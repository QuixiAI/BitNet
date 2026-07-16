from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from bitnet_train.tq1.calibration import (
    CalibrationCollector,
    ModuleSums,
    iter_calibration_records,
    load_calibration_artifact,
    load_calibration_sums,
    merge_calibration_artifacts,
    normalized_statistics,
    save_calibration_artifact,
)


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        return type("Tokens", (), {"input_ids": torch.arange(1, len(text) + 2)[None]})

    def apply_chat_template(self, messages, **_kwargs):
        length = sum(len(message["content"]) + 1 for message in messages)
        return torch.arange(1, length + 1)[None]


def test_jsonl_parser_is_strict_and_truncation_is_recorded(tmp_path):
    path = tmp_path / "calibration.jsonl"
    path.write_text("\n".join([
        json.dumps({"text": "abcdef", "bucket": "prose", "source": "unit"}),
        json.dumps({"messages": [{"role": "user", "content": "hey"},
                                  {"role": "assistant", "content": "hello"}],
                    "bucket": "chat"}),
        "plain text",
    ]))
    records = list(iter_calibration_records(path, _Tokenizer(), limit=3, sequence_cap=5))
    assert len(records) == 3
    assert records[0].bucket == "prose" and records[0].source == "unit"
    assert records[0].input_ids.numel() == 5 and records[0].truncated_tokens == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('[{"text":"not jsonl"}]\n')
    with pytest.raises(ValueError, match="arrays"):
        list(iter_calibration_records(bad, _Tokenizer(), limit=1, sequence_cap=8))


def test_statistics_match_direct_formulas_and_merge(tmp_path):
    torch.manual_seed(3)
    x = torch.randn(7, 256)
    module = nn.Linear(256, 8, bias=False)
    with CalibrationCollector({"layer": module}, modes=("diagonal", "covariance8", "block256")) \
            as collector:
        module(x[:3])
        module(x[3:])
    sums = collector.sums["layer"]
    torch.testing.assert_close(sums.diag_sum, x.double().square().sum(0))
    expected8 = torch.einsum("tgi,tgj->gij", x.double().reshape(7, 32, 8),
                             x.double().reshape(7, 32, 8))
    torch.testing.assert_close(sums.cov8_sum, expected8)
    expected256 = torch.einsum("tbi,tbj->bij", x.double().reshape(7, 1, 256),
                               x.double().reshape(7, 1, 256))
    torch.testing.assert_close(sums.cov256_sum, expected256)
    stats = normalized_statistics(sums)
    assert stats["diag"].mean() == pytest.approx(1.0)
    assert stats["cov8"].diagonal(dim1=-2, dim2=-1).mean() == pytest.approx(1.0)

    path = tmp_path / "stats.safetensors"
    save_calibration_artifact(path, {"layer": sums}, metadata={"model": "unit"})
    tensors, metadata = load_calibration_artifact(path)
    assert metadata["model"] == "unit" and metadata["token_counts"] == {"layer": 7}
    assert set(tensors) >= {"layer.diag", "layer.cov8", "layer.cov256",
                            "layer.__raw_diag_sum"}

    left, right = ModuleSums(256, sums.modes), ModuleSums(256, sums.modes)
    left.add(x[:2]); right.add(x[2:]); left.merge(right)
    torch.testing.assert_close(left.diag_sum, sums.diag_sum)
    assert left.token_count == sums.token_count


def test_covariance_statistics_reject_zero_groups_and_invalid_ridge():
    sums = ModuleSums(256, frozenset(("diagonal", "covariance8")))
    value = torch.zeros(2, 256)
    value[:, :8] = torch.randn(2, 8)
    sums.add(value)
    with pytest.raises(ValueError, match="zero-statistic group"):
        normalized_statistics(sums)
    complete = ModuleSums(256, frozenset(("diagonal",)))
    complete.add(torch.randn(2, 256))
    with pytest.raises(ValueError, match="ridge_factor"):
        normalized_statistics(complete, ridge_factor=-1)


def test_artifact_merge_uses_raw_sums_and_rejects_identity_mismatch(tmp_path):
    torch.manual_seed(31)
    x = torch.randn(9, 256)
    identity = {
        "model": "unit", "model_revision": "a" * 40,
        "tokenizer": "unit", "tokenizer_revision": "b" * 40,
        "sequence_cap": 256, "accumulation_dtype": "float64_cpu",
        "target_modules": ["layer"], "modes": ["diagonal", "covariance8"],
    }
    paths = []
    for index, part in enumerate((x[:4], x[4:])):
        sums = ModuleSums(256, frozenset(("diagonal", "covariance8")))
        sums.add(part)
        path = tmp_path / f"part{index}.safetensors"
        frequencies = torch.tensor([index + 1, 2 * index + 1], dtype=torch.int64)
        save_calibration_artifact(path, {"layer": sums}, metadata={
            **identity, "records": 1, "retained_tokens": len(part),
            "truncated_tokens": index, "bucket_tokens": {"unit": len(part)},
            "source_tokens": {f"source_{index}": len(part)},
            "calibration_file_sha256": str(index) * 64,
        }, extra_tensors={"model.embed_tokens.token_frequency": frequencies})
        paths.append(path)
    merged = merge_calibration_artifacts(paths, tmp_path / "merged.safetensors")
    got, metadata = load_calibration_sums(merged)
    expected = ModuleSums(256, frozenset(("diagonal", "covariance8")))
    expected.add(x)
    torch.testing.assert_close(got["layer"].diag_sum, expected.diag_sum)
    torch.testing.assert_close(got["layer"].cov8_sum, expected.cov8_sum)
    assert got["layer"].token_count == 9
    assert metadata["records"] == 2 and metadata["bucket_tokens"] == {"unit": 9}
    assert metadata["source_tokens"] == {"source_0": 4, "source_1": 5}
    assert metadata["truncated_tokens"] == 1
    tensors, _ = load_calibration_artifact(merged)
    assert torch.equal(
        tensors["model.embed_tokens.token_frequency"], torch.tensor([3, 4]))

    bad_sums = ModuleSums(256, frozenset(("diagonal", "covariance8")))
    bad_sums.add(x[:1])
    bad = tmp_path / "bad.safetensors"
    save_calibration_artifact(bad, {"layer": bad_sums}, metadata={
        **identity, "model_revision": "c" * 40})
    with pytest.raises(ValueError, match="model_revision"):
        merge_calibration_artifacts((paths[0], bad), tmp_path / "nope.safetensors")


def test_calibration_loader_rejects_normalized_statistics_that_do_not_match_raw_sums(
        tmp_path):
    sums = ModuleSums(256, frozenset(("diagonal", "covariance8")))
    sums.add(torch.randn(4, 256))
    valid = tmp_path / "valid.safetensors"
    save_calibration_artifact(valid, {"layer": sums}, metadata={"model": "unit"})
    tensors = load_file(valid)
    tensors["layer.diag"] = tensors["layer.diag"] * 2
    with safe_open(valid, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    corrupt = tmp_path / "corrupt.safetensors"
    save_file(tensors, corrupt, metadata=metadata)
    with pytest.raises(ValueError, match="normalized diag does not match raw sums"):
        load_calibration_artifact(corrupt)


def test_calibration_metadata_and_tensor_inventory_fail_closed(tmp_path):
    sums = ModuleSums(256, frozenset(("diagonal", "covariance8")))
    sums.add(torch.randn(4, 256))
    with pytest.raises(ValueError, match="finite JSON"):
        save_calibration_artifact(
            tmp_path / "nan.safetensors", {"layer": sums},
            metadata={"invalid": float("nan")})
    with pytest.raises(ValueError, match="extra tensor"):
        save_calibration_artifact(
            tmp_path / "extra.safetensors", {"layer": sums}, metadata={},
            extra_tensors={"undeclared": torch.ones(1)})

    valid = tmp_path / "valid_metadata.safetensors"
    save_calibration_artifact(valid, {"layer": sums}, metadata={})
    tensors = load_file(valid)
    with safe_open(valid, framework="pt", device="cpu") as handle:
        transport = handle.metadata()
    metadata = json.loads(transport["metadata_json"])
    metadata["ridge_damping_before_normalization"]["layer"] *= 2
    transport["metadata_json"] = json.dumps(metadata, separators=(",", ":"))
    corrupt = tmp_path / "bad_damping.safetensors"
    save_file(tensors, corrupt, metadata=transport)
    with pytest.raises(ValueError, match="ridge-damping metadata"):
        load_calibration_artifact(corrupt)

    tensors["undeclared"] = torch.ones(1)
    unknown = tmp_path / "unknown_tensor.safetensors"
    metadata["ridge_damping_before_normalization"]["layer"] /= 2
    transport["metadata_json"] = json.dumps(metadata, separators=(",", ":"))
    save_file(tensors, unknown, metadata=transport)
    with pytest.raises(ValueError, match="unknown fields"):
        load_calibration_artifact(unknown)
