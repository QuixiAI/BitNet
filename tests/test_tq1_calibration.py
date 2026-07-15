from __future__ import annotations

import json

import pytest
import torch
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
        save_calibration_artifact(path, {"layer": sums}, metadata={
            **identity, "records": 1, "retained_tokens": len(part),
            "truncated_tokens": index, "bucket_tokens": {"unit": len(part)},
            "calibration_file_sha256": str(index) * 64,
        })
        paths.append(path)
    merged = merge_calibration_artifacts(paths, tmp_path / "merged.safetensors")
    got, metadata = load_calibration_sums(merged)
    expected = ModuleSums(256, frozenset(("diagonal", "covariance8")))
    expected.add(x)
    torch.testing.assert_close(got["layer"].diag_sum, expected.diag_sum)
    torch.testing.assert_close(got["layer"].cov8_sum, expected.cov8_sum)
    assert got["layer"].token_count == 9
    assert metadata["records"] == 2 and metadata["bucket_tokens"] == {"unit": 9}
    assert metadata["truncated_tokens"] == 1

    bad_sums = ModuleSums(256, frozenset(("diagonal", "covariance8")))
    bad_sums.add(x[:1])
    bad = tmp_path / "bad.safetensors"
    save_calibration_artifact(bad, {"layer": bad_sums}, metadata={
        **identity, "model_revision": "c" * 40})
    with pytest.raises(ValueError, match="model_revision"):
        merge_calibration_artifacts((paths[0], bad), tmp_path / "nope.safetensors")
