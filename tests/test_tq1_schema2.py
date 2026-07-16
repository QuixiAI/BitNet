"""Normative schema-2 TQ1 contracts independent of the experimental CLI."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from bitnet_train.tq1.artifact import ArtifactBuilder, ArtifactReader
from bitnet_train.tq1.codebook import (
    CodebookRegistry,
    base3_ids,
    direct_joint_codebook,
    product_codebook,
    sign_canonical_codebook,
)
from bitnet_train.tq1.oracle import (
    dequantize_weight,
    linear_w2a8,
    quantize_activation,
)
from bitnet_train.tq1.packing import (
    PROFILE_LAYOUTS,
    pack_indices,
    pack_payload,
    unpack_indices,
    unpack_payload,
)
from bitnet_train.tq1.spec import CodebookRef, QuantSpec, canonical_json


def _universe(lanes: int) -> torch.Tensor:
    value = torch.arange(3 ** lanes, dtype=torch.int64)
    columns = []
    for _ in range(lanes):
        columns.append((value % 3 - 1).to(torch.int8))
        value //= 3
    return torch.stack(columns, dim=1)


def _joint_book(fmt: str):
    universe = _universe(8)
    nonzero = universe.ne(0)
    first = nonzero.to(torch.int64).argmax(1)
    negative = nonzero.any(1) & (universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(negative, -1, 1).to(torch.int8)[:, None]
    canonical_ids = torch.unique(base3_ids(canonical), sorted=True)
    shapes = universe[canonical_ids]
    zero = shapes[(shapes == 0).all(1)]
    shapes = torch.cat((zero, shapes[~(shapes == 0).all(1)]))
    count = 1024 if fmt == "v11" else 2048
    return sign_canonical_codebook(f"test_{fmt}", fmt, shapes[:count])


def _product_book(fmt: str):
    universe = _universe(4)
    nonzero = universe.ne(0)
    first = nonzero.to(torch.int64).argmax(1)
    negative = nonzero.any(1) & (universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(negative, -1, 1).to(torch.int8)[:, None]
    ids = torch.unique(base3_ids(canonical), sorted=True)
    reps = universe[ids]
    zero = reps[(reps == 0).all(1)]
    reps = torch.cat((zero, reps[~(reps == 0).all(1)]))
    a = reps[:32]
    if fmt == "v11":
        b = reps[:32]
    else:
        selected = torch.cat((reps, -reps[1:24]))
        zero = selected[(selected == 0).all(1)]
        nonzero_rows = selected[~(selected == 0).all(1)]
        b = torch.cat((zero, nonzero_rows[torch.argsort(base3_ids(nonzero_rows))]))
    return product_codebook(f"test_{fmt}_p", fmt, a, b)


def _spec(book, profile="tq1_v11-j-r"):
    return QuantSpec.core(
        default_profile=profile,
        codebook=book.ref(),
        target_regexes=(r"model\.layers\.\d+\.self_attn\.q_proj",),
        keep_fp_regexes=(r"lm_head",),
    )


def test_canonical_json_and_quant_spec_hash_are_stable():
    assert canonical_json({"z": -0.0, "é": 1e-7, "a": [1.0, True]}) == \
        '{"a":[1,true],"z":0,"é":1e-7}'
    book = _joint_book("v11")
    spec = _spec(book)
    restored = QuantSpec.from_dict(json.loads(spec.canonical_json()))
    assert restored == spec
    assert restored.sha256() == spec.sha256()
    assert len(spec.sha256()) == 64


def test_quant_spec_rejects_incompatible_codebook():
    book = _joint_book("v11")
    bad = CodebookRef("bad", "v11", "product", "model", "0" * 64)
    with pytest.raises(ValueError, match="incompatible"):
        QuantSpec.core(
            default_profile="tq1_v11-j-r", codebook=bad,
            target_regexes=("x",), keep_fp_regexes=("y",),
        )
    assert book.ref().sha256 == book.sha256()


@pytest.mark.parametrize("profile", [
    "tq1_v11-j-r", "tq1_v12-j-r", "tq1_v11-j-b", "tq1_v12-j-b",
])
def test_payload_round_trip_and_physical_sizes(profile):
    spec = PROFILE_LAYOUTS[profile]
    torch.manual_seed(2)
    indices = torch.randint(0, 1 << spec.index_bits, (2, 64))
    if spec.index_bits == 11:
        indices[indices == 1024] = 0
    raw = pack_indices(indices, profile)
    assert raw.shape == (2, 2, spec.raw_index_bytes)
    assert torch.equal(unpack_indices(raw, profile), indices)
    if spec.scale_mode == "block256":
        scales = torch.tensor([[0.0, 0.5], [1.0, 2.0]], dtype=torch.float16)
        payload = pack_payload(indices, profile, block_scales=scales)
        got, got_scales, affine = unpack_payload(payload, profile)
        assert torch.equal(got, indices)
        assert torch.equal(got_scales, scales)
        assert affine is None
    else:
        payload = pack_payload(indices, profile)
    assert payload.shape[-1] == spec.block_bytes


def test_a4_payload_round_trip_and_reserved_mu_rejected():
    book = _joint_book("v11")
    indices = torch.arange(32).reshape(1, 32)
    nibbles = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=torch.uint8)
    payload = pack_payload(indices, "tq1_v11-j-a4-r", affine_nibbles=nibbles)
    got, scales, affine = unpack_payload(payload, "tq1_v11-j-a4-r")
    assert torch.equal(got, indices)
    assert scales is None and torch.equal(affine, nibbles)
    with pytest.raises(ValueError, match="reserved"):
        pack_payload(indices, "tq1_v11-j-a4-r",
                     affine_nibbles=torch.full_like(nibbles, 12))
    assert book.legal_index_mask().sum() == 2047


@pytest.mark.parametrize(("fmt", "unique"), [("v11", 2047), ("v12", 4049)])
def test_product_codebook_invariants_and_canonical_representatives(fmt, unique):
    book = _product_book(fmt)
    expanded = book.decode(torch.arange(book.index_count))
    assert torch.unique(base3_ids(expanded)).numel() == unique
    legal = book.legal_index_mask()
    assert int(legal.sum()) == unique
    book.validate_indices(torch.nonzero(legal).flatten())
    with pytest.raises(ValueError, match="reserved"):
        book.validate_indices(torch.nonzero(~legal).flatten()[:1])


def test_w2a8_oracle_matches_dequantized_reference():
    torch.manual_seed(4)
    book = _joint_book("v12")
    indices = torch.randint(0, 4096, (5, 64))
    indices[indices == 2048] = 0
    payload = pack_payload(indices, "tq1_v12-j-r")
    scales = torch.rand(5, dtype=torch.float16)
    x = torch.randn(3, 512)
    got = linear_w2a8(x, payload, "tq1_v12-j-r", book, row_scales=scales)
    aq = quantize_activation(x)
    weight = dequantize_weight(payload, "tq1_v12-j-r", book, row_scales=scales)
    expected = aq.dequantize() @ weight.T
    torch.testing.assert_close(got, expected, atol=5e-6, rtol=1e-6)


def test_a4_with_block256_activation_matches_direct_dequantization():
    torch.manual_seed(8)
    book = _joint_book("v11")
    indices = torch.randint(0, 2048, (3, 64))
    indices[indices == 1024] = 0
    nibbles = torch.randint(0, 12, (3, 2, 8), dtype=torch.uint8)
    payload = pack_payload(indices, "tq1_v11-j-a4-r", affine_nibbles=nibbles)
    scales = torch.rand(3, dtype=torch.float16)
    x = torch.randn(4, 512)
    got = linear_w2a8(
        x, payload, "tq1_v11-j-a4-r", book, row_scales=scales,
        activation_mode="a8_block256")
    expected = quantize_activation(x, "a8_block256").dequantize() @ dequantize_weight(
        payload, "tq1_v11-j-a4-r", book, row_scales=scales).T
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-6)


def test_schema2_artifact_round_trip(tmp_path):
    book = _joint_book("v11")
    registry = CodebookRegistry({book.id: book})
    spec = _spec(book)
    builder = ArtifactBuilder(
        spec, registry, source_model="unit/model", source_revision="a" * 40,
        tokenizer_sha256="b" * 64, chat_template_sha256="c" * 64,
        provenance={"test": True},
    )
    source_files = tmp_path / "source_files"
    source_files.mkdir()
    (source_files / "config.json").write_text('{"tie_word_embeddings": true}')
    (source_files / "tokenizer_config.json").write_text("{}")
    indices = torch.zeros((2, 32), dtype=torch.int64)
    payload = pack_payload(indices, "tq1_v11-j-r")
    builder.add_quantized(
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj", payload,
        logical_shape=(2, 256), profile="tq1_v11-j-r",
        codebook_id=book.id, row_scales=torch.tensor([0.0, 0.5], dtype=torch.float16),
    )
    builder.add_non_tq1("model.norm.weight", torch.ones(2, dtype=torch.float16))
    tied = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    builder.add_non_tq1("model.embed_tokens.weight", tied)
    builder.add_non_tq1("lm_head.weight", tied)
    with pytest.raises(ValueError, match="exact shared storage"):
        builder.add_alias("equal_but_untied.weight", "model.embed_tokens.weight",
                          tied.clone())
    with pytest.raises(ValueError, match="identical dtype"):
        builder.add_non_tq1("incompatible_tie.weight", tied, dtype=torch.float16)
    out = builder.write(tmp_path / "artifact", source_files=source_files,
                        quantization_report={"ok": True})
    reader = ArtifactReader(out)
    reader.validate()
    assert reader.quant_spec.sha256() == spec.sha256()
    assert reader.manifest["tensors"][0]["payload_key"].endswith(".__tq1_payload")
    assert reader.aliases == {
        "lm_head.weight": {
            "target": "model.embed_tokens.weight", "shape": [4, 2],
            "dtype": "float32", "kind": "parameter"}}
    restored = reader.non_tq1_state_dict()
    assert restored["lm_head.weight"] is restored["model.embed_tokens.weight"]
    sizes = reader.manifest["size_accounting"]
    assert sizes["unique_logical_parameters"] == 512 + 2 + 8
    assert sizes["logical_parameter_references"] == 512 + 2 + 8 + 8
    assert sizes["non_tq1_physical_bytes"] == 2 * 2 + 8 * 4
    assert sizes["non_tq1_logical_reference_bytes"] == 2 * 2 + 2 * 8 * 4
    assert sizes["canonical_artifact_bytes"] == sum(
        path.stat().st_size for path in out.iterdir() if path.is_file())

    corruptions = {
        "hash": lambda manifest: manifest.__setitem__("tensor_aliases_sha256", "0" * 64),
        "missing": lambda manifest: manifest["tensor_aliases"]["lm_head.weight"].__setitem__(
            "target", "missing.weight"),
        "shape": lambda manifest: manifest["tensor_aliases"]["lm_head.weight"].__setitem__(
            "shape", [8, 1]),
        "dtype": lambda manifest: manifest["tensor_aliases"]["lm_head.weight"].__setitem__(
            "dtype", "float16"),
    }
    for label, mutate in corruptions.items():
        broken = tmp_path / f"broken-{label}"
        shutil.copytree(out, broken)
        manifest_path = broken / "tq1_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        mutate(manifest)
        if label != "hash":
            manifest["tensor_aliases_sha256"] = hashlib.sha256(
                canonical_json(manifest["tensor_aliases"]).encode()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="alias"):
            ArtifactReader(broken).validate()

    cyclic = tmp_path / "broken-cycle"
    shutil.copytree(out, cyclic)
    manifest_path = cyclic / "tq1_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    alias = manifest["tensor_aliases"]["lm_head.weight"]
    manifest["tensor_aliases"] = {
        "lm_head.weight": {**alias, "target": "other.weight"},
        "other.weight": {**alias, "target": "lm_head.weight"},
    }
    manifest["tensor_aliases_sha256"] = hashlib.sha256(
        canonical_json(manifest["tensor_aliases"]).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="cycle"):
        ArtifactReader(cyclic).validate()


def test_pinned_iq1_grid_has_normative_schema2_hash():
    source = Path.home() / "llama.cpp" / "ggml" / "src" / "ggml-common.h"
    if not source.is_file():
        pytest.skip("read-only llama.cpp IQ1 source is unavailable")
    from quant.quant import load_iq1_grid_codebook
    legacy = load_iq1_grid_codebook(Path.home() / "llama.cpp")
    book = direct_joint_codebook("iq1_grid_v1", legacy.shapes)
    assert book.sha256() == \
        "1edfeb295366968940d5d4397dc046110f851acb59de9407fdf0c06982adaa72"
