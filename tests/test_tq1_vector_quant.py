"""Correctness oracle tests for quant/quant.py's TQ1_V11/V12 implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.quant import (  # noqa: E402
    FORMATS,
    PATTERN_COUNT,
    TQ1Projector,
    classify_linears,
    codebook_coverage,
    collect_pattern_counts,
    decode_artifact_weight,
    encode_ternary,
    learn_sign_canonical_codebook,
    load_iq1_grid_codebook,
    pack_indices,
    quantize_model,
    ternary_universe,
    unpack_indices,
)


@pytest.mark.parametrize("format_name", ["v11", "v12"])
def test_physical_payload_roundtrip_and_exact_rate(format_name):
    spec = FORMATS[format_name]
    # Include every index boundary that crosses low/high fields and sign bits.
    edge = torch.tensor([
        0, 1, 0xFF, 0x100, spec.n_shapes - 1, spec.n_shapes,
        spec.n_shapes + 1, spec.n_indices - 1,
    ], dtype=torch.int64)
    indices = edge.repeat(3, 8)                    # [3, 64] = two payload blocks
    packed = pack_indices(indices, spec)
    assert packed.shape == (3, 2, spec.payload_bytes)
    assert packed.numel() * 8 / indices.numel() == spec.index_bits
    assert torch.equal(unpack_indices(packed, spec), indices)


@pytest.mark.parametrize(
    ("format_name", "unique_vectors", "max_distance"),
    [("v11", 2047, 2), ("v12", 4095, 1)],
)
def test_sign_canonical_codebook_symmetry_and_coverage(
    format_name, unique_vectors, max_distance,
):
    counts = torch.arange(PATTERN_COUNT, dtype=torch.float64) % 17
    a = learn_sign_canonical_codebook(format_name, counts)
    b = learn_sign_canonical_codebook(format_name, counts)
    assert torch.equal(a.shapes, b.shapes)
    assert a.hash() == b.hash()

    expanded = a.expanded(dtype=torch.int8)
    assert torch.unique(encode_ternary(expanded)).numel() == unique_vectors
    encoded = set(encode_ternary(expanded).tolist())
    assert all(int(encode_ternary(-v)) in encoded for v in expanded)
    coverage = codebook_coverage(a)
    assert coverage["max_squared_trit_distance"] == max_distance
    assert sum(coverage["histogram"].values()) == PATTERN_COUNT


@pytest.mark.parametrize("format_name", ["v11", "v12"])
def test_alternating_projection_is_exactly_decodable(format_name):
    torch.manual_seed(7)
    weight = torch.randn(5, 256) * 0.03
    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    scalar = (weight / scale).round().clamp(-1, 1).reshape(-1, 8)
    counts = torch.bincount(encode_ternary(scalar), minlength=PATTERN_COUNT)
    book = learn_sign_canonical_codebook(format_name, counts)
    projector = TQ1Projector(book, candidate_count=16, chunk_groups=73)

    result = projector.project(
        weight,
        activation_importance=torch.linspace(0.5, 1.5, 256),
        metric="iq1",
        iterations=3,
        scale_dtype=torch.float16,
    )
    decoded = decode_artifact_weight(result.packed, result.scales, book)
    assert torch.equal(decoded, result.dequantized)
    assert torch.isfinite(result.dequantized).all()
    assert result.metrics.effective_bpw_with_row_scale == pytest.approx(
        FORMATS[format_name].raw_bpw + 16 / 256)

    indices = unpack_indices(result.packed, FORMATS[format_name])
    trits = book.decode(indices)
    assert set(torch.unique(trits).tolist()).issubset({-1, 0, 1})
    reconstructed = (trits.float() * result.scales.float()[:, None, None]).reshape_as(weight)
    assert torch.equal(reconstructed, result.dequantized)


def test_ternary_universe_encoding_is_bijective():
    universe = ternary_universe()
    assert universe.shape == (PATTERN_COUNT, 8)
    assert torch.equal(encode_ternary(universe), torch.arange(PATTERN_COUNT))


def test_read_only_llama_cpp_iq1_grid_baseline():
    source = Path.home() / "llama.cpp" / "ggml" / "src" / "ggml-common.h"
    if not source.is_file():
        pytest.skip("~/llama.cpp IQ1 reference is not available")
    book = load_iq1_grid_codebook(Path.home() / "llama.cpp")
    assert book.encoding == "joint"
    assert book.shapes.shape == (2048, 8)
    assert codebook_coverage(book) == {
        "max_squared_trit_distance": 2,
        "histogram": {"0": 2048, "1": 4252, "2": 261},
    }
    encoded = set(encode_ternary(book.shapes).tolist())
    assert sum(int(encode_ternary(-v)) in encoded for v in book.shapes) == 1331


def test_tiny_llama_inventory_and_end_to_end_conversion():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(11)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        tie_word_embeddings=True,
    )).float()
    config_before = model.config.to_dict()
    linears, kept, unmatched = classify_linears(model)
    assert len(linears) == 7
    assert kept == ["lm_head"]
    assert unmatched == []

    counts = collect_pattern_counts(linears, rows_per_linear=16)
    book = learn_sign_canonical_codebook("v12", counts)
    packed, reports = quantize_model(
        model,
        linears,
        book,
        device="cpu",
        candidate_count=8,
        chunk_groups=127,
        metric="iq1",
        iterations=2,
        scale_dtype=torch.float16,
    )
    assert model.config.to_dict() == config_before
    assert len(reports) == 7

    for name, module in linears:
        payload = packed[f"{name}.weight.__tq1_indices"]
        scales = packed[f"{name}.weight.__tq1_scale"]
        decoded = decode_artifact_weight(payload, scales, book)
        assert torch.equal(decoded, module.weight.detach())
