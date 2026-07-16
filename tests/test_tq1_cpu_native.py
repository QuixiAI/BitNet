from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

from bitnet_train.cpu import bitnet_cpu
from bitnet_train.tq1.codebook import (
    base3_ids, load_iq1_reference, product_codebook, sign_canonical_codebook)
from bitnet_train.tq1.oracle import linear_w2a8, quantize_activation
from bitnet_train.tq1.packing import layout, pack_payload
from bitnet_train.tq1.runtime import (
    DENSE_REPACK_LAYOUT_VERSION, NativeCPUTQ1Embedding, NativeCPUTQ1Linear,
    NativeRoutingPolicy)
from bitnet_train.tq1.oracle import dequantize_weight
from bitnet_train.tq1.solver import canonical_shapes


def _joint(fmt: str):
    shapes = canonical_shapes()
    count = 1024 if fmt == "v11" else 2048
    return sign_canonical_codebook(f"native_{fmt}", fmt, torch.cat((
        shapes[(shapes == 0).all(1)],
        shapes[~(shapes == 0).all(1)][:count - 1])))


def _four_lane_universe() -> torch.Tensor:
    value = torch.arange(3 ** 4, dtype=torch.int64)
    columns = []
    for _ in range(4):
        columns.append((value % 3 - 1).to(torch.int8))
        value //= 3
    return torch.stack(columns, 1)


def _product(fmt: str):
    universe = _four_lane_universe()
    nonzero = universe.ne(0)
    first = nonzero.to(torch.int64).argmax(1)
    negative = nonzero.any(1) & (universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(negative, -1, 1).to(torch.int8)[:, None]
    representatives = universe[torch.unique(base3_ids(canonical), sorted=True)]
    zero = representatives[(representatives == 0).all(1)]
    representatives = torch.cat((zero, representatives[~(representatives == 0).all(1)]))
    a = representatives[:32]
    if fmt == "v11":
        b = representatives[:32]
    else:
        selected = torch.cat((representatives, -representatives[1:24]))
        zero = selected[(selected == 0).all(1)]
        rest = selected[~(selected == 0).all(1)]
        b = torch.cat((zero, rest[torch.argsort(base3_ids(rest))]))
    return product_codebook(f"native_{fmt}_p", fmt, a, b)


def _case(profile: str, activation_mode: str, *, bf16_scale: bool = False):
    torch.manual_seed(173)
    fmt = "v11" if "v11" in profile else "v12"
    book = _product(fmt) if "-p-" in profile else _joint(fmt)
    N, K = 7, 512
    legal = torch.nonzero(book.legal_index_mask()).flatten()
    indices = legal[torch.randint(0, legal.numel(), (N, K // 8))]
    kwargs = {}
    if layout(profile).scale_mode == "block256":
        kwargs["block_scales"] = torch.rand(N, K // 256, dtype=torch.float16)
        kwargs["block_scales"][0, 0] = 0
        row_scales = None
    else:
        row_scales = torch.rand(N).to(torch.bfloat16 if bf16_scale else torch.float16)
        row_scales[0] = 0
    if layout(profile).affine:
        kwargs["affine_nibbles"] = torch.randint(
            0, 12, (N, K // 256, 8), dtype=torch.uint8)
    payload = pack_payload(indices, profile, **kwargs)
    x = torch.randn(1, K)
    expected = linear_w2a8(
        x, payload, profile, book, row_scales=row_scales,
        activation_mode=activation_mode).squeeze(0).numpy()
    aq = quantize_activation(x, activation_mode)
    scale_bits = None if row_scales is None else \
        row_scales.contiguous().view(torch.uint16).numpy()
    args = (
        payload.numpy(), scale_bits,
        book.decode(torch.arange(book.index_count)).numpy(),
        book.legal_index_mask().numpy(), aq.codes[0].numpy(),
        aq.scales[0].reshape(-1).numpy(), profile)
    kwargs = {
        "activation_mode": activation_mode,
        "row_scale_dtype": "bf16" if bf16_scale else "f16",
    }
    return expected, args, kwargs, payload, row_scales, book, x


@pytest.mark.parametrize(("profile", "activation_mode", "bf16_scale"), [
    ("tq1_v11-j-r", "a8_token", False),
    ("tq1_v12-j-r", "a8_block256", True),
    ("tq1_v11-p-r", "a8_token", False),
    ("tq1_v12-p-r", "a8_block256", False),
    ("tq1_v11-j-b", "a8_token", False),
    ("tq1_v12-j-b", "a8_block256", False),
    ("tq1_v11-j-a4-r", "a8_token", False),
    ("tq1_v11-j-a4-r", "a8_block256", False),
])
def test_native_scalar_and_neon_match_oracle(
        profile, activation_mode, bf16_scale):
    expected, args, kwargs, *_ = _case(
        profile, activation_mode, bf16_scale=bf16_scale)
    scalar = bitnet_cpu.gemv_tq1(*args, **kwargs, impl="scalar")
    np.testing.assert_allclose(scalar, expected, atol=1e-6, rtol=1e-6)
    if hasattr(bitnet_cpu._lib, "bn_tq1_gemv_neon"):
        neon = bitnet_cpu.gemv_tq1(*args, **kwargs, impl="neon")
        np.testing.assert_allclose(neon, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(neon, scalar)


def test_native_module_reports_deterministic_repack_and_preserves_payload():
    expected, _, _, payload, scales, book, x = _case(
        "tq1_v11-j-r", "a8_token")
    first = NativeCPUTQ1Linear(
        payload, "tq1_v11-j-r", book, row_scales=scales,
        activation_mode="a8_token", state_dict_name="test.weight", impl="scalar")
    second = NativeCPUTQ1Linear(
        payload, "tq1_v11-j-r", book, row_scales=scales,
        activation_mode="a8_token", state_dict_name="test.weight", impl="scalar")
    np.testing.assert_allclose(first(x).numpy()[0], expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(first.payload, payload)
    assert first.repack_report["repack_sha256"] == second.repack_report["repack_sha256"]
    assert first.repack_report["canonical_packed_remains_resident"] is True


def test_versioned_dense_prefill_repack_is_exact_budgeted_and_routed():
    _, _, _, payload, scales, book, _ = _case(
        "tq1_v11-j-r", "a8_token")
    required = payload.shape[0] * payload.shape[1] * 256 * 4
    policy = NativeRoutingPolicy(
        dense_repack_budget_bytes=required,
        short_prefill_min_tokens=2, long_prefill_min_tokens=4)
    module = NativeCPUTQ1Linear(
        payload, "tq1_v11-j-r", book, row_scales=scales,
        activation_mode="a8_token", state_dict_name="test.weight", impl="scalar",
        routing_policy=policy)
    x = torch.randn(5, payload.shape[1] * 256)
    expected = linear_w2a8(
        x, payload, "tq1_v11-j-r", book, row_scales=scales,
        activation_mode="a8_token")
    got = module(x)
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        module.dense_prefill_weight,
        dequantize_weight(payload, "tq1_v11-j-r", book, row_scales=scales),
        atol=0, rtol=0)
    assert module.repack_report["dense_layout_version"] == DENSE_REPACK_LAYOUT_VERSION
    assert module.repack_report["dense_repack_bytes"] == required
    assert module.repack_report["last_route"] == "dense_long_prefill"
    assert module.repack_report["canonical_packed_remains_resident"] is True

    denied = NativeCPUTQ1Linear(
        payload, "tq1_v11-j-r", book, row_scales=scales,
        activation_mode="a8_token", state_dict_name="test.weight", impl="scalar",
        routing_policy=NativeRoutingPolicy(
            dense_repack_budget_bytes=required - 1,
            short_prefill_min_tokens=2, long_prefill_min_tokens=4))
    denied(x)
    assert denied.dense_prefill_weight is None
    assert denied.repack_report["last_route"] == "packed_small_batch"


def test_output_head_dense_decode_is_an_explicit_shared_repack_route():
    expected, _, _, payload, scales, book, x = _case(
        "tq1_v11-j-r", "a8_token")
    required = payload.shape[0] * payload.shape[1] * 256 * 4
    shared = NativeCPUTQ1Embedding(
        payload, "tq1_v11-j-r", book, row_scales=scales,
        activation_mode="a8_token", state_dict_name="embed.weight", impl="scalar",
        routing_policy=NativeRoutingPolicy(
            dense_repack_budget_bytes=required,
            output_head_dense_decode=True))
    got = shared.linear(x).numpy()[0]
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)
    assert shared.repack_report["last_route"] == "dense_output_head_decode"
    assert shared.repack_report["dense_repack_materialized"] is True


@pytest.mark.parametrize(("profile", "activation_mode", "bf16_scale"), [
    ("tq1_v11-j-r", "a8_token", False),
    ("tq1_v12-j-r", "a8_block256", True),
    ("tq1_v11-p-r", "a8_token", False),
    ("tq1_v12-p-r", "a8_block256", False),
    ("tq1_v11-j-b", "a8_token", False),
    ("tq1_v12-j-b", "a8_block256", False),
    ("tq1_v11-j-a4-r", "a8_token", False),
])
def test_native_small_batch_and_prefill_match_oracle(
        profile, activation_mode, bf16_scale):
    _, args, kwargs, payload, row_scales, book, _ = _case(
        profile, activation_mode, bf16_scale=bf16_scale)
    x = torch.randn(5, payload.shape[1] * 256)
    expected = linear_w2a8(
        x, payload, profile, book, row_scales=row_scales,
        activation_mode=activation_mode).numpy()
    aq = quantize_activation(x, activation_mode)
    scale_bits = None if row_scales is None else \
        row_scales.contiguous().view(torch.uint16).numpy()
    got = bitnet_cpu.gemm_tq1(
        payload.numpy(), scale_bits,
        book.decode(torch.arange(book.index_count)).numpy(),
        book.legal_index_mask().numpy(), aq.codes.numpy(),
        aq.scales.reshape(x.shape[0], -1).numpy(), profile,
        activation_mode=activation_mode,
        row_scale_dtype=kwargs["row_scale_dtype"], impl="auto")
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)
    repeated = np.stack([
        bitnet_cpu.gemv_tq1(
            *args[:4], aq.codes[row].numpy(),
            aq.scales[row].reshape(-1).numpy(), profile,
            activation_mode=activation_mode,
            row_scale_dtype=kwargs["row_scale_dtype"], impl="auto")
        for row in range(x.shape[0])
    ])
    np.testing.assert_array_equal(got, repeated)


def test_native_rejects_reserved_product_index():
    _, args, kwargs, *_ = _case("tq1_v11-p-r", "a8_token")
    payload, scales, codebook, legal, xq, act, profile = args
    bad = np.flatnonzero(legal == 0)[0]
    # Low byte of the first group plus its three high bits in qh byte zero.
    payload = payload.copy()
    payload[0, 0, 0] = bad & 0xff
    payload[0, 0, 32] = (int(payload[0, 0, 32]) & 0xf8) | (bad >> 8)
    with pytest.raises(ValueError, match="reserved"):
        bitnet_cpu.gemv_tq1(
            payload, scales, codebook, legal, xq, act, profile,
            **kwargs, impl="scalar")


def test_pinned_iq1_direct_joint_runs_end_to_end():
    if not (Path.home() / "llama.cpp" / "ggml" / "src" / "ggml-common.h").is_file():
        pytest.skip("pinned read-only llama.cpp reference is unavailable")
    book = load_iq1_reference(reference_dir=Path.home() / "llama.cpp")
    torch.manual_seed(191)
    N, K = 3, 256
    indices = torch.randint(0, book.index_count, (N, K // 8))
    payload = pack_payload(indices, "tq1_v11-i-r")
    scales = torch.tensor([0.0, 0.125, 0.25], dtype=torch.float16)
    x = torch.randn(1, K)
    expected = linear_w2a8(
        x, payload, "tq1_v11-i-r", book, row_scales=scales).numpy()[0]
    aq = quantize_activation(x)
    got = bitnet_cpu.gemv_tq1(
        payload.numpy(), scales.numpy(),
        book.decode(torch.arange(book.index_count)).numpy(),
        book.legal_index_mask().numpy(), aq.codes[0].numpy(),
        aq.scales.reshape(-1).numpy(), "tq1_v11-i-r", impl="auto")
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)
