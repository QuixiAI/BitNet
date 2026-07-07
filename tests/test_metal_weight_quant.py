"""K1 `weight_quant_ternary` vs the host numpy oracle (docs/new-kernels.md §5).

The oracle functions are copied verbatim from QuixiCore-Metal `bindings/python/tk/quant.py`
(`quantize_bitnet` / `dequantize_bitnet`) — the packer the vendored GEMM kernels were
verified against.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


# ---- oracle: QuixiCore tk/quant.py, verbatim ----

def quantize_bitnet(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 32
    Wb = W.reshape(N, nb, 32)
    scale = np.abs(Wb).mean(axis=2).astype(np.float32)
    ssafe = np.where(scale == 0, 1.0, scale)
    wq = np.clip(np.rint(Wb / ssafe[..., None]), -1, 1).astype(np.int32)
    code = (wq + 1).astype(np.uint32).reshape(N, nb, 8, 4)
    out = np.zeros((N, nb, 10), np.uint8)
    out[:, :, 0:2] = scale.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:10] = (code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4) | (code[..., 3] << 6)).astype(np.uint8)
    return out


def dequantize_bitnet(packed):
    packed = np.ascontiguousarray(packed, np.uint8)
    N, nb, _ = packed.shape
    scale = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb, 1)
    qs = packed[:, :, 2:10].astype(np.int32)
    codes = np.stack([(qs >> (j * 2)) & 0x3 for j in range(4)], axis=-1).reshape(N, nb, 32)
    return (scale * (codes.astype(np.float32) - 1.0)).reshape(N, nb * 32)


def _tk():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bitnet_train" / "metal"))
    import tk_torch
    return tk_torch


@pytest.mark.parametrize("N,K", [(32, 32), (64, 128), (129, 2048), (2048, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_k1_matches_numpy_oracle(N, K, dtype):
    tk = _tk()
    torch.manual_seed(0)
    W = (torch.randn(N, K) * 0.05).to(dtype).to("mps")
    wq, w_deq = tk.weight_quant_ternary(W, 32)
    torch.mps.synchronize()

    ref = quantize_bitnet(W.float().cpu().numpy())
    got = wq.cpu().numpy()
    assert got.shape == (N, K // 32, 10)
    np.testing.assert_array_equal(got[:, :, 2:], ref[:, :, 2:])  # ternary codes exact
    # absmean scale: simd_sum's tree reduction vs numpy's pairwise sum can differ in the
    # last fp32 ulp, occasionally rounding to the adjacent half -> allow 1 half-ulp
    s_got = np.ascontiguousarray(got[:, :, 0:2]).reshape(N, -1).view(np.float16).astype(np.float32)
    s_ref = np.ascontiguousarray(ref[:, :, 0:2]).reshape(N, -1).view(np.float16).astype(np.float32)
    np.testing.assert_allclose(s_got, s_ref, rtol=2 ** -10, atol=0)

    deq_ref = dequantize_bitnet(got)                            # what the GEMM dequantizes
    deq_got = w_deq.float().cpu().numpy()
    assert w_deq.dtype == torch.bfloat16
    np.testing.assert_allclose(deq_got, deq_ref, rtol=1 / 128, atol=1e-6)  # bf16 of exact values


@pytest.mark.parametrize("group_k", [64, 512])
def test_k1_coarser_groups_replicate_scale(group_k):
    """group_k > 32 shares one absmean across group_k/32 packed blocks; group_k=K is per-tensor."""
    tk = _tk()
    torch.manual_seed(1)
    N, K = 48, 512
    W = (torch.randn(N, K) * 0.05).to("mps")
    wq, w_deq = tk.weight_quant_ternary(W, group_k)
    torch.mps.synchronize()

    got = wq.cpu().numpy()
    scales = got[:, :, 0:2].reshape(N, -1).view(np.float16).astype(np.float32).reshape(N, K // 32)
    bpg = group_k // 32
    for b in range(1, bpg):                                     # replicated within a scale group
        np.testing.assert_array_equal(scales[:, b::bpg], scales[:, 0::bpg])

    Wg = W.float().cpu().numpy().reshape(N, K // group_k, group_k)
    s_ref = np.maximum(np.abs(Wg).mean(axis=-1), 1e-5)
    np.testing.assert_allclose(scales[:, ::bpg], s_ref.astype(np.float16).astype(np.float32),
                               rtol=0, atol=0)

    q_ref = np.clip(np.rint(Wg / s_ref[..., None]), -1, 1)
    deq_ref = (q_ref * s_ref.astype(np.float16).astype(np.float32)[..., None]).reshape(N, K)
    np.testing.assert_allclose(w_deq.float().cpu().numpy(), deq_ref, rtol=1 / 128, atol=1e-6)


def test_k1_zero_and_ternary_range():
    tk = _tk()
    W = torch.zeros(32, 64, device="mps")
    wq, w_deq = tk.weight_quant_ternary(W, 32)
    torch.mps.synchronize()
    assert torch.all(w_deq == 0)                                # scale clamp, codes all 1 (=0)
    codes = wq.cpu().numpy()[:, :, 2:]
    assert np.all(np.stack([(codes >> (2 * j)) & 3 for j in range(4)]) == 1)

    torch.manual_seed(2)
    W = (torch.randn(64, 128) * 3).to("mps")
    _, w_deq = tk.weight_quant_ternary(W, 32)
    torch.mps.synchronize()
    scales = w_deq.float().abs().amax(dim=-1)
    ratio = w_deq.float().reshape(64, 4, 32) / w_deq.float().reshape(64, 4, 32).abs().amax(-1, keepdim=True).clamp_min(1e-20)
    vals = torch.unique(ratio.round())
    assert set(vals.tolist()).issubset({-1.0, 0.0, 1.0})        # ternary per group


def test_k1pt_per_tensor_absmean():
    """weight_quant_ternary_pt: ONE absmean scale over the whole (N,K) slice (the
    train_plan §4 formula), replicated into every packed block."""
    tk = _tk()
    torch.manual_seed(5)
    N, K = 96, 256
    W = (torch.randn(N, K) * 0.04).to("mps")
    wq, w_deq = tk.weight_quant_ternary_pt(W)
    torch.mps.synchronize()

    Wf = W.float().cpu().numpy()
    s_ref = max(np.abs(Wf).mean(), 1e-5)
    got = wq.cpu().numpy()
    scales = got[:, :, 0:2].reshape(N, -1).view(np.float16).astype(np.float32)
    assert np.unique(scales).size == 1                            # one scale everywhere
    np.testing.assert_allclose(scales[0, 0], np.float16(s_ref).astype(np.float32),
                               rtol=2 ** -10, atol=0)             # tree-sum ulp allowance

    q_ref = np.clip(np.rint(Wf / s_ref), -1, 1)
    deq_ref = q_ref * np.float16(s_ref).astype(np.float32)
    np.testing.assert_allclose(w_deq.float().cpu().numpy(), deq_ref, rtol=1 / 64, atol=1e-6)


@pytest.mark.parametrize("fn", ["group", "pt"])
def test_k5_batched_3d_matches_per_expert_2d(fn):
    """The (E,N,K) batched path must equal E independent 2-D calls."""
    tk = _tk()
    torch.manual_seed(6)
    E, N, K = 5, 64, 128
    W = (torch.randn(E, N, K) * 0.04).to("mps")
    call = (lambda w: tk.weight_quant_ternary(w, 32)) if fn == "group" else tk.weight_quant_ternary_pt
    wq3, wd3 = call(W)
    torch.mps.synchronize()
    assert wq3.shape == (E, N, K // 32, 10) and wd3.shape == (E, N, K)
    for e in range(E):
        wq2, wd2 = call(W[e].contiguous())
        torch.mps.synchronize()
        assert torch.equal(wq3[e], wq2), f"expert {e} packed mismatch"
        torch.testing.assert_close(wd3[e], wd2, rtol=0, atol=0)


def test_k1_agrees_with_pure_pytorch_oracle():
    """The Phase-0 fake-quant (bitlinear_metal.weight_quant_pergroup) must equal K1's w_deq."""
    from bitnet_train.bitlinear_metal import weight_quant_pergroup
    tk = _tk()
    torch.manual_seed(3)
    W = (torch.randn(96, 256) * 0.04).to("mps")
    _, w_deq = tk.weight_quant_ternary(W, 32)
    torch.mps.synchronize()
    ref = weight_quant_pergroup(W, 32)
    torch.testing.assert_close(w_deq.float(), ref.float(), rtol=1 / 128, atol=1e-6)
