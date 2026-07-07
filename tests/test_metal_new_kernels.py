"""New-kernel batch (2026-07-06 gap-closing drop) vs PyTorch/numpy oracles:

  ternary_stats / code_flip_count   — §10.2/§6.2 health monitors over packed wq
  fake_quant_fp8                    — mode-b e4m3 fake-quant vs torch.float8_e4m3fn
  kd_kl_dense_fwd/bwd               — A6b full-KL vs dense PyTorch KL (+ autograd)
  qgemm_w2a8_fused                  — K2 vs the composed quantize + qgemm_w2a8 path
  attn_decode                       — batch-1 GQA decode vs SDPA
  moe_*_bwd                         — MoE dense backward vs a PyTorch autograd loop
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _tk():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bitnet_train" / "metal"))
    import tk_torch
    return tk_torch


def _unpack_codes(wq):
    """(rows, nb, 10) uint8 -> (rows, nb*32) int codes in {-1, 0, +1}."""
    qs = wq[:, :, 2:10].astype(np.int32)
    codes = np.stack([(qs >> (2 * j)) & 3 for j in range(4)], axis=-1)
    return codes.reshape(wq.shape[0], -1) - 1


# ---- ternary health monitors ----

def test_ternary_stats_matches_unpacked():
    tk = _tk()
    torch.manual_seed(0)
    W = (torch.randn(96, 256) * 0.05).to("mps")
    wq, _ = tk.weight_quant_ternary(W, 32)
    counts = tk.ternary_stats(wq)
    torch.mps.synchronize()
    codes = _unpack_codes(wq.cpu().numpy())
    ref = np.stack([(codes == v).sum(axis=1) for v in (-1, 0, 1)], axis=1)
    np.testing.assert_array_equal(counts.cpu().numpy(), ref)
    assert (counts.sum(dim=1) == 256).all()


def test_ternary_stats_expert_stack_rows():
    """(E, N, nb, 10) flattens to E*N rows — the per-expert zero-code tail readout."""
    tk = _tk()
    torch.manual_seed(1)
    E, N, K = 4, 32, 128
    W = (torch.randn(E, N, K) * 0.05).to("mps")
    wq, _ = tk.weight_quant_ternary_pt(W)
    counts = tk.ternary_stats(wq)
    torch.mps.synchronize()
    assert counts.shape == (E * N, 3)
    per_expert_zero = counts[:, 1].reshape(E, N).sum(dim=1).float() / (N * K)
    assert ((per_expert_zero >= 0) & (per_expert_zero <= 1)).all()


def test_code_flip_count():
    tk = _tk()
    torch.manual_seed(2)
    W = (torch.randn(64, 128) * 0.05).to("mps")
    wq_a, _ = tk.weight_quant_ternary(W, 32)
    wq_b, _ = tk.weight_quant_ternary(W + 0.02 * torch.randn_like(W), 32)
    flips = tk.code_flip_count(wq_a, wq_b)
    torch.mps.synchronize()
    ca, cb = _unpack_codes(wq_a.cpu().numpy()), _unpack_codes(wq_b.cpu().numpy())
    np.testing.assert_array_equal(flips.cpu().numpy(), (ca != cb).sum(axis=1))
    assert tk.code_flip_count(wq_a, wq_a).abs().sum().item() == 0


# ---- fp8 fake-quant ----

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fake_quant_fp8_matches_torch_e4m3(dtype):
    tk = _tk()
    torch.manual_seed(3)
    x = (torch.randn(64, 96) * 2.5).to(dtype).to("mps")
    x_fq, scale = tk.fake_quant_fp8(x)
    torch.mps.synchronize()
    assert x_fq.dtype == dtype and x_fq.shape == x.shape

    xf = x.float().cpu()
    s_ref = xf.abs().max().item() / 448.0
    assert np.isclose(scale.item(), s_ref, rtol=1e-6)
    ref = ((xf / s_ref).to(torch.float8_e4m3fn).float() * s_ref).to(dtype).float()
    # bit-exact: the kernel re-rounds the fast-math scale division and hand-rolls RNE
    torch.testing.assert_close(x_fq.float().cpu(), ref, rtol=0, atol=0)


# ---- dense KD-KL (A6b) ----

@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_kd_kl_dense_fwd_bwd_match_pytorch(tau):
    tk = _tk()
    torch.manual_seed(4)
    Tn, V = 24, 1000
    t = (torch.randn(Tn, V) * 2).to("mps")
    s = (torch.randn(Tn, V) * 2).to("mps")
    loss, lse_t, lse_s = tk.kd_kl_dense_fwd(t, s, invtemp=1.0 / tau)
    torch.mps.synchronize()

    tc, sc = t.cpu().double(), s.cpu().double().requires_grad_(True)
    p_t = F.softmax(tc / tau, dim=-1)
    ref = (p_t * (F.log_softmax(tc / tau, -1) - F.log_softmax(sc / tau, -1))).sum(-1)
    torch.testing.assert_close(loss.cpu().double(), ref, rtol=1e-4, atol=1e-5)

    go = torch.rand(Tn, device="mps")
    grad = tk.kd_kl_dense_bwd(t, s, lse_t, lse_s, go, invtemp=1.0 / tau)
    torch.mps.synchronize()
    ref.backward(go.cpu().double())
    torch.testing.assert_close(grad.cpu().double(), sc.grad, rtol=1e-4, atol=1e-6)


# ---- K2 fused forward ----

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qgemm_w2a8_fused_matches_composed(dtype):
    tk = _tk()
    torch.manual_seed(5)
    M, N, K = 33, 96, 256
    W = (torch.randn(N, K) * 0.05).to("mps")
    wq, _ = tk.weight_quant_ternary(W, 32)
    x = (torch.randn(M, K) * 0.7).to(dtype).to("mps")

    fused = tk.qgemm_w2a8_fused(wq, x)                       # (M, N)
    xq, a_scale = tk.quantize_per_token_int8(x)
    composed = tk.qgemm_w2a8(wq, xq, a_scale.half())         # (N, M)
    torch.mps.synchronize()
    assert fused.shape == (M, N) and fused.dtype == torch.half
    torch.testing.assert_close(fused, composed.t().contiguous(), rtol=1e-3, atol=1e-3)


# ---- attention decode ----

@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("D", [64, 128])
def test_attn_decode_matches_sdpa(dtype, tol, D):
    tk = _tk()
    torch.manual_seed(6)
    Tk, Hq, Hkv = 100, 8, 2
    q = torch.randn(Hq, D, dtype=dtype, device="mps")
    kc = torch.randn(Tk, Hkv, D, dtype=dtype, device="mps")
    vc = torch.randn(Tk, Hkv, D, dtype=dtype, device="mps")
    out = tk.attn_decode(q, kc, vc)
    torch.mps.synchronize()

    rep = Hq // Hkv
    qf = q.float().cpu()
    kf = kc.float().cpu().permute(1, 0, 2).repeat_interleave(rep, dim=0)   # (Hq, Tk, D)
    vf = vc.float().cpu().permute(1, 0, 2).repeat_interleave(rep, dim=0)
    ref = F.scaled_dot_product_attention(qf.unsqueeze(1), kf, vf).squeeze(1)
    torch.testing.assert_close(out.float().cpu(), ref, rtol=tol, atol=tol)


# ---- MoE backward ----

def test_moe_backward_matches_autograd():
    tk = _tk()
    torch.manual_seed(7)
    T, H, N_out, E, k = 40, 64, 64, 4, 2
    x = torch.randn(T, H, device="mps", requires_grad=True)
    W = torch.randn(E, H, N_out, device="mps", requires_grad=True) * 0.2
    W = W.detach().requires_grad_(True)
    logits = torch.randn(T, E, device="mps")
    ids, weights = tk.moe_route_topk(logits, k)
    weights = weights.detach().requires_grad_(True)

    # reference: pure autograd loop over tokens/experts
    y_ref = torch.zeros(T, N_out, device="mps")
    for j in range(k):
        sel = W[ids[:, j].long()]                            # (T, H, N_out)
        y_ref = y_ref + weights[:, j, None] * torch.bmm(x.unsqueeze(1), sel).squeeze(1)
    G = torch.randn(T, N_out, device="mps")
    (y_ref * G).sum().backward()
    torch.mps.synchronize()

    # metal path: schedule -> gather -> rect GEMM -> finalize; then the new backward
    ext = tk._ext
    sorted_idx, offsets, _ = ext.moe_permute(ids, E)
    eot, gather_idx, inv_pad, off_pad = ext.moe_pad_schedule(sorted_idx, offsets, k)
    A = ext.moe_gather(x.detach(), gather_idx)               # (P, H), zero pad rows
    Y = ext.moe_grouped_gemm_rect(A, W.detach(), eot)        # (P, N_out)
    out = ext.moe_finalize(Y, inv_pad, weights.detach(), k)  # (T, N_out)
    torch.mps.synchronize()
    torch.testing.assert_close(out, y_ref.detach(), rtol=1e-4, atol=1e-4)

    grad_eo, grad_w = tk.moe_finalize_bwd(G, Y, inv_pad, weights.detach())
    dA = tk.moe_grouped_gemm_bwd_dx(grad_eo, W.detach(), eot)
    dW = tk.moe_grouped_gemm_bwd_dw(A, grad_eo, off_pad, E)
    dx = tk.moe_gather_bwd(dA, inv_pad, k)
    torch.mps.synchronize()

    torch.testing.assert_close(dx, x.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dW, W.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad_w, weights.grad, rtol=1e-4, atol=1e-4)
