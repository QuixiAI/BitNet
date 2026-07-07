"""Fused ternary-expert MoE FFN (the new `bitnet` grouped-GEMM instantiation) vs a
pure-PyTorch reference MoE with per-group-32 fake-quant experts (W-only activations —
the mode-a0 / rollout-prefill path of moe_train_plan §7.5)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bitnet_train" / "metal"))

from bitnet_train.bitlinear_metal import weight_quant_pergroup  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

T, H, INTER, E, K = 64, 64, 64, 8, 2


def _rel(a, b):
    return (a - b).norm() / b.norm().clamp_min(1e-12)


def _fixtures():
    torch.manual_seed(0)
    x = (torch.randn(T, H) * 0.5).to(torch.bfloat16).to("mps")
    w_gate = torch.randn(E, INTER, H, device="mps") * 0.1
    w_up = torch.randn(E, INTER, H, device="mps") * 0.1
    w_down = torch.randn(E, H, INTER, device="mps") * 0.1
    logits = torch.randn(T, E, device="mps")
    return x, w_gate, w_up, w_down, logits


def test_route_topk_matches_torch():
    import tk_torch as tk
    _, _, _, _, logits = _fixtures()
    ids, w = tk.moe_route_topk(logits, K)
    torch.mps.synchronize()
    p = torch.softmax(logits.float(), -1)
    top_p, top_i = p.topk(K, dim=-1)
    top_p = top_p / top_p.sum(-1, keepdim=True)          # renormalized over selected
    assert torch.equal(torch.sort(ids.long(), -1).values, torch.sort(top_i, -1).values)
    torch.testing.assert_close(w.sum(-1), torch.ones(T, device="mps"), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.sort(w, -1).values, torch.sort(top_p, -1).values,
                               rtol=1e-4, atol=1e-5)


def test_moe_ffn_bitnet_matches_reference():
    import tk_torch as tk
    x, w_gate, w_up, w_down, logits = _fixtures()

    w1 = torch.cat([w_gate, w_up], dim=1).contiguous()          # (E, 2*inter, H)
    w1q, _ = tk.weight_quant_ternary(w1, 32)
    w2q, _ = tk.weight_quant_ternary(w_down.contiguous(), 32)
    ids, wts = tk.moe_route_topk(logits, K)
    y = tk.moe_ffn_bitnet(x, w1q.reshape(E, 2 * INTER, -1), w2q.reshape(E, H, -1), ids, wts)
    torch.mps.synchronize()

    # reference: dense fp32 on the fake-quant experts, same routing
    xf = x.float()
    ref = torch.zeros(T, H, device="mps")
    for e in range(E):
        g_q = weight_quant_pergroup(w_gate[e], 32).float()
        u_q = weight_quant_pergroup(w_up[e], 32).float()
        d_q = weight_quant_pergroup(w_down[e], 32).float()
        h = torch.nn.functional.silu(xf @ g_q.t()) * (xf @ u_q.t())
        y_e = h @ d_q.t()
        for slot in range(K):
            m = ids[:, slot].long() == e
            ref[m] += wts[m, slot].unsqueeze(-1) * y_e[m]

    assert y.shape == (T, H)
    assert _rel(y.float(), ref) < 3e-2      # bf16 activations + half dequant MMA vs fp32


def test_moe_ffn_bitnet_uneven_expert_load():
    """Routing skew (some experts empty) exercises the pad tiles / -1 sentinels."""
    import tk_torch as tk
    x, w_gate, w_up, w_down, _ = _fixtures()
    # force all tokens onto experts 0 and 3
    logits = torch.full((T, E), -10.0, device="mps")
    logits[:, 0] = 5.0
    logits[:, 3] = 4.0
    w1 = torch.cat([w_gate, w_up], dim=1).contiguous()
    w1q, _ = tk.weight_quant_ternary(w1, 32)
    w2q, _ = tk.weight_quant_ternary(w_down.contiguous(), 32)
    ids, wts = tk.moe_route_topk(logits, K)
    y = tk.moe_ffn_bitnet(x, w1q.reshape(E, 2 * INTER, -1), w2q.reshape(E, H, -1), ids, wts)
    torch.mps.synchronize()

    xf = x.float()
    ref = torch.zeros(T, H, device="mps")
    for e in (0, 3):
        g_q = weight_quant_pergroup(w_gate[e], 32).float()
        u_q = weight_quant_pergroup(w_up[e], 32).float()
        d_q = weight_quant_pergroup(w_down[e], 32).float()
        h = torch.nn.functional.silu(xf @ g_q.t()) * (xf @ u_q.t())
        y_e = h @ d_q.t()
        for slot in range(K):
            m = ids[:, slot].long() == e
            ref[m] += wts[m, slot].unsqueeze(-1) * y_e[m]
    assert _rel(y.float(), ref) < 3e-2
