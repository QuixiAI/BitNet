"""STE gradients of the reference BitLinear == the dense analytic formula
(train_plan §4: grad_x = g @ w_q, grad_W = g^T @ x_q; scales constant)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train import quant  # noqa: E402
from bitnet_train.bitlinear import BitLinear  # noqa: E402

torch.manual_seed(0)


@pytest.mark.parametrize("granularity", ["tensor", "group"])
@pytest.mark.parametrize("act_quant", [True, False])
def test_ste_grads_match_dense_formula(granularity, act_quant):
    lin = BitLinear(64, 32, backend="reference", granularity=granularity)
    lin.act_quant = act_quant
    x = torch.randn(8, 64, requires_grad=True)
    y = lin(x)
    g = torch.randn_like(y)
    y.backward(g)

    with torch.no_grad():
        w_q = quant.weight_quant(lin.weight, granularity, lin.group_k)
        x_q = quant.activation_quant(x.detach()) if act_quant else x.detach()
    torch.testing.assert_close(x.grad, g @ w_q, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(lin.weight.grad, g.t() @ x_q, rtol=1e-5, atol=1e-6)


def test_grad_flows_to_latent_not_ternary():
    """AdamW accumulates sub-threshold updates into the LATENT — the healing
    mechanism (train_plan §4). One step must move the latent even when no code flips."""
    lin = BitLinear(32, 32, backend="reference", granularity="tensor")
    w0 = lin.weight.detach().clone()
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
    lin(torch.randn(4, 32)).sum().backward()
    opt.step()
    assert not torch.equal(lin.weight.detach(), w0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_act_quant_sibling_sharing_metal():
    """q/k/v-style siblings fed the SAME input tensor run K4 once (routing
    optimization 2026-07-07) and produce outputs/grads identical to distinct
    per-module quantization; a mutated input misses the cache (_version guard)."""
    from bitnet_train import bitlinear_metal as bm

    dev = "mps"
    sibs = [BitLinear(96, 64, backend="metal", granularity="tensor",
                      device=dev) for _ in range(3)]
    x_shared = torch.randn(4, 8, 96, device=dev, requires_grad=True)

    calls = {"n": 0}
    real_tk = bm._tk()
    real_fq = real_tk.fake_quant_int8

    def counting_fq(t):
        calls["n"] += 1
        return real_fq(t)

    real_tk.fake_quant_int8 = counting_fq
    try:
        ys = [m(x_shared) for m in sibs]                 # same object -> 1 quant
        assert calls["n"] == 1
        sum(y.sum() for y in ys).backward()
        gx_shared = x_shared.grad.clone()

        calls["n"] = 0
        x2 = x_shared.detach().clone().requires_grad_(True)
        ys2 = [m(x2.clone()) for m in sibs]              # distinct objects -> 3 quants
        assert calls["n"] == 3
        sum(y.sum() for y in ys2).backward()
        for a, b in zip(ys, ys2):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(gx_shared, x2.grad)

        calls["n"] = 0
        x3 = torch.randn(4, 8, 96, device=dev)
        sibs[0](x3)
        x3.mul_(2.0)                                     # bump _version
        sibs[1](x3)
        assert calls["n"] == 2                           # mutation defeats the cache
    finally:
        real_tk.fake_quant_int8 = real_fq
