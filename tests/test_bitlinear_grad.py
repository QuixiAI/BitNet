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
