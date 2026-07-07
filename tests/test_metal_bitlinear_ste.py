"""BitLinearSTE parity: Metal composition vs the pure-PyTorch Phase-0 oracle
(docs/new-kernels.md §5). Forward within bf16/half tolerance; gradients must match
the analytic STE formulas grad_x = g @ w_q, grad_W = g^T @ x_q (the quantizers are
identity in the backward — a numerical-jacobian gradcheck is meaningless on a
staircase function, so we check the analytic form, like the reference trainer).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bitnet_train" / "metal"))

from bitnet_train.bitlinear_metal import (  # noqa: E402
    BitLinear, BitLinearSTE, act_quant_int8, bitlinear_reference, weight_quant_pergroup,
)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _rel(a, b):
    return (a - b).norm() / b.norm().clamp_min(1e-12)


@pytest.mark.parametrize("M,K,N", [(1, 64, 32), (33, 128, 96), (128, 2048, 512)])
def test_forward_matches_reference(M, K, N):
    torch.manual_seed(0)
    x = (torch.randn(M, K) * 0.5).to("mps")
    W = (torch.randn(N, K) * 0.03).to("mps")
    y_metal = BitLinearSTE.apply(x, W, 32)
    torch.mps.synchronize()
    y_ref = bitlinear_reference(x, W, 32)
    assert y_metal.shape == (M, N)
    # integer-exact GEMM vs fp32 dense on fake-quant values; half a_scale/output rounding
    assert _rel(y_metal.float(), y_ref.float()) < 2e-2


def test_backward_matches_analytic_ste():
    torch.manual_seed(1)
    M, K, N = 40, 256, 64
    x = (torch.randn(M, K) * 0.5).to("mps").requires_grad_(True)
    W = (torch.randn(N, K) * 0.03).to("mps").requires_grad_(True)
    g = torch.randn(M, N, device="mps")

    y = BitLinearSTE.apply(x, W, 32)
    y.backward(g)

    w_q = weight_quant_pergroup(W.detach(), 32).float()
    x_q = act_quant_int8(x.detach()).float()
    grad_x_ref = g.float() @ w_q                      # (M,N)@(N,K)
    grad_W_ref = g.float().t() @ x_q                  # (N,M)@(M,K)

    assert _rel(x.grad.float(), grad_x_ref) < 3e-2    # bf16 backward GEMMs
    assert _rel(W.grad.float(), grad_W_ref) < 3e-2


def test_backends_agree_on_grads():
    """Same layer, same batch: reference-backend autograd vs metal-backend autograd."""
    torch.manual_seed(2)
    M, K, N = 16, 128, 64
    x = (torch.randn(M, K) * 0.5).to("mps")
    W0 = (torch.randn(N, K) * 0.03).to("mps")

    grads = {}
    for backend in ("reference", "metal"):
        lin = BitLinear(K, N, backend=backend, device="mps")
        with torch.no_grad():
            lin.weight.copy_(W0)
        xb = x.clone().requires_grad_(True)
        loss = lin(xb).float().square().mean()
        loss.backward()
        grads[backend] = (loss.detach(), xb.grad.detach(), lin.weight.grad.detach())

    l_r, gx_r, gw_r = grads["reference"]
    l_m, gx_m, gw_m = grads["metal"]
    assert abs(l_m - l_r) / l_r.clamp_min(1e-12) < 3e-2
    assert _rel(gx_m.float(), gx_r.float()) < 5e-2
    assert _rel(gw_m.float(), gw_r.float()) < 5e-2


def test_training_step_moves_loss_together():
    """One SGD step per backend from identical init lands at ~the same loss (layer-level
    portability gate; the 1B end-to-end gate lives in the trainer)."""
    torch.manual_seed(3)
    M, K, N = 32, 256, 128
    x = (torch.randn(M, K) * 0.5).to("mps")
    tgt = torch.randn(M, N, device="mps")
    W0 = (torch.randn(N, K) * 0.03).to("mps")

    losses = {}
    for backend in ("reference", "metal"):
        lin = BitLinear(K, N, backend=backend, device="mps")
        with torch.no_grad():
            lin.weight.copy_(W0)
        opt = torch.optim.SGD(lin.parameters(), lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            loss = (lin(x).float() - tgt).square().mean()
            loss.backward()
            opt.step()
        losses[backend] = float(loss.detach())
    assert abs(losses["metal"] - losses["reference"]) / losses["reference"] < 5e-2


def test_fake_quant_int8_matches_eager_chain():
    """K4 one-pass fake-quant == quantize_per_token_int8 + eager half-grid dequant."""
    import tk_torch as tk
    torch.manual_seed(7)
    x = (torch.randn(64, 512) * 0.5).to("mps")
    x_q, codes, scale = tk.fake_quant_int8(x)
    codes_ref, scale_ref = tk.quantize_per_token_int8(x)
    torch.mps.synchronize()
    assert torch.equal(codes, codes_ref)
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
    ref = (codes_ref.float() * scale_ref.to(torch.float16).float().unsqueeze(-1)).to(torch.bfloat16)
    torch.testing.assert_close(x_q, ref, rtol=0, atol=0)


def test_silu_mul_fake_quant_matches_composition():
    import tk_torch as tk
    torch.manual_seed(8)
    x = (torch.randn(32, 256)).to(torch.bfloat16).to("mps")
    gate = (torch.randn(32, 256)).to(torch.bfloat16).to("mps")
    x_q, codes, scale = tk.silu_mul_fake_quant_int8(x, gate)
    torch.mps.synchronize()
    act = (torch.nn.functional.silu(x.float()) * gate.float())
    s_ref = act.abs().amax(-1, keepdim=True) / 127.0
    q_ref = torch.where(s_ref > 0, (act / s_ref.clamp_min(1e-30)).round().clamp(-127, 127),
                        torch.zeros_like(act))
    torch.testing.assert_close(scale, s_ref.squeeze(-1), rtol=1e-3, atol=1e-6)
    assert (codes.float() - q_ref).abs().max() <= 1     # bf16 activation rounding at code edges
    ref = (codes.float() * s_ref.to(torch.float16).float()).to(torch.bfloat16)
    torch.testing.assert_close(x_q, ref, rtol=0, atol=0)


def test_weight_quant_cache_per_step():
    """The version-keyed cache: hits within a step (grad accum / checkpoint recompute),
    misses after an in-place optimizer update."""
    torch.manual_seed(9)
    lin = BitLinear(128, 64, backend="metal", device="mps")
    x = (torch.randn(8, 128) * 0.5).to("mps")
    lin(x); v0 = lin._wcache[0]
    lin(x)                                        # second micro-batch, same step
    assert lin._wcache[0] == v0
    wd0 = lin._wcache[2]
    with torch.no_grad():
        lin.weight.mul_(1.001)                    # what an optimizer step does
    lin(x)
    assert lin._wcache[0] != v0 and lin._wcache[2] is not wd0

    # per-tensor granularity path
    lin_t = BitLinear(128, 64, backend="metal", granularity="tensor", device="mps")
    y = lin_t(x)
    from bitnet_train.bitlinear_metal import weight_quant_pertensor
    ref = bitlinear_reference(x, lin_t.weight.detach(), granularity="tensor")
    assert _rel(y.float(), ref.float()) < 2e-2


def test_leading_dims_and_frozen_inference():
    torch.manual_seed(4)
    K, N = 128, 64
    lin = BitLinear(K, N, backend="metal", device="mps")

    x3 = (torch.randn(2, 5, K) * 0.5).to("mps")     # (B, T, K) flattens/unflattens
    assert lin(x3).shape == (2, 5, N)

    lin.eval().freeze()
    y_train_path = BitLinearSTE.apply(x3.reshape(-1, K), lin.weight, 32)
    y_frozen = lin(x3).reshape(-1, N)
    torch.mps.synchronize()
    # same quantized grid; train path is a bf16 dense GEMM, frozen prefill the half MMA
    assert _rel(y_frozen.float(), y_train_path.float()) < 2e-2

    x1 = (torch.randn(1, K) * 0.5).to("mps")        # batch-1 -> qgemv_w2a8 decode path
    y1 = lin(x1)
    y1_ref = bitlinear_reference(x1, lin.weight.detach(), 32)
    assert y1.shape == (1, N)
    assert _rel(y1.float(), y1_ref.float()) < 2e-2
