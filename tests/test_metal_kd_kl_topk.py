"""kd_kl_topk fwd/bwd vs a full-vocab PyTorch reference at toy vocab (the
test_chunked_gkd_matches_full discipline from train_plan §7.2)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bitnet_train" / "metal"))

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

T_, V, K = 16, 64, 8


def _fixtures(tau=2.0, pad_some=False):
    torch.manual_seed(0)
    student = torch.randn(T_, V, device="mps") * 2
    teacher = torch.randn(T_, V, device="mps") * 2
    p_full = torch.softmax(teacher / tau, dim=-1)
    p_top, idx = p_full.topk(K, dim=-1)
    idx = idx.int()
    if pad_some:
        idx[:, -2:] = -1                      # variable-k caches pad with -1
        p_top[:, -2:] = 0.0
    return student, idx, p_top.float(), tau


def _ref_loss(student, idx, p_top, tau, tail_mode):
    logq = torch.log_softmax(student.float() / tau, dim=-1)
    loss = torch.zeros(T_, device=student.device)
    for t in range(T_):
        valid = idx[t] >= 0
        ii, pp = idx[t][valid].long(), p_top[t][valid].float()
        qq = logq[t][ii].float()
        if tail_mode == 0:
            pt = pp / pp.sum()
            loss[t] = (pt * (pt.clamp_min(1e-30).log() - qq)).sum().float()
        else:
            tail = (1 - pp.sum()).clamp_min(0)
            q_other = (1 - logq[t][ii].exp().sum()).clamp_min(1e-30)
            terms = (pp * (pp.clamp_min(1e-30).log() - qq)).sum()
            if tail > 0:
                terms = terms + tail * (tail.clamp_min(1e-30).log() - q_other.log())
            loss[t] = terms.float()
    return loss


@pytest.mark.parametrize("tail_mode", [0, 1])
@pytest.mark.parametrize("pad", [False, True])
def test_fwd_matches_reference(tail_mode, pad):
    import tk_torch as tk
    student, idx, p_top, tau = _fixtures(pad_some=pad)
    loss, lse = tk.kd_kl_topk_fwd(student, idx, p_top, invtemp=1.0 / tau, tail_mode=tail_mode)
    torch.mps.synchronize()
    ref = _ref_loss(student, idx, p_top, tau, tail_mode)
    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-5)
    lse_ref = torch.logsumexp(student.float() / tau, dim=-1)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tail_mode", [0, 1])
def test_bwd_matches_autograd(tail_mode):
    import tk_torch as tk
    student, idx, p_top, tau = _fixtures()

    # autograd reference through the same math
    s_ref = student.detach().clone().requires_grad_(True)
    logq = torch.log_softmax(s_ref.float() / tau, dim=-1)
    losses = []
    for t in range(T_):
        ii, pp = idx[t].long(), p_top[t]
        qq = logq[t][ii]
        if tail_mode == 0:
            pt = pp / pp.sum()
            losses.append((pt * (pt.clamp_min(1e-30).log() - qq)).sum())
        else:
            tail = (1 - pp.sum()).clamp_min(0)
            q_other = (1 - qq.exp().sum()).clamp_min(1e-30)
            losses.append((pp * (pp.clamp_min(1e-30).log() - qq)).sum()
                          + tail * (tail.clamp_min(1e-30).log() - q_other.log()))
    go = torch.rand(T_, device="mps") + 0.5
    torch.stack(losses).mul(go).sum().backward()

    _, lse = tk.kd_kl_topk_fwd(student, idx, p_top, invtemp=1.0 / tau, tail_mode=tail_mode)
    grad = tk.kd_kl_topk_bwd(student, idx, p_top, lse, go, invtemp=1.0 / tau,
                             tail_mode=tail_mode)
    torch.mps.synchronize()
    torch.testing.assert_close(grad, s_ref.grad, rtol=2e-4, atol=1e-5)


def test_full_k_equals_dense_kl():
    """K = V with renorm == the exact dense KL(teacher || student) — the E0 sanity."""
    import tk_torch as tk
    tau = 1.0
    torch.manual_seed(1)
    student = torch.randn(T_, V, device="mps")
    teacher = torch.randn(T_, V, device="mps")
    p = torch.softmax(teacher, -1)
    idx = torch.argsort(p, -1, descending=True).int()
    p_sorted = torch.gather(p, 1, idx.long()).float()
    loss, _ = tk.kd_kl_topk_fwd(student, idx, p_sorted, invtemp=1.0, tail_mode=0)
    torch.mps.synchronize()
    ref = torch.nn.functional.kl_div(torch.log_softmax(student.float(), -1), p.float(),
                                     reduction="none").sum(-1)
    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-5)
