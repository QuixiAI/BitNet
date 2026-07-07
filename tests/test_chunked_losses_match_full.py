"""Chunked losses == full-tensor references (values AND grads) on CPU at toy
vocab; fused MPS kernels == chunked on MPS (train_plan §7.2)."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train.distill import (  # noqa: E402
    IGNORE, LossComputer, LossConfig, chunked_ce, chunked_kd_dense, chunked_kd_topk)

torch.manual_seed(0)
T, H, V, K = 32, 64, 512, 16


def _inputs(requires_grad=True):
    hidden = torch.randn(T, H, requires_grad=requires_grad)
    head_w = (torch.randn(V, H) * 0.05).requires_grad_(requires_grad)  # leaf!
    targets = torch.randint(0, V, (T,))
    targets[-3:] = IGNORE
    return hidden, head_w, targets


def test_chunked_ce_matches_full():
    hidden, head_w, targets = _inputs()
    loss = chunked_ce(hidden, head_w, targets, vchunk=128)
    go = torch.rand(T)
    (loss * go).sum().backward()

    h2 = hidden.detach().clone().requires_grad_(True)
    w2 = head_w.detach().clone().requires_grad_(True)
    logits = h2.float() @ w2.float().t()
    ref = F.cross_entropy(logits, targets, reduction="none", ignore_index=IGNORE)
    (ref * go).sum().backward()

    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(hidden.grad, h2.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(head_w.grad, w2.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_chunked_kd_dense_matches_full(tau):
    hidden, head_w, _ = _inputs()
    t_logits = torch.randn(T, V) * 2
    loss = chunked_kd_dense(hidden, head_w, t_logits, tau=tau, vchunk=128)
    go = torch.rand(T)
    (loss * go).sum().backward()

    h2 = hidden.detach().clone().requires_grad_(True)
    w2 = head_w.detach().clone().requires_grad_(True)
    zs = (h2.float() @ w2.float().t()) / tau
    zt = t_logits.float() / tau
    p_t = F.softmax(zt, -1)
    ref = (p_t * (F.log_softmax(zt, -1) - F.log_softmax(zs, -1))).sum(-1)
    (ref * go).sum().backward()

    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(hidden.grad, h2.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(head_w.grad, w2.grad, rtol=1e-4, atol=1e-5)


def _topk_reference(h, w, t_idx, t_prob, tau, tail_mode):
    """Full-tensor implementation of the SAME sparse-support loss."""
    zs = (h.float() @ w.float().t()) / tau
    logq_full = F.log_softmax(zs, -1)
    pad = t_idx < 0
    idx = t_idx.clamp_min(0)
    logq = logq_full.gather(-1, idx).masked_fill(pad, 0.0)
    p = t_prob.float().masked_fill(pad, 0.0)
    P = p.sum(-1)
    tiny = 1e-30
    if tail_mode == 0:
        pt = p / P.clamp_min(tiny).unsqueeze(-1)
        return (pt * (pt.clamp_min(tiny).log() - logq)).masked_fill(pad, 0.0).sum(-1)
    S = logq_full.gather(-1, idx).exp().masked_fill(pad, 0.0).sum(-1)
    loss = (p * (p.clamp_min(tiny).log() - logq)).masked_fill(pad, 0.0).sum(-1)
    tail = (1 - P).clamp_min(0)
    return loss + torch.where(tail > 0,
                              tail * (tail.clamp_min(tiny).log()
                                      - (1 - S).clamp_min(tiny).log()),
                              torch.zeros_like(tail))


@pytest.mark.parametrize("tail_mode", [0, 1])
def test_chunked_kd_topk_matches_reference(tail_mode):
    hidden, head_w, _ = _inputs()
    t_full = F.softmax(torch.randn(T, V) / 0.5, -1)
    top = t_full.topk(K, -1)
    t_idx = top.indices.int()
    t_prob = top.values
    t_idx[0, -4:] = -1                                   # padded cache entries

    loss = chunked_kd_topk(hidden, head_w, t_idx, t_prob, tau=2.0,
                           tail_mode=tail_mode, vchunk=128)
    go = torch.rand(T)
    (loss * go).sum().backward()

    h2 = hidden.detach().clone().requires_grad_(True)
    w2 = head_w.detach().clone().requires_grad_(True)
    ref = _topk_reference(h2, w2, t_idx.long(), t_prob, 2.0, tail_mode)
    (ref * go).sum().backward()

    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(hidden.grad, h2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(head_w.grad, w2.grad, rtol=1e-4, atol=1e-4)


def test_tail_modes_change_gradients():
    """The config-recorded tail choice 'changes gradients' (train_plan §5.1)."""
    hidden, head_w, _ = _inputs()
    t_full = F.softmax(torch.randn(T, V), -1)
    top = t_full.topk(K, -1)
    g0 = torch.autograd.grad(
        chunked_kd_topk(hidden, head_w, top.indices.int(), top.values,
                        tail_mode=0).sum(), hidden, retain_graph=False)[0]
    g1 = torch.autograd.grad(
        chunked_kd_topk(hidden, head_w, top.indices.int(), top.values,
                        tail_mode=1).sum(), hidden)[0]
    assert not torch.allclose(g0, g1)


def test_loss_computer_kd_zero_for_identical_teacher():
    hidden, head_w, targets = _inputs(requires_grad=False)
    lc = LossComputer(LossConfig(kd_mode="dense", tchunk=8, vchunk=128,
                                 prefer_fused=False))
    with torch.no_grad():
        s_logits = hidden.float() @ head_w.float().t()
    out = lc(hidden.requires_grad_(True), head_w, targets,
             teacher_batch=lambda sl: s_logits[sl])
    assert abs(float(out["kd"])) < 1e-5                  # KL(p‖p) = 0
    assert float(out["ce"]) > 0
    out["loss"].backward()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_fused_matches_chunked_on_mps():
    dev = "mps"
    hidden = torch.randn(T, H, device=dev, requires_grad=True)
    head_w = (torch.randn(V, H, device=dev) * 0.05).requires_grad_(True)
    targets = torch.randint(0, V, (T,), device=dev)
    t_logits = torch.randn(T, V, device=dev) * 2

    for kd_mode, teacher in (("dense", lambda sl: t_logits[sl]), ("none", None)):
        outs = {}
        for fused in (True, False):
            h = hidden.detach().clone().requires_grad_(True)
            w = head_w.detach().clone().requires_grad_(True)
            lc = LossComputer(LossConfig(kd_mode=kd_mode, tchunk=16, vchunk=128,
                                         prefer_fused=fused))
            out = lc(h, w, targets, teacher_batch=teacher)
            out["loss"].backward()
            outs[fused] = (out, h.grad, w.grad)
        torch.testing.assert_close(outs[True][0]["loss"], outs[False][0]["loss"],
                                   rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(outs[True][1], outs[False][1], rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(outs[True][2], outs[False][2], rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_fused_cekd_ignore_index_on_mps():
    """IGNORE rows carry no CE loss/grad through the fused CE+KD kernel but the
    KD term still applies to them (matches LossComputer's chunked semantics)."""
    dev = "mps"
    hidden = torch.randn(T, H, device=dev)
    head_w = torch.randn(V, H, device=dev) * 0.05
    targets = torch.randint(0, V, (T,), device=dev)
    targets[::4] = IGNORE
    t_logits = torch.randn(T, V, device=dev) * 2

    outs = {}
    for fused in (True, False):
        h = hidden.detach().clone().requires_grad_(True)
        w = head_w.detach().clone().requires_grad_(True)
        lc = LossComputer(LossConfig(kd_mode="dense", tchunk=16, vchunk=128,
                                     prefer_fused=fused))
        out = lc(h, w, targets, teacher_batch=lambda sl: t_logits[sl])
        out["loss"].backward()
        outs[fused] = (out, h.grad, w.grad)
    for key in ("loss", "ce", "kd"):
        torch.testing.assert_close(outs[True][0][key], outs[False][0][key],
                                   rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(outs[True][1], outs[False][1], rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(outs[True][2], outs[False][2], rtol=1e-2, atol=1e-3)
