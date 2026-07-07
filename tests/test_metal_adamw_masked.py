"""adamw_masked: the moe_train_plan §5.6 decay-mask semantics at kernel level.

  route zero tokens to expert j -> step -> expert j's latents unchanged   (mask_mode 0)
  route tokens to expert j      -> step -> decay applied exactly now
plus mask_mode 1 (skip decay only) and exact agreement with the unmasked kernel /
torch.optim.AdamW on fully-active masks.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bitnet_train" / "metal"))

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

LR, B1, B2, EPS, WD = 1e-2, 0.9, 0.95, 1e-8, 0.1


def _setup(E=4, seg=64):
    torch.manual_seed(0)
    p = torch.randn(E * seg, device="mps")
    g = torch.randn(E * seg, device="mps")
    m = torch.randn(E * seg, device="mps").abs() * 0.01
    v = torch.randn(E * seg, device="mps").abs() * 0.01
    return p, g, m, v


def test_masked_segment_untouched_mode0():
    import tk_torch as tk
    E, seg = 4, 64
    p, g, m, v = _setup(E, seg)
    mask = torch.tensor([1, 0, 1, 0], dtype=torch.uint8, device="mps")
    p2, m2, v2 = tk.adamw_masked(p, g, m, v, LR, B1, B2, EPS, WD, 1, mask, seg)
    torch.mps.synchronize()
    for e in range(E):
        s = slice(e * seg, (e + 1) * seg)
        if mask[e] == 0:   # cold expert: bit-identical pass-through
            assert torch.equal(p2[s], p[s]) and torch.equal(m2[s], m[s]) and torch.equal(v2[s], v[s])
        else:              # routed expert: must have moved (grad is nonzero)
            assert not torch.equal(p2[s], p[s])


def test_active_segments_match_unmasked_kernel_and_torch():
    import tk_torch as tk
    E, seg = 4, 64
    p, g, m, v = _setup(E, seg)
    ones = torch.ones(E, dtype=torch.uint8, device="mps")
    pm, mm, vm = tk.adamw_masked(p, g, m, v, LR, B1, B2, EPS, WD, 3, ones, seg)
    pu, mu, vu = tk.adamw(p, g, m, v, LR, B1, B2, EPS, WD, 3)
    torch.mps.synchronize()
    assert torch.equal(pm, pu) and torch.equal(mm, mu) and torch.equal(vm, vu)

    # against torch.optim.AdamW from the same state (step count 1 -> bias correction t=1)
    p0, g0, m0, v0 = _setup(E, seg)
    ref = p0.detach().clone().requires_grad_(True)
    opt = torch.optim.AdamW([ref], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)
    opt.state[ref] = {"step": torch.tensor(0.0), "exp_avg": m0.clone(), "exp_avg_sq": v0.clone()}
    ref.grad = g0.clone()
    opt.step()
    pk, _, _ = tk.adamw_masked(p0, g0, m0, v0, LR, B1, B2, EPS, WD, 1, ones, seg)
    torch.mps.synchronize()
    torch.testing.assert_close(pk, ref.detach(), rtol=1e-5, atol=1e-6)


def test_mode1_skips_decay_only():
    import tk_torch as tk
    E, seg = 2, 32
    p, g, m, v = _setup(E, seg)
    mask = torch.tensor([1, 0], dtype=torch.uint8, device="mps")
    p2, m2, v2 = tk.adamw_masked(p, g, m, v, LR, B1, B2, EPS, WD, 2, mask, seg, mask_mode=1)
    # reference: same step with wd zeroed on the cold segment
    pw, mw, vw = tk.adamw(p, g, m, v, LR, B1, B2, EPS, WD, 2)      # decayed everywhere
    pn, mn, vn = tk.adamw(p, g, m, v, LR, B1, B2, EPS, 0.0, 2)     # no decay anywhere
    torch.mps.synchronize()
    s0, s1 = slice(0, seg), slice(seg, 2 * seg)
    assert torch.equal(p2[s0], pw[s0])          # hot: full decay
    assert torch.equal(p2[s1], pn[s1])          # cold: moments updated, decay skipped
    assert torch.equal(m2, mw) and torch.equal(v2, vw)   # moments identical either way


def test_erosion_demo_qa_decay():
    """The §4.3 pathology in miniature (ablation Q-A-decay): a never-routed expert under
    UNMASKED decay shrinks monotonically; under the mask it is bit-stable."""
    import tk_torch as tk
    seg = 128
    p = torch.randn(2 * seg, device="mps")
    zeros = torch.zeros_like(p)
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    mask = torch.tensor([1, 0], dtype=torch.uint8, device="mps")
    p_unmasked, p_masked = p.clone(), p.clone()
    mu, vu = m.clone(), v.clone()
    mm, vm = m.clone(), v.clone()
    for t in range(1, 51):
        p_unmasked, mu, vu = tk.adamw(p_unmasked, zeros, mu, vu, LR, B1, B2, EPS, WD, t)
        p_masked, mm, vm = tk.adamw_masked(p_masked, zeros, mm, vm, LR, B1, B2, EPS, WD, t,
                                           mask, seg)
    torch.mps.synchronize()
    cold = slice(seg, 2 * seg)
    shrink = (p_unmasked[cold].abs() / p[cold].abs().clamp_min(1e-12)).mean()
    assert shrink < 0.96                        # ~ (1 - lr*wd)^50 ≈ 0.951: measurable erosion
    assert torch.equal(p_masked[cold], p[cold])  # masked: bit-identical after 50 steps
