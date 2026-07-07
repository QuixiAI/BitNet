"""Optimizer machinery: cold-expert decay masking + the §5.4 precision hatches.

Hatches (train_plan §5.4, in pull order — promoted to defaults per track only
after A1 has a healthy fp32 baseline to compare against):
  1. 8-bit optimizer states  (MasterAdamW(moments_bits=8): blockwise-quantized moments)
  2. bf16 latents + explicit fp32 masters  (MasterAdamW: PyTorch does not
     maintain fp32 masters for bf16 params unless you build it)

Cold-expert decay masking (moe_train_plan §4.3 — MANDATORY for Q-track).

AdamW's decoupled decay fires every optimizer step whether or not a parameter
received gradient; a cold expert gets few updates but the full decay schedule,
so its latents shrink monotonically toward the zero-code region — decay
MANUFACTURES dead ternary experts, masquerading downstream as router collapse.

Mechanism (portable, exactly equivalent to selective decoupled decay): the
expert param groups run in the underlying torch.optim.AdamW with
weight_decay=0; after each optimizer step this masker applies
p[e] *= (1 - lr * wd) to the expert slices that WERE routed this step (the
utilization-floor exemption variant of §4.3). Routed ids come from RouterHooks.

Safety ordering (§5.2): the trainer forces intended_wd = 0 until
tests/test_decay_mask.py is green under the real optimizer wrapping — running
UNMASKED decoupled decay by accident is the failure this ordering forbids.
The fused MPS kernel path (tk.adamw_masked, already kernel-tested) is a T4
optimization behind the same semantics.
"""

from __future__ import annotations

import torch
from torch import nn

from bitnet_train.bitlinear import BitExperts


# ---------------------------------------------------------------------------
# §5.4 precision hatches
# ---------------------------------------------------------------------------

_Q8_BLOCK = 256


def _q8_encode_signed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise absmax int8 of a flat fp32 tensor (padded to _Q8_BLOCK)."""
    n = x.numel()
    pad = (-n) % _Q8_BLOCK
    xb = torch.nn.functional.pad(x.reshape(-1), (0, pad)).reshape(-1, _Q8_BLOCK)
    s = xb.abs().amax(dim=1, keepdim=True) / 127.0
    codes = torch.where(s > 0, (xb / s.clamp_min(1e-30)).round().clamp(-127, 127), xb)
    return codes.to(torch.int8), s.squeeze(1)


def _q8_decode_signed(codes: torch.Tensor, s: torch.Tensor, n: int) -> torch.Tensor:
    return (codes.float() * s.unsqueeze(1)).reshape(-1)[:n]


def _q8_encode_unsigned(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise max uint8 for the nonnegative second moment, encoded in the
    SQRT domain (bnb-style trick: v spans many decades; sqrt halves the dynamic
    range the 8-bit grid must cover)."""
    n = x.numel()
    pad = (-n) % _Q8_BLOCK
    r = torch.nn.functional.pad(x.reshape(-1).clamp_min(0).sqrt(), (0, pad))
    rb = r.reshape(-1, _Q8_BLOCK)
    s = rb.amax(dim=1, keepdim=True) / 255.0
    codes = torch.where(s > 0, (rb / s.clamp_min(1e-30)).round().clamp(0, 255), rb)
    return codes.to(torch.uint8), s.squeeze(1)


def _q8_decode_unsigned(codes: torch.Tensor, s: torch.Tensor, n: int) -> torch.Tensor:
    r = codes.float() * s.unsqueeze(1)
    return (r * r).reshape(-1)[:n]


class MasterAdamW(torch.optim.Optimizer):
    """AdamW with explicit fp32 MASTER weights for low-precision (bf16) params and
    optionally blockwise 8-bit moments — the §5.4 hatches, portable (CPU/MPS/CUDA).

    Semantics match torch.optim.AdamW (decoupled decay applied to the master);
    after each step the param is the master rounded to its storage dtype. Groups
    carry the same dicts build_param_groups emits (decay_masked groups must run
    wd=0 here too — pair with ColdExpertDecayMasker exactly as with stock AdamW).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, moments_bits=32):
        assert moments_bits in (32, 8)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.moments_bits = moments_bits

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    if p.dtype != torch.float32:
                        state["master"] = p.detach().float().clone()
                    if self.moments_bits == 32:
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    else:
                        state["m_q"] = _q8_encode_signed(torch.zeros_like(g))
                        state["v_q"] = _q8_encode_unsigned(torch.zeros_like(g))
                state["step"] += 1
                t = state["step"]
                master = state.get("master", p)

                if self.moments_bits == 32:
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                else:
                    n = g.numel()
                    m = _q8_decode_signed(*state["m_q"], n).reshape(g.shape)
                    v = _q8_decode_unsigned(*state["v_q"], n).reshape(g.shape)

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                if wd != 0.0:
                    master.mul_(1.0 - lr * wd)           # decoupled decay on the master
                mhat = m / (1 - beta1 ** t)
                vhat = v / (1 - beta2 ** t)
                master.addcdiv_(mhat, vhat.sqrt().add_(eps), value=-lr)

                if self.moments_bits == 8:
                    state["m_q"] = _q8_encode_signed(m)
                    state["v_q"] = _q8_encode_unsigned(v)
                if "master" in state:
                    p.copy_(master.to(p.dtype))          # rounded storage copy
        return loss


class ColdExpertDecayMasker:
    """Wire-up: masker = ColdExpertDecayMasker(model, optimizer, intended_wd)
    ... after optimizer.step():  masker.step(router_hooks.routed_and_reset())"""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 intended_wd: float):
        self.intended_wd = float(intended_wd)
        self.experts: list[BitExperts] = [m for m in model.modules()
                                          if isinstance(m, BitExperts)]
        # map each expert 3-D param to its group's live lr (masked groups only)
        self.param_lr: dict[int, dict] = {}
        for group in optimizer.param_groups:
            if not group.get("decay_masked"):
                continue
            if group.get("weight_decay", 0.0) != 0.0:
                raise ValueError(
                    "decay-masked group must run with weight_decay=0 in the "
                    "optimizer; the masker applies the decay itself (§5.2 ordering)")
            for p in group["params"]:
                self.param_lr[id(p)] = group

    @torch.no_grad()
    def step(self, routed: dict[int, list[int]]):
        """routed: {id(BitExperts) -> expert ids that received tokens this step}."""
        if self.intended_wd == 0.0:
            return
        for mod in self.experts:
            ids = routed.get(id(mod), [])
            if not ids:
                continue
            idx = torch.as_tensor(ids, device=mod.gate_up_proj.device)
            for p in (mod.gate_up_proj, mod.down_proj):
                group = self.param_lr.get(id(p))
                if group is None:
                    continue                       # frozen / non-masked param
                factor = 1.0 - group["lr"] * self.intended_wd
                p.data.index_copy_(0, idx, p.data.index_select(0, idx) * factor)
