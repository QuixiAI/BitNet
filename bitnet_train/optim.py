"""Cold-expert decay masking (moe_train_plan §4.3 — MANDATORY for Q-track).

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
