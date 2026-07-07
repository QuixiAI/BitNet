"""Router health panel + router hooks (moe_train_plan §6.2: "if healing goes
sideways it shows up first as the router collapsing onto a few experts, not as
a clean perplexity regression").

RouterHooks serves three consumers from ONE set of forward hooks on the
`mlp.gate` routers (transformers-5 Qwen3MoeTopKRouter returns
(router_logits, router_scores, router_indices)):

  * the aux load-balance loss — live (in-graph) logits per layer, handed to the
    model's OWN load_balancing_loss_func with the model's OWN coefficient
    (§5.1: never invented);
  * the decay masker — the union of routed expert ids per optimizer step,
    per experts module (moe §4.3);
  * the §6.2 stats panel — per-layer routing entropy, expert-load histogram,
    max-load, dead experts, plus the zero-code-in-the-cold-tail alarm input
    (cross-referencing ternary health per expert against utilization).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from bitnet_train.bitlinear import BitExperts


class RouterHooks:
    def __init__(self, model: nn.Module):
        self.pairs: list[tuple[str, nn.Module, BitExperts]] = []
        for name, mod in model.named_modules():
            if isinstance(mod, BitExperts):
                block = name.rsplit(".experts", 1)[0]
                router = model.get_submodule(f"{block}.gate")
                self.pairs.append((block, router, mod))
        self._handles = []
        self.collect_aux = False
        self._aux_logits: list[torch.Tensor] = []
        self._routed: dict[int, set[int]] = {}          # id(experts_mod) -> expert ids
        self._load: dict[str, torch.Tensor] = {}        # layer -> expert token counts
        self._entropy: dict[str, list[float]] = {}
        self._tokens: dict[str, int] = {}

    def attach(self):
        for block, router, experts in self.pairs:
            self._handles.append(router.register_forward_hook(
                self._make_hook(block, experts)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, block: str, experts: BitExperts):
        E = experts.num_experts

        def hook(_mod, _inp, out):
            logits, _scores, idx = out
            if self.collect_aux:
                self._aux_logits.append(logits)
            with torch.no_grad():
                ids = idx.reshape(-1)
                self._routed.setdefault(id(experts), set()).update(
                    ids.unique().tolist())
                load = torch.bincount(ids, minlength=E).cpu()
                self._load[block] = self._load.get(block, torch.zeros(E,
                                                                      dtype=torch.long)) + load
                p = F.softmax(logits.float(), dim=-1)
                ent = float(-(p * p.clamp_min(1e-20).log()).sum(-1).mean())
                self._entropy.setdefault(block, []).append(ent)
                self._tokens[block] = self._tokens.get(block, 0) + idx.shape[0]
        return hook

    # ---- aux loss (per micro-batch; call between forward and backward) ----
    def aux_loss(self, num_experts: int, top_k: int, coef: float) -> torch.Tensor:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import \
            load_balancing_loss_func
        if not self._aux_logits:
            return torch.zeros((), device="cpu")
        loss = load_balancing_loss_func(tuple(self._aux_logits), num_experts,
                                        top_k) * coef
        self._aux_logits = []
        return loss

    def clear_aux(self):
        self._aux_logits = []

    # ---- decay masker feed (per optimizer step) ----
    def routed_and_reset(self) -> dict[int, list[int]]:
        out = {k: sorted(v) for k, v in self._routed.items()}
        self._routed = {}
        return out

    # ---- §6.2 panel (per eval interval) ----
    def stats_and_reset(self, model: nn.Module | None = None,
                        dead_eps: float = 1e-3) -> dict:
        per_layer, agg = {}, {"entropy": [], "max_load": [], "dead": 0}
        zero_tail = []
        health = None
        if model is not None:
            from bitnet_train.bitlinear import ternary_health
            health = ternary_health(model)
        for block, load in self._load.items():
            tokens = max(self._tokens.get(block, 1), 1)
            frac = load.float() / (load.sum().clamp_min(1))
            util = load.float() / tokens
            ent = sum(self._entropy[block]) / max(len(self._entropy[block]), 1)
            dead = int((util < dead_eps).sum())
            row = {"entropy": ent, "max_load": float(frac.max()),
                   "dead_experts": dead, "load": load.tolist()}
            if health is not None:
                E = load.shape[0]
                zf = torch.zeros(E)
                for e in range(E):
                    keys = [f"{block}.experts.expert{e}.gate",
                            f"{block}.experts.expert{e}.up",
                            f"{block}.experts.expert{e}.down"]
                    vals = [health[k]["frac_zero"] for k in keys if k in health]
                    zf[e] = sum(vals) / max(len(vals), 1)
                q = max(1, E // 4)
                cold = util.argsort()[:q]                # bottom-quartile utilization
                hot = util.argsort()[-q:]
                row["zero_code_cold_tail"] = float(zf[cold].mean())
                row["zero_code_hot"] = float(zf[hot].mean())
                zero_tail.append(row["zero_code_cold_tail"])
            per_layer[block] = row
            agg["entropy"].append(ent)
            agg["max_load"].append(row["max_load"])
            agg["dead"] += dead
        out = {
            "router/entropy_mean": (sum(agg["entropy"]) / len(agg["entropy"])
                                    if agg["entropy"] else math.nan),
            "router/max_load_mean": (sum(agg["max_load"]) / len(agg["max_load"])
                                     if agg["max_load"] else math.nan),
            "router/dead_experts": agg["dead"],
            "per_layer": per_layer,
        }
        if zero_tail:
            out["router/zero_code_cold_tail"] = sum(zero_tail) / len(zero_tail)
        self._load, self._entropy, self._tokens = {}, {}, {}
        return out


def top8_agreement(student_idx: torch.Tensor, teacher_idx: torch.Tensor) -> float:
    """Per-layer top-k agreement with the teacher's routing (moe §6.2 — the MoE
    analogue of KL_tf; depth-aware interpretation is the caller's job)."""
    k = student_idx.shape[-1]
    s = student_idx.unsqueeze(-1) == teacher_idx.unsqueeze(-2)
    return float(s.any(-1).float().mean() / 1.0) * 1.0 if k else float("nan")
