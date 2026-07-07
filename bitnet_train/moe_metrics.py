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
        self.capture_routing = False                    # per-token top-k for §6.2 agreement
        self._aux_logits: list[torch.Tensor] = []
        self._routed: dict[int, set[int]] = {}          # id(experts_mod) -> expert ids
        self._load: dict[str, torch.Tensor] = {}        # layer -> expert token counts
        self._entropy: dict[str, list[float]] = {}
        self._tokens: dict[str, int] = {}
        self._routing: dict[str, list] = {}             # block -> per-token top-k ids

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
                if self.capture_routing:
                    self._routing.setdefault(block, []).append(
                        idx.reshape(-1, idx.shape[-1]).cpu())
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

    # ---- per-layer top-8 teacher agreement (§6.2), after a capture pass ----
    def routing_and_reset(self) -> dict[str, torch.Tensor]:
        out = {b: torch.cat(v, 0) for b, v in self._routing.items() if v}
        self._routing = {}
        return out

    def agreement_vs_teacher(self, teacher_routing: dict[str, torch.Tensor]) -> dict:
        """Per-block top-k agreement of the just-captured student routing vs the
        teacher fixture (same calibration windows, same order). Returns
        {block: agreement} + a depth-ordered '_by_depth' list + '_mean'."""
        student = self.routing_and_reset()
        per, depth = {}, []
        for block in sorted(student, key=lambda b: int(b.split(".layers.")[1].split(".")[0])
                            if ".layers." in b else 0):
            if block not in teacher_routing:
                continue
            n = min(student[block].shape[0], teacher_routing[block].shape[0])
            a = top8_agreement(student[block][:n], teacher_routing[block][:n])
            per[block] = a
            depth.append(a)
        per["_by_depth"] = depth
        per["_mean"] = sum(depth) / len(depth) if depth else float("nan")
        return per

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
    """Per-token fraction of the student's selected experts that are also in the
    teacher's selection (moe §6.2 — the MoE analogue of KL_tf). Depth-aware
    interpretation is the caller's job: only layer 1 is exact at t=0; deeper
    routers see inputs already carrying accumulated error, so expect a depth
    gradient dipping then re-stabilizing."""
    k = student_idx.shape[-1]
    if not k:
        return float("nan")
    hit = (student_idx.unsqueeze(-1) == teacher_idx.unsqueeze(-2)).any(-1)
    return float(hit.float().mean())


@torch.no_grad()
def capture_teacher_routing(teacher, windows: torch.Tensor, device,
                           top_k: int, batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Q-T0 §8.1 item 8: the teacher's per-layer routed top-k on the calibration
    set, the fixture the §6.2 agreement metric compares against. Keyed by the
    experts-block name (matching RouterHooks.pairs), value (n_tokens, top_k)."""
    hooks = []
    # the teacher is dense — hook every mlp.gate router (Qwen3MoeTopKRouter, whose
    # forward returns (logits, scores, top_k_indices)); block key matches
    # RouterHooks.pairs (name up to '.experts' / '.gate').
    routers = [(name.rsplit(".gate", 1)[0], mod)
               for name, mod in teacher.named_modules()
               if name.endswith(".mlp.gate") and hasattr(mod, "weight")]

    store: dict[str, list] = {b: [] for b, _ in routers}

    def mk(block):
        def hook(_m, _i, out):
            idx = out[2] if isinstance(out, tuple) and len(out) >= 3 else None
            if idx is not None:
                store[block].append(idx.reshape(-1, idx.shape[-1]).cpu())
        return hook

    for block, r in routers:
        hooks.append(r.register_forward_hook(mk(block)))
    teacher.eval()
    for i in range(0, windows.shape[0], batch_size):
        teacher(windows[i:i + batch_size].to(device))
    for h in hooks:
        h.remove()
    return {b: torch.cat(v, 0) for b, v in store.items() if v}
