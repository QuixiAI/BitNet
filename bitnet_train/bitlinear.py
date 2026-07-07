"""BitLinear + ternary stats + code-flip tracking (train_plan §7.0 file #2, §7.1 map).

Re-exports the layer from `bitlinear_metal.py` and adds the model-level machinery
the plans mandate from step 0:

  * eval modes (train_plan §8.4 / moe_train_plan §7.5): w_a8 (= a1, the training
    forward), w_only (= a0, activation quant off), b (a1 + per-tensor e4m3
    fake-quant on the keep-FP linears/embeddings — the Q-T4 cast delta's eval side);
  * the damage-map toggles (train_plan §11.1: A1d / A1b / A2 / module-family subsets);
  * code snapshots + flip rates (§10.2 — the LR sweep's mechanistic readout) and
    per-layer ternary health (§10.2 / moe §6.2), all portable with an MPS fast path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from bitnet_train import quant
from bitnet_train.bitlinear_metal import (  # noqa: F401  (re-exports)
    BitLinear,
    BitLinearSTE,
    act_quant_int8,
    bitlinear_reference,
)

EVAL_MODES = ("w_a8", "w_only", "a0", "a1", "b")
_DEFAULT_FP8_EXCLUDE = (r".*\.mlp\.gate",)          # the Qwen3 router: BF16 forever


def iter_bitlinears(model: nn.Module) -> Iterator[tuple[str, BitLinear]]:
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            yield name, mod


# ---------------------------------------------------------------------------
# fused expert stacks (Q-track, transformers 5 layout)
# ---------------------------------------------------------------------------

class BitExperts(nn.Module):
    """Ternary-QAT drop-in for transformers 5's fused MoE experts module
    (Qwen3MoeExperts-shaped: gate_up_proj (E, 2I, H), down_proj (E, H, I)).

    Q-T0 recon fact (recorded here, 2026-07-06, transformers 5.13): the v5
    qwen3_moe implementation fuses all experts into two 3-D parameters and the
    router is a Qwen3MoeTopKRouter, NOT nn.Linear — the per-expert
    `mlp.experts.N.gate_proj` Linears of the plan doc's §1.2 regexes only exist
    in transformers 4. Conversion therefore swaps the experts MODULE, keeping
    parameter names/shapes state-dict-identical.

    Numerics: per-tensor absmean ternary per LOGICAL matrix — each expert's gate
    slice, up slice, and down matrix get their own scale (the fused gate_up rows
    [0:I] and [I:2I] are two tensors in the plan's sense); per-token int8
    activations on both expert inputs (h before down included, matching the CPU
    engine's bn_expert_ffn). STE throughout; latents stay fp32. This is the
    reference (correctness) path — the fused Metal stacked-expert module is a
    deferred optimization (plan R6).
    """

    def __init__(self, experts_mod: nn.Module, granularity: str = "tensor",
                 group_k: int = 32):
        super().__init__()
        assert granularity in ("tensor", "group")
        self.num_experts = experts_mod.num_experts
        self.act_fn = experts_mod.act_fn
        self.granularity, self.group_k = granularity, group_k
        self.act_quant = True
        self.lam = 1.0                               # A3/Q-A3 ramp (see set_lambda)
        gu, dn = experts_mod.gate_up_proj, experts_mod.down_proj
        self.intermediate = gu.shape[1] // 2
        self.gate_up_proj = nn.Parameter(gu.detach().float().clone())
        self.down_proj = nn.Parameter(dn.detach().float().clone())

    def _fq(self, w2d: torch.Tensor) -> torch.Tensor:
        wq = quant.weight_quant(w2d, self.granularity, self.group_k)
        return w2d + self.lam * (wq - w2d).detach()  # STE on the latent slice (+A3 ramp)

    def _aq(self, x: torch.Tensor) -> torch.Tensor:
        if not self.act_quant:
            return x
        return x + (act_quant_int8(x) - x).detach()

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        idt = hidden_states.dtype
        for e in hit:
            e = int(e[0])
            top_k_pos, token_idx = torch.where(mask[e])
            x = self._aq(hidden_states[token_idx].float())
            wgu = self.gate_up_proj[e]
            w_q = torch.cat([self._fq(wgu[: self.intermediate]),
                             self._fq(wgu[self.intermediate:])], dim=0)
            gate, up = F.linear(x, w_q).chunk(2, dim=-1)
            h = self._aq(self.act_fn(gate) * up)
            out = F.linear(h, self._fq(self.down_proj[e]))
            final.index_add_(0, token_idx,
                             (out * top_k_weights[token_idx, top_k_pos, None]).to(idt))
        return final

    def expert_slices(self) -> Iterator[tuple[str, torch.Tensor]]:
        """The logical 2-D ternary matrices (for health panels / baking):
        (name, latent) per (expert, {gate,up,down})."""
        for e in range(self.num_experts):
            wgu = self.gate_up_proj[e]
            yield f"expert{e}.gate", wgu[: self.intermediate]
            yield f"expert{e}.up", wgu[self.intermediate:]
            yield f"expert{e}.down", self.down_proj[e]


def iter_bitexperts(model: nn.Module) -> Iterator[tuple[str, BitExperts]]:
    for name, mod in model.named_modules():
        if isinstance(mod, BitExperts):
            yield name, mod


# ---------------------------------------------------------------------------
# eval modes
# ---------------------------------------------------------------------------

class _FP8FakeQuant(nn.Module):
    """Weight parametrization: per-tensor e4m3 fake-quant (mode b). MPS uses the
    bit-exact fake_quant_fp8 kernel; elsewhere torch.float8_e4m3fn round-trip."""

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if w.device.type == "mps":
                from bitnet_train.bitlinear_metal import _tk
                return _tk().fake_quant_fp8(w)[0]
            s = w.float().abs().max().clamp_min(1e-12) / 448.0
            return ((w.float() / s).to(torch.float8_e4m3fn).float() * s).to(w.dtype)


def set_eval_mode(model: nn.Module, mode: str,
                  fp8_exclude: tuple[str, ...] = _DEFAULT_FP8_EXCLUDE) -> nn.Module:
    """Flip the model between the parity-matrix eval modes. Idempotent; call with
    'w_a8'/'a1' to restore the training forward. Mode 'b' adds a removable weight
    parametrization to every non-BitLinear nn.Linear and nn.Embedding not matching
    fp8_exclude (the router never quantizes — moe_train_plan §3.2)."""
    import re

    if mode not in EVAL_MODES:
        raise ValueError(f"mode must be one of {EVAL_MODES}, got {mode!r}")
    aq = mode in ("w_a8", "a1", "b")
    for _, mod in iter_bitlinears(model):
        mod.act_quant = aq
    for _, mod in iter_bitexperts(model):
        mod.act_quant = aq

    from torch.nn.utils import parametrize
    want_fp8 = (mode == "b")
    for name, mod in model.named_modules():
        is_target = (isinstance(mod, (nn.Linear, nn.Embedding))
                     and not isinstance(mod, BitLinear)
                     and not any(re.fullmatch(p, name) for p in fp8_exclude))
        has = parametrize.is_parametrized(mod, "weight")
        if want_fp8 and is_target and not has:
            parametrize.register_parametrization(mod, "weight", _FP8FakeQuant())
        elif not want_fp8 and has and any(isinstance(p, _FP8FakeQuant)
                                          for p in mod.parametrizations.weight):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
    return model


def set_lambda(model: nn.Module, lam: float) -> None:
    """A3/Q-A3 one-flag stability ramp: w_eff = (1-lam)*w + lam*quant(w) on every
    ternary module. Ramp 0 -> 1 over warmup only if step-0 spikes (train_plan §9.1);
    lam = 1 (default) is the plain quantized forward."""
    lam = float(min(max(lam, 0.0), 1.0))
    for _, mod in iter_bitlinears(model):
        mod.lam = lam
    for _, mod in iter_bitexperts(model):
        mod.lam = lam


# ---------------------------------------------------------------------------
# damage-map toggles (eval-only; train_plan §11.1)
# ---------------------------------------------------------------------------

@contextmanager
def quant_toggled(model: nn.Module, names: set[str] | None = None,
                  act_quant: bool = True, weight_ternary: bool = True):
    """Temporarily override selected BitLinears with a functional eval forward:
    weight_ternary=False -> dense latent weight (A1b: A8 only); act_quant=False ->
    FP activations (A1d: ternary W only). names=None affects every BitLinear;
    otherwise only the named subset (module-family / per-layer damage passes)."""
    saved = []
    for name, mod in iter_bitlinears(model):
        if names is not None and name not in names:
            continue
        saved.append((mod, mod.__dict__.get("forward")))
        mod.forward = _damage_forward(mod, act_quant, weight_ternary)
    try:
        yield model
    finally:
        for mod, prev in saved:
            if prev is None:
                mod.__dict__.pop("forward", None)
            else:
                mod.forward = prev


def _damage_forward(mod: BitLinear, aq: bool, wq_on: bool):
    def fwd(x: torch.Tensor) -> torch.Tensor:
        w = mod.weight
        if wq_on:
            w = quant.weight_quant(w, mod.granularity, mod.group_k)
        if aq:
            x = act_quant_int8(x)
        return F.linear(x.to(w.dtype), w).to(x.dtype)
    return fwd


# ---------------------------------------------------------------------------
# code snapshots / flip rates / ternary health
# ---------------------------------------------------------------------------

def snapshot_codes(model: nn.Module) -> dict[str, torch.Tensor]:
    """Per-BitLinear ternary code snapshot, held on CPU between eval intervals.
    Metal backend: the packed uint8 wq (reuses the module's weight-version cache,
    ~params/3.2 bytes). Reference backend: int8 codes from quant.ternary_codes."""
    out = {}
    for name, mod in iter_bitlinears(model):
        if mod.backend == "metal":
            out[name] = mod._quant_weight()[0].cpu()
        else:
            out[name] = quant.ternary_codes(mod.weight, mod.granularity,
                                            mod.group_k)[0].cpu()
    for name, mod in iter_bitexperts(model):
        for slice_name, w in mod.expert_slices():
            out[f"{name}.{slice_name}"] = quant.ternary_codes(
                w, mod.granularity, mod.group_k)[0].cpu()
    return out


def _packed_flips(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """Count differing 2-bit codes between two packed (..., nblocks, 10) tensors.
    Pure torch on CPU — no unpack, no device round-trip."""
    qa = a.reshape(-1, 10)[:, 2:]
    qb = b.reshape(-1, 10)[:, 2:]
    x = torch.bitwise_xor(qa, qb)
    flips = 0
    for s in range(0, 8, 2):
        flips += int((torch.bitwise_and(torch.bitwise_right_shift(x, s), 3) != 0).sum())
    return flips, qa.numel() * 4


def code_flip_rates(prev: dict[str, torch.Tensor],
                    curr: dict[str, torch.Tensor]) -> dict[str, float]:
    """Per-layer fraction of ternary codes that changed between two snapshots,
    plus '_total'. Near-zero early = frozen effective model (the low-LR failure);
    sustained very high = thrashing (train_plan §10.2)."""
    rates, tot_f, tot_n = {}, 0, 0
    for name, a in prev.items():
        b = curr[name]
        if a.dtype == torch.uint8:
            f, n = _packed_flips(a, b)
        else:
            f, n = int((a != b).sum()), a.numel()
        rates[name] = f / max(n, 1)
        tot_f += f
        tot_n += n
    rates["_total"] = tot_f / max(tot_n, 1)
    return rates


@torch.no_grad()
def ternary_health(model: nn.Module) -> dict[str, dict[str, float]]:
    """Per-BitLinear {-1,0,+1} code fractions, absmean scale, latent norm, and
    relative quantization error (train_plan §10.2 / moe_train_plan §6.2 panel)."""
    out = {}
    for name, mod in iter_bitlinears(model):
        w = mod.weight
        if mod.backend == "metal":
            wq, w_deq = mod._quant_weight()
            from bitnet_train.bitlinear_metal import _tk
            counts = _tk().ternary_stats(wq).sum(dim=0).cpu()    # (3,) over all rows
        else:
            codes, _ = quant.ternary_codes(w, mod.granularity, mod.group_k)
            counts = torch.stack([(codes == v).sum() for v in (-1, 0, 1)]).cpu()
            w_deq = quant.weight_quant(w, mod.granularity, mod.group_k)
        n = float(w.numel())
        wn = float(w.detach().float().norm())
        out[name] = {
            "frac_neg": float(counts[0]) / n,
            "frac_zero": float(counts[1]) / n,
            "frac_pos": float(counts[2]) / n,
            "absmean_scale": float(w.detach().float().abs().mean()),
            "latent_norm": wn,
            "quant_rel_err": float((w_deq.float() - w.float()).norm()) / max(wn, 1e-20),
        }
    for name, mod in iter_bitexperts(model):
        for slice_name, w in mod.expert_slices():
            codes, _ = quant.ternary_codes(w, mod.granularity, mod.group_k)
            w_deq = quant.weight_quant(w, mod.granularity, mod.group_k)
            n, wn = float(w.numel()), float(w.detach().float().norm())
            out[f"{name}.{slice_name}"] = {
                "frac_neg": float((codes == -1).sum()) / n,
                "frac_zero": float((codes == 0).sum()) / n,
                "frac_pos": float((codes == 1).sum()) / n,
                "absmean_scale": float(w.detach().float().abs().mean()),
                "latent_norm": wn,
                "quant_rel_err": float((w_deq.float() - w.float()).norm()) / max(wn, 1e-20),
            }
    return out
