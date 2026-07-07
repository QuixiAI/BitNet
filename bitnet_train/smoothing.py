"""A7 — pre-conversion outlier smoothing (train_plan §3.3, §9.1).

Rescaling THROUGH an existing norm is exactly function-preserving in FP:
for any RMSNorm -> Linear pair and any positive per-channel s,

    gamma' = gamma / s,   W' = W · diag(s)      =>      W'(x̂ ⊙ gamma') = W(x̂ ⊙ gamma)

(unlike inserting a norm, which is not foldable — §3.1). SmoothQuant-style
choice s_j = amax_x_j^alpha / amax_w_j^(1-alpha) shifts activation-outlier
magnitude into the weights, cutting t=0 activation-quant error BEFORE any
training. Applies only to norm-adjacent pairs: Llama/Qwen `input_layernorm ->
q/k/v_proj` and `post_attention_layernorm -> gate/up_proj`; o_proj/down_proj
have no preceding norm. Gated on the damage map showing activation-dominant
damage (run A7 only then). Apply PRE-conversion, on the dense model.
"""

from __future__ import annotations

import re

import torch
from torch import nn

# (norm attr, [linear attrs]) within one decoder layer — Llama and Qwen3 share names
_PAIRS = (
    ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
)


def _layers(model: nn.Module):
    for name, mod in model.named_modules():
        if re.fullmatch(r"model\.layers\.\d+", name):
            yield name, mod


@torch.no_grad()
def collect_act_stats(model: nn.Module, windows: torch.Tensor, device,
                      batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Per-channel absmax of each pair-norm's OUTPUT (the linears' input) over
    calibration windows. Returns {"<layer>.<norm attr>": amax (H,)}."""
    stats: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(key):
        def hook(_m, _i, out):
            amax = out.detach().float().abs().reshape(-1, out.shape[-1]).amax(0).cpu()
            stats[key] = torch.maximum(stats[key], amax) if key in stats else amax
        return hook

    for lname, layer in _layers(model):
        for norm_attr, _ in _PAIRS:
            handles.append(layer.get_submodule(norm_attr).register_forward_hook(
                make_hook(f"{lname}.{norm_attr}")))
    model.eval()
    for i in range(0, windows.shape[0], batch_size):
        model(windows[i:i + batch_size].to(device))
    for h in handles:
        h.remove()
    return stats


@torch.no_grad()
def apply_smoothing(model: nn.Module, act_stats: dict[str, torch.Tensor],
                    alpha: float = 0.5, min_scale: float = 1e-5) -> dict[str, dict]:
    """Fold s into every discovered pair in place. Returns a per-pair report
    (scale min/mean/max) for the track record. Exactly function-preserving."""
    report = {}
    for lname, layer in _layers(model):
        for norm_attr, lin_attrs in _PAIRS:
            key = f"{lname}.{norm_attr}"
            if key not in act_stats:
                continue
            norm = layer.get_submodule(norm_attr)
            lins = [layer.get_submodule(a) for a in lin_attrs]
            a_amax = act_stats[key].to(norm.weight.device).float().clamp_min(min_scale)
            w_amax = torch.stack([l.weight.detach().float().abs().amax(0)
                                  for l in lins]).amax(0).clamp_min(min_scale)
            s = (a_amax.pow(alpha) / w_amax.pow(1.0 - alpha)).clamp_min(min_scale)
            norm.weight.copy_((norm.weight.float() / s).to(norm.weight.dtype))
            for l in lins:
                l.weight.copy_((l.weight.float() * s.unsqueeze(0)).to(l.weight.dtype))
            report[key] = {"s_min": float(s.min()), "s_mean": float(s.mean()),
                           "s_max": float(s.max())}
    if not report:
        raise ValueError("apply_smoothing: no norm->linear pairs matched "
                         "(unexpected architecture, or stats collected elsewhere)")
    return report


@torch.no_grad()
def smooth_model(model: nn.Module, windows: torch.Tensor, device,
                 alpha: float = 0.5) -> dict[str, dict]:
    """Convenience: collect stats then fold. Run on the DENSE model, before
    conversion (§3.3); the conversion then ternarizes the smoothed latents."""
    stats = collect_act_stats(model, windows, device)
    return apply_smoothing(model, stats, alpha=alpha)
