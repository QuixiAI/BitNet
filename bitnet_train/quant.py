"""Quantizer math — the single import point (train_plan §7.0 file #1).

Everything here is portable pure-PyTorch (CPU/CUDA/MPS) and re-exports the
existing oracles in `bitlinear_metal.py` rather than duplicating them. The
canonical numerics (the delta from train_plan §4's pseudocode, decided once):

  * ternary codes are formed against the fp32 absmean scale, but the DEQUANT
    scale is that value rounded to float16 — the packed `bitnet` block and
    GGUF block scales both store f16, so training, export baking, and the
    parity gate all live on the same grid;
  * activations clamp to [-127, 127] (tk_int8_encode's range, not -128);
  * rounding is round-half-to-even everywhere.

Baseline granularity is per-TENSOR absmean (train_plan §3.4 / moe_train_plan
§3.7): the only choice where baked {-s, 0, +s} tensors survive I2_S and TQ2_0
block re-quantization with exact codes. Per-group is the ablation path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from bitnet_train.bitlinear_metal import (  # noqa: F401  (re-exports)
    act_quant_int8,
    bitlinear_reference,
    weight_quant_pergroup,
    weight_quant_pertensor,
)

QUANTIZER_VERSION = "1"                    # bump on ANY numerics change


def quantizer_hash() -> str:
    """sha256 over the quantizer sources + version — pinned into checkpoints,
    bake reports, and parity reports (train_plan §5.6 / §8.3 provenance)."""
    h = hashlib.sha256()
    h.update(QUANTIZER_VERSION.encode())
    here = Path(__file__).resolve().parent
    for src in (here / "quant.py", here / "bitlinear_metal.py"):
        h.update(src.read_bytes())
    return h.hexdigest()[:16]


def weight_quant(w: torch.Tensor, granularity: str = "tensor",
                 group_k: int = 32) -> torch.Tensor:
    """Fake-quant dispatch under the plan docs' name (train_plan §4 weight_quant)."""
    if granularity == "tensor":
        return weight_quant_pertensor(w)
    if granularity == "group":
        return weight_quant_pergroup(w, group_k)
    raise ValueError(f"granularity must be 'tensor' or 'group', got {granularity!r}")


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token absmax int8 fake-quant (train_plan §4 activation_quant)."""
    return act_quant_int8(x)


def ternary_codes(w: torch.Tensor, granularity: str = "tensor",
                  group_k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Latent weight -> (codes int8 in {-1,0,+1}, fp32 scale). Portable — feeds
    export baking, parity decoding, and non-MPS code-flip snapshots.
    scale shape: () for per-tensor, (N, K/group_k) for per-group.
    Dequant convention: codes * scale.to(float16).float()."""
    wf = w.detach().float()
    if granularity == "tensor":
        s = wf.abs().mean().clamp_min(1e-5)
        q = (wf / s).round().clamp_(-1.0, 1.0).to(torch.int8)
        return q, s
    if granularity == "group":
        N, K = wf.shape
        assert K % group_k == 0 and group_k % 32 == 0
        wg = wf.reshape(N, K // group_k, group_k)
        s = wg.abs().mean(dim=-1).clamp_min(1e-5)
        q = (wg / s.unsqueeze(-1)).round().clamp_(-1.0, 1.0).to(torch.int8)
        return q.reshape(N, K), s
    raise ValueError(f"granularity must be 'tensor' or 'group', got {granularity!r}")


def dequant_codes(codes: torch.Tensor, scale: torch.Tensor,
                  group_k: int = 32) -> torch.Tensor:
    """Inverse of ternary_codes on the f16-rounded grid (what the runtimes see)."""
    sh = scale.to(torch.float16).float()
    if scale.dim() == 0:
        return codes.float() * sh
    N, K = codes.shape
    return (codes.float().reshape(N, K // group_k, group_k)
            * sh.unsqueeze(-1)).reshape(N, K)


def lambda_ramp(w: torch.Tensor, lam: float, granularity: str = "tensor",
                group_k: int = 32) -> torch.Tensor:
    """A3/Q-A3 stability ablation, one flag: w_eff = (1-lam)*w + lam*weight_quant(w).
    The fixed ramp stays outside the backward graph (train_plan §4 caveat)."""
    if lam <= 0.0:
        return w
    wq = weight_quant(w, granularity, group_k)
    if lam >= 1.0:
        return wq
    return (1.0 - lam) * w + lam * wq
