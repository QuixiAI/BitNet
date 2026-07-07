"""BitLinear with STE on the Metal kernels (build spec: docs/new-kernels.md).

Two interchangeable backends for the same layer:

  reference — pure-PyTorch per-group fake-quant STE (Phase 0). Runs on any device
              (CPU included), is fully autograd-differentiable, and is the
              correctness oracle the Metal path must match.
  metal     — K1 `weight_quant_ternary` + `quantize_per_token_int8`, dense bf16 GEMM
              on the fake-quant operands, dense STE backward (Phase 1). MPS only.
              Inference from frozen packed weights routes to the integer/MMA kernels
              (`qgemv_w2a8` / `qgemm`); routes are measured, not guessed — see
              metal/perf/bitnet_training_kernels.md.

Numerics are pinned to what the vendored kernels actually do (§2 of new-kernels.md:
the forward quantization and the STE fake-quant must be identical):

  * weights: per-group-of-`group_k` (default 32) absmean ternary along K,
    scale = max(mean(|W_g|), 1e-5). Codes are formed against the fp32 scale;
    the DEQUANT uses that scale rounded to float16, because the packed `bitnet`
    block stores a half scale and the GEMM dequantizes with it.
    group_k = K recovers the b1.58 reference's per-tensor scaling.
  * activations: per-token symmetric int8, scale = absmax/127, codes clamped to
    [-127, 127] (tk_int8_encode's range — NOT -128), zero rows quantize to zero.
  * rounding: round-half-to-even everywhere (torch.round == metal rint).
  * STE: both scales are constants in the backward; grad_W flows to the FP latent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_TK = None


def _tk():
    """Lazy tk_torch import so the reference backend works without the Metal toolchain.
    The VENDORED copy (bitnet_train/metal) is pinned ahead of any installed tk_torch
    (e.g. an editable QuixiCore checkout) — the BitNet training kernels and their
    numerics live in the snapshot, not upstream."""
    global _TK
    if _TK is None:
        vendored = str(Path(__file__).resolve().parent / "metal")
        if vendored not in sys.path:
            sys.path.insert(0, vendored)
        import tk_torch as m
        assert hasattr(m, "weight_quant_ternary"), (
            f"tk_torch resolved to {m.__file__} without the BitNet kernels; "
            "expected the vendored bitnet_train/metal copy")
        _TK = m
    return _TK


# ---------------------------------------------------------------------------
# Phase 0 — pure-PyTorch fake-quant (the oracle)
# ---------------------------------------------------------------------------

def weight_quant_pergroup(w: torch.Tensor, group_k: int = 32) -> torch.Tensor:
    """Per-group absmean ternary fake-quant of a (N, K) weight; dequant via the
    half-rounded scale, matching K1's w_deq / the packed-block scale exactly."""
    N, K = w.shape
    assert K % group_k == 0 and group_k % 32 == 0, "group_k must be a multiple of 32 dividing K"
    wg = w.float().reshape(N, K // group_k, group_k)
    s = wg.abs().mean(dim=-1, keepdim=True).clamp_min(1e-5)
    q = (wg / s).round().clamp_(-1.0, 1.0)
    sh = s.to(torch.float16).float()               # the stored (half) scale dequantizes
    return (q * sh).reshape(N, K).to(w.dtype)


def weight_quant_pertensor(w: torch.Tensor) -> torch.Tensor:
    """Per-TENSOR absmean ternary fake-quant (train_plan §4 / moe_train_plan §3.7
    baseline), half-rounded dequant scale — the oracle for weight_quant_ternary_pt."""
    wf = w.float()
    s = wf.abs().mean().clamp_min(1e-5)
    q = (wf / s).round().clamp_(-1.0, 1.0)
    return (q * s.to(torch.float16).float()).to(w.dtype)


def act_quant_int8(x: torch.Tensor) -> torch.Tensor:
    """Per-token symmetric int8 fake-quant over the last axis, kernel-matched:
    scale = absmax/127, clamp [-127, 127], zero rows -> zero."""
    xf = x.float()
    s = xf.abs().amax(dim=-1, keepdim=True) / 127.0
    q = torch.where(s > 0, (xf / s.clamp_min(1e-30)).round().clamp(-127.0, 127.0), xf)
    return (q * s).to(x.dtype)


def bitlinear_reference(x: torch.Tensor, w: torch.Tensor, group_k: int = 32,
                        granularity: str = "group", act_quant: bool = True) -> torch.Tensor:
    """STE BitLinear forward, pure PyTorch (train_plan.md §4 with §2-of-new-kernels scales).
    act_quant=False is the W-only forward (eval modes w_only/a0; A2w/Q-A1w training)."""
    wq_f = weight_quant_pertensor(w) if granularity == "tensor" else weight_quant_pergroup(w, group_k)
    x_q = x + (act_quant_int8(x) - x).detach() if act_quant else x
    w_q = w + (wq_f - w).detach()
    return F.linear(x_q, w_q)


# ---------------------------------------------------------------------------
# Phase 1 — the Metal composition (K1 + reused kernels), STE backward
# ---------------------------------------------------------------------------

class BitLinearSTE(torch.autograd.Function):
    """forward:  quantize with the Metal kernels (K1 weights, per-token int8 acts), then a
    dense bf16 GEMM on the fake-quant operands; backward: grad_x = grad_y @ w_deq,
    grad_W = grad_y^T @ x_q (dense, STE). x: (M, K) float32/bf16 MPS; W: (N, K) latent.
    Scales are constants in the backward.

    Route (measured 2026-07-06, perf/bitnet_training_kernels.md): the integer-exact
    qgemm_w2a8 has no tensor cores (one simdgroup/row) and loses to a dense GEMM on the
    fake-quant operands at every training M — F.linear(x_q, w_deq) is the fast W2A8
    fake-quant forward on Apple (same reasoning as QuixiCore tk/quant.py's A8 note).
    The integer kernels win only at decode (see _infer_w2a8). torch.matmul measured
    equal-or-better than matmul_custom on the backward shapes, without the transpose copy."""

    @staticmethod
    def forward(ctx, x, W, group_k, w_deq=None, act_quant=True):
        tk = _tk()
        if w_deq is None:                                        # uncached path
            _, w_deq = tk.weight_quant_ternary(W, group_k)       # K1 (packed wq unused in training)
        if act_quant:
            x_q, _, _ = tk.fake_quant_int8(x.contiguous())       # K4: one pass -> bf16 grid
        else:
            x_q = x.to(torch.bfloat16)                           # W-only forward (A2w/Q-A1w, eval)
        y = F.linear(x_q, w_deq)                                 # (M, N) bf16 MPS GEMM
        ctx.save_for_backward(w_deq, x_q)
        ctx.dtypes = (x.dtype, W.dtype)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_y):
        w_deq, x_q = ctx.saved_tensors
        x_dtype, w_dtype = ctx.dtypes
        g = grad_y.to(torch.bfloat16)
        grad_x = torch.matmul(g, w_deq)                          # (M,N)@(N,K) -> (M,K)
        grad_W = torch.matmul(g.t(), x_q)                        # (N,M)@(M,K) -> (N,K)
        return grad_x.to(x_dtype), grad_W.to(w_dtype), None, None, None


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------

class BitLinear(nn.Module):
    """Drop-in nn.Linear replacement (bias-free, like Llama/Gemma projections).

    backend='reference' (any device) or 'metal' (MPS). The latent weight stays FP;
    ternary values are recomputed every forward and never stored as a parameter.
    """

    def __init__(self, in_features: int, out_features: int, group_k: int = 32,
                 backend: str = "reference", granularity: str = "group",
                 device=None, dtype=None):
        super().__init__()
        assert backend in ("reference", "metal")
        assert granularity in ("group", "tensor")   # "tensor" = per-tensor absmean (Q-track baseline)
        assert in_features % group_k == 0 and group_k % 32 == 0
        self.in_features, self.out_features = in_features, out_features
        self.group_k, self.backend, self.granularity = group_k, backend, granularity
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               device=device, dtype=dtype or torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.act_quant = True                     # False = W-only forward (eval modes / A2w)
        self._packed = None                       # eval-time cached wq — see freeze()
        self._wcache = (-1, None, None)           # (weight._version, wq, w_deq): the weight
        # only changes at optimizer steps, so grad-accum micro-batches and gradient-checkpoint
        # recomputes reuse one quantization instead of re-running K1 each forward.

    def _quant_weight(self):
        v = self.weight._version
        if self._wcache[0] != v:
            with torch.no_grad():
                if self.granularity == "tensor":
                    wq, w_deq = _tk().weight_quant_ternary_pt(self.weight)
                else:
                    wq, w_deq = _tk().weight_quant_ternary(self.weight, self.group_k)
            self._wcache = (v, wq, w_deq)
        return self._wcache[1], self._wcache[2]

    @torch.no_grad()
    def freeze(self):
        """Pack the current ternary weights once for inference (decode GEMV / prefill GEMM)."""
        self._packed = self._quant_weight()[0]
        return self

    def unfreeze(self):
        self._packed = None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead, K = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, K)
        if self.backend == "reference":
            y = bitlinear_reference(x2, self.weight.to(x2.dtype) if x2.dtype != self.weight.dtype
                                    else self.weight, self.group_k, self.granularity,
                                    self.act_quant)
        elif self._packed is not None and not self.training:
            y = _infer_w2a8(x2, self._packed, self.act_quant)
        else:
            _, w_deq = self._quant_weight()
            y = BitLinearSTE.apply(x2, self.weight, self.group_k, w_deq, self.act_quant)
        return y.reshape(*lead, self.out_features)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias=False, group_k={self.group_k}, backend={self.backend}, "
                f"granularity={self.granularity}")


def _infer_w2a8(x2: torch.Tensor, wq: torch.Tensor, act_quant: bool = True) -> torch.Tensor:
    """W2A8 inference forward from pre-packed blocks (no dense weight materialized):
    integer GEMV at batch 1 (measured ~2x the dense matmul), dequant-to-half MMA
    `qgemm(...,'bitnet')` on fake-quant activations otherwise. act_quant=False is
    the W-only route (eval mode w_only/a0): half activations straight into the MMA."""
    tk = _tk()
    M = x2.shape[0]
    if not act_quant:
        xt = x2.to(torch.float16).t()                            # (K, M) half, no act quant
        Mp = (M + 31) & ~31
        if Mp != M:
            xt = F.pad(xt, (0, Mp - M))
        return tk.qgemm(wq, xt.contiguous(), "bitnet")[:, :M].t().to(x2.dtype)
    xq, a_scale = tk.quantize_per_token_int8(x2.contiguous())
    a_half = a_scale.to(torch.float16)
    if M == 1:
        y = tk.qgemv_w2a8(wq, xq.reshape(-1, 1), a_half)         # (N, 1) integer-exact
    else:
        xt = (xq.to(torch.float16) * a_half.unsqueeze(-1)).t()   # (K, M) half fake-quant
        Mp = (M + 31) & ~31                                      # qgemm needs M % 32 == 0
        if Mp != M:
            xt = F.pad(xt, (0, Mp - M))
        y = tk.qgemm(wq, xt.contiguous(), "bitnet")[:, :M]       # (N, M) half
    return y.t().to(x2.dtype)
