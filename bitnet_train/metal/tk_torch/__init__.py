"""PyTorch/MPS backend for the BitNet-training Metal kernels.

Vendored from QuixiCore-Metal (`bindings/pytorch_mps`), trimmed to the kernels the BitNet
healing trainer needs. The compute lives in framework-agnostic `.metal` sources under
`../kernels`; this package:
  1. compiles them into a standalone `bitnet.metallib` with `xcrun metal` (no MLX, no CMake), and
  2. JIT-compiles a thin ObjC++ extension (`torch.utils.cpp_extension.load`) that dispatches
     those kernels onto PyTorch's MPS command stream.

Requirements: PyTorch (MPS build) + Xcode's Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`). No MLX, no CMake.

Layout (vendored root = `bitnet_train/metal`):
    include/metal/**            header-only substrate (ThunderMittens-derived; see ../LICENSE)
    kernels/common/tk_launch.h  host dispatch (all launchers; self-contained)
    kernels/<family>/*.metal    the curated kernel subset
    kernels/bitnet/*.metal      NEW BitNet-training kernels (auto-compiled; see docs/new-kernels.md)
    tk_torch/torch_kernels.mm   ObjC++ dispatch onto torch's MPS stream (verbatim from QuixiCore)
"""

import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent                     # bitnet_train/metal (the vendored root)
_KERNELS = _REPO_ROOT / "kernels"
_INCLUDE = _REPO_ROOT / "include" / "metal"
_KERNEL_COMMON = _KERNELS / "common"
_METALLIB = _HERE / "bitnet.metallib"


def _kernel_source(path: str) -> Path:
    return _KERNELS / path


# The curated .metal kernel sources for BitNet healing. Adding a file here (or dropping a new
# .metal into kernels/bitnet/) makes it part of the compiled metallib.
_METAL_SOURCES = [
    # activation int8 quant (per-token absmax) + fused gated-act+quant
    _kernel_source("quantization/quant_rt/quant_rt.metal"),
    _kernel_source("quantization/act_quant/act_quant.metal"),
    # BitNet ternary matmuls: dequant-to-half (fast) + integer (exact) + decode GEMV
    _kernel_source("quantization/qgemm/qgemm.metal"),
    _kernel_source("quantization/qgemm_int/qgemm_int.metal"),
    _kernel_source("quantization/qgemv/qgemv.metal"),
    _kernel_source("quantization/qgemv_int/qgemv_int.metal"),
    # dense bf16/f32 GEMMs for the STE backward products
    _kernel_source("matmul/matmul_custom/matmul_custom.metal"),
    _kernel_source("matmul/gemm_staged/gemm_staged.metal"),
    _kernel_source("matmul/gemm_v3/gemm_v3.metal"),
    # MoE pipeline (routing, permute/pad/gather, grouped expert GEMMs incl. the
    # "bitnet" ternary-expert instantiation, finalize) — moe_train_plan tracks
    _kernel_source("moe/moe/moe.metal"),
    # whole-model hot path
    _kernel_source("norms/rms_norm/rms_norm.metal"),
    _kernel_source("activations/glu/glu.metal"),
    _kernel_source("utils/cross_entropy/cross_entropy.metal"),
    _kernel_source("optimizers/optim/adamw.metal"),
]
# NEW BitNet-specific kernels we develop live in kernels/bitnet/ and auto-compile.
_METAL_SOURCES += sorted((_KERNELS / "bitnet").glob("*.metal"))


def build_metallib(force: bool = False) -> str:
    """Compile the .metal kernels into bitnet.metallib via xcrun metal. MLX-independent."""
    sources = [s for s in _METAL_SOURCES if s.exists()]
    if not force and _METALLIB.exists():
        # staleness must also track the header-only substrate under include/ (tk.metal pulls
        # in everything there), not just the listed kernel sources
        deps = list(sources)
        deps.extend(_INCLUDE.rglob("*.metal"))
        newest_src = max(s.stat().st_mtime for s in deps)
        if _METALLIB.stat().st_mtime >= newest_src:
            return str(_METALLIB)
    cmd = ["xcrun", "metal", "-std=metal3.1", "-O2", "-I", str(_INCLUDE),
           *map(str, sources), "-o", str(_METALLIB)]
    subprocess.run(cmd, check=True)
    return str(_METALLIB)


# Build the metallib (if missing/stale) and the ObjC++ extension on import.
build_metallib()

_ext = load(
    name="bitnet_metal_ext",
    sources=[str(_HERE / "torch_kernels.mm")],
    extra_cflags=["-std=c++17"],
    extra_include_paths=[str(_KERNEL_COMMON)],
    extra_ldflags=["-framework", "Metal", "-framework", "Foundation", "-framework", "QuartzCore"],
    verbose=False,
)
_ext._set_library(str(_METALLIB))


# ---------------------------------------------------------------------------
# Training-relevant API (a focused subset of the QuixiCore torch binding).
# The full op set is still registered in torch_kernels.mm (accessible via `_ext`),
# but only the kernels compiled into bitnet.metallib above are callable.
# ---------------------------------------------------------------------------

# --- normalization (RMSNorm fwd + fused backward) ---

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """RMSNorm over the last axis. bf16 MPS tensors; D in {256,512,768,1024}."""
    return _ext.rms_norm(x, weight, float(eps))


def rms_norm_bwd_fused(x, weight, dy, eps):
    """Fused RMSNorm backward -> (dx, dweight) in one pass. MPS."""
    return tuple(_ext.rms_norm_bwd_fused(x, weight, dy, float(eps)))


# --- dense GEMM (STE backward products; also FP fallback) ---

def _ceil(a, m):
    return ((a + m - 1) // m) * m


def matmul_custom(x: torch.Tensor, y: torch.Tensor):
    """(N,K) @ (K,M) GEMM, arbitrary shapes (f32/bf16, MPS). Zero-pads to tile multiples
    (N%32, M%32, K%16) and slices back."""
    import torch.nn.functional as F
    N, K = x.shape[-2], x.shape[-1]
    M = y.shape[-1]
    Np, Kp, Mp = _ceil(N, 32), _ceil(K, 16), _ceil(M, 32)
    xp = F.pad(x, (0, Kp - K, 0, Np - N)) if (Np != N or Kp != K) else x
    yp = F.pad(y, (0, Mp - M, 0, Kp - K)) if (Kp != K or Mp != M) else y
    out = _ext.matmul_custom(xp.contiguous(), yp.contiguous())
    return out[:N, :M].contiguous()


# --- GLU family (Llama FFN is SwiGLU) fwd + bwd ---

def glu(x: torch.Tensor, gate: torch.Tensor, mode: str = "swiglu",
        alpha: float = 1.0, limit: float = 1.0e20):
    """GLU-family activation. mode in reglu/geglu/swiglu/swiglu_oai/geglu_erf/geglu_quick."""
    return _ext.glu(x, gate, mode, float(alpha), float(limit))


def glu_backward(x: torch.Tensor, gate: torch.Tensor, dc: torch.Tensor, mode: str = "swiglu",
                 alpha: float = 1.0, limit: float = 1.0e20):
    """GLU-family backward. Returns (da, db) = grads wrt x, gate given upstream grad dc. MPS."""
    return tuple(_ext.glu_backward(x, gate, dc, mode, float(alpha), float(limit)))


def swiglu(x: torch.Tensor, gate: torch.Tensor):
    return glu(x, gate, "swiglu")


# --- fused cross-entropy (never materializes (T,V) probs) ---

def cross_entropy_fwd(logits, targets, ignore_index=-100, label_smoothing=0.0, z_loss=0.0,
                      softcap=0.0):
    """Fused cross-entropy forward. Returns (loss (T,), lse (T,)) f32. MPS."""
    return _ext.cross_entropy_fwd(logits, targets, int(ignore_index), float(label_smoothing),
                                  float(z_loss), float(softcap))


def cross_entropy_bwd(logits, targets, lse, grad_out, ignore_index=-100, label_smoothing=0.0,
                      z_loss=0.0, softcap=0.0):
    """Fused cross-entropy backward -> grad_logits (T,V), out-of-place. MPS."""
    return _ext.cross_entropy_bwd(logits, targets, lse, grad_out, int(ignore_index),
                                  float(label_smoothing), float(z_loss), float(softcap))


# --- sparse-teacher KD-KL (top-k cache distillation loss) ---

def kd_kl_topk_fwd(logits, t_idx, t_prob, invtemp=1.0, tail_mode=0):
    """KD-KL against a top-k teacher cache. Returns (loss (T,), lse (T,)) f32.
    tail_mode 0 = renormalize teacher mass over top-k; 1 = one 'other' bucket.
    Pass invtemp = 1/tau; scale loss/grad_out by alpha*tau^2 yourself. MPS."""
    return tuple(_ext.kd_kl_topk_fwd(logits, t_idx, t_prob, float(invtemp), int(tail_mode)))


def kd_kl_topk_bwd(logits, t_idx, t_prob, lse, grad_out, invtemp=1.0, tail_mode=0):
    """Backward for kd_kl_topk_fwd -> grad_logits (T, V), out-of-place. MPS."""
    return _ext.kd_kl_topk_bwd(logits, t_idx, t_prob, lse, grad_out, float(invtemp),
                               int(tail_mode))


# --- fused AdamW step ---

def adamw(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step):
    """AdamW step. Returns (param', m', v'); m/v fp32 moment state, step (t) >= 1. MPS."""
    return tuple(_ext.adamw(param, grad, m, v, float(lr), float(beta1), float(beta2),
                            float(eps), float(weight_decay), int(step)))


def adamw_masked(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step,
                 mask, seg_size, mask_mode=0):
    """Cold-expert masked AdamW (moe_train_plan §4.3): element i belongs to segment
    i//seg_size; mask (uint8[S]) == 0 skips that segment's update (mask_mode=0) or just
    its decay term (mask_mode=1) — decoupled decay must not erode unrouted experts. MPS."""
    return tuple(_ext.adamw_masked(param, grad, m, v, float(lr), float(beta1), float(beta2),
                                   float(eps), float(weight_decay), int(step),
                                   mask, int(seg_size), int(mask_mode)))


# --- activation int8 quantization (per-token absmax) ---

def quantize_per_token_int8(x: torch.Tensor):
    """Per-token (last-axis) symmetric int8 absmax quant. Returns (codes i8, scale f32). MPS."""
    return tuple(_ext.quantize_per_token_int8(x))


def fake_quant_int8(x: torch.Tensor):
    """One-pass per-token int8 FAKE-quant (K4): returns (x_q bf16 on the half-rounded
    grid, codes i8, scale f32). Replaces quantize_per_token_int8 + eager dequant. MPS."""
    return tuple(_ext.fake_quant_int8(x))


def silu_mul_fake_quant_int8(x, gate, act="swiglu", alpha=1.702, limit=7.0):
    """Fused SwiGLU + one-pass int8 fake-quant (K4 FFN epilogue). Returns
    (x_q bf16, codes i8, scale f32). x = activated operand, gate = multiplier. MPS."""
    _modes = {"swiglu": 0, "swiglu_oai": 1}
    return tuple(_ext.silu_mul_fake_quant_int8(x, gate, _modes[act], float(alpha), float(limit)))


def silu_mul_quant_int8(x, gate, act="swiglu", alpha=1.702, limit=7.0):
    """Fused gated-activation -> dynamic per-token int8 (feeds qgemm_w8a8). Returns (codes,scale). MPS."""
    _modes = {"swiglu": 0, "swiglu_oai": 1}
    return tuple(_ext.silu_mul_quant_int8(x, gate, _modes[act], float(alpha), float(limit)))


# --- BitNet weight quantization (K1/K5, docs/new-kernels.md §3) ---

def weight_quant_ternary(W: torch.Tensor, group_k: int = 32):
    """Latent weight (N,K) or expert stack (E,N,K) -> per-GROUP absmean ternary, one pass:
    (wq uint8 (..., K/32, 10) packed bitnet blocks for qgemm_w2a8/qgemm,
     w_deq bf16 (like W) dequantized ternary for the dense STE backward).
    group_k must be a multiple of 32 dividing K. NOTE group_k=K is per-ROW, not
    per-tensor — use weight_quant_ternary_pt for the per-tensor baseline. MPS."""
    return tuple(_ext.weight_quant_ternary(W, int(group_k)))


def weight_quant_ternary_pt(W: torch.Tensor):
    """Latent weight (N,K) or expert stack (E,N,K) -> per-TENSOR absmean ternary (one
    scale per (N,K) slice, the train_plan §4 / moe_train_plan §3.7 baseline; replicated
    into every packed block so the GEMM layout is unchanged). Two fused passes. MPS."""
    return tuple(_ext.weight_quant_ternary_pt(W))


# --- MoE with ternary experts (moe_train_plan; eval / OPD-rollout prefill path) ---

def moe_route_topk(logits: torch.Tensor, k: int):
    """Top-k routing: (T, E) logits -> (topk_ids i32 (T,k), topk_weights f32 (T,k)),
    softmax renormalized over the selected experts (Qwen3 norm_topk_prob=True). MPS."""
    return tuple(_ext.moe_route_topk(logits, int(k)))


def moe_ffn_bitnet(x: torch.Tensor, w1q: torch.Tensor, w2q: torch.Tensor,
                   topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                   act: str = "swiglu", alpha: float = 1.702, limit: float = 7.0):
    """Fused MoE FFN over packed TERNARY experts (W-only activations — the a0/rollout
    path): permute/pad/gather -> swiglu_q GEMM1 -> rect_q GEMM2 -> weighted finalize.
    x (T, H) bf16; w1q (E, 2*inter, H/32*10) packed [gate|up]; w2q (E, H, inter/32*10);
    both from weight_quant_ternary on (E, 2*inter, H) / (E, H, inter) stacks. MPS."""
    T, H = x.shape
    E, k = w1q.shape[0], topk_ids.shape[1]
    sorted_idx, offsets, _ = _ext.moe_permute(topk_ids, E)
    eot, gather_idx, inv_pad, _ = _ext.moe_pad_schedule(sorted_idx, offsets, k)
    A = _ext.moe_gather(x, gather_idx)                                   # (P, H)
    dummy = torch.zeros(1, 1, dtype=torch.bfloat16, device=x.device)
    mode = {"swiglu": 0, "swiglu_oai": 1}[act]
    h = _ext.moe_grouped_gemm_swiglu_q(A, w1q, eot, dummy, False, mode,
                                       float(alpha), float(limit), "bitnet")
    y = _ext.moe_grouped_gemm_rect_q(h, w2q, eot, dummy, False, "bitnet")
    return _ext.moe_finalize(y, inv_pad, topk_weights, k)                # (T, H)


# --- BitNet ternary matmuls (forward path) ---
# wq: packed uint8 BitNet blocks (group=32, per-group absmean scale, 10 bytes/32 weights).
# xq: int8 activations from quantize_per_token_int8. a_scale: per-token f16 scale.

def qgemm_w2a8(wq, xq, a_scale):
    """BitNet W2A8 prefill GEMM (integer-exact, int32 accum). y = (W_ternary @ Xq^T) * gscale * a_scale. MPS."""
    return _ext.qgemm_w2a8(wq, xq, a_scale)


def qgemv_w2a8(wq, xq, a_scale, version=2):
    """BitNet W2A8 decode GEMV (batch-1). version=2: one block per lane, arithmetic
    2-bit->int8 spread + idot4 (the measured winner); version=1: 4-lanes-per-block. MPS."""
    return _ext.qgemv_w2a8(wq, xq, a_scale, int(version))


def qgemm(wq: torch.Tensor, x: torch.Tensor, format: str = "bitnet"):
    """Dequant-to-half quantized GEMM (fast prefill; uses half tensor cores). format='bitnet'. MPS."""
    return _ext.qgemm(wq, x, format)


def qgemv(wq: torch.Tensor, x: torch.Tensor, format: str = "bitnet"):
    """Dequant-to-half quantized GEMV decode. format='bitnet'. MPS."""
    return _ext.qgemv(wq, x, format)


def qgemm_w2a8_fused(wq, x):
    """K2: fused per-token int8 act-quant + W2A8 GEMM (no int8 round-trip through
    device memory). x (M, K) float; wq packed (N, K/32, 10). -> (M, N) half, same
    half-scale grid as quantize_per_token_int8 + qgemm_w2a8. K <= 8192. MPS."""
    return _ext.qgemm_w2a8_fused(wq, x)


# --- Ternary health monitors (train_plan §10.2 / moe_train_plan §6.2) ---

def ternary_stats(wq: torch.Tensor):
    """Packed wq (..., nblocks, 10) -> int32 (rows, 3) counts of {-1, 0, +1} codes per
    row (rows = leading dims flattened). zero-code fraction = counts[:,1]/K. MPS."""
    return _ext.ternary_stats(wq)


def code_flip_count(wq_a: torch.Tensor, wq_b: torch.Tensor):
    """Per-row count of ternary codes that differ between two identically-shaped packs
    (the code-flip-rate numerator; keep a snapshot pack per eval interval). MPS."""
    return _ext.code_flip_count(wq_a, wq_b)


# --- FP8 fake-quant (moe_train_plan §7.5 mode b / §8.7 Q-T4) ---

def fake_quant_fp8(x: torch.Tensor):
    """Per-tensor e4m3 fake-quant: (x_fq same dtype/shape, scale f32 (1,)), with
    scale = absmax/448. The eval-side FP8 cast for the mode-b parity delta. MPS."""
    return tuple(_ext.fake_quant_fp8(x))


# --- Dense-teacher KD-KL (A6b: the full-KL ablation arm) ---

def kd_kl_dense_fwd(t_logits, s_logits, invtemp=1.0):
    """Fused KL(softmax(t/tau) || softmax(s/tau)) per row over live teacher logits —
    no (T, V) prob/log-prob materialization. Returns (loss (Tn,), lse_t, lse_s).
    Pass invtemp = 1/tau; scale loss/grad_out by alpha*tau^2 yourself. MPS."""
    return tuple(_ext.kd_kl_dense_fwd(t_logits, s_logits, float(invtemp)))


def kd_kl_dense_bwd(t_logits, s_logits, lse_t, lse_s, grad_out, invtemp=1.0):
    """Backward of kd_kl_dense_fwd -> grad wrt student logits. MPS."""
    return _ext.kd_kl_dense_bwd(t_logits, s_logits, lse_t, lse_s, grad_out, float(invtemp))


# --- Attention decode (rollout-generate customer; training keeps SDPA) ---

def attn_decode(q, kc, vc):
    """Batch-1 GQA online-softmax decode: q (Hq, D) against dense caches (Tk, Hkv, D),
    head h reads kv head h // (Hq//Hkv); D <= 128. -> (Hq, D). MPS."""
    return _ext.attn_decode(q, kc, vc)


# --- MoE backward (dense STE backward over the padded schedule) ---

def moe_grouped_gemm_bwd_dx(dy, W, eot):
    """dX(rows, K) = dY(rows, N) @ W[e]^T per row-tile's expert; W (E, K, N) in its
    forward layout (no transpose copy). MPS."""
    return _ext.moe_grouped_gemm_bwd_dx(dy, W, eot)


def moe_grouped_gemm_bwd_dw(A, dy, off_pad, num_experts):
    """dW (E, K, N) = A^T @ dY per padded expert segment (off_pad from
    moe_pad_schedule). A/dY pad rows must be zero — moe_gather and moe_finalize_bwd
    guarantee that. MPS."""
    return _ext.moe_grouped_gemm_bwd_dw(A, dy, off_pad, int(num_experts))


def moe_finalize_bwd(grad_out, expert_out, inv_pad, topk_weights):
    """Backward of moe_finalize: (grad_expert_out (P, H) with zero pad rows,
    grad_weights f32 (T, k) — the router-weight grads). MPS."""
    return tuple(_ext.moe_finalize_bwd(grad_out, expert_out, inv_pad, topk_weights))


def moe_gather_bwd(dA, inv_pad, k):
    """Backward of moe_gather: dx (T, H) = sum over the k routed copies of each token.
    MPS."""
    return _ext.moe_gather_bwd(dA, inv_pad, int(k))
