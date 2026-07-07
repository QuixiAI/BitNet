"""ctypes wrapper for the K-track CPU engine kernels (src/bitnet_cpu.c).

Auto-rebuilds the shared library when the C source is newer (same convention as the
Metal tree's metallib staleness check). All arrays are numpy, C-contiguous; the
packed-ternary format is the shared 10-byte/32-weight `bitnet` block layout, so
tensors packed by `tk_torch.weight_quant_ternary`(`_pt`) feed these kernels directly.
"""

from __future__ import annotations

import ctypes as C
import subprocess
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = _HERE / ("libbitnet_cpu.dylib" if __import__("sys").platform == "darwin"
                else "libbitnet_cpu.so")
_SRC = _HERE / "src" / "bitnet_cpu.c"

if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
    subprocess.run(["sh", str(_HERE / "build.sh")], check=True)

_lib = C.CDLL(str(_LIB))
_lib.bn_init()

_u8 = np.ctypeslib.ndpointer(np.uint8, flags="C")
_i8 = np.ctypeslib.ndpointer(np.int8, flags="C")
_i32 = np.ctypeslib.ndpointer(np.int32, flags="C")
_f32 = np.ctypeslib.ndpointer(np.float32, flags="C")
_i64 = C.c_int64

_lib.bn_act_quant_int8.restype = C.c_float
_lib.bn_act_quant_int8.argtypes = [_f32, _i64, _i8]
for _n in ("bn_gemv_w2a8", "bn_gemv_w2a8_scalar", "bn_gemv_w2a8_neon"):
    if hasattr(_lib, _n):
        getattr(_lib, _n).argtypes = [_u8, _i64, _i64, _i8, C.c_float, C.c_int, _f32]
_lib.bn_expert_ffn_w2a8.argtypes = [_f32, _i64, _i64, _u8, _u8, _u8, C.c_int, C.c_float,
                                    _i8, _i8, _f32, _f32]
_lib.bn_moe_ffn_w2a8.argtypes = [_f32, _i64, _i64, _i64, _i32, _f32, _u8, _u8, _u8,
                                 _i64, _i64, _i64, C.c_int, _i8, _i8, _f32, _f32]
_lib.bn_route_topk.argtypes = [_f32, _i64, _i64, _i32, _f32]
for _n in ("bn_gemv_fp8", "bn_gemv_fp8_scalar", "bn_gemv_fp8_neon"):
    if hasattr(_lib, _n):
        getattr(_lib, _n).argtypes = [_u8, _f32, _i64, _i64, _f32, _f32]
_lib.bn_attn_decode_kv8.argtypes = [_f32, _i8, _f32, _i8, _f32, _i64, _i64, _i64, _i64, _f32]
_lib.bn_unpack_ternary_f32.argtypes = [_u8, _i64, _i64, _f32]
_u16 = np.ctypeslib.ndpointer(np.uint16, flags="C")
_lib.bn_rms_norm.argtypes = [_f32, _f32, _i64, _i64, C.c_float, _f32]
_lib.bn_rope_neox.argtypes = [_f32, _i64, _i64, _i64, C.c_float]
_lib.bn_kv_quant_append.argtypes = [_f32, _f32, _i64, _i64, _i64, _i8, _f32, _i8, _f32]
for _n in ("bn_gemv_q8", "bn_gemv_q8_scalar", "bn_gemv_q8_neon"):
    if hasattr(_lib, _n):
        getattr(_lib, _n).argtypes = [_u8, _i64, _i64, _i8, C.c_float, _f32]
for _n in ("bn_gemv_bf16", "bn_gemv_bf16_scalar", "bn_gemv_bf16_neon"):
    if hasattr(_lib, _n):
        getattr(_lib, _n).argtypes = [_u16, _i64, _i64, _f32, _f32]
_lib.bn_pack_b3.argtypes = [_u8, _i64, _i64, _u8]
_lib.bn_gemv_b3_scalar.argtypes = [_u8, _i64, _i64, _i8, C.c_float, C.c_int, _f32]
_lib.bn_pack_tl1.argtypes = [_u8, _i64, _i64, _u8]
_lib.bn_gemv_tl1_scalar.argtypes = [_u8, _i64, _i64, _i8, C.c_float, C.c_int, _f32]
for _n in ("bn_gemv_tl1", "bn_gemv_tl1_neon"):
    if hasattr(_lib, _n):
        getattr(_lib, _n).argtypes = [_u8, _i64, _i64, _i8, C.c_float, C.c_int, _f32, _i8]


def act_quant_int8(x: np.ndarray):
    xq = np.empty(x.shape[-1], np.int8)
    s = _lib.bn_act_quant_int8(np.ascontiguousarray(x, np.float32).reshape(-1), x.shape[-1], xq)
    return xq, float(s)


def gemv_w2a8(wq: np.ndarray, xq: np.ndarray, a_scale: float, pt: bool = False,
              impl: str = "auto"):
    """wq (N, K/32, 10) packed; xq int8 (K,). impl: auto | scalar | neon."""
    N, nb, _ = wq.shape
    out = np.empty(N, np.float32)
    fn = {"auto": _lib.bn_gemv_w2a8, "scalar": _lib.bn_gemv_w2a8_scalar,
          "neon": getattr(_lib, "bn_gemv_w2a8_neon", None)}[impl]
    fn(np.ascontiguousarray(wq).reshape(-1), N, nb * 32, xq, a_scale, int(pt), out)
    return out


def expert_ffn_w2a8(x, gate_wq, up_wq, down_wq, w_r=1.0, pt=False, out=None):
    """One expert's fused decode FFN, accumulated into out (H,) with router weight w_r."""
    H = x.shape[-1]
    I = gate_wq.shape[0]
    out = np.zeros(H, np.float32) if out is None else out
    xq = np.empty(H, np.int8)
    hq = np.empty(I, np.int8)
    h = np.empty(I, np.float32)
    _lib.bn_expert_ffn_w2a8(np.ascontiguousarray(x, np.float32), H, I,
                            np.ascontiguousarray(gate_wq).reshape(-1),
                            np.ascontiguousarray(up_wq).reshape(-1),
                            np.ascontiguousarray(down_wq).reshape(-1),
                            int(pt), float(w_r), xq, hq, h, out)
    return out


def moe_ffn_w2a8(x, gate_stack, up_stack, down_stack, expert_ids, expert_w, pt=False):
    """Fused MoE decode step. *_stack: (E, rows, K/32, 10) packed expert stacks."""
    H = x.shape[-1]
    E, I = gate_stack.shape[0], gate_stack.shape[1]
    out = np.empty(H, np.float32)
    xq = np.empty(H, np.int8)
    hq = np.empty(I, np.int8)
    h = np.empty(I, np.float32)
    gs = gate_stack.reshape(E, -1)
    us = up_stack.reshape(E, -1)
    ds = down_stack.reshape(E, -1)
    _lib.bn_moe_ffn_w2a8(np.ascontiguousarray(x, np.float32), H, I, len(expert_ids),
                         np.ascontiguousarray(expert_ids, np.int32),
                         np.ascontiguousarray(expert_w, np.float32),
                         np.ascontiguousarray(gs).reshape(-1),
                         np.ascontiguousarray(us).reshape(-1),
                         np.ascontiguousarray(ds).reshape(-1),
                         gs.shape[1], us.shape[1], ds.shape[1], int(pt), xq, hq, h, out)
    return out


def route_topk(logits: np.ndarray, k: int):
    E = logits.shape[-1]
    ids = np.empty(k, np.int32)
    w = np.empty(k, np.float32)
    _lib.bn_route_topk(np.ascontiguousarray(logits, np.float32), E, k, ids, w)
    return ids, w


def gemv_fp8(w_codes: np.ndarray, row_scale: np.ndarray, x: np.ndarray, impl="auto"):
    """w_codes uint8 (N,K) e4m3; row_scale f32 (N,); x f32 (K,)."""
    N, K = w_codes.shape
    out = np.empty(N, np.float32)
    fn = {"auto": _lib.bn_gemv_fp8, "scalar": _lib.bn_gemv_fp8_scalar,
          "neon": getattr(_lib, "bn_gemv_fp8_neon", None)}[impl]
    fn(np.ascontiguousarray(w_codes).reshape(-1), np.ascontiguousarray(row_scale, np.float32),
       N, K, np.ascontiguousarray(x, np.float32), out)
    return out


def attn_decode_kv8(q, kc, k_scale, vc, v_scale):
    """q (Hq, D) f32; kc/vc (T, Hkv, D) int8; scales (T, Hkv) f32. GQA online softmax."""
    Hq, D = q.shape
    T, Hkv, _ = kc.shape
    out = np.empty((Hq, D), np.float32)
    _lib.bn_attn_decode_kv8(np.ascontiguousarray(q, np.float32),
                            np.ascontiguousarray(kc).reshape(-1),
                            np.ascontiguousarray(k_scale, np.float32).reshape(-1),
                            np.ascontiguousarray(vc).reshape(-1),
                            np.ascontiguousarray(v_scale, np.float32).reshape(-1),
                            T, Hq, Hkv, D, out.reshape(-1))
    return out


def unpack_ternary_f32(wq: np.ndarray):
    """(N, K/32, 10) packed -> (N, K) f32 dense (prefill dequant-once; feed BLAS)."""
    N, nb, _ = wq.shape
    out = np.empty((N, nb * 32), np.float32)
    _lib.bn_unpack_ternary_f32(np.ascontiguousarray(wq).reshape(-1), N, nb * 32,
                               out.reshape(-1))
    return out


def prefill_ternary(wq: np.ndarray, X: np.ndarray):
    """Prefill mode (§7.4): dequant the expert block ONCE, then a real BLAS GEMM
    (numpy dot -> Accelerate on macOS) where the unpack cost amortizes over the chunk.
    X (M, K) f32 -> (M, N)."""
    W = unpack_ternary_f32(wq)                       # (N, K)
    return X.astype(np.float32) @ W.T


# ---- decode glue (RMSNorm / RoPE / KV writer) ----

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6):
    """x (..., D) f32; w (D,) f32. QK-norm = same call with x (heads, head_dim)."""
    D = x.shape[-1]
    xf = np.ascontiguousarray(x, np.float32).reshape(-1, D)
    out = np.empty_like(xf)
    _lib.bn_rms_norm(xf.reshape(-1), np.ascontiguousarray(w, np.float32),
                     xf.shape[0], D, float(eps), out.reshape(-1))
    return out.reshape(x.shape)


def rope_neox(x: np.ndarray, pos: int, theta: float = 1e6):
    """In-place NeoX/HF half-split RoPE on x (H, D) f32 at position pos; returns x."""
    H, D = x.shape
    x = np.ascontiguousarray(x, np.float32)
    _lib.bn_rope_neox(x.reshape(-1), H, D, int(pos), float(theta))
    return x


def kv_quant_append(k_new, v_new, pos, kc, k_scale, vc, v_scale):
    """Quantize k_new/v_new (Hkv, D) f32 per (token, head) absmax int8 and write them
    at position pos of the (T, Hkv, D) caches (the layout attn_decode_kv8 reads)."""
    Hkv, D = k_new.shape
    _lib.bn_kv_quant_append(np.ascontiguousarray(k_new, np.float32),
                            np.ascontiguousarray(v_new, np.float32),
                            int(pos), Hkv, D,
                            kc.reshape(-1), k_scale.reshape(-1),
                            vc.reshape(-1), v_scale.reshape(-1))


# ---- head GEMVs (Q-A-head8 contenders) ----

def pack_q8(W: np.ndarray):
    """Q8_0-shaped pack: blocks of 32 { fp16 d; int8 qs[32] } = 34 B. Returns
    (packed uint8 (N, K/32, 34), dequant f32 (N, K))."""
    W = np.ascontiguousarray(W, np.float32)
    N, K = W.shape
    nb = K // 32
    Wb = W.reshape(N, nb, 32)
    d = (np.abs(Wb).max(axis=2) / 127.0).astype(np.float32)
    d16 = d.astype(np.float16)
    inv = np.where(d > 0, 1.0 / np.maximum(d, 1e-30), 0.0)
    q = np.clip(np.rint(Wb * inv[..., None]), -127, 127).astype(np.int8)
    out = np.zeros((N, nb, 34), np.uint8)
    out[:, :, 0:2] = d16.view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:34] = q.view(np.uint8)
    deq = (d16.astype(np.float32)[..., None] * q.astype(np.float32)).reshape(N, K)
    return out, deq


def gemv_q8(wq: np.ndarray, xq: np.ndarray, a_scale: float, impl: str = "auto"):
    """wq (N, K/32, 34) Q8_0-shaped; xq int8 (K,)."""
    N, nb, _ = wq.shape
    out = np.empty(N, np.float32)
    fn = {"auto": _lib.bn_gemv_q8, "scalar": _lib.bn_gemv_q8_scalar,
          "neon": getattr(_lib, "bn_gemv_q8_neon", None)}[impl]
    fn(np.ascontiguousarray(wq).reshape(-1), N, nb * 32, xq, float(a_scale), out)
    return out


def gemv_bf16(w: np.ndarray, x: np.ndarray, impl: str = "auto"):
    """w uint16 bf16 codes (N, K) (e.g. torch bf16 tensor .view(uint16)); x f32 (K,)."""
    N, K = w.shape
    out = np.empty(N, np.float32)
    fn = {"auto": _lib.bn_gemv_bf16, "scalar": _lib.bn_gemv_bf16_scalar,
          "neon": getattr(_lib, "bn_gemv_bf16_neon", None)}[impl]
    fn(np.ascontiguousarray(w, np.uint16).reshape(-1), N, K,
       np.ascontiguousarray(x, np.float32), out)
    return out


# ---- packing bake-off formats (repacked from the tested format-A blocks) ----

def pack_b3(wq: np.ndarray):
    """Format B (base-3 dense, 9 B/32): repack format-A blocks (N, K/32, 10)."""
    N, nb, _ = wq.shape
    out = np.empty((N, nb, 9), np.uint8)
    _lib.bn_pack_b3(np.ascontiguousarray(wq).reshape(-1), N, nb * 32, out.reshape(-1))
    return out


def gemv_b3(wb: np.ndarray, xq: np.ndarray, a_scale: float, pt: bool = False):
    """Format-B GEMV (scalar; the min-bytes contender)."""
    N, nb, _ = wb.shape
    out = np.empty(N, np.float32)
    _lib.bn_gemv_b3_scalar(np.ascontiguousarray(wb).reshape(-1), N, nb * 32, xq,
                           float(a_scale), int(pt), out)
    return out


def pack_tl1(wq: np.ndarray):
    """Format C (TL1-style LUT pair-index tiles, 160 B per 16 rows x 32 k): repack
    format-A blocks. N must be a multiple of 16."""
    N, nb, _ = wq.shape
    assert N % 16 == 0, "tl1 tiles are 16 rows"
    out = np.empty((N // 16, nb, 160), np.uint8)
    _lib.bn_pack_tl1(np.ascontiguousarray(wq).reshape(-1), N, nb * 32, out.reshape(-1))
    return out


def gemv_tl1(wt: np.ndarray, xq: np.ndarray, a_scale: float, pt: bool = False,
             impl: str = "auto", lut_scratch: np.ndarray | None = None):
    """Format-C GEMV. wt (N/16, K/32, 160); xq int8 (K,). impl: auto|scalar|neon."""
    nt, nb, _ = wt.shape
    N, K = nt * 16, nb * 32
    out = np.empty(N, np.float32)
    flat = np.ascontiguousarray(wt).reshape(-1)
    if impl == "scalar":
        _lib.bn_gemv_tl1_scalar(flat, N, K, xq, float(a_scale), int(pt), out)
        return out
    if lut_scratch is None:
        lut_scratch = np.empty(K // 2 * 32, np.int8)
    fn = {"auto": _lib.bn_gemv_tl1, "neon": getattr(_lib, "bn_gemv_tl1_neon", None)}[impl]
    fn(flat, N, K, xq, float(a_scale), int(pt), out, lut_scratch)
    return out
