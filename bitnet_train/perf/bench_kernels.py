#!/usr/bin/env python3
"""BitNet kernel benchmark harness (schema v1).

Modeled on QuixiCore-Metal's perf/bench_kernels.py. Covers the two kernel stacks
this repo owns, each case pairing:
  - the target kernel (tk_torch on MPS, or the bn_* CPU engine),
  - a baseline (a framework op, or the composed/naive equivalent),
  - a one-shot correctness check (max abs/rel error vs a float64 numpy reference),
  - derived throughput (GB/s, packed-weight GB/s for quant decode, GFLOP/s).

Backends:
  torch   — tk_torch Metal kernels on MPS (the training kernels).
  cpu     — the bn_* K-track engine (numpy/ctypes), the deployment kernels.

Run from the repo root:
    .venv/bin/python bitnet_train/perf/bench_kernels.py --backend torch --preset smoke --kernel all
    .venv/bin/python bitnet_train/perf/bench_kernels.py --backend cpu --preset quick --kernel gemv_w2a8,expert_ffn
    .venv/bin/python bitnet_train/perf/bench_kernels.py --backend torch --kernel qgemv --formats bitnet,tq2_0

Each run writes (results/ is git-ignored — copy summaries into optimization_status.md):
    bitnet_train/perf/results/YYYY-MM-DD/<run-id>/run.json
    bitnet_train/perf/results/YYYY-MM-DD/<run-id>/results.jsonl   (schema v1)
    bitnet_train/perf/results/YYYY-MM-DD/<run-id>/summary.md

Cases self-skip (recorded with a reason, not fatal) when a kernel, format, or
backend is unavailable.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "bitnet_train" / "metal"),
           str(REPO_ROOT / "bitnet_train" / "cpu")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- backends
class TorchBackend:
    """tk_torch Metal kernels on MPS."""
    name = "torch"

    def __init__(self):
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("torch MPS not available")
        import tk_torch as tk
        self.torch = torch
        self.tk = tk
        self._dt = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}

    def arr(self, a, dtype="f32"):
        return self.torch.from_numpy(np.ascontiguousarray(a)).to(self._dt[dtype]).to("mps")

    def sync(self, _=None):
        self.torch.mps.synchronize()

    def to_numpy(self, t):
        return t.detach().float().cpu().numpy()


class CPUBackend:
    """The bn_* K-track engine (numpy/ctypes)."""
    name = "cpu"

    def __init__(self):
        import bitnet_cpu as bn
        self.bn = bn

    def arr(self, a, dtype="f32"):
        return np.ascontiguousarray(a, {"f32": np.float32}.get(dtype, np.float32))

    def sync(self, _=None):
        pass

    def to_numpy(self, t):
        return np.asarray(t)


def make_backend(name):
    return TorchBackend() if name == "torch" else CPUBackend()


# --------------------------------------------------------------------------- timing
def time_thunk(fn, be, warmup, iters, min_sample_ms=2.0):
    """Median/p20/p80 per-call latency (ms). Small kernels are batched per sync so
    the submit+sync floor (~0.2 ms on MPS) does not swamp the kernel time; the
    reported number is throughput-style per-call latency."""
    t0 = time.perf_counter()
    calls = 0
    while calls < warmup or time.perf_counter() - t0 < 0.05:
        be.sync(fn())
        calls += 1
    t0 = time.perf_counter()
    be.sync(fn())
    est_ms = 1e3 * (time.perf_counter() - t0)
    batch = max(1, min(64, math.ceil(min_sample_ms / max(est_ms, 1e-3))))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(batch)]
        be.sync(outs)
        samples.append(1e3 * (time.perf_counter() - t0) / batch)
    samples.sort()
    n = len(samples)
    med = statistics.median(samples)
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples)
    return {"ms": med,
            "p20_ms": samples[max(0, int(0.20 * n) - 1)] if n > 1 else med,
            "p80_ms": samples[min(n - 1, int(0.80 * n))] if n > 1 else med,
            "cv": (stdev / mean) if mean > 0 else 0.0, "batch": batch}


# --------------------------------------------------------------------------- case model
@dataclass
class Case:
    kernel: str
    variant: str
    shape: dict
    dtype: str = "f32"
    fmt: str | None = None
    target: object = None
    baselines: dict = field(default_factory=dict)
    ref: object = None
    out_to_numpy: object = None
    bytes_moved: float | None = None
    weight_bytes: float | None = None
    flops: float | None = None
    notes: str = ""
    skip_reason: str | None = None


def _rel_err(out, ref):
    return float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-9))


def run_case(case, be, warmup, iters, check):
    row = {"schema": SCHEMA_VERSION, "kernel": case.kernel, "variant": case.variant,
           "shape": case.shape, "dtype": case.dtype, "format": case.fmt,
           "status": "ok", "notes": case.notes}
    if case.skip_reason:
        row["status"] = "skip"
        row["skip_reason"] = case.skip_reason
        return row
    try:
        if check and case.ref is not None:
            out = case.target()
            be.sync(out)
            out_np = (case.out_to_numpy(out) if case.out_to_numpy else be.to_numpy(out))
            ref_np = np.asarray(case.ref() if callable(case.ref) else case.ref, np.float64)
            out_np = np.asarray(out_np, np.float64)
            if out_np.shape != ref_np.shape:
                raise RuntimeError(f"shape out {out_np.shape} vs ref {ref_np.shape}")
            row["max_abs_err"] = float(np.max(np.abs(out_np - ref_np)))
            row["max_rel_err"] = _rel_err(out_np, ref_np)
        t = time_thunk(case.target, be, warmup, iters)
        row.update(target_ms=t["ms"], target_p20_ms=t["p20_ms"],
                   target_p80_ms=t["p80_ms"], target_cv=round(t["cv"], 4),
                   batch=t["batch"], baselines={})
        for name, thunk in case.baselines.items():
            try:
                b = time_thunk(thunk, be, warmup, iters)
                row["baselines"][name] = {"ms": b["ms"],
                                          "speedup": (b["ms"] / t["ms"]) if t["ms"] else None}
            except Exception as e:  # noqa: BLE001
                row["baselines"][name] = {"error": f"{type(e).__name__}: {e}"}
        sec = t["ms"] / 1e3
        if case.bytes_moved:
            row["gbps"] = case.bytes_moved / sec / 1e9
        if case.weight_bytes:
            row["weight_gbps"] = case.weight_bytes / sec / 1e9
        if case.flops:
            row["gflops"] = case.flops / sec / 1e9
    except Exception as e:  # noqa: BLE001
        row["status"] = "error"
        row["skip_reason"] = f"{type(e).__name__}: {e}"
    return row


# --------------------------------------------------------------------------- numpy oracles
def pack_bitnet(W, pt=True):
    W = np.ascontiguousarray(W, np.float32)
    N, K = W.shape
    nb = K // 32
    Wb = W.reshape(N, nb, 32)
    s = (np.full((N, nb), max(np.abs(W).mean(), 1e-5), np.float32) if pt
         else np.maximum(np.abs(Wb).mean(2), 1e-5).astype(np.float32))
    q = np.clip(np.rint(Wb / s[..., None]), -1, 1).astype(np.int32)
    code = (q + 1).astype(np.uint32).reshape(N, nb, 8, 4)
    out = np.zeros((N, nb, 10), np.uint8)
    out[:, :, 0:2] = s.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:10] = (code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4)
                       | (code[..., 3] << 6)).astype(np.uint8)
    deq = (s.astype(np.float16).astype(np.float32)[..., None] * q).reshape(N, K)
    return out, deq


def act_q(x):
    s = np.abs(x).max() / 127.0
    q = np.clip(np.rint(x / max(s, 1e-30)), -127, 127) if s > 0 else np.zeros_like(x)
    return q.astype(np.int8), np.float32(s)


# --------------------------------------------------------------------------- registry
KERNEL_BUILDERS = {}


def register(name, backends):
    def deco(fn):
        KERNEL_BUILDERS[name] = (fn, set(backends))
        return fn
    return deco


SHAPES = {  # (N, K) weight-matrix shapes: LLM projections + Q-track expert
    "smoke": [(768, 2048)],
    "quick": [(768, 2048), (2048, 2048)],
    "comprehensive": [(768, 2048), (2048, 768), (2048, 2048), (4096, 4096),
                      (14336, 4096)],
    # Llama-3.2-1B heal shapes (a1 profile): q/o 2048x2048, k/v 512x2048,
    # gate/up 8192x2048, down 2048x8192
    "a1": [(2048, 2048), (512, 2048), (8192, 2048), (2048, 8192)],
}


# ===== Metal (torch) kernels =====

@register("weight_quant", ["torch"])
def _wq(be, preset, formats):
    for N, K in SHAPES[preset]:
        W = (np.random.randn(N, K) * 0.05).astype(np.float32)
        Wd = be.arr(W, "f32")
        yield Case("weight_quant", f"N{N}_K{K}", {"N": N, "K": K}, "f32",
                   target=lambda Wd=Wd: be.tk.weight_quant_ternary_pt(Wd)[0],
                   # read f32 twice (abssum pass + encode pass) + write w_deq f32 + wq
                   bytes_moved=N * K * (4 + 4 + 4) + N * K / 32 * 10,
                   notes="per-tensor ternary pack (K1/K5)")


@register("quantize_tq2_0", ["torch"])
def _qtq(be, preset, formats):
    for N, K in SHAPES[preset]:
        if K % 256:
            continue
        W = (np.random.randn(N, K) * 0.05).astype(np.float32)
        Wd = be.arr(W, "f32")
        yield Case("quantize_tq2_0", f"N{N}_K{K}", {"N": N, "K": K}, "f32",
                   target=lambda Wd=Wd: be.tk.quantize_tq2_0(Wd)[0],
                   bytes_moved=N * K * 4 + N * K / 256 * 66,
                   notes="llama.cpp-native TQ2_0 pack")


@register("qgemv", ["torch"])
def _qgemv(be, preset, formats):
    fmts = formats or ["bitnet", "tq2_0"]
    for N, K in SHAPES[preset]:
        for fmt in fmts:
            if fmt == "tq2_0" and K % 256:
                continue
            W = (np.random.randn(N, K) * 0.05).astype(np.float32)
            if fmt == "bitnet":
                wq, deq = pack_bitnet(W)
                wqd = be.arr(wq.astype(np.float32)).to(be.torch.uint8) if False else \
                    be.torch.from_numpy(wq).to("mps")
                wbytes = wq.nbytes
            else:
                wqt, deqt = be.tk.quantize_tq2_0(be.arr(W))
                wqd = wqt
                deq = be.to_numpy(deqt)
                wbytes = int(np.prod(wqt.shape))
            x = (np.random.randn(K, 1) * 0.7).astype(np.float32)
            xd = be.arr(x, "f16")
            ref = (deq.astype(np.float64) @ x.astype(np.float64)).reshape(N, 1)
            yield Case("qgemv", f"N{N}_K{K}", {"N": N, "K": K}, "f16", fmt=fmt,
                       target=lambda wqd=wqd, xd=xd, fmt=fmt: be.tk.qgemv(wqd, xd, fmt),
                       ref=ref, weight_bytes=wbytes,
                       flops=2 * N * K, notes="ternary decode GEMV")


@register("qgemm", ["torch"])
def _qgemm(be, preset, formats):
    fmts = formats or ["bitnet", "tq2_0"]
    for N, K in SHAPES[preset]:
        M = 64
        for fmt in fmts:
            if fmt == "tq2_0" and K % 256:
                continue
            W = (np.random.randn(N, K) * 0.05).astype(np.float32)
            if fmt == "bitnet":
                wq, deq = pack_bitnet(W)
                wqd = be.torch.from_numpy(wq).to("mps")
                wbytes = wq.nbytes
            else:
                wqt, deqt = be.tk.quantize_tq2_0(be.arr(W))
                wqd, deq, wbytes = wqt, be.to_numpy(deqt), int(np.prod(wqt.shape))
            x = (np.random.randn(K, M) * 0.7).astype(np.float32)
            xd = be.arr(x, "f16")
            ref = deq.astype(np.float64) @ x.astype(np.float64)
            yield Case("qgemm", f"N{N}_K{K}_M{M}", {"N": N, "K": K, "M": M}, "f16",
                       fmt=fmt,
                       target=lambda wqd=wqd, xd=xd, fmt=fmt: be.tk.qgemm(wqd, xd, fmt),
                       ref=ref, weight_bytes=wbytes, flops=2 * N * K * M,
                       notes="ternary prefill GEMM")


@register("fake_quant_int8", ["torch"])
def _fq(be, preset, formats):
    for N, K in SHAPES[preset]:
        x = (np.random.randn(N, K)).astype(np.float32)
        xd = be.arr(x, "bf16")
        yield Case("fake_quant_int8", f"M{N}_D{K}", {"M": N, "D": K}, "bf16",
                   target=lambda xd=xd: be.tk.fake_quant_int8(xd)[0],
                   bytes_moved=N * K * 2 + N * K * 2, notes="K4 one-pass fake-quant")


@register("kd_kl_dense", ["torch"])
def _kd(be, preset, formats):
    V = 128256
    for Tn in ({"smoke": [64], "quick": [64, 256], "a1": [1024],
                "comprehensive": [64, 256, 512]}[preset]):
        t = be.arr((np.random.randn(Tn, V) * 2).astype(np.float32), "bf16")
        s = be.arr((np.random.randn(Tn, V) * 2).astype(np.float32), "bf16")
        # fwd streams both rows twice (online lse pass + loss pass)
        yield Case("kd_kl_dense", f"T{Tn}_V{V}_fwd", {"T": Tn, "V": V}, "bf16",
                   target=lambda t=t, s=s: be.tk.kd_kl_dense_fwd(t, s, 0.5)[0],
                   bytes_moved=Tn * V * 2 * 2 * 2, notes="A6b full-KL fwd")
        loss, lse_t, lse_s = be.tk.kd_kl_dense_fwd(t, s, 0.5)
        go = be.arr(np.random.rand(Tn).astype(np.float32), "f32")
        yield Case("kd_kl_dense", f"T{Tn}_V{V}_bwd", {"T": Tn, "V": V}, "bf16",
                   target=lambda t=t, s=s, lse_t=lse_t, lse_s=lse_s, go=go:
                       be.tk.kd_kl_dense_bwd(t, s, lse_t, lse_s, go, 0.5),
                   bytes_moved=Tn * V * 2 * 3,       # read t+s, write grad
                   notes="A6b full-KL bwd")


@register("kd_ce_fused", ["torch"])
def _cekd(be, preset, formats):
    V = 128256
    for Tn in ({"smoke": [64], "quick": [256], "a1": [1024],
                "comprehensive": [256, 1024]}[preset]):
        t = be.arr((np.random.randn(Tn, V) * 2).astype(np.float32), "bf16")
        s = be.arr((np.random.randn(Tn, V) * 2).astype(np.float32), "bf16")
        tg = be.torch.randint(0, V, (Tn,), device="mps", dtype=be.torch.int32)
        yield Case("kd_ce_fused", f"T{Tn}_V{V}_fwd", {"T": Tn, "V": V}, "bf16",
                   target=lambda t=t, s=s, tg=tg:
                       be.tk.kd_ce_fused_fwd(t, s, tg, 0.5)[0],
                   baselines={"separate": lambda t=t, s=s, tg=tg: (
                       be.tk.cross_entropy_fwd(s, tg),
                       be.tk.kd_kl_dense_fwd(t, s, 0.5))[1][0]},
                   bytes_moved=Tn * V * 2 * 2, notes="heal-loss fused fwd (1-pass)")
        ce, kd, lse_sr, lse_st, lse_t = be.tk.kd_ce_fused_fwd(t, s, tg, 0.5)
        go = be.arr(np.random.rand(Tn).astype(np.float32), "f32")

        def _sep_bwd(t=t, s=s, tg=tg, lse=lse_sr, lse_t=lse_t, lse_s=lse_st, go=go):
            g1 = be.tk.cross_entropy_bwd(s, tg, lse, go)
            g2 = be.tk.kd_kl_dense_bwd(t, s, lse_t, lse_s, go, 0.5)
            return g1 + g2                      # the autograd grad-add pass
        yield Case("kd_ce_fused", f"T{Tn}_V{V}_bwd", {"T": Tn, "V": V}, "bf16",
                   target=lambda t=t, s=s, tg=tg, a=lse_sr, b=lse_st, c=lse_t, go=go:
                       be.tk.kd_ce_fused_bwd(t, s, tg, a, b, c, go, go, 0.5),
                   baselines={"separate": _sep_bwd},
                   bytes_moved=Tn * V * 2 * 3, notes="heal-loss fused bwd")


@register("cross_entropy", ["torch"])
def _ce(be, preset, formats):
    V = 128256
    for Tn in ({"smoke": [64], "quick": [256], "a1": [1024],
                "comprehensive": [256, 1024]}[preset]):
        lg = be.arr((np.random.randn(Tn, V) * 2).astype(np.float32), "bf16")
        tg = be.torch.randint(0, V, (Tn,), device="mps", dtype=be.torch.int32)
        yield Case("cross_entropy", f"T{Tn}_V{V}_fwd", {"T": Tn, "V": V}, "bf16",
                   target=lambda lg=lg, tg=tg: be.tk.cross_entropy_fwd(lg, tg)[0],
                   baselines={"torch_ce": lambda lg=lg, tg=tg:
                              be.torch.nn.functional.cross_entropy(
                                  lg.float(), tg.long(), reduction="none")},
                   bytes_moved=Tn * V * 2, notes="fused CE fwd")
        loss, lse = be.tk.cross_entropy_fwd(lg, tg)
        go = be.arr(np.random.rand(Tn).astype(np.float32), "f32")
        yield Case("cross_entropy", f"T{Tn}_V{V}_bwd", {"T": Tn, "V": V}, "bf16",
                   target=lambda lg=lg, tg=tg, lse=lse, go=go:
                       be.tk.cross_entropy_bwd(lg, tg, lse, go),
                   bytes_moved=Tn * V * 2 * 2, notes="fused CE bwd")


@register("ternary_stats", ["torch"])
def _ts(be, preset, formats):
    for N, K in SHAPES[preset]:
        W = (np.random.randn(N, K) * 0.05).astype(np.float32)
        wq, _ = pack_bitnet(W)
        wqd = be.torch.from_numpy(wq).to("mps")
        yield Case("ternary_stats", f"N{N}_K{K}", {"N": N, "K": K}, "u8",
                   target=lambda wqd=wqd: be.tk.ternary_stats(wqd),
                   bytes_moved=wq.nbytes, notes="§10.2 health monitor")


@register("attn_decode", ["torch"])
def _ad(be, preset, formats):
    for T in ({"smoke": [1024], "quick": [1024, 4096],
               "comprehensive": [1024, 4096, 8192]}[preset]):
        Hq, Hkv, D = 32, 8, 64
        q = be.arr(np.random.randn(Hq, D).astype(np.float32), "bf16")
        kc = be.arr(np.random.randn(T, Hkv, D).astype(np.float32), "bf16")
        vc = be.arr(np.random.randn(T, Hkv, D).astype(np.float32), "bf16")
        yield Case("attn_decode", f"T{T}", {"T": T, "Hq": Hq, "Hkv": Hkv, "D": D},
                   "bf16",
                   target=lambda q=q, kc=kc, vc=vc: be.tk.attn_decode(q, kc, vc),
                   bytes_moved=T * Hkv * D * 2 * 2, notes="ACADEMIC (SDPA wins)")


# ===== CPU (bn_*) kernels =====

@register("gemv_w2a8", ["cpu"])
def _gemv(be, preset, formats):
    for N, K in SHAPES[preset]:
        W = (np.random.randn(N, K) * 0.05).astype(np.float32)
        wq, deq = pack_bitnet(W, pt=True)
        x = np.random.randn(K).astype(np.float32)
        xq, s = act_q(x)
        ref = deq.astype(np.float64) @ (xq.astype(np.float64) * s)
        for impl in ("scalar", "neon"):
            yield Case("gemv_w2a8", f"N{N}_K{K}_{impl}", {"N": N, "K": K}, "i8",
                       fmt="bitnet",
                       target=lambda wq=wq, xq=xq, s=s, impl=impl:
                           be.bn.gemv_w2a8(wq, xq, float(s), pt=True, impl=impl),
                       ref=(ref if impl == "neon" else None), weight_bytes=wq.nbytes,
                       flops=2 * N * K, notes="A-format decode GEMV")


@register("gemv_tl1", ["cpu"])
def _tl1(be, preset, formats):
    for N, K in SHAPES[preset]:
        if N % 16:
            continue
        W = (np.random.randn(N, K) * 0.05).astype(np.float32)
        wq, deq = pack_bitnet(W, pt=True)
        wt = be.bn.pack_tl1(wq)
        x = np.random.randn(K).astype(np.float32)
        xq, s = act_q(x)
        ref = deq.astype(np.float64) @ (xq.astype(np.float64) * s)
        yield Case("gemv_tl1", f"N{N}_K{K}", {"N": N, "K": K}, "i8", fmt="tl1",
                   target=lambda wt=wt, xq=xq, s=s:
                       be.bn.gemv_tl1(wt, xq, float(s), pt=True, impl="auto"),
                   ref=ref, weight_bytes=wt.nbytes, flops=2 * N * K,
                   notes="TL1 LUT decode GEMV (bakeoff winner)")


@register("expert_ffn", ["cpu"])
def _ffn(be, preset, formats):
    for H, I in [(2048, 768)] if preset == "smoke" else [(2048, 768), (4096, 1024)]:
        Wg = (np.random.randn(I, H) * 0.1).astype(np.float32)
        Wu = (np.random.randn(I, H) * 0.1).astype(np.float32)
        Wd = (np.random.randn(H, I) * 0.1).astype(np.float32)
        gq, _ = pack_bitnet(Wg); uq, _ = pack_bitnet(Wu); dq, _ = pack_bitnet(Wd)
        x = (np.random.randn(H) * 0.5).astype(np.float32)
        wbytes = gq.nbytes + uq.nbytes + dq.nbytes
        yield Case("expert_ffn", f"H{H}_I{I}_A", {"H": H, "I": I}, "i8", fmt="bitnet",
                   target=lambda gq=gq, uq=uq, dq=dq, x=x:
                       be.bn.expert_ffn_w2a8(x, gq, uq, dq, pt=True),
                   weight_bytes=wbytes, notes="fused expert FFN, A-format")
        if I % 16 == 0 and H % 16 == 0:
            gt, ut, dt = be.bn.pack_tl1(gq), be.bn.pack_tl1(uq), be.bn.pack_tl1(dq)
            ref = be.bn.expert_ffn_w2a8(x, gq, uq, dq, pt=True).astype(np.float64)
            yield Case("expert_ffn", f"H{H}_I{I}_TL1", {"H": H, "I": I}, "i8",
                       fmt="tl1",
                       target=lambda gt=gt, ut=ut, dt=dt, x=x:
                           be.bn.expert_ffn_tl1(x, gt, ut, dt, pt=True),
                       ref=ref, weight_bytes=wbytes,
                       notes="fused expert FFN, TL1 (~2x A)")


@register("gemv_fp8", ["cpu"])
def _fp8(be, preset, formats):
    for N, K in SHAPES[preset]:
        W = (np.random.randn(N, K) * 0.05).astype(np.float32)
        scale = (np.abs(W).max(1) / 448.0).astype(np.float32)
        lut = np.array([_e4m3(b) for b in range(256)], np.float32)
        codes = np.abs(W[:, :, None] / scale[:, None, None]
                       - lut[None, None, :]).argmin(2).astype(np.uint8)
        x = np.random.randn(K).astype(np.float32)
        ref = (lut[codes] * scale[:, None]).astype(np.float64) @ x.astype(np.float64)
        yield Case("gemv_fp8", f"N{N}_K{K}", {"N": N, "K": K}, "fp8",
                   target=lambda codes=codes, scale=scale, x=x:
                       be.bn.gemv_fp8(codes, scale, x, impl="auto"),
                   ref=ref, weight_bytes=codes.nbytes, flops=2 * N * K,
                   notes="fp8 attention/head GEMV")


def _e4m3(b):
    s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 7
    if e == 0:
        v = m * 2.0 ** -9
    elif e == 15 and m == 7:
        v = 0.0
    else:
        v = (1 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


# --------------------------------------------------------------------------- output
def _git_label():
    try:
        c = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_meta(backend_name, args):
    meta = {"git": _git_label(), "platform": platform.platform(),
            "python": platform.python_version(), "backend": backend_name,
            "preset": args.preset, "warmup": args.warmup, "iters": args.iters,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds")}
    try:
        meta["device"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                        capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        meta["device"] = "?"
    try:
        import torch
        meta["torch"] = torch.__version__
    except ImportError:
        pass
    return meta


def _shape_str(shape):
    return "×".join(str(v) for v in shape.values()) if isinstance(shape, dict) else str(shape)


def write_outputs(rows, meta, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (out_dir / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lines = ["# BitNet kernel benchmarks", "",
             f"- `{meta['git']}` · {meta.get('device','?')} · backend `{meta['backend']}` "
             f"· preset `{meta['preset']}` · warmup/iters {meta['warmup']}/{meta['iters']}",
             "",
             "| kernel | variant | shape | fmt | tk ms | best base | base ms | speedup | GB/s | W-GB/s | GFLOP/s | rel err |",
             "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['kernel']} | {r['variant']} | {_shape_str(r['shape'])} "
                         f"| {r.get('format') or ''} | _{r['status']}_ "
                         f"| {r.get('skip_reason','')[:40]} | | | | | | |")
            continue
        valid = {k: v for k, v in r.get("baselines", {}).items() if "ms" in v}
        bn_name = min(valid, key=lambda k: valid[k]["ms"]) if valid else ""
        bl = valid.get(bn_name, {})
        lines.append(
            f"| {r['kernel']} | {r['variant']} | {_shape_str(r['shape'])} "
            f"| {r.get('format') or ''} | {r['target_ms']:.4f} | {bn_name} "
            f"| {bl.get('ms', float('nan')):.4f} | {bl.get('speedup', float('nan')):.2f} "
            f"| {r.get('gbps', float('nan')):.1f} | {r.get('weight_gbps', float('nan')):.1f} "
            f"| {r.get('gflops', float('nan')):.0f} | {r.get('max_rel_err', float('nan')):.2e} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["torch", "cpu"], default="torch")
    ap.add_argument("--preset", choices=["smoke", "quick", "comprehensive", "a1"],
                    default="quick")
    ap.add_argument("--kernel", default="all")
    ap.add_argument("--formats", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    be = make_backend(args.backend)
    formats = args.formats.split(",") if args.formats else None
    want = (set(KERNEL_BUILDERS) if args.kernel == "all"
            else set(args.kernel.split(",")))
    rows = []
    for name, (builder, backends) in KERNEL_BUILDERS.items():
        if name not in want:
            continue
        if args.backend not in backends:
            rows.append({"schema": SCHEMA_VERSION, "kernel": name, "variant": "-",
                         "shape": {}, "status": "skip",
                         "skip_reason": f"not on {args.backend} backend"})
            continue
        try:
            cases = list(builder(be, args.preset, formats))
        except Exception as e:  # noqa: BLE001
            rows.append({"schema": SCHEMA_VERSION, "kernel": name, "variant": "-",
                         "shape": {}, "status": "error", "skip_reason": str(e)})
            continue
        for case in cases:
            print(f"  {case.kernel:16s} {case.variant:20s} {case.fmt or '':8s} ...",
                  flush=True)
            rows.append(run_case(case, be, args.warmup, args.iters, not args.no_check))

    meta = _env_meta(args.backend, args)
    run_id = f"{be.name}-{args.preset}-{args.kernel.replace(',', '+')[:24]}"
    out_dir = RESULTS_ROOT / _dt.date.today().isoformat() / run_id
    table = write_outputs(rows, meta, out_dir)
    print("\n" + table)
    print(f"\nwrote {out_dir}/{{run.json,results.jsonl,summary.md}}")


if __name__ == "__main__":
    main()
