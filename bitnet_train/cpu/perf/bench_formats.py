"""Packing-format bake-off bench (moe_train_plan §7.3: 'prototype >=2, measure').

Times the ternary decode GEMV in each packed format on one core at the Qwen3-15B-A2B
expert shapes, and reports packed-weight GB/s (the roofline quantity). Run:

    .venv/bin/python bitnet_train/cpu/perf/bench_formats.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bitnet_cpu as bn  # noqa: E402

rng = np.random.default_rng(0)


def pack_bitnet(W, per_tensor=False):
    W = np.ascontiguousarray(W, np.float32)
    N, K = W.shape
    nb = K // 32
    Wb = W.reshape(N, nb, 32)
    if per_tensor:
        scale = np.full((N, nb), max(np.abs(W).mean(), 1e-5), np.float32)
    else:
        scale = np.maximum(np.abs(Wb).mean(axis=2), 1e-5).astype(np.float32)
    q = np.clip(np.rint(Wb / scale[..., None]), -1, 1).astype(np.int32)
    code = (q + 1).astype(np.uint32).reshape(N, nb, 8, 4)
    out = np.zeros((N, nb, 10), np.uint8)
    out[:, :, 0:2] = scale.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:10] = (code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4)
                       | (code[..., 3] << 6)).astype(np.uint8)
    return out


def timeit(fn, reps=200, warmup=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    shapes = [(768, 2048), (2048, 768), (2048, 2048)]
    print(f"{'(N, K)':>14} {'variant':>22} {'µs':>8} {'GB/s packed':>12}")
    for N, K in shapes:
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.05
        wq = pack_bitnet(W)
        wb = bn.pack_b3(wq)
        wt = bn.pack_tl1(wq)
        xq = rng.integers(-127, 128, K).astype(np.int8)
        scratch = np.empty(K // 2 * 32, np.int8)
        runs = [
            ("A 2-bit scalar", wq.nbytes, lambda: bn.gemv_w2a8(wq, xq, 0.01, impl="scalar")),
            ("A 2-bit NEON", wq.nbytes, lambda: bn.gemv_w2a8(wq, xq, 0.01, impl="neon")),
            ("A 2-bit NEON pt", wq.nbytes, lambda: bn.gemv_w2a8(wq, xq, 0.01, pt=True, impl="neon")),
            ("B base-3 scalar", wb.nbytes, lambda: bn.gemv_b3(wb, xq, 0.01)),
            ("C tl1 scalar", wt.nbytes, lambda: bn.gemv_tl1(wt, xq, 0.01, impl="scalar")),
            ("C tl1 NEON", wt.nbytes,
             lambda: bn.gemv_tl1(wt, xq, 0.01, impl="neon", lut_scratch=scratch)),
            ("C tl1 NEON pt", wt.nbytes,
             lambda: bn.gemv_tl1(wt, xq, 0.01, pt=True, impl="neon", lut_scratch=scratch)),
        ]
        for name, nbytes, fn in runs:
            us = timeit(fn) * 1e6
            print(f"{str((N, K)):>14} {name:>22} {us:8.1f} {nbytes / (us * 1e-6) / 1e9:12.2f}")
        print()


if __name__ == "__main__":
    main()
