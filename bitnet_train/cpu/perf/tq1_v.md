# TQ1_V native CPU kernel

Standing correctness and performance record for the schema-2 canonical packed
CPU path (`bn_tq1_gemv_scalar`, `bn_tq1_gemv_neon`, and their batched GEMM
entry points). The permanent scalar path
unpacks each physical V11/V12 index, rejects reserved indices, decodes the
backend-private expanded codebook, accumulates exact integer dots, and applies
the declared row/block and activation scale units. The optimized arm64 path
groups four codewords and uses signed dot-product instructions without changing
the scale epilogue.

## Coverage

- Profiles: V11/V12 J and P row-scale, V11/V12 J block-scale, V11 J A4 row-scale.
- Codebooks: any validated expanded J/I/P codebook (the pinned IQ1 table uses
  the same direct-joint path).
- Activations: A8 token and A8 block256.
- Row scales: FP16 and BF16; embedded block scales: FP16.
- Output: FP32 native, cast by the PyTorch wrapper to the requested model dtype.
- Shapes: any positive `N`, `K % 256 == 0`; the A1 run covers all four distinct
  Llama-3.2-1B projection dimensions, including 512-row q/k/v asymmetry.
- Workloads: batch-one decode plus small-batch/prefill M=4 and M=32. The batch
  entry point retains exact packed execution but is a looped GEMV path, not a
  tiled prefill kernel.

The loader reports canonical/original bytes, expanded resident bytes, repack
time/hash, peak temporary bytes, and whether the canonical representation stays
resident. V11/V12 expanded codebook plus legal bitmap uses 18,432/36,864 bytes.

## 2026-07-15 focused pass

Device/toolchain: Apple M4 Max; macOS 26.5.2 build 25F84; Apple clang 21.0.0;
Python 3.12.9; PyTorch 2.14.0.dev20260706; git label `6f5c3a6-dirty`.

```bash
sh bitnet_train/cpu/build.sh
.venv/bin/python -m pytest tests/test_tq1_cpu_native.py tests/test_cpu_engine.py -q
.venv/bin/python bitnet_train/perf/bench_kernels.py \
  --backend cpu --preset a1 --kernel gemv_tq1 --warmup 10 --iters 50
.venv/bin/python bitnet_train/perf/bench_kernels.py \
  --backend cpu --preset a1 --kernel gemm_tq1 --warmup 10 --iters 50
```

Correctness was 15/15 tests. Tolerance was FP32 `atol=1e-6, rtol=1e-6`;
benchmark maxima were 1.48e-6 absolute and 8.42e-8 relative. The optimized and
scalar integer accumulators match exactly.

Across the four Llama-3.2-1B shape classes, grouped ARM dot-product is
2.78–3.17× faster than the scalar kernel. V11 medians are
0.1136/0.4040/1.5939/1.5621 ms for 512×2048, 2048×2048, 8192×2048, and
2048×8192; V12 medians are 0.1159/0.4080/1.5878/1.5609 ms. Dispersion and the
full baseline table are preserved in `bitnet_train/perf/optimization_status.md`.

Decision: keep grouped dot-product. Reject the activation subset-LUT candidate,
which regressed 768×2048 from about 0.23 ms to 0.41 ms. Dequantized dense BLAS
remains 5.3–8.0× faster when its much larger dense matrix is assumed resident;
this result is a packed-memory implementation milestone, not a claim of speed
leadership over Accelerate.

The batch-dispatch follow-up hoists Python/ctypes dispatch for M tokens. It is
1.00–1.19× faster than repeated native GEMV at M=4 and 1.01–1.17× at M=32,
with bit-identical outputs. Focused correctness is 25/25; benchmark max
absolute/relative error is 1.73e-6/7.82e-8 under the combined FP32
`atol=1e-6, rtol=1e-6` gate. Keep it for small-batch integration. Reject any
prefill speed claim: dense resident BLAS remains 3.36–58.45× faster, so the
next candidate must reuse decoded weights across tokens rather than loop GEMV.

## Pinned llama.cpp scalar reference

The exact integration is carried as a revision-locked patch under
quant/llama_cpp/, not as an import from the read-only local reference tree.
Its standalone test covers the five physical types, A8 token quantization,
batch/prefill dimensions, FP16 row/block scales, affine rational arithmetic,
and reserved-index rejection.

A separate 2026-07-15 focused pass used the same four Llama-3.2-1B projection
shapes at M=1 and M=32, all five types, 8 threads, 3 warmups, and 15 measured
iterations. Every one of the 40 rows was bit-identical to the packed scalar
oracle. V12-J-R medians range from 0.136 to 0.696 ms at M=1 and 1.251 to
19.100 ms at M=32. The optimistic decoded-F32 baseline is 1.21–2.23× faster
for decode and 3.60–5.03× faster for M=32 across the complete matrix.

Decision: keep the patched llama.cpp implementation as the permanent portable
CPU reference and reject any speed claim. The run also fixed activation-code
parity in both native CPU implementations by evaluating the specified
round_to_even(x / scale) directly instead of multiplying by a rounded
reciprocal. Full setup, p20/median/p80 values, and errors are recorded in the
optimization notebook and raw JSONL result.

## A8 half-even follow-up

The shared native activation quantizer was A/B tested against the exact
pre-change `6f5c3a6` dylib after the tie-case fix. A targeted FP32 fixture proves
the prior reciprocal multiply can be off by one int8 code; the kept direct
division plus `__builtin_roundevenf` has zero code error and all 53 focused CPU
tests pass. At K=512/2048/8192 its medians are 0.002531/0.003650/0.008042 ms,
1.010×/1.024×/1.037× the old implementation in the same-process A/B. The
manual `floorf`/`fmodf` candidate was rejected for a 1.15–1.74× regression.
Raw p20/p80/CV, environment, and the complete reproducible command are in
`bitnet_train/perf/results/a8_round_even_20260715/run.jsonl` and the optimization
notebook.
