# BitNet Kernel Performance Handbook

Operating guide for optimizing the two kernel stacks in this repo — the Metal
training kernels (`metal/tk_torch`, MPS) and the CPU K-track engine (`cpu/`). The
goal is not to collect tricks; it is a disciplined loop: find references, form a
bottleneck hypothesis, measure a clean baseline, run controlled experiments, keep
only verified wins, and record enough that the next pass starts from evidence.

Adapted from QuixiCore-Metal's `perf/perf.md`; the running notebook is
`optimization_status.md`.

## Principles

Optimization starts from correctness and measurement. A change is not a win until
it passes the kernel's correctness tests, improves the target metric on realistic
shapes, and does not regress a supported edge shape. The repo doctrine
(`CLAUDE.md`/plan docs): scalar/reference oracle first, kept forever; every fast
variant diffs against it before a number is trusted; a measured perf run is
recorded before any optimization claim.

Attack a specific bottleneck:

- **Memory-bound** (most quant decode): reduce bytes moved, improve coalescing
  and cache reuse, use narrower packed formats, avoid extra passes.
- **Compute-bound** (our ternary decode — `first_cut.md` found the CPU engine
  unpack-bound at ~7 GB/s/core against a >100 GB/s machine): raise arithmetic
  intensity, remove per-weight scalar work, use LUT/dotprod paths,
  `simdgroup_matrix` where it applies.
- **Latency/launch-bound**: fuse tiny kernels, batch dispatches, resident work.
- **Synchronization-bound**: drop unnecessary `threadgroup_barrier`, prefer
  simdgroup-local reductions when cross-simdgroup sharing does not pay.

Apple baseline assumption (do not blindly port CUDA): Metal has no
`cp.async`/TMA; explicit threadgroup-memory staging often does not beat enough
occupancy. Our own measurements agree — the integer-exact `qgemm_w2a8` loses to a
dense GEMM on fake-quant operands at every training batch size (why the training
forward is `F.linear(x_q, w_deq)`, `metal/perf/bitnet_training_kernels.md`).

## Device context

Apple M4 Max, 40-core GPU, 128 GB, theoretical DRAM ~546 GB/s. Judge
memory-bound kernels as a fraction of that roofline. Repeatedly re-read buffers
can sit in the SLC and report above-DRAM bandwidth — A/B comparisons stay valid,
but "above peak GB/s" means cache-resident, not magic. The CPU engine's
single-core numbers cite ~135 GB/s single-stream copy (`cpu/perf/first_cut.md`).

## What already exists (start from evidence)

- `metal/perf/bitnet_training_kernels.md` — training forward routing, the
  measured integer-vs-dense verdict.
- `metal/perf/gap_kernels_2026-07-06.md` — ternary_stats/code_flip (134–259× vs
  PyTorch unpack), fused dense-KL crossover, K2/attn_decode ACADEMIC verdicts,
  TQ2_0 addendum.
- `cpu/perf/first_cut.md` — the ~7 GB/s unpack-bound diagnosis.
- `cpu/perf/formats_bakeoff.md` — TL1 wins the packing bake-off 2.5–2.7× over the
  2-bit NEON path; the `bench_kernels.py --backend cpu` `expert_ffn` case now
  reproduces the FFN-level ~2× directly.

## Measurement harness

`bench_kernels.py` (schema v1). Every result carries: git label, kernel + exact
entry point, backend, device/platform/versions, shape/dtype/format, warmup +
timed-iter counts, median + p20/p80 + CV, correctness tolerance and observed max
abs/rel error, and derived throughput (GB/s, packed-weight GB/s for quant decode,
GFLOP/s). Fields are written to `results/.../results.jsonl`.

Timing discipline (`time_thunk`): warm by TIME not call count (GPU clocks decay
during host setup); batch several calls per sync so the submit+sync floor
(~0.15–0.25 ms on MPS) does not swamp small kernels; report throughput-style
per-call latency. One small kernel per sync measures dispatch latency, not the
kernel — relevant only for launch-bound routing decisions.

Force work and synchronize: MPS via `torch.mps.synchronize()`; the CPU backend
is synchronous. Do not time only Python dispatch.

### Metrics

Quantized decode — the decisive metric is effective PACKED-weight bandwidth:

```text
weight_GBps = packed_weight_bytes_read / time_s / 1e9
```

GEMM-like: `FLOPs = 2*M*N*K`. Attention decode reads `T*Hkv*D` of KV per step.
Elementwise/quantizer kernels: a conservative read+write byte count (be explicit
when it ignores cache reuse).

### Baselines

Bench against at least: (1) a framework baseline (`torch` dense matmul, SDPA,
`F.cross_entropy`); (2) a naive decomposed baseline for fused/quant kernels
(`dequantize(wq) @ x`, `quantize + qgemm` composed); (3) the current in-repo
kernel before the experiment. The harness records a `baselines` dict per case.

## Shape strategy

Do not optimize only square toys. Each kernel needs: small edge shapes,
tile-aligned fast-path shapes, tile-ragged shapes, real model shapes (Qwen3-15B
expert 2048×768 / 768×2048, Llama attention/MLP, `K=4096`, `N=14336`), and stress
shapes (long context, large K/N). The harness `SHAPES` presets encode these;
extend them per kernel rather than hand-timing one case.

## The loop

1. Pick a kernel and a bottleneck hypothesis; read the reference (`~/llama.cpp`
   ggml-metal for quant GEMV geometry, `~/QuixiCore` for the mittens substrate —
   READ ONLY, we vendor, we do not depend).
2. `bench_kernels.py --preset comprehensive --kernel <k>` for a clean baseline.
3. Change ONE thing; re-run correctness tests (`pytest tests/`) then the bench.
4. Keep the win only if correctness holds AND no supported shape regresses.
5. Copy the summary delta into `optimization_status.md` with the git label.

## Named open digs (from the status notebook)

- CPU ternary unpack was the bound; TL1 broke it (18–19 GB/s/core). Next: the
  multicore dispatch layer across experts (`first_cut.md` conclusion 3).
- fp8 GEMV is unpack-bound (~5 GB/s/core) and carries the larger decode-byte
  share; a nibble-split `tbl` decode is the candidate (`first_cut.md`).
- TQ2_0 Metal decode-scatters worse than the 10 B/32 bitnet format (66 B/256
  blocks); fine for eval, a ushort/mask-batched inner loop is the dig if it ever
  matters for training.
