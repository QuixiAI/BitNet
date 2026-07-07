# BitNet kernel benchmarking

Two kernel stacks live in this repo and both are benchmarked here:

- **Metal training kernels** (`metal/tk_torch`, MPS) — the healing forward/backward
  path: weight/act quantizers, ternary GEMM/GEMV, fused KD/CE, MoE, monitors.
- **CPU K-track engine** (`cpu/`, numpy+ctypes) — the batch-1 decode deployment
  path: ternary/fp8 GEMVs, fused expert FFN, int8-KV attention.

The operating guide is [`perf.md`](perf.md). The running notebook is
[`optimization_status.md`](optimization_status.md). Standing per-kernel numbers
already recorded during development live in
[`../metal/perf/`](../metal/perf/) and [`../cpu/perf/`](../cpu/perf/); this
directory is the disciplined harness that supersedes ad-hoc bench scripts.

## Entry point

```bash
.venv/bin/python bitnet_train/perf/bench_kernels.py --backend torch --preset smoke  --kernel all
.venv/bin/python bitnet_train/perf/bench_kernels.py --backend cpu   --preset quick  --kernel gemv_w2a8,gemv_tl1,expert_ffn
.venv/bin/python bitnet_train/perf/bench_kernels.py --backend torch --kernel qgemv  --formats bitnet,tq2_0
```

`--backend torch` runs the MPS kernels; `--backend cpu` runs the bn_* engine.
Cases self-skip (with a recorded reason) when a kernel/format/backend is
absent. Each run writes, under the git-ignored `results/`:

```text
results/YYYY-MM-DD/<run-id>/run.json       environment + invocation metadata
results/YYYY-MM-DD/<run-id>/results.jsonl  schema v1, one row per case
results/YYYY-MM-DD/<run-id>/summary.md     human-readable table
```

Copy summaries that matter into `optimization_status.md`. Record enough context
to reproduce: Apple Silicon model, macOS/toolchain version, backend, kernel
variant, shape, dtype, quant format, warmups/iters, correctness tolerance,
observed error, git commit (the harness captures most of this in `run.json`).
