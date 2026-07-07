# Agent Instructions

This repo hosts the 1.58-bit healing training stack (`bitnet_train/`) on top of
the BitNet inference trees. It owns two kernel stacks — the Metal training
kernels (`bitnet_train/metal/tk_torch`, MPS) and the CPU K-track decode engine
(`bitnet_train/cpu/`). Kernel work must be correctness-first, measurement-driven,
and recorded in the performance notebook.

## Read First

- Plans (single source of truth): `docs/train_plan.md`, `docs/moe_train_plan.md`.
- Performance operating guide: `bitnet_train/perf/perf.md`.
- Optimization notebook: `bitnet_train/perf/optimization_status.md`.
- Standing per-kernel numbers: `bitnet_train/metal/perf/`, `bitnet_train/cpu/perf/`.

## Performance Optimization Requirement (hard rule)

**Any time you touch kernel code, you must run a focused optimization pass on the
affected kernel(s) before pushing.** "Kernel code" means: `.metal` sources,
`tk_launch.h` launchers, `tk_torch/torch_kernels.mm` dispatch, the mittens
substrate under `metal/include/`, the CPU engine `cpu/src/bitnet_cpu.c`, any
kernel routing/threshold change, or the benchmark harness itself. A performance
claim also triggers this rule.

A valid pass includes:

- The kernel, integration path (torch-MPS / cpu), dtype/quant format, and the
  shape set benched.
- Correctness for the touched path (its `pytest` test green; the new number
  carries a correctness tolerance + observed max abs/rel error).
- Baseline/current timing and candidate timing when testing a variant.
- Device (Apple Silicon model), macOS/toolchain version, command line, warmups,
  iterations, median, and variance or p20/p80. `bench_kernels.py` captures most
  of this in `results/.../run.json`.
- A keep/reject decision written into `bitnet_train/perf/optimization_status.md`.

If the required runtime is unavailable (no Metal/MPS for a Metal-kernel change),
do not push a kernel optimization or speedup claim — report the blocker, or
restrict the change to scaffolding/docs with no performance claim. Pure
docs/metadata commits may skip the pass but must not claim a speedup.

## How To Optimize

- Start from `bitnet_train/perf/perf.md`; form a bottleneck hypothesis before
  editing (memory- / compute- / launch- / sync-bound).
- Change one meaningful factor at a time: tile/launch geometry, memory layout,
  fusion, barrier placement, dequant/unpack strategy, routing threshold, or
  format specialization.
- Compare against framework and naive-decomposed baselines
  (`dequantize(wq) @ x`, `quantize + qgemm` composed).
- Keep only wins that pass correctness, improve realistic priority shapes, and do
  not regress supported edge shapes or tolerances.
- Store raw output under `bitnet_train/perf/results/` (git-ignored); copy durable
  conclusions into `optimization_status.md` with the git label.

## Useful Commands

```bash
.venv/bin/python -m pytest tests/ -q
sh bitnet_train/metal/build.sh check              # compile + list metallib functions
sh bitnet_train/cpu/build.sh                      # rebuild libbitnet_cpu
.venv/bin/python bitnet_train/perf/bench_kernels.py --backend torch --preset quick --kernel <kernel>
.venv/bin/python bitnet_train/perf/bench_kernels.py --backend cpu   --preset quick --kernel <kernel>
```

Use Xcode/Instruments or Metal command-buffer timing when a benchmark A/B does
not explain a bottleneck.

## Engineering Hygiene

- Check `git status` before editing; do not revert user changes.
- We take NO external dependencies for the kernel stacks — `~/llama.cpp` and
  `~/QuixiCore` are READ-ONLY references, never imports. Vendor, transcribe, cite.
- Scalar/reference oracle first, kept forever; every fast variant diffs against it
  before a number is trusted.
- Update tests, launchers, bindings, and the Python wrapper when changing a
  kernel's public behavior.
- Keep commits scoped and descriptive; commit/push only when asked.
