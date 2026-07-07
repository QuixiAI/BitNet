# Claude Instructions

Follow `AGENTS.md` in this repository.

**Hard rule — kernel work:** any time you touch kernel code (`.metal` sources,
`tk_launch.h`, `tk_torch/torch_kernels.mm`, the mittens substrate under
`bitnet_train/metal/include/`, `bitnet_train/cpu/src/bitnet_cpu.c`, kernel
routing/thresholds, or the benchmark harness), run a focused optimization pass on
the affected kernel(s) **before pushing** — read `bitnet_train/perf/perf.md`
first, bench with `bitnet_train/perf/bench_kernels.py`, and record a keep/reject
decision (with device, shapes, correctness tolerance, and git label) in
`bitnet_train/perf/optimization_status.md`. If no Metal/MPS runtime is available
for a Metal-kernel change, report the blocker instead of pushing a speedup claim.
