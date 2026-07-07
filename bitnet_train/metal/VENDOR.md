# Vendored Metal kernels (from QuixiCore-Metal)

This directory is a **curated, self-contained copy** of the PyTorch/MPS Metal kernel stack
from `QuixiCore-Metal`, brought in to build the BitNet healing trainer's forward/backward
kernels here without depending on the source repo. Everything builds with only PyTorch (MPS)
and Xcode's Metal toolchain — no MLX, no CMake.

## Provenance

- Source: `~/QuixiCore/QuixiCore-Metal` (a ThunderMittens / ThunderKittens–derived tiled
  Metal kernel library; internal Metal namespace `mittens`).
- License: MIT — see `LICENSE` (© HazyResearch, © QuixiAI). The header-only substrate under
  `include/metal/` is derived from ThunderMittens (Apple MSL port of ThunderKittens).
- Copied on 2026-07-06. This is a *snapshot*; it does not track upstream automatically.

## What was copied (and what was not)

Copied:
- `include/metal/**` — the full header-only substrate (tile/vec types, `mma`, reductions,
  `dequant`, RNG, …). Small (~770 KB, 83 headers); pulled in by every kernel via `tk.metal`.
- `kernels/common/tk_launch.h` — host-side dispatch (all `launch_*` functions; self-contained).
- A **curated subset** of kernel `.metal` sources (see `tk_torch/__init__.py` `_METAL_SOURCES`):
  - `quantization/{quant_rt, act_quant, qgemm, qgemm_int, qgemv, qgemv_int}` — activation int8
    quant + BitNet ternary matmuls (dequant-to-half and integer W2A8, prefill + decode).
  - `matmul/{matmul_custom, gemm_staged}` — dense bf16/f32 GEMMs (for the STE backward products).
  - `norms/rms_norm`, `activations/glu`, `utils/cross_entropy`, `optimizers/optim/adamw` — the
    whole-model hot path (fwd + fused bwd where present).
- `tk_torch/torch_kernels.mm` — the ObjC++ extension (verbatim). It compiles against
  `tk_launch.h` alone, so it builds regardless of which kernel subset ships; pipelines are
  created lazily per call, so uncompiled ops are simply never requested.

**Not** copied: the MLX bindings and the vendored MLX source tree (`bindings/mlx`, hundreds of
files), the Xcode project, per-op MLX `Primitive` `.h`/`.cpp` (not needed for the MPS path),
`perf/`, `tests/`, and all inference-only kernels not on the training path (paged attention,
speculative decoding, MoE, MLA, sampling, KV-cache ops, …).

## Layout

```
include/metal/**              substrate (compile with -I include/metal)
kernels/common/tk_launch.h    host dispatch
kernels/<family>/*.metal      curated kernel subset
kernels/bitnet/*.metal        NEW BitNet-training kernels we develop (auto-compiled)
tk_torch/__init__.py          builds the metallib + JIT ObjC++ ext; the training Python API
tk_torch/torch_kernels.mm     ObjC++ dispatch onto torch's MPS command stream
pyproject.toml                optional editable install (`pip install -e bitnet_train/metal`)
build.sh                      standalone metallib build/verify (no torch)
```

## Build & use

```bash
# Requires the Metal toolchain: xcodebuild -downloadComponent MetalToolchain
./build.sh check                      # standalone: compile metallib + list functions

# From Python (no install needed):
python -c "import sys; sys.path.insert(0,'bitnet_train/metal'); import tk_torch as tk; print(tk.rms_norm)"
```

`import tk_torch` compiles `bitnet.metallib` (via `xcrun metal`) and JIT-builds the ObjC++
extension on first import (~10 s), caching both. The metallib is rebuilt automatically when any
`.metal` source or substrate header changes.

Verified on 2026-07-06 (Apple Silicon, torch 2.14 MPS): `rms_norm`, `matmul_custom`,
`quantize_per_token_int8`, `adamw`, `cross_entropy_fwd`, `swiglu` all load and run correctly;
`cross_entropy_fwd` matches `torch.nn.functional.cross_entropy` exactly.

## Adding the new BitNet kernels

Drop new `.metal` sources into `kernels/bitnet/` — they are globbed into the metallib
automatically. Add the host launcher to `kernels/common/tk_launch.h`, a dispatch function +
`m.def` to `tk_torch/torch_kernels.mm`, and a Python wrapper to `tk_torch/__init__.py`. The
kernels to develop are specified in `docs/new-kernels.md`.

## Attribution / license discipline

Keep `LICENSE` in this directory. When adding kernels derived from the substrate, preserve the
MIT notices. Per QuixiCore's `CLAUDE.md`/`AGENTS.md`: correctness-first, and record a measured
perf run before claiming any kernel optimization.
