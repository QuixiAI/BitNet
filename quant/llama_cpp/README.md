# Pinned llama.cpp TQ1_V integration

This directory carries the self-contained llama.cpp integration required by
[`quant_spec.md`](../quant_spec.md). It never imports or edits `~/llama.cpp`.
The local tree is only a read-only reference. The patch is locked to upstream
revision `a5822222909b785f23ddc74ce3c8f85bd0e38562`; its source revision,
digest, registry IDs, coverage, and limitations are machine-readable in
[`integration.json`](./integration.json).

## What the patch implements

The patch adds exact GGML registry-revision-1 types 43 through 47, their
46/50/44/48/48-byte block layouts, `gguf-py` sizes, full packed-payload
validation, model-local codebook loading and SHA-256 verification, row-scale
loading, and a permanent scalar CPU `MUL_MAT` path. The runtime performs
ties-to-even A8 token quantization once per input token and supports decode,
small batch, and prefill through the same graph operation.

J, the pinned llama.cpp IQ1 grid (I), and product (P) encodings share the
physical V11/V12 types and are dispatched by the embedded codebook metadata.
Generic block scales, external row scales, and affine A4 recovery are decoded
without first materializing dense weights. Unsupported devices fail closed:
TQ1 tensors are forced to the CPU backend because this patch makes no Metal,
CUDA, Vulkan, or other accelerator claim.

The generic `llama-quantize` path intentionally does not construct TQ1 data.
A codebook, exact indices, rounded scales, tensor policy, and provenance are
model-specific canonical state; reconstructing them from a dense GGUF would
violate the no-rediscovery contract. Produce them with `quant/quant.py` and
write them byte-for-byte with `quant/export_gguf.py`.

## Apply to a disposable clone

Create a separate clone from the read-only reference and run the guarded
helper from the BitNet repository root:

```bash
git clone --no-local ~/llama.cpp /tmp/llama.cpp-tq1
git -C /tmp/llama.cpp-tq1 checkout a5822222909b785f23ddc74ce3c8f85bd0e38562
.venv/bin/python quant/llama_cpp/apply_and_test.py \
  --target /tmp/llama.cpp-tq1 \
  --jobs 8
```

The helper refuses the wrong revision, a dirty target, a changed patch hash,
or an inapplicable patch. It then configures a CPU-only Release build, builds
the llama library and upstream backend tests, and compiles/runs the standalone
conformance test. It also constructs a deterministic one-layer Llama through
canonical PTQ and exact GGUF export, then loads and prefills that GGUF with the
patched library. To validate without changing the clone:

```bash
.venv/bin/python quant/llama_cpp/apply_and_test.py \
  --target /tmp/llama.cpp-tq1 \
  --check-only
```

The conformance test covers all five physical profiles, batch/prefill
broadcasting, independent scalar output calculation, FP16 row/block scales,
A4 rational arithmetic, and rejection of a reserved index. Python exporter
tests separately prove canonical-artifact-to-GGUF tensor, index, scale,
codebook, and metadata identity. The tiny model test additionally exercises
QuantSpec/codebook SHA verification, runtime tensor bindings, row-scale graph
wiring, model loading, and finite prefill logits. Use `--skip-model-test` only
when the BitNet Python environment is unavailable.

## Coverage boundary

This is a scalar correctness/reference implementation for dense Llama
`MUL_MAT`. It does not claim `MUL_MAT_ID` for MoE models, GPU execution, or a
speedup. The repository-native CPU K-track is the optimized deployment
backend. Timing evidence and the keep/reference decision are recorded in
[`bitnet_train/perf/optimization_status.md`](../../bitnet_train/perf/optimization_status.md)
and [`bitnet_train/cpu/perf/tq1_v.md`](../../bitnet_train/cpu/perf/tq1_v.md).

The integration does not by itself establish Llama-3.2-1B model quality.
A release still needs the held-out quality matrix described by the spec.
