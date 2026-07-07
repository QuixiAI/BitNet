# BitNet Training — Metal Kernels to Develop

Companion to `docs/train_plan.md`. This document specifies the Metal kernels the BitNet
healing trainer needs, distinguishing what we **reuse** (vendored from QuixiCore-Metal into
`bitnet_train/metal/`) from what we must **develop**. It is the build spec for the new
kernels.

## 0. The one principle that shrinks this work

BitLinear trains through a **straight-through estimator (STE)**: the forward uses the
quantized value; the backward passes gradients through `round`/`clamp` as identity. Concretely,
with fake-quant weight `w_q = weight_quant(W)` and fake-quant input `x_q = act_quant(x)`:

```
forward :  y      = x_q @ w_qᵀ                        (quantized matmul)
backward:  grad_x = grad_y @ w_q                       (DENSE matmul, dequantized weight)
           grad_W = grad_yᵀ @ x_q   → applied to the FP latent weight W
```

**The backward is two ordinary dense matmuls.** We therefore do **not** write ternary/int
backward kernels. We need only:

1. a fast **forward** ternary×int8 matmul — *reused* (`qgemm_w2a8` / `qgemm_bitnet`);
2. fast **dense bf16/f32 GEMMs** for the two backward products — *reused* (`matmul_custom`,
   `gemm_staged`);
3. cheap **elementwise quantizers** — activation int8 is *reused* (`quantize_per_token_int8`);
   weight→ternary is the **one missing kernel** (§3, K1).

So the net new kernel surface is small: a weight-quantize/pack kernel, plus optional fusions.

## 1. Kernels we reuse (already vendored + verified on MPS)

All are in `bitnet_train/metal/` and callable from `tk_torch` (verified 2026-07-06, torch 2.14
MPS). Exact Python API in `bitnet_train/metal/tk_torch/__init__.py`.

| Role in BitLinear | Kernel (tk_torch) | Notes |
|---|---|---|
| Activation → int8 (per-token absmax) | `quantize_per_token_int8(x) → (codes i8, scale f32)` | the A8 in W1.58A8 |
| Fused SwiGLU + int8 (FFN input) | `silu_mul_quant_int8(x, gate)` | Llama FFN is SwiGLU |
| Forward ternary×int8 GEMM (prefill) | `qgemm_w2a8(wq, xq, a_scale)` | integer-exact; wq = packed bitnet blocks |
| Forward ternary GEMM (dequant-to-half) | `qgemm(wq, x, "bitnet")` | uses half tensor cores; faster on Apple |
| Forward ternary decode GEMV | `qgemv_w2a8(wq, xq, a_scale)` / `qgemv(wq,x,"bitnet")` | batch-1 eval/generate |
| Backward dense GEMMs (STE) | `matmul_custom(a, b)`, `gemm_staged` (via `_ext`) | grad_x, grad_W |
| RMSNorm fwd + fused bwd | `rms_norm`, `rms_norm_bwd_fused` | pre-norms + sub-norms |
| SwiGLU fwd + bwd | `swiglu` / `glu`, `glu_backward` | FFN |
| Fused cross-entropy fwd + bwd | `cross_entropy_fwd/bwd` | never materializes (T,V) logits |
| Fused AdamW step | `adamw(param, grad, m, v, …)` | fp32 moments |

### The BitNet packing format (reused `bitnet` dequant struct)

`qgemm_w2a8`/`qgemm(...,"bitnet")` consume weights in this exact layout (see
`include/metal/ops/warp/register/tile/dequant.metal`, struct `bitnet`; host packer
`quantize_bitnet` in QuixiCore `tk/quant.py`):

- Ternary `{-1,0,+1}`, **group size 32 along K**, one `half` **absmean** scale per group.
- Block = **10 bytes** per 32 weights: `{ half scale; uint8 qs[8]; }`, each `qs` byte holds
  four 2-bit codes; `code ∈ {0,1,2} → value = scale·(code−1)`.
- Packed weight `wq`: `uint8[N, K/32, 10]`. Activations `xq`: `int8[M, K]` with a per-token
  `a_scale`. Output `y = (Σ code·xq) · gscale · a_scale`.

## 2. The scale-granularity decision (resolve before writing K1)

Three quantization granularities are in play, and they differ:

- **b1.58 reference training** (`.reference/bitnet`): **per-tensor** absmean scale
  (`1/mean(|W|)` over the whole matrix).
- **QuixiCore `bitnet` kernel** (what we reuse): **per-group-of-32** absmean scale.
- **bitnet.cpp I2_S export**: **per-row-block absmax** (block 128 x86 / 64 ARM).

For **training**, the only hard requirement is that the forward quantization and the STE
fake-quant used in the backward are **identical** — otherwise gradients are biased against a
weight the forward never used. The export granularity is independent (the exporter
re-quantizes at the end, §6).

**Decision: adopt per-group-of-32 absmean scaling for A-track training**, matching the reused
kernel format. It is a strictly finer (more accurate) quantization than per-tensor, it is what
the fast `qgemm_w2a8`/`qgemm_bitnet` kernels expect, and it removes any forward/backward
mismatch. The pure-PyTorch oracle (Phase 0) must use the **same** per-group-32 scheme so it
agrees with the Metal path bit-for-bit (within bf16 tolerance).

**Correction (2026-07-06):** an earlier revision claimed `group_k = K` recovers per-tensor —
wrong: it gives per-ROW absmean (one scale per row's K elements), not one scale over the whole
matrix. True per-tensor (the b1.58 reference formula, and the **baseline of
`moe_train_plan.md` §3.7** because it survives TQ2_0's 256-block export re-scaling) needs a
whole-tensor reduction and is its own two-pass kernel: `weight_quant_ternary_pt`
(abssum-reduce, then encode with the one scale replicated into every packed block). Both
kernels are batched over a leading expert axis (E, N, K) — one dispatch quantizes a fused MoE
expert stack with per-expert-slice scales (K5), instead of thousands of tiny per-expert
launches.

## 3. New kernels to develop

### K1 — `weight_quant_ternary` (the one required kernel)

On-device latent-weight → ternary. This is the piece with **no equivalent** in either repo:
QuixiCore quantizes *activations* on device but packs *weights* only in host numpy
(`quantize_bitnet`), and re-quantizing every latent weight each step on the CPU would dominate
step time.

- **Inputs:** `W` bf16/f32 `[N, K]` (the FP latent weight); `group_k` (default 32).
- **Outputs (two, from one pass):**
  1. `wq` `uint8[N, K/group_k, 10]` — the packed `bitnet` blocks (for `qgemm_w2a8` forward).
  2. `w_deq` bf16 `[N, K]` — the dequantized ternary weight (for the STE **backward** dense
     GEMM `grad_x = grad_y @ w_deq`, and for the Phase-0 dense forward).
  Optionally also `scale` `half[N, K/group_k]` if a caller wants it separately.
- **Math (per group g of 32 weights in a row):**
  `s = mean(|W_g|)` (absmean; clamp min 1e-5); `code = clip(round(W_g / s), −1, +1) + 1`
  (→ `{0,1,2}`); pack 4 codes/byte into `qs[8]`; store `half(s)` at bytes `[0:2]`; and
  `w_deq = s · (code − 1)`.
- **Threadgroup structure:** one **simdgroup (32 lanes) per row** `n ∈ [0,N)`; the 32 lanes
  map one-to-one onto a 32-wide group, so the absmean is a single `simd_sum(|w|)/32` with no
  threadgroup memory. Loop groups along K. Mirrors QuixiCore's `quantize_per_token_int8`
  (`quant_rt.metal`) row/simdgroup pattern — start from that file. Grid `(N, 1, 1)`, 32 threads.
- **Dtypes:** `W` in {f32, f16, bf16}; scale computed/stored in fp32→half; `w_deq` bf16.
- **Why both outputs:** forward uses `wq` (packed, fast kernel); backward uses `w_deq` (dense).
  They come from the *same* `s`, guaranteeing forward/backward consistency (§2). Recomputing
  `w_deq` from `wq` (a dequant kernel) is an alternative, but emitting both in one pass is
  cheaper and keeps the scale identical by construction.

### The differentiable layer — `BitLinearSTE` (composition, not a new kernel)

A `torch.autograd.Function` wiring the kernels (pattern: AUM's
`aum_ssm/ops/metal/unfold_metal.py`; QuixiCore's `bindings/python/tk/autograd.py`). No new
Metal code — this is the Python glue that makes the reused kernels + K1 trainable.
**Implemented:** `bitnet_train/bitlinear_metal.py`.

```
forward(ctx, x, W):                       # W: fp32/bf16 latent [N,K]; x: [M,K]
    wq, w_deq = weight_quant_ternary(W)   # K1
    xq, a_scale = quantize_per_token_int8(x)          # reused
    x_q = (xq * half(a_scale)).to(bf16)   # fake-quant activation (the half-scale grid)
    y = F.linear(x_q, w_deq)              # dense bf16 GEMM on the fake-quant operands ★
    ctx.save_for_backward(w_deq, x_q)
    return y

backward(ctx, grad_y):                    # grad_y: [M,N]
    w_deq, x_q = ctx.saved
    grad_x = grad_y @ w_deq               # [M,K]  (STE: dense)
    grad_W = grad_y.T @ x_q               # [N,K]  (STE: dense; scales detached)
    return grad_x, grad_W
```

★ **Measured routing (2026-07-06, `metal/perf/bitnet_training_kernels.md`)** — this
deviates from the draft above it replaced: the integer-exact `qgemm_w2a8` has no tensor
cores and loses to a dense GEMM on the fake-quant operands at every training batch size
(29× at M=2048), so the training forward is `F.linear(x_q, w_deq)` — the same
"A8 = snap to the 8-bit grid, then run the fast GEMM" convention as QuixiCore's host-side
W·A8 parity path. The integer kernels are the *inference* route: `qgemv_w2a8` for frozen
batch-1 decode (2.2–3.8× over dense) and `qgemm(wq,·,"bitnet")` for frozen prefill from
packed-only weights. The backward uses `torch.matmul` (measured ≥ `matmul_custom` at the
A1 shapes, no transpose copy).

Notes: the per-group/per-token scales are treated as **constants** in the backward (standard
BitNet STE) — do not backprop through `mean(|W|)` or the activation absmax. `grad_W` updates
the **FP latent** `W`; the ternary `wq` is never a parameter. Activation codes clamp to
**[-127, 127]** (`tk_int8_encode`), not −128 — the Phase-0 oracle mirrors this.

### K2 (optional, perf) — `bitlinear_forward` fused

Fuse `quantize_per_token_int8` + `qgemm_w2a8` into one dispatch to save an activation
round-trip (QuixiCore already fuses the FFN case as `silu_mul_quant_int8` → GEMM; this is the
attention/qkv analogue). Build only after Phase-0/1 correctness and a measured baseline. Not
required for correctness.

### K3 (optional, perf) — `weight_quant_ternary_bwd_fused`

If profiling shows the two dense backward GEMMs plus the K1 dequant dominate, a fused
`grad_x`-with-in-kernel-dequant kernel avoids materializing `w_deq`. Deferred; the dense path
(K1 emits `w_deq`, reuse `matmul_custom`) is the baseline.

**Summary: K1 is the only kernel required to train.** K2/K3 are speedups gated on measured
need.

## 4. Wiring a new kernel into the vendored stack

For K1 (and any new kernel), touch four places — all shown by the existing
`quantize_per_token_int8` as a worked example:

1. **Kernel source:** `bitnet_train/metal/kernels/bitnet/weight_quant_ternary.metal`
   (`#include "tk.metal"`). Auto-globbed into `bitnet.metallib` by `tk_torch/__init__.py`
   (`_METAL_SOURCES += kernels/bitnet/*.metal`).
2. **Host launcher:** add `template<class E> void launch_weight_quant_ternary(E& e, …)` to
   `kernels/common/tk_launch.h` (set pipeline name, bind buffers, dispatch `(N,1,1)×32`).
3. **ObjC++ dispatch + binding:** add a `weight_quant_ternary_mps(...)` function and an
   `m.def("weight_quant_ternary", …)` to `tk_torch/torch_kernels.mm` (mirror
   `quantize_per_token_int8_mps`).
4. **Python wrapper:** add `def weight_quant_ternary(W, group_k=32): return _ext…` to
   `tk_torch/__init__.py`, and the `BitLinearSTE` autograd.Function (in `bitnet_train/` model
   code, not the vendored dir).

Rebuild is automatic on next `import tk_torch` (metallib staleness is tracked); or run
`bitnet_train/metal/build.sh` standalone.

## 5. Testing & acceptance

Follow QuixiCore's discipline (`CLAUDE.md`/`AGENTS.md`): correctness-first; a measured perf
run recorded before any optimization claim.

- **K1 correctness:** compare `wq`/`w_deq` against the numpy oracle `quantize_bitnet` /
  `dequantize_bitnet` (host packer, already in QuixiCore `tk/quant.py`) — codes ⊆ {−1,0,+1},
  per-group scale = absmean, dequant rel-err 0 (integer codes) / < 2e-2 for `w_deq` bf16.
- **BitLinearSTE parity:** the fused Metal forward must match the **pure-PyTorch** BitLinear
  (Phase 0) within bf16 tolerance; gradients must match `torch.autograd.gradcheck` on the
  pure-PyTorch STE oracle (the reference trainer's `test_quant.py` already gradchecks the STE
  activation path — extend it to the weight path).
- **End-to-end:** one training step of the 1B model on the `metal` backend must equal the
  `reference` backend loss within tolerance (the portability gate).
- **Perf:** record K1 and the fused forward in a `perf/` note (µs, GB/s) vs the pure-PyTorch
  path before claiming a speedup; measure route thresholds, don't guess.

## 6. Relationship to export (not a training kernel)

Training keeps bf16 latent weights + on-the-fly ternary (per-group-32). **Export is separate**
and already tooled: `utils/convert-hf-to-gguf-bitnet.py --outtype f32` → `llama-quantize I2_S`
re-quantizes to bitnet.cpp's row-block absmax format (§7 of `train_plan.md`). Do **not** try to
make the training packer emit I2_S; the granularities differ by design and the exporter owns
the conversion.

## 7. Kernel inventory as built (2026-07-06 drop)

Beyond K1 and the §3 composition, the tree now carries (perf where measured in
`bitnet_train/metal/perf/bitnet_training_kernels.md` and `bitnet_train/cpu/perf/first_cut.md`):

| Kernel | Where | Serves | Status |
|---|---|---|---|
| `weight_quant_ternary` (batched E,N,K) | metal/bitnet | K1/K5, expert stacks in one dispatch | tested |
| `weight_quant_ternary_pt` (per-tensor, two-pass) | metal/bitnet | moe_train_plan §3.7 baseline (group_k=K is per-ROW, not per-tensor — §2 correction) | tested |
| `fake_quant_int8`, `silu_mul_fake_quant_int8` (K4) | metal/bitnet | one-pass fake-quant fwd; + per-step quant cache in `bitlinear_metal.py` | tested, 1727→1437 µs fwd |
| `adamw_masked` | metal/optim | cold-expert decay mask (moe_train_plan §4.3/§5.6) | tested incl. erosion demo |
| `kd_kl_topk_fwd/bwd` | metal/bitnet | sparse-teacher KD-KL (A6c cache), renorm + other-bucket tails | tested vs dense KL |
| `rms_norm_dyn` | metal/norms | D=2048/3072+ (register-resident template capped at 1024) | tested |
| MoE pipeline + `bitnet` grouped-GEMM instantiation | metal/moe | fused ternary-expert MoE fwd (a0/rollout prefill) | tested |
| `qgemv_w2a8` v2 | metal/qgemv_int | decode GEMV, 1.25–4.1× over v1, up to 259 GB/s | tested + default |
| `qgemm_bwd` (bitnet) | metal/bitnet | ACADEMIC: the forbidden ternary backward — loses 0.5–0.9×, §0 confirmed by measurement. (This IS K3's spec — grad_x with in-kernel dequant — so K3 is closed.) | tested |
| `gemm_v3` | metal/matmul | ACADEMIC: 94–99% of MPS, doesn't beat it | tested |
| `bn_gemv_w2a8`, `bn_expert_ffn_w2a8`, `bn_moe_ffn_w2a8`, `bn_route_topk`, `bn_gemv_fp8`, `bn_attn_decode_kv8`, `bn_unpack_ternary_f32` + prefill driver | `bitnet_train/cpu/` | the K-track engine, scalar oracles + NEON; verdict: compute-bound on unpack (~7 GB/s/core), LUT decode is the named next dig | tested |

### 2026-07-06 gap-closing drop (perf: `metal/perf/gap_kernels_2026-07-06.md`, `cpu/perf/formats_bakeoff.md`)

| Kernel | Where | Serves | Status |
|---|---|---|---|
| `ternary_stats`, `code_flip_count` | metal/bitnet | §10.2/§6.2 step-0 monitors over packed wq (zero-code tail, flip rate) | tested, 134–259× vs PyTorch unpack; ~30 ms/full 14.5B sweep |
| `fake_quant_fp8` | metal/bitnet | mode b eval / Q-T4 cast delta (per-tensor e4m3) | tested BIT-EXACT vs torch.float8_e4m3fn (fast-math div + RNE ties both hand-fixed) |
| `kd_kl_dense_fwd/bwd` | metal/bitnet | A6b full-KL arm, fused (no (T,V) log-softmax materialization) | tested vs PyTorch KL + autograd; 3.16× at Tn=256, V=128K (crossover ~Tn 100) |
| `qgemm_w2a8_fused` (K2) | metal/bitnet | ACADEMIC: fused act-quant+GEMM — 0.16× at M=32, breakeven ~M=256, 1.17× at M=2048; the §3 profiling gate confirmed | tested |
| `attn_decode` | metal/bitnet | ACADEMIC: batch-1 GQA decode — SDPA wins 12–25×; §11.7 "SDPA suffices" confirmed by measurement | tested vs SDPA |
| `moe_grouped_gemm_bwd_dx/dw`, `moe_finalize_bwd`, `moe_gather_bwd` | metal/moe | complete fused-Metal MoE dense-STE backward over the padded schedule (dx ≈ dw ≈ fwd cost) | tested vs full autograd loop (dx, dW, router-weight grads) |
| `bn_rms_norm`, `bn_rope_neox`, `bn_kv_quant_append` | `bitnet_train/cpu/` | Q-K0 decode glue: norms/QK-norm, HF-convention RoPE, the int8 KV-cache WRITER (matches `bn_attn_decode_kv8`'s read side) | tested incl. write→attend roundtrip |
| `bn_gemv_q8` (Q8_0-shaped, sdot), `bn_gemv_bf16` | `bitnet_train/cpu/` | Q-A-head8 contenders (FP8 already existed) | tested, scalar+NEON |
| `bn_pack_b3`/`bn_gemv_b3` (base-3, 2.25 b/w) | `bitnet_train/cpu/` | bake-off min-bytes contender; scalar-only dead end at decode | tested + measured |
| `bn_pack_tl1`/`bn_gemv_tl1` (TL1-style LUT partial sums) | `bitnet_train/cpu/` | **the bake-off winner: 2.5–2.7× over the 2-bit NEON path (18–19.5 GB/s/core)** — unpack bound broken; next constraint is multicore dispatch | tested (scalar oracle + NEON, per-group & pt) + measured |

## 8. Build order

1. **Phase 0 (no new kernels):** pure-PyTorch `BitLinearSTE` (per-group-32 fake-quant) — the
   correctness oracle; train the 1B heal on MPS with plain `F.linear`.
2. **Phase 1 (K1 + composition):** `weight_quant_ternary` + the `BitLinearSTE` autograd.Function
   wiring reused kernels; validate parity + gradcheck vs Phase 0; measure speedup.
3. **Phase 2 (optional):** K2/K3 fusions, gated on measured need.
