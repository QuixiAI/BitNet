# Perf note — BitNet training kernels (K1 + BitLinearSTE routing)

Measured 2026-07-06 · Apple M4 Max · torch 2.14.0.dev20260706 (MPS) · macOS 25.5.0
Method: 20–50 iters after warmup, `torch.mps.synchronize()` around the timed loop.
Shapes are the A1 (Llama-3.2-1B) BitLinear projections; M = 2048 tokens.

## K1 `weight_quant_ternary` (fp32 latent → packed blocks + bf16 dequant, group 32)

| (N, K) | K1 | GB/s | pure-PyTorch fake-quant | speedup |
|---|---|---|---|---|
| (2048, 2048) | 272 µs | 97 | 708 µs | 2.6× |
| (512, 2048) | 37 µs | 180 | 176 µs | 4.8× |
| (8192, 2048) | 340 µs | 312 | 2834 µs | 8.3× |
| (2048, 8192) | 379 µs | 280 | 2819 µs | 7.4× |

## Forward-route decision (why BitLinearSTE does NOT use `qgemm_w2a8` for training)

`qgemm_w2a8` is the integer-exact path — one simdgroup per output row, no tensor
cores (qgemm_int.metal's own header calls it the slower, bit-exact route). Measured
at (N,K)=(2048,2048), µs, `qgemm` column includes the (K,M) transpose copy it needs:

| M | qgemm_w2a8 (int) | qgemm mma "bitnet" | F.linear(x_q, w_deq) |
|---|---|---|---|
| 1 | 96 | — | 31 (but qgemv_w2a8: 16) |
| 32 | 761 | 107 | 52 |
| 128 | 2126 | 153 | 117 |
| 512 | 8693 | 389 | 361 |
| 2048 | 36616 | 1398 | 1266 |

**Routes chosen:** training forward = `F.linear(x_q, w_deq)` on the kernel-produced
fake-quant operands (fastest at every M, numerically the fake-quant semantics);
frozen decode (M=1) = `qgemv_w2a8` (integer-exact, 2.2–3.8× over dense); frozen
prefill = `qgemm(wq, x_qᵀ, "bitnet")` (keeps only packed weights resident).
Backward = `torch.matmul` (measured equal-or-better than `matmul_custom` at all
three A1 backward shapes, and it consumes the transposed view without a copy).

## BitLinearSTE (metal backend) vs pure-PyTorch reference, M = 2048, fp32 latents

| (N, K) | fwd+bwd metal | fwd+bwd ref | × | fwd metal | fwd ref | × |
|---|---|---|---|---|---|---|
| (2048, 2048) | 4.25 ms | 5.54 ms | 1.30 | 1.73 ms | 2.87 ms | 1.66 |
| (512, 2048) | 1.45 ms | 1.97 ms | 1.36 | 0.74 ms | 1.22 ms | 1.66 |
| (8192, 2048) | 15.3 ms | 20.2 ms | 1.33 | 5.6 ms | 9.7 ms | 1.71 |
| (2048, 8192) | 16.1 ms | 22.8 ms | 1.42 | 6.5 ms | 12.5 ms | 1.91 |

## Decode GEMV, M = 1, frozen packed weights

| (N, K) | qgemv_w2a8 | dense x@Wᵀ |
|---|---|---|
| (2048, 2048) | 15.5 µs | 34.9 µs |
| (512, 2048) | 9.7 µs | 19.3 µs |
| (8192, 2048) | 38.2 µs | 146.8 µs |
| (2048, 8192) | 39.1 µs | 146.5 µs |

## 2026-07-06 kernel drop (same machine/method)

**qgemv_w2a8 v2** (one 10-byte block/lane, arithmetic 2-bit→int8 spread + idot4, one
scale multiply per 32 codes) — new default; v1 kept for comparison:

| (N, K) | v1 | v2 | speedup |
|---|---|---|---|
| (2048, 2048) | 58.5 µs (22 GB/s) | 14.1 µs (93 GB/s) | 4.1× |
| (512, 2048) | 10.0 µs (33) | 8.0 µs (41) | 1.25× |
| (8192, 2048) | 57.7 µs (91) | 33.6 µs (156) | 1.7× |
| (2048, 8192) | 47.4 µs (111) | 28.8 µs (182) | 1.6× |
| (152064, 2048) lm-head | 596 µs (163) | 376 µs (259) | 1.6× |

Remaining headroom to ~546 GB/s is launch-latency at small shapes and reduce overhead
at large; split-K is the named next experiment if decode profiling demands it.

**K4 fused fake-quant + per-step weight-quant cache**: the eager x_q chain
(343 µs) and per-forward K1 re-runs are gone. BitLinear metal forward at
M=2048: (2048,2048) 1727 → **1437 µs** steady-state; (8192,2048) → 5125 µs —
non-GEMM overhead is now ≈ the single fused act-quant pass. With grad-accum 8 the
cache also cuts K1 invocations ~8–16× per optimizer step.

**Academic results (built to measure the arguments, and they measured):**

- `qgemm_bwd` (ternary-operand backward GEMM, the kernel train_plan §4 forbids):
  correct (rel ~1.5e-3), and **loses everywhere** — 0.89–0.90× vs `torch.matmul` on
  w_deq at training shapes, 0.51× at M=32. The dense-backward doctrine now has a
  number attached.
- `gemm_v3` (2×2-warp 64×64 tile, both operands staged): 13.8–14.2 TFLOP/s bf16 =
  **94–99% of MPS**, edging `gemm_staged` and tying `matmul_custom`, beating neither
  MPS nor the doctrine that hand GEMMs don't out-run Apple's. 70 lines gets within
  1–6%; the last 6% stays bought.

**MoE `bitnet` grouped GEMMs** (`moe_grouped_gemm_rect_q/swiglu_q` bitnet
instantiation + full route→permute→pad→gather→finalize pipeline): correctness-tested
vs the PyTorch reference (uneven routing incl.); not yet perf-profiled — first
customer is OPD rollout prefill, bench when that exists.

New correctness-only kernels this drop (no perf claims): `weight_quant_ternary_pt`
(per-tensor two-pass; the moe_train_plan §3.7 baseline the old K1 could not produce),
batched (E,N,K) quantization, `adamw_masked` (cold-expert decay mask, §4.3),
`kd_kl_topk_fwd/bwd` (sparse-teacher KD-KL, both tail policies), `fake_quant_int8` /
`silu_mul_fake_quant_int8`, `rms_norm_dyn` (D=2048/3072+ widths).

## K2/K3 outlook (gated on this baseline)

The fwd+bwd win is 1.3–1.4×; the remaining forward gap to pure GEMM time is the
quantize dispatches + x_q materialization — K2 (fused act-quant + GEMM) attacks
that. K3 (fused dequant in grad_x) saves the w_deq write (~3% of K1 traffic) and a
GEMM read. Neither is required for correctness; revisit after profiling a real
A1 training step.
