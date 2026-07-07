# Gap-closing kernel drop (2026-07-06) — measured

Apple M4 Max, torch 2.14 MPS, `tests/test_metal_new_kernels.py` green before any number
below was taken. Companion to `bitnet_training_kernels.md`; CPU-side numbers live in
`cpu/perf/formats_bakeoff.md`.

## Ternary-health monitors — the real win of the drop

`ternary_stats` / `code_flip_count` over packed wq vs unpacking codes in PyTorch:

| pack (rows, K) | ternary_stats | code_flip_count | PyTorch unpack path | speedup |
|---|---|---|---|---|
| (2048, 2048) — 4M w | 15 µs | 17 µs | 1992 µs | 134× |
| (16384, 2048) — 33M w | 69 µs | 39 µs | 17823 µs | 259× |

Extrapolated to the Qwen3-15B expert pack (14.5B weights): **~30 ms per full sweep** —
the §10.2/§6.2 step-0 monitors (zero-code tail, flip rate, code fractions) are now
free enough to run at every eval without touching the latents.

## kd_kl_dense (A6b full-KL, fused) vs PyTorch dense KL — bf16 logits, V = 128,256

| Tn | fused fwd | PyTorch | speedup |
|---|---|---|---|
| 64 | 2536 µs | 1963 µs | 0.77× |
| 256 | 2728 µs | 8624 µs | **3.16×** |

Crossover ~Tn≈100; above it the fused kernel wins and, more importantly, never
materializes the two (T, V) fp32 log-softmax tensors (~131 MB each at Tn=256) — the
chunked-losses mandate's "fused equivalent". Use PyTorch below the crossover if VRAM
allows; the kernel above it.

## MoE backward grouped GEMMs — f32, T=4096, H=2048, I=768, E=128, k=8

| op | µs |
|---|---|
| forward `moe_grouped_gemm_rect` | 8858 |
| `moe_grouped_gemm_bwd_dx` | 9069 |
| `moe_grouped_gemm_bwd_dw` | 9730 |

Both backward products cost ≈ the forward GEMM (the expected ratio for same-FLOP
mma tiles); the fused-Metal MoE training path is now complete
(finalize_bwd/gather_bwd are µs-scale glue). Verified against a full PyTorch
autograd loop (dx, dW, and router-weight grads) in the test.

## K2 `qgemm_w2a8_fused` — the gate was right (built on request)

Fused per-token quant + W2A8 GEMM vs composed `quantize_per_token_int8` + `qgemm_w2a8`:

| (M, N, K) | fused | composed | fused/composed |
|---|---|---|---|
| (32, 2048, 2048) | 3856 µs | 612 µs | **0.16×** |
| (256, 2048, 2048) | 5552 µs | 4988 µs | 0.90× |
| (2048, 2048, 2048) | 34517 µs | 40523 µs | 1.17× |

The one-threadgroup-per-token shape re-reads the whole weight pack per token and only
amortizes at large M; +17% at M=2048 does not buy back the loss below. **Verdict:
ACADEMIC — keep routing through the composed path (and recall the training forward
doesn't use the integer path at all; measured routing in `bitnet_training_kernels.md`).**
docs/new-kernels.md §3's "build only after a measured baseline" gate is hereby
confirmed by measurement, matching the qgemm_bwd/K3 precedent.

## `attn_decode` — SDPA suffices, now with a number (built on request)

Batch-1 GQA decode (Hq=32, Hkv=8, D=64, bf16) vs MPS SDPA with repeat_kv:

| T | attn_decode | SDPA | ratio |
|---|---|---|---|
| 1024 | 822 µs | 63 µs | 0.08× |
| 4096 | 5300 µs | 191 µs | 0.04× |

One simdgroup per head with a serial T loop is 12–25× behind MPS's attention.
**Verdict: ACADEMIC — train_plan §11.7's "custom attention: never built (SDPA
suffices)" is confirmed by measurement.** A competitive version needs T-parallel
partials + a reduce pass (the paged_attention shape) — not worth building unless
rollout profiling ever says attention is the bottleneck, which this number makes
unlikely.

## fake_quant_fp8 — correctness note, no perf claim

Bit-exact against `torch.float8_e4m3fn` (CPU reference) for f32/bf16 inputs. Two
device pitfalls were load-bearing and are documented in the kernel: fast-math
division computes absmax/448 one ulp off (fixed with an fma Newton re-round), and
fast-math tie rounding is not RNE (fixed with an exact fma halfway comparison).
tk_e4m3_encode (round-half-away) was deliberately not used.

## Addendum (2026-07-07): TQ2_0 as a first-class in-repo format

llama.cpp's native ternary GGUF type, implemented in the VENDORED stack (no
external runtime dependency): a `tq2_0` dequant struct in the mittens format
framework (which lights up `qgemm`/`qgemm_frag`/`qgemv`/`qdequant` and both MoE
grouped-GEMM `_q` kernels via one-line instantiations) plus an on-device
`quantize_tq2_0` pack kernel. Numerics transcribed from ggml-quants.c
(reference tree, read 2026-07-07): per-256 ABSMAX half scale stored LAST
({qs[64]; half d}), element 128j+32n+m in qs[32j+m] bits 2n, lroundf codes.

Byte-exactness vs the ggml reference required the two now-familiar fast-math
fixes (see fake_quant_fp8): an fma-Newton re-round of 1/d, and hand-rolled
round-half-away (fast round sends exact ±0.5 ties — which bf16-grid inputs
hit — to zero). `tests/test_metal_tq2_0.py` asserts byte-exact packs vs the
numpy oracle, the §8.2 baked-ternary preserve regime end-to-end in-repo, and
qgemv/qgemm parity vs dense.

Measured (M4 Max, 2048×2048): quantize 93 µs, qgemv 49 µs (vs 33 µs for the
10B/32 bitnet format — TQ2_0's 66B/256 blocks dequant-scatter worse; fine for
eval, not a training-path kernel), qgemm M=64 via the qdequant route 234 µs,
full dequant 35 µs.
