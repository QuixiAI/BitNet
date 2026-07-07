# CPU engine (K-track) — first-cut kernels, roofline read

Measured 2026-07-06 · Apple M4 Max (arm64 NEON) · single core · clang -O3
Machine context: 256 MB numpy copy ≈ 135 GB/s r+w single-stream.

## The regime verdict (the number that matters)

The NEON ternary GEMV runs at **~7 GB/s of packed weight traffic on one core** against
a machine that streams >100 GB/s: **we are compute-bound on unpack, not
bandwidth-bound** — precisely the failure mode moe_train_plan §7.3's roofline
discipline exists to name, and the reason its packing bake-off lists LUT-indexed
partial sums (T-SAR-style) as a first-class contender. The next dig is unpack cost
(in-register LUTs / wider planes / int8 dotprod-instruction paths), not memory.

## Ternary GEMV (`bn_gemv_w2a8`), single core

| (N, K) | scalar | NEON | NEON speedup | GB/s (packed) |
|---|---|---|---|---|
| (768, 2048) gate/up | 358 µs | 78 µs (pt: 70) | 4.6× | 6.3–7.0 |
| (2048, 768) down | 350 µs | 82 µs (pt: 70) | 4.3× | 6.0–7.0 |
| (2048, 2048) | 953 µs | 212 µs (pt: 181) | 4.5× | 6.2–7.2 |

The per-tensor-scale path (deferred single multiply — the §3.7 baseline's reward)
is ~10–15% faster than per-group.

## Fused expert FFN (`bn_expert_ffn_w2a8`), H=2048 I=768 (one Qwen3-15B-A2B expert)

187 µs/expert (7.9 GB/s) → 8 experts × 24 layers = **35.9 ms/token single-core** for
the expert path. Perfectly parallelized across ~12 P-cores that is ~3 ms/token →
the expert share of the §7.4 byte budget supports the "tens of tok/s" target ONLY
after (a) multicore dispatch across experts and (b) an unpack that beats 7 GB/s/core.
Neither is exotic; both are now quantified prerequisites rather than hopes.

## fp8 e4m3 GEMV (attention / lm_head — 0.75 of the ~1 GB/token budget)

| (N, K) | scalar | NEON |
|---|---|---|
| (2048, 2048) | 3535 µs (1.2 GB/s) | 847 µs (4.9 GB/s) |
| (16384, 2048) | 28.0 ms | 6.75 ms (5.0 GB/s) |

Also unpack-bound (scalar LUT gathers feeding vfma). This kernel carries MORE decode
bytes than the ternary one; a nibble-split `tbl` decode or fp16 storage for the hot
tensors are the candidate fixes. At 5 GB/s/core the fp8 share alone would cap decode
well below target — this is the K-track's second-priority dig after ternary unpack.

## int8-KV attention decode

T=4096, Hq=32, Hkv=8, D=64: 1.97 ms/token single-core (8.5 GB/s effective incl. the
4× GQA re-read). Fine at short context, needs the same unpack/parallelism treatment
at long context.

## Prefill: dequant-once + BLAS (`bn_prefill_ternary`) vs per-token GEMV

| M | dequant+BLAS | GEMV × M | amortization |
|---|---|---|---|
| 32 | 0.40 ms | 2.34 ms | 5.9× |
| 128 | 0.53 ms | 8.78 ms | 16.6× |
| 512 | 0.96 ms | 35.5 ms | 37× |

Validates §7.4's separate-prefill-mode mandate: unpack-once amortizes almost
immediately; never run decode kernels over a prompt.

## Standing conclusions for Q-K0

1. Scalar oracles exist for every kernel and NEON is diffed against them in CI.
2. The engine is unpack-compute-bound; the packing bake-off should be run with an
   unpack-cost-first mindset (2-bit shift/mask measured here; LUT next; base-3 last).
3. Threading across experts is mandatory to reach the target; add the dispatch layer
   before drawing tok/s conclusions.
4. The fp8 dense GEMVs are on the critical decode path and currently slower per byte
   than the ternary path — budget them equal attention.
