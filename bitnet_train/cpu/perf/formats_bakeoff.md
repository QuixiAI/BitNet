# Packing-format bake-off (moe_train_plan §7.3) — measured

Measured 2026-07-06 · Apple M4 Max (arm64 NEON) · single core · clang -O3 ·
`perf/bench_formats.py`. Follow-up to `first_cut.md`, which named the ~7 GB/s/core
unpack bound and LUT decode as the next dig. Three contenders, per the plan's list:

- **A — 2-bit shift/mask** (incumbent): 10 B/32 = 2.5 b/w. Trivial unpack.
- **B — base-3 dense** (`bn_pack_b3`/`bn_gemv_b3`): 9 B/32 = 2.25 b/w, min bytes;
  decode via a 256×5 byte→trits LUT. Scalar only.
- **C — TL1-style LUT partial sums** (`bn_pack_tl1`/`bn_gemv_tl1`): 2.5 b/w, 4-bit
  pair indices in 16-row tiles; per token a 9-entry int16 LUT per k-pair
  (lo/hi split for `tbl`), decode = table lookup + add, no per-weight arithmetic.

## Ternary GEMV, single core (per-group scales unless `pt`)

| (N, K) | A scalar | A NEON | A NEON pt | B scalar | C scalar | C NEON | C NEON pt |
|---|---|---|---|---|---|---|---|
| (768, 2048) | 362 µs | 81.9 µs (6.0 GB/s) | 70.1 (7.0) | 105 (4.2) | 453 | **31.8 (15.4)** | **29.6 (16.6)** |
| (2048, 768) | 345 µs | 81.1 µs (6.1) | 70.6 (7.0) | 105 (4.2) | 457 | **30.9 (15.9)** | **28.5 (17.3)** |
| (2048, 2048) | 956 µs | 204 µs (6.4) | 179.8 (7.3) | 269 (4.4) | 1200 | **72.9 (18.0)** | **67.3 (19.5)** |

GB/s = packed weight bytes / time (the roofline quantity; B moves 10% fewer bytes
per weight, so compare µs, not GB/s, across formats).

## Verdict

1. **Format C (TL1 LUT) wins decisively: 2.5–2.7× over the incumbent NEON path**
   (2048×2048 decode GEMV 204 → 73 µs). The LUT-build cost (K/2 × 9 int16 entries
   per token) is amortized inside these numbers. This confirms `first_cut.md`'s
   diagnosis: the engine was unpack-compute-bound, and removing per-weight
   shift/mask/mul arithmetic recovers most of it.
2. At ~18–19 GB/s/core, ~8–10 P-cores would aggregate to ~150+ GB/s — the kernel is
   now **within reach of the machine's bandwidth ceiling**; the next constraint is
   the multicore dispatch layer, no longer unpack.
3. **Format B (base-3) is the min-bytes contender but a scalar-only dead end at
   decode**: 3.4× faster than scalar-A (byte→5-trit LUT beats shift/mask) yet 3×
   slower than NEON-A. Its 2.25 b/w only pays where bytes, not cycles, dominate
   (archival/network); keep the packer, don't route decode through it.
4. The per-tensor deferred-scale reward persists in format C (~6–8%).
5. Expert-FFN projection: gate+up+down per expert ≈ (2×768×2048 + 2048×768) at C-NEON
   ≈ ~92 µs vs 187 µs measured for the fused A-path FFN — re-plumbing
   `bn_expert_ffn_w2a8` onto format C is the follow-up with a ~2× decode payoff.

Scalar C exists as the permanent oracle (`bn_gemv_tl1_scalar`, arithmetic decode, no
LUT); NEON is diffed against it in `tests/test_cpu_kernels.py` (per-group and pt).
