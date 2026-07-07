# BitNet kernel optimization — running notebook

The disciplined-loop log (`perf.md` is the guide). One section per kernel:
current status, best-known numbers with git label + device, the bottleneck
hypothesis, experiments tried (kept / rejected), and the next dig. Copy summary
tables here from `results/` (which is git-ignored); do not paste raw JSONL.

Device for all current numbers unless noted: Apple M4 Max, 40-core GPU, 128 GB,
macOS 26, torch 2.14 MPS. Seeded 2026-07-07 from the standing perf notes in
`../metal/perf/` and `../cpu/perf/` and a first `bench_kernels.py` run.

---

## CPU K-track engine (`--backend cpu`)

### gemv_w2a8 / gemv_tl1 — ternary decode GEMV

**Status: TL1 is the winner; unpack bound broken.** `formats_bakeoff.md`: TL1
LUT partial sums 2.5–2.7× over the 2-bit NEON path. First harness run
(smoke, 768×2048): scalar 0.314 ms, NEON-A 0.065 ms (7.6 GB/s), **TL1 0.028 ms
(17.4 GB/s)**. rel err vs f64 oracle 4e-8.

Bottleneck (pre-TL1, `first_cut.md`): compute-bound on 2-bit shift/mask unpack at
~7 GB/s/core against a >100 GB/s machine. TL1 replaces per-weight arithmetic with
a per-token LUT + table lookups. **Next dig:** multicore dispatch across the 8
active experts (conclusion 3) — the engine is now within reach of the bandwidth
ceiling per core, so aggregate scaling is the constraint, not decode.

### expert_ffn — fused whole-expert FFN

**Status: TL1 re-plumb landed, ~2× confirmed at FFN level.** Harness (smoke,
H2048 I768): A-format 0.174 ms (8.5 GB/s), **TL1 0.084 ms (17.7 GB/s)**, TL1
output byte-identical-within-1e-5 to the A path. This is the `formats_bakeoff.md`
follow-up delivered. Next: thread across experts; measure a full decode step
(`cpu/engine.py bench`) TTFT + tok/s on a real baked Qwen3-15B slice.

### gemv_fp8 — attention/head decode GEMV

**Status: unpack-bound, second-priority dig.** Harness (smoke, 768×2048): 0.307
ms (5.1 GB/s). Carries more decode bytes than the ternary path (`first_cut.md`).
**Next dig:** nibble-split `tbl` decode, or fp16 storage for the hot tensors.

### attn_decode_kv8 — int8-KV attention

Standing number (`first_cut.md`): T=4096 Hq32 Hkv8 D64, 1.97 ms/token
single-core (8.5 GB/s incl. 4× GQA re-read). Fine short-context; needs the same
unpack/parallelism treatment long-context. Not yet in the harness registry.

---

## Metal training kernels (`--backend torch`)

### weight_quant / quantize_tq2_0 — on-device packers

Harness (smoke, 768×2048): weight_quant_ternary_pt 0.074 ms (91.8 GB/s),
quantize_tq2_0 0.024 ms (276 GB/s). Both bandwidth-bound single-pass reads;
near roofline, no dig outstanding.

2026-07-07 A1-shape pass: earlier bytes accounting under-counted K1-pt (it
reads f32 TWICE — abssum pass + encode pass — and writes w_deq f32 + wq);
with honest bytes, a1 preset: 8192×2048 **439 GB/s**, 2048×8192 407 GB/s —
at roofline for a two-pass kernel. 2048×2048 196 GB/s is launch/occupancy
bound but costs 0.26 ms and amortizes over grad accum (weight-version cache).
No dig outstanding; the two reads are inherent to a per-tensor reduce.

### qgemv — ternary decode GEMV (bitnet / tq2_0)

Harness (smoke, 768×2048): bitnet 0.012 ms (264 W-GB/s), tq2_0 0.012 ms (255
W-GB/s). The TQ2_0 addendum (`gap_kernels_2026-07-06.md`) notes TQ2_0's 66 B/256
blocks decode-scatter worse than bitnet's 10 B/32; both fine for eval.

### qgemm — ternary prefill GEMM

The measured routing verdict (`bitnet_training_kernels.md`): the integer-exact
`qgemm_w2a8` has no tensor cores and loses to a dense GEMM on fake-quant operands
at every training batch size — the training forward uses `F.linear(x_q, w_deq)`.
qgemm(bitnet/tq2_0) is the FROZEN-inference / eval route. Harness measures the
dequant-to-half MMA path.

### fake_quant_int8 — K4 one-pass fake-quant

Harness (smoke, 768×2048): 0.028 ms (223 GB/s). Bandwidth-bound; the per-step
quant cache in `bitlinear_metal.py` removes the recompute across grad-accum.
A1 shapes (2026-07-07): 8192×2048 0.22 ms (310 GB/s), 2048×8192 0.25 ms
(267 GB/s) — small shapes launch-bound but cost ≤ 40 µs. No dig outstanding.

### ternary_stats / code_flip_count — §10.2 monitors

134–259× vs PyTorch unpack (`gap_kernels_2026-07-06.md`); harness (smoke): 0.012
ms. A full 14.5B-param sweep ≈ 30 ms — free at every eval. No dig.

### kd_kl_dense — A6b full-KL fwd

3.16× over PyTorch dense KL at Tn=256, V=128K, crossover ~Tn≈100
(`gap_kernels_2026-07-06.md`). Never materializes the (T,V) log-softmax tensors.
A1 shape (T1024, V128256): fwd 3.74 ms (281 GB/s), bwd 2.73 ms (288 GB/s).
Still the A6b ablation arm; the production heal loss is kd_ce_fused below.

### kd_ce_fused — fused CE + dense-KD (the heal loss) — KEPT 2026-07-07

**The 2026-07-07 optimization-pass win.** Baseline profiling at A1 shapes showed
the loss stack dominating per-step our-kernel time, with structural redundancy:
CE and KD both stream the same student logits, and their backwards emitted two
(T,V) grads that autograd then ADDED in another full pass. One kernel pair now
computes both losses (fwd: three online LSEs in one t+s read — student@1 for CE,
student@1/τ + teacher@1/τ for KD — plus the CE target gather; second read for
the KL sum) and one COMBINED grad (bwd: single pass, single bf16 store — also
one rounding instead of round-twice-then-add, so strictly more accurate).

Measured (a1 preset, T1024 V128256, vs separate cross_entropy + kd_kl_dense +
grad-add): fwd 3.76 ms vs 5.11 ms (**1.36×**), bwd 2.77 ms vs 6.42 ms
(**2.32×**); whole loss stack per chunk 11.5 → 6.5 ms (**1.77×**), ~740 → ~420
ms per T1 optimizer step (64 chunks). Both at ~280 GB/s — same bandwidth as the
parts, i.e. the win is pure traffic elimination. Wired as LossComputer's
fused+dense path (`_FusedCEKD`); _FusedCE/_FusedKDDense remain for kd_mode=none
/ ablations. Equivalence: fwd bit-identical to the separate kernels; suite
tests fused-vs-chunked incl. ignore_index rows.

**2026-07-07 second dig — three experiments, all KEPT** (a1 preset, T1024
V128256, M4 Max, tests 175-green after each):

1. **Single-pass fwd.** The KL pass-2 is eliminable: KL = (S1−S2)/L + lse_s −
   lse_t with S1 = Σ exp(zt−m)·zt, S2 = Σ exp(zt−m)·zs accumulated ONLINE in
   pass 1 (flash-attention-style rescale). Halves fwd traffic: fwd 3.76 →
   1.99 ms. Cancellation checked hostile (logits ×30, both τ): kd rel err
   ≤ 3e-6 — the feared (S1−S2) subtraction is benign because KL magnitude
   grows with spread.
2. **vec2 bwd.** Scalar bf16 lane loads = 64 B per simdgroup fetch (half a
   cache line); vec2 = 128 B. bwd 2.75 → 2.06 ms (**383 GB/s**). Scalar tail
   covers odd V; even-V rows are always vec2-aligned.
3. **vec2 fwd** (float2 online accumulators, pairwise merge before simd
   reduce): fwd 1.99 → **1.24 ms (424 GB/s)**.

Net vs the original separate CE+KD path: fwd **4.17×**, bwd **3.16×**; loss
stack per 1024-row chunk 11.5 → 3.3 ms (**3.5×**), ~740 → ~210 ms per T1
optimizer step. Both kernels now sit at the machine's practical streaming
roofline (383–424 of 546 GB/s theoretical) — remaining headroom ≤ 10%;
**no dig outstanding.** Correctness oracle: fp32 torch, even/odd/real V,
ignore_index rows, spreads to ±30 logits; suite covers fused-vs-chunked.

Possible future port (not done, low value): the 1-pass + vec2 tricks apply
verbatim to the ABLATION arms kd_kl_dense_fwd/bwd and cross_entropy_fwd/bwd;
port only if an A6b/A6c ablation or eval loop shows loss time in traces.

### attn_decode — ACADEMIC

SDPA wins 12–25× (`gap_kernels_2026-07-06.md`); §11.7 "SDPA suffices" confirmed.
In the registry for regression tracking only.

---

## Rejected / confirmed-negative (do not re-try without new evidence)

- **Integer W2A8 for the training forward** — loses to dense fake-quant GEMM at
  every M (no Apple int tensor cores). Kept as the inference route only.
- **K2 fused act-quant + GEMM** — 0.16× at M=32, breakeven ~M=256; the §3
  profiling gate confirmed. ACADEMIC.
- **ternary backward (qgemm_bwd / K3)** — 0.5–0.9× vs dense; §0 forbids it. Built
  once to measure; ACADEMIC.
- **base-3 packing** — min bytes but scalar-only dead end at decode (3× slower
  than NEON-A). Keep the packer, never route decode through it.
