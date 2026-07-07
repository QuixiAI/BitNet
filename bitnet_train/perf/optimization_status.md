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

### ternary_stats / code_flip_count — §10.2 monitors

134–259× vs PyTorch unpack (`gap_kernels_2026-07-06.md`); harness (smoke): 0.012
ms. A full 14.5B-param sweep ≈ 30 ms — free at every eval. No dig.

### kd_kl_dense — A6b full-KL fwd

3.16× over PyTorch dense KL at Tn=256, V=128K, crossover ~Tn≈100
(`gap_kernels_2026-07-06.md`). Never materializes the (T,V) log-softmax tensors.

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
