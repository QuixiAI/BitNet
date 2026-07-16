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

### gemv_tq1 — schema-2 codebook TQ1 decode GEMV

**2026-07-15 focused pass — grouped ARM dot-product KEPT; subset-LUT REJECTED.**
The initial scalar path was index-unpack/codebook-dot compute bound. The kept
variant decodes four adjacent 8-weight indices together and uses two signed
ARM `sdot` operations per 32 weights, while retaining the exact scalar loop as
the oracle and the canonical 44/48-byte payload as the weight source. V11/V12
expanded codebooks occupy 18,432/36,864 resident bytes including the legal-index
bitmap; canonical payload and codebook remain resident.

Apple M4 Max, macOS 26.5.2 (25F84), Apple clang 21.0.0, Python 3.12.9,
PyTorch 2.14.0.dev20260706, `6f5c3a6-dirty`. Command:
`.venv/bin/python bitnet_train/perf/bench_kernels.py --backend cpu --preset a1
--kernel gemv_tq1 --warmup 10 --iters 50`. Medians and p20/p80 are per call:

| profile / N×K | grouped sdot ms | p20/p80 ms | scalar ms | speedup | dense dequant+BLAS ms | canonical W-GB/s |
|---|---:|---:|---:|---:|---:|---:|
| V11 / 2048×2048 | 0.4040 | 0.3995/0.4146 | 1.2354 | 3.06× | 0.0648 | 1.8 |
| V11 / 512×2048 | 0.1136 | 0.1121/0.1163 | 0.3213 | 2.83× | 0.0144 | 1.6 |
| V11 / 8192×2048 | 1.5939 | 1.5669/1.6157 | 4.9138 | 3.08× | 0.2855 | 1.8 |
| V11 / 2048×8192 | 1.5621 | 1.5280/1.5992 | 4.9564 | 3.17× | 0.3011 | 1.8 |
| V12 / 2048×2048 | 0.4080 | 0.4031/0.4202 | 1.2621 | 3.09× | 0.0740 | 1.9 |
| V12 / 512×2048 | 0.1159 | 0.1138/0.1184 | 0.3219 | 2.78× | 0.0148 | 1.7 |
| V12 / 8192×2048 | 1.5878 | 1.5681/1.6234 | 4.9147 | 3.10× | 0.2949 | 2.0 |
| V12 / 2048×8192 | 1.5609 | 1.5346/1.5926 | 4.8718 | 3.12× | 0.2989 | 2.0 |

Correctness: `pytest tests/test_tq1_cpu_native.py tests/test_cpu_engine.py -q`
is 15/15 green. It covers J/P codebooks, V11/V12, row/block FP16 scales,
BF16 row scales, token/block A8, A4, zero/ragged rows, and corrupt reserved
indices. FP32 tolerance is `atol=1e-6, rtol=1e-6`; across the A1 benchmark the
observed maximum absolute/relative errors were 1.48e-6/8.42e-8 (combined
tolerance passes; integer accumulators are exact and scalar/optimized outputs
are bit-identical in the focused unit cases).

The subset-sum experiment built a 256-entry activation LUT per 8-value group.
At 768×2048 it regressed V11/V12 from expanded-codebook NEON
0.233/0.234 ms to 0.408/0.413 ms, so it was removed. Dense dequantized BLAS is
still 5.3–8.0× faster when a full dense matrix is already resident; no claim is
made that this first packed kernel beats that memory-expansive baseline. Next
dig: tile output rows and hoist packed high-bit extraction/codebook gathers.

**2026-07-15 small-batch/prefill follow-up — native batch dispatch KEPT.** The
hypothesis was launch-bound execution for multiple tokens: one C entry point now
loops over the same correctness-checked grouped-dot kernel instead of paying one
Python/ctypes call per token. Command: `.venv/bin/python
bitnet_train/perf/bench_kernels.py --backend cpu --preset a1 --kernel gemm_tq1
--warmup 10 --iters 50`, on the same device/toolchain and git label above. The
shape matrix was every Llama-3.2-1B projection dimension at M=4 and M=32, V11
and V12 J-R, A8-token, FP16 row scale.

Native batch dispatch is 1.00–1.19× the repeated-GEMV integration baseline at
M=4 (the 0.9955× result is inside observed dispersion) and 1.01–1.17× at M=32.
V11/V12 medians span 0.410/0.412 ms at M4 512×2048 through 49.07/49.25 ms at
M32 2048×8192. p20/p80 is recorded in the raw run; all but one target CV are
<=0.061, and the one OS-outlier case still has tight p20/p80
(0.402/0.423 ms around a 0.410 ms median). Correctness is 25/25 focused tests;
all seven format-v1 profile families are exercised by the batch API. FP32
`atol=1e-6, rtol=1e-6` passes, with observed benchmark maxima 1.73e-6 absolute
and 7.82e-8 relative; batched and repeated optimized outputs are bit-identical.

Decision: keep the native batch entry point as the small-batch correctness and
dispatch path. It is not a competitive prefill GEMM: dense dequantized BLAS is
3.36–58.45× faster when the expanded dense matrix is resident. A future prefill
candidate must tile/reuse decoded weights across M instead of looping GEMV.

**2026-07-15 QI-5 cached-repack routing pass — budgeted dense prefill KEPT;
eager/unbudgeted and sub-128 first-use routing REJECTED.** The hypothesis was
that repeated packed GEMV is decode-appropriate but fails to reuse decoded
weights across prompt tokens. The candidate lazily creates the exact,
deterministically hashed `tq1_dense_f32_row_major_v1` private layout, quantizes
activations with the same A8-token contract, and uses Accelerate-backed F32
BLAS. Canonical payload and expanded-codebook state remain resident. A zero-byte
policy budget keeps the route disabled; an enabled route must account for the
entire dense tensor.

Device/toolchain: Apple M4 Max (16 CPU cores, 128 GB), macOS 26.5.2 build
25F84, LLVM clang 22.1.0, Python 3.12.9, PyTorch
2.14.0.dev20260706, 12 Torch threads, git label `fed90f2-dirty`. Command:

```bash
.venv/bin/python bitnet_train/perf/bench_tq1_runtime_routes.py \
  --warmups 10 --iterations 30 --include-head
```

Coverage is V11-J-R and V12-J-R with FP16 row scales and A8-token activations;
all four distinct Llama-3.2-1B projection shapes, ragged `N=513`, M=32/64/128,
plus the real tied output head `M=1,N=128256,K=2048`. The scalar packed oracle
is retained. Packed-native output was bit-identical to it. The dense candidate
passes combined FP32 `atol=2e-5, rtol=2e-5`; observed maximum absolute/relative
errors were `1.0872e-4 / 3.4973e-6` (the absolute maximum occurs on
`K=8192` and passes the combined bound). The complete 31-row run, hashes,
repack time/bytes, CV, and raw p20/p80 are in
`perf/results/2026-07-15/tq1-runtime-routes/run.json` (git-ignored).

The following is the conservative first-use decision set at M=128. “Hot” omits
the separately reported one-time repack; “first-use” includes repack plus the
hot median. Times are p20/median/p80 milliseconds:

| profile / N×K | packed M128 ms | dense hot M128 ms | hot speedup | repack ms / MiB | first-use speedup |
|---|---:|---:|---:|---:|---:|
| V11 / 512×2048 | 12.673/12.813/12.898 | 0.754/0.774/0.782 | 16.54× | 4.67 / 4.0 | 2.35× |
| V11 / 2048×2048 | 49.270/49.483/49.635 | 1.113/1.140/1.159 | 43.42× | 10.16 / 16.0 | 4.38× |
| V11 / 8192×2048 | 194.105/194.720/195.112 | 2.234/2.253/2.280 | 86.43× | 37.29 / 64.0 | 4.92× |
| V11 / 2048×8192 | 189.147/189.777/190.675 | 2.629/2.669/2.797 | 71.11× | 39.12 / 64.0 | 4.54× |
| V11 / 513×2048 | 12.796/12.897/12.978 | 0.883/0.893/0.907 | 14.44× | 4.70 / 4.0 | 2.31× |
| V12 / 512×2048 | 12.881/12.934/13.044 | 0.879/0.892/0.914 | 14.50× | 6.32 / 4.0 | 1.79× |
| V12 / 2048×2048 | 49.187/49.308/49.471 | 1.255/1.278/1.292 | 38.58× | 11.79 / 16.0 | 3.77× |
| V12 / 8192×2048 | 195.200/196.010/196.590 | 2.333/2.356/2.371 | 83.20× | 40.76 / 64.0 | 4.55× |
| V12 / 2048×8192 | 189.876/190.678/190.997 | 2.632/2.680/2.865 | 71.16× | 40.48 / 64.0 | 4.42× |
| V12 / 513×2048 | 12.901/12.945/12.999 | 0.884/0.897/0.912 | 14.44× | 6.38 / 4.0 | 1.78× |

M=32 first-use speedups ranged from 0.51–1.20× and M=64 from 0.93–2.48×,
so neither threshold is safe across supported edge shapes. M=128 was
1.78–4.92× across every profile/shape, including `N=513`; keep the default
opt-in short-prefill threshold at 128. The long-prefill threshold is separately
named (512 by default) but currently selects the same cached-BLAS implementation.
The widest candidate had a few OS outliers (CV 0.354) while its p20/p80 remained
tight around the median; no conclusion uses mean timing.

The tied output-head hot path was 4.3307/4.3818/4.4075 ms versus packed native
23.8957/24.0240/24.0936 ms, a 5.48× hot speedup with `1.91e-6 / 1.25e-7`
maximum absolute/relative error. Its one-time repack is 550.3 ms and 1,050,673,152
bytes (1002 MiB), so first-token latency is a 0.043× regression; the measured
kernel break-even is about 29 decode tokens. Keep this only behind the explicit
`output_head_dense_decode` and byte-budget policy. Reject enabling it by default
until full-model `tg128`, TTFT, resident/peak memory, thermal, and joules/token
evidence passes the separate QI-5 report contract. This is a kernel-route win,
not yet a model-throughput or energy claim.

The QI-2 embedding-gather coverage follow-up used the same device/toolchain and
real V12-J-R vocabulary matrix:

```bash
.venv/bin/python bitnet_train/perf/bench_tq1_runtime_routes.py \
  --head-only --warmups 10 --iterations 30 \
  --output bitnet_train/perf/results/2026-07-15/tq1-embedding-head-routes/run.json
```

Packed lookup matched the unique-row scalar oracle exactly (`atol=rtol=0`) for
32 repeated copies of one token, a 128-token/16-unique prompt, a
512-token/256-unique prompt, and ragged `3x17` IDs including vocabulary row
128255. Their p20/median/p80 times were respectively
3.4231/3.4937/3.5699, 3.5774/3.6291/3.6654,
4.3920/4.4440/4.4725, and 3.7185/3.7731/3.8322 ms. Keep the packed
unique-row gather and the new harness coverage; no embedding speedup is claimed
because this pass establishes correctness/shape behavior rather than an
alternative gather kernel. In the same rerun the optional dense head remained a
5.44x hot win (23.6384 vs 4.3449 ms median) with 1.91e-6/1.31e-7 maximum
absolute/relative error, a 546.1 ms repack, and the unchanged 1,050,673,152-byte
residency cost.

**2026-07-15 pinned llama.cpp scalar integration — KEPT as the permanent
reference; speed claim REJECTED.** The hypothesis was that direct packed decode
would save weight traffic but remain codebook-gather/loop bound, especially for
prefill. The revision-locked patch
quant/llama_cpp/patches/a582222-tq1-v.patch was built at llama.cpp
a5822222909b785f23ddc74ce3c8f85bd0e38562 with Metal, Accelerate, BLAS, and
OpenMP disabled so the affected CPU path and decoded-F32 baseline were measured
directly. Device/toolchain: Apple M4 Max, macOS 26.5.2 (25F84), Apple clang
21.0.0, CMake 4.0.3, 8 threads. The focused command, 3 warmups, all 15 samples,
and 40 JSON rows are in
bitnet_train/perf/results/tq1_llama_cpp_scalar_20260715/run.jsonl.

Coverage was all five physical types (V11/V12 block scale, V11/V12 row scale,
and V11 A4 row scale), every distinct Llama-3.2-1B projection shape
(N×K 512×2048, 2048×2048, 8192×2048, 2048×8192), and M=1/M=32. The baseline
is deliberately optimistic: decoded F32 weights multiplied by already
dequantized A8 inputs, so its timing excludes activation quantization and
assumes the dense matrix is resident. Representative V12-J-R results:

| N×K / M | packed scalar ms p20/median/p80 | decoded-F32 ms p20/median/p80 | dense / packed |
|---|---:|---:|---:|
| 512×2048 / 1 | 0.1318 / 0.1359 / 0.1402 | 0.1022 / 0.1041 / 0.1082 | 0.77× |
| 2048×2048 / 1 | 0.2440 / 0.2489 / 0.2596 | 0.1326 / 0.1402 / 0.1451 | 0.56× |
| 8192×2048 / 1 | 0.6880 / 0.6958 / 0.7029 | 0.3164 / 0.3256 / 0.3446 | 0.47× |
| 2048×8192 / 1 | 0.6643 / 0.6669 / 0.6862 | 0.3197 / 0.3292 / 0.3347 | 0.49× |
| 512×2048 / 32 | 1.2424 / 1.2515 / 1.2620 | 0.3273 / 0.3323 / 0.3458 | 0.27× |
| 2048×2048 / 32 | 4.8462 / 5.0620 / 6.3546 | 1.0601 / 1.0701 / 1.0891 | 0.21× |
| 8192×2048 / 32 | 19.0495 / 19.1002 / 19.2530 | 3.9376 / 3.9489 / 3.9748 | 0.21× |
| 2048×8192 / 32 | 17.9149 / 17.9579 / 18.0795 | 4.1760 / 4.2268 / 4.2678 | 0.24× |

All 40 rows match an independent packed integer/scalar-order oracle with zero
observed absolute and relative error (gate: atol=1e-6, rtol=1e-6). The
decoded-F32 comparison differs only by floating accumulation order; maximum
absolute difference is 6.11e-5. The pass exposed and fixed an A8 parity bug in
both the repository-native and patched paths: multiplying by a rounded
reciprocal can choose a different tie than the specified round_to_even(x/a),
so both now divide by the stored scale and use explicit half-even rounding.
After measurement, the integration patch added FP16/BF16 row-scale casts in
the Llama FFN graph to complete model loading/prefill correctness. That change
does not affect the low-level kernel or benchmark graph; both the measured and
final patch hashes are preserved in the raw run metadata.

Across every profile/shape, the packed scalar path is 1.21–2.23× slower for
decode and 3.60–5.03× slower for M=32 than the optimistic dense baseline.
Decision: retain it as the fail-closed llama.cpp loader/scalar oracle and make
no speed claim. Production CPU performance remains the grouped-dot K-track;
any llama.cpp speed candidate must add tiled/SIMD codebook decode and rerun this
matrix.

**2026-07-15 CPU A8 exact-rounding follow-up — builtin divide variant KEPT;
manual variant REJECTED.** The correctness hypothesis was that the historical
`rintf(x * (1 / scale))` can cross a half-even boundary because the FP32
reciprocal is rounded before multiplication. At the exact fixture
`[1.6625983, -1.0669429, 0]`, the old `6f5c3a6` library emits
`[127, -81, 0]`; the specified FP32 `round_to_even(x / scale)` and the kept
library emit `[127, -82, 0]`. Candidate maximum integer-code error is zero; the
old path's observed targeted maximum is one. The focused native CPU suite is
53/53 green, including all TQ1 physical profiles and A8 token/block modes.

The first explicit `floorf`/`fmodf` implementation fixed correctness but was
1.15–1.74× slower and was rejected. The kept implementation divides by the
stored scale and uses Clang's rounding-mode-independent, vectorizable
`__builtin_roundevenf`, with a portable scalar fallback. On Apple M4 Max,
macOS 26.5.2, Apple clang 21.0.0, Python 3.12.9, it is 1.010×, 1.024×, and
1.037× the old path at K=512/2048/8192. Candidate p20/median/p80 times are
0.002511/0.002531/0.002567 ms, 0.003613/0.003650/0.003672 ms, and
0.007984/0.008042/0.008161 ms; CV is 0.018/0.024/0.020. The A/B used 100
warmups, 31 alternating samples, and at least two million elements per sample.
The command is:

```bash
git worktree add --detach /tmp/bitnet-a8-baseline.20260715 6f5c3a6
sh /tmp/bitnet-a8-baseline.20260715/bitnet_train/cpu/build.sh
sh bitnet_train/cpu/build.sh
.venv/bin/python bitnet_train/perf/bench_a8_rounding.py \
  --baseline-lib /tmp/bitnet-a8-baseline.20260715/bitnet_train/cpu/libbitnet_cpu.dylib \
  --candidate-lib bitnet_train/cpu/libbitnet_cpu.dylib \
  --baseline-label reciprocal_rint_head \
  --candidate-label divide_roundeven_kept \
  --baseline-revision 6f5c3a6 \
  --output bitnet_train/perf/results/a8_round_even_20260715/run.jsonl
```

Decision: keep the exact builtin path. The small measured improvement is a
compiler-vectorization result, not a broader decode/prefill speed claim.

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

**2026-07-07 third pass — sibling act-quant sharing (routing): KEPT, for
MEMORY not speed.** q/k/v receive the same layernorm output and gate/up the
same block input, so per Llama layer 7 K4 quants covered only 4 distinct
inputs. A `WeakTensorKeyDictionary` memo keyed on the module-input tensor
(guarded by `_version`; weak keys die with the activations) now runs K4 once
per distinct input and shares one x_q across siblings' saved-for-backward.

Measured (2-layer real-shape converted slice, micro-batch 8×1024, fwd+bwd):
- Throughput: **no measurable win** — 676.8 vs 677.1 ms/micro-step, inside
  the 3 ms run-to-run drift. The ~0.65 ms/layer of eliminated kernel time is
  invisible in a GEMM-dominated step. Do NOT cite this as a speedup.
- Memory: **302 MB less fwd-graph allocation on 2 layers** (7259 vs 7561 MB,
  stable across reruns) ≈ **2.4 GB across 16 layers** per micro-batch graph
  — real headroom for micro-batch size / avoiding gradient checkpointing.
- Semantics identical by construction (same tensor → same quant); suite
  covers shared-vs-distinct outputs+grads and the mutation guard (176 green).

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
