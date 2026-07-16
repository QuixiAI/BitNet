# Quantization and Healing Improvement Plan

## 1. Purpose

This document turns the reusable findings from the Bonsai reference release into
an implementation plan for the Llama-3.2-1B TQ1 quantization and healing stack.
It covers training, PTQ/QAT, artifact accounting, evaluation, export, KV-cache
quantization, and inference kernels.

This plan is subordinate to the contracts in [`quant/quant_spec.md`](../quant/quant_spec.md)
and the training doctrine in [`docs/train_plan.md`](train_plan.md). Changes to a
public format or training contract must update those documents as part of the
same implementation milestone.

The primary references are:

- [1-bit Bonsai 8B whitepaper](../.reference/Bonsai-demo/1-bit-bonsai-8b-whitepaper.pdf),
  especially pages 5–7 and 14–16;
- [Ternary Bonsai 8B whitepaper](../.reference/Bonsai-demo/ternary-bonsai-8b-whitepaper.pdf),
  especially pages 4–5;
- [Bonsai 27B whitepaper](../.reference/Bonsai-demo/bonsai-27b-whitepaper.pdf),
  especially pages 6–15 and 17–18;
- [Bonsai KV-cache notes](../.reference/Bonsai-demo/KV-CACHE.md);
- [Bonsai speculative-decoding notes](../.reference/Bonsai-demo/SPECULATIVE.md).

## 2. Evidence boundary

The Bonsai papers disclose useful system choices, but not the core model
conversion recipe. The 1-bit paper describes the underlying method as
proprietary, and none of the papers specifies the conversion objective, optimizer,
learning-rate schedule, training-token count, calibration algorithm, teacher,
loss weights, or ablations needed to reproduce its model quality.

Consequently:

- use the disclosed format, coverage, evaluation, memory-accounting, KV, and
  runtime ideas as engineering hypotheses;
- do not claim that the current TQ1 recipe reproduces Bonsai training;
- do not treat Bonsai quality or speed numbers as acceptance evidence for this
  repository;
- reproduce every quality and performance conclusion on the actual
  Llama-3.2-1B model, our formats, and each supported backend.

The 27B model is predominantly linear attention, while Llama-3.2-1B uses a
standard full-attention stack. Its context-memory and throughput numbers do not
transfer directly. The 27B evaluation also uses different recommended sampling
temperatures for the dense and Bonsai models, so its scores are useful for
identifying failure categories, not as controlled representation-only deltas.

## 3. Current blockers and baseline facts

### 3.1 The canonical QAT run cannot freeze

The checked-in configuration uses:

```text
seq_len        = 4096
batch_size     = 1
grad_accum     = 32
total_tokens   = 200,000,000
```

On one process this is 131,072 global tokens per optimizer step, or approximately
1,526 optimizer steps. The TQ1 profile specifies 1,000 soft steps, 4,000 hard
steps, an earliest freeze at step 5,000, and a maximum at step 9,000. The run
therefore cannot enter the frozen/export-qualified phase. Adding processes
increases tokens per optimizer step and reduces the number of steps further.

This must be fixed before another production-sized healing run.

### 3.2 The current model is not end-to-end low-bit

The primary profile quantizes the seven attention/MLP projections in each block
and leaves the tied embedding/LM head floating point. For Llama-3.2-1B:

| Component | Parameters | Approximate V12/FP16 bytes |
|---|---:|---:|
| Quantized block matrices | 973,078,528 | 183,205,888 including row scales |
| Tied embedding/LM head | 262,668,288 | 525,336,576 if stored once in FP16 |
| Combined | 1,235,746,816 unique | 708,542,464 |

This is approximately 4.59 model-wide bits per unique parameter, even though the
targeted matrices are near 1.5 bpw. The floating tied matrix accounts for about
74% of these compressed weight bytes.

The canonical artifact builder currently iterates both state-dict names and
clones every non-TQ1 value. A tied Hugging Face model exposes both
`model.embed_tokens.weight` and `lm_head.weight`, so the artifact can physically
store the same matrix twice. At FP16, the block payload plus both clones is about
1.23 GB before norms and metadata. Artifact and deployed-GGUF accounting must
distinguish logical aliases, physical storage, and final runtime residency.

### 3.3 Current kernels do not yet justify tighter packing by themselves

The standing CPU measurements show the optimized packed TQ1 GEMV at roughly
1.8–2.0 canonical GB/s. It is faster than the scalar oracle but remains slower
than an already-resident dense BLAS matrix. The current multi-token path loops
GEMV and is 3.36–58.45 times slower than the corresponding resident dense BLAS
baseline. This makes separate decode and prefill paths mandatory.

## 4. Milestone QI-0: make runs valid and accounting honest

### 4.1 Replace step-based QAT phases with token-based phases

Add canonical token-domain fields, for example:

```yaml
soft_tokens: ...
hard_tokens: ...
freeze_indices_at_tokens: ...
freeze_max_tokens: ...
freeze_eval_every_tokens: ...
```

The exact values are selected from a pilot; this plan does not prescribe them
without evidence. The implementation must:

1. derive the QAT phase and temperature from global `tokens_seen`;
2. evaluate freeze gates on a token-domain cadence so device count cannot change
   the number or placement of gate observations;
3. serialize the token-domain schedule, phase, history, and exact token position;
4. resume without replaying or skipping a transition;
5. reject a configuration at startup when `total_tokens` cannot reach the
   earliest freeze point or cannot provide the required sustained evaluations;
6. migrate or explicitly reject legacy step-domain checkpoints rather than
   silently interpreting their units differently.

Acceptance:

- one-, two-, and four-process simulations transition at the same global token;
- an interrupted/resumed run produces the same phase, temperature, gate history,
  and frozen indices as an uninterrupted run;
- the canonical 200M-token configuration either contains a feasible token
  schedule or fails before loading the model;
- a run cannot finish successfully while being structurally unable to qualify.

### 4.2 Make tied storage alias-aware

Extend the canonical artifact schema with explicit aliases. Store one physical
tensor or payload and let all logical state-dict consumers reference it. Hash the
canonical value once and hash the alias mapping separately.

Acceptance:

- tied embedding/head values occupy one physical payload;
- save/reload restores Python-level tying, not merely equal values;
- corrupt, cyclic, missing, shape-incompatible, or dtype-incompatible aliases
  fail closed;
- artifact validation reports unique logical parameters, logical references, and
  physical bytes separately;
- GGUF export preserves the target architecture's tied-output convention.

### 4.3 Report model-wide physical cost

Replace a single target-only bpw headline with the following fields:

- unique logical parameters;
- low-bit and high-precision parameter counts;
- ideal code bits per low-bit weight;
- packed code, scale, affine, codebook, and alignment bytes;
- canonical-artifact bytes;
- final GGUF bytes;
- backend-private repack bytes;
- resident language-model bytes;
- optional component bytes;
- model-wide effective bpw over unique parameters;
- estimated and measured bytes streamed per decode token;
- peak memory at named context lengths, including KV cache and workspaces.

Acceptance requires reconciling the byte sum against actual file sizes and
runtime allocation measurements. Tied aliases must not inflate unique parameter
counts, but any deliberately duplicated backend storage must appear in physical
resident bytes.

### 4.4 Implementation status (2026-07-15)

QI-0 is implemented in the training and schema-2 artifact paths:

- `QATSchedule` and `QATController` use global-token positions exclusively.
  Training validates reachability, launch alignment, and sustained-observation
  capacity before loading model weights. Controller schema 3, the exact gate
  code snapshot, and the data position are checkpointed; step-domain controller
  checkpoints and schema-2 histories lacking finite KL/zero/underflow gate
  measurements are rejected by resume and export.
- The canonical 200M-token profile has boundaries aligned for one, two, and four
  processes. These values make the run structurally valid; they remain pilot
  inputs rather than evidence that the quality thresholds are optimal.
- New schema-2 artifacts emit the separately hashed `tensor_aliases` extension.
  The Llama embedding is stored once, `lm_head.weight` points to it, runtime load
  verifies Python object identity, and GGUF validation requires the tied-output
  convention. Alias hash, target, chain/cycle, shape, dtype, and kind failures
  are fail-closed.
- `size_accounting` now decomposes codes, embedded/row scales, affine data,
  codebooks, non-TQ1 storage, container overhead, and the entire artifact. It
  distinguishes unique parameters from logical references and reports
  model-wide bpw. The baseline decode-stream estimate is one physical weight
  traversal; measured deployment fields remain nullable until observed. GGUF
  validation records its actual file bytes, while packed runtime load reports
  unique live tensor storage and backend-private repacks. Named
  optional-component/context maps accept only nonnegative byte measurements.

Focused acceptance tests cover invariant one/two/four-process transitions,
interrupted resume, impossible schedules, legacy rejection, tied physical
storage and reload, corrupt aliases, model-wide accounting reconciliation,
runtime residency, and tied GGUF output.

## 5. Milestone QI-1: establish the real 1B baseline matrix

No format or training change is promoted until the actual 1B model has a pinned,
reproducible baseline matrix. Use the same source revision, tokenizer, calibration
set, evaluation set, and runtime configuration for every comparable row.

Required weight-format rows:

- dense teacher;
- unhealed source projected into each candidate format;
- existing lossless/full ternary reference;
- llama.cpp TQ1_0 and TQ2_0 references;
- IQ1_S on the same source;
- TQ1 V11 and V12, PTQ and QAT;
- ordinary ternary g128 with one FP16 scale per 128 weights;
- binary g128 with one FP16 scale per 128 weights.

Required activation/training axes:

- weight-only training and evaluation with higher-precision activations;
- deployment-exact W+A8 training and evaluation;
- cross-evaluation of each trained model in both modes where legal.

Required scale axes for current TQ1 formats:

- row scale;
- block-256 scale where supported;
- scale-128 only after the g128 baseline demonstrates enough benefit to justify a
  new format and kernel path.

Every row records model-wide physical bytes, CE/PPL, teacher-KL distribution,
top-token agreement, task results, export parity, decode, prefill, and peak memory.
Quality and performance gates are declared after the dense baseline is measured
and before candidate results are opened.

This milestone also prevents terminology collisions from contaminating reports:
Bonsai `Q1_0` is binary, llama.cpp `TQ1_0` is ternary, and repository
`TQ1_V11/V12` types are restricted-codebook ternary. Reports must include the
full physical type and scale granularity, not a short marketing label.

## 6. Milestone QI-2: end-to-end tied embedding/head quantization

The Bonsai papers consistently quantize embeddings and the LM head in addition
to block projections. For this model, doing so is both the largest remaining
storage reduction and a major decode-bandwidth opportunity.

### 6.1 Representation and projection

Generalize tensor policy beyond `nn.Linear` so a tied embedding/output matrix can
be one quantized tensor with two consumers. Its K dimension is 2,048 and satisfies
the existing TQ1 divisibility requirement.

The initial projection must account for both uses:

- output-head sensitivity from final-hidden-state calibration statistics;
- embedding-row sensitivity from deployment token frequency and embedding
  reconstruction;
- end-to-end held-out KL and task impact as the deciding objective.

Do not create independently projected embedding and head payloads and then call
them tied.

### 6.2 QAT

Use one latent parameter and one assignment/scale state. Gradients from embedding
lookups and output logits must accumulate into the same latent. Checkpointing,
freezing, code-flip metrics, margin metrics, and exact-index export apply to this
shared tensor exactly as they do to block projections.

Add per-family diagnostics so damage from the output head is visible rather than
averaged across transformer blocks.

### 6.3 Runtime and export

Implement:

- a packed row-gather path for input embeddings;
- a packed output-head GEMV for decode;
- a tiled/repacked output-head GEMM for prefill or batched scoring;
- one canonical payload and one set of row scales shared by both graph consumers;
- GGUF import/export behavior consistent with tied Llama output weights.

The performance harness must add the real `N=128256, K=2048` output-head shape,
representative prompt-token counts, repeated-token embedding lookups, and ragged
edge cases.

### 6.4 Quality fallback

End-to-end TQ1 is a measured target, not an unconditional release requirement.
If the shared matrix fails a predeclared quality gate, evaluate explicit shared
Q2/Q4/Q8 alternatives. If the head is untied as a fallback, report the extra
parameters and bytes and stop describing the artifact as tied or end-to-end at
the lower bit width.

## 7. Milestone QI-3: improve the healing mixture and quality gates

### 7.1 Capability-balanced healing data

The current instruction builder supports exact assistant masks but defaults to a
single SmolTalk source. Build a multi-source, quota-controlled mixture containing:

- instruction and exact-format compliance;
- single- and multi-turn tool calls;
- longer math/reasoning traces;
- executable code and repair examples;
- representative multilingual traffic;
- long-context conversations and document use;
- ordinary chat and explanatory prose so specialization does not erase general
  behavior.

Because CE/KD is assistant-masked, enforce training quotas by supervised
assistant-token share. Also record total context-token share because prompt tokens
still determine activations. Dataset manifests must include source revisions,
licenses, selected IDs, bucket counts, assistant-token counts, length statistics,
deduplication information, and chat-template/tokenizer hashes.

The teacher remains the dense source model for the reproducible baseline. Larger
same-tokenizer teachers and on-policy distillation remain separately identified
ablations; they must not be folded into the baseline without attribution.

### 7.2 Capability-specific evaluation

The quality-report schema currently requires nonempty downstream sections but
does not enforce benchmark identity or meaningful thresholds. Define a pinned
non-thinking Llama-appropriate suite covering:

- knowledge: MMLU-Redux;
- reasoning: MuSR;
- math: GSM8K and MATH-500;
- code: HumanEval+ and MBPP+;
- instruction following: IFEval and IFBench;
- tool calling: BFCL v3, including a reported multi-turn slice;
- long context: a pinned retrieval/reasoning suite at multiple context lengths.

Use task-appropriate rule-based scoring first, AST/execution verification for
tools and code, and an LLM extraction fallback only when deterministic parsing
fails. Pin dataset revisions, prompts, templates, scorer versions, execution
images, seeds, token budgets, backend, and determinism flags.

Acceptance gates include:

- aggregate retention against the dense source;
- per-capability retention and maximum regression;
- teacher-KL mean, p50, p95, and p99;
- short/medium/long and language buckets;
- W-only and W+A8 results named separately;
- at least one exact rerun demonstrating score stability.

ARC, WinoGrande, PIQA, HellaSwag, and similar tasks may remain secondary
diagnostics, but they must not be the sole quality gate when saturated or when
they fail to expose instruction, tool, code, and long-reasoning damage.

## 8. Milestone QI-4: calibrated KV-cache quantization

Treat Q4 KV as an optional memory feature, not a speed feature. For the
full-attention 1B architecture, measure its value independently rather than
copying the 27B hybrid-attention memory numbers.

### 8.1 Calibration and artifact contract

Collect model- and layer-specific K-channel means from a representative corpus.
Record whether statistics were collected before or after RoPE/K rotation, the
attention implementation, KV type, tokenizer/model revisions, context lengths,
record and token counts, and source hashes.

Store the mean-centering bias in a separately hashed artifact linked to the exact
model artifact. The loader must reject model, layer-count, head-shape, dtype,
rotation-state, or calibration-contract mismatches.

The existing calibration-data guide remains the base corpus procedure. Extend it
with KV-specific long-context and generation prefixes rather than creating an
untracked synthetic-only path.

### 8.2 Evaluation

For FP16, Q8, and Q4 KV:

1. compare each mode with the same model's own FP16-KV output;
2. measure forward KL on the model's own generations;
3. measure forward KL on off-policy long-context inputs;
4. run downstream and long-context accuracy, not KL alone;
5. report memory, decode latency, and prefill latency by context length;
6. compare Q4 with and without mean centering.

Only after the inference baseline is understood should training add an optional
deployment-exact Q4-KV fake-quant or noise curriculum. The Bonsai result suggests
noise tolerance may be trainable, but does not establish its cause.

## 9. Milestone QI-5: physical-layout and kernel work

Logical format and backend representation must be separable. A canonical TQ1
artifact may be repacked into a backend-private layout as long as the repack is
exact, versioned, validated against the scalar oracle, and included in resident
memory accounting.

### 9.1 Decode

Keep a direct packed batch-1 GEMV path. Prioritize:

- tiled output-row processing;
- hoisted index/high-bit extraction;
- codebook or ternary decode reuse;
- fused scale application;
- output-head specialization;
- backend-specific binary or 2-bit dot-product paths where they win.

### 9.2 Prefill and batching

Do not use repeated GEMV as the final prefill design. Evaluate:

- tiled quantized GEMM that reuses decoded weights across M;
- a backend-private 2-bit ternary expansion;
- a cached dense or wider repack when its extra residency is acceptable;
- separate routing thresholds for short prefill, long prefill, and decode.

The ternary Bonsai deployment is evidence that a physical 2-bit representation
can be preferable to tighter 1.58-bit packing when the hardware maps it better.
Select the path by end-to-end latency, resident bytes, and energy—not bpw alone.

### 9.3 Required measurement

Every kernel implementation follows the repository performance rule. Before it
is pushed, record the scalar-oracle correctness result, tolerances and observed
errors, baseline and candidate timing, device/toolchain, exact shapes and formats,
warmups, iterations, median and p20/p80, and a keep/reject decision in
`bitnet_train/perf/optimization_status.md`.

Add model-level measurements for:

- `tg128` generation;
- `pp512` prompt processing;
- TTFT at several prompt lengths;
- output-head share of decode time;
- packed/repack memory residency;
- peak memory at named contexts;
- sustained throughput and thermal behavior;
- average power and energy per token.

Report energy per token rather than inferring efficiency from instantaneous
power. A faster kernel may draw more power while completing each token with less
energy.

## 10. Milestone QI-6: speculative decoding, only after profiling

Speculative decoding is not on the critical path for a 1B target. The Bonsai
release shows both positive CUDA results and severe regressions when the low-bit
target forward is already cheap. Accepted length alone is therefore insufficient.

Do not begin a drafter implementation until a measured verification-cost model
predicts positive return on a named backend and workload. If it is pursued, the
reference design to test is:

- target-specific lossless verification;
- block size around four, selected from the measured cost model;
- a small block-parallel drafter;
- normalized hidden taps from several target layers;
- a lightweight sequential head to reduce suffix decay;
- a confidence/survival head for scheduling;
- survival-weighted distillation;
- a quantized drafter with verified parity to its reference precision.

Acceptance is end-to-end latency and service behavior, including prompt-cache
reuse, concurrency, resident drafter bytes, acceptance distribution, and
workload-specific regressions. A drafter that improves a one-shot code prompt but
forces costly multi-turn re-prefill is not a general win.

## 10.1 Implementation status (2026-07-15)

QI-1 through QI-6 now have executable reference implementations and fail-closed
evidence contracts. This is deliberately not recorded as production promotion:

- **QI-1:** `baseline.py` and `quant/baseline_matrix.py` define the full,
  unambiguously named matrix. Dense evidence must be recorded before immutable
  gates, and candidate task/timing inventories, parity, quality, storage, and
  hashes are enforced. The actual Llama-3.2-1B matrix is not yet populated.
- **QI-2:** one shared TQ1 latent/payload now serves embedding lookup and output
  logits in PTQ, QAT, schema-2 artifacts, GGUF, scalar runtime, and native CPU
  runtime. Calibration includes final-hidden sensitivity and mergeable token
  frequencies. Explicit shared Q2/Q4/Q8 G128 references are available for the
  quality fallback, but none is silently promoted to a physical GGUF type.
- **QI-3:** the instruction builder accepts a schema-3 multi-source mixture,
  schedules by supervised assistant-token deficit, globally deduplicates, and
  records full token/source provenance. The capability evaluator pins the
  required task inventory, task-appropriate deterministic scorers, W-only/W+A8,
  KL tails, stratification, and exact-rerun gates. Production dataset revisions,
  execution images, and scores still have to be supplied and measured.
- **QI-4:** a separately hashed, model-linked per-layer key-mean artifact,
  centered Q4 and ordinary Q8 scalar oracles, attention reference, optional STE,
  and the required per-context cache/decode/prefill evaluation validator are
  implemented. A real
  long-context calibration/evaluation run has not yet promoted Q4 KV.
- **QI-5:** the native CPU path retains packed decode and adds the opt-in,
  byte-budgeted, versioned `tq1_dense_f32_row_major_v1` repack with distinct
  short-prefill, long-prefill, and output-head routes. The required focused
  M4-Max A/B is recorded in the performance notebook. Model-level `tg128`,
  `pp512`, TTFT, thermal, peak-context, and joules/token evidence remains a
  separate promotion gate.
- **QI-6:** measured-cost selection, gated construction, normalized multi-tap
  block drafting, sequential correction, survival-weighted distillation,
  lossless greedy verification, Q8 drafter parity, and service-report validation
  against the frozen memory, workload-regression, and full parity thresholds are
  implemented. No production drafter is enabled: a real named backend and
  workload must first produce a positive cost decision.

Synthetic tests establish correctness and failure behavior only. They are not
substitutes for the missing 1B quality, long-context, model-throughput, energy,
or service measurements.

## 11. Execution order and gates

Implement in this order:

1. **QI-0:** token-domain QAT schedule, startup feasibility checks, tied aliases,
   and model-wide accounting.
2. **QI-1:** actual 1B baseline matrix with pinned quality and performance gates.
3. **QI-2:** shared embedding/head quantization, export, and runtime support.
4. **QI-3:** capability-balanced healing data and enforceable evaluation suite.
5. **QI-4:** calibrated Q4 KV support and robustness evaluation.
6. **QI-5:** separate decode, prefill, and output-head kernel optimization.
7. **QI-6:** speculative drafting only if profiling justifies it.

Production training spend is blocked until QI-0 is complete. A new low-bit format
is blocked until QI-1 demonstrates a quality or physical-runtime advantage that
cannot be obtained with an existing type. End-to-end low-bit claims are blocked
until QI-2 ships without a large floating embedding/head escape hatch. Speed or
energy claims are blocked until the affected full-model path and kernels have the
required correctness and performance records.

## 12. Required final report for each promoted profile

Each promoted quantization/training profile publishes:

- exact source model, revision, license, tokenizer, and chat-template hashes;
- exact training/calibration manifests and token counts;
- format, codebook, scale granularity, activation mode, and KV mode;
- low-bit coverage by unique parameters and decode-byte share;
- ideal, packed, artifact, GGUF, repack, resident, and peak-context sizes;
- dense, unhealed, PTQ, and QAT quality results by capability;
- teacher-KL distribution and calibration-convergence evidence;
- PyTorch/artifact/GGUF/runtime parity results;
- decode, prefill, TTFT, memory, thermal, and energy measurements;
- known fallbacks and their honest footprint cost;
- commands, device/toolchain, seeds, versions, and immutable artifact hashes.

The report must make it impossible to mistake target-only bpw for model-wide bpw,
ideal entropy for deployed storage, a kernel microbenchmark for model throughput,
or aggregate quality for capability retention.
