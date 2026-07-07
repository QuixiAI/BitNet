# Qwen3-MoE 1.58-Bit Healing Plan

## 0. What this project is

### 0.1 The idea in one paragraph

MoE and ternary quantization attack opposite problems and compose: MoE buys parameters
per FLOP; ternary buys bits per parameter. In Qwen3-15B-A2B, ~93% of stored weights
are the 128-per-layer experts; ternarizing **only the experts** captures nearly all
the memory win while leaving every precision-sensitive component — router, norms,
attention, embeddings — untouched. The deployment story is the payoff: CPU **decode**
is bandwidth-bound (tokens/sec ≈ bandwidth ÷ bytes/token); top-8-of-128 routing reads
~1/16th of expert weights per token, ternary packing cuts each byte ~8× vs BF16, and
per-token weight traffic lands near ~1 GB — tens of tokens/sec on dual-channel DDR5.
That is usable local batch-1 decode on a normal machine, the one shape where CPUs
don't have to beat GPUs. Two scope truths up front: batch/throughput serving inverts
the story (GPUs win; out of scope), and **the sparsity advantage largely vanishes at
prefill** — prompt tokens route to the union of experts, so TTFT reads nearly the
whole expert pack (§7.5). Decode is the design target; TTFT is measured first and
reported per prompt length, never hidden.

### 0.2 Strategy: quantize-then-heal

Start from the pretrained BF16 checkpoint; replace expert linears with quantized
layers trained through a straight-through estimator; continue pretraining — with the
original model as a distillation teacher — until quality recovers. Healing from a
strong base costs orders of magnitude fewer tokens than training a ternary model from
scratch, at a modest quality ceiling below native 1-bit training. Expectation, to be
measured, not asserted: the unhealed converted model will be substantially damaged
(though likely less than a fully-ternarized dense model, since attention and
embeddings stay intact); healing then behaves more like early pretraining than
fine-tuning, which shapes the learning-rate policy (§5.3) and monitor tolerances (§6).

### 0.3 The two-projects doctrine

Track Q is two projects with different risk profiles, kept strictly apart:

```text
The HEALING project (Q-T*):  produce Qwen3-15B-A2B-1.58, validated in mainline
                             llama.cpp. MVS lives here.
The ENGINE project  (Q-K*):  a custom CPU runtime that must beat the llama.cpp
                             floor to justify existing.

Rules: training never depends on custom kernels; kernel design never blocks Q-T0;
the K-track can fail WITHOUT killing the model artifact (the llama.cpp route always
exists); neither project spends at scale until its own gate passes.
```

### 0.4 Gates

```text
Q-T0 — architecture/conversion/export recon spike        (§8.1; days, no training)
Q-K0 — CPU tokens/sec de-risk, decode AND TTFT           (§8.2; days, no training)
Q-T1 — tiny heal, doubling as the STACK SHAKEDOWN        (§8.3; first training)

Production spend (Q-T2+) requires Q-T0 pass, Q-K0 pass, and Q-T1 exit criteria.
The CI micro-loop (§5.6) guards the pipeline continuously from the first commit.
```

The spikes are front-loaded because the riskiest assumptions are not about healing —
they are "the checkpoint converts and exports" and "the deployment premise is
physically real." Both are answerable in days for zero training compute.

### 0.5 Naming, honesty, and the model card

Artifact: `Qwen3-15B-A2B-1.58`. Every released artifact ships a model card stating,
at minimum: base model, exact source repo/hash, and license; **experts-only
ternarization** — ~93% of *stored* params ternary but only a minority of *active
decode compute* through ternary weights (a **Q-T0-computed figure**; back-of-envelope
from known shapes: ~0.91B active expert params vs ~0.45B attention + ~0.31B lm_head
⇒ point estimate **~54%**, finalized once head counts and tying are pinned — the
storage story is strong, the compute story weaker; say both); packed size vs the
naive bits-per-param estimate; the gap to the BF16 baseline on standard evals; **the
eval mode and runtime every number was measured in** (§7.5); heal token count and
teacher; decode tok/s AND TTFT per stated prompt length on the named CPU; known
limitations. This is a heal of a Qwen3-shaped MoE — not a MoTE reproduction and not
a BitNet b1.58 2B4T reproduction; the recipe descends from the
quantize-and-heal lineage (e.g. the Llama3-8B-1.58 work), and the card says so.

### 0.6 Minimum viable success

```text
1. The converted model trains ≥100M tokens without numerical failure.
2. Validation PPL improves over the unhealed fake-quant baseline — in PyTorch AND
   in the exported GGUF.
3. The export runs in mainline llama.cpp (qwen3moe + TQ2_0 experts).
4. Exported PPL matches PyTorch mode a0 (§7.5) within a predeclared tolerance.
5. KL-to-teacher on the fixed calibration set decreases through the heal.
6. Router health holds: per-layer routing entropy and expert-load histograms stay
   within the Q-T1-measured healthy band; no persistent dead experts; no
   decay-driven cold-expert erosion (§4).
7. The custom runtime, WHEN it exists, matches mode a1 — but its existence is a
   K-track deliverable, never an MVS dependency.

NOT in MVS: on-policy distillation, SFT/DPO, kernel acceleration, benchmark
targets, the FP8 cast (§3.3, a Q-T4 step), any tokens/sec target (Q-K0/K-track).
```

Clause 2's "in the exported GGUF" is deliberate: "healed in PyTorch, broken through
export" is the failure mode the parity apparatus exists to catch. The PPL bar is low
from a damaged start — it gates the *loop*; quality bars live at Q-T2/Q-T3.

---

## 1. Target model specification

### 1.1 Known facts vs Q-T0-recorded facts

Known (from the checkpoint in hand; anything else is **recorded at Q-T0, never
assumed** — enumerate, don't assume):

| Field | Value |
|---|---|
| layers / hidden | 24 / 2048 |
| experts / top-k / shared expert | 128 / 8 / none |
| expert intermediate (SwiGLU) | 768 |
| params total / active | ~15B / ~2B |
| expert params | 24 × 128 × 3 × (2048×768) ≈ 14.5B (~93%) |
| active expert params/token | 24 × 8 × 3 × (2048×768) ≈ 0.91B |

Record at Q-T0: full HF config (heads/GQA, head_dim, QK-Norm presence/shape,
rope_theta/scaling, vocab, **tied embeddings or not — load-bearing for the §2 memory
endgame, the FP8 plan, and Q-A-head8**, `norm_topk_prob`, `router_aux_loss_coef`,
context length), tokenizer identity, the full `nn.Linear` inventory from walking the
module tree, and — hard requirement — **provenance and license pinned to an exact
repo/hash**. 15B-A2B is not a headline Qwen3 release; unpinned provenance stops the
track.

**Config preservation is a hard requirement:** the conversion copies the config
verbatim and a test diffs every field (`rope_theta`/scaling, `norm_topk_prob`,
`router_aux_loss_coef`, tying above all). Silent config drift is a bug that survives
until production if not caught here.

### 1.2 Conversion profile

```text
target_linear_regexes:  mlp.experts.\d+.(gate_proj|up_proj|down_proj)   # experts ONLY
keep_fp_list:           mlp.gate (the ROUTER — trap below), embed_tokens, lm_head,
                        self_attn.{q,k,v,o}_proj, every *norm* incl. q_norm/k_norm,
                        all 1-D tensors, rope buffers
freeze_fp_params:       per §5.2 (attention/embeddings frozen at baseline)
decay_masking:          cold-expert mask ON (§4), CI-tested (§5.6), fallback §5.2
quant_granularity:      per-tensor absmean (baseline; §3.7)
teacher:                the original BF16 MoE checkpoint
export_route:           mainline llama.cpp TQ2_0 (validation) + custom pack
                        (deployment, K-track)
eval_modes:             a0 / a1 / b (§7.5)
```

**The regex trap:** Qwen3-MoE has both `mlp.gate` (the router, 128×2048 — the one
tensor where quantization noise changes the *computation graph*, not just adds error;
BF16 forever) and `mlp.experts.N.gate_proj` (an expert linear; ternarize). An
unanchored `gate` regex ternarizes the router. The conversion test asserts by name
that `mlp.gate` survives as BF16 `nn.Linear` and that exactly 24×128×3 = 9,216
modules became BitLinear.

---

## 2. The precision map (the whole design in one table)

| Component | ~Params | Training | Export | Why |
|---|---|---|---|---|
| Expert gate/up/down (×9,216) | 14.5B | **ternary QAT** (BitLinear, W2A8 fake-quant) | packed ternary | where most stored parameters live; the memory win |
| Router `mlp.gate` (×24) | 6.3M | BF16, trainable | **BF16** | discrete top-8 flips downstream of tiny logit gaps; never quantize |
| Norms (incl. QK-Norm) | ~0.01B | BF16, trainable | **BF16** | tiny; scales everything downstream |
| Attention q/k/v/o | ~0.4B* | BF16, **frozen** (baseline) | **FP8** (cast at Q-T4, measured) | dense per token; precision-sensitive; undamaged by conversion |
| embed_tokens / lm_head | ~0.6B* | BF16, frozen | FP8 **pending Q-A-head8** (or tie) | lookup + one logit matmul; head precision measured separately (§3.3) |

\* measured at Q-T0; tying changes both rows. Memory endgame (recomputed at Q-T0):
experts packed ~2.9–3.7 GB (format-dependent) + attention FP8 ~0.45 GB + embed/head
~0.6 GB (less if tied; more if Q-A-head8 keeps the head at Q8_0/BF16) + router/norms
~0.015 GB → **~4.5–5 GB from ~31 GB BF16**.

---

## 3. Architecture & precision decisions (with reasoning, so they are not relitigated)

### 3.1 Experts-only ternarization — attention stays full precision

(1) Experts are ~93% of storage — ternarizing them captures nearly all the win.
(2) Individually small experts (2048×768) are exactly where ternary hurts most
(ternary-vs-dense parity in the literature is a ≥3B-dense-scale result); spend the
damage budget only where the payoff is. (3) Attention is read every decoded token —
but at FP8 it is ~0.45 GB/token, affordable inside the byte budget. **Q-A-attn**
(also ternarize attention) is an ablation for the extra ~4× on that 0.45 GB, gated on
Q-T2 quality margin. This split follows the MoTE precedent (ternary experts from a
dense checkpoint, high-precision router).

### 3.2 Router and norms: BF16 always, trainable always

The router is ~0.04% of params and 100% of the discrete structure; quantizing it buys
nothing and can flip expert assignments. Norms likewise (including Qwen3's QK-norms).
Both stay trainable — as experts change under QAT, optimal routing drifts and the
router must follow (§5.2, with the optional warm-freeze ablation §9). The
load-balance aux loss stays ON throughout (§5.1).

### 3.3 FP8 non-experts at EXPORT, not during training — and the head measured alone

Heal with attention/embeddings/lm_head in BF16; cast to FP8 only at Q-T4. That the
8-bit residual needs no expert co-adaptation is an *expectation measured at Q-T4*:
the parity suite runs the export with BF16 vs FP8 non-experts and the delta (mode b −
mode a1, §7.5) must sit inside a pre-declared tolerance; fallback on a material delta
is **Q-A-fp8** (fold FP8 fake-quant into QAT, heal an increment). **Q-A-head8:** the
lm_head is logit-sensitive, vocabulary-wide, dense every token, and entangled with
tying — its precision is measured *separately* at Q-T4 (FP8 vs Q8_0 vs BF16) rather
than riding inside the aggregate delta. GGUF note: no FP8 tensor type exists — the
llama.cpp route ships non-experts as Q8_0/F16; true FP8 lives only in the custom
pack. The §7.5 matrix carries this axis explicitly.

### 3.4 No SubLN in the baseline

Two reasons: there is no function-preserving way to insert RMSNorm (at γ=1 it maps
`x → x/rms(x)` — data-dependent, unfoldable into an adjacent linear; inserting it
adds damage on top of quantization damage at t=0), and it exits the `qwen3moe` GGUF
architecture (no tensor slots), forfeiting the validation route. The custom runtime
*could* host invented tensors, so SubLN survives as **ablation Q-A4,
custom-runtime-only, still not baseline** — run only on Q-T2 shortfall with the
damage map pointing at expert-output variance.

### 3.5 No shared expert in the baseline; a gated contingency

The checkpoint has none. A BF16 always-on shared expert is the single biggest quality
lever available — and it costs ~+5% active decode bytes (the optimized quantity),
breaks teacher-architecture match, and exits the `qwen3moe` GGUF arch. Pre-registered
rule: start without; consider only if Q-T2 lands meaningfully below its gate after
the §10 escalation order. If added: init from the utilization-weighted mean of expert
weights, keep BF16, re-run Q-T1-scale validation before committing tokens.

### 3.6 No structural expert edits

No widening, count changes, or merging — real advice for a from-scratch program; a
heal inherits its skeleton, and structural edits invalidate the teacher and the
cheap-recovery premise. Document the quality ceiling instead.

### 3.7 Quantization granularity and the export-alignment rule

**Baseline: per-tensor absmean** (the §4 formula), reconciled against the actual
`quant.py` implementation at Q-T0 before this document is cited as authority. The
choice is entangled with parity: TQ2_0 packs 256-weight blocks under one F16 scale,
so baked tensors carrying multiple distinct magnitudes per 256-block cannot be
represented by that one scale — the exporter must re-quantize, codes can flip near
thresholds, and exact-code parity dies on the validation route *by construction*.
Rules:

```text
Per-tensor absmean (baseline): exact-code parity achievable on BOTH routes.
Group scales (quality option): permitted ONLY export-block-aligned — group-256 for
  the TQ2_0 route (ablation Q-A-g256, gated on the Q-T0 damage map) — or confined to
  custom-pack-only variants with the validation route explicitly downgraded to the
  bounded-mismatch regime (§7.4). Group-32 is custom-pack-only, never a free upgrade.
```

Activations: per-token absmax int8 on expert inputs (§4), unchanged across variants.

### 3.8 Every gradient goes through the ternary forward

From the first Q-T1 step through the last post-training step (SFT/DPO included):
experts always forward fake-quant. Never "full precision just for this phase" — that
trains a model subtly different from the one deployed. (Mode a0 in §7.5 is an *eval*
mode; it never trains.)

---

## 4. The quantization math (inlined; the correctness core)

### 4.1 Quantizers and BitLinear

```python
def weight_quant(w):                       # per-tensor absmean ternary
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    return (w * scale).round().clamp(-1, 1) / scale

def activation_quant(x):                   # per-token absmax int8
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    return (x * scale).round().clamp(-128, 127) / scale

class BitLinear(nn.Linear):                # bias=False (Qwen3 expert projections have none)
    def forward(self, x):
        x_q = x + (activation_quant(x) - x).detach()                        # STE
        w_q = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return F.linear(x_q, w_q)
```

Forward uses quantized values; the `detach` trick makes the quantizers identity in
the backward graph (the straight-through estimator). **Latent weights and optimizer
state stay high precision**; ternary values are recomputed every forward and stored
only at export.

### 4.2 STE ⇒ dense backward (no ternary backward kernels, ever)

Autograd sees only `F.linear(x_q, w_q)`:

```text
grad_x = grad_y @ w_q        # dense bf16/fp32 GEMM
grad_w = grad_y.T @ x_q      # dense GEMM, applied to the FP latent weight
```

No ternary backward kernel exists to build: `grad_y` is dense (a ternary×dense
backward GEMM would optimize the small operand for marginal gain), and `grad_w` has
no ternary operand at all — its output *must* be full precision because AdamW
accumulates it into the latent weight, and that accumulation of sub-threshold updates
is the mechanism by which healing works. Caveat: this holds because the quantizer's
backward is identity by construction; learned scales or soft quantization would
re-enter the backward graph. The fixed λ-ramp ablation (Q-A3) does not.

### 4.3 MoE footnotes

- **Sparse gradients are unchanged in kind.** Autograd routes grads only to the top-8
  experts a token used; BitLinear is stateless; routing works untouched.
- **Cold experts heal slower — by construction.** 16× less routing ⇒ 16× fewer
  gradient steps; latents may drift without crossing ternarization thresholds.
  Detected by per-expert code-flip rate vs routing frequency (§6.2).
- **Cold experts are actively ERODED by weight decay — a bug, not slow healing.**
  AdamW's decoupled decay fires every optimizer step whether or not a parameter
  received gradient; a cold expert gets few updates but the full decay schedule, so
  its latents shrink monotonically toward the zero-code region — decay *manufactures*
  dead ternary experts, masquerading downstream as router collapse. **Fix, mandatory:
  mask decay for expert parameters that received no gradient that step**
  (utilization-floor exemption is the acceptable simpler variant). Easy to state,
  fiddly under FSDP flat parameters and zero-vs-absent gradients — hence the CI test
  (§5.6) and the fallback (§5.2). Dedicated alarm: **zero-code fraction rising
  specifically in the low-utilization tail** (§6.2).

---

## 5. Training recipe

### 5.1 Loss

```text
L = CE(student, tokens)
  + α · T² · KL( softmax(teacher/T) ‖ softmax(student/T) )     # α=1.0, T=2.0 start
  + β_aux · L_load_balance                                      # ON the whole time
```

- **Teacher = the original BF16 MoE**, frozen, no-grad. Healing's objective is
  *match the original function*; the KL term optimizes that directly and heals
  quantized models substantially faster per token than CE alone — decisive at
  PoC-scale token budgets.
- **Mandatory implementation requirement — chunked losses:** the training loop MUST
  compute CE and KL over vocabulary chunks (or fused equivalents) such that full
  (T, V) teacher and student logit tensors are never simultaneously materialized.
  V ≈ 152K makes this a day-one correctness-of-design constraint, not an
  optimization.
- **Top-k teacher caching (the recommended default):** precompute top-k teacher
  logits offline over the frozen corpus, removing the ~30 GB teacher from step
  memory. Two rules: (1) tail mass handled by an explicit, config-recorded choice —
  renormalize over top-k, or one "other" bucket — the choice changes gradients;
  (2) caches are valid only while corpus, tokenization, and windowing are frozen —
  pin the data first, hash the manifest into checkpoint metadata.
- **Load-balance aux loss:** the base model's own mechanism and coefficient
  (`router_aux_loss_coef`, read at Q-T0, never invented). Without it, the router
  concentrates onto the fastest-healing experts → concentrated gradients → widening
  gap → the collapse spiral. Sweep β_aux only on the §6 trigger.
- **Router distillation** (teacher-vs-student routing KL): ablation Q-A-route, not
  baseline — logit KD constrains routing indirectly; over-constraining fights the aux
  loss. **Intermediate-layer distillation:** attention is neither quantized nor
  trainable at baseline; reserve hidden-state matching for the case where logit-KD
  plateaus.

### 5.2 Parameter groups, trainability, decay

| Group | Trainable? | LR (rel.) | Decay | Rationale |
|---|---|---|---|---|
| Expert latents (QAT) | yes | 1.0 (§5.3 grid) | 0.1, **cold-masked (§4.3)** | the point |
| Router `mlp.gate` | yes | 0.1× (T1 sweep {0.05, 0.1, 0.3}×) | 0 | must track expert drift; too fast destabilizes top-8 |
| Norms | yes | 0.1× | 0 | tiny; standard practice |
| Attention q/k/v/o | **frozen** (ablation: 0.1×) | — | — | undamaged, teacher-shared anchor; saves optimizer state, removes a drift channel |
| embed / lm_head | frozen | — | — | undamaged; teacher-shared; Adam on 0.6B saved |

Optimizer: AdamW, betas (0.9, 0.95), eps 1e-8, gradient clipping 1.0. Freezing
rationale: the teacher IS the base model, so undamaged teacher-shared components are
an anchor; **Q-T1 runs the unfreeze ablation once before locking.**

**Decay-mask fallback (safety ordering):** until the cold-mask CI test (§5.6) is
green under the *real* optimizer/FSDP wrapping, **expert weight decay is 0 for
Q-T1.** Zero decay trivially prevents erosion; regularization returns only once the
mask is proven. Running unmasked decoupled decay by accident is the failure this
ordering forbids.

### 5.3 Learning rate

Philosophy: sweep wide at Q-T1, decide from data, with the code-flip rate (§6.2) as
the mechanistic readout — near-zero ternary code flips means latents drift without
crossing quantization thresholds (the effective model is frozen no matter how healthy
optimizer steps look); sustained very high flips means thrashing. Grid, shifted high
because narrow experts + sparse routing mean fewer, noisier updates per expert, and
because heal-from-damaged behaves like early pretraining (the quantize-and-heal
literature found too-small LR fails to recover):

```text
Q-T1 expert-LR grid: 1e-4, 2e-4, 4e-4, 8e-4
```

Watch per-expert flip rates across the utilization spectrum, not the aggregate.
Schedule: linear warmup 200 steps → cosine to 10% of peak (scale warmup with run
length).

### 5.4 Precision and memory (~14.6B trainable)

| Configuration | B/param | ≈14.6B | Notes |
|---|---|---|---|
| fp32 W+G+Adam | 16 | ~233 GB | multi-node only |
| bf16 latents + fp32 master + bf16 grad + 8-bit Adam | ~10 | ~146 GB | sharded state only |
| + frozen non-experts (baseline) | — | −~16 GB | |
| teacher | +30 GB bf16, or ~0 with top-k caching | | |

**Planning floor: 4-way FSDP (≥4×80 GB).** 2×H100-80 is a stretch config: ~146 GB
sharded across 160 GB leaves single-digit GB per GPU for activations, frozen params,
buffers, fragmentation. No single-GPU option, no fp32-everything option. Note:
PyTorch does not maintain fp32 masters for bf16 params unless you build it — verify
the optimizer-state dtypes explicitly. Gradient checkpointing + chunked losses
mandatory day-one. **De-risk the reduced-precision stack before trusting it:** the CI
micro-loop (§5.6) plus a short single-node smoke run on a truncated model (e.g.
4 layers × 16 experts sliced from the checkpoint) validates bf16-latents/fp32-master/
8-bit-Adam mechanics before the first multi-node Q-T1 job.

### 5.5 Data

- **Tokenizer:** Qwen3's (~152K vocab) — any corpus tokenized otherwise is
  **re-tokenized, never remapped**.
- **Shard dtype: uint32** — 152K does not fit uint16; a uint16 pipeline silently
  wraps a third of the vocab and trains on garbage.
- **Format:** flat EOS-separated token shards + `manifest.json`, non-overlapping
  windows via lazy `np.memmap`, window-level shuffle, `drop_last=True`.
- **Freeze early:** corpus, tokenization, and windowing pinned at Q-T1 start — for
  run comparability and because teacher caches are invalid if data moves. Manifest
  hash, quantizer version/hash, RNG seeds, and config hash go into checkpoint
  metadata; parity reports are only meaningful with provenance pinned.
- **Corpus:** match the base model's pretraining distribution as practical — heal,
  don't domain-shift. General/educational web, code, multilingual in Qwen-like
  proportions (Qwen3 is heavily multilingual; a pure-English heal corpus is a mild
  domain shift — note it in the card). PoC 2–5B tokens; production 50–150B — sourcing
  at that scale is the main data lift; start during Q-T1.
- **Batch sizing, the MoE rule: per-expert effective batch = tokens/step ÷ (E/k) =
  tokens/step ÷ 16.** A 65K-token step gives ~4K tokens/step/expert — enough to see
  learning, too noisy to judge it. T1 at 65–130K tokens/step (grid comparability),
  T2 at 0.5–1M, T3 at 1–4M.
- **Sequence length:** 1024–2048 (T1) → 2048 (T2) → toward the target context (T3).

### 5.6 Harness and CI

Accelerate/FSDP; `save_pretrained` HF checkpoints (required by the exporter) +
`trainer_state.pt` + `latest` symlink + `--resume`; wandb; the §6 monitor wired in
before the first Q-T1 step; FSDP config in checkpoint metadata. **Distributed-flag
note:** `find_unused_parameters=False` (a common DDP default for dense models) is
WRONG for sparse MoE under DDP — unrouted experts are absent from a step's graph.
Training runs under FSDP where the flag is moot; any DDP debug config sets it True.

**CI micro-loop, from the first commit:** a random tiny MoE (2 layers, 8 experts,
top-2, tiny vocab) runs the entire pipeline on every commit — convert (profile-driven
swap, `gate`/`gate_proj` assertion) → train 10 steps (chunked CE+KD, aux loss, decay
mask) → save/reload → bake + export → TQ2_0-style quantize → tensor parity →
tiny-runtime PPL parity. Minutes, no GPUs. The program's highest compound risk is
loop breakage discovered at scale; the micro-loop converts it into red CI.

**Required before Q-T1 — the decay-mask test, under the real wrapping:**

```text
route zero tokens to expert j → optimizer.step()
  → assert expert j's latents unchanged (modulo allowed global bookkeeping)
route tokens to expert j → optimizer.step()
  → assert decay applied exactly now
```

Run it against the actual optimizer/FSDP configuration used in training, not a toy
AdamW — zero-vs-absent gradients and flat-param sharding are exactly where the mask
silently breaks while the doc claims it's fixed. Until green: the §5.2 fallback.

---

## 6. Monitoring

### 6.1 Core metrics

Train/val loss (CE and KD terms separately), val PPL **per eval mode (§7.5)**, KL_tf
(teacher-forced KL-to-teacher on a fixed calibration set — the truest healing gauge),
grad norm, LR, tokens/s.

### 6.2 Ternary and router health (logged from step 0)

Doctrine: *if healing goes sideways it shows up first as the router collapsing onto a
few experts, not as a clean perplexity regression.*

```text
per-layer routing entropy              (the canary; alarmed on sustained decline)
expert-load histogram / max-load       (concentration)
dead-expert count                      (< ε utilization for K consecutive evals)
zero-code fraction, low-utilization tail   (the decay-erosion alarm — distinct from
                                        generic ternary degeneracy)
top-8 agreement with teacher, PER LAYER    (the MoE analogue of KL_tf. Depth-aware:
                                        only layer 1 is exact at t=0 — deeper routers
                                        see inputs already carrying accumulated
                                        activation error from quantized experts
                                        below; expect a depth gradient dipping then
                                        re-stabilizing; an aggregate number would
                                        misread this as an anomaly)
per-expert code-flip rate vs routing frequency   (the cold-expert scatter; also the
                                        LR-grid readout, §5.3)
per-BitLinear {-1,0,+1} code fractions, absmean scale, quantization error
aux-loss value                         (trend, not just weight)
```

Pre-registered rules: entropy declining + max-load rising over N evals → raise β_aux
one step; not arrested within the next window → drop expert LR one grid step.
Zero-code tail firing with routing otherwise healthy → **verify the decay mask is
active before touching anything else.** Perplexity is NOT the trigger — by the time
PPL regresses, the collapse is mature. Tolerate ugly early absolute values: the first
steps look like early pretraining, and quantized-model loss curves are often S-shaped
(plateau, then drop); never kill runs during the plateau.

### 6.3 Dual-mode tracking: a0 and a1 at every eval

**Both mode a0 (experts W-only) and mode a1 (W2A8) PPL/KL_tf are logged at every
Q-T1/Q-T2 eval**, not only at milestones. Reason: W2A8 training can learn weights
that co-adapt to activation-quant noise, and the W-only eval — the one llama.cpp
validation depends on — can drift from the training forward as healing progresses.
Expected picture: a small, stable a0−a1 gap (the inherent two-runtimes delta,
typically neutral-to-slightly-favorable for a0). **Alarm on a diverging TREND** — a0
improving while a1 degrades or vice versa means the validation and deployment
objectives are separating — not on the mere existence of the gap.

---

## 7. Runtimes, export, and parity

### 7.1 The runtime split

```text
VALIDATION runtime:  mainline llama.cpp — qwen3moe architecture, TQ2_0 ternary
                     experts (2.06 bits/weight native ternary type), Q8_0/F16
                     non-experts. Weight-only quantization: the runtime does NOT
                     quantize activations. Exists today; MVS validates here.
DEPLOYMENT runtime:  the custom CPU engine (K-track): W2A8 ternary-GEMV experts +
                     FP8 non-experts. Matures in parallel; never an MVS dependency.
```

### 7.2 Baked-ternary export (the parity foundation)

The export checkpoint contains `w_baked = weight_quant(w_latent)` — dequantized
ternary, every value exactly `{-s, 0, +s}` — never raw latents (latents stay in the
training checkpoint format). Baking is *intended* to make exact code recovery
achievable: a quantizer that sees clean `{-s, 0, +s}` input has no reason to flip
codes. That is an intent, not a guarantee — a route with its own block optimization,
clipping, or scale search could deviate — so **Q-T0 determines the regime per route**:

```text
preserve     — packing recovers codes from baked ternary  ⇒ EXACT code match required
re-quantize  — exporter re-ternarizes                      ⇒ bounded mismatch rate,
               every mismatch explained by scale convention; then fix the pipeline
```

Write `w_baked` in fp32 (or verify the bf16 round-trip of the actual `{-s,0,+s}`
values is exact) so dtype conversion cannot flip codes. **F16 scale bound:** llama.cpp
stores block scales in F16, so the per-tensor scale `s` is rounded per block; the
tensor-parity report asserts exact codes PLUS uniform dequant error bounded by F16
rounding of `s` — anything beyond fails. Per exported tensor, log: code mismatch
rate, dequantized max/mean/relative error, quantizer hash. The full acceptance loop —
a few-steps-trained checkpoint can save → reload → bake+export → quantize → run →
match PPL — must pass before any production run.

### 7.3 The K-track (deployment engine), condensed — including the activation contract

- **★ K-track activation contract:** matching mode a1 means the custom runtime is not
  just a ternary-weight GEMV engine — **it must implement the same expert-input
  activation quantization as training** (per-token absmax int8, §4.1), or Q-K0/Q-T4
  must record the measured divergence and resolve it explicitly: either update
  training-side `activation_quant` to the runtime's convention, or downgrade the
  custom-runtime parity expectation from "match a1" to a measured-and-bounded gap.
  Silence is not an option; the convention is recorded once at Q-K0.
- **Hot primitive: batch-1 ternary GEMV** (unpack + add/sub/skip; the multiply
  disappears). Scalar reference kernel first, kept forever as the oracle; every SIMD
  variant diffs against it before a token is trusted.
- **Packing format is the highest-leverage open decision** — 2-bit (trivial unpack,
  +25% traffic) vs base-3 dense (min bytes, painful unpack) vs LUT-indexed partial
  sums. Prototype ≥2, measure against the bandwidth roofline (achieved ÷ machine GB/s
  names the regime), pick with numbers. Training does not couple to the choice
  (export re-quantizes), modulo §3.7's group-scale alignment rule.
- **The MoE gather lives inside the kernel** — read each selected expert's packed
  block in place, accumulate router-weighted output directly; never stage 8 experts
  into a buffer (doubles weight traffic; fatal when weights ARE the traffic).
- **Scales fold into the epilogue**; no dequantized intermediates. **Target CPU class
  pinned at Q-K0** (AVX-512/AMX vs ARM SME).

### 7.4 Byte budgets: decode vs prefill (both stated, neither hidden)

Decode (the design target; recomputed at Q-T0): experts ~0.23 GB + attention FP8
~0.45 GB + lm_head ~0.3 GB ≈ **~1 GB/token** → 50–80 GB/s DDR5 → ceiling ~50–80
tok/s, "tens after overhead"; KV reads add with context. **Prefill is a different
story:** distinct tokens route to distinct experts, so even a few hundred prompt
tokens activate the union — approaching all 128 per layer — and TTFT reads **nearly
the whole ~3–3.7 GB expert pack**; prefill is compute-heavy AND bandwidth-heavy in a
way dense models are not. Prefill is a separate kernel/mode; the 8K-prompt TTFT
number may not be pretty; the card reports tok/s and TTFT per prompt length.

### 7.5 The parity matrix: three eval modes of ONE trained model

llama.cpp is weight-only; the custom runtime is W2A8. One number cannot serve both.
Three PyTorch eval modes of the **single baseline-trained (W2A8) model**:

```text
mode a0:  experts W-only (activation fake-quant OFF at eval) + all-else BF16
          ↔ llama.cpp TQ2_0 experts + Q8_0/F16 non-experts        [validation parity]
mode a1:  experts W2A8 + all-else BF16 (the training forward)
          ↔ custom runtime, pre-FP8 non-experts                   [deployment parity]
mode b:   mode a1 + FP8 fake-quant on attention/embed/lm_head
          ↔ final custom runtime                                  [final parity]

FP8-cast delta (§3.3's Q-T4 gate) = mode b − mode a1.
a0 − a1 = the inherent two-runtimes delta: measured continuously (§6.3), alarmed on
trend, never treated as failure by existence. Only a0's parity against llama.cpp
within tolerance is gated.
```

a0 is an **eval mode, not a training run** (§3.8). Pre-registered decision rule per
pairing: tensor parity clean + PPL parity bad ⇒ divergence is activation-side ⇒ align
training-side `activation_quant` with *that runtime's* actual behavior (the §7.3
contract for the custom runtime; confirmed weight-only-ness for llama.cpp, a Q-T0
read-the-code item).

**Q-A1w — the conditional W-only training variant, trigger pre-registered:** training
with FP activations (experts ternary-only) contests the baseline **only if Q-K0 kills
the custom runtime and llama.cpp is promoted from validation to deployment.** The
logic: when the sole runtime is weight-only, A8 activation fake-quant is training
damage the runtime never imposes — so run W2A8-trained and W-only-trained variants at
Q-T2 and let W-only runtime PPL pick the baseline. While the custom W2A8 runtime
remains the deployment target, W2A8 training stays baseline and a0 stays an eval
mode.

---

## 8. Milestones

### 8.1 Q-T0 — architecture/conversion/export spike (no training; gates all Q work)

(1) enumerate every `nn.Linear`, classify, write the profile (the `gate`/`gate_proj`
assertion in the same commit); (2) config diff incl. `norm_topk_prob`,
`router_aux_loss_coef`, tying; (3) reconcile §3.7/§4.1 against the actual `quant.py`;
(4) convert; fake-quant eval per mode incl. the first a0−a1 delta; **damage
decomposition, MoE edition** — eval-only passes for experts-ternary-only, +A8,
per-layer, per-expert-family; expectations to verify: damage milder than a
fully-ternarized dense model (attention intact; 93%-by-storage but top-8-by-compute
damaged) and roughly uniform per expert; (5) dense→GGUF via the mainline converter,
runs in llama-cli; (6) TQ2_0 quantize incl. the **3-D fused expert stacks**
(`ffn_gate_exps` etc. — row-block quant types should apply, *verified not assumed*)
+ `--tensor-type` per-tensor override verification (garbage quality expected;
pack/load/run is the test); (7) baked-ternary export → preserve-vs-requantize regime
(§7.2); (8) teacher fixture: calibration logits + per-layer routing recorded (feeds
§6.2's agreement metric); (9) provenance/license pinned. *Exit: all nine, no blocker;
damage map delivered.* Fail → the track stops, cheaply.

### 8.2 Q-K0 — CPU tokens/sec spike (gates training spend)

On the Q-T0 **unhealed** export (quality irrelevant; bytes and plumbing are the
test), on the pinned CPU: (1) decode tok/s in llama.cpp TQ2_0 — **the floor**;
(2) scalar + first-SIMD custom-GEMV prototype vs the roofline, activation-quant
convention recorded per the §7.3 contract; (3) **TTFT at 1K and 8K prompt tokens,
measured FIRST** (the union-routing number most likely to disappoint); (4) packing
bake-off entry. *Exit gate, pre-registered:*

```text
(i)   projected healed-model decode ≥ ~10 tok/s on the target machine;
(ii)  TTFT at 1K tokens within the tolerance declared BEFORE measuring;
(iii) the custom-kernel path projects to beat the llama.cpp floor by a pre-declared
      margin (default ≥2×) — otherwise DO NOT build the custom runtime yet.
Corollary: if the llama.cpp floor ALONE satisfies (i)+(ii), that is the cheapest
good outcome — ship on llama.cpp, demote the engine to an optimization project, and
trigger Q-A1w's rule (§7.5).
Below (i) or (ii) on every path: the deployment premise fails — stop or re-scope
before burning training compute.
```

### 8.3 Q-T1 — tiny heal + stack shakedown (100–200M tokens)

The first training milestone doubles as the harness proof (there is no external
pathfinder in a standalone plan): expert-LR grid (§5.3); router-LR sub-sweep;
frozen-vs-unfrozen ablation; **decay mask per §5.2's fallback ordering — CI test
green or decay zeroed**; teacher-cache validation; router panel live, healthy bands
recorded; a0/a1 dual-mode metrics at every eval; save/reload; the reduced-precision
smoke run (§5.4) before the first multi-node job; **export + parity re-pass after
training, per mode pairing**. *Exit: the Q-T2 recipe is chosen from data; health
bands defined; the loop survives a trained MoE checkpoint.*

### 8.4 Q-T2 — PoC heal (2–5B tokens) → MVS decision

Chosen recipe; ablations as budget allows (Q-A-route, β_aux on trigger, λ-ramp on
step-0 spikes, Q-A-g256 per damage map, Q-A-router-warm if early entropy shifted
sharply at T1). Deliver: PPL/KL_tf recovery vs dense and unhealed, router traces
incl. the zero-code tail, the a0/a1 trend, generations, exported artifact in the
validation runtime, §0.6 MVS verdict. Q-T1→Q-T2 tokens-to-recovery becomes the
forecasting basis for Q-T3's budget.

### 8.5 Q-T3 — production heal (50–150B tokens; working target ~100B, sized by Q-T2)

Scale tokens and context; downstream evals (standard zero-shot suite via
lm-eval-harness + a fixed generation smoke set — only after LM loss is healthy);
**mid-heal exports to the target CPU every ~20B tokens** (tok/s, TTFT, and quality
re-measured together). Contingency order on shortfall: LR/KD escalations (retune
grid, α/T sweep, λ-ramp) → group-256 scales → shared expert (§3.5) → per-tensor
mixed-precision fallback via `--tensor-type` (dilutes the claim; reported as such).

### 8.6 Q-T3.4 / Q-T3.5 — on-policy distillation (optional polish; inlined summary)

The teacher-forced heal never trains the student on states it visits when
*generating*, where quantization errors compound (exposure bias). OPD closes this:
the student samples, the teacher scores every token, the student minimizes per-token
reverse KL on its own state distribution. Discipline, self-contained:

- **Q-T3.4, measurement before training:** sample rollouts from the healed student on
  a frozen prompt set, score with the teacher, compute KL_op (on-policy KL), entropy,
  length/repetition/early-EOS stats; compare to KL_tf. Large gap → Q-T3.5 proceeds;
  small gap → shorten or skip. Zero weight updates; OPD earns its compute or doesn't
  run. **No OPD code is written before Q-T2 succeeds.**
- **Q-T3.5:** reverse KL over a support set of **student-top-k ∪ teacher-top-k ∪ the
  sampled token** (teacher-top-k alone misses exactly the tokens where the student
  errs — the tokens OPD exists to correct), chunked, explicit tail policy. Rollouts
  through fake-quant `generate()` (BitLinear stateless; routing is just compute).
  Trainer: an HF-ecosystem GKD-style trainer, version-pinned and subclassed for the
  chunked support-set loss; heavier RL-framework machinery only if OPD proves itself
  AND rollout throughput is the measured bottleneck. Exit: gap toward the floor AND
  generation evals improve **at fixed KL_tf** — improving free-running behavior while
  degrading teacher-forced parity is overfitting the teacher's modes. Monitor rollout
  entropy, distinct-n, early-EOS, length distribution (reverse KL is mode-seeking).
- MoE footnote: a rollout engine that changes routing numerics changes the policy; if
  rollouts ever move off the training forward, log KL(training‖rollout) per sync.
- SFT/DPO after, if at all; experts stay fake-quant throughout (§3.8).

### 8.7 Q-T4 — FP8 conversion + final export

Cast attention/embed to FP8 (per-tensor scales recorded); **run Q-A-head8** and set
the head's precision from its result; router/norms BF16; measure mode b − mode a1
against the pre-declared tolerance → pass, or trigger Q-A-fp8 + heal increment. Pack
experts per the K-track's chosen format under the §7.3 contract. Ship with the §0.5
card; speed claims measured per context length (decode AND TTFT) on the named CPU,
eval mode and runtime stated per number.

---

## 9. Ablation matrix

| ID | Description | Purpose | Gate |
|---|---|---|---|
| Q-A0 | dense continued pretrain (LoRA-scale control) | data/loop control | with Q-T1 |
| Q-A1 | experts-only ternary + A8 | **baseline** | — |
| Q-A1w | experts ternary, FP activations (trained W-only) | contests baseline **only if** llama.cpp is promoted to deployment (§7.5 trigger) | Q-K0 outcome |
| Q-A-g256 | group-256 absmean weight scales | quality vs per-tensor, TQ2_0-aligned | Q-T0 damage map |
| Q-A-route | + router-distillation KL | does constraining routing help? | Q-T2 budget |
| Q-A-router-warm ★ | freeze router for the first warmup window, then unfreeze at 0.1× | prevent the router routing AROUND damage before experts heal | run if Q-T1 shows sharp early entropy shifts |
| Q-A-aux | β_aux sweep | collapse response curve | §6.2 trigger only |
| Q-A-unfreeze | attention/embeddings at 0.1× | frozen-vs-unfrozen | Q-T1, before locking |
| Q-A-decay | decay-mask off (control) | demonstrate the §4.3 erosion — **tiny and bounded; a pathology demo, never a variant** | Q-T1, optional |
| Q-A3 | λ-ramp on expert quant (`w_eff=(1−λ)w+λ·quant(w)`, one flag) | stability | step-0 spikes only |
| Q-A4 | SubLN in experts (custom-runtime-only) | variance scouting | Q-T2 shortfall + damage map |
| Q-A-attn | also ternarize attention | decode-bytes ceiling | Q-T2 quality margin |
| Q-A-fp8 | FP8 fake-quant folded into QAT | §3.3 fallback | Q-T4 aggregate-delta failure |
| Q-A-head8 | lm_head FP8 vs Q8_0 vs BF16, measured alone | the logit-sensitive tensor decides its own precision | Q-T4, always run |
| Q-A-shared | add BF16 shared expert | the big lever, big cost | §3.5 rule |

## 10. Risks

- **Router collapse** — the signature MoE failure; §6.2 panel + pre-registered rules;
  aux loss never off.
- **Cold-expert decay erosion** — §4.3; mask mandatory, CI-tested under real FSDP,
  decay zeroed until green; zero-code-tail alarm; Q-A-decay demonstrates it once.
- **Cold experts under-heal (drift, not decay)** — flip×utilization monitor; accepted
  if utilization-consistent, escalated (β_aux, data mix) if utilization is itself the
  pathology.
- **Parity-mode confusion** — the structural fix is §7.5's three-mode matrix +
  continuous a0/a1 tracking; every reported number names its mode and runtime.
- **Validation/deployment objective divergence** — the §6.3 trend alarm.
- **Granularity/parity mismatch** — §3.7 rule; Q-T0 reconciles docs against code.
- **Small-expert ternary floor** — the founding tension; measured by the damage map
  and Q-T2 curve; contingencies §8.5; structural fixes out of scope (§3.6).
- **Prefill/TTFT disappointment** — union routing makes MoE prefill bandwidth-heavy
  too; measured FIRST at Q-K0 with a pre-declared tolerance; scoped and reported.
- **Custom-runtime opportunity cost** — must beat the llama.cpp floor by the §8.2
  margin; the floor sufficing is a good outcome, not a defeat.
- **15B training memory** — 4-way FSDP planning floor; reduced-precision stack
  smoke-tested (§5.4) before multi-node spend.
- **Teacher cost** — ~30 GB no-grad; top-k caching default; frozen-corpus cache
  discipline; manifest hashes.
- **Chunked-loss omission** — full (T, 152K) logits are a day-one OOM; chunking is a
  design requirement with its own equivalence test in CI.
- **Base checkpoint provenance/license** — pinned at Q-T0 or the track stops.
- **GGUF/FP8 confusion** — no FP8 in GGUF; the §7.5 matrix keeps Q8_0-vs-FP8 deltas
  measured, never mysterious.
- **Fast-moving OPD tooling** — pin versions; re-check trainer docs at Q-T3.4
  kickoff; this document's snapshot is not the implementation reference.

## 11. Source map

- **MoTE (Mixture of Ternary Experts)** — the precision-split precedent: ternary
  experts from dense checkpoints, high-precision router.
- **BitNet / BitNet b1.58 line** — the quantizers, STE training, latent-weight
  recipe, large-LR tolerance, S-curve loss behavior.
- **The quantize-and-heal lineage (e.g. Llama3-8B-1.58)** — heal-from-pretrained
  precedent; the too-small-LR-fails-to-recover finding; step-0 full quantization.
- **GKD / on-policy distillation literature** — §8.6's algorithm, support-set
  estimator, and failure modes (entropy collapse, support mismatch, staleness).
- **Qwen3 technical report + HF Qwen3-MoE modeling code** — architecture ground
  truth (128-expert MoE, QK-Norm, no shared expert, aux-loss convention).
- **CPU-kernel literature** — bitnet.cpp (ternary CPU kernels, dense-2B precedent
  numbers), T-SAR (in-register LUT GEMV), FairyFuse (fused AVX-512 ternary loops),
  KTransformers (sparse-on-cheap-DRAM).
- **mainline llama.cpp** — qwen3moe architecture, TQ1_0/TQ2_0 ternary types,
  `--tensor-type` per-tensor overrides, F16 block scales.

---

## 12. The recipe on one page

```text
DOCTRINE  Two projects, kept apart: the HEAL (validated in llama.cpp; MVS lives
          here) and the ENGINE (K-track; may fail without killing the artifact).

GATES     Q-T0 (converts/exports; quant.py reconciled; damage mapped) and Q-K0
          (decode, TTFT, beat-the-floor margin — all pre-declared) before training;
          Q-T1 doubles as the stack shakedown; Q-T2+ requires all three.

PRECISION Ternarize experts ONLY (93% of storage; ~54% of active decode compute,
          T0-computed, in the card). Router + norms BF16 forever; never quantize
          the router; watch the gate/gate_proj regex. Attention/embed BF16 through
          training → FP8 at export, measured; the lm_head decides its own precision
          (Q-A-head8). Per-tensor scales baseline; group scales only export-aligned.

HEAL      CE + chunked KD from the original MoE teacher (top-k cache) + the model's
          own aux loss, always on. Experts QAT at the high grid; router 0.1×;
          attention/embeddings frozen (ablated once). Weight decay COLD-MASKED —
          and ZERO until the mask's CI test is green under real FSDP. 4-way FSDP
          floor; reduced-precision stack smoke-tested first. Batches per-EXPERT
          (tokens/16). uint32 shards, frozen corpus, hashed provenance. QAT always
          on — a0 is an eval mode, never a training phase.

WATCH     Routing entropy first — collapse shows there before PPL. Zero-code cold
          tail = decay erosion. Flip-vs-utilization = the cold scatter. Top-8
          teacher agreement PER LAYER (depth gradient expected). a0 AND a1 at every
          eval — alarm on divergence TREND, not the gap's existence.

VERIFY    Baked ternary; exact codes where the route preserves them, bounded
          mismatch where it doesn't (determined at Q-T0); F16 scale bound. Three
          modes, one model: a0 W-only ↔ llama.cpp, a1 W2A8 ↔ custom runtime
          (activation contract binds it), b +FP8 ↔ final. FP8 delta = b − a1.
          Q-A1w trains only if the floor becomes the deployment target. Tiny random
          MoE + decay-mask test in CI from the first commit.

DEPLOY    Batch-1 CPU decode ~1 GB/token → tens of tok/s on DDR5. Prefill reads
          ~the whole pack (union routing) — TTFT measured first, per prompt length.
          Scalar-oracle kernels, roofline discipline, gather inside the kernel.
          Mid-heal checkpoints on real hardware every ~20B tokens.

SCOPE     No SubLN, no shared expert, no structural edits, no router quant — each
          has a numbered ablation and a pre-registered gate if reality disagrees.
```