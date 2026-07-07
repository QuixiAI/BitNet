# 1.58-Bit Healing Program — Unified Plan v5

Status: v5.1, 2026-07-06. This document supersedes and consolidates: the v4 core plan, the
v4.1 multi-track supplement (2B/3B/4B), the v4.2 on-policy-distillation supplement, and
all accepted review feedback. It is the single source of truth; earlier documents are
history.

**What v5 consolidates and adds:**

```text
From v4:        quantize-then-heal recipe, BitNet math, KD-by-default, parity gates,
                milestones, kernel plan, all architecture decisions
From v4.1:      tracks A3 (Llama-3.2-3B), G2 (Gemma-4-E2B), G4 (Gemma-4-E4B),
                arch profiles, TQ2_0 upstream export route, per-track compute plans
From v4.2:      OPD polish phase (T3.5), on-policy gap metric, TRL/verl/Tinker decision
OPD review:     T3.4 measurement gate; E0–E4 estimator variants with E3 (student ∪
                teacher support) as baseline; TRL GKD labeled experimental+pinned;
                OPD excluded from MVS; "OPD competes with SFT" (I0–I3); stricter verl
                gate; generation-collapse monitors
New in v5:   ★  runtime eval-mode matrix — I2_S is W2A8 but upstream TQ2_0 is
                weight-only, so parity must compare like with like (§8.4)
             ★  A2w ablation: weight-only training variant for TQ2_0-only targets
             ★  CI micro-loop: full convert→train→export→parity on a random tiny
                model, minutes, every commit (§7.3)
             ★  heal scaling study: A1-vs-A3 tokens-to-recovery as the forecasting
                basis for G4/production budgets (§11.6)
             ★  release checklist / model-card template (§13.3)
v5.1 (review): A1-only critical path stated up front; baked-ternary "exact codes"
                softened to a T0-verified intent; A2-vs-A2w made a formal
                deployment-dependent decision (G-track baseline contested at T2);
                OPD implementation forbidden before A1 T2; Track B moved out of the
                operational track map; first-sprint build order added (§7.0)
```

**THE CRITICAL PATH — read before anything else:**

```text
A1 T0  →  A1 T1  →  A1 T2 / MVS
```

Everything else in this document waits on that line: A3 waits, Gemma waits, OPD waits,
kernels wait, SFT/DPO waits. The breadth below is a program map, not a to-do list for
the first sprint (§7.0). If a piece of work does not advance A1 toward MVS, it is not
next.

---

## 0. Goal, framing, and success

### 0.1 One paragraph

Take a strong full-precision base model, replace every transformer-block linear layer
with a BitLinear (ternary forward weights `{-1,0,+1}` + per-token int8 activations,
trained through a straight-through estimator over full-precision latent weights),
continue pretraining — with the original dense model as a distillation teacher — until
quality recovers, then optionally polish with on-policy distillation. Export through a
ternary GGUF route and validate in the target runtime. The deliverable is not just the
models: it is a reusable BitLinear training stack (convert → heal → polish → export →
validate) that treats BitNet layers as an architectural component.

### 0.2 Naming and stakeholder language

Artifacts are named `<Base>-1.58`: `Llama-3.2-1B-1.58`, `Llama-3.2-3B-1.58`,
`Gemma-4-E2B-1.58`, `Gemma-4-E4B-1.58` — *Llama/Gemma-shaped 1.58-bit models* in the
lineage of `HF1BitLLM/Llama3-8B-1.58-100B-tokens`. **None is a reproduction of
Microsoft's BitNet b1.58 2B4T** (different architectures, FFNs, norms, and a 4T-token
from-scratch regime we are deliberately not attempting). Say this in every external
summary. The canonical Microsoft-shaped architecture (SubLN, squared-ReLU FFN, BitNet
GGUF arch) is **Track B** — future work that reuses this stack.

### 0.3 Minimum viable success (per track; decided before production spend)

```text
A converted <Base>-1.58 model:
  1. trains for at least 100M tokens without numerical failure,
  2. improves validation PPL over the unhealed fake-quant baseline —
     measured in PyTorch AND in the exported GGUF,
  3. exports through its designated route and runs in its designated runtime,
  4. matches PyTorch PPL within the predeclared tolerance under the matched
     eval mode (§8.4),
  5. shows decreasing KL-to-teacher on the calibration set.

Explicitly NOT part of MVS: on-policy distillation (T3.4/T3.5 are additive polish),
SFT/DPO, kernel acceleration, downstream benchmark targets.
```

Clause 2's "in the exported GGUF" is deliberate: "healed in PyTorch, broken through
export" is the failure mode the parity apparatus exists to catch. The PPL bar itself is
low from a severely damaged start — it gates the *loop*; quality bars live at T2/T3.

---

## 1. Strategy and track map

### 1.1 Quantize-then-heal

| | Heal from a pretrained base (chosen) | Native 1-bit from scratch |
|---|---|---|
| Starting point | strong pretrained model | random init |
| Token budget | 10M (smoke) → 1–3B (PoC) → 10–150B (prod) | multi-trillion for best results |
| Compute risk | low; single node viable for PoC | high |
| Quality ceiling | somewhat below native-trained 1-bit | highest |
| What it proves | the whole training/export stack | mostly the same stack, expensively |

**Expectation, measured at T0, not asserted:** the unhealed converted model will be
severely damaged; PPL in the hundreds or worse would not be surprising. If T0 instead
shows mild damage, that is good news (fewer heal tokens needed), not a plan failure.
Either way, healing behaves dynamically more like early pretraining than fine-tuning —
this shapes the LR sweep (§5.3), monitor tolerances (§10.4), and T0 framing (§11.1).

### 1.2 Tracks

| Track | Base model | Params (eff/total) | Tokenizer (vocab) | Export route | Runtime |
|---|---|---|---|---|---|
| **A1** (pathfinder) | meta-llama/Llama-3.2-1B | 1.24B | LLaMA-3 (128,256) | I2_S primary, TQ2_0 secondary | bitnet.cpp / mainline llama.cpp |
| **A3** | meta-llama/Llama-3.2-3B | ~3.21B | LLaMA-3 (128,256) | same as A1 | same |
| **G2** | google/gemma-4-E2B | 2.3B / 5.1B | Gemma (262,144) | **TQ2_0 only** | mainline llama.cpp |
| **G4** | google/gemma-4-E4B | 4.5B / 8B | Gemma (262,144) | same as G2 | same |

Track B (the Microsoft-canonical b1.58 architecture) is deliberately **not** in the
operational map — it is roadmap context (§0.2, §14) until someone actively plans it.
The working program is A1/A3/G2/G4.

### 1.3 Sequencing and gates

```text
A1 ──reaches MVS──► A3 (lowest-risk scale-up: same everything, bigger)
               └──► G-T0 export spike ──passes──► G2 ──reaches MVS──► G4
OPD (T3.4/T3.5) per track, only after that track's T2, never on the critical path.
```

- A3 needs no new feasibility work, only compute; it starts once A1 passes MVS.
- G2 is gated on the **G-T0 export spike** (§11.2) *before any Gemma training code is
  written*. G4 is G2 scaled and starts only after G2's MVS.
- Milestones are written per-track as **T0–T4** (instantiating one template, §11), with
  T0.5/T3.4/T3.5 as optional gated phases.

---

## 2. Target model specifications

### 2.1 The arch-profile mechanism (how one stack serves four models)

The stack wraps the HF model and swaps `nn.Linear` modules; everything else (attention
variants, RoPE flavors, norm conventions) lives in HF modeling code and is invisible to
the swap. Per-track behavior is data, not code — an **arch profile** consumed by a
generic `conversion.py`:

```text
arch_profile:
  base_model:             HF id
  target_linear_regexes:  which nn.Linear modules become BitLinear
  keep_fp_list:           embeddings, PLE tables, all norms, heads, 1-D tensors
  freeze_fp_params:       bool (§5.4)
  tokenizer / shard_set:  which prepared corpus this track reads
  teacher:                HF id (default: the track's own dense base)
  export_route:           i2s_bitnet_cpp | tq2_upstream | both
  eval_modes:             which §8.4 modes this track's parity uses
  seq_len_schedule, lr_grid, token_budgets
```

**Rule: enumerate, don't assume.** Each profile's target list is produced by walking the
actual module tree at that track's T0 and classifying every `nn.Linear` — never copied
from another track. The A1 profile is first extracted from the existing behavior as a
pure refactor verified by the existing tests.

### 2.2 Track A1 — Llama-3.2-1B (pathfinder)

| Field | Value | Decision |
|---|---|---|
| params | ~1.24B (tied embeddings) | accepted |
| hidden / layers | 2048 / 16 | unchanged |
| heads / KV heads | 32 / 8 (GQA), head_dim 64 | unchanged |
| intermediate | 8192, SwiGLU (SiLU gate) | unchanged — §3.2 |
| vocab / rope_theta | 128,256 / 500000 | unchanged |
| **rope_scaling** | llama3 type, factor 32 | **must survive conversion** |
| norms / tying | Llama RMSNorm placement / tied | unchanged |

Ternarizes: `q,k,v,o_proj, gate,up,down_proj` per block. Stays FP: `embed_tokens`
(= tied head), every `*norm.weight`, 1-D tensors, RoPE buffers — matching what the
exporters refuse to ternary-pack (§8). FP embedding fraction ~21% of params: state it in
the track report so packed-size expectations are honest.

**Config preservation is a hard requirement** — verbatim copy plus a field-by-field diff
test (`rope_scaling` above all; silently dropping it changes long-context behavior and is
the kind of bug that survives until T3).

### 2.3 Track A3 — Llama-3.2-3B

The deliberately boring track: every A1 decision carries over unchanged; same tokenizer,
same shards, same export precedent class (Llama3-8B-1.58 brackets it from above).

| Field | Value |
|---|---|
| params | ~3.21B (tied) |
| hidden / layers | 3072 / 28 |
| heads / KV heads | 24 / 8 (GQA), head_dim 128 |
| intermediate / vocab / rope | 8192 SwiGLU / 128,256 / θ=500000, scaling factor 32 |

Deltas from A1, exhaustively: memory class (§5.4); rerun the LR grid at T1 rather than
assuming A1's winner transfers (prior: optimum at or slightly below A1's); FP embedding
fraction improves to ~12% — the 1.58-bit story is *stronger* at 3B; the T0 damage map is
re-measured (comparing 1B and 3B maps is itself informative, §11.6).

### 2.4 Tracks G2/G4 — Gemma-4-E2B / E4B

Known facts (model card / release material; anything not listed here is a **T0-recorded
fact, not an assumption**): E2B = 2.3B effective / 5.1B total, 35 layers; E4B = 4.5B /
8B, 42 layers; hybrid attention interleaving 512-token sliding-window local layers with
full global layers (final layer global); global layers use unified Keys/Values and
proportional RoPE; 262,144-token vocabulary; Per-Layer Embeddings (PLE — each decoder
layer has its own small per-token embedding lookup; hence total ≫ effective params);
multimodal with ~150M vision and ~300M audio encoders; 128K context; Apache-2.0;
requires transformers 5 (pin the training env per track).

To record at G-T0 (§11.2): hidden sizes, FFN type and activation, norm convention
((1+γ) style and placement), QK-norm presence, embedding scaling, weight tying, PLE
wiring and dtypes, the exact `nn.Linear` inventory. **No implementation may depend on a
Gemma internal this document does not list as verified; the G-T0 outputs are the sole
authority.**

**Conversion stance:**

- **Text-only extraction.** Drop the vision/audio encoders; the heal target is the
  language model, and llama.cpp's text path doesn't load them. Stated limitation:
  `Gemma-4-*-1.58` are **text-only** artifacts; multimodal alignment is expected to
  degrade and is not a deliverable.
- **Ternarize** the decoder-block attention/FFN linears per the T0 enumeration; hybrid
  attention and unified-KV change shapes, not the recipe.
- **Keep FP — a longer list than Llama's:** token embeddings, **all PLE tables**
  (embedding-like lookups; same rule, bigger consequence), every norm including
  QK-norms, 1-D tensors, and anything the T0 classification marks ambiguous (FP by
  default; ablate later, never guess now).
- **Keep the native FFN activation** — same reasoning as §3.2.
- **Honest accounting:** ~2.8B of E2B's 5.1B total is FP embeddings/PLE; the
  ternarizable fraction of *stored* params is well under half (the compute path
  ternarizes properly). Track reports quote ternarized-FLOPs fraction and packed size —
  "1.58-bit model" oversells the G tracks more than the A tracks.
- **Base-checkpoint option (T0 recon):** Google's Gemma-4 QAT variants (the mobile
  mixtures even carry 2-bit layers) have quantization-robust weights; if a *pretrained*
  (non-it) QAT checkpoint exists, starting the heal from it may cut t=0 damage. If only
  -it QAT exists, the tradeoff becomes a T2 ablation.

---

## 3. Architecture decisions (with reasoning, so they are not relitigated)

### 3.1 No added SubLN in any baseline

Two independent reasons: (1) **there is no function-preserving way to insert RMSNorm** —
at γ=1 it maps `x → x/rms(x)`, data-dependent, unfoldable into an adjacent linear;
inserting it adds damage on top of quantization damage at t=0. (2) **The Llama/Gemma
GGUF architectures have no tensor slots for sub-norms**; keeping the base-shaped export
target excludes them. SubLN is ablation A4 — trainable in PyTorch, not exportable, and
belongs to Track B.

**A-track T0 action item:** dump the released Llama3-8B-1.58 GGUF tensor list — the
ground truth for tensors, names, dtypes, and architecture string the I2_S converter and
the Eddie-Wang1120 llama.cpp fork accept. Matching it beats reasoning about it.

### 3.2 Keep each base's native gated-FFN activation

Pretrained gate/up/down weights are tied to their activation (SiLU for Llama; Gemma's
recorded at T0). Switching to squared-ReLU would invalidate the inherited FFN semantics
and forfeit the main advantage of healing. Squared-ReLU is Track B.

### 3.3 Function-preserving reparameterizations are allowed pre-conversion

Unlike inserting norms, *rescaling through* an existing norm is exactly
function-preserving in FP: `γ' = γ/s`, `W' = W·diag(s)` for any RMSNorm→Linear pair.
This is ablation A7 (SmoothQuant-style outlier smoothing): choose `s` to shift
activation-outlier magnitude into the weights, reducing t=0 quantization error before
any training. Applies only to norm-adjacent pairs (e.g. Llama's `input_layernorm →
q/k/v` and `post_attention_layernorm → gate/up`; `o_proj`/`down_proj` have no preceding
norm). Gated on the T0 damage map showing activation-dominant damage.

### 3.4 Quantization granularity

Per-tensor absmean ternary weights; per-token absmax int8 activations (§4). Packed
formats regroup scales at export; reconciling conventions is the parity gate's job
(§8), not a training concern.

---

## 4. BitNet math

Adopted verbatim from the agreeing reference implementations
(`.reference/bitnet/bitnet/model.py`, `gpu/convert_checkpoint.py`, QuixiCore
`tk.quant.quantize_bitnet`).

```python
def weight_quant(w):                       # per-tensor absmean ternary
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    return (w * scale).round().clamp(-1, 1) / scale

def activation_quant(x):                   # per-token absmax int8
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    return (x * scale).round().clamp(-128, 127) / scale

class BitLinear(nn.Linear):                # bias=False (Llama/Gemma projections have none)
    def forward(self, x):
        x_q = x + (activation_quant(x) - x).detach()                        # STE
        w_q = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return F.linear(x_q, w_q)
```

**STE ⇒ dense backward.** Autograd sees only `F.linear(x_q, w_q)`; quantizers are
identity in the backward graph:

```text
grad_x = grad_y @ w_q        # dense bf16/fp32 GEMM
grad_w = grad_y.T @ x_q      # dense GEMM, applied to the FP latent weight
```

No ternary backward kernel exists to build: `grad_y` is dense (a ternary×dense backward
GEMM optimizes the small operand for marginal gain), and `grad_w` has no ternary operand
at all — its output *must* be full precision because AdamW accumulates it into the
latent weight, and that accumulation of sub-threshold updates is the mechanism by which
healing works. This deletes the hardest kernel work from the plan (§12).

Latent weights and optimizer state stay high precision; ternary values are recomputed
every forward and stored only at export. **Caveat:** the argument holds because the
quantizer's backward is identity by construction. LSQ-style learned scales, soft
quantization, or a *learned* λ would re-enter the backward graph; the fixed λ-ramp
(ablation A3) does not. OPD (§6) changes the loss and data source, not the backward
math — the conclusion survives there too.

---

## 5. Training recipe (off-policy heal — the workhorse)

### 5.1 Loss: distillation from the dense teacher, by default

```text
L = CE(student, tokens) + α · T² · KL( softmax(teacher/T) ‖ softmax(student/T) )
```

Frozen dense base as teacher; starting point `α = 1.0`, `T = 2.0` (swept at T1/T2).
Healing's objective is *match the original function*; the KL term optimizes that
directly and is known to heal quantized models substantially faster per token than CE
alone — decisive at 1–3B-token PoC budgets. Cost: one extra no-grad teacher forward per
step, or none with cached top-k (below).

**Mandatory implementation requirement — chunked losses:**

```text
train.py MUST compute CE and KL over vocabulary chunks (or fused equivalents) such
that full (T, V) teacher and student logit tensors are never simultaneously
materialized. Day-one requirement, not an escape hatch. V = 128,256 (A) / 262,144 (G).
```

**Top-k teacher caching (A6c):** precomputed top-k teacher logits remove the teacher
from step memory — the **recommended default for G tracks** (262K vocab, ~10–15 GB
teachers). Two rules: (1) tail mass handled by an explicit config-recorded choice
(renormalize over top-k, or one "other" bucket) — the choice changes gradients;
(2) valid only over a frozen corpus/tokenization/windowing — pin the data first.

CE-only (A6a) and full-KL vs top-k (A6b vs A6c) are ablations so the KD gain and the
top-k approximation cost are measured, not assumed. Optional extension only if logit-KD
plateaus: intermediate-layer distillation (hidden-state MSE) — adds hook complexity, not
baseline.

### 5.2 Optimizer and regularization

- **AdamW**, betas (0.9, 0.95), eps 1e-8. Muon on ternary latents is unvalidated —
  deferred experiment, never baseline.
- **Gradient clipping 1.0.**
- **Weight decay 0.1** on 2-D BitLinear latent matrices only (none on norms, embeddings,
  1-D tensors); decay-to-zero at the midpoint for long runs, flat for short.

### 5.3 Learning rate: sweep wide at T1, decide from data

Two real, competing considerations: the Llama3-8B-1.58 precedent (a **CE-only** heal)
found too-small LR fails to recover — consistent with post-conversion training behaving
like early pretraining; but **KD changes the loss geometry** (dense smooth supervision
everywhere), so lower LRs may work where CE-only under-heals. The precedent does not
directly transfer to this recipe. Therefore:

```text
T1 grid, all primary, nothing prejudged: 3e-5, 1e-4, 2e-4, 4e-4
```

Watch beyond loss: at the top, re-damage of the init (diverging from the teacher); at
the bottom, a *frozen* model — detected mechanistically by the **code-flip rate**
(§10.2): near-zero ternary code flips means latents drift without crossing quantization
thresholds, i.e. the effective model is not training regardless of healthy-looking
optimizer steps. Sustained very high flips = thrashing. This metric turns the LR debate
into a measurement. Schedule: linear warmup (100–500 steps short runs, 1k–2k long) →
cosine to ~10% of peak. Per track: rerun the grid at T1 scale; do not inherit winners.

### 5.4 Precision and memory, per track

Correctness baseline (A1 only): **fp32 latent weights, fp32 Adam moments, bf16 autocast
forward.** PyTorch does not maintain fp32 masters for bf16 params unless you build it.

| | A1 | A3 | G2 | G4 |
|---|---|---|---|---|
| trainable params (see freeze note) | 1.24B | 3.21B | ~2B-class* | ~4.5B-class* |
| W+G+Adam @ fp32, 16 B/param | ~20 GB | ~51 GB | ~32 GB* | ~72 GB* |
| teacher, bf16, no-grad | ~2.5 GB | ~6.4 GB | ~9–10 GB | ~15 GB |
| plan | fp32 baseline + hatches | 80 GB: as-is; else hatches day-one | hatches day-one | multi-GPU/FSDP, no fp32 option |

\* assumes the G-track default: **freeze FP embeddings and PLE tables** — quantization
did not damage them, the teacher shares them exactly, and unfreezing puts Adam state on
~2.8B extra params for marginal benefit. This differs from A-tracks (tied embeddings
stay trainable), so it gets a T1 frozen-vs-unfrozen ablation before locking. Exact
G-track numbers are measured at T0 from the real config.

Always-on from day one, every track: **gradient checkpointing** and **chunked losses**.
Remaining hatches in pull order: 8-bit optimizer states; then bf16-latents + explicit
fp32 masters — adopted only after A1 has produced a healthy fp32 T1 run to compare
against. A3/G2/G4 promote hatches to defaults (the recipe having been validated at A1
in fp32 means nothing is being *proven* in reduced precision).

### 5.5 Data

- **Tokenizer per family, non-negotiable** (we reuse each base's embedding table):
  LLaMA-3 tokenizer for A-tracks, Gemma tokenizer for G-tracks. Existing corpora with
  other tokenizers (e.g. the ~1B-token AUM shards, SmolLM2 vocab) are **re-tokenized**,
  never remapped.
- **Shard dtype: uint32.** Neither 128,256 nor 262,144 fits uint16; a uint16 pipeline
  silently wraps half the vocab and trains on garbage.
- **Format:** flat EOS-separated token shards + `manifest.json`, non-overlapping windows
  via lazy `np.memmap`, window-level shuffle, `drop_last=True` (AUM's `PackedWindows`,
  dtype-widened).
- **Freeze early:** each family's PoC corpus, tokenization, and windowing are pinned at
  T1 start — for run comparability and because A6c caches are invalid if data moves.
  Manifest hash goes into checkpoint metadata (§5.6).
- **Sharing:** A1/A3 share the LLaMA-3 shard set verbatim; G2/G4 share one Gemma set.
  Teacher caches share within a family, never across.
- **Sequence length:** 1024–2048 (T1) → 2048 (T2) → 2048→4096 (T3).
- **Corpus:** match each base's pretraining distribution as practical — heal, don't
  domain-shift. General/educational web, wiki, code; Gemma is heavily multilingual, so a
  pure-English G corpus is a mild domain shift (note it in the track report). PoC 10M–3B
  tokens; production 10–150B — sourcing at that scale is the main data lift; start
  during A1's T1, and size it using the §11.6 scaling study.

### 5.6 Harness, provenance, starter config

Accelerate DDP (`find_unused_parameters=False`); `save_pretrained` HF checkpoints
(required by exporters) + `trainer_state.pt` + `latest` symlink + `--resume`; wandb;
HealthMonitor (§10.4). **Checkpoint metadata must record:** quantizer version/hash, RNG
seeds, config hash, data-manifest hash — parity reports and ablation comparisons are
only meaningful with provenance pinned.

A1 T1 starting point (so the first run needs no design meeting):

```text
seq_len 1024 · micro_batch 8 · grad_accum 8 · ≈65k tokens/step
warmup 200 · cosine to 10% · eval every 200 steps on the fixed calibration set
bf16 autocast · fp32 latents/Adam · grad checkpointing on · chunked CE/KL on
```

Scale tokens/step toward 0.25–0.5M for T2/T3 via accumulation and devices, not design
changes.

---

## 6. On-policy distillation (OPD) — the polish phase

### 6.1 What OPD adds that the §5 heal cannot

The §5 heal is off-policy: teacher-forced forward-KL on data-distribution states. The
student is never trained on the states *it* visits when generating, where
quantization-induced errors compound token by token — so it can reach teacher-forced
parity while still degrading over its own generations (exposure bias). OPD closes this:
the **student samples**, the **teacher scores every token** of those samples, and the
student minimizes per-token **reverse KL on its own state distribution** — dense
token-level supervision at inference-time states, no reward model. With the teacher
being the exact dense model we converted from, OPD is the most literal implementation of
"match the original where the quantized model actually goes."

Directionality: §5's KD is forward KL on data states (mass-covering); OPD is reverse KL
on student states (mode-seeking) — what a heal wants, at the known cost of some entropy
reduction, monitored (§10.3).

### 6.2 Placement, gates, and what OPD is not

```text
Phase 1 — off-policy heal (T1–T3):            the workhorse. UNCHANGED by this section.
Phase 2 — T3.4 measurement, then T3.5 OPD:    additive polish, per gates below.
Phase 3 — instruction variants (§6.6):        OPD competes with SFT; measured, not assumed.
```

**Hard rules:** no OPD before the track's T2; no OPD before coherent generation; OPD is
never part of MVS; OPD never touches the critical path (convert → heal → export →
validate comes first). **Operationally: no OPD code is written — no `train/opd.py`, no
GKD subclass — until A1's T2 has succeeded.** The OPD literature is still working out
staleness, estimator, and reliability issues (asynchronous staleness and finite
teacher-score caches are active problem areas for reverse-KL OPD); engineering time
spent there before the basic heal has proven value is misallocated.

**★ T3.4 — measurement before training (new, gated entry).** Before any OPD weight
update: sample rollouts from the healed student on the frozen prompt set, score with the
teacher, and compute KL_op, entropy, length/repetition stats, early-EOS rate; compare to
KL_tf. Cost: a few GPU-hours, zero updates. Decision rule:

```text
large on-policy gap (KL_op ≫ KL_tf)  → T3.5 proceeds, budget per O4
small gap                            → shorten or skip T3.5 for this track; record it
```

OPD earns its compute or doesn't run. **T3.5 exit:** gap reduced toward the measurement
floor AND generation evals improve **at fixed KL_tf/PPL** — improving free-running
behavior while damaging teacher-forced parity is overfitting the teacher's modes, not
healing. Budget: start 5–15% of heal compute; O4 measures the curve.

### 6.3 The algorithm, pinned

```text
Given prompt x from the frozen prompt set:
  sample y ~ student(·|x), temperature 1.0, up to L tokens
  at every position t: per-token loss = D_KL( p_S(·|x,y<t) ‖ p_T(·|x,y<t) )
  no reward model; no discounting; no reference-policy KL
optionally mix fraction λ of on-policy batches with 1−λ teacher-forced (GKD's λ);
  sweep λ ∈ {0.5, 0.75, 1.0} (O3)
```

**★ 6.3.1 Estimator variants — and why teacher-top-k alone is wrong.** Reverse KL is an
expectation over the **student's** distribution: `Σ p_S · log(p_S/p_T)`. A sparse
estimator supported only on the *teacher's* top-k can miss exactly the tokens where the
student puts mass and the teacher doesn't — which are precisely the error tokens OPD
exists to correct — biasing the loss toward "match the teacher's favorite modes" and away
from "stop doing the wrong thing you currently do."

| ID | Estimator | Status |
|---|---|---|
| E0 | full-vocab reverse KL | small-vocab unit tests only (memory) |
| E1 | sampled-token (score-function/REINFORCE-style) | not baseline — high gradient variance; the control-variate literature exists because of this |
| E2 | teacher-top-k sparse KL | **rejected as baseline** (support bias above); note verl's stock async recipe is E2-shaped |
| E3 | support S = student-top-k ∪ teacher-top-k ∪ sampled token; truncated/renormalized reverse KL on S, explicit config-recorded tail policy | **baseline** |
| E4 | E3 + teachability/high-disagreement token filtering | future ablation |

E3 is nearly free: the student's full distribution is already materialized locally
in-process — the sparsity constraint only ever applied to *teacher* communication — so
E3 costs merely gathering teacher logprobs at the student's top-k indices too. E1 vs
E2/E3 taxonomy note: E1 differentiates through the sampling (high variance); E2/E3 are
directly differentiable distribution matching at visited states — the GKD-style path is
naturally E3-shaped.

Inherited mandates, both binding here: **chunked/top-k distributions** (full (T,V)
tensors never materialize; §5.1's rule extends to the OPD loss, with its own
small-vocab equivalence test) and **rollout policy = training policy** (sampling goes
through the fake-quant forward; KV-cache generation works unmodified — BitLinear is
stateless — and any substitution is measured and corrected, §6.5).

### 6.4 Framework: TRL first (experimental, pinned), verl only after OPD proves itself

**TRL GKD — the T3.5 baseline.** Our student *is* an HF model (wrapped base with swapped
linears): it loads, `generate()`s, and trains like any `AutoModelForCausalLM`, so TRL's
GKD trainer (student, teacher, λ, β, temperature; Accelerate; our hardware) is the
minimal-new-code path. **Label it what it is:** the implementation lives in TRL's
*experimental* area — pin the exact version, treat stock behavior as a starting point,
and subclass regardless, because two adaptations are preconditions, not optimizations:
(1) the **E3 chunked support-set loss** (stock GKD materializes full logits —
disqualifying at 262K vocab); (2) **β pinned to the reverse-KL end** for the heal
use-case. Prompt sets: base-model polish uses document prefixes from the frozen heal
corpus; instruction OPD uses instruction prompts with the family's chat template. Known
cost: fake-quant `generate()` is slow — acceptable at polish scale, and it makes rollout
throughput a second customer for the T4 forward kernels.

**Tinker — the recipe, not the platform.** The Tinker cookbook's OPD recipe (from the
originating Thinking Machines work) is the reference specification §6.3 is pinned
against, and the hosted service is fine for quick dense-proxy prototyping of OPD
hyperparameters. It cannot train a custom BitLinear architecture with an STE loop,
custom optimizer partitioning, and our checkpoint/export formats — not a substrate for
any 1.58-bit student in this plan.

**verl — the scale path, strictly gated.** verl carries OPD as a first-class algorithm
(config note that matters: disable the PPO/GRPO reference-policy KL, or the student is
simultaneously regularized toward a reference and distilled toward the teacher) and an
async on-policy KD recipe (teacher returns top-k, token-wise sparse KL, overlapped
generation/teacher/update stages). **Adoption gate — all three required:**

```text
1. TRL-based OPD has demonstrably improved A1 in the O-series, AND
2. OPD is a material fraction of program compute, AND
3. rollout throughput is the measured limiting bottleneck.
```

Below that bar, verl adds Ray, a serving stack, weight-sync hooks, and operational
surface before OPD has proven it helps a BitLinear heal. **Adoption checklist items when
the gate opens:** extend verl's E2-shaped support set to E3 (the stock recipe inherits
§6.3.1's critique); and implement the rollout bridge of §6.5.

### 6.5 The verl rollout bridge (recorded now, built only after the gate)

verl's rollout backends (vLLM/SGLang) cannot serve fake-quant BitLinear. Bridge: at each
weight sync, bake `w_q = weight_quant(w_latent)` and push bf16 `{-s,0,+s}` weights to
the rollout engine — weight-side exact; but the engine does **not** apply per-token int8
activation quant, so rollout policy ≈ training policy, not ==. Pre-registered handling:
(a) measure — log KL(training ‖ rollout policy) on a fixed prompt set at every sync,
alarmed on drift; (b) rely on the truncated importance-sampling correction these
pipelines already apply (the async recipe explicitly trades strict on-policyness for
throughput; our mismatch is one more small off-policy term); (c) if the measured gap is
non-trivial: fall back to exact HF-generate rollouts, or implement activation quant in
the rollout engine (real work, only if justified).

### 6.6 Instruction variants: OPD competes with SFT — it does not replace it yet

SFT on curated responses is cheap, stable, and well-understood; OPD from an instruction
teacher is more expensive and mode-seeking. It *may* be better; delete nothing until
measured:

| ID | Curriculum | Note |
|---|---|---|
| I0 | SFT only | control (v4's SFT settings: chat template, small LR, summed-vs-mean token-loss test) |
| I1 | SFT → OPD polish | prior favorite: SFT injects the response distribution cheaply, OPD fixes free-running behavior |
| I2 | OPD directly from -it teacher | the pure-OPD hypothesis |
| I3 | SFT → OPD → optional DPO | full stack |

Run I0–I2 on A1 after its T3.5; the winner sets the instruction default for other
tracks. Teachers: the same-family instruction-tuned dense model; for Gemma, distill with
the teacher in the thinking-mode the student is meant to serve, mode recorded in config.
DPO remains optional, after, short and conservative.

### 6.7 Teachers under OPD

Default: the track's own dense base. **Cross-size within a tokenizer family** (Llama-3B →
1B student, Gemma-E4B → E2B student) is *more* natural on-policy — the teacher only
scores states, never has to have generated the tokens — and runs as ablation O2.
Cross-tokenizer OPD is open research, out of scope. Teacher serving at T3.5 scale: a
no-grad HF model or a small endpoint returning logprobs at the E3 support set. A6c
*caching* does not apply on-policy (states are novel by construction) — top-k *querying*
replaces top-k *caching* in this phase.

---

## 7. Repo layout, tests, and CI

### 7.0 ★ The first sprint (build order for A1 T0)

The ten files that exist before anything else, in order:

```text
 1. bitnet_train/quant.py                  # §4 formulas
 2. bitnet_train/bitlinear.py              # BitLinear + STE
 3. bitnet_train/conversion.py             # profile-driven module swap
 4. train/profiles/a1.yaml                 # the A1 arch profile
 5. train/init_from_base.py                # convert + T0 validation + damage map
 6. train/eval_ppl.py                      # calibration PPL, per eval mode
 7. bitnet_train/export/export_gguf.py     # bake + route converter + quantize
 8. bitnet_train/export/compare_gguf.py    # parity report
 9. tests/test_conversion_shapes.py
10. tests/test_pytorch_vs_gguf_ppl.py
```

That set **is** A1 T0. Only after T0 passes: `train.py`, `distill.py`, code-flip
metrics, the CI micro-loop. Only after A1 T2: the OPD package (§6.2). Everything else
in §7.1 is a map of the finished state, not the sprint plan.

### 7.1 Layout

```text
train/
  init_from_base.py             # convert + T0 validation + damage decomposition
  prepare_data.py               # HF datasets -> uint32 EOS-separated shards + manifest
  train.py                      # shards -> Accelerate -> AdamW (+chunked CE/KD) -> cosine
  opd.py                        # T3.4 + T3.5 driver — NOT built until A1 T2 (§6.2)
  eval_ppl.py                   # fixed calibration PPL, per eval mode (§8.4)
  launch.sh
  configs/<track>/{t0,t1,t2,t3,t34,t35}.yaml
  profiles/{a1,a3,g2,g4}.yaml   # arch profiles (§2.1)

bitnet_train/
  quant.py                      # weight_quant / activation_quant / λ-ramp
  bitlinear.py                  # BitLinear + ternary stats + code-flip tracking
  conversion.py                 # generic module swap driven by arch profiles
  smoothing.py                  # A7 norm→linear scale folding (function-preserving)
  distill.py                    # teacher wrapper, chunked KD, top-k cache reader
  opd/                          # entire package deferred until A1 T2 (§6.2)
    gkd_chunked.py              # TRL GKD subclass: E3 support-set chunked reverse KL
    estimators.py               # E0–E4
    rollout_metrics.py          # KL_op, entropy, distinct-n, EOS/length stats
  export/
    export_gguf.py              # bake ternary -> HF ckpt -> route converter -> quantize
    compare_gguf.py             # parity report per route × eval mode
  kernels/                      # empty until T4; reference.py is the oracle

tests/  (see 7.2)
```

Python training code stays separate from the C++ inference trees. A-track export
prerequisite: `git submodule update --init --recursive` (the Eddie-Wang1120 fork is not
checked out by default), then build. G-track export prerequisite: a pinned mainline
llama.cpp build.

### 7.2 Tests

```text
test_quant.py                       test_bitlinear_forward.py
test_bitlinear_grad.py              test_conversion_shapes.py       # per profile
test_config_preservation.py        test_save_reload.py
test_smoothing_preserves_fn.py     test_chunked_losses_match_full.py
test_chunked_gkd_matches_full.py   # E3 vs E0 at toy vocab
test_estimator_support.py          # E3 support contains sampled + student-top-k tokens
test_export_tensor_parity.py       test_pytorch_vs_gguf_ppl.py      # per route × mode
```

### 7.3 ★ CI micro-loop (new)

A random-weight 2-layer, tiny-vocab model runs the **entire** pipeline in CI on every
commit: build profile → convert → train 10 steps (CE+KD, chunked) → save/reload → bake +
export → quantize → tensor parity → tiny-runtime PPL parity (and, once the OPD package exists
after A1 T2, a 5-step E3 OPD on 4 prompts). Minutes, no real models, no GPUs required. Rationale: the program's highest
compound risk is loop breakage discovered late (a converter change that breaks parity, a
loss refactor that breaks resume); the micro-loop converts those from T2-scale
discoveries into red CI. Real-model T0s remain the correctness gates; CI guards the
plumbing.

---

## 8. Train/export parity — the gate before any long run

Training uses a per-tensor scale; packed formats use their own row/group scale
conventions; runtimes differ in whether they quantize activations. All reconciled by
*measurement* at T0, never discovered as a T3 quality gap.

### 8.1 Export routes

| Route | Tooling | Tracks | Notes |
|---|---|---|---|
| **I2_S** | `utils/convert-hf-to-gguf-bitnet.py` → `llama-quantize ... I2_S 1` (Eddie-Wang1120 fork) | A1/A3 primary | the Llama-1.58 precedent path; `--token-embedding-type q6_k` optional |
| **TQ2_0** | mainline `convert_hf_to_gguf.py` → `llama-quantize ... TQ2_0` | A1/A3 secondary; **G2/G4 only route** | native ternary 2.06 bpw (TQ1_0 = 1.69 bpw size variant); per-tensor overrides via `--tensor-type` regex |

The fork predates Gemma-4 and will never learn it; porting either direction is C++ work
with no payoff over TQ2_0. **Stakeholder flag:** if "runs in bitnet.cpp specifically" is
a hard requirement, G tracks do not meet it — they meet "runs in mainline llama.cpp."
Decide which requirement is real before G2's T1.

### 8.2 Baseline export path: bake the ternary

The export checkpoint contains `w_baked = weight_quant(w_latent)` — dequantized ternary,
every value exactly `{-s, 0, +s}` — never raw latents. Baking is *intended* to make
exact code recovery achievable: a quantizer that sees clean `{-s, 0, +s}` input has no
reason to flip codes. But that is an intent, not a guarantee — a route with its own
block optimization, clipping, scale search, or codebook could still deviate — so **T0
verifies per route whether codes are actually preserved** (§8.3); if not, use the
bounded-mismatch regime or modify the export. Write `w_baked` in fp32 (or verify the bf16 round-trip of
the actual `{-s,0,+s}` values is exact) so dtype conversion cannot flip codes. Latents
never leave the training checkpoint format.

**F16 scale bound (both routes):** llama.cpp stores block scales in F16, so the
per-tensor scale `s` is rounded per block. Codes still match exactly from baked input;
the parity report asserts **exact code match + uniform dequant error bounded by F16
rounding of `s`** — anything beyond that bound fails. (Not hypothetical: the Gemma-4
QAT→llama.cpp conversions hit this scale-dtype mismatch in the wild.)

### 8.3 Conditional tensor-parity rule (regime determined at T0)

```text
T0 recon: read the route's quantize path once and determine the regime:
  preserve     — packing recovers codes from baked ternary  ⇒ EXACT code match required
  re-quantize  — exporter re-ternarizes from latent-like values ⇒ bounded mismatch
                 rate, every mismatch explained by scale convention; then switch the
                 pipeline to the baked path
Log per tensor: code mismatch rate, dequantized max/mean/relative error, quantizer hash.
```

### 8.4 ★ Runtime eval-mode matrix (new — compare like with like)

The two runtimes differ in *activation* handling: **I2_S is W2A8** (runtime quantizes
activations), while **upstream llama.cpp quantization is weight-only** — a TQ2_0 model
runs with FP activations. A single PyTorch fake-quant number therefore cannot be the
parity target for both. Define two PyTorch eval modes and pair them:

```text
PyTorch eval modes:   W+A8 (training-time forward)   |   W-only (activation quant off)
Parity pairing:       W+A8  ↔ I2_S runtime           |   W-only ↔ TQ2_0 runtime
```

Pre-registered decision rule (per pairing): tensor parity clean + PPL parity bad ⇒ the
divergence is activation-side ⇒ align training-time `activation_quant` with **that
runtime's** actual behavior. **T0 action items:** read the I2_S activation-quant kernel
once and record its convention; confirm the TQ2_0 path's weight-only nature in the
pinned build.

Corollary — **A2 vs A2w is a deployment-dependent decision, not a purity question.**
For tracks whose *only* runtime is weight-only TQ2_0, A8 activation fake-quant is
training damage the runtime never imposes. Formal rule:

```text
A-tracks: A2 (W+A8) remains baseline — I2_S (W2A8) is the primary runtime.
G-tracks: A2 and A2w BOTH run at T2; whichever wins W-only runtime PPL becomes the
          G-track baseline. (A2w = ternary weights, FP activations, trained and
          evaluated W-only.)
```

Do not let the Gemma tracks carry unnecessary activation-quant damage for
BitNet-faithfulness reasons; equally, keep A2 in the race because it preserves the
W2A8-runtime option.

### 8.5 Model-level parity and the acceptance loop

Identical prompts through the PyTorch model (matched eval mode) and the GGUF: logits
where accessible, calibration PPL always; tolerance predeclared before T2; the parity
suite re-runs **after training** (T1 exit criterion), not just on the untrained convert.
Hard acceptance test before any production run: a few-steps-trained checkpoint can
save → reload → bake+export → quantize → run → match PPL. If this loop fails, do not
scale.

---

## 9. Ablation matrices

### 9.1 A-series (off-policy heal)

| ID | Description | Purpose | Exportable? |
|---|---|---|---|
| A0 | dense continued pretraining | data/loop control | n/a |
| A1d | ternary W only, FP activations — eval-only | isolate weight damage (T0 damage map) | diagnostic |
| A1b | A8 only, dense W — eval-only | isolate activation damage (T0 damage map) | diagnostic |
| A2 | ternary + A8 | **baseline** | yes (both routes; W+A8 mode) |
| A2w ★ | ternary, FP activations, trained W-only | contests the **G-track baseline** vs A2 at T2, per the §8.4 rule | yes (TQ2_0; W-only mode) |
| A3 | A2 + λ warm-up (`w_eff=(1-λ)w+λ·quant(w)`, one flag) | stability, only if step-0 spikes | yes |
| A4 | A2 + SubLN | Track B scouting | no (§3.1) |
| A5 | A2 across LR grid | tune healing | yes |
| A6a/b/c | CE-only / full-KL / cached-top-k-KL | measure KD gain and top-k cost | yes |
| A7 | A2 + pre-conversion outlier smoothing | reduce t=0 activation damage; gated on damage map | yes |

### 9.2 O-series (OPD, pathfinder first; prerequisites: T2 done, frozen prompts, gap instrumentation live)

| ID | Description | Question |
|---|---|---|
| O0 | no OPD (heal endpoint) | the gap off-policy healing leaves |
| O1 | +OPD, same-size teacher, λ=1, **E3** | does closing the gap improve generation evals at fixed KL_tf? |
| O2 | O1 with cross-size teacher (3B→1B / E4B→E2B) | ceiling lift from a stronger scorer |
| O3 | λ sweep {0.5, 0.75, 1.0} | do mixed batches stabilize/help? |
| O4 | budget sweep {5%, 15%, 30%} of heal compute | marginal-value curve → default budget |
| O5 | E2 vs E3 (one run) | measure the support-bias cost empirically, not just theoretically |

### 9.3 I-series (instruction variants; after A1's T3.5)

I0 SFT-only · I1 SFT→OPD · I2 OPD-from-it-teacher · I3 SFT→OPD→DPO. Winner sets the
default for other tracks (§6.6).

---

## 10. Evaluation and monitoring

### 10.1 Core metrics

Train/val loss (CE and KD terms separately), val PPL per eval mode, **KL_tf**
(teacher-forced KL-to-teacher on the fixed calibration set — the truest healing gauge),
grad norm, LR, tokens/s.

### 10.2 Ternary health (per BitLinear tensor)

{-1,0,+1} code fractions, absmean scale, latent norm, quantization error — catches
degeneracies loss hides. **Code-flip rate:** fraction of codes changed over the last K
steps (cached snapshots at eval intervals). High-then-decaying = healthy; near-zero
early = *frozen effective model* (the low-LR failure, invisible in latent norms);
sustained very high = thrashing (the high-LR failure). This is the LR sweep's
mechanistic readout.

### 10.3 On-policy metrics (T3.4 onward)

```text
KL_op  = E_{x ~ student rollouts}[ D_KL(p_S ‖ p_T) ]     (E3 estimator)
on-policy gap = KL_op − KL_tf                             (exposure bias, quantified)
```

Roles: T3.4 decision input; T3.5 exit criterion (gap → floor, generation evals up **at
fixed KL_tf**). **Collapse monitors during OPD:** rollout entropy, distinct-n /
repetition rate, early-EOS rate, length distribution vs teacher, and win-rate of
generations vs the pre-OPD checkpoint. verl-only: rollout-vs-training policy KL per
sync, alarmed on drift.

### 10.4 HealthMonitor tolerances

Early steps look like early pretraining; binarized loss curves are S-shaped (plateau,
then drop). Alerts: NaN/inf, exploding grads, *persistent* post-warmup loss increase,
ternary degeneracy, code-flip anomalies, checkpoint-reload mismatch, parity regression,
and (OPD phase) the §10.3 collapse channels. Never kill runs for ugly absolute values
during the plateau.

### 10.5 Baselines and downstream

Compare against: dense base, unhealed A2, each ablation, pre-OPD checkpoint.
Downstream only after LM loss is healthy: ARC, HellaSwag, PIQA, WinoGrande, BoolQ via
lm-eval-harness + a fixed generation smoke set. Don't over-index on benchmarks before
PPL recovers.

---

## 11. Milestones (one template, instantiated per track)

### 11.1 T0 — Convert, export, reconcile, map the damage (no training)

Dense PPL baseline → convert → fake-quant PPL per eval mode (severe damage expected;
the number is the measurement) → baked-ternary export checkpoint → route quantize →
runs in the designated runtime → full parity report (regime + eval-mode pairings
determined) → config-preservation report → **damage decomposition**: eval-only passes
for A1d, A1b, A2, then module-family subsets (only q/k/v, only o, only gate/up, only
down; optionally coarse per-layer). The damage map steers A7 (activation-dominant),
T0.5 (weight-dominant), and the mixed-precision contingency (tensor-concentrated).
*Exit: conversion/export/runtime work; every parity gap explained; damage map delivered.*

Reconnaissance items by track: **A:** dump the Llama3-8B-1.58 GGUF tensor list; read the
I2_S activation-quant kernel; determine preserve-vs-requantize; check per-tensor
override support in the fork. **G (= the G-T0 spike, §11.2).**

### 11.2 G-T0 export spike (gate for all Gemma work)

On the *dense* model, before any Gemma training code: (1) mainline
`convert_hf_to_gguf.py` on gemma-4-E2B (transformers 5) → bf16 GGUF runs in `llama-cli`;
(2) `llama-quantize` it to TQ2_0 — expect garbage quality (dense weights aren't
ternary); the test is pack/load/run; (3) dump and classify the full tensor list
(ternarizable / keep-FP incl. PLE and QK-norms / dropped encoders); (4) read the TQ2_0
code path — block size, F16 scale handling, weight-only confirmation; (5) verify
`--tensor-type` overrides on this arch; (6) enumerate every `nn.Linear`, draft the G2
profile; (7) record the §2.4 to-be-recorded facts; (8) check whether a pretrained
(non-it) QAT checkpoint exists. Pass = all eight, no blocker. Fail = G tracks stop here.

### 11.3 T0.5 (optional, gated) — layerwise reconstruction init

GPTQ/AdaRound/BRECQ-family: per block, briefly optimize latents (± per-channel input
scales) to minimize fake-quant-vs-dense block-output error on a small calibration set —
a better heal starting point. Run only if the damage map is weight-dominant or T2
under-delivers. Not baseline: real complexity, redundant if the KD heal works on budget.

### 11.4 T1 — tiny heal (10M–100M tokens) → T2 — PoC heal (1–3B tokens) → T3 — production (10–150B)

- **T1:** prove learning on the pinned corpus. Full LR grid; A0 control; first A6a-vs-b
  look; frozen-vs-unfrozen FP params (G); loss decreases, grads sane, flip-signature
  healthy, save/reload works, **export + parity re-pass after training**. *Exit: the T2
  LR/loss recipe is chosen from data; the loop survives a trained checkpoint.*
- **T2:** chosen recipe as baseline; A3/A6c/A7/A2w as budget and gates allow. Deliver
  PPL-recovery and KL_tf curves vs dense and vs unhealed, generations, exported model.
  *Exit: MVS met and exceeded; parity within tolerance.*
- **T3:** scale tokens and seq (2048→4096), multi-GPU; downstream evals; runtime
  speed/memory benchmarks; final export; documented tradeoffs. *Contingency (only if
  quality is short and the damage map points at specific tensors):* mixed precision via
  `--tensor-type` (e.g. `down_proj` or first/last blocks at F16) — dilutes the 1.58-bit
  claim, reported as such, fallback not target.

### 11.5 T3.4 / T3.5 — OPD measurement, then polish

Per §6.2: T3.4 measures the gap with zero updates and decides; T3.5 runs the O-series
recipe (TRL GKD subclass, E3, λ per O3, budget per O4) with the §10.3 exit criteria.
Never on the critical path; never in MVS.

### 11.6 ★ Cross-track deliverable: the heal scaling study

A1 and A3 share tokenizer, corpus, recipe, and instrumentation — their
**tokens-to-recovery curves** (PPL and KL_tf vs tokens, damage-map deltas, best-LR
shift) constitute a two-point scaling study that forecasts G-track and production
budgets from data instead of guesses. Small standing deliverable owned alongside T2
reports; it is how "PoC 1–3B / production 10–150B" stops being a range and becomes a
number per track.

### 11.7 T4 — kernel acceleration (last, and only where profiling points)

`torch.autograd.Function`: on-device weight quantizer (new, tiny) + QuixiCore
`qgemm_bitnet`/`qgemv_w2a8` forward + existing dense GEMMs backward; then fused
CE/RMSNorm/AdamW per profile. Grad-checked against the PyTorch oracle; oracle stays the
fallback backend. Rollout throughput (§6.4) is now a second customer. **Never built:**
ternary backward kernels (§4), training-time packed weight storage, custom attention
(SDPA suffices). Triton/NVIDIA mirrors later if needed.

---

## 12. What exists vs what we build

| Piece | Status | Source |
|---|---|---|
| quant formulas, BitLinear+STE | exist | `.reference/bitnet`, QuixiCore, `gpu/convert_checkpoint.py` |
| harness shape (DDP, shards, schedule, monitor) | template | `~/AUM/train` |
| I2_S export | exists | fork converter + `llama-quantize` |
| TQ2_0 export + `--tensor-type` | exists | mainline llama.cpp |
| Gemma-4 conversion/inference | exists | mainline llama.cpp (transformers 5) |
| OPD algorithm spec | exists | Tinker cookbook recipe (reference), GKD |
| OPD trainer scaffold | exists (experimental) | TRL GKD area — pinned, subclassed |
| OPD at scale | exists | verl OPD + async KD recipe — gated (§6.4), E2→E3 fix required |
| Metal fwd matmuls, act-quant, dense GEMMs, fused CE/norm/AdamW | exist | QuixiCore |
| conversion+profiles, chunked KD, E3 GKD subclass, parity suite (routes × modes), uint32 pipeline, smoothing, code-flip + rollout metrics, CI micro-loop, on-device weight quantizer | **build** | this plan |

---

## 13. Risks

### 13.1 Technical

- **Healing under-delivers at PoC budget.** Escalation order: KD (default) → LR grid →
  λ-ramp → A7 smoothing (activation-dominant) → T0.5 reconstruction (weight-dominant) →
  more tokens (sized by §11.6) → mixed-precision contingency (last; dilutes the claim).
  The FP-embedding quality floor is structural — document, don't fight.
- **Export rejects a model.** Export-first T0s; A-tracks match the released 8B-1.58
  tensor list; G tracks gated on the G-T0 spike.
- **Train/export mismatch.** Baked ternary (§8.2) + conditional rule (§8.3) + eval-mode
  pairings (§8.4) + F16 scale bound; pre-registered activation-side decision rule.
- **Config drift** (`rope_scaling` above all) — field-diff test. **Tokenizer/dtype
  reuse** — re-tokenize, uint32. **Teacher-cache invalidation** — frozen corpora,
  manifest hashes.
- **Memory walls.** §5.4 budgets; chunking + checkpointing mandatory day-one; hatches
  promoted to defaults per track.
- **OPD-specific:** started too early (hard T3.4 gate); entropy collapse (§10.3
  monitors; lower λ, shorten phase); estimator support bias (E3 baseline; O5 measures
  E2's cost); GKD full-logit memory (chunked subclass is a precondition, tested);
  rollout throughput (small budget fraction; T4 kernels); verl surface (triple gate
  §6.4 + bridge §6.5).
- **Fast-moving OPD ecosystem.** Pin framework versions per track; **hard rule:
  re-check TRL/verl docs at each track's T3.4 kickoff** — this document's snapshot is
  not the implementation reference. Re-check specifically: rollout/teacher staleness in
  async pipelines and finite teacher-score caches for reverse-KL OPD, both active
  problem areas.

### 13.2 Program

- **Corpus sourcing at 10–150B tokens × two tokenizations** — the main data lift; the
  LLaMA-3 half shared A1/A3, the Gemma half shared G2/G4; start during A1's T1; size by
  the scaling study.
- **Two runtimes to validate** (A: fork + mainline; G: mainline only) — the parity suite
  carries a `runtime` axis; the CI micro-loop guards the plumbing between real-model
  T0s.
- **Multimodal degradation (G)** — accepted and stated, not mitigated.
- **transformers 5 (G)** — per-track pinned environments.

### 13.3 ★ Communication (with the release checklist as mitigation)

Every released artifact ships a model card stating, at minimum: base model and license;
"Llama/Gemma-shaped 1.58-bit heal, **not** a BitNet b1.58 2B4T reproduction";
ternarized-parameter and ternarized-FLOPs fractions; packed size vs naive estimate;
dense-baseline gap on the standard evals; eval mode and runtime the numbers were
measured in; heal token count and teacher; known limitations (text-only for G;
long-context caveats if seq-4096 training was skipped). The checklist is the structural
fix for stakeholder-language drift — it makes the honest framing the path of least
resistance.

---

## 14. Source map

- **BitNet paper + b1.58 2B4T report** — quantization math, STE, LR/WD recipes, S-curve,
  SFT/DPO settings, native-arch ideas (Track B).
- **HF Llama3-8B-1.58 heal** — the precedent: Llama-shaped target, large-LR finding
  (CE-only; §5.3 for why it doesn't directly transfer), step-0 full quantization, I2_S
  support, the T0 tensor-list ground truth.
- **SmoothQuant/AWQ family** — A7. **GPTQ/AdaRound/BRECQ family** — T0.5.
- **GKD (the seminal on-policy distillation), Thinking Machines' OPD + Tinker cookbook
  recipe, verl OPD docs + async KD recipe, and the 2026 OPD failure-mode literature**
  (support mismatch, teachability, gradient-variance stabilization) — §6's algorithm,
  estimator taxonomy, and gates.
- **Gemma-4 release material + QAT→llama.cpp conversion reports** — §2.4 facts, the F16
  scale-mismatch precedent, the TQ2_0-on-Gemma precedent.
- **mainline llama.cpp** — TQ1_0/TQ2_0, `--tensor-type`, Gemma-4 support.
- **This repo** — I2_S path, high-precision tensor rules, the fork submodule.
- **~/AUM** — harness template. **QuixiCore-Metal** — every T4 kernel except the weight
  quantizer.

---

## 15. The recipe on one page

```text
CRITICAL  A1 T0 → A1 T1 → A1 MVS. Everything else waits.

TRACKS    A1 Llama-1B (pathfinder) → A3 Llama-3B → [G-T0 spike] → G2 Gemma-E2B → G4 Gemma-E4B
          Track B (Microsoft-canonical) is roadmap context, not in the operational map.

CONVERT   Swap block linears for BitLinear via arch profiles; enumerate, don't assume.
          No SubLN. Keep native FFN activations. Preserve every config field
          (rope_scaling!). Gemma: text-only; PLE + embeddings FP and frozen by default.

HEAL      CE + KD from the frozen dense teacher; chunked losses mandatory (128K/262K
          vocab). AdamW, clip 1.0, WD 0.1 on 2-D latents. T1 LR grid 3e-5…4e-4, decided
          by data + code-flip rate. fp32 latents first (A1), hatches by track. uint32
          shards, per-family tokenizer, corpus frozen at T1.

VERIFY    T0 before any training: export, parity (baked ternary; exact codes; F16 scale
          bound; eval-mode pairing W+A8↔I2_S, W-only↔TQ2_0), damage map. CI micro-loop
          guards the full pipeline on every commit. MVS gates production spend.

POLISH    T3.4 measures the on-policy gap with zero updates; T3.5 runs OPD only if it
          earns it: TRL GKD (experimental, pinned, subclassed), E3 support-set reverse
          KL, collapse monitors, exit at gap-closure with KL_tf held. verl only after
          the triple gate. Tinker = recipe, not platform. Instruction variants: OPD
          competes with SFT (I0–I3), winner measured on A1. No OPD code before A1 T2.

KERNELS   Last. Forward quantized matmul + weight quantizer; backward stays dense
          (STE). Never a ternary backward kernel.

SCALE     A1→A3 tokens-to-recovery curves forecast every later budget.
          Off-policy KD heals where the data goes; OPD heals where the model goes;
          in that order.
```