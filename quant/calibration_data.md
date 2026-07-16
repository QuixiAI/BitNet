# Building high-quality calibration data

This guide describes how to build the calibration input consumed by
[`quant.py`](./quant.py) when quantizing
`unsloth/Llama-3.2-1B-Instruct` to TQ1_V BitNet.

The short answer is:

> The best calibration set is a small, clean, deterministic sample of the
> tokens the unquantized source model will actually process in production,
> including its chat template, system prompts, structured inputs, languages,
> context lengths, and representative assistant continuations.

Semantic quality still matters, but calibration is not supervised training.
There are no labels and no gradients. The important property is that the token
and activation distribution matches deployment. A narrow collection of
beautiful prose is usually worse than a diverse, production-matched collection
containing chat, code, JSON, retrieval passages, and the imperfect inputs the
application really receives.

For a serious first quantization, start with a balanced prefix of **512 records
at `--calibration-seq-len 1024`**, then compare nested 128-, 256-, 512-, and, if
useful, 1,024-record runs on a separate evaluation set. Those values are
starting points, not universal optima. Stop increasing the set when held-out
quality has converged.

## 1. File format: it is JSONL, not a JSON array

`quant.py` reads one physical line at a time. The recommended name is:

```text
data/calibration.jsonl
```

The name `data/calibration.json` also works, but its contents must still be
JSONL: one complete JSON object per line. A conventional top-level JSON array
is **not supported**.

Recommended chat record:

```json
{"messages":[{"role":"system","content":"You are a concise technical assistant."},{"role":"user","content":"Why does this query use an index scan?"},{"role":"assistant","content":"The predicate matches the leading index column, so the planner can avoid a full table scan."}],"source":"production_replay","bucket":"chat_technical_en"}
```

Recommended prose or document record:

```json
{"text":"A coherent document passage goes here. Embedded newlines must be escaped as \n inside this one-line JSON object.","source":"licensed_docs","bucket":"prose_en"}
```

The accepted forms are:

| Input form | Collector behavior |
| --- | --- |
| `{"messages": [...]}` | Renders the messages with the model tokenizer's chat template and `add_generation_prompt=False`. |
| `{"text": "..."}` | Uses the string as raw text. |
| `{"prompt": "..."}` | Uses the string as raw text; it does **not** apply the chat template. |
| `{"content": "..."}` | Uses the string as raw text. |
| Plain non-JSON line | Uses that line as raw text. |
| Blank line | Ignores it. |

Extra metadata such as `source`, `bucket`, `language`, or `record_id` is
ignored by `quant.py` and is useful for audits. If `messages` is present and is
a list, it takes precedence over all text fields.

Do not write this:

```json
[
  {"text": "first document"},
  {"text": "second document"}
]
```

Do not use a source-specific field such as `conversations` without converting
it to `messages`; the current parser will not recognize it. Use JSON objects
even for prose when a logical record contains multiple lines. Otherwise every
physical line becomes an independent sample.

## 2. What the calibration collector measures

For every target linear layer, the collector hooks the layer input and always
accumulates the FP64 per-channel second-moment sum

```text
importance[channel] = mean_over_retained_tokens(input[channel] ** 2)
```

For `--importance-mode diagonal`, it divides by the exact retained-token count
and normalizes each layer's vector to mean one. With the default
`--weight-metric iq1`, that activation importance combines with the IQ1-inspired
weight metric during codeword assignment and row-scale fitting. This follows
the same basic motivation as activation-aware quantization: errors in frequently
or strongly activated input channels matter more. See the primary
[AWQ paper](https://arxiv.org/abs/2306.00978) and the read-only local
`~/llama.cpp/tools/imatrix` reference.

The schema-2 collector also supports two stronger, much larger statistics:

| `--importance-mode` | Stored statistic per input block | Intended use |
| --- | --- | --- |
| `diagonal` | Per-channel second moments | Recommended first production pass. |
| `covariance8` | Symmetric 8x8 activation covariance | Exact codeword assignment objective within each eight-weight group. |
| `block256` | Symmetric 256x256 covariance | Required by `--gptq-feedback`; highest memory and collection cost. |

Covariance modes also retain the diagonal. Covariances are symmetrized, receive
the declared ridge, and are normalized by mean diagonal; the artifact keeps
the original FP64 sums and token counts so separately collected shards can be
merged exactly. For quality work, establish the diagonal result first, then
ablate `covariance8`, and use `block256 --gptq-feedback` only if its held-out
gain justifies the much larger artifact and PTQ cost. A mode name alone is not
evidence that it improves the model.

Several consequences are important:

- Every retained token has equal weight. A 1,024-token record contributes
  roughly eight times as much as a 128-token record.
- `--calibration-samples` counts usable records, not tokens.
- Records are processed separately with batch size one. They are not packed
  together and padding does not affect the statistics.
- Each record is truncated independently to `--calibration-seq-len`.
- The collector reads only the first `--calibration-samples` usable records.
- Calibration changes the projection importance and fitted row scales. It does
  not choose the learned TQ1 codebook; the codebook frequency table is derived
  from the model weights.

Build and audit the mixture by **retained token share**, not just record count.
Also interleave the final file's strata so every useful prefix is balanced. A
file sorted as 500 prose records followed by 500 code records will make a
`--calibration-samples 512` run almost entirely prose.

## 3. Rank data sources in this order

### 3.1 Best: representative deployment transcripts

Use opt-in, de-identified, policy-compliant production traffic when it is
available. Preserve the exact inputs seen by the model:

- system prompts and prompt wrappers;
- user turns and representative multi-turn history;
- retrieval passages and citations;
- tool definitions, tool results, JSON, XML, Markdown, tables, and code fences;
- the actual language mix;
- short, typical, and long contexts;
- accepted answers, refusals, and error-recovery turns at their real rates.

Sample by expected production **token share**. Request share alone can be
misleading when one task produces much longer contexts or answers than another.

Do not export secrets or raw personal data merely because the calibration run
is local. Redact or replace identifiers before sampling, restrict access to the
raw pool, and do not commit sensitive calibration files. Keep the redaction
method in the manifest.

### 3.2 Next best: replay representative prompts through the source model

If prompts are available but assistant responses are not, run those prompts
through the original, unquantized checkpoint at the exact revision being
quantized. Use the deployment system prompt, chat template, generation settings,
and tool formatting, then save complete `messages` transcripts.

This is especially useful for the current collector because it calls
`apply_chat_template(..., add_generation_prompt=False)`. A record that ends in
`user` calibrates the prompt tokens but does not add an assistant-generation
header. Complete records ending in a representative `assistant` response also
cover the token prefixes and activations encountered during generation.

Use the source model rather than an unrelated generator when possible. The
goal is to match the source model's own continuation distribution, not to
create better training labels. If production sampling is stochastic, use a
small number of representative seeds rather than generating many near-duplicate
answers for the same prompt.

### 3.3 Public-data fallback

When no deployment sample exists, assemble a mixed public set and manually
inspect it. Useful upstream sources include:

| Need | Possible source | Important caveat |
| --- | --- | --- |
| Complete English conversations | [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) `train_sft` | Synthetic source data; the released set is filtered, but still inspect and remove malformed or repetitive conversations. |
| Broad instruction/task coverage | [Tulu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | Subsets have different licenses and some are non-commercial. Audit and select permitted components rather than assuming the mixture license is sufficient. |
| Clean English explanatory prose | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Web-derived data under ODC-By and Common Crawl terms; inspect excerpts and retain provenance. |
| Non-English prose | [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Select only deployment languages, pin a dataset revision, and review its language-specific quality and terms. |
| Code, tool calls, and private-domain documents | Your own licensed repositories, docs, schemas, and synthetic examples based on them | Preserve production syntax; remove secrets, credentials, generated files, and duplicate boilerplate. |

The official [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
lists English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai as
supported languages. That is not a reason to add all eight. Calibrate in the
languages the deployment will actually use, in their expected proportions.

Dataset cards and upstream contents can change. Pin immutable revisions, save
source IDs, and review the terms of every selected component. Do not download a
large public corpus and take its first 512 rows; streaming order often contains
source, crawl, or length structure that makes such a prefix unrepresentative.

### 3.4 A reasonable fallback mixture

If there is no usage forecast at all, this is a defensible starting mixture for
a general assistant. Percentages are retained token shares:

| Bucket | Token share |
| --- | ---: |
| Complete instruction/chat transcripts | 40% |
| Explanatory and general prose | 20% |
| Code, tool calls, and structured text | 15% |
| Math, reasoning, and planning | 15% |
| Expected non-English languages | 10% |

Long-context, multi-turn, safety, and formatting cases should be distributed
inside those buckets rather than treated as a separate topic. For an
English-only deployment, redistribute the multilingual share according to the
other expected tasks. For a coding agent, code and tool traffic may properly
become the majority. Production evidence always overrides this fallback.

## 4. Construct the candidate pool

Create source pools before selecting the final calibration set:

```text
data/calibration_sources/
  chat.jsonl
  prose.jsonl
  reasoning.jsonl
  code_tools.jsonl
  multilingual.jsonl
data/calibration.jsonl
data/calibration_holdout.jsonl
data/calibration_manifest.json
```

For each candidate record:

1. Normalize chat data to `messages` with string `role` and `content` fields.
2. Preserve the exact production system prompt and formatting where applicable.
3. Render and tokenize with the exact model tokenizer revision.
4. Assign explicit source, task, language, length, and format buckets.
5. Remove malformed, corrupt, empty, and accidental boilerplate records.
6. Deduplicate records within and across sources.
7. Exclude every benchmark, validation set, and application test case used to
   choose the quantized model.
8. Split near-duplicate clusters as a unit so calibration and evaluation do not
   receive different copies of the same content.
9. Select deterministically by stratum and retained token budget.
10. Interleave strata in the final file so nested prefixes remain balanced.

Exact record duplicates usually waste the small calibration budget. Near
duplicates can be detected with normalized text hashes followed by character
n-gram MinHash, SimHash, or manual inspection of a small candidate pool.
However, do not strip shared production scaffolding that truly occurs on every
request. A repeated system prompt should be represented at its real frequency;
500 copies of the same accidental web footer should not.

### Quality filters

Reject or repair records with:

- invalid UTF-8, replacement-character runs, or mojibake;
- empty turns, non-string content, impossible role ordering, or unfinished
  source conversion;
- repeated sentences, token spam, navigation menus, cookie banners, or scraped
  page fragments, unless such text is expected in deployment;
- assistant answers consisting mostly of generic disclaimers or template
  residue;
- leaked API keys, credentials, private URLs, personal data, or copyrighted
  material you are not permitted to use;
- exact or near overlap with the held-out evaluation set.

Do not over-clean away realistic inputs. Misspellings, terse prompts, logs, and
odd JSON are valuable when they are part of the target workload.

### Long records and truncation

The collector truncates; it does not automatically choose a useful window from
an over-length record. Blind truncation can over-sample document openings and
user prompts while dropping later assistant responses.

- Split long prose into coherent passages near the target length, using varied
  windows rather than always the first window.
- For chat, prefer complete turn boundaries. Preserve a system prompt plus a
  coherent sequence of turns that fits the budget.
- Keep some genuinely long records if long contexts are deployed. A collection
  of chopped 512-token passages cannot reproduce activations at later context
  positions.
- Set `--calibration-seq-len` from the workload's length distribution and the
  practical attention cost, not automatically from the model's maximum context.
- Measure the fraction of records and tokens truncated. Unexpected truncation
  is a dataset bug, not harmless noise.

## 5. Choose a token budget

Use these tiers as measurement points:

| Tier | Records | Sequence cap | Use |
| --- | ---: | ---: | --- |
| Smoke | 32 | 512 | Verify parsing, memory, and the end-to-end command. |
| First decision-quality run | 256 | 1,024 | Catch most major workload modes with a well-stratified set. |
| Recommended serious run | 512 | 1,024 | Strong default for this 1B model and collector. |
| Convergence check | 1,024 | 1,024-2,048 | Use only if 512 has not converged or the workload is broad/long-context. |

The actual retained token count matters more than the table. Five hundred
mostly 50-token prompts are not equivalent to 500 mixed 1,000-token
conversations. Conversely, a few giant records can dominate every second
moment. Aim first for roughly 200,000-500,000 well-matched retained tokens, then
measure whether more data improves held-out results.

If the deployment has distinct modes, it can be better to compare separately
calibrated artifacts—such as English chat versus multilingual code—than to
force one averaged importance matrix to serve unrelated distributions. Choose
a single global artifact only after evaluating the combined production mix.

## 6. Audit the exact consumed prefix

The following audit mirrors the current parser and tokenizer path. Set `LIMIT`
and `SEQ_LEN` to the intended CLI arguments. It reports token lengths,
truncation, duplicate rendered records, terminal chat roles, and retained token
share by bucket.

```bash
.venv/bin/python - <<'PY'
from collections import Counter
from pathlib import Path
import hashlib
import json

from transformers import AutoTokenizer

PATH = Path("data/calibration.jsonl")
MODEL = "unsloth/Llama-3.2-1B-Instruct"
REVISION = "main"  # Prefer the immutable commit used for quantization.
LIMIT = 512
SEQ_LEN = 1024

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    revision=REVISION,
    trust_remote_code=False,
)

lengths = []
retained_by_bucket = Counter()
terminal_roles = Counter()
rendered_hashes = Counter()
truncated_records = 0
full_tokens = 0
retained_tokens = 0
first_nonempty = None

with PATH.open(encoding="utf-8") as handle:
    for line_number, raw in enumerate(handle, 1):
        raw = raw.strip()
        if not raw:
            continue
        if first_nonempty is None:
            first_nonempty = raw
            if raw.startswith("["):
                raise ValueError("top-level JSON arrays are not supported")

        item = {}
        if raw.startswith("{"):
            item = json.loads(raw)
            messages = item.get("messages")
            if isinstance(messages, list):
                if not messages:
                    raise ValueError(f"line {line_number}: empty messages")
                for message in messages:
                    if not isinstance(message, dict):
                        raise ValueError(f"line {line_number}: non-object message")
                    if not isinstance(message.get("role"), str):
                        raise ValueError(f"line {line_number}: missing string role")
                    if not isinstance(message.get("content"), str):
                        raise ValueError(f"line {line_number}: missing string content")
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                terminal_roles[messages[-1]["role"]] += 1
            else:
                text = next(
                    (item[key] for key in ("text", "prompt", "content")
                     if isinstance(item.get(key), str)),
                    None,
                )
        else:
            text = raw

        if not text:
            raise ValueError(f"line {line_number}: no usable calibration text")

        token_ids = tokenizer(text, truncation=False)["input_ids"]
        length = len(token_ids)
        retained = min(length, SEQ_LEN)
        bucket = str(item.get("bucket", item.get("source", "unlabelled")))

        lengths.append(length)
        full_tokens += length
        retained_tokens += retained
        retained_by_bucket[bucket] += retained
        truncated_records += int(length > SEQ_LEN)
        rendered_hashes[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1

        if len(lengths) >= LIMIT:
            break

if not lengths:
    raise ValueError("no usable records")
if len(lengths) < LIMIT:
    raise ValueError(f"expected {LIMIT} usable records, found {len(lengths)}")

ordered = sorted(lengths)
def percentile(fraction):
    return ordered[round((len(ordered) - 1) * fraction)]

duplicate_records = sum(count - 1 for count in rendered_hashes.values())
print(f"records:             {len(lengths):,}")
print(f"tokens before cap:   {full_tokens:,}")
print(f"tokens retained:     {retained_tokens:,}")
print(f"records truncated:   {truncated_records:,} ({truncated_records/len(lengths):.1%})")
print(f"exact duplicates:    {duplicate_records:,}")
print(f"length p50/p90/p95:  {percentile(.50):,} / {percentile(.90):,} / {percentile(.95):,}")
print(f"length max:          {ordered[-1]:,}")
print(f"terminal chat roles: {dict(terminal_roles)}")
print("retained token share:")
for bucket, count in retained_by_bucket.most_common():
    print(f"  {bucket:24s} {count:10,d}  {count/retained_tokens:7.2%}")
PY
```

Also validate every line independently and record a file hash:

```bash
jq -c . data/calibration.jsonl >/dev/null
wc -l data/calibration.jsonl
shasum -a 256 data/calibration.jsonl
```

`jq` validation intentionally assumes every record is JSON. Plain-text lines
are accepted by `quant.py`, but all-JSONL input is easier to validate and audit.
After automated checks, manually inspect rendered examples from every bucket,
especially the longest records and every rare language or structured format.

## 7. Record a manifest

Keep `data/calibration_manifest.json` beside the dataset. At minimum record:

```json
{
  "schema_version": 1,
  "calibration_file": "data/calibration.jsonl",
  "sha256": "<sha256>",
  "model": "unsloth/Llama-3.2-1B-Instruct",
  "model_revision": "<immutable-hugging-face-commit>",
  "selection_seed": 20260715,
  "selected_records": 512,
  "calibration_seq_len": 1024,
  "retained_tokens": 0,
  "source_revisions": [],
  "bucket_token_counts": {},
  "filters": [],
  "dedup_method": "<method and threshold>",
  "evaluation_exclusion": "<holdout name or hash>",
  "privacy_review": "<status>",
  "license_review": "<status>"
}
```

Include the builder's source hash or commit if a script produced the file. A
seed is not sufficient when upstream streaming datasets change; immutable
source revisions and selected record IDs are required to reproduce a sample.

## 8. Run the quality-oriented quantization

After the smoke run succeeds, a strong correctness-first command is:

```bash
.venv/bin/python quant/quant.py \
  --model unsloth/Llama-3.2-1B-Instruct \
  --revision <MODEL_COMMIT> \
  --output runs/Llama-3.2-1B-Instruct-TQ1-V12-cal512 \
  --profile tq1_v12-j-r \
  --quantize-tied-embedding-head \
  --shared-head-importance 0.75 \
  --shared-embedding-importance 0.25 \
  --importance-mode diagonal \
  --weight-metric iq1 \
  --calibration-file data/calibration.jsonl \
  --calibration-samples 512 \
  --calibration-seq-len 1024 \
  --device mps \
  --codebook-source learned \
  --codebook-weighting family_equal \
  --candidate-count 32 \
  --alternating-iterations 3
```

Notes:

- Replace `<MODEL_COMMIT>` with an immutable checkpoint revision. If `main` is
  used, `quant.py` resolves it and records the commit in
  `quantization_report.json`, but passing the commit makes the run command
  independently reproducible.
- `--weight-metric iq1` combines the IQ1-inspired weight metric with collected
  activation importance. `--weight-metric uniform` removes that weight factor;
  do not assume it is better merely because calibration data is available.
- `--codebook-source learned --codebook-weighting family_equal` scans all target
  weights and gives each of the seven Llama projection families equal corpus
  mass. The production solver is deterministic; its objective trace and exact
  codebook hash are stored in the artifact.
- More candidates or alternating iterations are not guaranteed improvements.
  Keep those fixed while comparing calibration sets, then ablate them
  separately.
- If MPS memory is insufficient, reduce `--calibration-seq-len` only after
  checking which workload bucket will be lost, or use `--device cpu` for both
  collection and projection. Do not silently change the distribution.
- `--quantize-tied-embedding-head` hooks the output head's final-hidden input and
  stores a vocabulary-length token-frequency tensor. Both are required to
  project the tied matrix once for its lookup and output-logit consumers. The
  raw-sum merger verifies compatible frequency inventories and adds counts.

## 9. Prove that the set is good

Do not select calibration data using the calibration objective alone. Reserve a
disjoint, representative holdout before sampling, preferably a later time slice
for production traffic. Evaluate every artifact on the exact same holdout.

Use one interleaved calibration file and nested prefix sizes such as 128, 256,
512, and 1,024. Hold constant:

- source model and immutable revision;
- TQ1 format and codebook;
- metric, candidate count, iteration count, and scale dtype;
- sequence cap and runtime representation;
- evaluation records and decoding settings.

The learned codebook is independent of calibration data, but explicitly reuse
the first run's canonical artifact with
`--codebook-source artifact --codebook-artifact <ARTIFACT_DIR>` in later runs if
you want to eliminate any doubt about codebook identity. Confirm that all
compared reports contain the same full codebook SHA-256.

For each calibration size and mixture, record:

1. Held-out negative log-likelihood or perplexity versus the unquantized source
   model.
2. Teacher-to-quantized logit KL, preferably mean plus p95/p99, on held-out
   tokens.
3. Top-token agreement and task metrics important to the deployment.
4. Results broken down by task, language, format, and length bucket.
5. The per-tensor and aggregate errors in `quantization_report.json` as
   diagnostics.

Evaluate the actual deployed arithmetic. The optional baked Hugging Face
checkpoint uses decoded TQ1 weights, but ordinary Transformers linears do not
automatically apply the BitNet A8 activation quantizer. Load the canonical
artifact with `bitnet_train.tq1.runtime.load_packed_model` and the matching
`activation_mode` when the target is W2A8. A finite smoke forward is not a
quality evaluation.

### Merge statistics collected in shards

Large or distributed calibration runs may write several compatible statistics
artifacts. Merge them only with the provided raw-sum merger:

```bash
.venv/bin/python quant/merge_calibration.py \
  runs/calibration/part-000.safetensors \
  runs/calibration/part-001.safetensors \
  --output runs/calibration/merged.safetensors
```

The command verifies model/tokenizer revisions, sequence cap, target inventory,
modes, accumulation contract, collector source hash, and ridge setting. It then
adds FP64 raw diagonal/covariance sums and exact token counts and regenerates
the normalized tensors. It refuses to average normalized statistics. Input
artifact hashes, source calibration hashes, record/token/truncation counts, and
bucket totals are retained in the merged metadata. Use the result with
`--statistics-artifact runs/calibration/merged.safetensors`.

Choose the smallest set whose next doubling produces no meaningful held-out
gain and no important bucket regression under a predeclared tolerance. If a
specialized bucket improves while the aggregate does not, fix the mixture or
ship a specialized artifact based on deployment needs; do not hide the result
inside a single average.

## 10. Common failure modes

- Saving a JSON array instead of JSONL.
- Using only user prompts even though generation is the main workload.
- Using a generic Wikipedia or web-text slice for a chat, code, or tool-heavy
  deployment.
- Taking the first rows returned by a streaming dataset.
- Sorting the final file by source even though only its first `N` records are
  consumed.
- Counting records while a few long records dominate retained tokens.
- Allowing truncation to remove most assistant responses or long-context modes.
- Filling the set with random tokens, repeated boilerplate, or one synthetic
  generator's stylistic artifacts.
- Including benchmark test prompts or application acceptance tests.
- Rendering with a different tokenizer, model revision, or chat template.
- Optimizing only aggregate error instead of held-out end-to-end quality.
- Committing private production logs or losing license and source provenance.

## 11. KV-cache-specific calibration

KV Q4 uses the same representative-corpus principles, but it needs more long
contexts and generation prefixes than a weight-only second-moment run. Build a
tracked extension of the primary corpus rather than an unrelated synthetic set:

- preserve the deployed prompt and chat template;
- include own-model continuations so later generation positions are observed;
- include off-policy documents and conversations at each intended context tier;
- balance by retained tokens and position ranges, not record count;
- record context lengths, record/token counts, source hashes, and truncation;
- keep the quality/evaluation prompts disjoint.

Instrument the exact attention implementation and choose one collection point:
`pre_rope` or `post_rope`. Never combine them. Save captured key tensors as
safetensors with exact names `layer.0.key` through `layer.<L-1>.key`, each in
explicit `[batch, kv_head, token, channel]` layout. An optional shared
`token_mask` is boolean `[batch, token]`. All layers must cover the same tokens.
The capture job's own manifest should identify the hook location, attention
implementation, source/model/tokenizer revisions, and input file hashes.

Build the separately linked channel-mean artifact with:

```bash
.venv/bin/python quant/calibrate_kv.py \
  --captured-keys runs/kv/captured_keys.safetensors \
  --output runs/kv/key_channel_mean.safetensors \
  --model-artifact-sha256 <CANONICAL_MODEL_ARTIFACT_SHA256> \
  --model-id unsloth/Llama-3.2-1B-Instruct \
  --model-revision <MODEL_COMMIT> \
  --tokenizer-id unsloth/Llama-3.2-1B-Instruct \
  --tokenizer-revision <TOKENIZER_COMMIT> \
  --layer-count 16 \
  --num-kv-heads 8 \
  --head-dim 64 \
  --kv-dtype float16 \
  --rotation-state post_rope \
  --attention-implementation sdpa \
  --context-length 1024 \
  --context-length 4096 \
  --context-length 16384 \
  --record-count <RECORDS> \
  --source-sha256 <CAPTURE_MANIFEST_SHA256>
```

Use the dimensions and dtype from the exact model configuration/capture; the
values above are an example and are validated, not inferred. The output's
companion manifest hashes the mean artifact and links it to the model artifact.
Runtime loading supplies the expected model/layer/head/dtype/RoPE/attention
identity and rejects any mismatch.

Evaluate FP16, Q8, centered Q4, and uncentered-Q4 ablation on the same model.
For each, measure forward-KL mean/p50/p95/p99 against that model's own FP16 cache
on both own generations and off-policy long contexts, plus downstream scores,
several context lengths, cache bytes, and latency p20/median/p80. Q4 is promoted
as a memory option only if those results pass; reduced bytes alone do not imply
speed or acceptable quality.

## Recommended final checklist

- [ ] One JSON object per physical line; no top-level array.
- [ ] Complete `messages` transcripts for the majority of assistant traffic.
- [ ] Exact deployment system prompts, structured syntax, and source-model
      continuations represented.
- [ ] Token-weighted task, language, format, turn-count, and length mixture
      matches deployment.
- [ ] Records cleaned, deduplicated, de-identified, and license reviewed.
- [ ] Evaluation overlap removed by duplicate cluster, not only exact hash.
- [ ] Final ordering interleaves strata and gives balanced 128/256/512 prefixes.
- [ ] Exact tokenizer audit reports acceptable truncation and no unexplained
      bucket dominance.
- [ ] File, model, tokenizer, source revisions, selected IDs, and seed recorded
      in a manifest.
- [ ] Nested calibration sizes compared on a disjoint, per-bucket holdout using
      the actual W2A8 reference or packed runtime.
