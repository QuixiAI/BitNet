# TQ1_V quantization system specification

| Field | Value |
| --- | --- |
| Status | Normative implementation specification |
| Specification revision | 1.0.0 |
| Canonical artifact schema | 2 |
| Binary format version | 1 |
| GGML type-registry revision | 1 |
| Primary target | `unsloth/Llama-3.2-1B-Instruct` |

This document turns the rationale in [`quant.md`](./quant.md) into an
implementable and testable contract. It covers every feature proposed there:
strict V11/V12 formats, codebook construction, diagonal and covariance-aware
PTQ, GPTQ-style feedback, exact codebook-aware QAT, affine recovery, product
codebooks, artifacts, GGUF integration, CPU and GPU runtimes, mixed tensor
formats, validation, evaluation, and performance gates.

This specification is subordinate to [`docs/train_plan.md`](../docs/train_plan.md)
and [`docs/moe_train_plan.md`](../docs/moe_train_plan.md), which remain the
program-level sources of truth for the shared healing stack. If a requirement
here conflicts with either plan, implementation must stop and reconcile the
documents; it must not silently choose one behavior.

## 1. Normative language and conformance

The words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**, and
**MAY** are normative.

An implementation may claim one or more conformance classes:

| Class | Required capability |
| --- | --- |
| Oracle | Pure-PyTorch codebook, projection, packing, unpacking, decode, W2A8 forward, and exhaustive correctness tests. |
| PTQ producer | Oracle plus calibration, codebook construction, mixed tensor policy, canonical artifact writing, deterministic reports, and full-model PTQ. |
| QAT trainer | Exact deployed projection in every forward, STE/soft curriculum, health metrics, resumable checkpoints, and exact-index export. |
| Artifact exporter | Canonical artifact to GGUF without rediscovering codes or scales, including codebook and row-scale metadata. |
| Runtime backend | Packed inference for a named backend, tested against the scalar oracle for every supported profile and shape. |
| Core production | V12-J-R and V11-J-R PTQ/QAT, canonical artifact, GGUF, CPU reference runtime, at least one optimized runtime, parity, model-quality evaluation, and performance evidence. |
| Full design | Core production plus covariance8, GPTQ feedback, IQ1 baseline, product profiles, affine recovery, generic block-scale profiles, mixed V11/V12 policy, and every evaluation class in this document. |

A backend must name its own coverage. For example, “TQ1 Metal-conformant for
V12-J-R decode” does not imply V11, prefill, product codebooks, or affine
recovery.

The current `quant.py` is an experimental Oracle/PTQ starting point. Its schema
version 1 artifacts are not canonical schema 2 artifacts and must not be
reported as full conformance.

## 2. Goals and non-goals

### 2.1 Goals

The system must:

1. Represent every quantized weight group using a restricted vocabulary of
   eight-trit vectors.
2. Preserve genuinely ternary arithmetic for strict profiles.
3. Support 11-bit and 12-bit indices with exact physical packing.
4. Use one runtime scale per output row for native BitNet profiles.
5. Train against the exact codebook, scale rounding, packing semantics, and A8
   activation behavior used at inference.
6. Produce self-contained, reproducible artifacts containing exact indices,
   scales, codebooks, tensor policy, and provenance.
7. Export those exact values to GGUF without dense re-quantization.
8. Provide a permanent scalar oracle before optimized kernels.
9. Measure quality and performance separately for decode and prefill.
10. Support mixed V11, V12, higher precision, and optionally affine tensors in
    one model.

### 2.2 Non-goals

The system does not:

- claim lossless representation of arbitrary ternary matrices at 11 or 12 bits
  per group;
- call an affine profile “strict ternary”;
- treat a baked dense Hugging Face checkpoint as the canonical deployable
  representation;
- assume that smaller packed size implies faster inference;
- import `~/llama.cpp` or `~/QuixiCore` as runtime dependencies;
- silently pad unsupported matrix widths;
- silently substitute scalar ternarization for TQ1 vector assignment;
- claim BitNet model quality before held-out evaluation.

Local external trees are read-only implementation references. Required source
must be transcribed or vendored under the repository’s existing dependency
rules and attributed with an immutable source revision.

## 3. End-to-end system

The complete data flow is:

~~~text
dense or BitNet latent checkpoint
              |
              +--> representative calibration --> importance statistics
              |
              +--> pattern corpus -------------> codebook construction
              |
              v
       exact TQ1 PTQ initializer
              |
              +--> optional codebook-aware QAT/healing
              |
              v
   canonical packed artifact (indices + row scales + codebooks)
              |
              +--> baked HF debug checkpoint
              |
              +--> exact GGUF exporter
                            |
                            v
          scalar oracle --> packed CPU/Metal/CUDA runtime
                            |
                            v
             parity, quality, and performance gates
~~~

The canonical packed artifact is the handoff between quantization/training and
every exporter. Dense weights are never the source of truth after canonical
export.

## 4. Terminology, tensor inventory, and shape rules

### 4.1 Terms

| Term | Definition |
| --- | --- |
| Trit | One value in {-1, 0, +1}. |
| Lane | One position within an eight-trit codeword. |
| Group | Eight consecutive input-channel weights in one output row. |
| Payload block | Thirty-two groups, or 256 consecutive weights. |
| Row | One output channel of a logical weight matrix. |
| Shape | A sign-canonical eight-trit vector. |
| Codeword | The decoded eight-trit vector selected by an index. |
| Latent weight | Trainable FP32/BF16 value before TQ1 projection. |
| Runtime weight | `alpha * codeword` for strict profiles. |
| Canonical artifact | Packed indices, exact runtime scales, codebooks, non-TQ1 tensors, and manifest. |

For a logical matrix `W`:

- `W` has shape `[N, K]` in framework order.
- `N` is the number of output channels.
- `K` is the number of input channels.
- Group `g` in row `r` is `W[r, 8g:8g+8]`.
- Payload block `b` contains groups `32b:32b+32`.

### 4.2 Primary Llama tensor policy

The default dense Llama profile MUST enumerate every `nn.Linear`. It MUST
ternarize exactly these seven projections per transformer block:

~~~text
model.layers.L.self_attn.q_proj
model.layers.L.self_attn.k_proj
model.layers.L.self_attn.v_proj
model.layers.L.self_attn.o_proj
model.layers.L.mlp.gate_proj
model.layers.L.mlp.up_proj
model.layers.L.mlp.down_proj
~~~

The embedding, norms, rotary parameters, biases, and `lm_head` remain in their
profile-selected floating-point precision. Tied embeddings remain tied.

Rules match canonical `model.named_modules()` paths, such as
`model.layers.0.self_attn.q_proj`, using full regular-expression matching. The
corresponding state-dict weight name is that path plus `.weight`. Manifests
record both names; calibration tensors use the module path and packed artifact
keys use the state-dict weight name.

Every linear must match exactly one of `target` or `keep_fp`. Doubly matched or
unmatched linears are fatal unless an explicit, recorded opt-out names each
exception. A targeted matrix MUST be bias-free.

For the default Llama profile, the selected count is exactly
`7 * config.num_hidden_layers`; a mismatch is fatal even if every discovered
linear otherwise matched a rule.

### 4.3 Shape constraints

Core TQ1 types require:

- `K % 256 == 0`;
- `N >= 1`;
- `127 * K <= 2^31 - 1` for the required int32 W2A8 accumulator;
- finite source values;
- contiguous logical K-order before packing.

Unsupported widths MUST fail before quantization. Version 1 has no implicit
tail padding. A later padded format would require a new format version and an
explicit logical-width field.

For a fused MoE tensor, each expert projection is a separate logical `[N, K]`
matrix for scale fitting, grouping, health metrics, and export. Flattening
multiple experts into one shared scale is forbidden. Storage retains an
explicit leading expert dimension `E`; it never folds E into N or K.

### 4.4 Tensor-level mixed policy

Every target tensor receives one explicit assignment:

~~~text
fp32 | bf16 | fp16 | tq1_v11-j-r | tq1_v12-j-r |
tq1_v11-i-r | tq1_v11-p-r | tq1_v12-p-r | tq1_v11-j-a4-r |
tq1_v11-j-b | tq1_v12-j-b
~~~

Only profiles enabled by the artifact's codebook registry and implementation
coverage are legal. Profile strings are lowercase in machine-readable data and
uppercase in prose. A floating-point assignment has no codebook ID; every TQ1
assignment has exactly one compatible codebook ID.

Assignments may be selected by ordered full-match regular expressions, but the
resolved manifest MUST list the final format of every tensor. The first
matching rule wins only if the configuration declares that rule ordering;
otherwise overlaps are fatal.

Mixed policy search MAY promote sensitive tensors from V11 to V12 or floating
point. It MUST use a held-out objective and MUST record the search budget,
candidate formats, chosen result, and total effective bits per weight.

## 5. Serializable QuantSpec

All commands and checkpoints MUST carry one canonical QuantSpec. Equivalent
options expressed through different CLIs must serialize to the same object.

The logical schema is:

~~~python
@dataclass(frozen=True)
class CodebookRef:
    id: str                              # [a-z0-9][a-z0-9_-]{0,62}
    format: str                          # v11 | v12
    encoding: str                        # sign_canonical | direct_joint | product
    scope: str                           # universal | model
    sha256: str                          # full 64 hex characters

@dataclass(frozen=True)
class TensorRule:
    match: str                           # full-match regex over module path
    profile: str                         # fp32 | bf16 | fp16 | full TQ1 profile
    codebook_id: str | None

@dataclass(frozen=True)
class QuantSpec:
    spec_revision: str                 # "1.0.0"
    artifact_schema: int               # 2
    format_version: int                # 1
    ggml_type_registry_revision: int   # 1
    default_profile: str               # e.g. tq1_v12-j-r
    codebooks: tuple[CodebookRef, ...] # registry used by defaults/overrides
    default_codebook_id: str
    default_scale_mode: str            # row | block256
    default_scale_dtype: str           # float16 | bfloat16
    activation_mode: str               # model-wide: a8_token | a8_block256 | none
    importance_mode: str               # uniform | diagonal | covariance8 | block256
    weight_metric: str                 # uniform | iq1
    candidate_count: int
    assignment_mode: str               # exhaustive | shortlist
    alternating_iterations: int
    gptq_feedback: bool
    gptq_block_size: int
    gptq_damping: float
    qat_projection: str                # none | soft | hard | frozen
    default_affine_mode: str           # none | rho_mu_a4
    target_regexes: tuple[str, ...]
    keep_fp_regexes: tuple[str, ...]
    tensor_overrides: tuple[TensorRule, ...]
~~~

The canonical hash is SHA-256 over UTF-8 JSON encoded with:

- object keys sorted by Unicode code-point order;
- no insignificant whitespace;
- integers rather than integral floats;
- lowercase enum values;
- full source regex strings;
- the complete ordered codebook registry and every full codebook hash.

Strings use JSON escaping with no ASCII-only conversion. Arrays retain their
declared order. Finite non-integral IEEE-754 values use the shortest decimal
spelling that round-trips to the same binary64 value, lowercase `e`, no `+` in
the exponent, and no redundant exponent zeroes. Negative zero serializes as
`0`. The byte stream has no BOM and no trailing newline. NaN and infinity are
invalid before serialization.

The full 64-character QuantSpec hash MUST appear in training checkpoints,
canonical artifacts, GGUF metadata, parity reports, and evaluation reports.
Truncated hashes MAY be displayed to humans but are not identity keys.

Invalid combinations include:

- a V12 tensor with the 2,048-entry IQ1 joint grid;
- an affine tensor with `default_affine_mode=none` and no tensor override;
- row scale mode without one scale per logical output row;
- `activation_mode=none` while claiming W2A8 parity;
- covariance or GPTQ modes without matching calibration statistics;
- GPTQ feedback with an A4 profile;
- a tensor rule whose profile and referenced codebook encoding disagree;
- frozen QAT without serialized indices.

Schema validation MUST also enforce that codebook IDs match the documented
regular expression and are unique, every hash is 64 lowercase hexadecimal
characters, `default_codebook_id` exists and matches `default_profile`, every
TQ1 tensor rule references an existing compatible codebook, and every
floating-point rule has `codebook_id=null`. Every resolved target tensor must
match exactly one effective rule after defaults and overrides are applied.

## 6. Common numerical contract

### 6.1 Arithmetic precision

Unless a narrower runtime operation is explicitly named:

- source weights are converted to FP32 for statistics and assignment;
- calibration accumulators are FP32 at minimum and SHOULD be FP64 on CPU;
- scale numerator and denominator are accumulated in FP32 at minimum;
- candidate distances are FP32;
- integer dot products accumulate into signed int32;
- final scale multiplication accumulates or converts through FP32 before the
  requested output dtype.

NaN or infinity in source weights, calibration activations, statistics,
codebooks, scales, or outputs is fatal.

### 6.2 Rounding and ties

The numerical contract is round-to-nearest, ties-to-even for:

- scalar ternary initialization;
- A8 activation codes;
- FP16/BF16 scale conversion;
- any explicit integer conversion not otherwise specified.

Candidate-error ties MUST choose the lowest legal numerical index. Codebook
construction ties MUST choose the lowest base-3 pattern ID. Kernel results may
not depend on thread scheduling, unordered reductions, or hash-map iteration.

### 6.3 Activation quantization

The parity activation mode is `a8_token`. For each activation row `x`:

~~~text
a = max(abs(x)) / 127
q = round_to_even(x / a), clamped to [-127, 127]
x_hat = a * q
~~~

If `max(abs(x)) == 0`, `a=0`, every code is zero, and `x_hat` is exactly zero.
There is one activation scale per logical token row.

`a8_block256` is a separate optional mode with one scale per 256 channels. It
applies the same max/127, rounding, clamp, and zero rules independently to each
contiguous aligned block. The activation width must be divisible by 256 in
format version 1. It must have its own QuantSpec, training/evaluation results,
and runtime claim. It must never be substituted for `a8_token` under the same
artifact identity.

`none` is the W-only evaluation mode.

During QAT, `A8_STE(x)` has `x_hat` above as its exact forward value and an
identity gradient with respect to `x`. The dynamically computed maximum,
activation scale, rounded code, and clamp have no separate trainable gradient.
This rule applies identically in soft, hard, and frozen weight phases.

### 6.4 Scale invariants

Strict-profile scales are nonnegative. If a nonzero scale unit's analytic
refit has a nonpositive numerator or zero denominator, the solver retains its
previous positive scale, records a rejected refit, and performs the next
assignment step. If exhaustive reassignment at the final retained scale cannot
produce a valid positive refit, quantization of that tensor fails. A materially
negative optimum is never silently clamped.

An exactly zero source row is encoded with:

- row scale exactly zero;
- every index equal to the zero index.

For nonzero rows, a scale that underflows the chosen runtime dtype is clamped to
that dtype’s smallest positive normal value. The event is counted and reported.

The runtime scale is the value after FP16/BF16 rounding. Every reported
reconstruction metric and every final assignment MUST use that rounded value.

### 6.5 Trit and pattern encoding

The scalar initializer for a scale unit is:

~~~text
alpha_init = sum_i h_i * abs(W_i) / sum_i h_i
q_i = clamp(round_to_even(W_i / alpha_init), -1, +1)
~~~

Uniform initialization uses `h_i=1`. A zero denominator or all-zero scale unit
uses `alpha_init=0` and all-zero trits. This initializer seeds codebook search;
it never limits the final real-valued objective.

An L-lane trit vector maps to a base-3 ID by:

~~~text
base3_id(q) = sum_i=0..L-1 (q_i + 1) * 3^i
~~~

Lane 0 is the least-significant digit. Eight-lane IDs span 0..6560 and
four-lane IDs span 0..80. Sign canonicalization scans lanes 0 through 7 and,
for a nonzero vector whose first nonzero lane is negative, negates the entire
vector and sets the sign bit.

## 7. Strict format profiles

### 7.1 Profile naming

The canonical suffixes are:

| Suffix | Meaning |
| --- | --- |
| J | Joint eight-trit codebook. |
| I | Direct IQ1-grid joint baseline. |
| P | Factorized product codebook. |
| R | External scale per output row. |
| B | Embedded scale per 256-weight block. |
| A4 | Four affine metadata bits per 32 weights; not strict ternary. |

Examples:

- `TQ1_V12-J-R`: core 12-bit sign-canonical joint profile.
- `TQ1_V11-I-R`: exact llama.cpp IQ1 grid used as a strict codeword table.
- `TQ1_V12-P-R`: 12-bit product profile.
- `TQ1_V11-J-B`: generic block-scale compatibility profile.
- `TQ1_V11-J-A4-R`: 11-bit joint indices plus affine metadata at 1.5 raw bpw.

The shorthand `V11` and `V12` means J-R only when the surrounding QuantSpec is
available. Standalone files and reports MUST use the full profile.

### 7.2 Physical payload sizes

Each payload block covers 256 weights or 32 indices.

| Profile | Index payload | Scale payload | Total | Effective bpw |
| --- | ---: | ---: | ---: | ---: |
| V11-{J,I,P}-R | 44 bytes | external | 44 bytes | `1.375 + 16/K` for FP16 row scales |
| V12-{J,P}-R | 48 bytes | external | 48 bytes | `1.500 + 16/K` |
| V11-J-B | 44 bytes | 2-byte FP16 | 46 bytes | 1.4375 |
| V12-J-B | 48 bytes | 2-byte FP16 | 50 bytes | 1.5625 |
| V11-J-A4-R | 44 bytes + 4 bytes | external | 48 bytes | `1.500 + 16/K` |

Reports MUST distinguish:

1. raw index bpw;
2. row/block-scale overhead;
3. embedded codebook overhead;
4. alignment/container overhead;
5. actual on-disk model bpw over targeted parameters;
6. whole-model bpw including non-TQ1 tensors.

Format version 1 defines generic block-scale profiles only for J codebooks.
Adding I-B or P-B requires a later format revision even though the index bits
could fit physically.

### 7.3 Sign-canonical joint encoding

For J profiles:

- V11 stores 1,024 shapes and uses a 10-bit shape ID plus one sign bit.
- V12 stores 2,048 shapes and uses an 11-bit shape ID plus one sign bit.
- Shape 0 is exactly eight zeros.
- Every nonzero stored shape is oriented so its first nonzero lane is +1.
- Shape IDs occupy the low bits.
- The sign is the most significant index bit.

`shape[0]` is zero. After a J codebook's selected shape set is finalized, every
other shape is serialized in increasing eight-trit base-3 ID order. Solver
insertion order therefore cannot change indices or hashes.

Decoding is:

~~~text
shape_id = index & (n_shapes - 1)
negative = (index & n_shapes) != 0
codeword = shapes[shape_id] * (-1 if negative else +1)
~~~

The sign-set encoding of shape 0 is reserved:

- V11 reserved index: 1,024.
- V12 reserved index: 2,048.

Quantizers MUST NOT emit it. Validators MUST reject an artifact containing it.
Decoders MAY decode it as zero for memory safety but MUST surface a validation
error.

The number of unique representable vectors is therefore 2,047 for V11-J and
4,095 for V12-J.

### 7.4 Codebook representation and identity

Every canonical shape is serialized as:

~~~text
positive_mask: uint8
negative_mask: uint8
~~~

Bit `i` corresponds to lane `i`. Positive and negative masks MUST be disjoint.
A zero in both masks is a zero trit. No lane may appear in both masks.

The codebook identity is SHA-256 over one canonical binary stream. All integer
fields are unsigned little-endian. The common prefix is:

~~~text
magic                         13 bytes: "TQ1_CODEBOOK\0"
format_version                uint32
encoding_length               uint16
encoding                      UTF-8 bytes, no terminator
index_format_length           uint16
index_format                  UTF-8 bytes: "v11" or "v12"
table_count                   uint16

for each table in the required order:
    table_name_length         uint16
    table_name                UTF-8 bytes, no terminator
    dtype_code                uint8: 1 = uint8, 2 = int8
    rank                      uint8
    dimensions[rank]          uint32 each
    payload_length            uint64
    contiguous row-major payload bytes
~~~

The required tables and order are:

| Encoding | Table name, dtype, and shape |
| --- | --- |
| J/sign-canonical | `shapes_masks`, uint8 `[shape_count, 2]`; each row is positive mask then negative mask. |
| I/direct-joint | `joint_trits`, int8 `[index_count, 8]`. |
| P/product | `product_a`, int8 `[A_count, 4]`, then `product_b`, int8 `[B_count, 4]`. |

Int8 trits use their ordinary two's-complement byte representation. No padding,
alignment, path, scope, provenance, or container metadata enters this stream.
The full lowercase 64-hex SHA-256 is the codebook identity. This definition
makes the same mathematical table hash identically across safetensors, GGUF,
and runtime implementations while preventing encoding or dimension aliasing.

### 7.5 IQ1-grid baseline

V11-I uses exactly the 2,048 `iq1s_grid` rows from the recorded read-only
llama.cpp reference revision. The index is a direct table row; it has no
shape/sign interpretation. All indices 0..2047 are legal, and the unique
all-zero row is index 1,029.

Specification revision 1.0.0 pins the reference to:

~~~text
repository:  ~/llama.cpp (read-only source reference)
revision:    a5822222909b785f23ddc74ce3c8f85bd0e38562
file:        ggml/src/ggml-common.h
symbol:      iq1s_grid
~~~

For each `uint64_t` literal in source order, lane `i` is bits
`8*i..8*i+7`, interpreted as a two's-complement int8. The resulting table is
`int8[2048,8]`. Under Section 7.4's canonical codebook stream its required
SHA-256 is:

~~~text
1edfeb295366968940d5d4397dc046110f851acb59de9407fdf0c06982adaa72
~~~

The transcribed grid and its immutable source commit MUST be stored in the
repository or embedded in the artifact. A runtime MUST NOT read `~/llama.cpp`.
The expected complete-universe coverage is:

~~~text
squared distance 0: 2048 patterns
squared distance 1: 4252 patterns
squared distance 2:  261 patterns
maximum squared distance: 2
~~~

### 7.6 Product encoding

Product profiles split a codeword into lanes 0–3 and 4–7.

V11-P index bits:

~~~text
bits 0..4   A code (32 entries)
bits 5..9   B code (32 entries)
bit  10     global sign
~~~

V12-P index bits:

~~~text
bits 0..4   A code (32 entries)
bits 5..10  B code (64 entries)
bit  11     global sign
~~~

The decoded codeword is:

~~~text
sign * concat(A[a_id], B[b_id])
~~~

Half-codebooks contain four trits per row. The unsigned Cartesian product
contains exactly 1,024 base vectors for V11 or 2,048 for V12.

For V11-P, both 32-entry half-tables can avoid nonzero sign pairs. The expanded
table MUST then contain 2,047 unique vectors; only sign-set zero is duplicated.

For V12-P, a 64-entry subset of the 81 possible four-trit halves necessarily
contains nonzero sign pairs. If the 32-entry A table is sign-free except zero
and the 64-entry B table minimizes sign pairs, the expanded table has at most:

~~~text
2 * (32 * 64) - (1 * 47) = 4,049 unique vectors
~~~

Conforming V12-P MUST achieve that maximum. A research table with a different
structural constraint uses an experimental profile name and cannot claim
V12-P conformance under format version 1. Every duplicate-equivalence class is
listed in the codebook manifest. The quantizer emits the lowest numerical index
in each class and treats all other encodings as reserved. Validators reject
reserved encodings in weight payloads.

Product and joint codebooks are different model representations. A checkpoint
trained with one cannot be relabeled as the other.

### 7.7 Index packing

Indices are ordered by increasing K within each output row. Each payload block
contains:

1. 32 low bytes, one per index;
2. a dense little-endian bitstream containing the remaining high bits.

For group `g` in 0..31:

~~~text
qs[g] = index[g] & 0xff
high = index[g] >> 8
bit_position = g * high_bits
byte = bit_position // 8
shift = bit_position % 8
qh[byte] |= (high << shift) & 0xff
if shift + high_bits > 8:
    qh[byte + 1] |= high >> (8 - shift)
~~~

Unpacking is the inverse:

~~~text
high = qh[byte] >> shift
if shift + high_bits > 8:
    high |= qh[byte + 1] << (8 - shift)
high &= (1 << high_bits) - 1
index[g] = qs[g] | (high << 8)
~~~

`high_bits` is 3 for V11 and 4 for V12. Unused high bits, if introduced by a
future format, MUST be zero. Version 1 has no unused bits.

Packing is independent for every row and block. The byte order is invariant
across host architectures. Implementations on big-endian systems must perform
explicit byte operations rather than struct reinterpretation.

### 7.8 Scale-mode block layout

External-row-scale payloads contain:

~~~text
byte 0..31                         qs
byte 32..43 (V11) or 32..47 (V12) qh
~~~

Generic block-scale payloads contain:

~~~text
byte 0..1                          FP16 scale, IEEE binary16 little-endian
byte 2..33                         qs
byte 34..45 (V11) or 34..49 (V12) qh
~~~

C structs MUST have static size assertions for 44, 48, 46, and 50 bytes as
applicable; implicit padding is forbidden. Block-scale QAT is not required for
format version 1, but PTQ, decode, export, and the scalar runtime are required
for a generic-profile conformance claim.

## 8. Affine recovery profile

`TQ1_V11-J-A4-R` is the optional non-strict PTQ fallback. It uses:

- one 11-bit joint codeword index per eight weights;
- one four-bit affine field per 32 weights;
- one external row scale.

Four consecutive codewords share one affine field:

~~~text
bits 0..1: rho_id
bits 2..3: mu_id
~~~

The fixed tables are:

~~~text
rho = [6/8, 7/8, 8/8, 9/8]
mu  = [0, +1/8, -1/8, reserved]
~~~

The reserved `mu_id=3` MUST NOT be emitted. The decoded weights are:

~~~text
w_hat = alpha_row * rho * (codeword + mu)
~~~

This profile MUST report `strict_ternary=false`. Its runtime evaluates:

~~~text
for 32-weight subblock s:
    subacc_s = rho_s * (
        sum_g=0..3 dot(codeword_s,g, q_activation_s,g)
        + mu_s * sum_i=0..31 q_activation_s,i
    )
row_output = alpha_row * activation_scale * sum_s subacc_s
~~~

The exact rational reference uses `rho_num={6,7,8,9}` and
`mu_num={0,1,-1}`:

~~~text
subacc_numerator_s = rho_num_s * (8 * dot_sum_s + mu_num_s * q_sum_s)
subacc_s = subacc_numerator_s / 64
~~~

No intermediate integer division is performed before subblocks are summed.

Affine parameters are assigned and refit jointly with codeword indices. The
quantizer MUST evaluate all legal rho/mu choices for each shortlisted codeword
or prove an equivalent exact minimization. QAT support is optional; if QAT is
implemented, the exact affine decode must appear in every forward.

The four affine bytes follow the 44 index bytes in each 256-weight payload
block. Subblock `s` covers weights `32s:32s+32`. Even subblock nibbles occupy
the low half of byte `s//2` and odd subblock nibbles occupy the high half:

~~~text
affine_byte[s // 2] |= affine_nibble[s] << (4 * (s % 2))
~~~

## 9. Codebook construction

### 9.1 Sources

A codebook may be:

- `universal`: frozen across models and eligible for compile-time embedding;
- `model`: constructed for one artifact and carried by that artifact;
- `iq1`: the exact V11-I baseline;
- `loaded`: reused bit-for-bit from a prior canonical artifact.

The source, source hashes, solver configuration, input model revisions, and
training/held-out split MUST be recorded.

### 9.2 Pattern corpus

For each source matrix:

1. fit its ordinary BitNet initializer scale at the declared granularity;
2. form scalar ternary codes with the common rounding contract;
3. split rows into aligned groups of eight;
4. encode groups as base-3 IDs;
5. aggregate pattern frequency and, when available, sensitivity statistics.

Codebook-corpus weighting MUST be explicit:

- `parameter` weights every observed group equally;
- `tensor_equal` normalizes every logical tensor to total weight one;
- `family_equal` normalizes q/k/v/o/gate/up/down families first;
- a custom policy names exact weights.

The production default is `family_equal`, preventing large FFN tensors from
silently determining the entire table.

### 9.3 Required anchors

J codebooks MUST contain:

- zero;
- all canonical support-1 shapes;
- all canonical support-2 shapes;
- all canonical dense shapes;
- any additional architecture-specific anchors declared in QuantSpec.

The anchor list and count are part of codebook provenance.

### 9.4 Facility-location objective

The production learned codebook minimizes:

~~~text
sum_u demand(u) * min_c_in_C (
    distance(u, c) + lambda_nz * nonzero_count(c)
)
~~~

Distance is one of:

- unweighted squared trit distance;
- diagonal importance-weighted squared distance;
- covariance-weighted quadratic distance.

Sign symmetry is enforced through canonical shapes rather than a soft penalty.
For J construction, each observed nonzero `u` is canonicalized together with
its sign; demand is aggregated on the canonical shape, while `distance(u,c)`
is the minimum over the positive and negative encoding of selected shape `c`.
Zero has one legal representative. Thus the solver selects exactly 1,024 V11
or 2,048 V12 stored shapes while optimizing the corresponding expanded legal
codeword set.

The required deterministic solver is:

1. insert required anchors;
2. greedily add the candidate with the largest exact reduction in objective;
3. run deterministic best-improvement medoid swaps;
4. stop after a full pass with no improvement or a recorded iteration limit;
5. break every tie by the lowest canonical base-3 ID.

Frequency-plus-farthest-first is allowed as a named baseline, not as the
production facility-location result.

### 9.5 Product half-table construction

Product tables obey these structural invariants:

- `A[0]` and `B[0]` are the all-zero half;
- rows within each half-table are unique;
- V11 A and B each contain 32 rows and contain no nonzero row together with
  its negation;
- V12 A contains 32 rows with the same sign-free rule;
- V12 B contains 64 rows: zero, one representative from every 40 nonzero sign
  pairs, and the opposite representative for exactly 23 of those pairs;
- after selection, zero is first and all other rows are ordered by increasing
  four-trit base-3 ID.

The production product solver minimizes the same demand-weighted objective as
the joint solver, but its candidate set is constrained to:

~~~text
{s * concat(A[a], B[b]) : s in {-1,+1}}
~~~

It is deterministic:

1. build weighted four-trit marginals for the first and second halves;
2. seed each table with zero and greedily add the marginal medoid that gives
   the largest reduction while respecting the sign-pair constraints;
3. for V12 B, add the required 23 opposite representatives by the same exact
   reduction criterion;
4. evaluate the full eight-trit structured objective;
5. alternate deterministic best-improvement single-row swaps in A then B,
   preserving all invariants, until neither pass improves the objective;
6. break ties by table name, then the lowest incoming four-trit base-3 ID, then
   the lowest outgoing ID.

The selected tables, initialization objective, per-pass objective, iteration
limit, and termination reason are recorded. A frequency-only half-table is a
named baseline and cannot be labeled the production product solver.

### 9.6 Universal codebook acceptance

A universal codebook is accepted only after evaluation on held-out model
checkpoints not used to construct it. Report:

- exact-hit rate;
- coverage histogram over all 6,561 patterns;
- frequency- and sensitivity-weighted distortion;
- per-model and per-family distortion;
- PTQ and QAT model-quality deltas;
- kernel performance for its representation.

### 9.7 Codebook validators

Every codebook load MUST validate:

- exact shape and dtype;
- trits only;
- canonical orientation for J;
- zero at the encoding-required index;
- no duplicate stored J shapes or rows within an I/P component table;
- legal unique-codeword count;
- reserved indices;
- full SHA-256;
- complete-universe coverage;
- product-table invariants and exact declared equivalence classes where
  applicable.

## 10. Calibration and importance statistics

The dataset contract is specified in
[`calibration_data.md`](./calibration_data.md). The collector and this section
are jointly normative.

### 10.1 Dataset requirements

Calibration input MUST:

- be parsed exactly as JSONL/plain text described in the guide;
- use the exact source tokenizer and chat template revision;
- record the file SHA-256 and retained token count;
- be disjoint from model-selection evaluation data;
- use deterministic, interleaved prefixes;
- report per-bucket retained token share and truncation.

### 10.2 Diagonal statistics

For target module input `x` with final dimension `K`:

~~~text
I[j] = sum_tokens x[j]^2 / token_count
~~~

The stored vector has shape `[K]`. Before assignment it is normalized so its
mean is one. Counts are accumulated over non-padding retained tokens only.

### 10.3 Covariance8 statistics

For aligned group `g`:

~~~text
H[g] = sum_tokens outer(x_g, x_g) / token_count
~~~

The stored tensor has shape `[K/8, 8, 8]`. The collector MUST:

1. accumulate in FP32 or FP64;
2. symmetrize with `(H + H.T)/2`;
3. add recorded ridge damping;
4. validate positive semidefiniteness within numerical tolerance;
5. normalize the mean diagonal over all groups to one.

Default ridge damping is:

~~~text
epsilon = 1e-5 * mean_diagonal_before_normalization
~~~

Zero-statistic groups are fatal unless the module was provably never executed
and explicitly excluded.

### 10.4 Block256 statistics for GPTQ

GPTQ feedback requires cross-group covariance. For each aligned 256-channel
block, collect `E[x_block x_block^T]` with shape `[K/256, 256, 256]`, or an
equivalent factorization that reproduces the same solver update within the
declared tolerance.

The artifact records damping, factorization, token count, and any rank
truncation. Covariance8 alone MUST NOT be labeled GPTQ-capable.

### 10.5 Calibration artifact

Importance statistics are stored separately in safetensors:

~~~text
<module>.diag                  float32 [K]
<module>.cov8                  float32 [K/8, 8, 8]
<module>.cov256 or factor      implementation-defined, declared in metadata
~~~

Metadata MUST include:

- schema and collector source hash;
- model/tokenizer IDs and immutable revisions;
- calibration file SHA-256;
- parsing and chat-template mode;
- requested and retained records/tokens;
- sequence cap and truncation statistics;
- device and accumulation dtype;
- normalization and damping;
- target module inventory.

Statistics may be merged only by combining raw sums and counts. Averaging
already normalized statistics is forbidden.

## 11. PTQ algorithm

### 11.1 Source weights

PTQ consumes latent real-valued weights directly when available. It MUST NOT
first bake them through ordinary scalar ternarization. If only baked ternary
weights exist, the manifest records that limitation.

Direct PTQ of a dense non-BitNet Llama checkpoint is supported as an experiment,
but its output is not considered quality-qualified until the full model gates
pass.

### 11.2 Objective

For row scale mode:

~~~text
min over alpha_r >= 0 and indices k_g:
    sum_g error(W[r,g] - alpha_r * C[k_g], H[g])
~~~

For block scale mode, the objective separates over each aligned payload block:

~~~text
min over d_r,b >= 0 and indices k_g in block b:
    sum_g_in_b error(W[r,g] - d_r,b * C[k_g], H[g])
~~~

`d_r,b` is encoded as the block's little-endian FP16 value. Every assignment,
refit, and reported reconstruction uses that rounded value. A source block that
is exactly zero uses `d_r,b=0` and zero indices. Block-scale and row-scale
results are different profiles and cannot share an artifact identity.

For diagonal importance:

~~~text
error(delta, h) = sum_i h_i * delta_i^2
~~~

For covariance8:

~~~text
error(delta, H) = delta.T * H * delta
~~~

`weight_metric=iq1` multiplies diagonal activation importance by:

~~~text
sqrt(sigma_block^2 + W_i^2)
sigma_block^2 = 2 * mean(W_block^2)
~~~

where `W_block` is the aligned 256-weight source block. With covariance8, the
IQ1 factor is applied as `D H D` where `D` is diagonal with the square root of
the IQ1 per-lane factor, preserving a quadratic objective.

### 11.3 Candidate construction

Every ordinary ternary group maps to a base-3 ID in 0..6560.

The exhaustive oracle evaluates every legal codeword. The shortlist path:

1. includes the exact codeword when legal;
2. adds unique legal codewords in increasing squared-trit-distance order until
   the configured candidate count is reached;
3. breaks shell ties by numerical index;
4. evaluates shortlisted candidates with the actual declared importance metric.

The default shortlist is 32 candidates. Production use MUST compare shortlist
assignments against exhaustive assignments on a deterministic sample from
every tensor family and report mismatch rate and excess objective.

For A4, candidate selection is performed per 32-weight affine subblock. For
each of the 12 legal `(rho_id, mu_id)` pairs, the solver selects the best legal
codeword for each of the four contained groups using the actual objective, then
chooses the joint affine/codeword result with the lowest summed error. Ties use
the lowest affine nibble and then the lowest sequence of four codeword indices.

### 11.4 Scale initialization and alternating solve

Diagonal weighted initialization:

~~~text
alpha = sum_i h_i * abs(W_i) / sum_i h_i
~~~

Here `h_i` is one for uniform mode, the normalized diagonal statistic for
diagonal mode, or `diag(H)` for covariance modes.

For fixed codewords under diagonal importance:

~~~text
alpha = sum_gi h_gi * W_gi * C_gi
        / sum_gi h_gi * C_gi^2
~~~

For covariance8:

~~~text
alpha = sum_g C_g.T * H_g * W_g
        / sum_g C_g.T * H_g * C_g
~~~

The solver runs the configured 2–4 alternating assignment/refit iterations.
For row profiles the sums cover the whole row; for block profiles they cover
only one 256-weight block and every block is solved independently. For A4,
replace every `C` in the refit equations with the normalized decoded vector
`rho * (C + mu)` chosen for its 32-weight subblock.

Afterward it:

1. rounds the scale to the runtime dtype;
2. reassigns all groups using that exact scale;
3. computes one additional analytic refit from the new assignments;
4. rounds that candidate scale;
5. reassigns once more if the rounded scale changed;
6. chooses the lower exact runtime objective of the two rounded candidates.

No unexplained empirical multiplier is permitted.

### 11.5 GPTQ-style feedback

When enabled, input channels are partitioned into recorded 256-channel blocks.
For one row and one block, let `T` be a mutable copy of the original source
weights and let `H` be the damped 256x256 covariance. Groups are visited in
increasing K order. After assigning group `g` at the current rounded scale:

~~~text
E_g = T_g - alpha_rt * C[k_g]
T_remaining = T_remaining - E_g * inverse(H_g,g) * H_g,remaining
~~~

The 8x8 solve uses the recorded Cholesky factor; implementations do not form an
explicit inverse. Assignment of `g` uses its current adjusted target `T_g` and
the exact declared group metric. The row-scale refit after a complete sweep is
computed against the original unadjusted weights using the full block metric:

~~~text
alpha = sum_blocks C_block.T * H_block * W_block
        / sum_blocks C_block.T * H_block * C_block
~~~

For a B profile, the same equation is evaluated independently for each block
without the outer sum. Format version 1 GPTQ supports strict J, I, and P
profiles, not A4.

The deterministic feedback procedure is:

1. retain the ordinary alternating-solver result as candidate zero;
2. run one complete feedback sweep at its rounded runtime scale;
3. refit and round the scale with the full-block equation;
4. if the rounded scale changed, rerun exactly one sweep from the original
   weights at the new scale and refit once more;
5. score candidate zero and every feedback candidate on the original weights
   with `sum_blocks delta.T * H_block * delta`;
6. select the lowest objective, breaking ties in favor of candidate zero, then
   the earliest sweep.

The selected row scales and indices still use the exact TQ1 runtime
representation. GPTQ feedback cannot make the declared full-block objective
worse because the ordinary result remains a candidate.

The report records group order, block size, damping, factorization failures,
fallbacks, every sweep scale, and objective before/after feedback. A failed
Cholesky or invalid covariance is fatal unless an explicitly configured
diagonal fallback is used; fallback use is counted per tensor.

### 11.6 PTQ report

Every tensor report includes:

- logical shape and resolved profile;
- codebook ID/hash;
- source and rounded scale ranges;
- raw/effective bpw;
- RMSE and relative L2;
- declared-importance weighted relative error;
- maximum absolute error;
- scalar-pattern exact-hit rate;
- mean and histogram of changed trits per group;
- codeword entropy, dead codewords, and top usages;
- candidate-oracle mismatch statistics;
- iteration objectives;
- elapsed time and peak memory;
- underflow, zero-row, and fallback counts.

The aggregate report includes parameter-weighted and tensor-family summaries.

### 11.7 Mixed-format policy search

An automatic mixed-format run uses a policy-selection split that is disjoint
from calibration and final evaluation. Its configuration declares:

- a starting profile for every target tensor;
- legal promotion edges, such as V11-J-R to V12-J-R to BF16;
- whether moves apply to one tensor, one projection family, or one layer;
- a hard packed-byte or effective-bpw budget;
- the held-out objective and maximum number of trials.

The required baseline search is deterministic greedy promotion. At each step it
temporarily applies every legal move that fits the remaining budget, evaluates
the complete model, and selects the move with the largest objective improvement
per added physical byte. Non-improving moves stop the search. Ties choose lower
total bytes, then the lexicographically lowest resolved tensor-name set. The
model is re-evaluated after every accepted move; isolated per-tensor errors are
not assumed additive.

Alternative exact knapsack, beam, or Bayesian searches are allowed when named,
but must use the same split and budget accounting. The report retains every
trial, objective, byte delta, accepted order, final resolved tensor policy, and
policy-selection data hash. Final quality is measured once on the untouched
evaluation set.

## 12. Exact codebook-aware QAT

This contract applies to J, I, and P row-scale profiles; only their decode and
legal-index sets differ. Core conformance requires J. Full-design conformance
also exercises the I baseline and P profiles. Block-scale QAT is outside format
version 1 requirements, and A4 QAT remains optional as stated in Section 8.

### 12.1 Module state

A TQ1-trainable linear stores:

- latent FP32 or BF16 weight `V[N,K]`;
- trainable FP32 unconstrained row-scale parameter `a[N]`;
- immutable codebook buffers and hash;
- QuantSpec and resolved tensor profile;
- optional fixed calibration importance;
- current phase, temperature, and candidate configuration;
- current or frozen indices `[N,K/8]`.

A fused expert wrapper applies this state independently to each expert and adds
a leading E dimension to latent weights, scales, and indices. All equations
below are then evaluated independently for each `(expert,row)` pair.

The positive continuous scale is:

~~~text
alpha_fp = softplus(a)
alpha_rt = STE_cast_to_runtime_dtype(alpha_fp)
~~~

`STE_cast` uses the rounded FP16/BF16 value in the forward and identity
gradient with respect to `alpha_fp`.

Initial `a` is the numerically stable inverse-softplus of the positive PTQ row
scale before runtime rounding. Runtime conversion applies Section 6.4's normal
minimum, rounding, and finite-value rules.

An exactly zero row is a non-trainable special case: its scale is fixed to zero
and its indices are fixed to the encoding's legal zero index. It bypasses the
softplus parameterization.

### 12.2 Hard forward

For every forward in hard or frozen phases:

~~~text
Z = V / alpha_rt[:, None]
indices = exact_or_shortlisted_assignment(Z, importance)
C = decode(indices)
W_hard = alpha_rt[:, None] * C
surrogate = V + alpha_rt[:, None] * stop_gradient(C)
W_STE = stop_gradient(W_hard) + surrogate - stop_gradient(surrogate)
output = linear(A8_STE(input), W_STE)
~~~

The deployed rounded scale and exact codebook appear in every forward. Cached
indices may be reused only while the latent weight and scale parameter versions
are unchanged. The surrogate gives the latent an identity STE and gives the row
scale the gradient of the selected codeword while preserving `W_hard` exactly
in the forward.

For group `g`, `distance_k` is the declared Section 11.2 error of
`V_g - alpha_rt*C_k`. Exhaustive mode considers every legal index. Shortlist
mode builds the ordinary trit initializer from `Z_g`, uses Section 11.3's
unique legal candidates, and then ranks them by this real-valued distance. All
ties choose the lowest legal numerical index.

### 12.3 Soft projection

For each group’s top-M legal candidates:

~~~text
p_k = softmax(-distance_k / temperature)
C_soft = sum_k p_k * C_k
C_hard = C[argmin distance]
C_ST = C_soft + stop_gradient(C_hard - C_soft)
W_hard = alpha_rt * C_hard
surrogate = V + alpha_rt * C_ST
W_STE = stop_gradient(W_hard) + surrogate - stop_gradient(surrogate)
~~~

Forward values are hard. Gradients follow the soft mixture. Temperature,
top-M, schedule, and minimum temperature are checkpointed.
Temperature must be finite and strictly positive, and
`1 <= top_m <= candidate_count <= legal_index_count`. Candidate ties use the
same numerical-index rule as hard assignment.

### 12.4 Backward contract

The default hard-phase surrogate is:

~~~text
surrogate = V + alpha_rt * stop_gradient(C_hard)
W_STE = stop_gradient(alpha_rt * C_hard)
      + surrogate
      - stop_gradient(surrogate)
~~~

Therefore the forward sees the exact hard runtime weight, `V` receives an
identity STE, and runtime scales receive gradients through `alpha_rt`’s STE
cast. In the soft phase, `C_ST` replaces the stopped hard codeword inside the
surrogate so assignment gradients follow `C_soft`. Codebook tables are frozen
by default. Experimental trainable codebooks require a different spec revision
and are outside format version 1.

Activations use the shared BitNet A8 STE from the training stack. Backward is
dense; no ternary backward kernel is required.

### 12.5 Loss

The QAT loss is:

~~~text
L = L_CE
  + lambda_KL * tau^2 * KL(teacher || student)
  + lambda_hidden * L_hidden
  + lambda_margin * L_margin
~~~

Requirements:

- `p_t=softmax(teacher_logits/tau)`, `log_p_s=log_softmax(student_logits/tau)`,
  and KL is `mean(sum(p_t * (log(p_t) - log_p_s)))`, masking exactly the same
  non-padding prediction positions as CE. This is `KL(teacher || student)` and
  must match `bitnet_train.distill`.
- Hidden loss names exact layers and uses normalized MSE or a recorded metric.
- If `D1 <= D2` are best and second-best candidate distances:

~~~text
L_margin = mean(max(0, margin - (D2 - D1)))
~~~

- Loss weights, teacher revision, cache manifest, and hidden-layer selection
  are checkpointed.

### 12.6 Curriculum

#### Phase A: PTQ initialization

- Build or load the final codebook.
- Run exact PTQ with rounded scales.
- Initialize latent weights, scales, indices, and the frozen teacher.
- Record step-zero CE, KL, hidden error, assignment margin, and codebook health.

#### Phase B: soft projection

- Use top-M soft gradients with hard forward values.
- Anneal temperature according to a recorded schedule.
- Train latent weights, row scales, norms, and explicitly allowed FP tensors.
- Candidate refresh occurs when weight/scale versions change.

#### Phase C: hard projection

- Use hard assignments and latent STE in every forward.
- Continue CE/KL/hidden/margin objectives.
- Monitor codeword-index churn and per-family damage.

#### Phase D: frozen indices

Entry requires:

- sustained index-flip rate below a declared threshold;
- no worsening held-out CE/KL trend;
- acceptable assignment margins;
- no dead-codeword or scale pathology requiring intervention.

A configured freeze step is the earliest eligibility point, not permission to
bypass these gates. If the gates are unmet at the configured maximum hard-phase
step, the run ends without a frozen/export-qualified claim and records which
gates failed.

Indices become immutable. Latent weights are frozen or removed from the
forward. Trainable parameters are limited to row scales, norms, allowed FP
tensors, and an explicitly temporary adapter. Any merged adapter must be
reprojected before indices are frozen again.

#### Phase E: exact export

- Export stored indices and runtime-rounded scales directly.
- Pack with the production packer.
- Run canonical and GGUF validators.
- Compare CPU oracle and deployed runtime.
- Reject any index, scale, codebook-hash, or tensor-inventory mismatch.

### 12.7 QAT health metrics

At step zero and every evaluation interval, record per tensor and aggregate:

- index flip rate;
- changed trits per group;
- exact scalar-pattern hit rate;
- best/second assignment margin distribution;
- codebook entropy and perplexity;
- dead and newly activated codewords;
- zero-codeword rate;
- per-lane {-1,0,+1} fractions;
- scale min/median/max and underflow count;
- weighted projection error;
- CE, teacher KL, and selected hidden errors;
- W-only versus W2A8 mode gap.

Frozen indices changing is a fatal error.

### 12.8 Training-stack integration

TQ1 must extend the existing `ArchProfile.quant` data model rather than create
an unrelated trainer. The shared trainer continues to own data, KD, optimizer,
monitoring, checkpoint cadence, and eval modes.

Checkpoints MUST include:

- full QuantSpec and hash;
- complete codebooks and hashes;
- latent weights, scale parameters, phase, temperature, and indices;
- optimizer/scheduler state;
- calibration-statistics hash;
- teacher/cache/data manifests;
- RNG states;
- distributed model state;
- exact data-stream resume position.

FSDP saves MUST use a gathered full state dict or the framework’s distributed
checkpoint API. Rank-zero `save_pretrained` on a sharded module is not
conformant.

## 13. Canonical artifact

### 13.1 Directory contents

A canonical artifact directory contains:

~~~text
config.json
tokenizer files
tq1_manifest.json
tq1_packed.safetensors
non_tq1_model.safetensors or shards
quantization_report.json
evaluation_report.json                 required for a quality-qualified release
~~~

The packed artifact, not a baked checkpoint, is canonical.

### 13.2 Packed tensor keys

For each quantized HF weight name `name`:

~~~text
name.__tq1_payload       uint8 [N, K/256, block_bytes]
name.__tq1_scale         float16 or bfloat16 [N]          row mode only
~~~

For a fused E-expert weight, the corresponding shapes are
`[E,N,K/256,block_bytes]` and `[E,N]`. The manifest names the expert, output,
and input axes explicitly.

`__tq1_payload` is the complete physical block, not a normalized index array:

| Profile | `block_bytes` | Contents |
| --- | ---: | --- |
| V11-{J,I,P}-R | 44 | V11 `qs`, `qh` |
| V12-{J,P}-R | 48 | V12 `qs`, `qh` |
| V11-J-B | 46 | embedded FP16 scale, V11 `qs`, `qh` |
| V12-J-B | 50 | embedded FP16 scale, V12 `qs`, `qh` |
| V11-J-A4-R | 48 | V11 `qs`, `qh`, four affine bytes |

Thus block scales and affine nibbles have exactly one source of truth and are
never duplicated as side tensors. Row scales remain companions because `_R`
payloads deliberately omit them. The schema-1 `__tq1_indices` key is not a
schema-2 alias; a reader must reject it or run an explicit versioned migration.

Codebook registry tensors use:

~~~text
__tq1_codebook.<id>.positive_mask       uint8 [shape_count]
__tq1_codebook.<id>.negative_mask       uint8 [shape_count]
__tq1_codebook.<id>.joint_trits         int8 [index_count, 8]
__tq1_codebook.<id>.product_a           int8 [A_count, 4]
__tq1_codebook.<id>.product_b           int8 [B_count, 4]
~~~

Only tensors appropriate to the encoding are present.

### 13.3 Manifest

`tq1_manifest.json` includes:

- schema/spec/format versions;
- source model and immutable revision;
- tokenizer and chat-template hashes;
- full QuantSpec and hash;
- codebook registry with full hashes;
- resolved tensor inventory, logical shapes, physical block sizes, formats,
  codebook IDs, payload hashes, and scale hashes;
- legal and reserved index policy, including the canonical representative and
  members of every product-codebook duplicate-equivalence class;
- scale modes/dtypes;
- calibration and statistics hashes;
- PTQ/QAT phase and training checkpoint identity;
- source hashes for executable quantizer, packer, decoder, activation
  quantizer, exporter, and relevant kernels;
- software/toolchain/device provenance;
- exact command/config;
- payload and whole-model size accounting;
- known runtime compatibility.

Each tensor hash is SHA-256 over its contiguous logical row-major data bytes,
not over a safetensors header or file offset. Uint8 payloads are already
canonical bytes; FP16/BF16 companions use their little-endian 16-bit payloads.
Other multibyte tensors likewise use their declared dtype's little-endian bit
representation.
File-level SHA-256 values are recorded separately for transport integrity.

Unknown required fields are fatal. Readers may ignore unknown optional fields
only when their names are listed in `optional_extensions`.

### 13.4 Baked Hugging Face checkpoint

A dense decoded HF checkpoint MAY be emitted for debugging and framework
evaluation. It MUST be labeled:

~~~text
canonical_packed = false
debug_baked = true
activation_quantization_automatic = false
~~~

It is never fed to a generic quantizer to reconstruct TQ1 indices. Exporters
consume the canonical artifact.

## 14. GGUF and llama.cpp contract

### 14.1 GGML types

Append new enum values; never reuse retired IDs:

~~~text
GGML_TYPE_TQ1_V11          = 43  generic block-scale V11-J
GGML_TYPE_TQ1_V12          = 44  generic block-scale V12-J
GGML_TYPE_TQ1_V11_R        = 45  external-row-scale V11
GGML_TYPE_TQ1_V12_R        = 46  external-row-scale V12
GGML_TYPE_TQ1_V11_J_A4_R   = 47  affine recovery
GGML_TYPE_COUNT             = 48
~~~

These are TQ1 GGML type-registry revision 1 IDs, allocated after type 42 in the
pinned llama.cpp reference used by this specification. They must be mirrored
exactly in C/C++, `gguf-py`, converter enums, tests, and every inference tree.
If upstream assigns any of 43–47 to a different type before integration, the
conflict requires a new TQ1 registry/spec revision and an explicit GGUF
migration; a producer must never emit an ambiguous numeric type.

J versus I versus P codebook encoding is model metadata, not a different
physical index width. Backends must dispatch through the resolved encoding.
If a backend cannot do so without harming performance, a backend-private repack
is allowed but does not alter the GGUF tensor type.

### 14.2 GGUF metadata

Required model metadata:

~~~text
tq1.spec_revision
tq1.format_version
tq1.ggml_type_registry_revision
tq1.quant_spec_json
tq1.quant_spec_sha256
tq1.tensor_policy_json
tq1.codebook.count
tq1.codebook.ids
tq1.codebook.<id>.encoding
tq1.codebook.<id>.index_format
tq1.codebook.<id>.sha256
tq1.codebook.<id>.table_shapes_json
tq1.codebook.<id>.legal_index_count
tq1.codebook.<id>.reserved_index_count
tq1.activation_mode
tq1.strict_ternary
tq1.source_model
tq1.source_revision
~~~

Codebook mask/trit arrays may be GGUF arrays or dedicated immutable tensors.
Dedicated tensors retain the Section 13.2 names. In either representation they
must be self-contained and hash-identical to the canonical artifact.
`tq1.codebook.ids` is the ordered QuantSpec registry ID array, and its length
must equal `tq1.codebook.count`.
`tq1.tensor_policy_json` maps every quantized GGUF tensor name to its complete
profile, codebook ID, logical `[N,K]`, physical block size, and companion-scale
name or `null`. Its tensor-name set must equal the manifest's mapped target set.
`tq1.strict_ternary` is true only when every TQ1 tensor uses a non-affine J, I,
or P profile; a mixed artifact containing any A4 tensor sets it false.

### 14.3 Row-scale tensors

For a GGUF weight tensor `X.weight` using an `_R` type, the companion tensor is
`X.scale` with shape `[N]` and the declared scale dtype. Existing BitNet scale
hooks are extended from optional scalar shape `[1]` to accept row vector shape
`[N]`.

A fused expert tensor uses logical scale shape `[E,N]`; GGUF physical dimension
ordering follows its tensor convention and is declared in tensor policy
metadata. Each expert row is scaled independently.

The scale is applied to output channels after the ternary/int8 accumulation.
Broadcasting must be validated for batch, sequence, and MoE dimensions.
It is applied exactly once: a fused quantized matmul epilogue and a separate
graph multiply are mutually exclusive dispatch choices.

### 14.4 Exact exporter

The exporter:

1. reads the canonical manifest;
2. verifies every codebook and tensor hash;
3. applies required architecture row permutations directly to packed rows and
   row scales;
4. writes packed bytes without dequantizing and reassigning;
5. writes non-TQ1 tensors through the ordinary converter;
6. emits resolved type and scale companions;
7. validates the written GGUF before success.

For Llama q/k rotary permutations, packed index rows and their corresponding
row scales MUST receive the identical output-row permutation.

No “best effort” tensor mapping is allowed. Expected quantized tensor names
come from the canonical manifest. Missing, extra, unsupported, or duplicate
target tensors are fatal.

### 14.5 llama.cpp implementation surfaces

Implementation covers:

- `ggml/include/ggml.h`: types;
- `ggml/src/ggml-common.h`: blocks and compile-time size assertions;
- `ggml/src/ggml-quants.h/.c`: reference pack/decode/quantize;
- `ggml/src/ggml.c`: traits and quantization dispatch;
- `ggml/src/ggml-cpu` and architecture folders: CPU kernels;
- `ggml/src/ggml-metal`, `ggml-cuda`, and claimed secondary backends;
- `include/llama.h` and quantize CLI file types;
- model loader and graph row-scale tensors;
- `gguf-py` constants, quants, and converters;
- quantization/backend/type-selection tests.

## 15. Runtime and kernels

### 15.1 Permanent scalar oracle

The scalar runtime is retained permanently. It:

1. unpacks each index;
2. validates or safely handles reserved values;
3. decodes the exact codeword/affine parameters;
4. computes the integer activation dot;
5. applies activation and row/block scales;
6. returns the declared output dtype.

Every optimized path diffs against this oracle before performance is measured.

### 15.2 Strict joint dot

For `a8_token`:

~~~text
acc_int32 = sum_groups dot_int8_ternary(q_activation[group], codeword[index])
output = alpha_row * activation_scale_token * acc_int32
~~~

The sign bit is applied to the integer group sum. No local scale or offset is
used in strict profiles.

Other strict scale combinations use:

~~~text
R weight + a8_block256: alpha_row * sum_b activation_scale[b] * acc_int32[b]
B weight + a8_token:    activation_scale * sum_b weight_scale[b] * acc_int32[b]
B weight + a8_block256: sum_b weight_scale[b] * activation_scale[b] * acc_int32[b]
W-only R:               alpha_row * sum_g dot(codeword[g], x[g])
W-only B:               sum_b weight_scale[b] * sum_g_in_b dot(codeword[g], x[g])
~~~

Floating epilogues follow Section 6.1. The integer accumulator is exact within
each scale unit; implementations must not incorrectly combine accumulators that
have different activation or weight scales.

### 15.3 Positive/negative-mask kernel

Given masks `P` and `N`:

~~~text
dot = sum(q_activation[i] for i in P)
    - sum(q_activation[i] for i in N)
~~~

Scalar bit iteration is the reference. SIMD-expanded codewords, shuffles, or
mask tables are optimizations and must preserve the exact int32 accumulator.

### 15.4 Product LUT kernel

For each activation group:

~~~text
lut_a[id] = dot4(q[0:4], A[id])
lut_b[id] = dot4(q[4:8], B[id])
group_sum = sign * (lut_a[a_id] + lut_b[b_id])
~~~

LUT construction must be amortized over enough output rows or tiles to justify
its cost. Decode and prefill are benchmarked independently.

### 15.5 Affine kernel

The affine path additionally computes the activation sum for each 32-weight
subblock and applies rho/mu exactly. Integer rational factors SHOULD be folded
before the final floating-point epilogue where possible. The scalar oracle uses
an int64 numerator for the `/8` rho and mu factors; a narrower optimized path
must prove that its supported K range cannot overflow.

### 15.6 Backend-private repacking

A backend MAY repack codebooks or weight payloads at load time. It MUST report:

- original and resident bytes;
- repack time;
- peak temporary memory;
- deterministic repack hash;
- whether the canonical packed form remains resident.

Resident memory, not only GGUF file size, appears in performance reports.

### 15.7 Required runtime shape classes

Each claimed backend tests:

- decode: batch 1, one token;
- small batch decode;
- prefill at representative prompt lengths;
- every target K/N shape in Llama-3.2-1B;
- non-square q/k/v/o and MLP shapes;
- row counts not aligned to SIMD tile boundaries;
- zero rows, sparse and dense codewords;
- mixed-format layers;
- supported output dtypes.

## 16. CLI and configuration contract

### 16.1 PTQ CLI

`quant/quant.py` remains the PTQ/canonical-artifact entry point. It MUST expose
or accept through a config file:

- source model/revision/local-only behavior;
- output and overwrite policy;
- full QuantSpec or equivalent validated flags;
- codebook build/load/source and scope;
- codebook-corpus weighting and solver settings;
- calibration file/statistics artifact;
- diagonal/covariance8/block256 importance;
- candidate/exhaustive mode;
- alternating and GPTQ settings;
- mixed tensor policy;
- device/dtype/chunking;
- artifact-only and optional baked-debug output;
- verification/evaluation prompts or datasets.

The final report stores the normalized QuantSpec, not only raw command flags.

### 16.2 QAT configuration

The shared `train/train.py` configuration gains a TQ1 quant block:

~~~yaml
quant:
  scheme: tq1_v
  spec: path/to/quant_spec.json
  default_profile: tq1_v12-j-r
  default_codebook_id: llama32_v12j
  codebook_artifact: path/to/canonical_codebooks.safetensors
  activation_mode: a8_token
  importance_stats: path/to/calibration_stats.safetensors
  qat_projection: soft
  candidate_count: 32
  top_m: 8
  temperature_start: 1.0
  temperature_end: 0.05
  soft_steps: 1000
  hard_steps: 4000
  freeze_indices_at: null
  margin: 0.1
  lambda_margin: 0.0
  tensor_overrides: []
~~~

Unknown keys are fatal. Resolved values appear in checkpoint provenance.

### 16.3 Export and verification

The existing export route adds `tq1_v` and accepts only canonical artifacts.
The parity tool gains TQ1 decoders and requires:

- exact expected/observed tensor-set equality;
- exact codebook hashes;
- exact packed indices;
- bit-exact scale payloads;
- exact resolved profile metadata;
- model-level logits/PPL pairing with the matching W2A8 mode.

Any skipped or unmapped target makes the command fail.

## 17. Validation and evaluation

### 17.1 Representation tests

Required tests include:

1. base-3 encoding is bijective over all 6,561 vectors;
2. sign canonicalization is exact and symmetric;
3. every codebook validator failure mode;
4. pack/unpack round-trip for every boundary index and randomized matrices;
5. V11/V12/A4 physical size assertions;
6. reserved indices rejected;
7. all codewords decode exactly;
8. scale rounding and zero-row behavior;
9. alternating objective trace and final rounded-scale selection;
10. exhaustive-versus-shortlist comparison;
11. diagonal and covariance objective against direct formulas;
12. GPTQ feedback against a small dense reference;
13. canonical codebook-hash vectors, including the pinned IQ1 hash;
14. cross-language golden payload bytes for V11, V12, block scale, and A4;
15. deterministic results across repeated runs.

### 17.2 QAT tests

Required QAT tests include:

- hard forward equals decoded runtime weights;
- forward uses rounded runtime scales;
- latent STE gradient equals the intended dense gradient;
- scale parameter receives finite gradients;
- soft forward is hard while gradients follow the soft candidates;
- temperature schedule and resume are exact;
- frozen indices never change;
- index-flip/entropy/margin metrics match direct references;
- one optimizer step changes latent state without violating format invariants;
- QAT checkpoint to canonical artifact is index/scale exact;
- distributed save/reload reproduces the next loss and indices.

### 17.3 Runtime correctness

For every backend/profile:

- unpacked indices and scales match the canonical artifact bit-for-bit;
- integer accumulators match the scalar oracle exactly;
- strict profile codewords contain only {-1,0,+1};
- W2A8 activation codes and scales match the activation oracle;
- FP32 final outputs use `atol=1e-6, rtol=1e-6` unless the same operation order
  permits bitwise equality;
- FP16/BF16 outputs use predeclared dtype-specific tolerances, with observed
  max absolute and relative error recorded;
- end-to-end logits, top-token agreement, and PPL use the track’s parity gates.

Passing “finite output” alone is never sufficient.

### 17.4 Model-quality matrix

Every quality-qualified model compares:

| Baseline/profile | Required |
| --- | --- |
| Original dense or BitNet teacher | Yes |
| Existing lossless ternary representation | Yes |
| TQ1_0 | Yes |
| TQ2_0 | Yes |
| I2_S/TL when available | Yes |
| IQ1_S on the same source | Yes |
| V12-J PTQ | Yes |
| V12-J QAT | Yes |
| V11-J PTQ | Yes |
| V11-J QAT | Yes |
| Product V11/V12 | Full-design claim |
| Affine recovery | Full-design claim |
| Mixed policy | Yes for a mixed release |

Measure:

- held-out CE/perplexity;
- teacher-to-student token KL, mean and tail percentiles;
- top-token agreement;
- standard downstream tasks;
- long-context behavior;
- instruction/chat evaluations for instruct models;
- results by language, task, length, and tensor-policy ablation;
- calibration-set convergence.

Acceptance thresholds are predeclared per track in the governing training plan.
They may not be chosen after seeing the final run.

### 17.5 Systems-performance matrix

Report:

- actual GGUF and resident bpw;
- load/repack time and memory;
- batch-one decode tokens/s;
- small-batch decode;
- prompt-evaluation tokens/s;
- median and p20/p80 or variance;
- energy per token when measurable;
- index-unpack and codebook-cache behavior from profiling;
- comparison with dequantize-then-matmul and existing ternary baselines;
- CPU and GPU separately.

Kernel work follows [`bitnet_train/perf/perf.md`](../bitnet_train/perf/perf.md).
Every touched kernel or routing threshold requires the repository’s focused
correctness/performance pass and a keep/reject entry in
`bitnet_train/perf/optimization_status.md`.

No speed claim is valid without device, OS/toolchain, command, shapes, dtype,
format, warmups, iterations, median, dispersion, correctness tolerance, and
observed errors.

## 18. Failure behavior

The implementation MUST fail closed for:

- existing output directory unless explicit safe overwrite is requested;
- source revision that cannot be resolved when reproducibility is required;
- unmatched/doubly matched target tensors;
- unsupported width, bias, rank, or dtype;
- invalid calibration JSON or empty retained set;
- missing statistics for a requested importance mode;
- codebook hash, cardinality, or invariant failure;
- NaN/Inf statistics, scales, weights, or outputs;
- index outside range or reserved negative zero;
- packed shape/byte-count mismatch;
- scale count not matching output rows/blocks;
- canonical/GGUF tensor inventory mismatch;
- unsupported backend/profile combination;
- QAT/export QuantSpec mismatch;
- any parity row marked skipped, unmapped, or mismatch.

Fallbacks such as diagonal importance, CPU execution, or dense dequantization
occur only when explicitly enabled and are recorded in the report.

## 19. Reproducibility, provenance, and data handling

Every released result records:

- repository commit and dirty-worktree state;
- full hashes of all executable quantizer/kernel/export sources;
- model, tokenizer, teacher, dataset, and codebook revisions;
- calibration file/statistics hashes;
- normalized QuantSpec/config/profile hashes;
- Python, PyTorch, Transformers, compiler, OS, and device versions;
- seeds and deterministic-algorithm settings;
- complete commands;
- artifact and GGUF hashes;
- evaluation dataset revisions and exclusions.

Sensitive production calibration data is not committed. A redacted manifest may
be committed only if it does not expose prompts, identifiers, secrets, or
private source paths.

## 20. Implementation sequence and gates

### Gate 0: specification and oracle

- QuantSpec and validators.
- V12-J-R and V11-J-R packing/decode.
- IQ1 baseline.
- scalar W2A8 oracle.
- exhaustive unit/property tests.

### Gate 1: full PTQ

- representative calibration and diagonal stats;
- production codebook solver;
- exact alternating solver;
- canonical artifact schema 2;
- real Llama-3.2-1B PTQ and representation report.

### Gate 2: exact QAT

- TQ1 training modules integrated with `bitnet_train`;
- soft/hard/frozen phases;
- CE/KL/hidden/margin losses;
- complete health/resume/provenance;
- V12 then V11 recovery results.

### Gate 3: export and CPU reference

- exact canonical-to-GGUF exporter;
- GGML types and row-scale graph support;
- scalar packed CPU runtime;
- tensor-set, index, scale, codebook, and logits parity.

### Gate 4: quality decision

- required baseline matrix;
- calibration convergence;
- PTQ versus QAT;
- mixed V11/V12 policy;
- predeclared quality gates.

### Gate 5: optimized kernels

- profile-driven CPU optimization;
- Metal and CUDA according to deployment need;
- decode and prefill measured independently;
- performance notebook decisions.

### Gate 6: enhanced profiles

- covariance8 and GPTQ feedback;
- product codebooks/LUT runtime;
- affine recovery;
- universal codebook held-out study;
- full-design evaluation.

No production training spend is justified by packed-size calculations alone.
Each gate requires its own correctness and quality evidence.

## 21. Definition of complete

### 21.1 Core implementation complete

Core is complete only when all of these are true:

- [ ] Canonical QuantSpec and full hashes are used everywhere.
- [ ] V12-J-R and V11-J-R pack/decode oracles are exhaustive and deterministic.
- [ ] Diagonal calibration and production codebook learning are implemented.
- [ ] Full-model PTQ emits schema 2 artifacts.
- [ ] Exact TQ1 QAT runs soft, hard, and frozen phases.
- [ ] Canonical artifacts contain exact indices, scales, and codebooks.
- [ ] GGUF export performs no dense rediscovery.
- [ ] llama.cpp loads row scales and executes a scalar packed CPU path.
- [ ] At least one optimized deployment backend passes oracle tests.
- [ ] Tensor inventory, indices, scales, codebook, and model-level parity pass.
- [ ] V11/V12 PTQ and QAT quality are measured against all required baselines.
- [ ] Decode and prefill performance are measured under the repository protocol.
- [ ] No unresolved P0 correctness issue affects checkpoint, export, or runtime.

### 21.2 `quant.md` fully implemented

The design document is fully implemented only when Core is complete and:

- [ ] Exact llama.cpp IQ1-grid baseline is supported end to end.
- [ ] Covariance8 collection and assignment are supported.
- [ ] Block256 GPTQ-style error feedback is supported.
- [ ] Product V11/V12 codebooks have PTQ, QAT, export, and runtime support.
- [ ] Affine A4 recovery has PTQ, export, oracle, and runtime support.
- [ ] Mixed V11/V12/FP tensor search and export are supported.
- [ ] Generic block-scale V11/V12 types are supported or formally removed from
      the design in a new spec revision.
- [ ] Full codebook construction/held-out study is complete.
- [ ] Every representation, model-quality, and systems metric in this
      specification has a durable report.
- [ ] Every claimed backend names and passes its exact profile/shape coverage.
- [ ] Performance conclusions are recorded in the optimization notebook.

Until every item in `21.2` is satisfied, the accurate project description is a
more limited conformance claim such as “TQ1 V12 PTQ oracle” or “TQ1 V12-J-R
CPU runtime,” not “complete implementation of `quant.md`.”

## 22. Design traceability

This table is the coverage audit for [`quant.md`](./quant.md). A future design
addition is not considered specified until this table and the applicable
normative sections are updated.

| `quant.md` feature | Normative contract here |
| --- | --- |
| IQ1 ternary grid, importance projection, and refit lessons | 6, 7.5, 10, 11 |
| Difference from TQ1_0 | 2, 7, 13, 14 |
| TQ1_V11 and TQ1_V12 bit allocation | 7.1–7.8 |
| Native row scales and generic block scales | 6.4, 7.2, 7.8, 11.2–11.4, 13, 14.3 |
| IQ1 baseline | 7.4–7.5, 9.1, 12, 17 |
| Sign-canonical J codebooks | 7.3–7.4, 9 |
| Weighted facility-location learning | 9, 10 |
| Product P codebooks and LUT execution | 7.6, 9.5, 12, 15.4 |
| Direct-latent PTQ and importance objectives | 10, 11.1–11.4 |
| Covariance8 | 10.3, 11.2, 17.1 |
| Fast candidate search | 11.3, 17.1 |
| Alternating scale/codeword solve | 11.4, 17.1 |
| GPTQ-style feedback | 10.4, 11.5, 17.1 |
| Exact codebook-aware QAT and curriculum | 12, 16.2, 17.2 |
| Affine recovery | 8, 11.3–11.4, 13.2, 14.1, 15.5 |
| Scalar, mask, product, CPU, and GPU kernels | 15, 17.3, 17.5 |
| llama.cpp type, GGUF, converter, and row-scale work | 14, 16.3 |
| Development sequence | 20 |
| Representation, quality, and systems evaluation | 17 |
