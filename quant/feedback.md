# Verdict

This is a **substantial and well-designed research branch**, not a toy implementation. The numerical discipline, QAT infrastructure, distillation path, export checks, CPU reference kernels, Metal kernels, and performance methodology are all unusually thoughtful.

However:

> **It does not yet implement the IQ1-inspired BitNet vector quantizer we discussed.**

The current weight representation is still scalar absmean ternarization:

[
s=\operatorname{mean}(|W|),\qquad
q_i=\operatorname{clip}(\operatorname{round}(W_i/s),-1,1),
\qquad
\widehat W_i=s_{\mathrm{fp16}}q_i
]

Each trit is selected independently. There is no restricted eight-trit codebook, importance-aware vector assignment, neighbor search, or vector-codebook-aware QAT in the current quantization path.

The runtime formats are also not sub-1.58 bpw:

* **TQ2_0:** 66 bytes per 256 weights, or **2.0625 bpw**.
* **I2_S:** 2-bit codes plus a tensor scale, asymptotically **about 2 bpw**.
* **TL1:** **2.5 bpw**, including one FP16 scale per 32 weights.
* **Base-3:** **2.25 bpw** in the implemented block layout.

So “1.58-bit” currently describes the ternary alphabet’s information content, not the actual stored artifact.

I would characterize the branch as:

> **A strong scalar-ternary healing and runtime platform that is well positioned to become the IQ1-inspired vector-quantized BitNet implementation.**

---

# What is especially good

## 1. The numerical contract is explicit

The branch pins several details that frequently cause training/export discrepancies:

* Code formation uses an FP32 absmean.
* Dequantization uses the FP16-rounded stored scale.
* Activations use `[-127,127]`.
* Rounding is round-half-to-even.
* The Python, Metal, and export implementations are intended to use the same convention.

That is the right engineering discipline. The Metal tests compare packed codes, scale rounding, and reconstructed values against a host oracle rather than merely checking that outputs are finite.

## 2. The STE is tested correctly

The tests verify the actual intended gradients:

[
\nabla_x L = \nabla_y L,\widehat W
]

[
\nabla_W L = \nabla_y L^T,\widehat x
]

This is much stronger than checking that a gradient tensor exists. The custom Metal autograd path also records the original dtypes and returns gradients in those dtypes.

## 3. Shared activation quantization is a sensible optimization

Caching the quantized activation by input tensor identity is a good optimization for sibling projections such as Q/K/V and gate/up. The tensor version check handles in-place mutation, and weak keys avoid manually managing most lifetimes.

## 4. The kernel work is driven by measurements

The CPU investigation correctly discovered that the original decode kernel was unpack-compute-bound rather than bandwidth-bound. The TL1-style lookup implementation then improved the measured ternary GEMV by roughly 2.5–2.7× over the original NEON format.

This strongly supports one part of our earlier design:

> The codebook must be designed jointly with the lookup/matrix-multiplication kernel.

The current TL1 pair-index representation is not the final compressed codebook, but it is useful evidence that lookup-based ternary computation can be substantially faster than unpacking individual trits.

## 5. The scalar-reference-first approach is correct

The CPU implementation retains scalar reference kernels and compares optimized implementations against them. That is exactly how the eventual V11/V12 codebook kernel should be developed.

## 6. Training recovery infrastructure is already rich

The branch already contains much of what a vector-codebook QAT experiment needs:

* Hard ternary forward passes.
* Distillation.
* Code-flip monitoring.
* Quantization error statistics.
* Fixed calibration windows.
* Blockwise reconstruction.
* Export baking.
* Exact-code parity logic.
* CPU and Metal runtime paths.

That is valuable. The central missing piece is the vector representation itself.

---

# Blocking correctness issues

I would fix the following before trusting CPU quality measurements, long training runs, or export claims.

## P0: The CPU engine rescales already-baked weights incorrectly

The CPU engine says it loads a baked checkpoint, but its packer recomputes an absmean scale from the baked tensor:

```python
scale = np.full(
    (N, nb),
    max(np.abs(W).mean(), 1e-5),
    np.float32,
)
```

It then packs that newly computed scale.

For a baked tensor,

[
W_{\mathrm{baked}}=s q,\qquad q\in{-1,0,+1}
]

the recomputed scale is:

[
s'
=
\operatorname{mean}(|W_{\mathrm{baked}}|)
=

s\operatorname{Pr}(q\ne0)
]

If 70% of the weights are nonzero, the CPU engine stores approximately (0.7s). The trit codes remain the same because the values are clipped back into ({-1,0,+1}), but every nonzero reconstructed weight is 30% too small.

This is especially dangerous because the existing CPU tests will not detect it:

* They instantiate and save a random dense model, not a baked model.
* They check that outputs are finite.
* They verify that TL1 matches the original 2-bit format.
* They do not compare the CPU engine logits against the baked PyTorch model.

Both CPU formats can therefore agree perfectly while both use the wrong scale.

### Fix

Separate latent quantization from baked-weight packing:

```python
def pack_latent_ternary(W):
    # Training-style conversion.
    scale = np.abs(W).mean(...)
    codes = np.clip(np.rint(W / scale), -1, 1)
    return codes, scale


def pack_baked_ternary(W):
    # W must already contain only 0 and ±scale.
    scale = np.abs(W).max(...)
    codes = np.sign(W).astype(np.int8)

    reconstructed = codes * scale
    np.testing.assert_allclose(reconstructed, W, rtol=0, atol=0)
    return codes, scale
```

Even better, load the exact stored scale from `bake_report.json`, and fail if the baked values do not match it.

This should be accompanied by an end-to-end test:

[
\operatorname{logits}*{\mathrm{CPU}}
\approx
\operatorname{logits}*{\mathrm{baked\ PyTorch}}
]

not just TL1-versus-format-A equality.

---

## P0: The export parity gate can report success while tensors are missing

The parity routine currently:

1. Guesses which tensors are ternary by sampling only the first four rows.
2. Records an absent mapped tensor as `unmapped`.
3. Records unsupported GGUF types as `skipped`.
4. Sets the overall result to failure only when a row has status `mismatch`.

Thus `unmapped` and `skipped` tensors do not fail the preserve-regime gate.

There is a second issue: for TQ2_0, code equality and reconstruction error are tracked separately, but the overall `ok` value is changed only by code status. A row can have zero code mismatches, exceed the supposed FP16 reconstruction bound, and still leave the overall gate at `PASS`.

The integration test partially masks this by requiring only that at least one tensor was decoded. It does not require that every baked target tensor was mapped and checked.

### Fix

Make `bake_report.json` the source of truth:

```text
expected tensor set = all tensors explicitly marked ternary in bake_report
observed tensor set = all successfully decoded target GGUF tensors

require expected == observed
```

In preserve mode, fail on any of:

* Missing tensor.
* Unmapped name.
* Unsupported GGUF type.
* Shape mismatch.
* Nonzero code mismatch.
* Scale mismatch.
* Reconstruction error beyond the defined bound.
* Unexpected additional ternary target.

The gate should effectively become:

```python
ok = (
    observed_names == expected_names
    and all(row.status == "exact" for row in rows)
    and all(row.within_f16_bound for row in rows)
)
```

For a future V11/V12 format, compare the **packed codeword indices and scale values directly**, rather than dequantizing and inferring whether they happen to agree.

---

## P0 for the Q-track: FSDP checkpoint saving is unsafe

The save function currently runs `save_pretrained()` directly on the unwrapped model from the main process:

```python
accelerator.unwrap_model(model).save_pretrained(path)
```

It separately saves optimizer state and RNG information.

For FSDP, Hugging Face’s current Accelerate guidance is to either:

* Use `accelerator.save_state()` and `accelerator.load_state()`, or
* Pass `state_dict=accelerator.get_state_dict(model)` to `save_pretrained()`.

`get_state_dict()` enters the appropriate full-state-dict context and gathers the model correctly on rank zero. ([Hugging Face][1])

The current code can therefore produce incomplete or inappropriate model state when parameters are sharded.

### Fix

For final model checkpoints:

```python
unwrapped = accelerator.unwrap_model(model)
state_dict = accelerator.get_state_dict(model)

unwrapped.save_pretrained(
    path,
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
    state_dict=state_dict,
)
```

For resumable training checkpoints, prefer:

```python
accelerator.save_state(path)
```

and:

```python
accelerator.load_state(path)
```

Keep the provenance metadata as a supplementary file.

A real two-process FSDP test should train, save, destroy the process group, reload, and verify:

* Parameter equality.
* Optimizer-state restoration.
* Next-step loss equality.
* Same codeword/code-flip state.

---

# Important P1 issues

## The BF16 latent-memory option currently appears ineffective

The trainer exposes:

```text
--latent-dtype bf16
```

and loads the checkpoint with `torch_dtype=torch.bfloat16`.

But conversion then creates every `BitLinear` without specifying a dtype or device. `BitLinear` defaults its parameter to FP32, and conversion explicitly copies `old.weight.float()`.

As a result:

* `--latent-dtype bf16` loads BF16 linears.
* Conversion replaces them with FP32 `BitLinear` parameters.
* The trainer selects `MasterAdamW` as though the parameters were BF16.

So the claimed BF16 latent memory reduction is probably not being realized.

The same code also does not preserve the source parameter’s device, making direct conversion of a model already on CUDA or MPS unsafe.

### Fix

Make latent dtype an explicit conversion input:

```python
def convert(
    model,
    profile,
    backend="reference",
    latent_dtype=None,
):
    ...
    new = BitLinear(
        old.in_features,
        old.out_features,
        group_k=profile.group_k,
        backend=backend,
        granularity=profile.granularity,
        device=old.weight.device,
        dtype=latent_dtype or old.weight.dtype,
    )
    new.weight.copy_(old.weight.to(new.weight.dtype))
```

Add tests asserting actual parameter dtypes and devices after conversion, not just testing `MasterAdamW` in isolation.

---

## Resume does not resume the data stream

The data loader creates its own seeded `torch.Generator`, and `cycle(loader)` starts from the beginning of that loader.

The checkpoint records step, tokens, optimizer state, and global RNG states, but it does not store:

* The loader generator state.
* Sampler epoch.
* Position within the current epoch.
* Number of already-consumed microbatches.

On resume, the loader is recreated before RNG restoration, then iteration starts at its beginning.

This means resumed training can repeat earlier batches rather than continue the exact token stream.

### Fix

Save and restore either:

* Sampler epoch plus batch offset, or
* The data-loader generator state plus exact iterator offset.

For deterministic packed-window training, an even cleaner approach is a stateless sampler:

[
\operatorname{window_index}
=

f(\text{seed},\text{global step},\text{rank},\text{microstep})
]

Then resume position is determined entirely from the global step.

---

## Top-k cache validation is weaker than its documentation claims

The cache manifest stores:

* Corpus hash.
* Split.
* Sequence length.
* (k).
* Temperature.
* Number of windows.

But the reader validates only the corpus hash and temperature.

A cache made for a different split, sequence length, top-(k), or limited window count can therefore be accepted. This can produce incorrect teacher alignment or a later shape/index failure.

### Fix

Require all of these at construction:

```python
TopkCacheReader(
    cache_dir,
    data_dir,
    split="train",
    seq_len=args.seq_len,
    k=expected_k,
    tau=args.kd_tau,
    n_windows=len(train_ds),
)
```

Also verify the actual `.npy` shapes against the manifest.

---

## The quantizer provenance hash omits executable quantizer sources

`quantizer_hash()` currently hashes:

* `quant.py`
* `bitlinear_metal.py`
* A manual version string.

It does not hash the Metal quantization kernel, bindings, runtime packing implementation, or eventual codebook.

The actual Metal code determines scale reduction, rounding, code formation, packing, and dequantization. A numerical change there would not invalidate existing checkpoints.

### Fix

Create a quantization-spec manifest containing hashes of:

* Python reference implementation.
* Metal quantization sources.
* CPU packer.
* Export packer.
* Activation quantizer.
* Codebook data.
* Codebook-generation configuration.
* Scale and rounding specification.
* Format version.

For V11/V12, the codebook SHA is part of the model’s mathematical identity and must be stored in both the checkpoint and GGUF metadata.

---

## No CI result currently backs the branch

The head commit currently has no reported workflow runs.

The test suite is broad, but without an active workflow it is easy for platform-specific tests and integration checks to become aspirational rather than enforced.

At minimum, I would add:

* Linux CPU reference suite.
* Two-process distributed/FSDP suite.
* Export/parity suite with a pinned llama.cpp commit.
* macOS MPS/Metal suite on a self-hosted runner.
* A baked-model CPU-engine parity test.
* Static checks for accidental skipped tests.

---

# Smaller issues

The Base-3 C comment says 9 bytes per 32 weights is 1.8 bpw. It is:

[
9\times8/32=2.25\text{ bpw}
]

The performance document gives the correct 2.25-bpw value.

The reconstruction path captures hidden states and outputs onto CPU, but captured keyword tensors are not detached or moved, which can retain device allocations across calibration passes. It also switches the reconstructed block to training mode even though dropout-free deterministic block reconstruction normally benefits from evaluation mode.

The vendored Metal documentation identifies the source only as a local filesystem path and copy date. For an upstream submission, it should identify an immutable public commit or release in addition to the license.

The compare also removes upstream `CODE_OF_CONDUCT.md` and `SECURITY.md`; those should be restored before opening an upstream pull request.

---

# How close this is to the IQ1-inspired design

The branch already supplies most of the surrounding infrastructure, but not the central quantizer.

## Present today

* Scalar ternary QAT.
* Distillation and healing.
* Exact activation quantization.
* Code-flip monitoring.
* CPU and Metal execution.
* GGUF export infrastructure.
* TQ2/I2_S parity decoders.
* Pair-index LUT inference.
* Blockwise reconstruction.
* Benchmarking discipline.

## Still missing

* A restricted (K=2048) or (K=4096) codebook of eight-trit vectors.
* Codeword assignment rather than independent scalar rounding.
* Importance-weighted or covariance-weighted assignment.
* Exact V11/V12 fake quantization in every QAT forward.
* Codeword-index flip monitoring.
* Eleven- or twelve-bit index packing.
* External row-scale storage.
* A GGUF type for the vector format.
* Joint-codebook or product-codebook inference kernels.
* Exact index/scale export parity.

So the branch is not far away architecturally, but the missing element is the element that creates the compression benefit.

---

# Recommended implementation inside this branch

## Phase 1: Add a first-class quantization specification

Do not continue adding boolean flags to the scalar quantizer. Introduce a serializable specification:

```python
@dataclass(frozen=True)
class QuantSpec:
    scheme: str                 # "scalar_ternary", "tq1_v11", "tq1_v12"
    vector_width: int           # 1 or 8
    codebook_bits: int          # 0, 11, or 12
    scale_granularity: str      # tensor, row, group
    scale_dtype: str            # fp16
    activation_bits: int        # 8
    rounding: str               # half_even
    codebook_sha256: str | None
    format_version: int
```

Hash this complete object together with executable source hashes.

## Phase 2: Implement V12 in pure PyTorch first

Start with 4,096 legal eight-trit vectors:

[
C\in{-1,0,+1}^{4096\times8}
]

Use one FP16 scale per output row:

[
\widehat W_{r,g}
=

\alpha_r C_{k_{r,g}}
]

For QAT:

```python
def tq1_v12_ste(weight, row_scale, codebook):
    z = weight.float() / row_scale[:, None]

    # Conceptual implementation; production version uses candidate lists.
    groups = z.reshape(z.shape[0], -1, 8)
    index = nearest_codeword(groups, codebook)
    projected = codebook[index].reshape_as(weight)

    wq = projected * row_scale[:, None]
    return weight + (wq.to(weight.dtype) - weight).detach(), index
```

Do not form the full `[rows, groups, 4096, 8]` distance tensor. Use:

1. Ordinary ternary initialization.
2. Base-3 encoding into one of 6,561 source patterns.
3. A precomputed candidate list for each source pattern.
4. Exact weighted evaluation over those candidates.

That directly mirrors IQ1’s efficient neighbor-search strategy.

## Phase 3: Reuse the existing healing stack

The current distillation and health monitoring should transfer naturally.

Replace scalar-code monitoring with:

* Codeword-index flip rate.
* Mean changed trits per group.
* Codebook entropy.
* Dead codewords.
* Per-layer exact-hit rate.
* Best-versus-second-best assignment margin.
* Weighted projection error.

The existing teacher KL and blockwise reconstruction are excellent tools for adapting a BitNet checkpoint to the restricted vocabulary.

## Phase 4: Export indices, not reconstructed dense weights

For V12, the model artifact should contain:

* 12-bit index per eight weights.
* FP16 row scale.
* Codebook identifier or codebook data.
* Format/version metadata.

Do not export a dense baked tensor and ask a generic quantizer to rediscover the indices. That recreates the same class of scale and parity problems currently visible in the CPU engine.

The exporter should receive the exact QAT indices:

```text
checkpoint latent weights
        ↓ exact QAT projection
codeword indices + row scales
        ↓ bit packing only
GGUF TQ1_V12
```

The parity test then becomes bit-exact:

```text
checkpoint index == GGUF decoded index
checkpoint scale == GGUF decoded scale
codebook hash == GGUF codebook hash
```

## Phase 5: Use the TL1 work to design the kernel-friendly codebook

The TL1 results are one of the strongest parts of the branch. They show that lookup partial sums can outperform per-weight unpacking decisively.

I would benchmark two V12 codebooks.

### Joint codebook

A general 4,096-entry table of eight-trit vectors.

Expected benefit:

* Best reconstruction quality.
* Straightforward offline learning.
* Possibly more expensive lookup/decode.

### Product codebook

Factor an eight-trit vector into two four-trit halves:

[
C_{i,j}=[A_i,B_j]
]

For example:

* 6 bits for (A_i).
* 5 bits for (B_j).
* 1 sign bit.

Then:

[
a^T C_{i,j}
=

a_{0:4}^T A_i
+
a_{4:8}^T B_j
]

This is much closer to the existing TL1 LUT machinery. It may give up some codebook quality while being considerably easier to vectorize.

The current TL1 implementation uses every possible pair pattern and is therefore lossless. The next experiment is to **restrict combinations across multiple pairs during QAT**, creating the actual compression.

## Phase 6: V11 only after V12 is stable

Once V12:

* Exports bit-exactly.
* Recovers model quality under QAT.
* Has a working CPU reference kernel.
* Shows acceptable lookup cost.

Then reduce the vocabulary to 2,048 codewords:

[
11/8=1.375\text{ index bpw}
]

With external row scales, scale overhead is negligible for normal model widths.

Use mixed assignment if necessary:

* V12 for sensitive attention and down-projection tensors.
* V11 for robust gate/up and expert tensors.
* Higher precision for embeddings and output head.

---

# How I would split this for upstream review

This is too broad to submit as one monolithic Microsoft BitNet pull request. I would separate it into:

1. **Scalar QAT reference and tests**

   * Quantizer contract.
   * BitLinear.
   * Conversion.
   * Small CPU tests.

2. **Training and distillation**

   * Data.
   * KD.
   * Monitoring.
   * Checkpointing.
   * FSDP corrections.

3. **Export and parity**

   * Baking.
   * GGUF conversion.
   * Strict completeness gate.
   * Pinned llama.cpp integration.

4. **CPU runtime experiments**

   * Scalar oracle.
   * Format A.
   * Base-3.
   * TL1.
   * End-to-end baked parity.

5. **Metal backend**

   * Vendored dependency with immutable provenance.
   * Kernels.
   * MPS tests.

6. **TQ1_V12 research implementation**

   * Codebook.
   * Exact QAT.
   * Packing.
   * Reference kernel.
   * Quality results.

7. **TQ1_V11 optimization**

   * Learned codebook.
   * Mixed-format policy.
   * Optimized kernels.

---

# Bottom line

The branch looks **very promising as infrastructure**. The training stack and kernel methodology are stronger than most early quantization projects.

I would not yet trust:

* CPU-engine model-quality results.
* Preserve-regime export success.
* BF16-memory estimates.
* FSDP checkpoints.
* Exact resume claims.

Those are fixable engineering issues.

Most importantly, this branch validates several premises behind our proposal—especially exact QAT numerics, distillation-based healing, and LUT-oriented ternary kernels—but it has not yet crossed the key conceptual boundary:

> **From independently quantized ternary weights to QAT-trained, restricted vector patterns.**

After fixing the P0 issues, **V12 with an exact eight-trit codebook and row scales is the natural next implementation**, and the existing branch is a strong platform on which to build it.

[1]: https://huggingface.co/docs/accelerate/en/usage_guides/fsdp?utm_source=chatgpt.com "Fully Sharded Data Parallel · Hugging Face"
