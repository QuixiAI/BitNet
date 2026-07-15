# Conclusion

After reading the implementation, your intuition is stronger than I initially realized:

> **IQ1 is already a ternary vector quantizer.**

Its 2,048-entry grid consists of eight-element vectors whose entries are exactly ({-1,0,+1}). IQ1’s additional machinery exists mainly because it must approximate arbitrary floating-point weights: importance weighting, local scales, a small signed offset, neighbor search, and compact packing.

For BitNet, I would reuse the vector-codebook and importance-aware ideas, but redesign the bit allocation around the fact that the source weights are already ternary.

My proposed family is:

* **TQ1_V11:** 11 bits per eight ternary weights, using a 2,048-entry codebook.
* **TQ1_V12:** 12 bits per eight ternary weights, using a 4,096-entry codebook.
* Optional **affine recovery mode** if post-training quality is more important than preserving strict ternary weights.
* QAT against the exact codebook and exact runtime representation.

The first two retain genuinely ternary arithmetic. With native BitNet row scales, they require approximately **1.375 and 1.500 bits per weight**, respectively.

---

# 1. What IQ1 actually does

All source references in this section refer to your uploaded archive.

## 1.1 The IQ1 grid is ternary

The relevant constants are:

```c
#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f
```

The runtime table `iq1s_grid` contains 2,048 eight-byte vectors. Each byte is interpreted as signed `int8_t`, giving entries of (-1), (0), or (+1).

See:

* `ggml/src/ggml-common.h:1131–1135`
* `ggml/src/ggml-quants.c:2911–3106`

Each group of eight weights therefore selects:

[
\mathbf c_k \in {-1,0,+1}^8,\qquad k\in[0,2047]
]

using an 11-bit index.

Because:

[
3^8=6561
]

there are 6,561 possible ternary eight-vectors, but only 2,048 are directly representable by IQ1.

I enumerated the complete `kgrid_1bit_2048` table from the archive against all 6,561 possible ternary vectors. Under squared Euclidean distance in trit units:

| Nearest-codeword distance | Number of ternary vectors |
| ------------------------: | ------------------------: |
|       (0), exact codeword |                     2,048 |
|                       (1) |                     4,252 |
|                       (2) |                       261 |
|          Greater than (2) |                         0 |

So the IQ1 table behaves as a very good generic covering code: every ternary pattern can be reached by at most two one-step trit changes.

It is not completely sign-symmetric, however. Only 1,331 of its 2,048 entries have their negated vector also present. That is one area where a BitNet-specific codebook could improve.

## 1.2 IQ1_S bit allocation

The layout is:

```c
// 1.5625 bpw
typedef struct {
    ggml_half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;
```

See `ggml/src/ggml-common.h:424–430`.

For 256 weights:

* 32 low index bytes: 256 bits.
* 32 codewords × 3 high index bits: 96 bits.
* Eight 32-weight subblocks × 3-bit local scale: 24 bits.
* Eight 32-weight subblocks × 1 offset-sign bit: 8 bits.
* One FP16 superblock scale: 16 bits.

Total:

[
256+96+24+8+16=400\text{ bits}
]

and:

[
400/256=1.5625\text{ bpw}
]

Its decoded weight is:

[
\widehat w_i
=

d,(2s_g+1)\left(c_{k,i}+\epsilon_g\frac18\right)
]

where:

* (c_{k,i}\in{-1,0,+1}),
* (s_g\in{0,\ldots,7}), producing odd scale multipliers (1,3,\ldots,15),
* (\epsilon_g\in{-1,+1}),
* the scale and offset sign are shared by 32 weights.

See `ggml/src/ggml-quants.c:2650–2673`.

So although the codeword is ternary, **the final IQ1_S weights are not strictly ternary** because of the scale and (\pm 1/8) offset.

## 1.3 IQ1_M spends more metadata for finer correction

IQ1_M uses:

* A local scale for every 16 weights.
* An independent offset sign for every eight weights.
* The same 11-bit codeword index.
* A global FP16 scale hidden in unused high nibbles of the scale array.

See:

* `ggml/src/ggml-common.h:432–444`
* `ggml/src/ggml-quants.c:2675–2723`
* `ggml/src/ggml-quants.c:4692–4943`

That gives 1.75 bpw.

## 1.4 The quantization process is more important than the packing

The IQ1_S quantizer does the following for each 256-weight block.

### Importance-weighted objective

It derives per-weight objective weights as:

[
h_i
=

I_i\sqrt{\sigma^2+x_i^2}
]

where (I_i) comes from the imatrix and:

[
\sigma^2=\frac{2}{256}\sum_i x_i^2
]

See `ggml/src/ggml-quants.c:4550–4558`.

The imatrix itself accumulates activation second moments:

[
I_i \approx \mathbb E[x_i^2]
]

See `tools/imatrix/imatrix.cpp:369–389`.

This approximates the fact that errors on heavily activated input channels affect the layer output more.

### Exact three-level scalar assignment

For every 32-weight IQ1_S subblock, the quantizer sorts the weights and searches all pairs of split boundaries. These boundaries divide the weights into three labels.

It tests both shifted alphabets:

[
{-1+\delta,\delta,1+\delta}
]

and:

[
{-1-\delta,-\delta,1-\delta}
]

with (\delta=1/8).

For every split, it analytically fits the optimum scale by maximizing:

[
\frac{\left(\sum_i h_i q_i x_i\right)^2}
{\sum_i h_i q_i^2}
]

See `ggml/src/ggml-quants.c:4567–4616`.

### Projection onto the 2,048-vector codebook

The resulting labels define four ternary eight-vectors.

For each eight-vector:

1. Construct its compact base-4 pattern.
2. Look it up directly if it belongs to the grid.
3. Otherwise fetch a precomputed list of nearby grid points.
4. Evaluate those candidates using the actual importance-weighted error.
5. Select the best one.

See `ggml/src/ggml-quants.c:4617–4629`.

The neighbor lists contain the first three distinct Euclidean-distance shells for IQ1. They are built in `ggml/src/ggml-quants.c:3108–3253`.

### Refit after codebook projection

If any eight-vector had to be changed to a nearby codeword, IQ1 refits the continuous subblock scale around the codewords actually selected.

See `ggml/src/ggml-quants.c:4631–4643`.

### Quantize the local scales and pack

Finally, it chooses a global superblock scale, turns the local scales into odd multipliers, and packs the scale and offset sign into the same words as the high index bits.

See `ggml/src/ggml-quants.c:4644–4668`.

IQ1_S multiplies the fitted scale by an empirical factor of 1.125, while IQ1_M uses 1.1125. The source explicitly describes these as empirical correction factors rather than derived optima.

I would avoid carrying those constants into a new format. A final alternating refit after all discrete choices should make them unnecessary.

## 1.5 The dot kernel makes the offset inexpensive

The generic IQ1_S dot product computes:

[
\sum_i a_i\left(c_i+\epsilon\delta\right)
=

\sum_i a_i c_i
+
\epsilon\delta\sum_i a_i
]

The first term is a ternary dot product. The second uses the sums already stored in `block_q8_K`.

See `ggml/src/ggml-cpu/quants.c:1150–1190`.

That is an important design pattern: a seemingly non-ternary correction can be implemented as a cheap activation-sum correction rather than a full floating-point multiply per weight.

---

# 2. How this differs from llama.cpp’s TQ1_0

Current TQ1_0 is essentially scalar ternarization plus base-3 packing:

```c
// 1.6875 bpw
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5];
    uint8_t qh[QK_K/64];
    ggml_half d;
} block_tq1_0;
```

See `ggml/src/ggml-common.h:271–281`.

Its quantizer:

1. Finds the absolute maximum over 256 weights.
2. Divides by that value.
3. Rounds every scalar independently to (-1,0,+1).
4. Packs five trits into most bytes.
5. Does not use the imatrix.

See `ggml/src/ggml-quants.c:2314–2418`.

For a model whose weights are already exactly ternary times a scale, this is mostly a storage format. It does not exploit correlations between adjacent trits and does not adapt the model to a restricted pattern vocabulary.

BitNet b1.58 itself uses ternary weights, with its training quantizer based on absmean scaling rather than the per-block absolute maximum used by this TQ1_0 converter. ([arXiv][1])

---

# 3. Proposed format: Ternary Vector Quantization for BitNet

I would call the overall method **TQ1_V**, for “ternary vector quantization.”

The fundamental unit remains eight trits:

[
\mathbf c_k\in{-1,0,+1}^8
]

but only selected joint patterns are legal.

## 3.1 TQ1_V11: 2,048 patterns

Each eight-weight group receives an 11-bit index.

For 256 weights:

[
32\times11=352\text{ bits}=44\text{ bytes}
]

A generic GGML block with its own FP16 scale would be:

```c
#define QK_TQ1_V 256

// 1.4375 bpw
typedef struct {
    ggml_half d;
    uint8_t qs[QK_TQ1_V/8];        // low 8 bits of each index
    uint8_t qh[3*QK_TQ1_V/64];     // packed high 3 bits
} block_tq1_v11;

static_assert(sizeof(block_tq1_v11) == 46);
```

The size is:

[
\frac{46\times8}{256}=1.4375\text{ bpw}
]

That is about 14.8% smaller than TQ1_0.

For a native BitNet model whose scale is held externally, omit `d`:

```c
typedef struct {
    uint8_t qs[QK_TQ1_V/8];
    uint8_t qh[3*QK_TQ1_V/64];
} block_tq1_v11_native;
```

That is:

[
44\times8/256=1.375\text{ bpw}
]

or about 18.5% smaller than TQ1_0.

## 3.2 TQ1_V12: 4,096 patterns

For a safer first implementation, use 12 bits per eight weights:

[
32\times12=384\text{ bits}=48\text{ bytes}
]

A generic block can reuse the physical footprint of IQ1_S:

```c
// 1.5625 bpw
typedef struct {
    ggml_half d;
    uint8_t  qs[QK_K/8];       // low 8 bits
    uint16_t qh[QK_K/32];      // four high nibbles per 32 weights
} block_tq1_v12;
```

Each `qh[ib]` contains the high four bits of four codeword indices:

```c
idx0 = qs[4*ib + 0] | (((qh[ib] >>  0) & 0x0f) << 8);
idx1 = qs[4*ib + 1] | (((qh[ib] >>  4) & 0x0f) << 8);
idx2 = qs[4*ib + 2] | (((qh[ib] >>  8) & 0x0f) << 8);
idx3 = qs[4*ib + 3] | (((qh[ib] >> 12) & 0x0f) << 8);
```

With an external BitNet scale, omit `d`, giving exactly:

[
1.5\text{ bpw}
]

This format spends the same number of bits as IQ1_S’s native payload, but allocates all four metadata bits per 32 weights to a twelfth codeword bit. In other words:

* IQ1_S: 2,048 patterns plus scale and offset correction.
* TQ1_V12: 4,096 strictly ternary patterns.

For an already-ternary source model, I expect this to be a very useful comparison.

## 3.3 Why a restricted codebook is necessarily lossy without QAT

An arbitrary eight-trit group has 6,561 possibilities.

A lossless fixed-width index needs:

[
\left\lceil \log_2 6561 \right\rceil=13\text{ bits}
]

Thirteen bits per eight weights is:

[
13/8=1.625\text{ raw bpw}
]

Adding one FP16 scale per 256 weights gives:

[
1.625+16/256=1.6875\text{ bpw}
]

which is exactly the storage rate of TQ1_0.

Therefore:

* V11 and V12 are lossy for arbitrary existing ternary matrices.
* They become exact storage formats for a model trained or fine-tuned so every eight-weight group belongs to their codebook.

That is the central role of QAT.

---

# 4. Use external per-row scales rather than block scales

The BitNet model implementation in this archive already loads separate scale tensors such as `wq_s`, `wk_s`, and `ffn_down_s`:

* `src/models/bitnet.cpp:26–43`

The graph applies the scale after matrix multiplication:

* `src/llama-graph.cpp:1382–1390`

The current scale tensors are scalar, but I would extend this to one FP16 or BF16 scale per output row:

[
\widehat W_{r,i}=\alpha_r C_{k(r,i),i}
]

The scale overhead is:

[
\frac{16}{n_{\text{in}}}\text{ bpw}
]

For an input width of 4,096:

[
16/4096=0.00390625\text{ bpw}
]

That is far cheaper than a 16-bit scale every 256 weights, which costs 0.0625 bpw.

Per-row scaling also gives the quantizer significantly more freedom while preserving:

* Strict ternary dot products.
* One output multiplication per row.
* Easy fusion into the matrix-multiplication epilogue.

For an initial upstreamable implementation, retaining `d` inside each block is simpler. For a native BitNet implementation, external row scales are the better eventual design.

---

# 5. Codebook design

## 5.1 First baseline: reuse IQ1’s existing grid

The fastest baseline is simply:

```c
codebook_v11 = iq1s_grid;
```

Advantages:

* Already present in llama.cpp.
* Known neighbor-map infrastructure.
* Every possible ternary eight-vector is within squared trit distance two.
* Existing IQ1 backend kernels provide useful implementation templates.

This establishes whether the general idea works before spending time on codebook learning.

It should not be assumed to be optimal for BitNet, because it was designed as a generic covering grid rather than from BitNet pattern frequencies and activation sensitivity.

## 5.2 Recommended production codebook: sign-canonical shapes

For a BitNet-specific codebook, exploit the fact that if a pattern is useful, its negation is usually useful too.

Canonicalize every nonzero ternary vector so that its first nonzero trit is positive. Store:

* A canonical shape identifier.
* One global sign bit for the eight-vector.

### V11

Use:

* 10-bit shape ID: 1,024 shapes.
* 1-bit sign.

This represents:

[
1+2(1024-1)=2047
]

unique vectors; the duplicate encoding of the all-zero shape can be reserved.

### V12

Use:

* 11-bit shape ID: 2,048 shapes.
* 1-bit sign.

This represents 4,095 unique vectors.

A shape can be stored as two eight-bit masks:

```c
typedef struct {
    uint8_t positive_mask;
    uint8_t negative_mask;
} tq_shape8;
```

Thus the static shape tables require only:

* V11: (1024\times2=2) KiB.
* V12: (2048\times2=4) KiB.

An expanded `int8_t[8]` table would require 8 KiB or 16 KiB.

At runtime, negating a codeword merely swaps its positive and negative masks or negates the accumulated dot product.

This gives exact sign symmetry, unlike the current IQ1 grid, and resembles the sign/mirror consolidation that efficient ternary lookup kernels already exploit. The Bitnet.cpp work shows that representation and SIMD-friendly packing have to be designed jointly rather than treating file size as the only objective. ([arXiv][2])

## 5.3 Learn the codebook with weighted discrete facility location

All possible codewords come from a small universe of only 6,561 vectors. Codebook construction can therefore be treated as a discrete optimization problem rather than unconstrained neural codebook learning.

Let (u) be an observed eight-trit source pattern and (H_u) its importance metric. Select a set (C) by minimizing:

[
\min_{\substack{C\subset{-1,0,+1}^8\|C|=K}}
\sum_u p(u)
\min_{c\in C}
(u-c)^T H_u(u-c)
+
\lambda_{\mathrm{nz}}|c|_0
]

where:

* (p(u)) includes empirical frequency.
* (H_u) comes from activation statistics.
* The optional nonzero penalty encourages hardware-friendly sparse patterns.
* Sign symmetry is enforced by the canonical-shape representation.

A practical solver would be:

1. Aggregate patterns from representative BitNet checkpoints.
2. Normalize contributions so large FFN tensors do not completely dominate.
3. Initialize with the most frequent patterns plus required anchors:

   * Zero.
   * Dense sign patterns.
   * One-hot and low-density patterns.
4. Run weighted k-medoids or greedy facility-location swaps.
5. Evaluate the resulting codebook on held-out models.
6. Freeze one universal compile-time codebook for the first implementation.

Model-specific codebooks could improve quality, but they would require passing a model-owned codebook pointer through GGML kernels. That is substantially more invasive than a static table and should be a later version.

## 5.4 Kernel-first alternative: factorized product codebooks

A joint 2,048- or 4,096-vector table maximizes representational quality. A factorized codebook may provide faster inference.

Split each eight-trit group into two four-trit halves.

### V11 product layout

Use:

* 5-bit first-half code.
* 5-bit second-half code.
* 1-bit global sign.

That provides:

[
32\times32\times2=2048
]

patterns.

### V12 product layout

Use:

* 5-bit first-half code.
* 6-bit second-half code.
* 1-bit global sign.

That provides:

[
32\times64\times2=4096
]

patterns.

For each activation group, construct small lookup tables:

[
L_A[j]=\mathbf a_{0:4}^T A_j
]

[
L_B[j]=\mathbf a_{4:8}^T B_j
]

Then every weight group evaluates as:

[
\operatorname{sign}\left(L_A[i_A]+L_B[i_B]\right)
]

This converts the inner loop from eight scalar ternary operations into two small table lookups and an addition. It is directly compatible with the LUT-centric direction explored by Bitnet.cpp. ([arXiv][2])

I would benchmark two profiles:

* **J profile:** joint codebook, best expected quality.
* **P profile:** product codebook, best expected CPU lookup performance.

They use the same 11- or 12-bit budget but require separately trained or fine-tuned models.

---

# 6. Post-training quantization algorithm

For strict TQ1_V, the objective for a row is:

[
\min_{\alpha_r,{k_g}}
\sum_g
\left(
W_{r,g}-\alpha_r C_{k_g}
\right)^T
H_g
\left(
W_{r,g}-\alpha_r C_{k_g}
\right)
]

where:

* (g) identifies an eight-weight group.
* (C_{k_g}) is a ternary codeword.
* (H_g) is an importance metric.
* (\alpha_r) is the row scale.

For the generic block-scale version, replace (\alpha_r) with one (d_b) per 256-weight block.

## 6.1 Quantize the latent weights directly when available

A BitNet training checkpoint generally has latent real-valued weights whose forward pass produces ternary weights. The best exporter should project those latent weights directly into the vector codebook.

Do not unnecessarily perform:

[
W_{\rm latent}
\rightarrow
Q_{\rm ordinary\ ternary}
\rightarrow
Q_{\rm codebook}
]

when it can instead solve:

[
W_{\rm latent}
\rightarrow
Q_{\rm codebook\ ternary}
]

directly.

If only exported ternary weights are available, use those as the target.

BitNet’s standard ternarization uses absmean scaling and nearest-trit rounding. That remains a good initializer, but it should not constrain the final vector assignment. ([arXiv][1])

## 6.2 Importance metric

For the first implementation, copy IQ1’s metric:

[
h_i=I_i\sqrt{\sigma^2+W_i^2}
]

This provides a controlled comparison with IQ1.

A cleaner BitNet-specific baseline is simply:

[
h_i=I_i
]

because (I_i) already approximates the diagonal of the input covariance.

## 6.3 Better version: collect (8\times8) activation covariance

The expected squared output error of one row is:

[
\mathbb E[(\Delta w^T x)^2]
=

\Delta w^T\Sigma_x\Delta w
]

The current imatrix keeps only the diagonal of (\Sigma_x).

For each aligned group of eight input channels, collect:

[
H_g=\mathbb E[x_gx_g^T]\in\mathbb R^{8\times8}
]

Then candidate error becomes:

[
D(k)=
(W_g-\alpha C_k)^T
H_g
(W_g-\alpha C_k)
]

This captures correlations between lanes inside the exact unit being vector-quantized.

It remains inexpensive as an offline calibration artifact, and the optimum scale for a fixed codeword still has a closed form:

[
\alpha^*
=

\frac{C_k^T H_g W_g}
{C_k^T H_g C_k}
]

For one shared row scale:

[
\alpha_r^*
=

\frac{\sum_g C_{k_g}^T H_g W_g}
{\sum_g C_{k_g}^T H_g C_{k_g}}
]

## 6.4 Fast candidate search

Every ordinary ternary group can be encoded as a base-3 integer:

[
u=\sum_{i=0}^{7}(q_i+1)3^i
]

so the map needs only 6,561 entries.

For each source pattern, precompute:

* Exact codeword index, when present.
* Otherwise, 16–32 nearby codewords.
* Optionally, candidates from the first three distinct distance shells, mirroring IQ1.

During quantization, evaluate those candidates using the real importance metric.

The neighbor maps are needed only by the quantizer, not inference.

## 6.5 Alternating optimization

A practical quantizer is a Lloyd-style alternating solver:

```text
initialize alpha from the BitNet scale or weighted absmean

repeat 2–4 times:
    for every 8-weight group:
        normalize by alpha
        obtain an ordinary ternary initializer
        fetch exact or nearby codebook candidates
        select the candidate minimizing importance-weighted error

    refit alpha analytically from all selected codewords

optionally:
    run one final reassignment after alpha is rounded to FP16/BF16

pack codeword indices and alpha
```

More explicitly:

```c
for each row r {
    float alpha = initial_scale(W[r]);

    for (int iteration = 0; iteration < 3; ++iteration) {
        for each 8-weight group g {
            ternary_pattern u = initialize_pattern(W[r][g], alpha);

            candidates = neighbour_table[u];

            best_idx = -1;
            best_error = INFINITY;

            for (idx in candidates) {
                codeword c = codebook[idx];

                error = quadratic_error(
                    W[r][g] - alpha*c,
                    H[g]
                );

                if (error < best_error) {
                    best_error = error;
                    best_idx = idx;
                }
            }

            indices[g] = best_idx;
        }

        alpha = refit_row_scale(W[r], indices, H);
    }

    pack(indices, alpha);
}
```

## 6.6 Add GPTQ-style error feedback as a second-stage enhancement

The block-covariance objective ignores correlations between different eight-channel groups.

A higher-quality PTQ implementation could quantize groups sequentially and propagate their error into unquantized groups using a Hessian inverse approximation:

[
W_{g+1:}
\leftarrow
W_{g+1:}
-

\Delta W_g
H_{gg}^{-1}
H_{g,g+1:}
]

This would combine:

* IQ-style vector codebooks.
* Imatrix or Hessian sensitivity.
* GPTQ-style error compensation.

It is likely more useful for V11 than V12.

## 6.7 Do not use unexplained scale multipliers

After codeword assignment and FP16 scale rounding:

1. Re-evaluate the actual decoded representation.
2. Analytically refit the remaining continuous scale.
3. Reassign once if needed.

This is preferable to inheriting IQ1’s fixed 1.125 or 1.1125 empirical factors.

---

# 7. QAT design

V11 will probably benefit substantially from QAT. V12 may work acceptably with PTQ, but QAT is still the cleaner formulation.

The training-time quantizer must exactly emulate the deployed format.

## 7.1 Forward pass

Keep latent BF16 or FP32 weights (V).

For each row:

[
\alpha_r=\operatorname{softplus}(a_r)
]

or initialize (\alpha_r) from the ordinary BitNet absmean scale.

Normalize:

[
Z_{r,g}=V_{r,g}/\alpha_r
]

Project onto the codebook:

[
k_{r,g}
=

\arg\min_k
\left|
Z_{r,g}-C_k
\right|_{H_g}^2
]

Then use:

[
\widehat W_{r,g}
=

\alpha_r C_{k_{r,g}}
]

in every forward pass.

Activations should continue to use the model’s normal BitNet activation quantization so the training path matches inference.

## 7.2 Backward pass

The simplest straight-through estimator is:

[
W_{\rm STE}
=

V+\operatorname{stopgrad}(\widehat W-V)
]

Forward evaluation sees (\widehat W), while gradients flow approximately through (V).

A smoother warm-up can use only the nearest (M) codeword candidates:

[
p_k
=

\operatorname{softmax}(-D_k/\tau)
]

[
C_{\rm soft}
=

\sum_k p_k C_k
]

and the straight-through hard assignment:

[
C_{\rm ST}
=

C_{\rm hard}
+
C_{\rm soft}
-

\operatorname{stopgrad}(C_{\rm soft})
]

As temperature (\tau) decreases, the assignments become discrete.

## 7.3 Training objective

I would use:

[
\mathcal L
=

\mathcal L_{\rm LM}
+
\lambda_{\rm KL}\mathcal L_{\rm distill}
+
\lambda_h\mathcal L_{\rm hidden}
+
\lambda_m\mathcal L_{\rm margin}
]

where:

* (\mathcal L_{\rm LM}): ordinary language-model loss.
* (\mathcal L_{\rm distill}): logit KL against the original BitNet.
* (\mathcal L_{\rm hidden}): selected hidden-state or layer-output matching.
* (\mathcal L_{\rm margin}): encourages the best codeword to remain clearly better than the second-best codeword.

The margin term reduces index churn late in training and makes the exported discrete model more stable.

## 7.4 Curriculum

A robust schedule would be:

### Phase A: initialization

* Start from an existing BitNet checkpoint.
* Initialize row scales from its original scales.
* Initialize every group to its nearest codeword.
* Keep the original model as a frozen teacher.

### Phase B: soft projection

* Use top-(M) soft candidates.
* Gradually increase the codebook commitment.
* Train scales, latent weights, and norms.

### Phase C: hard projection

* Use hard codeword assignments in every forward pass.
* Continue STE and distillation.
* Refresh candidate lists from the current latent patterns.

### Phase D: freeze indices

* Freeze codeword indices.
* Fine-tune only:

  * Row scales.
  * Norm parameters.
  * Embeddings and output head if desired.
  * A temporary LoRA adapter, which can later be merged and reprojected.

### Phase E: exact export validation

* Pack the indices using the real C exporter.
* Run the llama.cpp dequantizer or integer kernel.
* Confirm bit-exact logits within the expected activation-quantization tolerance.

Once the indices are frozen, there is no post-training codebook-projection loss: the trained model is already exactly representable by the runtime format.

---

# 8. Optional affine recovery format

If PTQ quality is poor and QAT is unavailable, add IQ1-like correction metadata.

Using four metadata bits per 32 weights keeps the native payload at 1.5 bpw:

* Two bits for a small local scale.
* Two bits for offset mode.

I would not copy IQ1’s (1,3,\ldots,15) scale range verbatim. BitNet already has a meaningful global or row scale, so local scales should remain near one.

A hardware-friendly choice is:

[
\rho\in
\left{
\frac68,\frac78,\frac88,\frac98
\right}
]

and:

[
\mu\in
\left{
0,+\frac18,-\frac18
\right}
]

giving:

[
\widehat w
=

\alpha_r\rho_g(c+\mu_g)
]

The integer kernel can evaluate:

[
\rho_g
\left(
c^Ta+\mu_g\sum_i a_i
\right)
]

using the same activation-sum technique as IQ1.

This is likely useful as a PTQ fallback, but it should be treated as a different tradeoff:

* Better approximation flexibility.
* No longer strictly ternary weights.
* More local scaling work in the kernel.

My preference is to try V12 and QAT before adding affine correction.

---

# 9. Kernel design

## 9.1 Direct joint-codebook kernel

The simplest generic CPU kernel mirrors `ggml_vec_dot_iq1_s_q8_K_generic`, but removes the local-scale and delta paths:

```c
for each 256-weight block {
    int32_t block_sum = 0;

    for each 8-weight group {
        uint32_t index = unpack_index(block, group);

        uint32_t shape = index & SHAPE_MASK;
        int sign = index & SIGN_BIT ? -1 : +1;

        const int8_t * c = tq1_shape_grid[shape];

        int32_t group_sum = dot_i8x8(q8, c);
        block_sum += sign * group_sum;

        q8 += 8;
    }

    result += weight_scale * activation_scale * block_sum;
}
```

Compared with IQ1:

* No odd local-scale multiply.
* No offset sum correction.
* No subblock accumulation hierarchy.
* Only codeword lookup, signed ternary dot, and final scale.

For V12, the physical block is almost a drop-in structural sibling of IQ1_S.

## 9.2 Positive/negative mask kernel

With the mask table:

[
c^Ta
=
\sum_{i\in P}a_i
-
\sum_{i\in N}a_i
]

This can be implemented using:

* Scalar bit iteration as a reference.
* SIMD-expanded codewords.
* Architecture-specific shuffle or dot-product instructions.
* A small mask-to-expanded-vector table.

The sign bit simply swaps (P) and (N).

## 9.3 Product-codebook LUT kernel

For the factorized format:

```c
lut_a[id] = dot4(q8 + 0, codebook_a[id]);
lut_b[id] = dot4(q8 + 4, codebook_b[id]);

group_sum = lut_a[id_a] + lut_b[id_b];

if (negative) {
    group_sum = -group_sum;
}
```

The activation lookup tables are built once for an input tile and reused over many output rows.

This is the design most likely to compete with the specialized TL family on CPUs, but the codebook restriction is stronger and therefore needs its own QAT experiment.

Bitnet.cpp’s results are a warning that dense bit packing alone does not guarantee speed: aligned data layout, lookup construction, SIMD compatibility, and matrix-shape handling are decisive. Its I2_S and TL implementations specifically target fast, lossless ternary inference rather than merely minimum file size. ([arXiv][2])

## 9.4 GPU path

For the joint codebook:

* Keep the canonical-shape table in constant memory.
* Load an eight-trit shape per index.
* Apply the sign to the accumulation rather than negating the vector.
* Decode several neighboring indices per thread to amortize bit extraction.

For prompt processing, a backend-private tiled repack may be beneficial. For token generation, retaining the packed representation may be better because weight bandwidth dominates.

The two cases should be benchmarked independently:

* Batch-one decoding.
* Batched prompt evaluation.

---

# 10. llama.cpp implementation map

## Type and block declarations

Modify:

* `ggml/include/ggml.h`

  * Add `GGML_TYPE_TQ1_V11`.
  * Add `GGML_TYPE_TQ1_V12`.
  * Append enum values rather than reusing removed IDs.

* `ggml/src/ggml-common.h`

  * Add block structs.
  * Add codebook shape tables.
  * Add compile-time size assertions.

## Quantization and dequantization

Modify:

* `ggml/src/ggml-quants.h`

  * Declarations for quantizers and dequantizers.

* `ggml/src/ggml-quants.c`

  * Base-3 ternary pattern encoding.
  * Exact map and neighbor-list support.
  * Weighted candidate search.
  * Row/block scale refitting.
  * Index packing.
  * Reference dequantizers.

I would create a new generalized helper rather than further overloading `iq2_data`, whose assertions and fixed four-entry storage are currently tied to specific IQ types.

## Generic GGML traits and dispatch

Modify:

* `ggml/src/ggml.c`

  * Type traits.
  * Type sizes and block sizes.
  * Quantization initialization and cleanup.
  * `ggml_quantize_chunk()` dispatch.
  * Mark V11 as requiring an imatrix for PTQ.
  * Probably require one for V12 initially as well.

## CPU kernels

Modify:

* `ggml/src/ggml-cpu/quants.h`
* `ggml/src/ggml-cpu/quants.c`
* `ggml/src/ggml-cpu/ggml-cpu.c`

Add:

* `ggml_vec_dot_tq1_v11_q8_K`
* `ggml_vec_dot_tq1_v12_q8_K`

Then add optimized versions under:

* `ggml/src/ggml-cpu/arch/x86/quants.c`
* `ggml/src/ggml-cpu/arch/arm/quants.c`
* Other architectures after correctness is established.

## GPU and secondary backends

Port from the IQ1 paths in:

* `ggml/src/ggml-cuda`
* `ggml/src/ggml-metal`
* `ggml/src/ggml-vulkan`
* `ggml/src/ggml-sycl`
* `ggml/src/ggml-webgpu`

V12 should be ported first because its block layout is closest to IQ1_S.

## GGUF and CLI integration

Modify:

* `include/llama.h`
* `src/llama-quant.cpp`
* `src/llama-model-loader.cpp`
* `tools/quantize/quantize.cpp`
* `gguf-py/gguf/constants.py`
* `gguf-py/gguf/quants.py`

Add file types such as:

* `LLAMA_FTYPE_MOSTLY_TQ1_V11`
* `LLAMA_FTYPE_MOSTLY_TQ1_V12`

The quantizer should permit mixed formats by tensor.

## Native per-row BitNet scales

Modify:

* `src/models/bitnet.cpp`
* The converter that emits BitNet scale tensors.
* Potentially `build_lora_mm` scale broadcasting or a fused matmul epilogue.

Support both:

* Existing one-element scalar scale tensors.
* New output-channel scale vectors.

## Tests

Extend:

* `tests/test-quantize-fns.cpp`
* `tests/test-quant-type-selection.cpp`
* `tests/test-backend-ops.cpp`

Add focused tests for:

1. Exact packing and unpacking of all boundary-crossing indices.
2. Every codebook entry round-trips exactly.
3. Reference dot product equals dequantize-then-FP32 dot.
4. Sign-bit behavior.
5. Zero, dense, and sparse codewords.
6. Random Q8_K activations.
7. V11/V12 file-size assertions.
8. Cross-backend numerical agreement.
9. Quantization determinism with the same imatrix.
10. Native row-scale broadcasting.

---

# 11. Recommended development sequence

## Phase 0: prove the training idea with existing IQ1_S

Before adding a new GGML type, build a fake-quantized BitNet training path whose forward representation exactly matches IQ1_S:

* Existing 2,048-vector ternary grid.
* Existing odd local scales.
* Existing (\pm1/8) offset.
* Existing block layout.

Export `block_iq1_s` directly rather than running the heuristic converter afterward.

This gives an immediate answer to:

> Can a BitNet checkpoint adapt to an IQ-style vector codebook under QAT?

It will not preserve strict ternary weights, but it requires almost no new inference work.

## Phase 1: implement V12 reference CPU support

Implement TQ1_V12 first because:

* It has the same 50-byte generic block size as IQ1_S.
* Its index packing is simple.
* It uses 4,096 ternary patterns.
* Its kernel is simpler than IQ1_S.
* It is the safer PTQ and QAT target.

Use:

* Static joint sign-canonical codebook.
* FP16 scale per 256 weights initially.
* Diagonal imatrix weighting.
* Generic CPU kernel.

## Phase 2: native row scales

Remove the scale from the weight blocks for native BitNet models and use per-output-row scale tensors.

This changes V12 from 1.5625 to approximately 1.5 bpw and V11 from 1.4375 to approximately 1.375 bpw.

## Phase 3: implement V11

Reuse the validated quantizer and QAT machinery, but use:

* 1,024 canonical shapes.
* 10-bit shape ID plus sign.
* Tight 11-bit index packing.

V11 is the more ambitious compression target and is more likely to require QAT.

## Phase 4: codebook learning and full covariance

Compare:

* Existing IQ1 grid.
* Frequency-trained joint codebook.
* Sensitivity-trained joint codebook.
* Product codebook.
* Diagonal imatrix versus (8\times8) covariance.
* PTQ versus QAT.

## Phase 5: optimized kernels

Only after the quality frontier is understood:

* AVX2/AVX-512.
* ARM NEON/dot-product.
* Product-codebook LUT kernels.
* CUDA and Metal.
* Backend-specific runtime repacking.

This prevents optimizing a codebook structure that later proves impossible to train effectively.

---

# 12. Evaluation matrix

The essential baselines are:

| Baseline                                | Purpose                              |
| --------------------------------------- | ------------------------------------ |
| Original BitNet lossless representation | Quality and speed ceiling            |
| llama.cpp TQ1_0                         | Current 1.6875-bpw ternary baseline  |
| TQ2_0                                   | Aligned 2-bit ternary baseline       |
| Bitnet.cpp I2_S or TL                   | Specialized ternary-kernel baseline  |
| IQ1_S applied to latent BitNet weights  | Affine vector-codebook baseline      |
| V12 PTQ                                 | Strict ternary, moderate compression |
| V12 QAT                                 | Strict ternary, adapted model        |
| V11 PTQ                                 | Aggressive strict ternary            |
| V11 QAT                                 | Main research target                 |
| Product-codebook V11/V12                | Kernel-first alternative             |

Measure three distinct classes of outcomes.

### Representation quality

* Importance-weighted reconstruction error.
* Output activation MSE on calibration data.
* Exact codebook-hit rate before projection.
* Average changed trits per eight-weight group.
* Codeword usage distribution.
* Per-layer sensitivity.

### Model quality

* Validation perplexity.
* Logit KL from the original BitNet.
* Standard downstream tasks.
* Long-context behavior.
* Quantization degradation by tensor family.

### Systems performance

* Actual file bpw including scales and codebook.
* Resident memory after any runtime repacking.
* Batch-one tokens per second.
* Prompt-evaluation tokens per second.
* Energy per token.
* Codebook-table cache misses.
* Index-unpacking instruction count.
* CPU and GPU separately.

A format that saves 15% of weight traffic but adds 30% decoding work is not a successful format. The kernel comparison has to be part of the quantizer design from the beginning.

---

# Final recommendation

The most compelling implementation is:

1. **TQ1_V12 first:** 4,096 strictly ternary eight-weight patterns, 1.5 native bpw, sign-canonical codebook, row scales.
2. **Exact codebook-aware QAT:** the deployed projection runs in every forward pass.
3. **TQ1_V11 second:** 2,048 patterns, 1.375 native bpw, distilled from the V12 or original BitNet model.
4. **Use IQ1’s grid as a baseline, not necessarily the final codebook.**
5. **Replace IQ1’s heuristic scale multipliers with exact alternating refits.**
6. **Collect (8\times8) activation covariance as an enhanced imatrix.**
7. **Benchmark a product-codebook profile for LUT-centric CPU kernels.**
8. **Use mixed V11/V12 tensor selection rather than forcing the most aggressive format on every layer.**

The crucial reframing is:

> This is not “quantizing a ternary scalar again.” It is quantizing the joint distribution of eight trits, then using QAT to make those joint restrictions native to the model.

That makes sub-1.58-bpw BitNet storage entirely plausible while retaining genuinely ternary computation. The unresolved question is empirical rather than mathematical: whether the model can reorganize its ternary patterns enough under QAT that V11’s 2,048-pattern vocabulary delivers a better total quality–memory–speed point than existing lossless ternary formats.

[1]: https://arxiv.org/html/2402.17764v1 "https://arxiv.org/html/2402.17764v1"
[2]: https://arxiv.org/html/2502.11880v1 "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs"
