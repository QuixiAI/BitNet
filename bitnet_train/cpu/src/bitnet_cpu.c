// bitnet_cpu — the K-track CPU engine kernels (moe_train_plan §7.3), first cut.
//
// Scope: batch-1 decode primitives for ternary experts + 8-bit non-experts, plus the
// prefill unpack. Every kernel has a SCALAR reference implementation (the permanent
// correctness oracle — moe_train_plan: "scalar reference kernel first, kept forever")
// and, where it matters, a NEON version selected at runtime via the *_neon entry
// points. The Python ctypes wrapper (bitnet_cpu.py) diffs NEON against scalar on
// random inputs before trusting it — same discipline as the Metal tree.
//
// Weight format: the same `bitnet` packed blocks as the Metal/GGUF-adjacent tooling —
// ternary {-1,0,+1}, group 32 along K, 10 bytes/block { fp16 scale; uint8 qs[8] },
// code in {0,1,2}, value = scale * (code-1). Per-tensor-scale checkpoints simply
// carry the same scale in every block (moe_train_plan §3.7 baseline), which the GEMV
// exploits via the deferred-scale fast path when told to (pt=1).
//
// Activations: per-token absmax int8, clamp [-127,127] (the kernel-matched convention
// of the training stack — the §7.3 "activation contract").
//
// Build: clang -O3 -shared (see build.sh). Threading: none inside these kernels —
// callers parallelize across experts/rows (decode-time parallelism is across the 8
// active experts; keep the kernel single-core-measurable for the roofline).

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define BN_BLOCK_K 32
#define BN_BLOCK_BYTES 10

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static inline float bn_half_to_float(const uint8_t *p) {
#if defined(__ARM_NEON)
    __fp16 h;
    memcpy(&h, p, 2);
    return (float)h;
#else
    // portable fp16 -> fp32
    uint16_t u; memcpy(&u, p, 2);
    uint32_t sign = (uint32_t)(u & 0x8000) << 16;
    uint32_t exp = (u >> 10) & 0x1F, mant = u & 0x3FF;
    float v;
    if (exp == 0) v = ldexpf((float)mant, -24);
    else if (exp == 31) v = mant ? NAN : INFINITY;
    else v = ldexpf((float)(mant | 0x400), (int)exp - 25);
    uint32_t bits; memcpy(&bits, &v, 4); bits |= sign; memcpy(&v, &bits, 4);
    return v;
#endif
}

static inline float bn_bfloat16_to_float(uint16_t value) {
    const uint32_t bits = (uint32_t) value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float bn_e4m3_lut[256];
static int8_t bn_b3_lut[256][5];           // base-3 byte -> 5 ternary values {-1,0,+1}
static int bn_e4m3_ready = 0;

void bn_init(void) {                       // build the decode LUTs (idempotent)
    if (bn_e4m3_ready) return;
    for (int b = 0; b < 256; b++) {
        int sign = (b >> 7) & 1, exp = (b >> 3) & 0xF, mant = b & 7;
        float v;
        if (exp == 0)               v = ldexpf((float)mant, -9);          // subnormal: m/8 * 2^-6
        else if (exp == 15 && mant == 7) v = 0.0f;                        // NaN encoding -> 0
        else                        v = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
        bn_e4m3_lut[b] = sign ? -v : v;
    }
    for (int b = 0; b < 256; b++) {        // bytes >= 243 are never emitted; decode as 0
        int t = b;
        for (int i = 0; i < 5; i++) {
            bn_b3_lut[b][i] = b < 243 ? (int8_t)(t % 3 - 1) : 0;
            t /= 3;
        }
    }
    bn_e4m3_ready = 1;
}

// ---------------------------------------------------------------------------
// activation quant: per-token absmax int8, clamp +/-127, round-half-even
// ---------------------------------------------------------------------------

static inline float bn_round_half_even(float value) {
#if defined(__has_builtin)
#if __has_builtin(__builtin_roundevenf)
    // Unlike rintf, this is independent of the process floating-point rounding
    // mode.  Clang lowers it to vectorizable ties-to-even instructions.
    return __builtin_roundevenf(value);
#endif
#endif
    // Portable fallback. TQ1 activation quotients are clamped to int8 below,
    // so the parity test is exact throughout the finite supported range.
    float lower = floorf(value);
    const float fraction = value - lower;
    if (fraction > 0.5f ||
        (fraction == 0.5f && fmodf(fabsf(lower), 2.0f) == 1.0f)) {
        lower += 1.0f;
    }
    return lower;
}

float bn_act_quant_int8(const float *x, int64_t K, int8_t *xq) {
    float amax = 0.0f;
    for (int64_t i = 0; i < K; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    const float s = amax / 127.0f;
    for (int64_t i = 0; i < K; i++) {
        float r = s > 0.0f ? bn_round_half_even(x[i] / s) : 0.0f;
        if (r > 127.0f) r = 127.0f; else if (r < -127.0f) r = -127.0f;
        xq[i] = (int8_t)r;
    }
    return s;
}

// ---------------------------------------------------------------------------
// ternary GEMV (decode): out[n] = a_scale * sum_g gscale_g * sum_{k in g} (code-1)*xq[k]
// pt != 0 asserts per-tensor scales (all blocks share block 0's scale) -> the integer
// accumulation defers the scale to one multiply per row.
// ---------------------------------------------------------------------------

void bn_gemv_w2a8_scalar(const uint8_t *wq, int64_t N, int64_t K, const int8_t *xq,
                         float a_scale, int pt, float *out) {
    const int64_t bpr = K / BN_BLOCK_K;
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = wq + n * bpr * BN_BLOCK_BYTES;
        float acc = 0.0f;
        int64_t iacc = 0;
        for (int64_t g = 0; g < bpr; g++) {
            const uint8_t *blk = row + g * BN_BLOCK_BYTES;
            const int8_t *x = xq + g * BN_BLOCK_K;
            int isum = 0;
            for (int j = 0; j < 8; j++) {
                const uint8_t b = blk[2 + j];
                isum += (((b >> 0) & 3) - 1) * (int)x[4 * j + 0];
                isum += (((b >> 2) & 3) - 1) * (int)x[4 * j + 1];
                isum += (((b >> 4) & 3) - 1) * (int)x[4 * j + 2];
                isum += (((b >> 6) & 3) - 1) * (int)x[4 * j + 3];
            }
            if (pt) iacc += isum;
            else    acc += bn_half_to_float(blk) * (float)isum;
        }
        if (pt) acc = bn_half_to_float(row) * (float)iacc;
        out[n] = acc * a_scale;
    }
}

#if defined(__ARM_NEON)
// NEON: vld4 de-interleaves x into the 4 code planes of each byte (plane s multiplies
// x[s], x[s+4], ...), so 2-bit extraction is a shift+mask per plane, the multiply is a
// widening vmull_s8, and the {0,1,2} bias is removed via the plane-summed x (code-1).
static inline int32_t bn_block_dot_neon(const uint8_t *blk, const int8_t *x) {
    const uint8x8_t qs = vld1_u8(blk + 2);
    const int8x8x4_t xd = vld4_s8(x);
    const uint8x8_t m3 = vdup_n_u8(3);
    int16x8_t acc = vdupq_n_s16(0), xs = vdupq_n_s16(0);
    #define BN_PLANE(s, extract)                                                     \
        {                                                                            \
            const int8x8_t c = vreinterpret_s8_u8(vand_u8((extract), m3));           \
            acc = vmlal_s8(acc, c, xd.val[s]);                                       \
            xs = vaddw_s8(xs, xd.val[s]);                                            \
        }
    BN_PLANE(0, qs)                       // vshr_n_u8 rejects shift 0
    BN_PLANE(1, vshr_n_u8(qs, 2))
    BN_PLANE(2, vshr_n_u8(qs, 4))
    BN_PLANE(3, vshr_n_u8(qs, 6))
    #undef BN_PLANE
    return vaddlvq_s16(acc) - vaddlvq_s16(xs);      // codes {0,1,2} -> {-1,0,+1}
}

void bn_gemv_w2a8_neon(const uint8_t *wq, int64_t N, int64_t K, const int8_t *xq,
                       float a_scale, int pt, float *out) {
    const int64_t bpr = K / BN_BLOCK_K;
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = wq + n * bpr * BN_BLOCK_BYTES;
        float acc = 0.0f;
        int64_t iacc = 0;
        for (int64_t g = 0; g < bpr; g++) {
            const uint8_t *blk = row + g * BN_BLOCK_BYTES;
            const int32_t isum = bn_block_dot_neon(blk, xq + g * BN_BLOCK_K);
            if (pt) iacc += isum;
            else    acc += bn_half_to_float(blk) * (float)isum;
        }
        if (pt) acc = bn_half_to_float(row) * (float)iacc;
        out[n] = acc * a_scale;
    }
}
#endif

void bn_gemv_w2a8(const uint8_t *wq, int64_t N, int64_t K, const int8_t *xq,
                  float a_scale, int pt, float *out) {
#if defined(__ARM_NEON)
    bn_gemv_w2a8_neon(wq, N, K, xq, a_scale, pt, out);
#else
    bn_gemv_w2a8_scalar(wq, N, K, xq, a_scale, pt, out);
#endif
}

// ---------------------------------------------------------------------------
// TQ1_V schema-2 GEMV (Llama decode): canonical payload + expanded codebook
//
// The canonical artifact stores 32 8-weight indices in each 256-weight block.
// A backend-private expanded codebook (one int8[8] per physical index) is built
// once by the loader and retained alongside the canonical bytes.  The scalar
// implementation is permanent; the NEON variant changes only dot8, so every
// profile shares the same validation, scale, and affine epilogue.
// ---------------------------------------------------------------------------

enum {
    BN_TQ1_V11_R = 0,
    BN_TQ1_V12_R = 1,
    BN_TQ1_V11_B = 2,
    BN_TQ1_V12_B = 3,
    BN_TQ1_V11_A4_R = 4,
};

static inline uint32_t bn_tq1_index(const uint8_t *block, int index_offset,
                                    int high_bits, int group) {
    const uint8_t *low = block + index_offset;
    const uint8_t *high = low + 32;
    const int bit = group * high_bits;
    const int byte = bit >> 3;
    const int shift = bit & 7;
    uint32_t word = high[byte];
    const int high_bytes = high_bits * 4;
    if (byte + 1 < high_bytes) word |= (uint32_t) high[byte + 1] << 8;
    return (uint32_t) low[group] |
           (((word >> shift) & ((1u << high_bits) - 1u)) << 8);
}

static inline int32_t bn_tq1_dot8_scalar(const int8_t *x, const int8_t *code) {
    int32_t sum = 0;
    for (int lane = 0; lane < 8; lane++) sum += (int32_t) x[lane] * code[lane];
    return sum;
}

#if defined(__ARM_NEON)
static inline int32_t bn_tq1_dot8_neon(const int8_t *x, const int8_t *code) {
    const int8x8_t xv = vld1_s8(x);
    const int8x8_t cv = vld1_s8(code);
    return (int32_t) vaddlvq_s16(vmull_s8(xv, cv));
}
#if defined(__ARM_FEATURE_DOTPROD)
static inline int32_t bn_tq1_dot32_dotprod(const int8_t *x,
                                           const int8_t *c0, const int8_t *c1,
                                           const int8_t *c2, const int8_t *c3) {
    const int8x16_t code01 = vcombine_s8(vld1_s8(c0), vld1_s8(c1));
    const int8x16_t code23 = vcombine_s8(vld1_s8(c2), vld1_s8(c3));
    int32x4_t sum = vdotq_s32(vdupq_n_s32(0), code01, vld1q_s8(x));
    sum = vdotq_s32(sum, code23, vld1q_s8(x + 16));
    return vaddvq_s32(sum);
}
#endif
#endif

// Return 0 on success.  Invalid profile, dimensions, scale, reserved index, or
// affine metadata returns a negative code and never silently decodes garbage.
static int bn_tq1_gemv_impl(const uint8_t *wq, const uint16_t *row_scale,
                            const int8_t *codebook, const uint8_t *legal,
                            int64_t index_count, int64_t N, int64_t K, int profile,
                            int row_scale_bf16,
                            const int8_t *xq, const float *act_scale,
                            int activation_block256, float *out, int use_neon) {
    int high_bits, raw_bytes, block_bytes, row_mode, affine;
    switch (profile) {
        case BN_TQ1_V11_R:    high_bits = 3; raw_bytes = 44; block_bytes = 44; row_mode = 1; affine = 0; break;
        case BN_TQ1_V12_R:    high_bits = 4; raw_bytes = 48; block_bytes = 48; row_mode = 1; affine = 0; break;
        case BN_TQ1_V11_B:    high_bits = 3; raw_bytes = 44; block_bytes = 46; row_mode = 0; affine = 0; break;
        case BN_TQ1_V12_B:    high_bits = 4; raw_bytes = 48; block_bytes = 50; row_mode = 0; affine = 0; break;
        case BN_TQ1_V11_A4_R: high_bits = 3; raw_bytes = 44; block_bytes = 48; row_mode = 1; affine = 1; break;
        default: return -1;
    }
    if (!wq || !row_scale || !codebook || !legal || !xq || !act_scale || !out ||
        N <= 0 || K <= 0 || K % 256 != 0 ||
        index_count != (int64_t) (1u << (high_bits + 8))) return -2;
    const int64_t blocks_per_row = K / 256;
    for (int64_t n = 0; n < N; n++) {
        float ws_row = 1.0f;
        if (row_mode) {
            ws_row = row_scale_bf16 ? bn_bfloat16_to_float(row_scale[n]) :
                                      bn_half_to_float((const uint8_t *) (row_scale + n));
            if (!isfinite(ws_row) || ws_row < 0.0f) return -3;
        }
        int64_t token_acc = 0;
        int64_t token_numerator = 0;
        float scaled_acc = 0.0f;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            const uint8_t *block = wq + (n * blocks_per_row + b) * block_bytes;
            const int index_offset = row_mode ? 0 : 2;
            const float as = act_scale[activation_block256 ? b : 0];
            if (!isfinite(as) || as < 0.0f) return -3;
            int64_t block_acc = 0;
            int64_t block_numerator = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            if (use_neon) {
                for (int subblock = 0; subblock < 8; subblock++) {
                    const int group = subblock * 4;
                    uint32_t index[4];
                    const int8_t *code[4];
                    for (int extra = 0; extra < 4; extra++) {
                        index[extra] = bn_tq1_index(
                            block, index_offset, high_bits, group + extra);
                        if (index[extra] >= (uint32_t) index_count ||
                            !legal[index[extra]]) return -4;
                        code[extra] = codebook + (int64_t) index[extra] * 8;
                    }
                    const int8_t *xa = xq + b * 256 + group * 8;
                    const int64_t dot = bn_tq1_dot32_dotprod(
                        xa, code[0], code[1], code[2], code[3]);
                    if (!affine) {
                        block_acc += dot;
                    } else {
                        const uint8_t byte = block[raw_bytes + (subblock >> 1)];
                        const uint8_t nibble = (subblock & 1) ? (byte >> 4) : (byte & 15);
                        const int rho_num = 6 + (nibble & 3);
                        const int mu_id = (nibble >> 2) & 3;
                        if (nibble > 11 || mu_id == 3) return -5;
                        const int64_t xsum =
                            (int64_t) vaddlvq_s8(vld1q_s8(xa)) +
                            (int64_t) vaddlvq_s8(vld1q_s8(xa + 16));
                        const int mu_num = mu_id == 0 ? 0 : (mu_id == 1 ? 1 : -1);
                        block_numerator += (int64_t) rho_num *
                                           (8 * dot + mu_num * xsum);
                    }
                }
            } else
#endif
            {
            for (int group = 0; group < 32; group++) {
                const uint32_t index = bn_tq1_index(block, index_offset, high_bits, group);
                if (index >= (uint32_t) index_count || !legal[index]) return -4;
                const int8_t *code = codebook + (int64_t) index * 8;
                const int8_t *xa = xq + b * 256 + group * 8;
                int32_t dot;
#if defined(__ARM_NEON)
                dot = use_neon ? bn_tq1_dot8_neon(xa, code) : bn_tq1_dot8_scalar(xa, code);
#else
                (void) use_neon;
                dot = bn_tq1_dot8_scalar(xa, code);
#endif
                if (!affine) {
                    block_acc += dot;
                } else if ((group & 3) == 0) {
                    // A4's metadata controls one 32-weight / four-codeword unit.
                    const int subblock = group >> 2;
                    const uint8_t byte = block[raw_bytes + (subblock >> 1)];
                    const uint8_t nibble = (subblock & 1) ? (byte >> 4) : (byte & 15);
                    const int rho_num = 6 + (nibble & 3);
                    const int mu_id = (nibble >> 2) & 3;
                    if (nibble > 11 || mu_id == 3) return -5;
                    int64_t sub_dot = dot;
                    int64_t xsum = 0;
                    for (int lane = 0; lane < 32; lane++) xsum += xa[lane];
                    for (int extra = 1; extra < 4; extra++) {
                        const int g2 = group + extra;
                        const uint32_t i2 = bn_tq1_index(block, index_offset, high_bits, g2);
                        if (i2 >= (uint32_t) index_count || !legal[i2]) return -4;
                        const int8_t *c2 = codebook + (int64_t) i2 * 8;
                        const int8_t *x2 = xq + b * 256 + g2 * 8;
#if defined(__ARM_NEON)
                        sub_dot += use_neon ? bn_tq1_dot8_neon(x2, c2) : bn_tq1_dot8_scalar(x2, c2);
#else
                        sub_dot += bn_tq1_dot8_scalar(x2, c2);
#endif
                    }
                    const int mu_num = mu_id == 0 ? 0 : (mu_id == 1 ? 1 : -1);
                    block_numerator += (int64_t) rho_num * (8 * sub_dot + mu_num * xsum);
                    group += 3;
                }
            }
            }
            if (affine) {
                if (activation_block256) scaled_acc += ((float) block_numerator / 64.0f) * as;
                else token_numerator += block_numerator;
            } else if (row_mode) {
                if (activation_block256) scaled_acc += (float) block_acc * as;
                else token_acc += block_acc;
            } else {
                const float ws = bn_half_to_float(block);
                if (!isfinite(ws) || ws < 0.0f) return -3;
                scaled_acc += (float) block_acc * ws * (activation_block256 ? as : 1.0f);
            }
        }
        if (affine) {
            out[n] = ws_row * (activation_block256 ? scaled_acc :
                               ((float) token_numerator / 64.0f) * act_scale[0]);
        } else if (row_mode) {
            out[n] = ws_row * (activation_block256 ? scaled_acc :
                               (float) token_acc * act_scale[0]);
        } else {
            out[n] = activation_block256 ? scaled_acc : scaled_acc * act_scale[0];
        }
        if (!isfinite(out[n])) return -6;
    }
    return 0;
}

int bn_tq1_gemv_scalar(const uint8_t *wq, const uint16_t *row_scale,
                       const int8_t *codebook, const uint8_t *legal,
                       int64_t index_count, int64_t N, int64_t K, int profile,
                       int row_scale_bf16,
                       const int8_t *xq, const float *act_scale,
                       int activation_block256, float *out) {
    return bn_tq1_gemv_impl(wq, row_scale, codebook, legal, index_count, N, K,
                            profile, row_scale_bf16, xq, act_scale,
                            activation_block256, out, 0);
}

#if defined(__ARM_NEON)
int bn_tq1_gemv_neon(const uint8_t *wq, const uint16_t *row_scale,
                     const int8_t *codebook, const uint8_t *legal,
                     int64_t index_count, int64_t N, int64_t K, int profile,
                     int row_scale_bf16,
                     const int8_t *xq, const float *act_scale,
                     int activation_block256, float *out) {
    return bn_tq1_gemv_impl(wq, row_scale, codebook, legal, index_count, N, K,
                            profile, row_scale_bf16, xq, act_scale,
                            activation_block256, out, 1);
}
#endif

int bn_tq1_gemv(const uint8_t *wq, const uint16_t *row_scale,
                const int8_t *codebook, const uint8_t *legal,
                int64_t index_count, int64_t N, int64_t K, int profile,
                int row_scale_bf16,
                const int8_t *xq, const float *act_scale,
                int activation_block256, float *out) {
#if defined(__ARM_NEON)
    return bn_tq1_gemv_neon(wq, row_scale, codebook, legal, index_count, N, K,
                            profile, row_scale_bf16, xq, act_scale,
                            activation_block256, out);
#else
    return bn_tq1_gemv_scalar(wq, row_scale, codebook, legal, index_count, N, K,
                              profile, row_scale_bf16, xq, act_scale,
                              activation_block256, out);
#endif
}

static int bn_tq1_gemm_impl(const uint8_t *wq, const uint16_t *row_scale,
                            const int8_t *codebook, const uint8_t *legal,
                            int64_t index_count, int64_t M, int64_t N, int64_t K,
                            int profile, int row_scale_bf16,
                            const int8_t *xq, const float *act_scale,
                            int activation_block256, float *out, int use_neon) {
    if (M <= 0 || !xq || !act_scale || !out) return -2;
    const int64_t scales_per_token = activation_block256 ? K / 256 : 1;
    for (int64_t m = 0; m < M; m++) {
        const int status = bn_tq1_gemv_impl(
            wq, row_scale, codebook, legal, index_count, N, K, profile,
            row_scale_bf16, xq + m * K, act_scale + m * scales_per_token,
            activation_block256, out + m * N, use_neon);
        if (status != 0) return status;
    }
    return 0;
}

int bn_tq1_gemm_scalar(const uint8_t *wq, const uint16_t *row_scale,
                       const int8_t *codebook, const uint8_t *legal,
                       int64_t index_count, int64_t M, int64_t N, int64_t K,
                       int profile, int row_scale_bf16,
                       const int8_t *xq, const float *act_scale,
                       int activation_block256, float *out) {
    return bn_tq1_gemm_impl(wq, row_scale, codebook, legal, index_count, M, N, K,
                            profile, row_scale_bf16, xq, act_scale,
                            activation_block256, out, 0);
}

#if defined(__ARM_NEON)
int bn_tq1_gemm_neon(const uint8_t *wq, const uint16_t *row_scale,
                     const int8_t *codebook, const uint8_t *legal,
                     int64_t index_count, int64_t M, int64_t N, int64_t K,
                     int profile, int row_scale_bf16,
                     const int8_t *xq, const float *act_scale,
                     int activation_block256, float *out) {
    return bn_tq1_gemm_impl(wq, row_scale, codebook, legal, index_count, M, N, K,
                            profile, row_scale_bf16, xq, act_scale,
                            activation_block256, out, 1);
}
#endif

int bn_tq1_gemm(const uint8_t *wq, const uint16_t *row_scale,
                const int8_t *codebook, const uint8_t *legal,
                int64_t index_count, int64_t M, int64_t N, int64_t K,
                int profile, int row_scale_bf16,
                const int8_t *xq, const float *act_scale,
                int activation_block256, float *out) {
#if defined(__ARM_NEON)
    return bn_tq1_gemm_neon(wq, row_scale, codebook, legal, index_count, M, N, K,
                            profile, row_scale_bf16, xq, act_scale,
                            activation_block256, out);
#else
    return bn_tq1_gemm_scalar(wq, row_scale, codebook, legal, index_count, M, N, K,
                              profile, row_scale_bf16, xq, act_scale,
                              activation_block256, out);
#endif
}

// ---------------------------------------------------------------------------
// fused whole-expert FFN (decode): out += w_r * down( silu(gate(x_q)) * up(x_q) )_q
// gate/up share one activation quant and one pass over the intermediate; h (I floats)
// lives on the caller-provided scratch (I <= a few K). The router weighting and the
// accumulate happen in the epilogue — the gather-inside-the-kernel rule: pass each
// selected expert's packed base pointers, no staging copies.
// ---------------------------------------------------------------------------

void bn_expert_ffn_w2a8(const float *x, int64_t H, int64_t I,
                        const uint8_t *gate_wq, const uint8_t *up_wq,
                        const uint8_t *down_wq, int pt, float w_r,
                        int8_t *xq_scratch, int8_t *hq_scratch, float *h_scratch,
                        float *out) {
    const float as = bn_act_quant_int8(x, H, xq_scratch);
    const int64_t bpr = H / BN_BLOCK_K;
    for (int64_t i = 0; i < I; i++) {                 // gate+up fused: one loop, two rows
        const uint8_t *grow = gate_wq + i * bpr * BN_BLOCK_BYTES;
        const uint8_t *urow = up_wq + i * bpr * BN_BLOCK_BYTES;
        float g = 0.0f, u = 0.0f;
        int64_t gi = 0, ui = 0;
        for (int64_t b = 0; b < bpr; b++) {
            const int8_t *xb = xq_scratch + b * BN_BLOCK_K;
            int gs, us;
#if defined(__ARM_NEON)
            gs = bn_block_dot_neon(grow + b * BN_BLOCK_BYTES, xb);
            us = bn_block_dot_neon(urow + b * BN_BLOCK_BYTES, xb);
#else
            gs = 0; us = 0;
            for (int j = 0; j < 8; j++) {
                const uint8_t gb = grow[b * BN_BLOCK_BYTES + 2 + j];
                const uint8_t ub = urow[b * BN_BLOCK_BYTES + 2 + j];
                for (int t = 0; t < 4; t++) {
                    const int xv = (int)xb[4 * j + t];
                    gs += (((gb >> (2 * t)) & 3) - 1) * xv;
                    us += (((ub >> (2 * t)) & 3) - 1) * xv;
                }
            }
#endif
            if (pt) { gi += gs; ui += us; }
            else {
                g += bn_half_to_float(grow + b * BN_BLOCK_BYTES) * (float)gs;
                u += bn_half_to_float(urow + b * BN_BLOCK_BYTES) * (float)us;
            }
        }
        if (pt) { g = bn_half_to_float(grow) * (float)gi; u = bn_half_to_float(urow) * (float)ui; }
        g *= as; u *= as;
        h_scratch[i] = (g / (1.0f + expf(-g))) * u;   // silu(gate) * up
    }
    const float hs = bn_act_quant_int8(h_scratch, I, hq_scratch);
    const int64_t bpr_d = I / BN_BLOCK_K;
    for (int64_t n = 0; n < H; n++) {
        const uint8_t *drow = down_wq + n * bpr_d * BN_BLOCK_BYTES;
        float acc = 0.0f;
        int64_t di = 0;
        for (int64_t b = 0; b < bpr_d; b++) {
            int ds;
#if defined(__ARM_NEON)
            ds = bn_block_dot_neon(drow + b * BN_BLOCK_BYTES, hq_scratch + b * BN_BLOCK_K);
#else
            ds = 0;
            for (int j = 0; j < 8; j++) {
                const uint8_t db = drow[b * BN_BLOCK_BYTES + 2 + j];
                for (int t = 0; t < 4; t++)
                    ds += (((db >> (2 * t)) & 3) - 1) * (int)hq_scratch[b * BN_BLOCK_K + 4 * j + t];
            }
#endif
            if (pt) di += ds;
            else    acc += bn_half_to_float(drow + b * BN_BLOCK_BYTES) * (float)ds;
        }
        if (pt) acc = bn_half_to_float(drow) * (float)di;
        out[n] += w_r * acc * hs;
    }
}

// ---------------------------------------------------------------------------
// TL1-format expert FFN (the formats_bakeoff.md follow-up: gate/up/down on the
// LUT-partial-sum format, projected ~2x the format-A decode). Same decode math
// as bn_expert_ffn_w2a8, but the three matmuls run through bn_gemv_tl1. Weights
// are pre-repacked via bn_pack_tl1 (I and H must be %16). The caller provides
// float scratch g_scratch/u_scratch (I each) + h_scratch (max(I,H)) + the tl1
// lut scratch (K/2*32 int8, sized for max(H,I)). The scalar/NEON split is inside
// bn_gemv_tl1, diffed against the scalar oracle in CI — same discipline as A.
// ---------------------------------------------------------------------------
void bn_gemv_tl1(const uint8_t *wt, int64_t N, int64_t K, const int8_t *xq,
                 float a_scale, int pt, float *out, int8_t *lut_scratch);   // fwd (defined below)

void bn_expert_ffn_tl1(const float *x, int64_t H, int64_t I,
                       const uint8_t *gate_wt, const uint8_t *up_wt,
                       const uint8_t *down_wt, int pt, float w_r,
                       int8_t *xq_scratch, int8_t *hq_scratch, float *g_scratch,
                       float *u_scratch, float *h_scratch, int8_t *lut_scratch,
                       float *out) {
    const float as = bn_act_quant_int8(x, H, xq_scratch);
    bn_gemv_tl1(gate_wt, I, H, xq_scratch, as, pt, g_scratch, lut_scratch);
    bn_gemv_tl1(up_wt,   I, H, xq_scratch, as, pt, u_scratch, lut_scratch);
    for (int64_t i = 0; i < I; i++) {
        const float g = g_scratch[i];
        h_scratch[i] = (g / (1.0f + expf(-g))) * u_scratch[i];   // silu(gate)*up
    }
    const float hs = bn_act_quant_int8(h_scratch, I, hq_scratch);
    // bn_gemv_tl1 applies hs internally (as a_scale), so the epilogue only weights
    bn_gemv_tl1(down_wt, H, I, hq_scratch, hs, pt, g_scratch, lut_scratch);  // reuse g_scratch
    for (int64_t n = 0; n < H; n++) out[n] += w_r * g_scratch[n];
}

// MoE decode step over the selected experts (weights read in place; ids/weights from
// the router). expert_stride_* are BYTE strides between consecutive experts' packs.
void bn_moe_ffn_w2a8(const float *x, int64_t H, int64_t I, int64_t k,
                     const int32_t *expert_ids, const float *expert_w,
                     const uint8_t *gate_base, const uint8_t *up_base,
                     const uint8_t *down_base, int64_t gate_stride, int64_t up_stride,
                     int64_t down_stride, int pt,
                     int8_t *xq_scratch, int8_t *hq_scratch, float *h_scratch,
                     float *out) {
    memset(out, 0, (size_t)H * sizeof(float));
    for (int64_t j = 0; j < k; j++) {
        const int64_t e = expert_ids[j];
        bn_expert_ffn_w2a8(x, H, I,
                           gate_base + e * gate_stride, up_base + e * up_stride,
                           down_base + e * down_stride, pt, expert_w[j],
                           xq_scratch, hq_scratch, h_scratch, out);
    }
}

// ---------------------------------------------------------------------------
// router: softmax over E logits, top-k, renormalize over the selected (norm_topk_prob)
// ---------------------------------------------------------------------------

void bn_route_topk(const float *logits, int64_t E, int64_t k,
                   int32_t *ids, float *weights) {
    float m = logits[0];
    for (int64_t e = 1; e < E; e++) if (logits[e] > m) m = logits[e];
    float Z = 0.0f;
    for (int64_t e = 0; e < E; e++) Z += expf(logits[e] - m);
    // selection sort for top-k (E ~ 128, k ~ 8: trivial)
    uint8_t taken[1024] = {0};
    float wsum = 0.0f;
    for (int64_t j = 0; j < k; j++) {
        int64_t best = -1;
        for (int64_t e = 0; e < E; e++)
            if (!taken[e] && (best < 0 || logits[e] > logits[best])) best = e;
        taken[best] = 1;
        ids[j] = (int32_t)best;
        weights[j] = expf(logits[best] - m) / Z;
        wsum += weights[j];
    }
    for (int64_t j = 0; j < k; j++) weights[j] /= wsum;
}

// ---------------------------------------------------------------------------
// fp8 e4m3 GEMV (attention / lm_head — the OTHER 0.75 GB/token): per-row scale.
//   out[n] = row_scale[n] * sum_k lut[w[n,k]] * x[k]
// ---------------------------------------------------------------------------

void bn_gemv_fp8_scalar(const uint8_t *w, const float *row_scale, int64_t N, int64_t K,
                        const float *x, float *out) {
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = w + n * K;
        float acc = 0.0f;
        for (int64_t j = 0; j < K; j++) acc += bn_e4m3_lut[row[j]] * x[j];
        out[n] = acc * row_scale[n];
    }
}

#if defined(__ARM_NEON)
void bn_gemv_fp8_neon(const uint8_t *w, const float *row_scale, int64_t N, int64_t K,
                      const float *x, float *out) {
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = w + n * K;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        int64_t j = 0;
        for (; j + 8 <= K; j += 8) {                    // LUT stays L1-hot (1 KB)
            const float32x4_t w0 = {bn_e4m3_lut[row[j + 0]], bn_e4m3_lut[row[j + 1]],
                                    bn_e4m3_lut[row[j + 2]], bn_e4m3_lut[row[j + 3]]};
            const float32x4_t w1 = {bn_e4m3_lut[row[j + 4]], bn_e4m3_lut[row[j + 5]],
                                    bn_e4m3_lut[row[j + 6]], bn_e4m3_lut[row[j + 7]]};
            acc0 = vfmaq_f32(acc0, w0, vld1q_f32(x + j));
            acc1 = vfmaq_f32(acc1, w1, vld1q_f32(x + j + 4));
        }
        float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; j < K; j++) acc += bn_e4m3_lut[row[j]] * x[j];
        out[n] = acc * row_scale[n];
    }
}
#endif

void bn_gemv_fp8(const uint8_t *w, const float *row_scale, int64_t N, int64_t K,
                 const float *x, float *out) {
#if defined(__ARM_NEON)
    bn_gemv_fp8_neon(w, row_scale, N, K, x, out);
#else
    bn_gemv_fp8_scalar(w, row_scale, N, K, x, out);
#endif
}

// ---------------------------------------------------------------------------
// int8-KV attention decode (GQA, online softmax): one new-token query against a
// quantized KV cache. K/V (T, Hkv, D) int8 with per-(token, head) scales.
// ---------------------------------------------------------------------------

void bn_attn_decode_kv8(const float *q, const int8_t *kc, const float *k_scale,
                        const int8_t *vc, const float *v_scale,
                        int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                        float *out) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)D);
    const int64_t rep = Hq / Hkv;
    for (int64_t h = 0; h < Hq; h++) {
        const int64_t hk = h / rep;
        const float *qh = q + h * D;
        float m = -3.4e38f, l = 0.0f;
        float *oh = out + h * D;
        for (int64_t d = 0; d < D; d++) oh[d] = 0.0f;
        for (int64_t t = 0; t < T; t++) {
            const int8_t *kt = kc + (t * Hkv + hk) * D;
            float dot = 0.0f;
#if defined(__ARM_NEON)
            float32x4_t acc = vdupq_n_f32(0);
            for (int64_t d = 0; d < D; d += 8) {
                const int16x8_t kw = vmovl_s8(vld1_s8(kt + d));
                acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(kw))), vld1q_f32(qh + d));
                acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(kw))), vld1q_f32(qh + d + 4));
            }
            dot = vaddvq_f32(acc);
#else
            for (int64_t d = 0; d < D; d++) dot += qh[d] * (float)kt[d];
#endif
            const float s = dot * k_scale[t * Hkv + hk] * inv_sqrt_d;
            const float nm = s > m ? s : m;
            const float corr = expf(m - nm);
            const float p = expf(s - nm);
            l = l * corr + p;
            const int8_t *vt = vc + (t * Hkv + hk) * D;
            const float pv = p * v_scale[t * Hkv + hk];
#if defined(__ARM_NEON)
            const float32x4_t corrv = vdupq_n_f32(corr), pvv = vdupq_n_f32(pv);
            for (int64_t d = 0; d < D; d += 8) {
                const int16x8_t vw = vmovl_s8(vld1_s8(vt + d));
                float32x4_t o0 = vld1q_f32(oh + d), o1 = vld1q_f32(oh + d + 4);
                o0 = vfmaq_f32(vmulq_f32(o0, corrv),
                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(vw))), pvv);
                o1 = vfmaq_f32(vmulq_f32(o1, corrv),
                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(vw))), pvv);
                vst1q_f32(oh + d, o0); vst1q_f32(oh + d + 4, o1);
            }
#else
            for (int64_t d = 0; d < D; d++) oh[d] = oh[d] * corr + pv * (float)vt[d];
#endif
            m = nm;
        }
        const float invl = 1.0f / l;
        for (int64_t d = 0; d < D; d++) oh[d] *= invl;
    }
}

// ---------------------------------------------------------------------------
// prefill support: unpack a whole packed tensor to f32 once (dequant-once, then hand
// the chunk to a real GEMM — Accelerate/BLAS — where unpack cost amortizes; §7.4).
// ---------------------------------------------------------------------------

void bn_unpack_ternary_f32(const uint8_t *wq, int64_t N, int64_t K, float *out) {
    const int64_t bpr = K / BN_BLOCK_K;
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = wq + n * bpr * BN_BLOCK_BYTES;
        float *o = out + n * K;
        for (int64_t g = 0; g < bpr; g++) {
            const uint8_t *blk = row + g * BN_BLOCK_BYTES;
            const float s = bn_half_to_float(blk);
            for (int j = 0; j < 8; j++) {
                const uint8_t b = blk[2 + j];
                o[g * 32 + 4 * j + 0] = s * (float)(((b >> 0) & 3) - 1);
                o[g * 32 + 4 * j + 1] = s * (float)(((b >> 2) & 3) - 1);
                o[g * 32 + 4 * j + 2] = s * (float)(((b >> 4) & 3) - 1);
                o[g * 32 + 4 * j + 3] = s * (float)(((b >> 6) & 3) - 1);
            }
        }
    }
}

// ===========================================================================
// DECODE GLUE (moe_train_plan Q-K0: the non-GEMV pieces of an end-to-end step)
// ===========================================================================

// ---------------------------------------------------------------------------
// RMSNorm: out = x / sqrt(mean(x^2) + eps) * w, row-wise over R rows of D.
// QK-norm is the same kernel with R = num_heads, D = head_dim (Qwen3 convention:
// plain RMSNorm weight; Gemma's (1+gamma) callers pass w+1 themselves).
// ---------------------------------------------------------------------------

void bn_rms_norm(const float *x, const float *w, int64_t R, int64_t D, float eps,
                 float *out) {
    for (int64_t r = 0; r < R; r++) {
        const float *xr = x + r * D;
        float *o = out + r * D;
        float ss = 0.0f;
#if defined(__ARM_NEON)
        float32x4_t acc = vdupq_n_f32(0);
        int64_t d = 0;
        for (; d + 4 <= D; d += 4) {
            const float32x4_t v = vld1q_f32(xr + d);
            acc = vfmaq_f32(acc, v, v);
        }
        ss = vaddvq_f32(acc);
        for (; d < D; d++) ss += xr[d] * xr[d];
#else
        for (int64_t d = 0; d < D; d++) ss += xr[d] * xr[d];
#endif
        const float inv = 1.0f / sqrtf(ss / (float)D + eps);
        for (int64_t d = 0; d < D; d++) o[d] = xr[d] * inv * w[d];
    }
}

// ---------------------------------------------------------------------------
// RoPE, NeoX/HF half-split convention (matches transformers apply_rotary_pos_emb
// for Llama/Qwen): pairs (d, d + D/2), angle = pos * theta^(-2d/D). In place,
// per head. Tiny (H*D flops/token) — scalar only.
// ---------------------------------------------------------------------------

void bn_rope_neox(float *x, int64_t H, int64_t D, int64_t pos, float theta) {
    const int64_t half = D / 2;
    for (int64_t d = 0; d < half; d++) {
        const float freq = powf(theta, -2.0f * (float)d / (float)D);
        const float a = (float)pos * freq;
        const float c = cosf(a), s = sinf(a);
        for (int64_t h = 0; h < H; h++) {
            float *xh = x + h * D;
            const float x1 = xh[d], x2 = xh[d + half];
            xh[d] = x1 * c - x2 * s;
            xh[d + half] = x1 * s + x2 * c;
        }
    }
}

// ---------------------------------------------------------------------------
// KV-cache int8 writer: quantize the new token's K/V (Hkv, D) per (token, head)
// absmax (clamp +/-127 — the same convention bn_attn_decode_kv8 reads) and append
// at position pos into the (T, Hkv, D) caches.
// ---------------------------------------------------------------------------

void bn_kv_quant_append(const float *k_new, const float *v_new, int64_t pos,
                        int64_t Hkv, int64_t D,
                        int8_t *kc, float *k_scale, int8_t *vc, float *v_scale) {
    for (int64_t h = 0; h < Hkv; h++) {
        k_scale[pos * Hkv + h] = bn_act_quant_int8(k_new + h * D, D,
                                                   kc + (pos * Hkv + h) * D);
        v_scale[pos * Hkv + h] = bn_act_quant_int8(v_new + h * D, D,
                                                   vc + (pos * Hkv + h) * D);
    }
}

// ===========================================================================
// HEAD GEMVs (Q-A-head8: the lm_head decides its own precision — FP8 already
// exists above; these add the Q8_0 and BF16 contenders)
// ===========================================================================

// ---------------------------------------------------------------------------
// Q8_0-shaped GEMV: blocks of 32 { fp16 d; int8 qs[32] } = 34 B, int8 activations
// (per-token absmax — the engine's activation contract, not llama.cpp's per-block).
//   out[n] = a_scale * sum_g d_g * sum_{k in g} qs[k] * xq[k]
// ---------------------------------------------------------------------------

#define BN_Q8_BLOCK_BYTES 34

void bn_gemv_q8_scalar(const uint8_t *wq, int64_t N, int64_t K, const int8_t *xq,
                       float a_scale, float *out) {
    const int64_t bpr = K / BN_BLOCK_K;
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = wq + n * bpr * BN_Q8_BLOCK_BYTES;
        float acc = 0.0f;
        for (int64_t g = 0; g < bpr; g++) {
            const uint8_t *blk = row + g * BN_Q8_BLOCK_BYTES;
            const int8_t *w = (const int8_t *)(blk + 2);
            const int8_t *x = xq + g * BN_BLOCK_K;
            int isum = 0;
            for (int j = 0; j < BN_BLOCK_K; j++) isum += (int)w[j] * (int)x[j];
            acc += bn_half_to_float(blk) * (float)isum;
        }
        out[n] = acc * a_scale;
    }
}

#if defined(__ARM_NEON)
void bn_gemv_q8_neon(const uint8_t *wq, int64_t N, int64_t K, const int8_t *xq,
                     float a_scale, float *out) {
    const int64_t bpr = K / BN_BLOCK_K;
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = wq + n * bpr * BN_Q8_BLOCK_BYTES;
        float acc = 0.0f;
        for (int64_t g = 0; g < bpr; g++) {
            const uint8_t *blk = row + g * BN_Q8_BLOCK_BYTES;
            const int8_t *w = (const int8_t *)(blk + 2);
            const int8_t *x = xq + g * BN_BLOCK_K;
            const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16);
            const int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16);
#if defined(__ARM_FEATURE_DOTPROD)
            int32x4_t d = vdupq_n_s32(0);
            d = vdotq_s32(d, w0, x0);
            d = vdotq_s32(d, w1, x1);
            const int32_t isum = vaddvq_s32(d);
#else
            int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
            p0 = vmlal_s8(p0, vget_high_s8(w0), vget_high_s8(x0));   // <= 2*127^2, fits
            int16x8_t p1 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
            p1 = vmlal_s8(p1, vget_high_s8(w1), vget_high_s8(x1));
            const int32_t isum = vaddlvq_s16(p0) + vaddlvq_s16(p1);
#endif
            acc += bn_half_to_float(blk) * (float)isum;
        }
        out[n] = acc * a_scale;
    }
}
#endif

void bn_gemv_q8(const uint8_t *wq, int64_t N, int64_t K, const int8_t *xq,
                float a_scale, float *out) {
#if defined(__ARM_NEON)
    bn_gemv_q8_neon(wq, N, K, xq, a_scale, out);
#else
    bn_gemv_q8_scalar(wq, N, K, xq, a_scale, out);
#endif
}

// ---------------------------------------------------------------------------
// BF16 GEMV (weight-only): w uint16 bf16 codes, x f32. Decode = shift to the
// high half of an f32.
// ---------------------------------------------------------------------------

void bn_gemv_bf16_scalar(const uint16_t *w, int64_t N, int64_t K, const float *x,
                         float *out) {
    for (int64_t n = 0; n < N; n++) {
        const uint16_t *row = w + n * K;
        float acc = 0.0f;
        for (int64_t j = 0; j < K; j++) {
            const uint32_t bits = (uint32_t)row[j] << 16;
            float v; memcpy(&v, &bits, 4);
            acc += v * x[j];
        }
        out[n] = acc;
    }
}

#if defined(__ARM_NEON)
void bn_gemv_bf16_neon(const uint16_t *w, int64_t N, int64_t K, const float *x,
                       float *out) {
    for (int64_t n = 0; n < N; n++) {
        const uint16_t *row = w + n * K;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        int64_t j = 0;
        for (; j + 8 <= K; j += 8) {
            const uint16x8_t u = vld1q_u16(row + j);
            const float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u), 16));
            const float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u), 16));
            acc0 = vfmaq_f32(acc0, w0, vld1q_f32(x + j));
            acc1 = vfmaq_f32(acc1, w1, vld1q_f32(x + j + 4));
        }
        float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; j < K; j++) {
            const uint32_t bits = (uint32_t)row[j] << 16;
            float v; memcpy(&v, &bits, 4);
            acc += v * x[j];
        }
        out[n] = acc;
    }
}
#endif

void bn_gemv_bf16(const uint16_t *w, int64_t N, int64_t K, const float *x, float *out) {
#if defined(__ARM_NEON)
    bn_gemv_bf16_neon(w, N, K, x, out);
#else
    bn_gemv_bf16_scalar(w, N, K, x, out);
#endif
}

// ===========================================================================
// PACKING BAKE-OFF (moe_train_plan §7.3: "prototype >=2, measure"). Format A is
// the 10 B/32 2-bit layout above. Both new formats repack FROM format A so the
// tested packer/scales are reused and parity is exact by construction.
// ===========================================================================

// ---------------------------------------------------------------------------
// Format B — base-3 dense: block = { fp16 scale; uint8 t[7] } = 9 B per 32
// weights (1.8 b/w vs A's 2.5). t[j<6] holds 5 trits (c0 + 3c1 + 9c2 + 27c3 +
// 81c4, values 0..242), t[6] holds the last 2. Decode via the bn_b3_lut
// byte->5-trit table (built in bn_init). The min-bytes contender; unpack cost
// is the question the bake-off answers.
// ---------------------------------------------------------------------------

#define BN_B3_BLOCK_BYTES 9

void bn_pack_b3(const uint8_t *wq, int64_t N, int64_t K, uint8_t *out) {
    const int64_t bpr = K / BN_BLOCK_K;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < bpr; g++) {
            const uint8_t *blk = wq + (n * bpr + g) * BN_BLOCK_BYTES;
            uint8_t *o = out + (n * bpr + g) * BN_B3_BLOCK_BYTES;
            o[0] = blk[0]; o[1] = blk[1];                  // fp16 scale, verbatim
            uint8_t c[32];
            for (int j = 0; j < 8; j++)
                for (int t = 0; t < 4; t++)
                    c[4 * j + t] = (blk[2 + j] >> (2 * t)) & 3;
            for (int j = 0; j < 6; j++)
                o[2 + j] = (uint8_t)(c[5 * j] + 3 * c[5 * j + 1] + 9 * c[5 * j + 2]
                                     + 27 * c[5 * j + 3] + 81 * c[5 * j + 4]);
            o[8] = (uint8_t)(c[30] + 3 * c[31]);
        }
    }
}

void bn_gemv_b3_scalar(const uint8_t *wb, int64_t N, int64_t K, const int8_t *xq,
                       float a_scale, int pt, float *out) {
    const int64_t bpr = K / BN_BLOCK_K;
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = wb + n * bpr * BN_B3_BLOCK_BYTES;
        float acc = 0.0f;
        int64_t iacc = 0;
        for (int64_t g = 0; g < bpr; g++) {
            const uint8_t *blk = row + g * BN_B3_BLOCK_BYTES;
            const int8_t *x = xq + g * BN_BLOCK_K;
            int isum = 0;
            for (int j = 0; j < 6; j++) {
                const int8_t *d = bn_b3_lut[blk[2 + j]];
                isum += d[0] * (int)x[5 * j + 0] + d[1] * (int)x[5 * j + 1]
                      + d[2] * (int)x[5 * j + 2] + d[3] * (int)x[5 * j + 3]
                      + d[4] * (int)x[5 * j + 4];
            }
            const int8_t *d = bn_b3_lut[blk[8]];
            isum += d[0] * (int)x[30] + d[1] * (int)x[31];
            if (pt) iacc += isum;
            else    acc += bn_half_to_float(blk) * (float)isum;
        }
        if (pt) acc = bn_half_to_float(row) * (float)iacc;
        out[n] = acc * a_scale;
    }
}

// ---------------------------------------------------------------------------
// Format C — TL1-style LUT-indexed partial sums (the T-SAR-shaped contender the
// perf note names as the next dig). Weights become 4-bit PAIR indices
// (idx = c0 + 3*c1 in [0,9)); per token a 9-entry int16 LUT per k-pair holds the
// precomputed (c0-1)*x[2j] + (c1-1)*x[2j+1], so decode is table lookups + adds —
// no shift/mask/multiply per weight. Layout is row-TILED for NEON tbl:
//
//   tile block = 160 B per (16 rows x 32 k):
//     bytes [0,128): p in 0..8: byte[p*16 + r] = idx(row r, pair 2p)
//                                              | idx(row r, pair 2p+1) << 4
//     bytes [128,160): fp16 scale of (row r, group g) at 128 + 2r
//   blocks ordered [N/16][K/32]. Same 2.5 b/w traffic as format A.
//
// N must be a multiple of 16 (real dims 768/2048/vocab all are).
// ---------------------------------------------------------------------------

#define BN_TL1_TILE_ROWS 16
#define BN_TL1_BLOCK_BYTES 160

void bn_pack_tl1(const uint8_t *wq, int64_t N, int64_t K, uint8_t *out) {
    const int64_t bpr = K / BN_BLOCK_K;
    for (int64_t rt = 0; rt < N / BN_TL1_TILE_ROWS; rt++) {
        for (int64_t g = 0; g < bpr; g++) {
            uint8_t *o = out + (rt * bpr + g) * BN_TL1_BLOCK_BYTES;
            for (int r = 0; r < BN_TL1_TILE_ROWS; r++) {
                const int64_t n = rt * BN_TL1_TILE_ROWS + r;
                const uint8_t *blk = wq + (n * bpr + g) * BN_BLOCK_BYTES;
                uint8_t c[32];
                for (int j = 0; j < 8; j++)
                    for (int t = 0; t < 4; t++)
                        c[4 * j + t] = (blk[2 + j] >> (2 * t)) & 3;
                for (int p = 0; p < 8; p++) {
                    const uint8_t ia = (uint8_t)(c[4 * p + 0] + 3 * c[4 * p + 1]);
                    const uint8_t ib = (uint8_t)(c[4 * p + 2] + 3 * c[4 * p + 3]);
                    o[p * 16 + r] = (uint8_t)(ia | (ib << 4));
                }
                o[128 + 2 * r] = blk[0]; o[128 + 2 * r + 1] = blk[1];
            }
        }
    }
}

// Scalar reference (the permanent oracle): decodes indices arithmetically.
void bn_gemv_tl1_scalar(const uint8_t *wt, int64_t N, int64_t K, const int8_t *xq,
                        float a_scale, int pt, float *out) {
    const int64_t bpr = K / BN_BLOCK_K;
    for (int64_t rt = 0; rt < N / BN_TL1_TILE_ROWS; rt++) {
        float facc[BN_TL1_TILE_ROWS];
        int64_t iacc[BN_TL1_TILE_ROWS];
        for (int r = 0; r < BN_TL1_TILE_ROWS; r++) { facc[r] = 0.0f; iacc[r] = 0; }
        float s0[BN_TL1_TILE_ROWS];
        for (int64_t g = 0; g < bpr; g++) {
            const uint8_t *blk = wt + (rt * bpr + g) * BN_TL1_BLOCK_BYTES;
            const int8_t *x = xq + g * BN_BLOCK_K;
            for (int r = 0; r < BN_TL1_TILE_ROWS; r++) {
                int isum = 0;
                for (int p = 0; p < 8; p++) {
                    const uint8_t b = blk[p * 16 + r];
                    const int ia = b & 0xF, ib = b >> 4;
                    isum += (ia % 3 - 1) * (int)x[4 * p + 0]
                          + (ia / 3 - 1) * (int)x[4 * p + 1]
                          + (ib % 3 - 1) * (int)x[4 * p + 2]
                          + (ib / 3 - 1) * (int)x[4 * p + 3];
                }
                if (pt) iacc[r] += isum;
                else    facc[r] += bn_half_to_float(blk + 128 + 2 * r) * (float)isum;
            }
            if (g == 0)
                for (int r = 0; r < BN_TL1_TILE_ROWS; r++)
                    s0[r] = bn_half_to_float(blk + 128 + 2 * r);
        }
        for (int r = 0; r < BN_TL1_TILE_ROWS; r++)
            out[rt * BN_TL1_TILE_ROWS + r] =
                (pt ? s0[r] * (float)iacc[r] : facc[r]) * a_scale;
    }
}

#if defined(__ARM_NEON)
// LUT build: per k-pair j, 16-byte lo and hi tables of the 9 int16 partial sums
// (entries 9..15 zero). lut_scratch: int8[K/2 * 32] — lo[16] then hi[16] per pair.
void bn_tl1_lut_build(const int8_t *xq, int64_t K, int8_t *lut_scratch) {
    for (int64_t j = 0; j < K / 2; j++) {
        int8_t *lo = lut_scratch + j * 32, *hi = lo + 16;
        const int x0 = xq[2 * j], x1 = xq[2 * j + 1];
        for (int e = 0; e < 16; e++) {
            const int16_t v = e < 9 ? (int16_t)((e % 3 - 1) * x0 + (e / 3 - 1) * x1) : 0;
            lo[e] = (int8_t)(v & 0xFF);
            hi[e] = (int8_t)((v >> 8) & 0xFF);
        }
    }
}

void bn_gemv_tl1_neon(const uint8_t *wt, int64_t N, int64_t K, const int8_t *xq,
                      float a_scale, int pt, float *out, int8_t *lut_scratch) {
    bn_tl1_lut_build(xq, K, lut_scratch);
    const int64_t bpr = K / BN_BLOCK_K;
    const uint8x16_t mlo = vdupq_n_u8(0x0F);
    for (int64_t rt = 0; rt < N / BN_TL1_TILE_ROWS; rt++) {
        float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0),
                    f2 = vdupq_n_f32(0), f3 = vdupq_n_f32(0);
        int32x4_t i0 = vdupq_n_s32(0), i1 = vdupq_n_s32(0),
                  i2 = vdupq_n_s32(0), i3 = vdupq_n_s32(0);
        const uint8_t *tile0 = wt + rt * bpr * BN_TL1_BLOCK_BYTES;
        for (int64_t g = 0; g < bpr; g++) {
            const uint8_t *blk = tile0 + g * BN_TL1_BLOCK_BYTES;
            const int8_t *lut = lut_scratch + g * 16 * 32;   // 16 pairs per group
            int16x8_t a0 = vdupq_n_s16(0), a1 = vdupq_n_s16(0);
            for (int p = 0; p < 8; p++) {
                const uint8x16_t idx = vld1q_u8(blk + p * 16);
                const uint8x16_t ia = vandq_u8(idx, mlo);
                const uint8x16_t ib = vshrq_n_u8(idx, 4);
                const int8_t *la = lut + (2 * p) * 32, *lb = lut + (2 * p + 1) * 32;
                const int8x16_t alo = vqtbl1q_s8(vld1q_s8(la), ia);
                const int8x16_t ahi = vqtbl1q_s8(vld1q_s8(la + 16), ia);
                const int8x16_t blo = vqtbl1q_s8(vld1q_s8(lb), ib);
                const int8x16_t bhi = vqtbl1q_s8(vld1q_s8(lb + 16), ib);
                a0 = vaddq_s16(a0, vreinterpretq_s16_s8(vzip1q_s8(alo, ahi)));
                a1 = vaddq_s16(a1, vreinterpretq_s16_s8(vzip2q_s8(alo, ahi)));
                a0 = vaddq_s16(a0, vreinterpretq_s16_s8(vzip1q_s8(blo, bhi)));
                a1 = vaddq_s16(a1, vreinterpretq_s16_s8(vzip2q_s8(blo, bhi)));
            }
            if (pt) {
                i0 = vaddw_s16(i0, vget_low_s16(a0));
                i1 = vaddw_s16(i1, vget_high_s16(a0));
                i2 = vaddw_s16(i2, vget_low_s16(a1));
                i3 = vaddw_s16(i3, vget_high_s16(a1));
            } else {
                const float16x8_t s01 = vld1q_f16((const __fp16 *)(blk + 128));
                const float16x8_t s23 = vld1q_f16((const __fp16 *)(blk + 144));
                f0 = vfmaq_f32(f0, vcvt_f32_f16(vget_low_f16(s01)),
                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0))));
                f1 = vfmaq_f32(f1, vcvt_f32_f16(vget_high_f16(s01)),
                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(a0))));
                f2 = vfmaq_f32(f2, vcvt_f32_f16(vget_low_f16(s23)),
                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1))));
                f3 = vfmaq_f32(f3, vcvt_f32_f16(vget_high_f16(s23)),
                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(a1))));
            }
        }
        float res[BN_TL1_TILE_ROWS];
        if (pt) {
            const float16x8_t s01 = vld1q_f16((const __fp16 *)(tile0 + 128));
            const float16x8_t s23 = vld1q_f16((const __fp16 *)(tile0 + 144));
            vst1q_f32(res + 0, vmulq_f32(vcvt_f32_f16(vget_low_f16(s01)), vcvtq_f32_s32(i0)));
            vst1q_f32(res + 4, vmulq_f32(vcvt_f32_f16(vget_high_f16(s01)), vcvtq_f32_s32(i1)));
            vst1q_f32(res + 8, vmulq_f32(vcvt_f32_f16(vget_low_f16(s23)), vcvtq_f32_s32(i2)));
            vst1q_f32(res + 12, vmulq_f32(vcvt_f32_f16(vget_high_f16(s23)), vcvtq_f32_s32(i3)));
        } else {
            vst1q_f32(res + 0, f0); vst1q_f32(res + 4, f1);
            vst1q_f32(res + 8, f2); vst1q_f32(res + 12, f3);
        }
        for (int r = 0; r < BN_TL1_TILE_ROWS; r++)
            out[rt * BN_TL1_TILE_ROWS + r] = res[r] * a_scale;
    }
}
#endif

void bn_gemv_tl1(const uint8_t *wt, int64_t N, int64_t K, const int8_t *xq,
                 float a_scale, int pt, float *out, int8_t *lut_scratch) {
#if defined(__ARM_NEON)
    bn_gemv_tl1_neon(wt, N, K, xq, a_scale, pt, out, lut_scratch);
#else
    (void)lut_scratch;
    bn_gemv_tl1_scalar(wt, N, K, xq, a_scale, pt, out);
#endif
}
