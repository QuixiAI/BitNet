#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// K1/K5 — weight_quant_ternary family (docs/new-kernels.md §3): on-device
// latent-weight -> ternary, emitting BOTH forward and backward operands from the
// same scale (STE forward/backward consistency):
//   wq    (E, N, K/32, 10) uint8 — packed `bitnet` blocks {half scale; uint8 qs[8]}
//   w_deq (E, N, K)        bf16  — dequantized ternary for the dense STE backward
//
// Two scale granularities, matching the two plans that consume them:
//   PER-GROUP (A-track baseline, group_k % 32 == 0): one absmean per group_k
//     weights along K. One kernel, one pass. group_k = K gives per-ROW absmean
//     (NOT per-tensor — see below).
//   PER-TENSOR (Q-track/MoE baseline, train_plan §4 / moe_train_plan §3.7): one
//     absmean over the ENTIRE (N, K) slice. Needs a whole-tensor reduction, so it
//     is two kernels: abssum (atomic float accumulate per expert slice) then
//     encode (reads the sum, packs). The batched E axis makes one dispatch cover
//     a whole fused MoE expert stack (per-expert-tensor scales), instead of
//     thousands of per-expert launches.
//
// All kernels are batched over E via grid.y; E = 1 is the plain 2-D case.
// Codes are formed against the fp32 scale; dequant uses that scale rounded to
// half (what the packed block stores and the GEMMs dequantize with).
// ---------------------------------------------------------------------------

// ---- per-group path (one pass). grid (N, E), 32 threads. ----
template <typename T>
kernel void weight_quant_ternary(device const T *W       [[buffer(0)]],
                                 device uchar   *wq      [[buffer(1)]],
                                 device bf16    *w_deq   [[buffer(2)]],
                                 constant int   &K       [[buffer(3)]],
                                 constant int   &group_k [[buffer(4)]],
                                 constant int   &N       [[buffer(5)]],
                                 uint3 tgid [[threadgroup_position_in_grid]],
                                 uint  lane [[thread_index_in_simdgroup]]) {
    const uint row = tgid.x, e = tgid.y;
    const long base    = ((long)e * N + row) * K;
    const int  nblocks = K / 32;
    const int  bpg     = group_k / 32;
    const int  ngroups = K / group_k;
    device uchar* row_blocks = wq + ((long)e * N + row) * nblocks * 10;

    for (int g = 0; g < ngroups; ++g) {
        const long gbase = base + (long)g * group_k;
        float asum = 0.0f;
        for (int b = 0; b < bpg; ++b) {
            asum += fabs(float(W[gbase + b * 32 + (int)lane]));
        }
        asum = simd_sum(asum);
        const float s   = max(asum / float(group_k), 1e-5f);
        const half  sh  = half(s);
        const float inv = 1.0f / s;

        for (int b = 0; b < bpg; ++b) {
            const long idx = gbase + b * 32 + (int)lane;
            const int  q   = int(clamp(rint(float(W[idx]) * inv), -1.0f, 1.0f));
            w_deq[idx] = bf16(float(sh) * float(q));
            const uint code = uint(q + 1);                       // {0,1,2}
            const ushort j  = ushort(4 * (lane & 7));
            const uint   c0 = simd_shuffle(code, j);
            const uint   c1 = simd_shuffle(code, ushort(j + 1));
            const uint   c2 = simd_shuffle(code, ushort(j + 2));
            const uint   c3 = simd_shuffle(code, ushort(j + 3));
            device uchar* blk = row_blocks + (long)(g * bpg + b) * 10;
            if (lane == 0) ((device half*)blk)[0] = sh;          // 10-byte blocks: 2-aligned
            if (lane < 8)  blk[2 + lane] = uchar(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
        }
    }
}

// ---- per-tensor pass 1: |W| sum per expert slice. grid (ceil(N*K/(256*16)), E),
//      256 threads; each thread strides 16 elements (vec4 x4); simdgroup partials
//      land in abssum[e] via atomic float add (fp32 accumulation of ~KB-scale
//      partials — plenty for a scale). Host zeroes abssum first. ----
template <typename T>
kernel void weight_quant_ternary_abssum(device const T *W          [[buffer(0)]],
                                        device atomic_float *abssum [[buffer(1)]],
                                        constant int   &NK         [[buffer(2)]],
                                        uint3 tgid [[threadgroup_position_in_grid]],
                                        uint3 tid  [[thread_position_in_threadgroup]],
                                        uint  lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const uint e = tgid.y;
    device const T* We = W + (long)e * NK;
    const long base = ((long)tgid.x * 256 + tid.x) * 16;
    float asum = 0.0f;
    if (base + 16 <= (long)NK) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            const float4 v = float4(((device const T4*)(We + base))[j]);
            asum += fabs(v.x) + fabs(v.y) + fabs(v.z) + fabs(v.w);
        }
    } else {
        for (long i = base; i < (long)NK; ++i) asum += fabs(float(We[i]));
    }
    asum = simd_sum(asum);
    if (lane == 0 && asum > 0.0f) {
        atomic_fetch_add_explicit(&abssum[e], asum, memory_order_relaxed);
    }
}

// ---- per-tensor pass 2: encode + pack with s = max(abssum/(N*K), 1e-5).
//      grid (N, E), 32 threads; same packing as the per-group kernel, the one
//      scale replicated into every 32-block slot (layout unchanged for the GEMMs). ----
template <typename T>
kernel void weight_quant_ternary_pt_encode(device const T     *W      [[buffer(0)]],
                                           device const float *abssum [[buffer(1)]],
                                           device uchar       *wq     [[buffer(2)]],
                                           device bf16        *w_deq  [[buffer(3)]],
                                           constant int       &K      [[buffer(4)]],
                                           constant int       &N      [[buffer(5)]],
                                           uint3 tgid [[threadgroup_position_in_grid]],
                                           uint  lane [[thread_index_in_simdgroup]]) {
    const uint row = tgid.x, e = tgid.y;
    const long base    = ((long)e * N + row) * K;
    const int  nblocks = K / 32;
    device uchar* row_blocks = wq + ((long)e * N + row) * nblocks * 10;

    const float s   = max(abssum[e] / (float(N) * float(K)), 1e-5f);
    const half  sh  = half(s);
    const float inv = 1.0f / s;

    for (int b = 0; b < nblocks; ++b) {
        const long idx = base + b * 32 + (int)lane;
        const int  q   = int(clamp(rint(float(W[idx]) * inv), -1.0f, 1.0f));
        w_deq[idx] = bf16(float(sh) * float(q));
        const uint code = uint(q + 1);
        const ushort j  = ushort(4 * (lane & 7));
        const uint   c0 = simd_shuffle(code, j);
        const uint   c1 = simd_shuffle(code, ushort(j + 1));
        const uint   c2 = simd_shuffle(code, ushort(j + 2));
        const uint   c3 = simd_shuffle(code, ushort(j + 3));
        device uchar* blk = row_blocks + (long)b * 10;
        if (lane == 0) ((device half*)blk)[0] = sh;
        if (lane < 8)  blk[2 + lane] = uchar(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
    }
}

#define instantiate_weight_quant_ternary(type_name, T)                              \
  template [[host_name("weight_quant_ternary_" #type_name)]] [[kernel]] void        \
  weight_quant_ternary<T>(device const T *W [[buffer(0)]],                          \
                          device uchar *wq [[buffer(1)]],                            \
                          device bf16 *w_deq [[buffer(2)]],                          \
                          constant int &K [[buffer(3)]],                             \
                          constant int &group_k [[buffer(4)]],                       \
                          constant int &N [[buffer(5)]],                             \
                          uint3 tgid [[threadgroup_position_in_grid]],               \
                          uint lane [[thread_index_in_simdgroup]]);                  \
  template [[host_name("weight_quant_ternary_abssum_" #type_name)]] [[kernel]] void \
  weight_quant_ternary_abssum<T>(device const T *W [[buffer(0)]],                    \
                                 device atomic_float *abssum [[buffer(1)]],          \
                                 constant int &NK [[buffer(2)]],                     \
                                 uint3 tgid [[threadgroup_position_in_grid]],        \
                                 uint3 tid [[thread_position_in_threadgroup]],       \
                                 uint lane [[thread_index_in_simdgroup]]);           \
  template [[host_name("weight_quant_ternary_pt_encode_" #type_name)]] [[kernel]]   \
  void weight_quant_ternary_pt_encode<T>(device const T *W [[buffer(0)]],           \
                                         device const float *abssum [[buffer(1)]],  \
                                         device uchar *wq [[buffer(2)]],             \
                                         device bf16 *w_deq [[buffer(3)]],           \
                                         constant int &K [[buffer(4)]],              \
                                         constant int &N [[buffer(5)]],              \
                                         uint3 tgid [[threadgroup_position_in_grid]],\
                                         uint lane [[thread_index_in_simdgroup]]);

instantiate_weight_quant_ternary(float32, float)
instantiate_weight_quant_ternary(float16, half)
instantiate_weight_quant_ternary(bfloat16, bf16)
