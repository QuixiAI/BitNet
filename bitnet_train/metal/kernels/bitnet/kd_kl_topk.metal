#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Sparse-teacher KD-KL fwd/bwd (train_plan §5.1 A6c / moe_train_plan §5.1): the
// distillation term when the teacher is a top-k CACHE — per token, K indices +
// teacher probabilities (already softmax(teacher/τ), computed at cache build).
// Never materializes teacher (T, V) anything; the student (T, V) logits row is
// streamed once per pass, like cross_entropy.
//
// Two tail policies (the config-recorded choice that "changes gradients"):
//   tail_mode 0 — RENORMALIZE: teacher mass renormalized over the k entries.
//     loss = Σ p̃ (log p̃ − log q),        p̃ = p/P,  P = Σp,  q = softmax(z·invtemp)
//     grad = (q − scatter(p̃)) · go · invtemp
//   tail_mode 1 — OTHER-BUCKET: the non-top-k vocab is ONE outcome with teacher
//     mass tail = 1−P and student mass 1−S (S = Σ q at the top-k indices).
//     loss = Σ p (log p − log q) + tail·(log tail − log(1−S))
//     grad = (P·q − scatter(p) + (tail/(1−S))·(scatter(q) − S·q)) · go · invtemp
//
// Temperature: pass invtemp = 1/τ; the caller applies the conventional α·τ²
// factor to loss and grad_out. Negative t_idx entries are padding (skipped).
// One simdgroup per row; grid (Tn, 1, 1), 32 threads. lse is at temperature τ.
// ---------------------------------------------------------------------------

constant float KD_NEG_INF = -3.4028234663852886e38f;
constant float KD_TINY    = 1e-30f;

template <typename T>
kernel void kd_kl_topk_fwd(device const T     *logits [[buffer(0)]],  // (Tn, V)
                           device const int   *t_idx  [[buffer(1)]],  // (Tn, K)
                           device const float *t_prob [[buffer(2)]],  // (Tn, K)
                           device float       *loss   [[buffer(3)]],  // (Tn,)
                           device float       *lse_out[[buffer(4)]],  // (Tn,)
                           constant int   &V         [[buffer(5)]],
                           constant int   &K         [[buffer(6)]],
                           constant float &invtemp   [[buffer(7)]],
                           constant int   &tail_mode [[buffer(8)]],
                           uint row  [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const long kbase = (long)row * K;
    // online (max, sumexp) over the student row at temperature
    float m = KD_NEG_INF, l = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float x = float(logits[base + i]) * invtemp;
        const float nm = max(m, x);
        l = l * exp(m - nm) + exp(x - nm);
        m = nm;
    }
    const float M = simd_max(m);
    l = simd_sum(l * exp(m - M));
    const float lse = M + log(l);

    // pass 1 over the k entries: teacher mass P and student top-k mass S
    float P = 0.0f, S = 0.0f;
    for (int k = (int)lane; k < K; k += 32) {
        const int idx = t_idx[kbase + k];
        if (idx >= 0) {
            P += t_prob[kbase + k];
            S += exp(float(logits[base + idx]) * invtemp - lse);
        }
    }
    P = simd_sum(P); S = simd_sum(S);

    // pass 2: the loss
    float acc = 0.0f;
    const float invP = 1.0f / max(P, KD_TINY);
    for (int k = (int)lane; k < K; k += 32) {
        const int idx = t_idx[kbase + k];
        if (idx < 0) continue;
        const float p = t_prob[kbase + k];
        const float logq = float(logits[base + idx]) * invtemp - lse;
        if (tail_mode == 0) {
            const float pt = p * invP;
            acc += (pt > 0.0f) ? pt * (log(max(pt, KD_TINY)) - logq) : 0.0f;
        } else {
            acc += (p > 0.0f) ? p * (log(max(p, KD_TINY)) - logq) : 0.0f;
        }
    }
    acc = simd_sum(acc);
    if (tail_mode == 1) {
        const float tail = max(1.0f - P, 0.0f);
        if (tail > 0.0f) acc += tail * (log(max(tail, KD_TINY)) - log(max(1.0f - S, KD_TINY)));
    }
    if (lane == 0) { loss[row] = acc; lse_out[row] = lse; }
}

template <typename T>
kernel void kd_kl_topk_bwd(device const T     *logits      [[buffer(0)]],
                           device const int   *t_idx       [[buffer(1)]],
                           device const float *t_prob      [[buffer(2)]],
                           device const float *lse_in      [[buffer(3)]],
                           device const float *grad_out    [[buffer(4)]],  // (Tn,)
                           device T           *grad_logits [[buffer(5)]],  // (Tn, V)
                           constant int   &V         [[buffer(6)]],
                           constant int   &K         [[buffer(7)]],
                           constant float &invtemp   [[buffer(8)]],
                           constant int   &tail_mode [[buffer(9)]],
                           uint row  [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const long kbase = (long)row * K;
    const float lse = lse_in[row];
    const float go = grad_out[row] * invtemp;

    // recompute P and S (cheap: K gathers)
    float P = 0.0f, S = 0.0f;
    for (int k = (int)lane; k < K; k += 32) {
        const int idx = t_idx[kbase + k];
        if (idx >= 0) {
            P += t_prob[kbase + k];
            S += exp(float(logits[base + idx]) * invtemp - lse);
        }
    }
    P = simd_sum(P); S = simd_sum(S);
    const float tail = max(1.0f - P, 0.0f);
    const float tail_c = (tail_mode == 1 && tail > 0.0f) ? tail / max(1.0f - S, KD_TINY) : 0.0f;
    const float qcoef = (tail_mode == 0) ? 1.0f : (P - tail_c * S);   // multiplies q everywhere
    const float invP = 1.0f / max(P, KD_TINY);

    // dense term: grad_j = qcoef * q_j (top-k entries corrected below)
    for (int i = (int)lane; i < V; i += 32) {
        const float q = exp(float(logits[base + i]) * invtemp - lse);
        grad_logits[base + i] = T(qcoef * q * go);
    }
    simdgroup_barrier(mem_flags::mem_device);

    // sparse correction at the teacher's indices
    for (int k = (int)lane; k < K; k += 32) {
        const int idx = t_idx[kbase + k];
        if (idx < 0) continue;
        const float p = t_prob[kbase + k];
        const float q = exp(float(logits[base + idx]) * invtemp - lse);
        float corr;
        if (tail_mode == 0) corr = -(p * invP);
        else                corr = -p + tail_c * q;
        grad_logits[base + idx] = T(float(grad_logits[base + idx]) + corr * go);
    }
}

#define instantiate_kd_kl_topk(type_name, T)                                        \
  template [[host_name("kd_kl_topk_fwd_" #type_name)]] [[kernel]] void              \
  kd_kl_topk_fwd<T>(device const T *logits [[buffer(0)]],                           \
                    device const int *t_idx [[buffer(1)]],                           \
                    device const float *t_prob [[buffer(2)]],                        \
                    device float *loss [[buffer(3)]], device float *lse_out [[buffer(4)]], \
                    constant int &V [[buffer(5)]], constant int &K [[buffer(6)]],    \
                    constant float &invtemp [[buffer(7)]],                           \
                    constant int &tail_mode [[buffer(8)]],                           \
                    uint row [[threadgroup_position_in_grid]],                       \
                    uint lane [[thread_index_in_simdgroup]]);                        \
  template [[host_name("kd_kl_topk_bwd_" #type_name)]] [[kernel]] void              \
  kd_kl_topk_bwd<T>(device const T *logits [[buffer(0)]],                           \
                    device const int *t_idx [[buffer(1)]],                           \
                    device const float *t_prob [[buffer(2)]],                        \
                    device const float *lse_in [[buffer(3)]],                        \
                    device const float *grad_out [[buffer(4)]],                      \
                    device T *grad_logits [[buffer(5)]],                             \
                    constant int &V [[buffer(6)]], constant int &K [[buffer(7)]],    \
                    constant float &invtemp [[buffer(8)]],                           \
                    constant int &tail_mode [[buffer(9)]],                           \
                    uint row [[threadgroup_position_in_grid]],                       \
                    uint lane [[thread_index_in_simdgroup]]);

instantiate_kd_kl_topk(float32, float)
instantiate_kd_kl_topk(float16, half)
instantiate_kd_kl_topk(bfloat16, bf16)
