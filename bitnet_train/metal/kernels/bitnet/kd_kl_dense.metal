#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// DENSE-teacher KD-KL fwd/bwd (train_plan §5.1 / §9.1 A6b — the full-KL arm of
// the A6b-vs-A6c ablation): loss = KL( softmax(t·invtemp) ‖ softmax(s·invtemp) )
// per row, with LIVE teacher logits (no top-k cache). Fused so the (T, V)
// log-softmax / probability tensors of the PyTorch chunked path are never
// materialized — both logit rows are streamed, per the chunked-losses mandate's
// "or fused equivalents" clause.
//
//   fwd:  loss = Σ_v p_t (log p_t − log q),  p_t = softmax(zt), q = softmax(zs),
//         zt = t·invtemp, zs = s·invtemp; emits both LSEs for the backward.
//   bwd:  d loss / d s_v = go · invtemp · (q_v − p_t,v)
//
// Temperature convention matches kd_kl_topk: pass invtemp = 1/τ, caller applies
// α·τ² to loss / grad_out. One simdgroup per row; grid (Tn, 1, 1), 32 threads.
// ---------------------------------------------------------------------------

constant float KDD_NEG_INF = -3.4028234663852886e38f;

template <typename T>
kernel void kd_kl_dense_fwd(device const T *t_logits [[buffer(0)]],  // (Tn, V)
                            device const T *s_logits [[buffer(1)]],  // (Tn, V)
                            device float   *loss     [[buffer(2)]],  // (Tn,)
                            device float   *lse_t_out[[buffer(3)]],  // (Tn,)
                            device float   *lse_s_out[[buffer(4)]],  // (Tn,)
                            constant int   &V        [[buffer(5)]],
                            constant float &invtemp  [[buffer(6)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    // pass 1: online (max, sumexp) for teacher and student simultaneously
    float mt = KDD_NEG_INF, lt = 0.0f, ms = KDD_NEG_INF, ls = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float zt = float(t_logits[base + i]) * invtemp;
        const float zs = float(s_logits[base + i]) * invtemp;
        float nm = max(mt, zt);
        lt = lt * exp(mt - nm) + exp(zt - nm); mt = nm;
        nm = max(ms, zs);
        ls = ls * exp(ms - nm) + exp(zs - nm); ms = nm;
    }
    const float Mt = simd_max(mt), Ms = simd_max(ms);
    lt = simd_sum(lt * exp(mt - Mt));
    ls = simd_sum(ls * exp(ms - Ms));
    const float lse_t = Mt + log(lt), lse_s = Ms + log(ls);

    // pass 2: Σ p_t · ((zt − lse_t) − (zs − lse_s))
    float acc = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float zt = float(t_logits[base + i]) * invtemp;
        const float zs = float(s_logits[base + i]) * invtemp;
        acc += exp(zt - lse_t) * ((zt - lse_t) - (zs - lse_s));
    }
    acc = simd_sum(acc);
    if (lane == 0) { loss[row] = acc; lse_t_out[row] = lse_t; lse_s_out[row] = lse_s; }
}

template <typename T>
kernel void kd_kl_dense_bwd(device const T     *t_logits [[buffer(0)]],
                            device const T     *s_logits [[buffer(1)]],
                            device const float *lse_t_in [[buffer(2)]],
                            device const float *lse_s_in [[buffer(3)]],
                            device const float *grad_out [[buffer(4)]],  // (Tn,)
                            device T           *grad_s   [[buffer(5)]],  // (Tn, V)
                            constant int   &V       [[buffer(6)]],
                            constant float &invtemp [[buffer(7)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const float lse_t = lse_t_in[row], lse_s = lse_s_in[row];
    const float go = grad_out[row] * invtemp;
    for (int i = (int)lane; i < V; i += 32) {
        const float q  = exp(float(s_logits[base + i]) * invtemp - lse_s);
        const float pt = exp(float(t_logits[base + i]) * invtemp - lse_t);
        grad_s[base + i] = T((q - pt) * go);
    }
}

#define instantiate_kd_kl_dense(type_name, T)                                        \
  template [[host_name("kd_kl_dense_fwd_" #type_name)]] [[kernel]] void              \
  kd_kl_dense_fwd<T>(device const T *t_logits [[buffer(0)]],                         \
                     device const T *s_logits [[buffer(1)]],                         \
                     device float *loss [[buffer(2)]],                               \
                     device float *lse_t_out [[buffer(3)]],                          \
                     device float *lse_s_out [[buffer(4)]],                          \
                     constant int &V [[buffer(5)]],                                  \
                     constant float &invtemp [[buffer(6)]],                          \
                     uint row [[threadgroup_position_in_grid]],                      \
                     uint lane [[thread_index_in_simdgroup]]);                       \
  template [[host_name("kd_kl_dense_bwd_" #type_name)]] [[kernel]] void              \
  kd_kl_dense_bwd<T>(device const T *t_logits [[buffer(0)]],                         \
                     device const T *s_logits [[buffer(1)]],                         \
                     device const float *lse_t_in [[buffer(2)]],                     \
                     device const float *lse_s_in [[buffer(3)]],                     \
                     device const float *grad_out [[buffer(4)]],                     \
                     device T *grad_s [[buffer(5)]],                                 \
                     constant int &V [[buffer(6)]],                                  \
                     constant float &invtemp [[buffer(7)]],                          \
                     uint row [[threadgroup_position_in_grid]],                      \
                     uint lane [[thread_index_in_simdgroup]]);

instantiate_kd_kl_dense(float32, float)
instantiate_kd_kl_dense(float16, half)
instantiate_kd_kl_dense(bfloat16, bf16)
