// Standalone conformance test for the pinned llama.cpp TQ1_V integration.
// Build through quant/llama_cpp/apply_and_test.py; this file is not linked into
// BitNet's own CPU engine.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t index_at(const uint8_t * block, int offset, int high_bits, int group) {
    const uint8_t * low = block + offset;
    const uint8_t * high = low + 32;
    const int bit = group * high_bits;
    const int byte = bit / 8;
    const int shift = bit % 8;
    uint32_t word = high[byte];
    if (byte + 1 < high_bits * 4) word |= (uint32_t) high[byte + 1] << 8;
    return low[group] | (((word >> shift) & ((1u << high_bits) - 1)) << 8);
}

static void set_index(uint8_t * block, int offset, int high_bits, int group, uint32_t index) {
    uint8_t * low = block + offset;
    uint8_t * high = low + 32;
    low[group] = (uint8_t) index;
    const uint32_t value = index >> 8;
    const int bit = group * high_bits;
    for (int lane = 0; lane < high_bits; ++lane) {
        if ((value >> lane) & 1) high[(bit + lane) / 8] |= (uint8_t) (1u << ((bit + lane) % 8));
    }
}

static int8_t round_even(float value) {
    float lower = std::floor(value);
    const float fraction = value - lower;
    if (fraction > 0.5f || (fraction == 0.5f && std::fmod(std::fabs(lower), 2.0f) == 1.0f)) lower += 1;
    return (int8_t) std::max(-127.0f, std::min(127.0f, lower));
}

static int run_profile(enum ggml_type type) {
    constexpr int64_t K = 256;
    constexpr int64_t N = 3;
    constexpr int64_t M = 4;
    const bool v11 = type == GGML_TYPE_TQ1_V11 || type == GGML_TYPE_TQ1_V11_R ||
                     type == GGML_TYPE_TQ1_V11_J_A4_R;
    const bool row_mode = type == GGML_TYPE_TQ1_V11_R || type == GGML_TYPE_TQ1_V12_R ||
                          type == GGML_TYPE_TQ1_V11_J_A4_R;
    const bool affine = type == GGML_TYPE_TQ1_V11_J_A4_R;
    const int high_bits = v11 ? 3 : 4;
    const int raw_bytes = v11 ? 44 : 48;
    const int block_bytes = (int) ggml_type_size(type);
    const int offset = row_mode ? 0 : 2;
    const int index_count = 1 << (high_bits + 8);

    std::vector<int8_t> codebook((size_t) index_count * 8);
    std::vector<uint8_t> legal(index_count, 1);
    for (int index = 0; index < index_count; ++index) {
        for (int lane = 0; lane < 8; ++lane) codebook[(size_t) index * 8 + lane] = (int8_t) ((index + 2*lane) % 3 - 1);
    }
    ggml_tq1_tensor_context binding = {
        GGML_TQ1_CONTEXT_MAGIC, (uint32_t) index_count, codebook.data(), legal.data()};

    std::vector<uint8_t> packed((size_t) N * block_bytes);
    for (int n = 0; n < N; ++n) {
        uint8_t * block = packed.data() + n * block_bytes;
        if (!row_mode) {
            const ggml_fp16_t scale = ggml_fp32_to_fp16(0.375f + 0.125f * n);
            std::memcpy(block, &scale, sizeof(scale));
        }
        for (int group = 0; group < 32; ++group) {
            set_index(block, offset, high_bits, group,
                      (uint32_t) ((17 + 73*group + 211*n) % index_count));
        }
        if (affine) {
            for (int sub = 0; sub < 8; ++sub) {
                const uint8_t nibble = (uint8_t) ((sub + 3*n) % 12);
                block[raw_bytes + sub/2] |= (uint8_t) (nibble << (4*(sub & 1)));
            }
        }
    }

    std::vector<float> activation((size_t) M * K);
    for (size_t i = 0; i < activation.size(); ++i) activation[i] = 1.7f*std::sin(0.013f*(float) i) + 0.2f*std::cos(0.11f*(float) i);
    std::vector<ggml_fp16_t> row_scale(N);
    for (int n = 0; n < N; ++n) row_scale[n] = ggml_fp32_to_fp16(0.25f*(n + 1));

    struct ggml_init_params params = {16u*1024u*1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, K, N);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    std::memcpy(weight->data, packed.data(), packed.size());
    std::memcpy(input->data, activation.data(), activation.size()*sizeof(float));
    weight->extra = &binding;
    if (!ggml_cpu_validate_tq1_tensor(weight)) return 10 + type;

    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    if (row_mode) {
        ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, N);
        std::memcpy(scale->data, row_scale.data(), row_scale.size()*sizeof(ggml_fp16_t));
        output = ggml_mul(ctx, output, ggml_cast(ctx, scale, GGML_TYPE_F32));
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) return 30 + type;

    const float * observed = (const float *) output->data;
    for (int m = 0; m < M; ++m) {
        float maximum = 0;
        for (int k = 0; k < K; ++k) maximum = std::max(maximum, std::fabs(activation[(size_t) m*K + k]));
        const float as = maximum / 127.0f;
        std::vector<int8_t> q(K);
        for (int k = 0; k < K; ++k) q[k] = round_even(activation[(size_t) m*K + k] / as);
        for (int n = 0; n < N; ++n) {
            const uint8_t * block = packed.data() + n*block_bytes;
            int64_t acc = 0;
            int64_t numerator = 0;
            if (affine) {
                for (int sub = 0; sub < 8; ++sub) {
                    int64_t dot = 0, xsum = 0;
                    for (int extra = 0; extra < 4; ++extra) {
                        const int group = 4*sub + extra;
                        const uint32_t index = index_at(block, offset, high_bits, group);
                        for (int lane = 0; lane < 8; ++lane) {
                            dot += q[8*group + lane]*codebook[8*index + lane];
                            xsum += q[8*group + lane];
                        }
                    }
                    const uint8_t nibble = (block[raw_bytes + sub/2] >> (4*(sub & 1))) & 15;
                    const int mu_id = (nibble >> 2) & 3;
                    const int mu = mu_id == 0 ? 0 : mu_id == 1 ? 1 : -1;
                    numerator += (6 + (nibble & 3))*(8*dot + mu*xsum);
                }
            } else {
                for (int group = 0; group < 32; ++group) {
                    const uint32_t index = index_at(block, offset, high_bits, group);
                    for (int lane = 0; lane < 8; ++lane) acc += q[8*group + lane]*codebook[8*index + lane];
                }
            }
            float expected;
            if (affine) expected = as*(float) numerator/64.0f;
            else if (row_mode) expected = as*(float) acc;
            else {
                ggml_fp16_t scale;
                std::memcpy(&scale, block, sizeof(scale));
                expected = as*ggml_fp16_to_fp32(scale)*(float) acc;
            }
            if (row_mode) expected *= ggml_fp16_to_fp32(row_scale[n]);
            const float error = std::fabs(observed[(size_t) m*N + n] - expected);
            if (error > 2e-5f*std::max(1.0f, std::fabs(expected))) {
                std::fprintf(stderr, "type %d m=%d n=%d observed=%g expected=%g error=%g\n",
                             (int) type, m, n, observed[(size_t) m*N + n], expected, error);
                return 50 + type;
            }
        }
    }
    legal[index_at(packed.data(), offset, high_bits, 0)] = 0;
    if (ggml_cpu_validate_tq1_tensor(weight)) return 80 + type;
    ggml_free(ctx);
    return 0;
}

int main() {
    const enum ggml_type profiles[] = {
        GGML_TYPE_TQ1_V11, GGML_TYPE_TQ1_V12, GGML_TYPE_TQ1_V11_R,
        GGML_TYPE_TQ1_V12_R, GGML_TYPE_TQ1_V11_J_A4_R};
    for (enum ggml_type type : profiles) {
        const int status = run_profile(type);
        if (status) return status;
    }
    std::puts("TQ1_V llama.cpp scalar CPU conformance: PASS");
    return 0;
}
