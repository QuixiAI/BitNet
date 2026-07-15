// Focused benchmark for the pinned llama.cpp scalar TQ1_V CPU path.
// The dense baseline uses decoded weights and pre-dequantized A8 activations,
// making it an intentionally optimistic dequantize-then-matmul comparison.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct profile_info {
    enum ggml_type type;
    const char * name;
    bool v11;
    bool row;
    bool affine;
};

static profile_info parse_profile(const std::string & name) {
    if (name == "v11-b") return {GGML_TYPE_TQ1_V11, name.c_str(), true, false, false};
    if (name == "v12-b") return {GGML_TYPE_TQ1_V12, name.c_str(), false, false, false};
    if (name == "v11-r") return {GGML_TYPE_TQ1_V11_R, name.c_str(), true, true, false};
    if (name == "v12-r") return {GGML_TYPE_TQ1_V12_R, name.c_str(), false, true, false};
    if (name == "a4-r") return {GGML_TYPE_TQ1_V11_J_A4_R, name.c_str(), true, true, true};
    std::fprintf(stderr, "unknown profile %s\n", name.c_str());
    std::exit(2);
}

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
        if ((value >> lane) & 1) {
            high[(bit + lane) / 8] |= (uint8_t) (1u << ((bit + lane) % 8));
        }
    }
}

static int8_t round_even(float value) {
    float lower = std::floor(value);
    const float fraction = value - lower;
    if (fraction > 0.5f ||
        (fraction == 0.5f && std::fmod(std::fabs(lower), 2.0f) == 1.0f)) {
        lower += 1.0f;
    }
    return (int8_t) std::max(-127.0f, std::min(127.0f, lower));
}

static double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const size_t index = (size_t) std::llround(p * (double) (values.size() - 1));
    return values[index];
}

static std::vector<double> measure(
        ggml_context * ctx, ggml_cgraph * graph, int threads, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) {
        if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "warmup graph failed\n");
            std::exit(3);
        }
    }
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        const enum ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, threads);
        const auto end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "benchmark graph failed\n");
            std::exit(3);
        }
        samples.push_back(
            std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return samples;
}

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
            "usage: %s <v11-b|v12-b|v11-r|v12-r|a4-r> N K M threads warmup iterations\n",
            argv[0]);
        return 2;
    }
    const std::string profile_name = argv[1];
    const profile_info profile = parse_profile(profile_name);
    const int64_t N = std::atoll(argv[2]);
    const int64_t K = std::atoll(argv[3]);
    const int64_t M = std::atoll(argv[4]);
    const int threads = std::atoi(argv[5]);
    const int warmup = std::atoi(argv[6]);
    const int iterations = std::atoi(argv[7]);
    if (N < 1 || K < 256 || K % 256 || M < 1 || threads < 1 ||
        warmup < 0 || iterations < 5) {
        std::fprintf(stderr, "invalid shape or benchmark controls\n");
        return 2;
    }

    const int high_bits = profile.v11 ? 3 : 4;
    const int index_count = 1 << (high_bits + 8);
    const int index_offset = profile.row ? 0 : 2;
    const int raw_bytes = profile.v11 ? 44 : 48;
    const int block_bytes = (int) ggml_type_size(profile.type);
    const int64_t blocks_per_row = K / 256;

    std::vector<int8_t> codebook((size_t) index_count * 8);
    std::vector<uint8_t> legal(index_count, 1);
    for (int index = 0; index < index_count; ++index) {
        int pattern = index;
        for (int lane = 0; lane < 8; ++lane) {
            codebook[(size_t) index * 8 + lane] = (int8_t) (pattern % 3 - 1);
            pattern /= 3;
        }
    }
    ggml_tq1_tensor_context binding = {
        GGML_TQ1_CONTEXT_MAGIC, (uint32_t) index_count, codebook.data(), legal.data()};

    std::vector<uint8_t> packed((size_t) N * blocks_per_row * block_bytes);
    std::vector<ggml_fp16_t> row_scales(N);
    for (int64_t n = 0; n < N; ++n) {
        row_scales[n] = ggml_fp32_to_fp16(0.03125f * (float) (1 + n % 17));
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            uint8_t * block = packed.data() + (n * blocks_per_row + b) * block_bytes;
            if (!profile.row) {
                const ggml_fp16_t scale = ggml_fp32_to_fp16(
                    0.0625f * (float) (1 + (n + 3*b) % 11));
                std::memcpy(block, &scale, sizeof(scale));
            }
            for (int group = 0; group < 32; ++group) {
                const uint32_t index = (uint32_t)
                    ((17 + 73*group + 211*n + 307*b) % index_count);
                set_index(block, index_offset, high_bits, group, index);
            }
            if (profile.affine) {
                for (int sub = 0; sub < 8; ++sub) {
                    const uint8_t nibble = (uint8_t) ((sub + n + 2*b) % 12);
                    block[raw_bytes + sub/2] |= (uint8_t) (nibble << (4*(sub & 1)));
                }
            }
        }
    }

    std::vector<float> activation((size_t) M * K);
    std::vector<float> activation_a8((size_t) M * K);
    std::vector<int8_t> activation_q((size_t) M * K);
    std::vector<float> activation_scales(M);
    for (size_t i = 0; i < activation.size(); ++i) {
        activation[i] =
            1.7f*std::sin(0.013f*(float) i) + 0.2f*std::cos(0.11f*(float) i);
    }
    for (int64_t m = 0; m < M; ++m) {
        float maximum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            maximum = std::max(maximum, std::fabs(activation[(size_t) m*K + k]));
        }
        const float scale = maximum / 127.0f;
        activation_scales[m] = scale;
        for (int64_t k = 0; k < K; ++k) {
            const int8_t q = round_even(activation[(size_t) m*K + k] / scale);
            activation_q[(size_t) m*K + k] = q;
            activation_a8[(size_t) m*K + k] = scale * q;
        }
    }

    std::vector<float> dense((size_t) N * K);
    for (int64_t n = 0; n < N; ++n) {
        const float row_scale = ggml_fp16_to_fp32(row_scales[n]);
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t * block =
                packed.data() + (n * blocks_per_row + b) * block_bytes;
            float block_scale = 1.0f;
            if (!profile.row) {
                ggml_fp16_t raw;
                std::memcpy(&raw, block, sizeof(raw));
                block_scale = ggml_fp16_to_fp32(raw);
            }
            for (int group = 0; group < 32; ++group) {
                const uint32_t index =
                    index_at(block, index_offset, high_bits, group);
                float rho = 1.0f;
                float mu = 0.0f;
                if (profile.affine) {
                    const int sub = group / 4;
                    const uint8_t nibble =
                        (block[raw_bytes + sub/2] >> (4*(sub & 1))) & 15;
                    rho = (float) (6 + (nibble & 3)) / 8.0f;
                    const int mu_id = (nibble >> 2) & 3;
                    mu = mu_id == 0 ? 0.0f : mu_id == 1 ? 0.125f : -0.125f;
                }
                for (int lane = 0; lane < 8; ++lane) {
                    float value = codebook[(size_t) index*8 + lane];
                    if (profile.affine) value = rho * (value + mu);
                    value *= profile.row ? row_scale : block_scale;
                    dense[(size_t) n*K + b*256 + group*8 + lane] = value;
                }
            }
        }
    }

    const size_t tensor_bytes =
        packed.size() + dense.size()*sizeof(float) +
        (activation.size() + activation_a8.size() + 2*(size_t) N*M)*sizeof(float);
    const size_t context_bytes = tensor_bytes + 128u*1024u*1024u;
    ggml_init_params params = {context_bytes, nullptr, false};
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * tq_weight = ggml_new_tensor_2d(ctx, profile.type, K, N);
    ggml_tensor * tq_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    std::memcpy(tq_weight->data, packed.data(), packed.size());
    std::memcpy(tq_input->data, activation.data(), activation.size()*sizeof(float));
    tq_weight->extra = &binding;
    if (!ggml_cpu_validate_tq1_tensor(tq_weight)) {
        std::fprintf(stderr, "generated packed tensor failed validation\n");
        return 4;
    }
    ggml_tensor * tq_output = ggml_mul_mat(ctx, tq_weight, tq_input);
    if (profile.row) {
        ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, N);
        std::memcpy(scale->data, row_scales.data(), row_scales.size()*sizeof(ggml_fp16_t));
        tq_output = ggml_mul(ctx, tq_output, ggml_cast(ctx, scale, GGML_TYPE_F32));
    }
    ggml_cgraph * tq_graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(tq_graph, tq_output);

    ggml_tensor * dense_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * dense_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    std::memcpy(dense_weight->data, dense.data(), dense.size()*sizeof(float));
    std::memcpy(dense_input->data, activation_a8.data(), activation_a8.size()*sizeof(float));
    ggml_tensor * dense_output = ggml_mul_mat(ctx, dense_weight, dense_input);
    ggml_cgraph * dense_graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(dense_graph, dense_output);

    if (ggml_graph_compute_with_ctx(ctx, tq_graph, threads) != GGML_STATUS_SUCCESS ||
        ggml_graph_compute_with_ctx(ctx, dense_graph, threads) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "correctness graph failed\n");
        return 5;
    }
    const float * observed = (const float *) tq_output->data;
    const float * reference = (const float *) dense_output->data;
    double dense_max_abs = 0.0;
    double dense_max_rel = 0.0;
    for (int64_t i = 0; i < N*M; ++i) {
        const double absolute = std::fabs((double) observed[i] - reference[i]);
        const double relative = absolute / std::max(1e-12, std::fabs((double) reference[i]));
        dense_max_abs = std::max(dense_max_abs, absolute);
        dense_max_rel = std::max(dense_max_rel, relative);
    }

    double oracle_max_abs = 0.0;
    double oracle_max_rel = 0.0;
    bool within_tolerance = true;
    for (int64_t m = 0; m < M; ++m) {
        const int8_t * q = activation_q.data() + m*K;
        for (int64_t n = 0; n < N; ++n) {
            int64_t integer_acc = 0;
            int64_t affine_numerator = 0;
            float block_acc = 0.0f;
            for (int64_t b = 0; b < blocks_per_row; ++b) {
                const uint8_t * block =
                    packed.data() + (n*blocks_per_row + b)*block_bytes;
                int64_t current = 0;
                if (profile.affine) {
                    for (int sub = 0; sub < 8; ++sub) {
                        int64_t dot = 0;
                        int64_t xsum = 0;
                        for (int extra = 0; extra < 4; ++extra) {
                            const int group = 4*sub + extra;
                            const uint32_t index =
                                index_at(block, index_offset, high_bits, group);
                            for (int lane = 0; lane < 8; ++lane) {
                                dot += q[b*256 + group*8 + lane] *
                                    codebook[(size_t) index*8 + lane];
                                xsum += q[b*256 + group*8 + lane];
                            }
                        }
                        const uint8_t nibble =
                            (block[raw_bytes + sub/2] >> (4*(sub & 1))) & 15;
                        const int mu_id = (nibble >> 2) & 3;
                        const int mu_num = mu_id == 0 ? 0 : mu_id == 1 ? 1 : -1;
                        affine_numerator +=
                            (int64_t) (6 + (nibble & 3)) * (8*dot + mu_num*xsum);
                    }
                    continue;
                }
                for (int group = 0; group < 32; ++group) {
                    const uint32_t index =
                        index_at(block, index_offset, high_bits, group);
                    for (int lane = 0; lane < 8; ++lane) {
                        current += q[b*256 + group*8 + lane] *
                            codebook[(size_t) index*8 + lane];
                    }
                }
                if (profile.row) {
                    integer_acc += current;
                } else {
                    ggml_fp16_t raw;
                    std::memcpy(&raw, block, sizeof(raw));
                    block_acc += ggml_fp16_to_fp32(raw) * (float) current;
                }
            }
            float expected;
            if (profile.affine) {
                expected = activation_scales[m] * ((float) affine_numerator / 64.0f);
            } else if (profile.row) {
                expected = activation_scales[m] * (float) integer_acc;
            } else {
                expected = activation_scales[m] * block_acc;
            }
            if (profile.row) expected *= ggml_fp16_to_fp32(row_scales[n]);
            const double absolute =
                std::fabs((double) observed[m*N + n] - expected);
            const double relative =
                absolute / std::max(1e-12, std::fabs((double) expected));
            oracle_max_abs = std::max(oracle_max_abs, absolute);
            oracle_max_rel = std::max(oracle_max_rel, relative);
            if (absolute > 1e-6 + 1e-6*std::fabs((double) expected)) {
                within_tolerance = false;
            }
        }
    }

    const std::vector<double> tq_samples =
        measure(ctx, tq_graph, threads, warmup, iterations);
    const std::vector<double> dense_samples =
        measure(ctx, dense_graph, threads, warmup, iterations);
    const double tq_p20 = percentile(tq_samples, 0.2);
    const double tq_median = percentile(tq_samples, 0.5);
    const double tq_p80 = percentile(tq_samples, 0.8);
    const double dense_p20 = percentile(dense_samples, 0.2);
    const double dense_median = percentile(dense_samples, 0.5);
    const double dense_p80 = percentile(dense_samples, 0.8);
    const size_t row_scale_bytes = profile.row ? (size_t) N*sizeof(ggml_fp16_t) : 0;
    const double actual_bpw =
        8.0 * (double) (packed.size() + row_scale_bytes) / (double) (N*K);

    std::printf(
        "{\"profile\":\"%s\",\"N\":%lld,\"K\":%lld,\"M\":%lld,"
        "\"threads\":%d,\"warmup\":%d,\"iterations\":%d,"
        "\"tq1_ms\":{\"p20\":%.6f,\"median\":%.6f,\"p80\":%.6f},"
        "\"dense_dequant_ms\":{\"p20\":%.6f,\"median\":%.6f,\"p80\":%.6f},"
        "\"dense_over_tq1\":%.6f,\"packed_bpw_with_row_scale\":%.9f,"
        "\"oracle_max_abs_error\":%.9g,\"oracle_max_rel_error\":%.9g,"
        "\"dense_max_abs_error\":%.9g,\"dense_max_rel_error\":%.9g,"
        "\"tolerance\":\"oracle abs <= 1e-6 + 1e-6 * abs(reference)\","
        "\"correct\":%s}\n",
        profile_name.c_str(), (long long) N, (long long) K, (long long) M,
        threads, warmup, iterations,
        tq_p20, tq_median, tq_p80,
        dense_p20, dense_median, dense_p80,
        dense_median / tq_median, actual_bpw,
        oracle_max_abs, oracle_max_rel, dense_max_abs, dense_max_rel,
        within_tolerance ? "true" : "false");
    ggml_free(ctx);
    return within_tolerance ? 0 : 6;
}
