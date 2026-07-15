// End-to-end loader/graph smoke test for the deterministic tiny TQ1 GGUF.

#include "llama.h"

#include <cmath>
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s tiny-tq1.gguf\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = false;
    model_params.check_tensors = true;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load TQ1 model\n");
        llama_backend_free();
        return 3;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 16;
    context_params.n_batch = 4;
    context_params.n_ubatch = 4;
    context_params.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::fprintf(stderr, "failed to create TQ1 context\n");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    llama_token tokens[] = {1, 4, 5, 2};
    llama_batch batch = llama_batch_get_one(tokens, 4);
    if (llama_decode(context, batch) != 0) {
        std::fprintf(stderr, "TQ1 prefill failed\n");
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t count = llama_vocab_n_tokens(vocab);
    float * logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr || count < 1) {
        std::fprintf(stderr, "TQ1 logits are unavailable\n");
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }
    for (int32_t i = 0; i < count; ++i) {
        if (!std::isfinite(logits[i])) {
            std::fprintf(stderr, "nonfinite TQ1 logit at %d\n", i);
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 7;
        }
    }
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    std::puts("TQ1_V llama.cpp model load + prefill: PASS");
    return 0;
}
