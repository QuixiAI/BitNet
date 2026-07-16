from __future__ import annotations

from dataclasses import replace

import torch

from bitnet_train.tq1.artifact import ArtifactReader
from bitnet_train.tq1.codebook import CodebookRegistry, sign_canonical_codebook
from bitnet_train.tq1.oracle import dequantize_weight
from bitnet_train.tq1.pipeline import (
    LLAMA_KEEP_FP_REGEXES, LLAMA_SHARED_EMBEDDING_REGEX, LLAMA_TARGET_REGEXES,
    classify_model_linears, run_full_model_ptq)
from bitnet_train.tq1.qat import TQ1Embedding, TQ1OutputHead
from bitnet_train.tq1.runtime import PackedTQ1Embedding, load_packed_model
from bitnet_train.tq1.solver import canonical_shapes
from bitnet_train.tq1.spec import QuantSpec


def _book():
    shapes = canonical_shapes()
    return sign_canonical_codebook("shared", "v11", torch.cat((
        shapes[(shapes == 0).all(1)],
        shapes[~(shapes == 0).all(1)][:1023])), scope="model")


def test_shared_embedding_head_is_one_quantized_payload_and_two_consumers(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(901)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=256, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
        tie_word_embeddings=True)).float()
    book = _book()
    spec = replace(QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=LLAMA_TARGET_REGEXES + (LLAMA_SHARED_EMBEDDING_REGEX,),
        keep_fp_regexes=LLAMA_KEEP_FP_REGEXES,
        activation_mode="a8_token", importance_mode="uniform"),
        weight_metric="uniform", candidate_count=4, alternating_iterations=2,
        shared_embedding_head=True, shared_head_importance=0.75,
        shared_embedding_importance=0.25)
    inventory = classify_model_linears(model, spec)
    assert inventory.shared_tied == (("model.embed_tokens", "lm_head"),)
    source = tmp_path / "source"
    model.config.save_pretrained(source)
    (source / "tokenizer_config.json").write_text("{}")
    statistics = {
        "model.embed_tokens.token_frequency": torch.arange(1, 33, dtype=torch.int64),
    }
    output = run_full_model_ptq(
        model, spec, CodebookRegistry({book.id: book}),
        output_dir=tmp_path / "artifact", source_model="tiny-shared",
        source_revision="9" * 40, source_files=source, statistics=statistics)
    reader = ArtifactReader(output)
    reader.validate()
    assert len(reader.manifest["tensors"]) == 8
    shared = next(item for item in reader.manifest["tensors"]
                  if item["consumer_kind"] == "shared_embedding_head")
    assert shared["state_dict_name"] == "model.embed_tokens.weight"
    assert reader.aliases["lm_head.weight"]["target"] == shared["state_dict_name"]
    from safetensors.torch import load_file
    non_tq1 = load_file(output / "non_tq1_model.safetensors")
    assert "model.embed_tokens.weight" not in non_tq1
    assert "lm_head.weight" not in non_tq1
    _, payload, scales = reader.tensor("lm_head.weight")
    _, target_payload, target_scales = reader.tensor("model.embed_tokens.weight")
    assert torch.equal(payload, target_payload) and torch.equal(scales, target_scales)
    accounting = reader.manifest["size_accounting"]
    assert accounting["low_bit_unique_parameters"] == sum(
        item["logical_shape"][0] * item["logical_shape"][1]
        for item in reader.manifest["tensors"])

    packed, _ = load_packed_model(output)
    assert isinstance(packed.model.embed_tokens, PackedTQ1Embedding)
    assert packed.lm_head.shared_weight is packed.model.embed_tokens
    assert packed.lm_head.weight is packed.model.embed_tokens
    ids = torch.tensor([[1, 1, 7, 31]])
    got_embedding = packed.model.embed_tokens(ids)
    dense = dequantize_weight(
        target_payload, shared["profile"], book, row_scales=target_scales)
    torch.testing.assert_close(got_embedding, dense[ids], atol=0, rtol=0)
    logits = packed(ids).logits
    assert tuple(logits.shape) == (1, 4, 32) and torch.isfinite(logits).all()


def test_shared_qat_accumulates_lookup_and_head_gradients_into_one_latent():
    torch.manual_seed(902)
    book = _book()
    spec = replace(QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=("embed",), keep_fp_regexes=("lm_head",),
        activation_mode="none", importance_mode="uniform"),
        candidate_count=4, shared_embedding_head=True)
    weight = torch.randn(8, 256) * 0.1
    scales = torch.full((8,), 0.1, dtype=torch.float16)
    indices = torch.zeros((8, 32), dtype=torch.int64)
    embed = TQ1Embedding(
        weight, scales, book, spec, profile="tq1_v11-j-r",
        initial_indices=indices, phase="hard", top_m=4)
    head = TQ1OutputHead(embed)
    ids = torch.tensor([[1, 3, 3]])
    hidden = torch.randn(1, 2, 256)

    embed(ids).sum().backward()
    lookup_grad = embed.weight.grad.detach().clone()
    embed.weight.grad = None
    head(hidden).sum().backward()
    head_grad = embed.weight.grad.detach().clone()
    embed.weight.grad = None
    (embed(ids).sum() + head(hidden).sum()).backward()
    torch.testing.assert_close(embed.weight.grad, lookup_grad + head_grad)
    assert head.weight is embed.weight
    assert list(head.parameters()) == []
