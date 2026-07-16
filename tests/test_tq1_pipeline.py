from dataclasses import replace

import torch

from bitnet_train.tq1.artifact import ArtifactReader
from bitnet_train.tq1.codebook import CodebookRegistry, sign_canonical_codebook
from bitnet_train.tq1.pipeline import (
    LLAMA_KEEP_FP_REGEXES, LLAMA_TARGET_REGEXES, classify_model_linears,
    run_full_model_ptq)
from bitnet_train.tq1.runtime import load_packed_model
from bitnet_train.tq1.solver import canonical_shapes
from bitnet_train.tq1.spec import QuantSpec, TensorRule


def test_full_model_ptq_emits_canonical_schema2(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(4)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=256, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        tie_word_embeddings=True,
    )).float()
    shapes = canonical_shapes()
    zero = shapes[(shapes == 0).all(1)]
    nonzero = shapes[~(shapes == 0).all(1)][:1023]
    book = sign_canonical_codebook(
        "tiny_v11j", "v11", torch.cat((zero, nonzero)), scope="model")
    spec = QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=LLAMA_TARGET_REGEXES,
        keep_fp_regexes=LLAMA_KEEP_FP_REGEXES,
        activation_mode="a8_token", importance_mode="uniform")
    spec = replace(spec, weight_metric="uniform", candidate_count=4,
                   alternating_iterations=2)
    inventory = classify_model_linears(model, spec)
    assert len(inventory.target) == 7 and inventory.keep_fp == ("lm_head",)
    source = tmp_path / "source"
    model.config.save_pretrained(source)
    (source / "tokenizer_config.json").write_text("{}")
    output = run_full_model_ptq(
        model, spec, CodebookRegistry({book.id: book}),
        output_dir=tmp_path / "artifact", source_model="tiny",
        source_revision="0" * 40, source_files=source,
        overwrite=False, command=("test",))
    reader = ArtifactReader(output)
    reader.validate()
    assert reader.manifest["artifact_schema"] == 2
    assert len(reader.manifest["tensors"]) == 7
    assert reader.manifest["quant_spec_sha256"] == spec.sha256()
    assert (output / "config.json").is_file()
    assert reader.aliases["lm_head.weight"]["target"] == \
        "model.embed_tokens.weight"
    from safetensors.torch import load_file
    physical_non_tq1 = load_file(output / "non_tq1_model.safetensors")
    assert "model.embed_tokens.weight" in physical_non_tq1
    assert "lm_head.weight" not in physical_non_tq1
    accounting = reader.manifest["size_accounting"]
    assert accounting["unique_logical_parameters"] == sum(
        parameter.numel() for parameter in model.parameters())
    assert accounting["logical_parameter_references"] == (
        accounting["unique_logical_parameters"] + model.lm_head.weight.numel())
    assert accounting["physical_model_storage_bytes"] < \
        accounting["non_tq1_logical_reference_bytes"] + accounting["payload_bytes"] \
        + accounting["row_scale_bytes"] + accounting["codebook_bytes"]
    input_ids = torch.randint(0, 64, (1, 4))
    packed_model, _ = load_packed_model(output)
    assert packed_model.lm_head.weight is packed_model.model.embed_tokens.weight
    assert packed_model.tq1_memory_report["backend_private_repack_bytes"] == 0
    assert packed_model.tq1_memory_report["resident_language_model_bytes"] > 0
    assert packed_model.tq1_memory_report["estimated_decode_weight_bytes_per_token"] == (
        accounting["payload_bytes"] + accounting["row_scale_bytes"]
        + accounting["codebook_bytes"] + accounting["non_tq1_parameter_bytes"])
    cast_model, _ = load_packed_model(output, dtype=torch.float16)
    assert cast_model.lm_head.weight is cast_model.model.embed_tokens.weight
    assert cast_model.lm_head.weight.dtype == torch.float16
    logits = packed_model(input_ids).logits
    assert tuple(logits.shape) == (1, 4, 64) and torch.isfinite(logits).all()
    native_model, _ = load_packed_model(
        output, runtime_backend="native_cpu", native_impl="scalar")
    assert native_model.tq1_memory_report["backend_private_repack_bytes"] > 0
    native_logits = native_model(input_ids).logits
    torch.testing.assert_close(native_logits, logits, atol=1e-6, rtol=1e-6)
    assert torch.equal(native_logits.argmax(-1), logits.argmax(-1))


def test_mixed_tq1_and_fp_policy_is_canonical_and_runnable(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(44)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=256, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        tie_word_embeddings=True)).float()
    shapes = canonical_shapes()
    book = sign_canonical_codebook("mixed", "v11", torch.cat((
        shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)][:1023])))
    spec = QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=LLAMA_TARGET_REGEXES, keep_fp_regexes=LLAMA_KEEP_FP_REGEXES,
        activation_mode="a8_token", importance_mode="uniform")
    spec = replace(
        spec, weight_metric="uniform", candidate_count=4,
        alternating_iterations=2, tensor_overrides=(TensorRule(
            r"model\.layers\.0\.self_attn\.q_proj", "bf16", None),))
    source = tmp_path / "source_mixed"
    model.config.save_pretrained(source)
    (source / "tokenizer_config.json").write_text("{}")
    output = run_full_model_ptq(
        model, spec, CodebookRegistry({book.id: book}),
        output_dir=tmp_path / "mixed", source_model="tiny",
        source_revision="1" * 40, source_files=source)
    reader = ArtifactReader(output)
    reader.validate()
    state_name = "model.layers.0.self_attn.q_proj.weight"
    assert len(reader.manifest["tensors"]) == 6
    assert reader.manifest["resolved_tensor_policy"][state_name] == {
        "profile": "bf16", "codebook_id": None,
        "storage": "non_tq1_model.safetensors"}
    from safetensors.torch import load_file
    assert load_file(output / "non_tq1_model.safetensors")[state_name].dtype == torch.bfloat16
    packed, _ = load_packed_model(output)
    assert torch.isfinite(packed(torch.randint(0, 64, (1, 3))).logits).all()
