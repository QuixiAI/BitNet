"""Profile-driven conversion on tiny Llama + tiny Qwen3-MoE (train_plan §7.0 #9;
moe_train_plan §1.2's router trap + exact-count assertions)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train.bitlinear import BitExperts, BitLinear  # noqa: E402
from bitnet_train.conversion import (  # noqa: E402
    classify_linears, convert, diff_config, load_profile)

transformers = pytest.importorskip("transformers")

PROFILES = Path(__file__).resolve().parents[1] / "train" / "profiles"


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
                      tie_word_embeddings=True)
    return LlamaForCausalLM(cfg)


def tiny_qwen_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(hidden_size=128, intermediate_size=256,
                         moe_intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, num_experts=8,
                         num_experts_per_tok=2, vocab_size=512, mlp_only_layers=[],
                         tie_word_embeddings=True, head_dim=32)
    return Qwen3MoeForCausalLM(cfg)


def test_llama_counts_and_names():
    model = tiny_llama()
    prof = load_profile(PROFILES / "ci_tiny.yaml")
    cfg_before = model.config.to_dict()
    rep = convert(model, prof, backend="reference")
    # 2 layers x (4 attn + 3 mlp) = 14 BitLinears; lm_head kept
    assert rep.n_ternarized == 14 and rep.n_kept_fp == 1
    assert all(isinstance(model.get_submodule(n), BitLinear) for n in rep.ternarized)
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert not isinstance(model.lm_head, BitLinear)
    assert not diff_config(cfg_before, model.config)
    assert rep.family_counts["self_attn.q_proj"] == 2
    out = model(torch.randint(0, 512, (2, 16))).logits
    assert out.shape == (2, 16, 512)


def test_qwen_moe_router_survives_and_expert_stacks_swap():
    model = tiny_qwen_moe()
    prof = load_profile(PROFILES / "ci_tiny_moe.yaml")
    rep = convert(model, prof, backend="reference")
    assert rep.n_expert_stacks == 2                    # one fused stack per layer
    assert rep.n_ternarized == 0                       # v5: no per-expert Linears
    for layer in model.model.layers:
        assert isinstance(layer.mlp.experts, BitExperts)
        # the router: untouched, trainable, NOT a BitLinear/BitExperts
        assert type(layer.mlp.gate).__name__ == "Qwen3MoeTopKRouter"
        assert layer.mlp.gate.weight.requires_grad
    out = model(torch.randint(0, 512, (2, 16))).logits
    assert out.shape == (2, 16, 512)


def test_unclassified_linear_is_hard_error():
    model = tiny_llama()
    prof = load_profile(PROFILES / "ci_tiny.yaml")
    prof.target_linear_regexes = [r"model\.layers\.\d+\.self_attn\.(q|k|v)_proj"]
    with pytest.raises(ValueError, match="enumerate-don't-assume"):
        classify_linears(model, prof)


def test_doubly_matched_linear_is_hard_error():
    model = tiny_llama()
    prof = load_profile(PROFILES / "ci_tiny.yaml")
    prof.keep_fp_regexes = prof.keep_fp_regexes + [r".*o_proj"]
    with pytest.raises(ValueError, match="enumerate-don't-assume"):
        classify_linears(model, prof)


def test_unanchored_gate_regex_overmatches_loudly():
    """The Qwen trap: on the v4-style per-expert layout an unanchored 'gate'
    regex ternarized the router silently. fullmatch + exactly-one-class makes any
    over-broad pattern collide with keep_fp and fail loudly instead."""
    model = tiny_llama()
    prof = load_profile(PROFILES / "ci_tiny.yaml")
    prof.target_linear_regexes = prof.target_linear_regexes + [r".*"]
    with pytest.raises(ValueError, match="enumerate-don't-assume"):
        classify_linears(model, prof)
