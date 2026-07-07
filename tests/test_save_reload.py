"""Converted-checkpoint save -> load_converted round-trip (plan risk R5: the
BitLinear/BitExperts state dicts are key/shape-identical to the dense modules,
so from_pretrained-then-convert must reproduce the model bit-for-bit)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("transformers")

from bitnet_train.conversion import convert, load_converted, load_profile  # noqa: E402

PROFILES = Path(__file__).resolve().parents[1] / "train" / "profiles"


@pytest.mark.parametrize("arch", ["llama", "qwen_moe"])
def test_save_reload_bit_identical(arch, tmp_path):
    torch.manual_seed(0)
    if arch == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        model = LlamaForCausalLM(LlamaConfig(
            hidden_size=128, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
            tie_word_embeddings=True))
        prof = load_profile(PROFILES / "ci_tiny.yaml")
    else:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
            hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_experts=8, num_experts_per_tok=2, vocab_size=512,
            mlp_only_layers=[], tie_word_embeddings=True, head_dim=32))
        prof = load_profile(PROFILES / "ci_tiny_moe.yaml")

    convert(model, prof, backend="reference")
    sd0 = {k: v.clone() for k, v in model.state_dict().items()}
    model.save_pretrained(tmp_path / "ckpt")

    reloaded, _ = load_converted(tmp_path / "ckpt", prof, backend="reference")
    sd1 = reloaded.state_dict()
    assert set(sd0) == set(sd1)
    for k in sd0:
        assert torch.equal(sd0[k].float(), sd1[k].float()), k

    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        y0 = model(ids).logits
        y1 = reloaded(ids).logits
    torch.testing.assert_close(y0, y1, rtol=0, atol=0)
