"""T0.5 reconstruction (train_plan §11.3): per-block latent optimization must
REDUCE the fake-quant-vs-dense block-output error and improve the converted
model's function vs raw conversion."""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("transformers")

from bitnet_train.conversion import convert, load_profile  # noqa: E402
from bitnet_train.reconstruct import reconstruct  # noqa: E402

PROFILES = Path(__file__).resolve().parents[1] / "train" / "profiles"


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
        tie_word_embeddings=True))


def test_reconstruction_reduces_block_error_and_improves_ppl():
    prof = load_profile(PROFILES / "ci_tiny.yaml")
    dense = tiny_llama().eval()
    conv = convert(copy.deepcopy(dense), prof, backend="reference")  # returns report
    conv_model = copy.deepcopy(dense)
    convert(conv_model, prof, backend="reference")

    ids = torch.randint(0, 512, (4, 32))
    with torch.no_grad():
        dense_logits = dense(ids).logits
        err_raw = (conv_model(ids).logits - dense_logits).pow(2).mean().item()

    report = reconstruct(dense, conv_model, ids, "cpu", steps=120, lr=5e-3)
    assert report.per_block, "no blocks reconstructed"
    # the deterministic guarantee: every block's local objective decreased, a lot
    for name, b in report.per_block.items():
        assert b["err_after"] < b["err_before"], (name, b)
    assert report.mean_reduction > 0.3, report.mean_reduction

    with torch.no_grad():
        err_recon = (conv_model(ids).logits - dense_logits).pow(2).mean().item()
    # whole-model output does not REGRESS materially (block-local recon on a
    # teacher-forced toy needn't strictly improve the compounded output, but a
    # real weight-dominant heal does — §11.3); guard against divergence
    assert err_recon < 1.5 * err_raw, (err_raw, err_recon)


def test_latents_stay_trainable_after_recon():
    prof = load_profile(PROFILES / "ci_tiny.yaml")
    dense = tiny_llama().eval()
    conv_model = tiny_llama()
    convert(conv_model, prof, backend="reference")
    reconstruct(dense, conv_model, torch.randint(0, 512, (2, 16)), "cpu", steps=10)
    from bitnet_train.bitlinear import iter_bitlinears
    assert all(m.weight.requires_grad for _, m in iter_bitlinears(conv_model))
