"""§5.4 precision hatches: MasterAdamW (fp32 masters over bf16 params, optional
8-bit moments) vs stock fp32 AdamW; truncate_model slicing."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from bitnet_train.optim import MasterAdamW  # noqa: E402

torch.manual_seed(0)


def _run(opt_factory, dtype, steps=30, moments_bits=32):
    torch.manual_seed(1)
    p = torch.randn(64, 32)
    ref = p.clone()
    param = torch.nn.Parameter(p.to(dtype))
    opt = opt_factory(param)
    grads = [torch.randn(64, 32) * 0.1 for _ in range(steps)]
    for g in grads:
        param.grad = g.to(dtype)
        opt.step()
        opt.zero_grad()
    return param.detach().float(), grads, ref


def test_master_adamw_matches_stock_fp32():
    """bf16 params + fp32 masters must track stock fp32 AdamW to bf16 rounding."""
    stock, grads, ref = _run(
        lambda p: torch.optim.AdamW([p], lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                    weight_decay=0.1), torch.float32)
    master, _, _ = _run(
        lambda p: MasterAdamW([p], lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                              weight_decay=0.1), torch.bfloat16)
    # grads differed only by bf16 rounding of the gradient; drift stays small
    torch.testing.assert_close(master, stock, rtol=2e-2, atol=2e-2)
    assert not torch.equal(master, ref)


def test_master_adamw_fp32_params_exact():
    """On fp32 params with fp32 moments the math is EXACTLY stock AdamW."""
    stock, _, _ = _run(lambda p: torch.optim.AdamW([p], lr=1e-2, betas=(0.9, 0.95),
                                                   eps=1e-8, weight_decay=0.1),
                       torch.float32)
    ours, _, _ = _run(lambda p: MasterAdamW([p], lr=1e-2, betas=(0.9, 0.95),
                                            eps=1e-8, weight_decay=0.1),
                      torch.float32)
    torch.testing.assert_close(ours, stock, rtol=1e-6, atol=1e-7)


def test_q8_moments_track_fp32_moments():
    """8-bit moments introduce bounded noise, not divergence: updates stay closely
    aligned with the fp32-moment run over 30 steps."""
    full, _, ref = _run(lambda p: MasterAdamW([p], lr=1e-2, betas=(0.9, 0.95),
                                              eps=1e-8), torch.float32)
    q8, _, _ = _run(lambda p: MasterAdamW([p], lr=1e-2, betas=(0.9, 0.95),
                                          eps=1e-8, moments_bits=8), torch.float32)
    d_full = full - ref
    d_q8 = q8 - ref
    cos = torch.nn.functional.cosine_similarity(d_full.reshape(1, -1),
                                                d_q8.reshape(1, -1)).item()
    assert cos > 0.98, cos
    assert (d_q8.norm() / d_full.norm()).item() == pytest.approx(1.0, abs=0.2)


def test_truncate_model_slices_moe(tmp_path):
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
    from truncate_model import truncate

    torch.manual_seed(0)
    full = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
        num_experts=8, num_experts_per_tok=2, vocab_size=256, mlp_only_layers=[],
        tie_word_embeddings=True, head_dim=32))
    full.save_pretrained(tmp_path / "full")

    rep = truncate(tmp_path / "full", tmp_path / "small", layers=1, experts=4)
    assert rep["dropped_tensors"] > 0
    small = AutoModelForCausalLM.from_pretrained(tmp_path / "small")
    assert small.config.num_hidden_layers == 1 and small.config.num_experts == 4
    experts = small.model.layers[0].mlp.experts
    assert experts.gate_up_proj.shape[0] == 4          # fused 3-D after load
    assert small.model.layers[0].mlp.gate.weight.shape[0] == 4
    out = small(torch.randint(0, 256, (2, 8))).logits
    assert out.shape == (2, 8, 256) and torch.isfinite(out).all()
    # sliced weights are the ORIGINAL layer-0 experts 0..3, not reinit
    torch.testing.assert_close(experts.down_proj,
                               full.model.layers[0].mlp.experts.down_proj[:4])
