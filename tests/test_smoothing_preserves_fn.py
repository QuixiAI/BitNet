"""A7 smoothing must be exactly function-preserving in FP (train_plan §3.3) and
the λ-ramp must blend dense -> ternary without entering the backward graph."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("transformers")

from bitnet_train import quant  # noqa: E402
from bitnet_train.bitlinear import BitLinear, set_lambda  # noqa: E402
from bitnet_train.smoothing import apply_smoothing, collect_act_stats, smooth_model  # noqa: E402

torch.manual_seed(0)


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
        tie_word_embeddings=True))


def test_smoothing_preserves_function():
    model = tiny_llama().eval()
    ids = torch.randint(0, 512, (4, 32))
    with torch.no_grad():
        before = model(ids).logits.clone()
    report = smooth_model(model, ids, "cpu")
    assert len(report) == 4                              # 2 layers x 2 pairs
    with torch.no_grad():
        after = model(ids).logits
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-4)
    # the fold actually did something (norm weights rescaled)
    assert all(r["s_max"] > r["s_min"] for r in report.values())


def test_smoothing_shifts_activation_scale_into_weights():
    model = tiny_llama().eval()
    ids = torch.randint(0, 512, (4, 32))
    stats_before = collect_act_stats(model, ids, "cpu")
    apply_smoothing(model, stats_before, alpha=0.5)
    stats_after = collect_act_stats(model, ids, "cpu")
    # per-channel absmax at the norm outputs flattens: max/mean ratio must not grow
    for k in stats_before:
        r_before = float(stats_before[k].max() / stats_before[k].mean().clamp_min(1e-9))
        r_after = float(stats_after[k].max() / stats_after[k].mean().clamp_min(1e-9))
        assert r_after <= r_before * 1.05, (k, r_before, r_after)


def test_lambda_ramp_blends_and_keeps_ste():
    lin = BitLinear(64, 32, backend="reference", granularity="tensor")
    x = torch.randn(4, 64)
    set_lambda(lin, 0.0)
    y0 = lin(x)
    # the ramp is on the WEIGHT quantizer only (A2 + lam warm-up, §9.1): at lam=0
    # the weight is dense but A8 activation fake-quant stays on
    torch.testing.assert_close(y0, F.linear(quant.activation_quant(x), lin.weight),
                               rtol=1e-5, atol=1e-6)

    set_lambda(lin, 1.0)
    y1 = lin(x)
    w_q = quant.weight_quant(lin.weight, "tensor")
    torch.testing.assert_close(y1, F.linear(quant.activation_quant(x), w_q),
                               rtol=1e-5, atol=1e-6)

    set_lambda(lin, 0.5)
    x = x.requires_grad_(True)
    y = lin(x)
    assert not torch.allclose(y, y0) and not torch.allclose(y, y1)
    g = torch.randn_like(y)
    y.backward(g)
    w_eff = 0.5 * lin.weight.detach() + 0.5 * w_q
    torch.testing.assert_close(x.grad, g @ w_eff, rtol=1e-4, atol=1e-5)  # STE: dense grad
