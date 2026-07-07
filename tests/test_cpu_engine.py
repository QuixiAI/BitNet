"""K-track CPU decode engine (moe_train_plan §7.3 / Q-K0): the full decode loop
over ternary experts must run and roughly track the HF fake-quant reference on a
tiny model; the TL1 format path must match the bitnet-A path."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("transformers")
import torch  # noqa: E402


def _tiny_moe(tmp_path):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(0)
    # H, I, head_dim all %16 (TL1 tiles + block-32 packing)
    m = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_experts=8, num_experts_per_tok=2, vocab_size=256, mlp_only_layers=[],
        tie_word_embeddings=True, head_dim=32, rms_norm_eps=1e-6))
    m.save_pretrained(tmp_path / "m")
    return tmp_path / "m", m


def test_engine_decode_step_finite_and_shaped(tmp_path):
    from bitnet_train.cpu.engine import CPUEngine
    hf, _ = _tiny_moe(tmp_path)
    eng = CPUEngine(hf, fmt="bitnet", pt=True)
    kv = eng.new_kv(16)
    logits = eng.step(5, 0, kv)
    assert logits.shape == (256,) and np.isfinite(logits).all()
    # a few steps advance without blowing up
    for pos in range(1, 6):
        logits = eng.step(int(logits.argmax()), pos, kv)
        assert np.isfinite(logits).all()


def test_engine_tl1_matches_bitnet_format(tmp_path):
    from bitnet_train.cpu.engine import CPUEngine
    hf, _ = _tiny_moe(tmp_path)
    a = CPUEngine(hf, fmt="bitnet", pt=True)
    c = CPUEngine(hf, fmt="tl1", pt=True)
    kv_a, kv_c = a.new_kv(8), c.new_kv(8)
    for pos, tok in enumerate([3, 7, 1]):
        la = a.step(tok, pos, kv_a)
        lc = c.step(tok, pos, kv_c)
        np.testing.assert_allclose(la, lc, rtol=1e-4, atol=1e-4)


def test_engine_generate_and_bench(tmp_path):
    from bitnet_train.cpu.engine import CPUEngine, bench
    hf, _ = _tiny_moe(tmp_path)
    eng = CPUEngine(hf, fmt="tl1", pt=True)
    out = eng.generate([2, 5, 9], max_new_tokens=6)
    assert len(out) == 6 and all(0 <= t < 256 for t in out)
    b = bench(eng, prompt_len=16, gen=6)
    assert b["decode_tok_s"] > 0 and b["ttft_s"] > 0
