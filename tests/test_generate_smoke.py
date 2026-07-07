"""Generation smoke driver produces coherent-shaped output in each eval mode and
via the frozen packed path (train_plan §10.5)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

pytest.importorskip("transformers")

from bitnet_train.bitlinear import BitLinear, set_eval_mode  # noqa: E402
from bitnet_train.conversion import convert, load_profile  # noqa: E402

PROFILES = Path(__file__).resolve().parents[1] / "train" / "profiles"


def test_generate_smoke_all_modes(tmp_path):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from generate_smoke import generate

    try:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        tok = None
    if tok is None:
        pytest.skip("no tokenizer available offline")

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=tok.vocab_size,
        tie_word_embeddings=True, eos_token_id=tok.eos_token_id or 0))
    prof = load_profile(PROFILES / "ci_tiny.yaml")
    convert(model, prof, backend="reference")

    for mode in ("w_a8", "w_only"):
        set_eval_mode(model, mode)
        out = generate(model, tok, ["hello world"], "cpu", max_new_tokens=8)
        assert len(out) == 1 and isinstance(out[0]["completion"], str)

    # frozen packed path is MPS-only (packs via the Metal kernel); exercise it
    # only there
    if torch.backends.mps.is_available():
        model.to("mps")
        set_eval_mode(model, "w_a8")
        for m in model.modules():
            if isinstance(m, BitLinear):
                m.backend = "metal"
                m.freeze()
        out = generate(model, tok, ["hello world"], "mps", max_new_tokens=8)
        assert len(out) == 1
