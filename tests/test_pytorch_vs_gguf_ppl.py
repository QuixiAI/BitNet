"""Tiny end-to-end route test (train_plan §7.0 file #10, §8.5 acceptance loop):
convert -> bake -> mainline TQ2_0 export -> tensor parity (exact codes) ->
runtime PPL vs the paired PyTorch eval mode (w_only <-> TQ2_0).

Skips gracefully when the mainline llama.cpp build or the HF tokenizer is
unavailable — everything upstream of the missing piece still asserts.
The tight PPL tolerance is a REAL-model T0 job; here the model is random and
the check is that the loop closes with finite, same-ballpark numbers.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

transformers = pytest.importorskip("transformers")

from bitnet_train.conversion import convert, load_profile  # noqa: E402
from bitnet_train.export.export_gguf import bake_checkpoint, export_tq2, mainline_dir, runtime_ppl  # noqa: E402
from bitnet_train.export.compare_gguf import tensor_parity  # noqa: E402

PROFILES = Path(__file__).resolve().parents[1] / "train" / "profiles"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"          # open BPE tokenizer, llama-arch friendly


def _need(path: Path, what: str):
    if not path.exists():
        pytest.skip(f"{what} not available at {path}")


@pytest.fixture(scope="module")
def baked_dir(tmp_path_factory):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:                        # offline / no cache
        pytest.skip(f"tokenizer {TOKENIZER} unavailable: {e}")
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=tok.vocab_size, tie_word_embeddings=True,
                      bos_token_id=tok.bos_token_id or 0,
                      eos_token_id=tok.eos_token_id or 0)
    model = LlamaForCausalLM(cfg)
    prof = load_profile(PROFILES / "ci_tiny.yaml")
    convert(model, prof, backend="reference")

    out = tmp_path_factory.mktemp("e2e")
    tok.save_pretrained(out / "baked")
    bake_checkpoint(model, prof, out / "baked")
    return out, model, tok, prof


def test_bake_then_tq2_export_and_parity(baked_dir):
    out, model, tok, prof = baked_dir
    _need(mainline_dir() / "convert_hf_to_gguf.py", "mainline llama.cpp")
    res = export_tq2(out / "baked", out / "model.tq2_0.gguf", python=sys.executable)
    if res.skipped:
        pytest.skip(res.reason)
    assert res.ok, f"{res.reason}\n{res.log_tail}"

    rows, ok = tensor_parity(out / "baked", res.gguf, regime="preserve")
    decoded = [r for r in rows if r.status in ("exact", "mismatch")]
    assert decoded, f"no ternary tensor decoded; rows={[(r.hf_name, r.status, r.detail) for r in rows]}"
    bad = [(r.hf_name, r.gguf_type, r.code_mismatch_rate) for r in rows
           if r.status == "mismatch"]
    assert ok, f"parity FAIL: {bad}"
    assert all(r.within_f16_bound for r in decoded)


def test_runtime_ppl_matches_python_mode(baked_dir, tmp_path):
    out, model, tok, prof = baked_dir
    gguf = out / "model.tq2_0.gguf"
    if not gguf.exists():
        pytest.skip("gguf not produced (export test skipped or failed)")
    _need(mainline_dir() / "build" / "bin" / "llama-perplexity", "llama-perplexity")

    text = ("The quick brown fox jumps over the lazy dog. " * 400)
    calib = tmp_path / "calib.txt"
    calib.write_text(text)

    # python side, w_only mode (the TQ2_0 pairing, §8.4) on the same token stream
    from bitnet_train.bitlinear import set_eval_mode
    from eval_ppl import evaluate_ppl
    ids = tok(text, return_tensors="pt").input_ids[:, :1024]
    set_eval_mode(model, "w_only")
    model.eval()
    r = evaluate_ppl(model, ids, device="cpu", mode=None)

    rt_ppl, log = runtime_ppl(gguf, calib, ctx=512, extra=["--chunks", "2"])
    if rt_ppl is None:
        pytest.skip(f"llama-perplexity unusable: {log[:300]}")
    rel = abs(rt_ppl - r["ppl"]) / r["ppl"]
    assert rel < 0.5, (f"runtime PPL {rt_ppl:.1f} vs python w_only {r['ppl']:.1f} "
                       f"(rel {rel:.2f}) — beyond even the random-model ballpark")
