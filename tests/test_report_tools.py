"""Model-card + scaling-report generators (train_plan §13.3 / §11.6) build from
run artifacts without a real training run."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from model_card import build_card  # noqa: E402
from scaling_report import build_report, tokens_to_target  # noqa: E402


def test_tokens_to_target_interpolates():
    curve = [(0, 100.0), (1_000_000, 40.0), (2_000_000, 18.0), (3_000_000, 10.0)]
    ttt = tokens_to_target(curve, 20.0, decreasing=True)
    assert 1_000_000 < ttt < 2_000_000
    assert tokens_to_target(curve, 5.0) is None          # never reached


def test_scaling_report_two_runs(tmp_path):
    for label, scale in (("A1", 1), ("A3", 2)):
        d = tmp_path / label
        d.mkdir()
        with (d / "metrics.jsonl").open("w") as f:
            for step in range(1, 6):
                tokens = step * 100_000 * scale
                f.write(json.dumps({"step": step, "tokens": tokens,
                                    "ppl_w_a8": 100.0 / step}) + "\n")
    rep = build_report([tmp_path / "A1", tmp_path / "A3"], ["A1", "A3"],
                       target_ppl=25.0, target_kl=None)
    assert "Scaling ratio (A3/A1)" in rep and "2.00×" in rep


def test_model_card_from_artifacts(tmp_path):
    conv = {"base_model": "unsloth/Llama-3.2-1B", "n_ternarized": 112,
            "n_expert_stacks": 0, "ternary_param_fraction": 0.79,
            "ternary_flop_fraction": 0.81, "kept_fp": ["lm_head"],
            "export_route": ["i2s_bitnet_cpp", "tq2_upstream"]}
    prov = {"quantizer_hash": "abc", "config_hash": "def", "git_rev": "123",
            "kd": "dense", "args": {"latent_dtype": "fp32", "teacher": "self"}}
    (tmp_path / "conv.json").write_text(json.dumps(conv))
    (tmp_path / "prov.json").write_text(json.dumps(prov))
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        json.dumps({"step": 100, "tokens": 100_000_000, "val_ce_primary": 3.0,
                    "ppl_w_a8": 20.1, "kl_tf": 0.5}) + "\n")

    import argparse
    args = argparse.Namespace(
        name="Llama-3.2-1B-1.58", run=str(run),
        conversion=str(tmp_path / "conv.json"), provenance=str(tmp_path / "prov.json"),
        parity=None, t0_report=None, teacher=None, license=None, routes=None,
        eval_mode=None, limitations=None)
    card = build_card(args)
    assert "NOT a reproduction" in card
    assert "79.0%" in card and "81.0%" in card           # fractions formatted
    assert "unsloth/Llama-3.2-1B" in card
    assert "100,000,000" in card                         # heal tokens
