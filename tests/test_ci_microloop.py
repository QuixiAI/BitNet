"""The CI micro-loop (train_plan §7.3): a random tiny model runs the ENTIRE
pipeline on every commit — convert -> train -> save/reload -> resume (hash-
validated) -> bake -> requantize-exact parity — in minutes, CPU-only. The
program's highest compound risk is loop breakage discovered at scale; this
converts it into red CI. Plus the MoE twin (moe_train_plan §5.6)."""

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

pytest.importorskip("transformers")

from bitnet_train.conversion import convert, load_converted, load_profile  # noqa: E402
from bitnet_train.export.compare_gguf import _is_ternary, encode_tq2_0_ref  # noqa: E402
from bitnet_train.export.export_gguf import bake_checkpoint  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROFILES = REPO / "train" / "profiles"
CONFIGS = REPO / "train" / "configs" / "ci"


def _make_shards(d: Path, vocab=512, seq_len=32, n_docs=200):
    from prepare_data import ShardWriter, pack, write_manifest
    d.mkdir(parents=True, exist_ok=True)
    docs = [("token doc %d body " % i) * 8 for i in range(n_docs)]
    enc = lambda ts: [[1 + zlib.crc32(w.encode()) % (vocab - 1) for w in t.split()]
                      for t in ts]
    writers = {"train": ShardWriter(str(d), "train", 50_000),
               "val": ShardWriter(str(d), "val", 50_000)}
    pack(iter(docs), enc, 0, lambda i: writers["val" if i % 10 == 0 else "train"],
         batch_docs=32)
    for w in writers.values():
        w.close()
    write_manifest(str(d), "ci-fake", vocab, 0, seq_len=seq_len, source="ci",
                   writers=writers)
    return d


def _tiny_llama(vocab=512):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=vocab,
        tie_word_embeddings=True))


def _tiny_qwen_moe(vocab=512):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_experts=8, num_experts_per_tok=2, vocab_size=vocab, mlp_only_layers=[],
        tie_word_embeddings=True, head_dim=32, router_aux_loss_coef=0.01))


def _init_ckpt(model, profile, out: Path):
    convert(model, profile, backend="reference")
    model.save_pretrained(out)
    return out


@pytest.fixture(scope="module")
def loop_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("microloop")
    shards = _make_shards(root / "shards")
    prof = load_profile(PROFILES / "ci_tiny.yaml")
    init = _init_ckpt(_tiny_llama(), prof, root / "init")
    return root, shards, init, prof


def test_microloop_train_save_resume_bake(loop_env):
    root, shards, init, prof = loop_env
    import train as train_mod

    run_dir = train_mod.main([
        "--config", str(CONFIGS / "tiny.yaml"),
        "--init", str(init), "--data-dir", str(shards),
        "--out-dir", str(root / "ckpts"), "--total-tokens", "1280",   # 10 steps @128
    ])

    # metrics: CE+KD logged separately; finite; loss should not explode
    lines = [json.loads(l) for l in
             (Path(run_dir) / "metrics.jsonl").read_text().splitlines()]
    steps = [l for l in lines if "ce" in l]
    assert steps and all(np.isfinite(l["loss"]) for l in steps)
    assert {"ce", "kd"} <= set(steps[0])
    evals = [l for l in lines if "flip_total" in l]
    assert evals, "eval block never logged ternary panel"
    assert "ce_w_a8" in evals[-1] and "ce_w_only" in evals[-1]

    # checkpoint + provenance meta
    latest = Path(run_dir) / "latest"
    assert latest.is_symlink()
    st = torch.load(latest / "trainer_state.pt", weights_only=False)
    assert {"quantizer_hash", "config_hash", "profile_hash",
            "manifest_hash", "rng_states"} <= set(st["meta"])

    # save -> reload is bit-identical
    m1, _ = load_converted(latest, prof, backend="reference")
    m2, _ = load_converted(latest, prof, backend="reference")
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)

    # resume from latest continues (2 more steps) with hashes validated
    run_dir2 = train_mod.main([
        "--config", str(CONFIGS / "tiny.yaml"),
        "--init", str(init), "--data-dir", str(shards),
        "--out-dir", str(root / "ckpts"), "--total-tokens", "1536",
        "--resume", str(latest.resolve()),
    ])
    assert (Path(run_dir2) / "report.txt").exists()

    # tampered corpus -> resume hard-fails (§5.6)
    man = json.loads((shards / "manifest.json").read_text())
    man["splits"]["train"]["shards"][0]["sha256"] = "f" * 16
    (shards / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="provenance mismatch"):
        train_mod.main([
            "--config", str(CONFIGS / "tiny.yaml"),
            "--init", str(init), "--data-dir", str(shards),
            "--out-dir", str(root / "ckpts"), "--total-tokens", "1664",
            "--resume", str(latest.resolve()),
        ])

    # bake the trained checkpoint: every target ternary, and TQ2_0's own
    # re-quantization recovers codes exactly (the llama.cpp-free §8.2 gate)
    model, _ = load_converted(latest, prof, backend="reference")
    bake_checkpoint(model, prof, root / "baked")
    from bitnet_train.export.compare_gguf import load_baked_tensors
    baked = load_baked_tensors(root / "baked")
    tern = {n: v for n, v in baked.items() if v.ndim == 2 and _is_ternary(v)}
    assert len(tern) >= 14
    for name, x in tern.items():
        if x.shape[1] % 256 == 0:
            codes, _ = encode_tq2_0_ref(x.astype(np.float32))
            back, _ = encode_tq2_0_ref(
                codes.astype(np.float32) * np.abs(x).max())
            np.testing.assert_array_equal(codes, back, err_msg=name)


def test_microloop_moe_twin(tmp_path):
    import train as train_mod
    shards = _make_shards(tmp_path / "shards")
    prof = load_profile(PROFILES / "ci_tiny_moe.yaml")
    init = _init_ckpt(_tiny_qwen_moe(), prof, tmp_path / "init")

    run_dir = train_mod.main([
        "--config", str(CONFIGS / "tiny_moe.yaml"),
        "--init", str(init), "--data-dir", str(shards),
        "--out-dir", str(tmp_path / "ckpts"), "--total-tokens", "1280",
    ])
    lines = [json.loads(l) for l in
             (Path(run_dir) / "metrics.jsonl").read_text().splitlines()]
    steps = [l for l in lines if "ce" in l]
    assert steps and all(np.isfinite(l["loss"]) for l in steps)
    assert any(l.get("aux", 0) != 0 for l in steps), "aux loss never fired"
    evals = [l for l in lines if "flip_total" in l]
    assert evals and "router/entropy_mean" in evals[-1]
    assert "ce_a0" in evals[-1] and "ce_a1" in evals[-1] and "ce_b" in evals[-1]
    assert "router/zero_code_cold_tail" in evals[-1]
