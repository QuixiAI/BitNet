"""Config preservation is a hard requirement (train_plan §2.2: rope_scaling
above all; silently dropping it changes long-context behavior)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train.conversion import diff_config  # noqa: E402

pytest.importorskip("transformers")


# transformers 5 serializes rope_theta + rope_scaling as one `rope_parameters`
# dict — the diff key the assertions below watch. (v4 configs diff under their
# old names; diff_config is name-agnostic either way.)

def _cfg(**over):
    from transformers import LlamaConfig
    base = dict(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
                max_position_embeddings=131072, rope_theta=500000.0,
                rope_scaling={"rope_type": "llama3", "factor": 32.0,
                              "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                              "original_max_position_embeddings": 8192})
    base.update(over)
    return LlamaConfig(**base)


def _rope_key(d: dict) -> bool:
    return any(k.startswith("rope") for k in d)


def test_identical_configs_diff_clean():
    assert diff_config(_cfg(), _cfg()) == {}


def test_rope_scaling_drop_detected():
    assert _rope_key(diff_config(_cfg(), _cfg(rope_scaling=None)))


def test_rope_scaling_field_change_detected():
    changed = _cfg(rope_scaling={"rope_type": "llama3", "factor": 8.0,
                                 "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                                 "original_max_position_embeddings": 8192})
    assert _rope_key(diff_config(_cfg(), changed))


def test_rope_theta_change_detected():
    assert _rope_key(diff_config(_cfg(), _cfg(rope_theta=10000.0)))


def test_dtype_and_path_fields_ignored():
    a, b = _cfg(), _cfg()
    b._name_or_path = "/somewhere/else"
    assert diff_config(a, b) == {}
