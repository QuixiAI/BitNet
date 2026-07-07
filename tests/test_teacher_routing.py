"""Teacher routing fixture + top-8 agreement (moe_train_plan §6.2, Q-T0 §8.1)."""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("transformers")

from bitnet_train.conversion import convert, load_profile  # noqa: E402
from bitnet_train.moe_metrics import (  # noqa: E402
    RouterHooks, capture_teacher_routing, top8_agreement)

PROFILES = Path(__file__).resolve().parents[1] / "train" / "profiles"


def tiny_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_experts=8, num_experts_per_tok=2, vocab_size=256, mlp_only_layers=[],
        tie_word_embeddings=True, head_dim=32, router_aux_loss_coef=0.01))


def test_top8_agreement_bounds():
    a = torch.tensor([[0, 1], [2, 3]])
    assert top8_agreement(a, a) == pytest.approx(1.0)
    assert top8_agreement(a, torch.tensor([[4, 5], [6, 7]])) == pytest.approx(0.0)
    half = top8_agreement(torch.tensor([[0, 1]]), torch.tensor([[0, 9]]))
    assert half == pytest.approx(0.5)


def test_capture_and_agreement_self_is_one():
    teacher = tiny_moe().eval()
    windows = torch.randint(0, 256, (3, 16))
    tr = capture_teacher_routing(teacher, windows, "cpu", top_k=2)
    assert len(tr) == 2                                 # one per layer
    assert all(v.shape[1] == 2 for v in tr.values())

    # a converted COPY with identical weights routes identically -> agreement 1.0
    prof = load_profile(PROFILES / "ci_tiny_moe.yaml")
    student = copy.deepcopy(teacher)
    convert(student, prof, backend="reference")
    student.eval()
    hooks = RouterHooks(student).attach()
    hooks.capture_routing = True
    with torch.no_grad():
        student(windows)
    hooks.capture_routing = False
    agree = hooks.agreement_vs_teacher(tr)
    hooks.detach()
    # student experts are ternarized (routing may drift a little) but the router
    # is untouched and layer 0 sees identical inputs -> layer-0 agreement is 1.0
    assert agree["_by_depth"][0] == pytest.approx(1.0)
    assert 0.0 <= agree["_mean"] <= 1.0
    assert len(agree["_by_depth"]) == 2


def test_agreement_drops_with_perturbed_router():
    teacher = tiny_moe().eval()
    windows = torch.randint(0, 256, (3, 16))
    tr = capture_teacher_routing(teacher, windows, "cpu", top_k=2)
    prof = load_profile(PROFILES / "ci_tiny_moe.yaml")
    student = copy.deepcopy(teacher)
    convert(student, prof, backend="reference")
    with torch.no_grad():
        for layer in student.model.layers:
            layer.mlp.gate.weight.add_(torch.randn_like(layer.mlp.gate.weight) * 5)
    hooks = RouterHooks(student).attach()
    hooks.capture_routing = True
    with torch.no_grad():
        student(windows)
    hooks.capture_routing = False
    agree = hooks.agreement_vs_teacher(tr)
    hooks.detach()
    assert agree["_mean"] < 0.9                          # scrambled router disagrees
