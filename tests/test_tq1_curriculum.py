from dataclasses import replace
import math

import pytest
import torch

from bitnet_train.conversion import load_profile
from bitnet_train.tq1.codebook import sign_canonical_codebook
from bitnet_train.tq1.curriculum import QATController, QATSchedule, schedule_from_config
from bitnet_train.tq1.qat import TQ1Linear
from bitnet_train.tq1.solver import canonical_shapes
from bitnet_train.tq1.spec import QuantSpec


def _module():
    shapes = canonical_shapes()
    book = sign_canonical_codebook("curriculum", "v11", torch.cat((
        shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)][:1023])))
    spec = replace(QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=("x",), keep_fp_regexes=("y",),
        importance_mode="uniform"), candidate_count=4)
    return TQ1Linear(torch.randn(2, 256), torch.ones(2), book, spec,
                     profile="tq1_v11-j-r", top_m=4, phase="soft")


def test_schedule_is_exact_and_resume_freezes_only_after_gates():
    module = _module()
    schedule = QATSchedule(
        soft_tokens=30, hard_tokens=20, freeze_eval_every_tokens=10,
        temperature_start=1.0,
        temperature_end=0.25, sustain_evals=2, flip_threshold=0.01,
        margin_threshold=0.1, trend_tolerance=0.02)
    schedule.validate_run(total_tokens=70, tokens_per_step=10)
    controller = QATController([module], schedule)
    controller.before_step(0)
    assert module.phase == "soft" and module.temperature == 1.0
    controller.after_step(10)
    controller.before_step(10)
    assert 0.25 < module.temperature < 1.0
    controller.after_step(20)
    controller.before_step(20)
    controller.after_step(30)
    controller.before_step(30)
    assert module.phase == "hard"
    controller.after_step(40)
    first = controller.observe(40, {
        "flip_total": 0.001, "tq1_margin_p05": 0.2, "val_ce_primary": 2.0})
    assert not first["transitioned"]
    saved = controller.state_dict()
    module_checkpoint = module.state_dict()

    resumed_module = _module()
    resumed_module.load_state_dict(module_checkpoint)
    resumed = QATController([resumed_module], schedule)
    resumed.load_state_dict(saved)
    resumed.validate_position(40)
    resumed.before_step(40)
    resumed.after_step(50)
    result = resumed.observe(50, {
        "flip_total": 0.001, "tq1_margin_p05": 0.2, "val_ce_primary": 2.01})
    assert result["transitioned"] and resumed_module.phase == "frozen"
    resumed.assert_export_qualified()
    controller.before_step(40)
    controller.after_step(50)
    uninterrupted = controller.observe(50, {
        "flip_total": 0.001, "tq1_margin_p05": 0.2, "val_ce_primary": 2.01})
    assert uninterrupted["transitioned"]
    assert resumed.state_dict() == controller.state_dict()
    assert torch.equal(resumed_module.indices, module.indices)


def test_schedule_is_world_size_invariant_and_rejects_misalignment():
    schedule = QATSchedule(
        soft_tokens=40, hard_tokens=120, freeze_eval_every_tokens=40,
        freeze_indices_at_tokens=160, freeze_max_tokens=200,
        sustain_evals=3)
    transitions = []
    for tokens_per_step in (10, 20, 40):
        schedule.validate_run(total_tokens=240, tokens_per_step=tokens_per_step)
        controller = QATController([_module()], schedule)
        for tokens in range(0, 240, tokens_per_step):
            controller.before_step(tokens)
            controller.after_step(tokens + tokens_per_step)
            if controller.observation_due(tokens + tokens_per_step):
                event = controller.observe(tokens + tokens_per_step, {
                    "flip_total": 0.0, "tq1_margin_p05": 1.0,
                    "val_ce_primary": 1.0, "kl_tf": 0.0})
                if event["transitioned"]:
                    transitions.append(tokens + tokens_per_step)
                    break
    assert transitions == [160, 160, 160]
    with pytest.raises(ValueError, match="aligned"):
        schedule.validate_run(total_tokens=240, tokens_per_step=30)


def test_schedule_rejects_impossible_run_and_legacy_checkpoint():
    schedule = QATSchedule(
        soft_tokens=20, hard_tokens=60, freeze_eval_every_tokens=20,
        freeze_max_tokens=100, sustain_evals=3)
    with pytest.raises(ValueError, match="cannot reach"):
        schedule.validate_run(total_tokens=80, tokens_per_step=20)
    controller = QATController([_module()], schedule)
    with pytest.raises(ValueError, match="legacy step-domain"):
        controller.load_state_dict({"schema": 1})

    for tokens in range(0, 100, 20):
        controller.before_step(tokens)
        controller.after_step(tokens + 20)
        controller.observe(tokens + 20, {
            "flip_total": 1.0, "tq1_margin_p05": 0.0,
            "val_ce_primary": 1.0, "kl_tf": 0.0})
    assert controller.failure_reason == "freeze gates unmet: flip"
    with pytest.raises(RuntimeError, match="freeze gates unmet"):
        controller.before_step(100)


def test_canonical_200m_profile_is_feasible_for_one_two_and_four_processes():
    profile = load_profile("train/profiles/tq1_llama32_1b_instruct.yaml")
    schedule = schedule_from_config(profile.quant)
    base_tokens_per_step = 1 * 32 * 4096
    for process_count in (1, 2, 4):
        tokens_per_step = base_tokens_per_step * process_count
        total_tokens = math.ceil(200_000_000 / tokens_per_step) * tokens_per_step
        schedule.validate_run(total_tokens=total_tokens,
                              tokens_per_step=tokens_per_step)


def test_extra_state_is_tensor_and_zero_rows_remain_zero():
    module = _module()
    state = module.state_dict()
    assert state["_extra_state"].dtype == torch.uint8
    with torch.no_grad():
        module.zero_rows[0] = True
        module.indices[0].fill_(1)
        projected = module.projection(module.runtime_scales())
    assert torch.all(projected.indices[0] == module.zero_index)
