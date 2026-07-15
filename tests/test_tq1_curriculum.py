from dataclasses import replace

import torch

from bitnet_train.tq1.codebook import sign_canonical_codebook
from bitnet_train.tq1.curriculum import QATController, QATSchedule
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
        soft_steps=3, hard_steps=2, temperature_start=1.0,
        temperature_end=0.25, sustain_evals=2, flip_threshold=0.01,
        margin_threshold=0.1, trend_tolerance=0.02)
    controller = QATController([module], schedule)
    controller.before_step(0)
    assert module.phase == "soft" and module.temperature == 1.0
    controller.before_step(2)
    assert module.temperature == 0.25
    controller.before_step(3)
    assert module.phase == "hard"
    first = controller.observe(4, {
        "flip_total": 0.001, "tq1_margin_p05": 0.2, "val_ce_primary": 2.0})
    assert not first["transitioned"]
    saved = controller.state_dict()

    resumed_module = _module()
    resumed = QATController([resumed_module], schedule)
    resumed.load_state_dict(saved)
    result = resumed.observe(5, {
        "flip_total": 0.001, "tq1_margin_p05": 0.2, "val_ce_primary": 2.01})
    assert result["transitioned"] and resumed_module.phase == "frozen"
    resumed.assert_export_qualified()


def test_extra_state_is_tensor_and_zero_rows_remain_zero():
    module = _module()
    state = module.state_dict()
    assert state["_extra_state"].dtype == torch.uint8
    with torch.no_grad():
        module.zero_rows[0] = True
        module.indices[0].fill_(1)
        projected = module.projection(module.runtime_scales())
    assert torch.all(projected.indices[0] == module.zero_index)
