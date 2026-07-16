from __future__ import annotations

import copy
import importlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bitnet_train.tq1.codebook import (
    base3_ids, load_iq1_reference, product_codebook,
    sign_canonical_codebook)
from bitnet_train.tq1.oracle import linear_w2a8, quantize_activation
from bitnet_train.tq1.packing import unpack_payload
from bitnet_train.tq1.ptq import Importance, PTQConfig, project_weight, ternary_universe
from bitnet_train.tq1.qat import (
    TQ1Experts, TQ1Linear, a8_ste, aggregate_qat_health)
from bitnet_train.tq1.spec import QuantSpec


def _training_entrypoint():
    """Resolve train.py in both script-style and namespace-package test orders."""
    module = importlib.import_module("train")
    if not hasattr(module, "tq1_hidden_loss"):
        module = importlib.import_module("train.train")
    return module


def _book():
    universe = ternary_universe()
    nz = universe != 0
    first = nz.long().argmax(1)
    negative = nz.any(1) & (universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(negative, -1, 1).to(torch.int8)[:, None]
    shapes = universe[torch.unique(base3_ids(canonical), sorted=True)]
    shapes = torch.cat((shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)]))
    return sign_canonical_codebook("qat", "v11", shapes[:1024])


def _module(phase="hard"):
    torch.manual_seed(19)
    book = _book()
    linear = nn.Linear(256, 4, bias=False)
    ptq = project_weight(
        linear.weight, book, Importance(),
        PTQConfig("tq1_v11-j-r", weight_metric="uniform",
                  assignment_mode="shortlist", candidate_count=16,
                  alternating_iterations=2),
    )
    spec = QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=("x",), keep_fp_regexes=("y",),
        importance_mode="uniform",
    )
    spec = replace(spec, weight_metric="uniform")
    module = TQ1Linear.from_ptq(
        linear, ptq, book, spec, profile="tq1_v11-j-r",
        phase=phase, top_m=4, assignment_chunk=64)
    return module, book


def _direct_book():
    try:
        return load_iq1_reference("qat_i")
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"pinned read-only IQ1 reference is unavailable: {exc}")


def _product_book(fmt):
    values = torch.arange(3 ** 4, dtype=torch.int64)
    lanes = []
    for _ in range(4):
        lanes.append((values % 3 - 1).to(torch.int8))
        values //= 3
    universe = torch.stack(lanes, 1)
    nonzero = universe != 0
    first = nonzero.long().argmax(1)
    negative = nonzero.any(1) & (
        universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(
        negative, -1, 1).to(torch.int8)[:, None]
    representatives = universe[torch.unique(base3_ids(canonical), sorted=True)]
    representatives = torch.cat((
        representatives[(representatives == 0).all(1)],
        representatives[~(representatives == 0).all(1)]))
    product_a = representatives[:32]
    if fmt == "v11":
        product_b = representatives[:32]
    else:
        selected = torch.cat((representatives, -representatives[1:24]))
        zero = selected[(selected == 0).all(1)]
        rest = selected[~(selected == 0).all(1)]
        product_b = torch.cat((zero, rest[torch.argsort(base3_ids(rest))]))
    return product_codebook(f"qat_{fmt}_p", fmt, product_a, product_b)


def test_a8_ste_forward_is_bit_exact_and_backward_is_identity():
    x = torch.randn(3, 256, requires_grad=True)
    got = a8_ste(x, "a8_token")
    expected = quantize_activation(x).dequantize()
    assert torch.equal(got, expected)
    got.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("phase", ["soft", "hard"])
def test_qat_forward_is_exact_hard_projection(phase):
    module, _ = _module(phase)
    x = torch.randn(2, 256)
    got = module(x)
    weight = module.projected_weight().detach()
    aq = quantize_activation(x).dequantize()
    expected = aq @ weight.T
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)
    hard_words = module.decoded_table[module.indices].float()
    hard = module.runtime_scales().detach()[:, None, None] * hard_words
    torch.testing.assert_close(weight, hard.reshape_as(weight), atol=0, rtol=0)


def test_hard_ste_gives_latent_identity_and_scale_gradients():
    module, _ = _module("hard")
    x = torch.randn(3, 256)
    module(x).square().mean().backward()
    assert module.latent_weight.grad is not None
    assert torch.isfinite(module.latent_weight.grad).all()
    assert module.scale_parameter.grad is not None
    assert torch.isfinite(module.scale_parameter.grad).all()


def test_qat_runtime_scale_clamps_nonzero_underflow_to_smallest_normal():
    module, _ = _module("hard")
    with torch.no_grad():
        module.scale_parameter.fill_(-1000)
    scales = module.runtime_scales()
    assert torch.equal(
        scales, torch.full_like(scales, torch.finfo(module.scale_dtype).tiny))
    assert module.health()["scale_underflow_count"] == module.out_features


def test_soft_forward_has_candidate_gradients_and_margin_metric():
    module, _ = _module("soft")
    x = torch.randn(2, 256)
    loss = module(x).square().mean() + 0.01 * module.margin_loss(0.1)
    loss.backward()
    assert torch.isfinite(module.latent_weight.grad).all()
    health = module.health(module.indices.clone())
    assert health["index_flip_rate"] == 0
    assert health["codebook_entropy"] >= 0


def test_qat_health_metrics_and_aggregate_reconcile_with_direct_formulas():
    module, _ = _module("hard")
    module(torch.randn(2, 256))
    previous = torch.full_like(module.indices, module.zero_index)
    health = module.health(previous)
    groups = module.indices.numel()
    assert health["group_count"] == groups
    assert sum(health["changed_trits_histogram"].values()) == groups
    assert health["scalar_pattern_exact_hit_rate"] == pytest.approx(
        health["changed_trits_histogram"]["0"] / groups)
    assert health["index_flip_count"] == int((module.indices != previous).sum())
    current_used = torch.bincount(
        module.indices.flatten(), minlength=module.decoded_table.shape[0]) > 0
    previous_used = torch.bincount(
        previous.flatten(), minlength=module.decoded_table.shape[0]) > 0
    assert health["newly_activated_codewords"] == int(
        (current_used & ~previous_used & module.legal_mask).sum())
    assert len(health["lane_trit_fractions"]) == 8
    for lane in health["lane_trit_fractions"]:
        assert sum(lane.values()) == pytest.approx(1.0)

    hard_words = module.decoded_table[module.indices].float()
    hard = module.runtime_scales().detach()[:, None, None] * hard_words
    latent = module.latent_weight.detach().reshape_as(hard)
    expected_error = float((hard - latent).square().sum())
    expected_energy = float(latent.square().sum())
    assert health["weighted_projection_error"] == pytest.approx(expected_error)
    assert health["weighted_projection_relative_error"] == pytest.approx(
        (expected_error / expected_energy) ** 0.5)

    aggregate = aggregate_qat_health({"first": health, "second": health})
    assert aggregate["tensor_count"] == 2
    assert aggregate["group_count"] == groups * 2
    assert aggregate["changed_trits_histogram"] == {
        key: value * 2 for key, value in health["changed_trits_histogram"].items()}
    assert aggregate["index_flip_count"] == health["index_flip_count"] * 2
    assert aggregate["weighted_projection_relative_error"] == pytest.approx(
        health["weighted_projection_relative_error"])

    tq1_health_record = _training_entrypoint().tq1_health_record
    record = tq1_health_record(
        {"model.layers.0.self_attn.q_proj": health},
        event="evaluation", step=3, tokens=1024,
        quant_spec_sha256="a" * 64,
        model_metrics={"ce_w_only": 1.5, "ignored": "text"},
        training_metrics={"hidden": 0.2},
        controller={"phase": "hard"})
    assert record["aggregate"]["group_count"] == groups
    assert record["model_metrics"] == {"ce_w_only": 1.5}
    assert record["training_metrics"] == {"hidden": 0.2}
    assert json.loads(json.dumps(record)) == record
    with pytest.raises(ValueError, match="model metric.*nonfinite"):
        tq1_health_record(
            {"model.layers.0.self_attn.q_proj": health},
            event="evaluation", step=3, tokens=1024,
            quant_spec_sha256="a" * 64,
            model_metrics={"ce_w_only": float("nan")})


def test_fixed_calibration_hidden_alignment_uses_masks_and_exact_layers():
    class StudentDecoder(nn.Module):
        def forward(self, input_ids, *, output_hidden_states, use_cache):
            assert output_hidden_states is True
            assert use_cache is False
            base = torch.nn.functional.one_hot(input_ids, num_classes=8).float()
            return SimpleNamespace(hidden_states=(base, base.roll(1, -1)))

    class Teacher:
        def slice_server(self, input_ids, *, hidden_layers):
            base = torch.nn.functional.one_hot(input_ids, num_classes=8).float()
            selected = {0: base[:, :-1], 1: base.roll(1, -1)[:, :-1]}
            return lambda _slice: None, {
                layer: selected[layer] for layer in hidden_layers
            }

    tq1_hidden_alignment = _training_entrypoint().tq1_hidden_alignment

    raw_model = SimpleNamespace(model=StudentDecoder())
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    # The first prediction in each window is retained; padding is excluded.
    masks = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    result = tq1_hidden_alignment(
        raw_model, Teacher(), (ids, masks), torch.device("cpu"), (0, 1))
    assert result == {
        "hidden_normalized_mse": 0.0,
        "hidden_normalized_mse_L0": 0.0,
        "hidden_normalized_mse_L1": 0.0,
    }


def test_fixed_calibration_hidden_alignment_fails_closed_on_missing_layer():
    class StudentDecoder(nn.Module):
        def forward(self, input_ids, **_kwargs):
            hidden = torch.ones((*input_ids.shape, 4))
            return SimpleNamespace(hidden_states=(hidden,))

    class Teacher:
        def slice_server(self, input_ids, *, hidden_layers):
            del hidden_layers
            hidden = torch.ones((*input_ids[:, :-1].shape, 4))
            return lambda _slice: None, {0: hidden}

    tq1_hidden_alignment = _training_entrypoint().tq1_hidden_alignment

    raw_model = SimpleNamespace(model=StudentDecoder())
    with pytest.raises(ValueError, match="hidden layer 1 is unavailable"):
        tq1_hidden_alignment(
            raw_model, Teacher(), torch.tensor([[0, 1, 2]]),
            torch.device("cpu"), (1,))


def test_training_hidden_loss_requires_exact_layers_shapes_and_tokens():
    training = _training_entrypoint()
    tq1_hidden_layers = training.tq1_hidden_layers
    tq1_hidden_loss = training.tq1_hidden_loss

    student = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))
    teacher = {0: student[0][:, :-1].clone(), 1: student[1][:, :-1].clone()}
    mask = torch.tensor([[True, False, True], [False, True, False]])
    loss = tq1_hidden_loss(student, teacher, (0, 1), mask)
    assert loss == 0
    with pytest.raises(ValueError, match="inventory mismatch"):
        tq1_hidden_loss(student, {0: teacher[0]}, (0, 1), mask)
    with pytest.raises(ValueError, match="shape mismatch"):
        tq1_hidden_loss(
            student, {0: teacher[0], 1: teacher[1][:, :-1]}, (0, 1), mask)
    with pytest.raises(ValueError, match="retained no prediction tokens"):
        tq1_hidden_loss(student, teacher, (0, 1), torch.zeros_like(mask))
    assert tq1_hidden_layers({"hidden_layers": [0, 1], "lambda_hidden": 0.1}) == (0, 1)
    with pytest.raises(ValueError, match="unique nonnegative"):
        tq1_hidden_layers({"hidden_layers": [1, 1], "lambda_hidden": 0.1})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        tq1_hidden_layers({"hidden_layers": [1], "lambda_hidden": float("nan")})


def test_exhaustive_qat_uses_every_legal_index_and_dynamic_iq1_metric():
    torch.manual_seed(191)
    book = _book()
    latent = torch.randn(1, 256) * 0.04
    scales = torch.tensor([0.04], dtype=torch.float16)
    spec = replace(QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=("x",), keep_fp_regexes=("y",),
        importance_mode="diagonal"),
        assignment_mode="exhaustive", candidate_count=4,
        weight_metric="iq1")
    diagonal = torch.linspace(0.5, 1.5, 256)
    module = TQ1Linear(
        latent, scales, book, spec, profile="tq1_v11-j-r",
        importance_diag=diagonal, phase="hard", top_m=4,
        assignment_chunk=8)
    alpha = module.runtime_scales().detach()
    state = module.projection(alpha)
    assert module.candidate_indices.numel() == 0

    legal = torch.nonzero(module.legal_mask).flatten()
    words = module.decoded_table[legal].float()
    groups = latent.reshape(-1, 8)
    blocks = latent.reshape(-1, 256)
    sigma2 = 2.0 * blocks.square().mean(1, keepdim=True)
    iq1 = torch.sqrt(sigma2 + blocks.square()).reshape(-1, 8)
    candidates = words[None] * alpha[0]
    errors = ((groups[:, None] - candidates).square()
              * diagonal.reshape(-1, 8)[:, None] * iq1[:, None]).sum(-1)
    expected = legal[errors.argmin(1)]
    assert torch.equal(state.indices.flatten(), expected)


def test_nonuniform_qat_requires_its_declared_statistics():
    book = _book()
    spec = QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=("x",), keep_fp_regexes=("y",),
        importance_mode="diagonal")
    with pytest.raises(ValueError, match="requires diagonal statistics"):
        TQ1Linear(
            torch.randn(1, 256), torch.ones(1), book, spec,
            profile="tq1_v11-j-r", phase="hard")
    covariance_spec = replace(spec, importance_mode="covariance8")
    with pytest.raises(ValueError, match="requires covariance statistics"):
        TQ1Linear(
            torch.randn(1, 256), torch.ones(1), book, covariance_spec,
            profile="tq1_v11-j-r", importance_diag=torch.ones(256),
            phase="hard")
    indefinite = torch.eye(8).repeat(32, 1, 1)
    indefinite[0, 0, 1] = indefinite[0, 1, 0] = 2
    with pytest.raises(ValueError, match="not positive semidefinite"):
        TQ1Linear(
            torch.randn(1, 256), torch.ones(1), book, covariance_spec,
            profile="tq1_v11-j-r", importance_cov8=indefinite,
            phase="hard")


def test_frozen_indices_are_immutable_and_export_exact():
    module, _ = _module("hard")
    module(torch.randn(1, 256))
    module.set_phase("frozen")
    payload, scales = module.export_projection()
    assert payload.shape == (4, 1, 44) and scales.shape == (4,)
    with torch.no_grad():
        module.indices[0, 0] = 1
    with pytest.raises(RuntimeError, match="frozen"):
        module(torch.randn(1, 256))


def test_qat_state_dict_resume_reproduces_next_forward():
    module, _ = _module("soft")
    module.set_temperature(0.25)
    state = copy.deepcopy(module.state_dict())
    restored, _ = _module("soft")
    restored.load_state_dict(state)
    x = torch.randn(2, 256)
    torch.testing.assert_close(module(x), restored(x), atol=0, rtol=0)
    assert restored.temperature == pytest.approx(0.25)


def test_qat_state_dict_rejects_cast_indices_and_changed_immutable_buffers():
    module, _ = _module("hard")
    state = copy.deepcopy(module.state_dict())
    cast_indices = copy.deepcopy(state)
    cast_indices["indices"] = cast_indices["indices"].float()
    restored, _ = _module("hard")
    with pytest.raises(RuntimeError, match="serialized QAT indices must be int64"):
        restored.load_state_dict(cast_indices)

    changed_table = copy.deepcopy(state)
    changed_table["decoded_table"][0, 0] = 1
    restored, _ = _module("hard")
    with pytest.raises(RuntimeError, match="immutable QAT buffer"):
        restored.load_state_dict(changed_table)


def test_expert_wrapper_keeps_independent_expert_state():
    first, _ = _module("hard")
    second, _ = _module("hard")
    experts = TQ1Experts([first, second])
    x = torch.randn(3, 256)
    ids = torch.tensor([0, 1, 0])
    got = experts(x, ids)
    torch.testing.assert_close(got[ids == 0], first(x[ids == 0]))
    torch.testing.assert_close(got[ids == 1], second(x[ids == 1]))


@pytest.mark.parametrize(("profile", "kind"), [
    ("tq1_v11-i-r", "i"),
    ("tq1_v11-p-r", "p"),
    ("tq1_v12-p-r", "p"),
])
def test_full_design_i_and_p_qat_freeze_and_export_are_exact(profile, kind):
    torch.manual_seed(29)
    fmt = "v11" if "v11" in profile else "v12"
    book = _direct_book() if kind == "i" else _product_book(fmt)
    linear = nn.Linear(256, 2, bias=False)
    ptq = project_weight(
        linear.weight, book, Importance(),
        PTQConfig(profile, weight_metric="uniform", candidate_count=8,
                  alternating_iterations=2),
    )
    spec = QuantSpec.core(
        default_profile=profile, codebook=book.ref(),
        target_regexes=("x",), keep_fp_regexes=("y",),
        importance_mode="uniform",
    )
    spec = replace(spec, candidate_count=8, weight_metric="uniform")
    module = TQ1Linear.from_ptq(
        linear, ptq, book, spec, profile=profile, phase="soft",
        top_m=4, assignment_chunk=64)
    x = torch.randn(2, 256)
    module(x).square().mean().backward()
    assert module.latent_weight.grad is not None
    assert module.scale_parameter.grad is not None

    module.set_phase("hard")
    module(x)
    module.set_phase("frozen")
    payload, scales = module.export_projection()
    indices, embedded_scales, affine = unpack_payload(payload, profile)
    assert embedded_scales is None and affine is None
    assert torch.equal(indices, module.indices.cpu())
    expected = linear_w2a8(
        x, payload, profile, book, row_scales=scales,
        activation_mode=spec.activation_mode)
    torch.testing.assert_close(module(x), expected, atol=1e-6, rtol=1e-6)
