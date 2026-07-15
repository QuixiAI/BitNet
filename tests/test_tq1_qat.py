from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from bitnet_train.tq1.codebook import (
    base3_ids, direct_joint_codebook, product_codebook,
    sign_canonical_codebook)
from bitnet_train.tq1.oracle import linear_w2a8, quantize_activation
from bitnet_train.tq1.packing import unpack_payload
from bitnet_train.tq1.ptq import Importance, PTQConfig, project_weight, ternary_universe
from bitnet_train.tq1.qat import TQ1Experts, TQ1Linear, a8_ste
from bitnet_train.tq1.spec import QuantSpec


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
    )
    module = TQ1Linear.from_ptq(
        linear, ptq, book, spec, profile="tq1_v11-j-r",
        phase=phase, top_m=4, assignment_chunk=64)
    return module, book


def _direct_book():
    universe = ternary_universe()
    zero = universe[(universe == 0).all(1)]
    nonzero = universe[~(universe == 0).all(1)][:2047]
    # Format-v1 direct/I codebooks reserve index 1029 as the zero row.
    table = torch.cat((nonzero[:1029], zero, nonzero[1029:]))
    return direct_joint_codebook("qat_i", table, scope="loaded")


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


def test_soft_forward_has_candidate_gradients_and_margin_metric():
    module, _ = _module("soft")
    x = torch.randn(2, 256)
    loss = module(x).square().mean() + 0.01 * module.margin_loss(0.1)
    loss.backward()
    assert torch.isfinite(module.latent_weight.grad).all()
    health = module.health(module.indices.clone())
    assert health["index_flip_rate"] == 0
    assert health["codebook_entropy"] >= 0


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
