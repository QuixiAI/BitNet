from __future__ import annotations

import pytest
import torch

from bitnet_train.tq1.codebook import base3_ids, product_codebook, sign_canonical_codebook
from bitnet_train.tq1.oracle import dequantize_weight
from bitnet_train.tq1.ptq import Importance, PTQConfig, project_weight, ternary_universe


def _joint(fmt="v11"):
    universe = ternary_universe()
    nz = universe != 0
    first = nz.long().argmax(1)
    negative = nz.any(1) & (universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(negative, -1, 1).to(torch.int8)[:, None]
    ids = torch.unique(base3_ids(canonical), sorted=True)
    shapes = universe[ids]
    zero = shapes[(shapes == 0).all(1)]
    shapes = torch.cat((zero, shapes[~(shapes == 0).all(1)]))
    return sign_canonical_codebook("joint", fmt, shapes[:1024 if fmt == "v11" else 2048])


def _product(fmt="v11"):
    value = torch.arange(81)
    lanes = []
    for _ in range(4):
        lanes.append((value % 3 - 1).to(torch.int8)); value //= 3
    universe = torch.stack(lanes, 1)
    nz = universe != 0
    first = nz.long().argmax(1)
    negative = nz.any(1) & (universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(negative, -1, 1).to(torch.int8)[:, None]
    reps = universe[torch.unique(base3_ids(canonical), sorted=True)]
    reps = torch.cat((reps[(reps == 0).all(1)], reps[~(reps == 0).all(1)]))
    a = reps[:32]
    if fmt == "v11":
        b = reps[:32]
    else:
        raw = torch.cat((reps, -reps[1:24]))
        b = torch.cat((raw[(raw == 0).all(1)],
                       raw[~(raw == 0).all(1)][torch.argsort(
                           base3_ids(raw[~(raw == 0).all(1)]))]))
    return product_codebook("product", fmt, a, b)


@pytest.mark.parametrize(("profile", "kind"), [
    ("tq1_v11-j-r", "j"),
    ("tq1_v12-j-r", "j"),
    ("tq1_v11-p-r", "p"),
    ("tq1_v12-p-r", "p"),
    ("tq1_v11-j-b", "j"),
    ("tq1_v12-j-b", "j"),
    ("tq1_v11-j-a4-r", "j"),
])
def test_all_ptq_profiles_emit_exactly_decodable_payloads(profile, kind):
    torch.manual_seed(7)
    fmt = "v11" if "v11" in profile else "v12"
    book = _joint(fmt) if kind == "j" else _product(fmt)
    weight = torch.randn(3, 256) * 0.03
    weight[0].zero_()
    result = project_weight(
        weight, book,
        Importance("diagonal", diag=torch.linspace(0.5, 1.5, 256)),
        PTQConfig(profile, assignment_mode="shortlist", candidate_count=16,
                  alternating_iterations=2, weight_metric="uniform"),
    )
    decoded = dequantize_weight(result.payload, profile, book,
                                row_scales=result.row_scales)
    assert torch.equal(decoded, result.dequantized)
    assert torch.isfinite(decoded).all()
    assert result.report["zero_rows"] == (1 if profile.endswith("-r") else 0)
    if result.row_scales is not None:
        assert result.row_scales[0] == 0
    assert result.payload.shape[-1] in {44, 46, 48, 50}


def test_covariance_objective_and_exhaustive_assignment_are_deterministic():
    torch.manual_seed(9)
    book = _joint("v11")
    weight = torch.randn(2, 256) * 0.02
    a = torch.randn(32, 8, 8)
    cov = a @ a.transpose(-1, -2) + torch.eye(8)[None] * 0.01
    cfg = PTQConfig("tq1_v11-j-r", assignment_mode="exhaustive",
                    alternating_iterations=2, weight_metric="uniform")
    first = project_weight(weight, book, Importance("covariance8", cov8=cov), cfg)
    second = project_weight(weight, book, Importance("covariance8", cov8=cov), cfg)
    assert torch.equal(first.payload, second.payload)
    assert torch.equal(first.row_scales, second.row_scales)
    assert first.report["iteration_objectives"] == second.report["iteration_objectives"]


def test_invalid_gptq_or_width_fail_closed():
    book = _joint("v11")
    with pytest.raises(ValueError, match="divisible by 256"):
        project_weight(torch.randn(2, 264), book, Importance(),
                       PTQConfig("tq1_v11-j-r"))
    with pytest.raises(ValueError, match="A4"):
        PTQConfig("tq1_v11-j-a4-r", gptq_feedback=True).validate()


def test_gptq_feedback_never_worsens_declared_full_block_objective():
    torch.manual_seed(13)
    book = _joint("v11")
    weight = torch.randn(1, 256) * 0.025
    a = torch.randn(256, 40)
    covariance = (a @ a.T / 40 + torch.eye(256) * 0.05)[None]
    result = project_weight(
        weight, book, Importance("block256", cov256=covariance),
        PTQConfig("tq1_v11-j-r", weight_metric="uniform",
                  assignment_mode="shortlist", candidate_count=8,
                  alternating_iterations=2, gptq_feedback=True,
                  gptq_damping=0.01),
    )
    report = result.report["gptq"][0]
    assert report["selected_objective"] <= report["ordinary_objective"]
    assert report["selected_candidate"] in {0, 1, 2}
