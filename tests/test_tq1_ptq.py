from __future__ import annotations

import pytest
import torch

from bitnet_train.tq1.codebook import base3_ids, product_codebook, sign_canonical_codebook
from bitnet_train.tq1.oracle import dequantize_weight
from bitnet_train.tq1.ptq import (
    Importance, PTQConfig, _assign_groups, project_weight, ternary_universe)


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
    assert result.report["zero_rows"] == 1
    assert result.report["zero_scale_units"] == 1
    if result.row_scales is not None:
        assert result.row_scales[0] == 0
    assert result.payload.shape[-1] in {44, 46, 48, 50}
    report = result.report
    assert report["logical_shape"] == [3, 256]
    assert set(report["source_scale_range"]) == {"min", "median", "max"}
    assert set(report["rounded_scale_range"]) == {"min", "median", "max"}
    groups = weight.numel() // 8
    assert sum(report["changed_trits_histogram"].values()) == groups
    histogram_mean = sum(
        int(changed) * count
        for changed, count in report["changed_trits_histogram"].items()) / groups
    assert report["mean_changed_trits_per_group"] == pytest.approx(histogram_mean)
    assert report["peak_memory_bytes"] >= report["peak_memory_baseline_bytes"] > 0
    assert report["fallback_count"] == report["factorization_fallbacks"] == 0
    assert report["candidate_oracle"]["comparison_performed"] is True
    assert report["candidate_oracle"]["sample_unit"] == (
        "affine_subblock32" if "-a4-" in profile else "codeword_group8")
    usages = report["top_codeword_usages"]
    assert usages == sorted(usages, key=lambda item: (-item["count"], item["index"]))
    assert all(item["count"] > 0 and 0 < item["fraction"] <= 1 for item in usages)


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
    assert first.report["candidate_oracle"] == {
        "comparison_performed": False,
        "reason": "assignment path is already the exhaustive oracle",
        "sample_unit": "codeword_group8",
        "population_count": weight.numel() // 8,
        "sample_count": 0,
        "mismatch_count": 0,
        "mismatch_rate": 0.0,
        "mean_excess_objective": 0.0,
        "max_excess_objective": 0.0,
    }


def test_covariance8_importance_rejects_nonsymmetric_and_indefinite_metrics():
    book = _joint("v11")
    weight = torch.randn(1, 256) * 0.02
    config = PTQConfig(
        "tq1_v11-j-r", assignment_mode="exhaustive",
        alternating_iterations=2, weight_metric="uniform")
    nonsymmetric = torch.eye(8).repeat(32, 1, 1)
    nonsymmetric[0, 0, 1] = 1
    with pytest.raises(ValueError, match="not symmetric"):
        project_weight(
            weight, book, Importance("covariance8", cov8=nonsymmetric), config)
    indefinite = torch.eye(8).repeat(32, 1, 1)
    indefinite[0, 0, 1] = indefinite[0, 1, 0] = 2
    with pytest.raises(ValueError, match="not positive semidefinite"):
        project_weight(
            weight, book, Importance("covariance8", cov8=indefinite), config)


def test_nonzero_scale_underflow_is_clamped_and_reported():
    book = _joint("v11")
    weight = torch.full((1, 256), 1e-8)
    result = project_weight(
        weight, book, Importance(),
        PTQConfig("tq1_v11-j-r", assignment_mode="shortlist", candidate_count=4,
                  alternating_iterations=2, weight_metric="uniform"),
    )
    tiny = torch.finfo(torch.float16).tiny
    assert result.report["source_scale_range"]["max"] < tiny
    assert result.report["rounded_scale_range"] == {
        "min": tiny, "median": tiny, "max": tiny}
    assert result.report["underflow_count"] == 1
    assert result.report["rounding_underflow_events"] >= 1
    assert result.report["zero_rows"] == 0
    assert result.report["zero_scale_units"] == 0


def test_assignment_objective_ties_choose_the_lowest_legal_index():
    book = _joint("v11")
    decoded = book.decode(torch.arange(book.index_count))
    legal = torch.nonzero(book.legal_index_mask()).flatten()
    by_pattern = {
        int(base3_ids(decoded[index:index + 1])[0]): int(index) for index in legal}
    pair = None
    for first in legal.tolist():
        word = decoded[first]
        for lane in range(8):
            for delta in (-1, 1):
                if not -1 <= int(word[lane]) + delta <= 1:
                    continue
                neighbor = word.clone()
                neighbor[lane] += delta
                second = by_pattern.get(int(base3_ids(neighbor[None])[0]))
                if second is not None:
                    pair = (first, second, word, neighbor)
                    break
            if pair is not None:
                break
        if pair is not None:
            break
    assert pair is not None
    first, second, first_word, second_word = pair
    target = (first_word.float() + second_word.float())[None] * 0.5
    diag = torch.ones(1, 8)
    expected = min(first, second)
    for mode in ("exhaustive", "shortlist"):
        index, _, error = _assign_groups(
            target, 1.0, book, diag, None,
            PTQConfig("tq1_v11-j-r", assignment_mode=mode,
                      candidate_count=16, alternating_iterations=2,
                      weight_metric="uniform"))
        assert int(index[0]) == expected
        assert float(error[0]) == pytest.approx(0.25)


def test_invalid_gptq_or_width_fail_closed():
    book = _joint("v11")
    with pytest.raises(ValueError, match="divisible by 256"):
        project_weight(torch.randn(2, 264), book, Importance(),
                       PTQConfig("tq1_v11-j-r"))
    with pytest.raises(ValueError, match="A4"):
        PTQConfig("tq1_v11-j-a4-r", gptq_feedback=True).validate()
    with pytest.raises(ValueError, match="source dtype"):
        project_weight(torch.randn(2, 256, dtype=torch.float64), book, Importance(),
                       PTQConfig("tq1_v11-j-r"))


def test_gptq_feedback_never_worsens_declared_full_block_objective():
    torch.manual_seed(13)
    book = _joint("v11")
    weight = torch.randn(1, 256) * 0.025
    a = torch.randn(256, 40)
    covariance = (a @ a.T / 40 + torch.eye(256) * 0.05)[None]
    base_config = PTQConfig(
        "tq1_v11-j-r", weight_metric="uniform",
        assignment_mode="shortlist", candidate_count=8,
        alternating_iterations=2)
    ordinary = project_weight(
        weight, book, Importance("block256", cov256=covariance), base_config)
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
    damping = 0.01 * float(covariance[0].diagonal().mean())
    damped = covariance[0] + torch.eye(256) * damping
    ordinary_delta = ordinary.dequantized[0] - weight[0]
    selected_delta = result.dequantized[0] - weight[0]
    assert report["ordinary_objective"] == pytest.approx(float(
        ordinary_delta @ damped @ ordinary_delta))
    assert report["selected_objective"] == pytest.approx(float(
        selected_delta @ damped @ selected_delta))
    assert report["group_order"] == "increasing_k"
    assert report["block_size"] == 256
    assert report["block_damping_values"] == pytest.approx([damping])
    assert report["factorization_failures"] == 0
    assert report["factorization_fallbacks"] == 0


def test_gptq_factorization_fails_closed_or_records_block_fallback():
    torch.manual_seed(31)
    book = _joint("v11")
    weight = torch.randn(1, 256) * 0.02
    covariance = torch.eye(256)[None]
    covariance[0, 0, 0] = 0
    importance = Importance("block256", cov256=covariance)
    base = dict(
        profile="tq1_v11-j-r", weight_metric="uniform",
        assignment_mode="shortlist", candidate_count=8,
        alternating_iterations=2, gptq_feedback=True, gptq_damping=0.0)
    with pytest.raises(ValueError, match="GPTQ Cholesky failed"):
        project_weight(weight, book, importance, PTQConfig(**base))
    result = project_weight(
        weight, book, importance,
        PTQConfig(**base, allow_diagonal_fallback=True))
    report = result.report["gptq"][0]
    assert report["factorization_failures"] == 1
    assert report["factorization_fallbacks"] == 1
    assert report["diagonal_fallback_blocks"] == [0]
    assert report["factorization_failure_locations"] == [
        {"block": 0, "stage": "block", "info": 1}]
    assert result.report["fallback_count"] == 1
