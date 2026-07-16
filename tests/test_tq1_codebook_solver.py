from __future__ import annotations

import pytest
import torch
from torch import nn

from bitnet_train.tq1.codebook import base3_ids
from bitnet_train.tq1.pipeline import (
    LinearInventory, learn_model_codebook, scalar_pattern_tensors)
from bitnet_train.tq1.solver import (
    PatternCorpus,
    _distance_matrix,
    build_product_codebook,
    canonical_shapes,
    combine_pattern_corpora,
    corpus_from_tensors,
    facility_location_select,
    required_anchor_rows,
    sensitivity_corpus_from_tensors,
)


def test_family_equal_corpus_gives_each_projection_family_equal_mass():
    q = torch.zeros((100, 8), dtype=torch.int8)
    down = torch.ones((10, 8), dtype=torch.int8)
    corpus = corpus_from_tensors({
        "model.layers.0.self_attn.q_proj": q,
        "model.layers.0.mlp.down_proj": down,
    })
    assert corpus.demand.sum() == 2
    assert corpus.demand[int(base3_ids(q[:1]))] == 1
    assert corpus.demand[int(base3_ids(down[:1]))] == 1


def test_pattern_corpus_rejects_fractional_values_before_integer_conversion():
    values = torch.zeros((1, 8))
    values[0, 0] = 0.5
    with pytest.raises(ValueError, match="non-trit"):
        corpus_from_tensors({"q_proj": values})


def test_sensitivity_corpus_aggregates_family_weighted_metric_without_row_expansion():
    q = torch.zeros((2, 8), dtype=torch.int8)
    down = torch.zeros((1, 8), dtype=torch.int8)
    corpus = sensitivity_corpus_from_tensors(
        {"layer.q_proj": q, "layer.down_proj": down},
        diagonal={"layer.q_proj": torch.ones(8),
                  "layer.down_proj": torch.full((8,), 3.0)},
        weighting="family_equal", row_chunk=1)
    zero_id = int(base3_ids(q[:1]))
    assert corpus.demand.sum() == 2
    assert corpus.demand[zero_id] == 2
    assert torch.equal(corpus.diagonal[zero_id], torch.full((8,), 2.0,
                                                           dtype=torch.float64))


def test_combining_corpora_weights_conditional_sensitivity_by_raw_demand():
    demand_a = torch.zeros(6561); demand_a[0] = 2
    demand_b = torch.zeros(6561); demand_b[0] = 1
    diag_a = torch.zeros((6561, 8)); diag_a[0] = 1
    diag_b = torch.zeros((6561, 8)); diag_b[0] = 3
    raw = combine_pattern_corpora([
        PatternCorpus(demand_a, diagonal=diag_a),
        PatternCorpus(demand_b, diagonal=diag_b),
    ])
    assert raw.demand[0] == 3
    assert torch.equal(raw.diagonal[0], torch.full((8,), 5 / 3,
                                                   dtype=torch.float64))
    equal = combine_pattern_corpora([
        PatternCorpus(demand_a, diagonal=diag_a),
        PatternCorpus(demand_b, diagonal=diag_b),
    ], normalize_each=True)
    assert equal.demand[0] == 2
    assert torch.equal(equal.diagonal[0], torch.full((8,), 2.0,
                                                     dtype=torch.float64))


def test_pattern_corpus_rejects_indefinite_covariance():
    covariance = torch.zeros((6561, 8, 8))
    covariance[0] = torch.eye(8)
    covariance[0, 0, 1] = covariance[0, 1, 0] = 2
    with pytest.raises(ValueError, match="positive semidefinite"):
        PatternCorpus(torch.ones(6561), covariance=covariance)


def test_model_codebook_learning_consumes_declared_sensitivity():
    model = nn.Module()
    model.q_proj = nn.Linear(256, 1, bias=False)
    book = learn_model_codebook(
        model, LinearInventory(("q_proj",), ()), codebook_id="sensitive",
        index_format="v11", statistics={"q_proj.diag": torch.ones(256)},
        importance_mode="diagonal", swap_limit=0)
    assert book.provenance["importance_mode"] == "diagonal"
    assert len(book.provenance["anchor_base3_ids"]) == book.provenance["anchor_count"]


def test_scalar_pattern_initializer_uses_declared_diagonal_importance():
    model = nn.Module()
    model.q_proj = nn.Linear(256, 1, bias=False)
    with torch.no_grad():
        model.q_proj.weight.zero_()
        model.q_proj.weight[0, :2] = torch.tensor([0.4, 1.0])
    inventory = LinearInventory(("q_proj",), ())
    uniform = scalar_pattern_tensors(model, inventory)["q_proj"]
    diagonal = torch.zeros(256)
    diagonal[1] = 1
    weighted = scalar_pattern_tensors(
        model, inventory, statistics={"q_proj.diag": diagonal},
        importance_mode="diagonal")["q_proj"]
    assert uniform[0, 0] == 1
    assert weighted[0, 0] == 0
    assert weighted[0, 1] == 1


def test_facility_location_is_deterministic_and_keeps_every_anchor():
    counts = torch.ones(6561, dtype=torch.float64)
    corpus = PatternCorpus.from_counts(counts)
    shapes = canonical_shapes()
    anchors = required_anchor_rows(shapes)
    first, report_a = facility_location_select(
        corpus, select_count=len(anchors) + 2, swap_limit=0, return_trace=True)
    second, report_b = facility_location_select(
        corpus, select_count=len(anchors) + 2, swap_limit=0, return_trace=True)
    assert torch.equal(first, second)
    assert report_a == report_b
    selected_ids = set(base3_ids(first).tolist())
    assert set(base3_ids(shapes[torch.tensor(anchors)]).tolist()) <= selected_ids
    assert report_a["anchor_base3_ids"] == base3_ids(
        shapes[torch.tensor(anchors)]).tolist()
    assert all(b <= a for a, b in zip(report_a["objectives"], report_a["objectives"][1:]))


def test_lazy_greedy_is_exactly_the_eager_canonical_selection():
    corpus = PatternCorpus.from_counts(torch.arange(6561, dtype=torch.float64) % 23 + 1)
    candidates = canonical_shapes()
    anchors = required_anchor_rows(candidates)
    count = len(anchors) + 8
    got, report = facility_location_select(
        corpus, select_count=count, swap_limit=0, return_trace=True)

    distances = _distance_matrix(corpus, candidates)
    selected = list(anchors)
    minimum = distances[:, selected].min(1).values
    eager_objectives = [float((corpus.demand * minimum).sum())]
    while len(selected) < count:
        gains = (corpus.demand[:, None] * (
            minimum[:, None] - torch.minimum(minimum[:, None], distances)
        ).clamp_min(0)).sum(0)
        gains[torch.tensor(selected)] = -torch.inf
        incoming = int(torch.argmax(gains))
        selected.append(incoming)
        minimum = torch.minimum(minimum, distances[:, incoming])
        eager_objectives.append(float((corpus.demand * minimum).sum()))
    assert torch.equal(got, candidates[torch.tensor(selected)])
    assert report["objectives"] == eager_objectives


def test_optimized_swap_report_matches_direct_final_objective():
    corpus = PatternCorpus.from_counts(torch.arange(6561, dtype=torch.float64) % 17 + 1)
    selected, report = facility_location_select(
        corpus, select_count=256, swap_limit=1, return_trace=True)
    direct = _distance_matrix(corpus, selected).min(1).values
    objective = float((corpus.demand * direct).sum())
    assert objective == report["objectives"][-1]


def test_v11_product_solver_meets_structural_unique_count():
    counts = torch.arange(6561, dtype=torch.float64) % 17 + 1
    book = build_product_codebook(
        "product", "v11", PatternCorpus.from_counts(counts), swap_limit=0)
    assert book.tables["product_a"].shape == (32, 4)
    assert book.tables["product_b"].shape == (32, 4)
    assert int(book.legal_index_mask().sum()) == 2047
