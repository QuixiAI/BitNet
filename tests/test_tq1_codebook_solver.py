from __future__ import annotations

import torch

from bitnet_train.tq1.codebook import base3_ids
from bitnet_train.tq1.solver import (
    PatternCorpus,
    _distance_matrix,
    build_product_codebook,
    canonical_shapes,
    corpus_from_tensors,
    facility_location_select,
    required_anchor_rows,
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
