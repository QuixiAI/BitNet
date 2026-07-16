from __future__ import annotations

import copy
import json
import math

import pytest
import torch
from torch import nn

from bitnet_train.tq1.codebook import sign_canonical_codebook
from bitnet_train.tq1.cli import _parse_args, _validate_arguments
from bitnet_train.tq1.calibration import file_sha256
from bitnet_train.tq1.codebook_study import (
    PROJECTION_FAMILIES, codebook_coverage, combine_distortion_metrics,
    distortion_metrics, finalize_universal_codebook_study, load_pattern_corpus,
    save_pattern_corpus, study_definition_sha256,
    validate_universal_codebook_study)
from bitnet_train.tq1.solver import PatternCorpus, canonical_shapes
from bitnet_train.tq1.pipeline import (
    LinearInventory, scalar_pattern_family_corpora)
from quant.codebook_study import _measure_model_directory


def _book():
    shapes = canonical_shapes()
    zero = shapes[(shapes == 0).all(1)]
    nonzero = shapes[~(shapes == 0).all(1)][:1023]
    return sign_canonical_codebook(
        "universal_v11", "v11", torch.cat((zero, nonzero)), scope="universal",
        provenance={
            "source": "construction_pattern_corpus",
            "pattern_corpus_sha256": "d" * 64,
            "pattern_corpus_metadata_sha256": "a" * 64,
            "solver_config_sha256": "e" * 64,
            "split_manifest_sha256": "f" * 64,
            "source_models": [{
                "model_id": "org/source", "revision": "a" * 40,
                "family": "llama"}],
        })


def _distortion(mass=10.0):
    return {
        "sensitivity_metric": "diagonal",
        "pattern_mass": mass,
        "exact_hit_mass": mass * 0.5,
        "exact_hit_rate": 0.5,
        "frequency_weighted_distortion_sum": mass * 0.2,
        "frequency_weighted_distortion_mean": 0.2,
        "sensitivity_weighted_distortion_sum": mass * 0.3,
        "sensitivity_weighted_distortion_mean": 0.3,
    }


def _metrics(*, ce=2.0, kl=0.1, agreement=0.8):
    return {
        "token_count": 100,
        "cross_entropy": ce,
        "perplexity": math.exp(ce),
        "teacher_kl_mean": kl,
        "teacher_kl_p50": kl * 0.5,
        "teacher_kl_p95": kl * 2,
        "teacher_kl_p99": kl * 3,
        "top_token_agreement": agreement,
    }


def _quality_row(book, model, phase, activation):
    reference = _metrics()
    candidate = _metrics(ce=2.01, kl=0.11, agreement=0.79)
    reference_tasks = {"knowledge": 0.8, "reasoning": 0.7}
    candidate_tasks = {"knowledge": 0.75, "reasoning": 0.68}
    return {
        "model": model,
        "phase": phase,
        "activation_mode": activation,
        "reference_artifact_sha256": "4" * 64,
        "reference_codebook_sha256": "5" * 64,
        "reference_codebook_scope": "model",
        "candidate_artifact_sha256": "6" * 64,
        "candidate_codebook_sha256": book.sha256(),
        "evaluation_data_sha256": "7" * 64,
        "reference_metrics": reference,
        "candidate_metrics": candidate,
        "reference_tasks": reference_tasks,
        "candidate_tasks": candidate_tasks,
        "deltas": {
            "cross_entropy": candidate["cross_entropy"] - reference["cross_entropy"],
            "perplexity": candidate["perplexity"] - reference["perplexity"],
            "teacher_kl_mean": (
                candidate["teacher_kl_mean"] - reference["teacher_kl_mean"]),
            "teacher_kl_p99": (
                candidate["teacher_kl_p99"] - reference["teacher_kl_p99"]),
            "top_token_agreement": (
                candidate["top_token_agreement"] - reference["top_token_agreement"]),
            "aggregate_task_score": (
                sum(candidate_tasks.values()) / len(candidate_tasks)
                - sum(reference_tasks.values()) / len(reference_tasks)),
        },
    }


def _kernel(workload):
    median = 1.0 if workload == "decode" else 10.0
    return {
        "id": f"cpu_{workload}",
        "backend": "native_cpu",
        "workload": workload,
        "shape": [1 if workload == "decode" else 128, 256, 256],
        "activation_mode": "a8_token",
        "output_dtype": "float32",
        "reference_representation": "model_loaded_codebook",
        "candidate_representation": "universal_embedded_codebook",
        "warmups": 5,
        "iterations": 20,
        "reference_timing": {
            "p20_ms": median * 0.9, "median_ms": median, "p80_ms": median * 1.1},
        "candidate_timing": {
            "p20_ms": median * 0.945, "median_ms": median * 1.05,
            "p80_ms": median * 1.155},
        "correctness": {
            "atol": 1e-6, "rtol": 1e-6, "max_abs": 1e-7,
            "max_rel": 1e-7, "passed": True},
        "device": {"model": "unit cpu"},
        "toolchain": {"compiler": "unit"},
        "command": f"bench --workload {workload}",
        "run_sha256": "8" * 64,
        "codebook_sha256": None,
    }


def _report(book):
    identities = [
        {"model_id": "org/heldout-a", "revision": "b" * 40, "family": "llama"},
        {"model_id": "org/heldout-b", "revision": "c" * 40, "family": "llama"},
    ]
    models = []
    for model_index, identity in enumerate(identities):
        families = {}
        for family_index, family in enumerate(PROJECTION_FAMILIES):
            families[family] = {
                "pattern_corpus_sha256": format(family_index + 9, "x")[-1] * 64,
                "metrics": _distortion(10.0 + family_index),
            }
        models.append({
            "identity": identity,
            "pattern_corpus_sha256": str(model_index + 1) * 64,
            "families": families,
            "metrics": combine_distortion_metrics(
                evidence["metrics"] for evidence in families.values()),
        })
    report = {
        "schema": 1,
        "codebook": book.ref().to_dict(),
        "construction": {
            "source_models": [{
                "model_id": "org/source", "revision": "a" * 40,
                "family": "llama"}],
            "pattern_corpus_sha256": "d" * 64,
            "pattern_corpus_metadata_sha256": "a" * 64,
            "solver_config_sha256": "e" * 64,
            "split_manifest_sha256": "f" * 64,
        },
        "heldout": {
            "split_manifest_sha256": "0" * 64,
            "models": models,
            "aggregate": combine_distortion_metrics(
                model["metrics"] for model in models),
        },
        "universe_coverage": codebook_coverage(book),
        "quality_comparisons": [
            _quality_row(book, identity, phase, activation)
            for identity in identities for phase in ("ptq", "qat")
            for activation in ("w_only", "w_a8")
        ],
        "kernel_performance": [_kernel("decode"), _kernel("prefill")],
        "predeclared_gates": {
            "declared_before_evaluation": True,
            "study_definition_sha256": "0" * 64,
            "minimum_aggregate_exact_hit_rate": 0.4,
            "minimum_model_exact_hit_rate": 0.4,
            "minimum_family_exact_hit_rate": 0.4,
            "maximum_universe_squared_trit_distance": 4,
            "maximum_aggregate_frequency_weighted_distortion": 0.25,
            "maximum_model_frequency_weighted_distortion": 0.25,
            "maximum_family_frequency_weighted_distortion": 0.25,
            "maximum_aggregate_sensitivity_weighted_distortion": 0.35,
            "maximum_model_sensitivity_weighted_distortion": 0.35,
            "maximum_family_sensitivity_weighted_distortion": 0.35,
            "maximum_quality_cross_entropy_increase": 0.02,
            "maximum_quality_teacher_kl_mean_increase": 0.02,
            "maximum_quality_teacher_kl_p99_increase": 0.04,
            "maximum_quality_top_token_agreement_drop": 0.02,
            "maximum_quality_task_score_drop": 0.05,
            "maximum_decode_latency_ratio": 1.1,
            "maximum_prefill_latency_ratio": 1.1,
        },
        "decision": None,
        "commands": ["quant/codebook_study.py validate --frozen-inputs"],
        "provenance": {"repository_commit": "9" * 40},
    }
    for row in report["kernel_performance"]:
        row["codebook_sha256"] = book.sha256()
    report["predeclared_gates"]["study_definition_sha256"] = \
        study_definition_sha256(report)
    return finalize_universal_codebook_study(report, book)


def test_complete_universe_coverage_and_sensitivity_distortion_are_exact():
    book = _book()
    coverage = codebook_coverage(book)
    assert sum(coverage["squared_trit_distance_histogram"].values()) == 6561
    assert coverage["squared_trit_distance_histogram"]["0"] == 2047
    assert coverage["max_squared_trit_distance"] == 4
    corpus = PatternCorpus(
        torch.ones(6561, dtype=torch.float64),
        diagonal=torch.ones((6561, 8), dtype=torch.float64))
    metrics = distortion_metrics(book, corpus)
    assert metrics["exact_hit_mass"] == 2047
    assert metrics["frequency_weighted_distortion_mean"] == pytest.approx(
        metrics["sensitivity_weighted_distortion_mean"])


def test_pattern_corpus_round_trip_is_safe_and_exact(tmp_path):
    corpus = PatternCorpus(
        torch.arange(6561, dtype=torch.float64) + 1,
        diagonal=torch.ones((6561, 8), dtype=torch.float64))
    path = save_pattern_corpus(
        tmp_path / "family.safetensors", corpus,
        metadata={"model_id": "org/model", "family": "q_proj"})
    loaded, metadata = load_pattern_corpus(path)
    assert torch.equal(loaded.demand, corpus.demand)
    assert torch.equal(loaded.diagonal, corpus.diagonal)
    assert metadata == {"model_id": "org/model", "family": "q_proj"}


def test_heldout_model_directory_measures_every_hashed_family(tmp_path):
    identity = {"model_id": "org/heldout", "revision": "b" * 40,
                "family": "llama"}
    split = "c" * 64
    files = {}
    for index, family in enumerate(PROJECTION_FAMILIES):
        demand = torch.zeros(6561, dtype=torch.float64)
        demand[index] = 1
        diagonal = torch.zeros((6561, 8), dtype=torch.float64)
        diagonal[index] = 1
        path = save_pattern_corpus(
            tmp_path / f"{family}.safetensors",
            PatternCorpus(demand, diagonal=diagonal), metadata={
                "role": "heldout", "model_identity": identity,
                "projection_family": family, "split_manifest_sha256": split})
        files[path.name] = file_sha256(path)
    manifest = {
        "schema": 1, "role": "heldout", "model_identity": identity,
        "statistics_sha256": "d" * 64,
        "statistics_metadata_sha256": "e" * 64,
        "split_manifest_sha256": split, "sensitivity_metric": "diagonal",
        "weighting": "parameter", "target_modules": ["unit"], "files": files,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = _measure_model_directory(tmp_path, _book())
    assert result["identity"] == identity
    assert set(result["families"]) == set(PROJECTION_FAMILIES)
    assert result["metrics"]["pattern_mass"] == len(PROJECTION_FAMILIES)


def test_distortion_rejects_non_psd_covariance():
    covariance = torch.eye(8, dtype=torch.float64)[None].repeat(6561, 1, 1)
    covariance[0, 0, 0] = -1
    with pytest.raises(ValueError, match="positive semidefinite"):
        PatternCorpus(torch.ones(6561), covariance=covariance)


def test_heldout_family_extraction_uses_calibration_sensitivity():
    model = nn.Module()
    model.layers = nn.ModuleDict({
        family: nn.Linear(8, 2, bias=False) for family in PROJECTION_FAMILIES})
    targets = tuple(f"layers.{family}" for family in PROJECTION_FAMILIES)
    statistics = {
        f"{name}.diag": torch.arange(1, 9, dtype=torch.float32)
        for name in targets}
    corpora = scalar_pattern_family_corpora(
        model, LinearInventory(targets, ()), statistics,
        importance_mode="diagonal", weighting="parameter")
    assert set(corpora) == set(PROJECTION_FAMILIES)
    assert all(corpus.demand.sum() == 2 for corpus in corpora.values())
    assert all(corpus.diagonal is not None for corpus in corpora.values())


def test_universal_study_recomputes_disjointness_aggregates_and_acceptance():
    book = _book()
    report = _report(book)
    decision = validate_universal_codebook_study(report, book)
    assert decision["accepted"]
    assert all(decision["checks"].values())

    tampered = copy.deepcopy(report)
    tampered["decision"]["accepted"] = False
    with pytest.raises(ValueError, match="decision differs"):
        validate_universal_codebook_study(tampered, book)

    overlap = copy.deepcopy(report)
    overlap["heldout"]["models"][0]["identity"] = copy.deepcopy(
        overlap["construction"]["source_models"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_universal_codebook_study(overlap, book)

    aggregate = copy.deepcopy(report)
    aggregate["heldout"]["aggregate"]["pattern_mass"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        validate_universal_codebook_study(aggregate, book)

    wrong_family = copy.deepcopy(report)
    wrong_family["quality_comparisons"][0]["model"] = {
        **wrong_family["quality_comparisons"][0]["model"], "family": "other"}
    with pytest.raises(ValueError, match="identity differs"):
        validate_universal_codebook_study(wrong_family, book)


def test_universal_study_requires_ptq_qat_and_decode_prefill_coverage():
    book = _book()
    quality = copy.deepcopy(_report(book))
    quality["quality_comparisons"].pop()
    with pytest.raises(ValueError, match="quality coverage"):
        validate_universal_codebook_study(quality, book)

    performance = copy.deepcopy(_report(book))
    performance["kernel_performance"].pop()
    with pytest.raises(ValueError, match="decode and prefill"):
        validate_universal_codebook_study(performance, book)


def test_quant_cli_fails_before_model_load_for_invalid_codebook_sources(tmp_path):
    output = str(tmp_path / "artifact")
    missing = _parse_args([
        "--output", output, "--codebook-source", "universal"])
    with pytest.raises(ValueError, match="requires --codebook-corpus"):
        _validate_arguments(missing)
    direct = _parse_args([
        "--output", output, "--profile", "tq1_v11-i-r"])
    with pytest.raises(ValueError, match="direct-joint"):
        _validate_arguments(direct)
    valid = _parse_args([
        "--output", output, "--codebook-source", "universal",
        "--codebook-corpus", str(tmp_path / "construction.safetensors")])
    _validate_arguments(valid)
