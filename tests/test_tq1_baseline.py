import hashlib
import math

import pytest

from bitnet_train.tq1.baseline import (
    BaselineMatrix, evaluate_candidate_gates, required_baseline_rows,
    result_template)
from bitnet_train.tq1.spec import canonical_json


def _hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _identity():
    runtime = {"backend": "cpu", "threads": 1}
    return {
        "source_model": "unit", "source_revision": "a" * 40,
        "source_license": "test", "tokenizer": "unit",
        "tokenizer_revision": "b" * 40, "tokenizer_sha256": "1" * 64,
        "chat_template_sha256": "2" * 64, "calibration_dataset": "cal",
        "calibration_revision": "c" * 40, "calibration_sha256": "3" * 64,
        "evaluation_dataset": "eval", "evaluation_revision": "d" * 40,
        "evaluation_sha256": "4" * 64,
        "runtime_configuration": runtime,
        "runtime_configuration_sha256": _hash(runtime),
        "repository_commit": "e" * 40, "seeds": {"model": 7},
        "tool_versions": {"torch": "unit"},
    }


def _metrics(ppl=2.0, agreement=0.9):
    return {"token_count": 10, "cross_entropy": math.log(ppl), "perplexity": ppl,
            "teacher_kl_mean": 0.01, "teacher_kl_p50": 0.005,
            "teacher_kl_p95": 0.02, "teacher_kl_p99": 0.03,
            "top_token_agreement": agreement}


def _timing(ms):
    return {"samples": 20, "warmups": 5, "iterations": 20,
            "median_ms": ms, "p20_ms": ms * 0.9, "p80_ms": ms * 1.1}


def _fill(template, row, *, dense=False):
    template["artifact_identity"] = {"sha256": "5" * 64}
    template["storage"] = {
        "unique_logical_parameters": 100, "low_bit_unique_parameters": 0 if dense else 80,
        "high_precision_unique_parameters": 100 if dense else 20,
        "packed_weight_bytes": 400 if dense else 100,
        "canonical_artifact_bytes": 500, "final_gguf_bytes": 450,
        "backend_repack_bytes": 0, "resident_language_model_bytes": 500,
        "model_effective_bpw": 32 if dense else 8,
    }
    template["quality_by_activation"] = {
        mode: _metrics(2.0 if dense else 2.1) for mode in row.evaluation_modes}
    template["task_results"] = {"knowledge": {"score": 0.8 if dense else 0.76},
                                "code": {"score": 0.7 if dense else 0.68}}
    template["export_parity"] = {"status": "not_applicable" if dense else "pass",
                                 "max_abs": None if dense else 0.0,
                                 "max_rel": None if dense else 0.0}
    template["performance"] = {
        "decode": {"tg1": _timing(1.0 if dense else 1.1)},
        "prefill": {"pp32": _timing(2.0 if dense else 2.2)},
        "peak_memory_bytes_by_context": {"4096": 1000},
    }
    template["commands"] = ["measure"]
    template["provenance"] = {"git": "abc"}
    return template


def _gates(matrix):
    return {
        "declared_after_dense_sha256": matrix.document["dense_result_sha256"],
        "maximum_perplexity_ratio": 1.1, "maximum_teacher_kl_p99": 0.1,
        "minimum_top_token_agreement": 0.8,
        "minimum_aggregate_task_retention": 0.9,
        "minimum_capability_retention": 0.9,
        "maximum_decode_latency_ratio": 1.2,
        "maximum_prefill_latency_ratio": 1.2,
        "maximum_model_effective_bpw": 10,
        "required_export_parity_status": "pass",
    }


def test_baseline_matrix_enforces_dense_then_gates_then_candidates(tmp_path):
    rows = required_baseline_rows()[:2]
    matrix = BaselineMatrix.create(_identity(), rows=rows)
    with pytest.raises(ValueError, match="before gates"):
        result_template(matrix, rows[1].id)
    dense = _fill(result_template(matrix, "dense_teacher"), rows[0], dense=True)
    matrix.record_result(dense)
    matrix.declare_gates(_gates(matrix))
    candidate = _fill(result_template(matrix, rows[1].id), rows[1])
    matrix.record_result(candidate)
    assert matrix.document["status"] == "complete"
    assert matrix.document["verdicts"][rows[1].id]["passed"]
    path = matrix.write(tmp_path / "matrix.json")
    assert BaselineMatrix.load(path).document == matrix.document


def test_candidate_cannot_omit_tasks_or_use_not_applicable_to_bypass_parity():
    rows = required_baseline_rows()[:2]
    matrix = BaselineMatrix.create(_identity(), rows=rows)
    dense = _fill(result_template(matrix, "dense_teacher"), rows[0], dense=True)
    matrix.record_result(dense)
    gates = _gates(matrix)
    matrix.declare_gates(gates)
    candidate = _fill(result_template(matrix, rows[1].id), rows[1])
    candidate["task_results"].pop("code")
    with pytest.raises(ValueError, match="inventories"):
        evaluate_candidate_gates(candidate, dense, gates)
    candidate = _fill(result_template(matrix, rows[1].id), rows[1])
    candidate["export_parity"]["status"] = "not_applicable"
    verdict = evaluate_candidate_gates(candidate, dense, gates)
    assert not verdict["passed"] and not verdict["checks"]["export.parity"]
