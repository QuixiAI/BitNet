import copy

import pytest

from bitnet_train.tq1.evaluation import (
    CAPABILITY_MODES, REQUIRED_CAPABILITY_TASKS, canonical_document_sha256,
    validate_capability_report, validate_capability_suite)
from bitnet_train.tq1.scoring import aggregate_scores, score_prediction


def _suite():
    context = {"long_context_4k": 4096, "long_context_16k": 16384,
               "long_context_32k": 32768}
    scorers = {
        "mmlu_redux": "multiple_choice", "musr": "multiple_choice",
        "gsm8k": "numeric", "math_500": "numeric",
        "humaneval_plus": "code_execution", "mbpp_plus": "code_execution",
        "ifeval": "constraint_fraction", "ifbench": "constraint_fraction",
        "bfcl_v3_single_turn": "bfcl_ast", "bfcl_v3_multi_turn": "bfcl_ast",
        "long_context_4k": "retrieval_exact",
        "long_context_16k": "retrieval_exact",
        "long_context_32k": "retrieval_exact",
    }
    return {
        "schema": 1, "name": "unit-pinned-suite",
        "aggregate_method": "macro_mean_task_normalized_score",
        "tasks": [{
            "id": name, "capability": capability, "dataset": f"org/{name}",
            "config": None, "revision": "a" * 40, "split": "test",
            "prompt_template_sha256": "b" * 64,
            "scorer": scorers[name],
            "scorer_version_sha256": "c" * 64,
            "execution_image_digest": "sha256:" + "d" * 64
                if capability in {"code", "tools"} else None,
            "seed": 7, "max_generation_tokens": 256, "backend": "torch-eager",
            "deterministic": True, "context_length": context.get(name),
        } for name, capability in REQUIRED_CAPABILITY_TASKS.items()],
    }


def _task_results(score):
    return {name: {"score": score, "sample_count": 10,
                   "deterministic_parse_failures": 0, "llm_fallback_count": 0,
                   "output_sha256": "e" * 64}
            for name in REQUIRED_CAPABILITY_TASKS}


def _report(suite):
    return {
        "schema": 1, "suite_sha256": canonical_document_sha256(suite),
        "quant_spec_sha256": "1" * 64, "dense_result_sha256": "2" * 64,
        "evaluation_data_sha256": "3" * 64,
        "predeclared_gates": {
            "declared_before_candidate": True, "dense_result_sha256": "2" * 64,
            "aggregate_retention": 0.9, "per_capability_retention": 0.85,
            "maximum_task_regression": 0.2, "teacher_kl_mean": 0.1,
            "teacher_kl_p95": 0.3, "teacher_kl_p99": 0.5,
            "rerun_max_score_delta": 0.001,
        },
        "dense": {"tasks": _task_results(0.8)},
        "modes": {mode: {"tasks": _task_results(0.76)}
                  for mode in CAPABILITY_MODES},
        "teacher_kl": {mode: {"mean": 0.02, "p50": 0.01,
                              "p95": 0.1, "p99": 0.2}
                       for mode in CAPABILITY_MODES},
        "stratified": {mode: {"length": {"short": 0.75, "long": 0.73},
                              "language": {"en": 0.76, "multi": 0.72}}
                       for mode in CAPABILITY_MODES},
        "rerun": {mode: {"score_delta": 0.0, "first_output_sha256": "4" * 64,
                         "second_output_sha256": "4" * 64}
                  for mode in CAPABILITY_MODES},
        "commands": ["evaluate --deterministic"], "provenance": {"git": "abc"},
    }


def test_capability_suite_and_predeclared_gates_are_enforced():
    suite = _suite()
    validate_capability_suite(suite)
    decisions = validate_capability_report(
        _report(suite), suite, quant_spec_sha256="1" * 64)
    assert all(result["passed"] for result in decisions.values())
    bad = _report(suite)
    bad["predeclared_gates"]["dense_result_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="predeclared"):
        validate_capability_report(bad, suite, quant_spec_sha256="1" * 64)
    bad = _report(suite)
    bad["rerun"]["w_only"]["score_delta"] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_capability_report(bad, suite, quant_spec_sha256="1" * 64)
    missing = copy.deepcopy(suite)
    missing["tasks"].pop()
    with pytest.raises(ValueError, match="inventory"):
        validate_capability_suite(missing)


def test_deterministic_scorers_fail_closed_without_llm_extraction():
    assert score_prediction("multiple_choice", " A ", ["a"]).score == 1
    assert score_prediction("numeric", "therefore 1,024", ["1024"]).score == 1
    assert score_prediction("bfcl_ast", '{"x":[1,2]}', [{"x": [1, 2]}]).score == 1
    failed = score_prediction("numeric", "unknown", ["1"])
    assert failed.score == 0 and failed.parse_error
    assert score_prediction("code_execution", "code", ["unused"],
                            execution_passed=True).score == 1
    assert aggregate_scores([failed])["deterministic_parse_failures"] == 1
