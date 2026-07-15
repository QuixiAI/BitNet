from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bitnet_train.tq1.evaluation import (
    REQUIRED_CORE_BASELINES, evaluate_records, validate_quality_report)


class _Tokenizer:
    def __len__(self):
        return 7

    def __call__(self, text, **kwargs):
        del kwargs
        values = [1] + [2 + ord(char) % 5 for char in text][:15]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}


class _Model(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.offset = float(offset)

    def forward(self, input_ids, use_cache=False):
        del use_cache
        vocab = torch.arange(7, device=input_ids.device).float()
        logits = -(vocab - input_ids[..., None].float()).square()
        logits[..., 3] += self.offset
        return SimpleNamespace(logits=logits)


def test_heldout_evaluator_reports_exact_metrics_and_strata():
    result = evaluate_records(
        _Model(0.1), _Model(0.4), _Tokenizer(), [
            {"text": "hello", "language": "en", "task": "chat"},
            {"text": "bonjour", "language": "fr", "task": "prose"},
        ])
    metrics = result["metrics"]
    assert result["record_count"] == 2
    assert metrics["token_count"] == 12
    assert metrics["perplexity"] == pytest.approx(math.exp(metrics["cross_entropy"]))
    assert metrics["teacher_kl_mean"] >= 0
    assert 0 <= metrics["top_token_agreement"] <= 1
    assert set(result["stratified"]["language"]) == {"en", "fr"}
    assert set(result["stratified"]["task"]) == {"chat", "prose"}


def _quality_report():
    metrics = {
        "token_count": 100,
        "cross_entropy": 2.0,
        "perplexity": math.exp(2.0),
        "teacher_kl_mean": 0.1,
        "teacher_kl_p50": 0.05,
        "teacher_kl_p95": 0.3,
        "teacher_kl_p99": 0.5,
        "top_token_agreement": 0.75,
    }
    return {
        "schema": 1,
        "quant_spec_sha256": "a" * 64,
        "evaluation_data": {
            "dataset": "heldout", "revision": "r1", "split": "test",
            "sha256": "b" * 64, "tokenizer_sha256": "c" * 64,
            "record_count": 10, "token_count": 100,
            "calibration_disjoint": True, "policy_selection_disjoint": True,
        },
        "predeclared_gates": {"declared_before_run": True, "max_ppl_delta": 0.1},
        "profiles": {name: dict(metrics) for name in REQUIRED_CORE_BASELINES},
        "stratified": {
            "language": {"en": dict(metrics)},
            "task": {"prose": dict(metrics)},
            "length": {"0001-0128": dict(metrics)},
        },
        "downstream_tasks": {"revision": "pinned", "results": {"task": 0.5}},
        "instruction_chat": {"suite": "pinned", "score": 0.5},
        "long_context": {"suite": "pinned", "score": 0.5},
        "calibration_convergence": {"sample_counts": [128, 256], "converged": True},
        "commands": ["evaluate --frozen-config"],
        "provenance": {"repository_commit": "d" * 40},
    }


def test_quality_report_requires_complete_disjoint_baseline_matrix():
    report = _quality_report()
    validate_quality_report(report, "a" * 64)
    missing = copy.deepcopy(report)
    del missing["profiles"][REQUIRED_CORE_BASELINES[-1]]
    with pytest.raises(ValueError, match="core baselines"):
        validate_quality_report(missing, "a" * 64)
    overlapping = copy.deepcopy(report)
    overlapping["evaluation_data"]["calibration_disjoint"] = False
    with pytest.raises(ValueError, match="disjoint"):
        validate_quality_report(overlapping, "a" * 64)
