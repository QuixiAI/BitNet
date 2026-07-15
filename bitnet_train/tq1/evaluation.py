"""Held-out model evaluation and fail-closed TQ1 quality-report validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


QUALITY_REPORT_SCHEMA = 1
REQUIRED_CORE_BASELINES = (
    "dense_teacher",
    "lossless_ternary",
    "tq1_0",
    "tq2_0",
    "iq1_s",
    "v12_j_ptq",
    "v12_j_qat",
    "v11_j_ptq",
    "v11_j_qat",
)
REQUIRED_METRICS = {
    "token_count", "cross_entropy", "perplexity", "teacher_kl_mean",
    "teacher_kl_p50", "teacher_kl_p95", "teacher_kl_p99",
    "top_token_agreement",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_metric_set(metrics: Mapping[str, Any], *, name: str) -> None:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{name} metrics must be an object")
    missing = REQUIRED_METRICS - set(metrics)
    if missing:
        raise ValueError(f"{name} metrics are missing {sorted(missing)}")
    count = metrics["token_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"{name}.token_count must be a positive integer")
    ce = _finite_number(metrics["cross_entropy"], f"{name}.cross_entropy")
    ppl = _finite_number(metrics["perplexity"], f"{name}.perplexity")
    if ce < 0 or ppl < 1:
        raise ValueError(f"{name} CE/perplexity is outside its legal range")
    expected_ppl = math.exp(min(ce, math.log(float.fromhex("0x1.fffffffffffffp+1023"))))
    if not math.isclose(ppl, expected_ppl, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"{name} perplexity is inconsistent with cross entropy")
    tails = [
        _finite_number(metrics[key], f"{name}.{key}")
        for key in ("teacher_kl_mean", "teacher_kl_p50", "teacher_kl_p95",
                    "teacher_kl_p99")
    ]
    if any(value < -1e-6 for value in tails) or not tails[1] <= tails[2] <= tails[3]:
        raise ValueError(f"{name} KL metrics are invalid or non-monotonic")
    agreement = _finite_number(
        metrics["top_token_agreement"], f"{name}.top_token_agreement")
    if not 0 <= agreement <= 1:
        raise ValueError(f"{name} top-token agreement must be in [0,1]")


def validate_quality_report(report: Mapping[str, Any], quant_spec_sha256: str, *,
                            require_core_matrix: bool = True) -> None:
    """Validate the evidence needed to set ``quality_qualified=true``.

    A report may carry additional benchmark fields, but it cannot omit identity,
    disjointness, predeclared gates, core baseline results, stratified results,
    downstream/chat/long-context evidence, or calibration convergence.
    """
    required = {
        "schema", "quant_spec_sha256", "evaluation_data", "predeclared_gates",
        "profiles", "stratified", "downstream_tasks", "instruction_chat",
        "long_context", "calibration_convergence", "commands", "provenance",
    }
    if not isinstance(report, Mapping) or required - set(report):
        raise ValueError(
            f"quality report is missing required fields {sorted(required - set(report))}")
    if report["schema"] != QUALITY_REPORT_SCHEMA:
        raise ValueError("unsupported quality report schema")
    if report["quant_spec_sha256"] != quant_spec_sha256:
        raise ValueError("quality report QuantSpec mismatch")
    data = report["evaluation_data"]
    data_required = {
        "dataset", "revision", "split", "sha256", "tokenizer_sha256",
        "record_count", "token_count", "calibration_disjoint",
        "policy_selection_disjoint",
    }
    if not isinstance(data, Mapping) or set(data) != data_required:
        raise ValueError("quality report evaluation_data has an invalid schema")
    for key in ("sha256", "tokenizer_sha256"):
        value = data[key]
        if not isinstance(value, str) or len(value) != 64 \
                or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"evaluation_data.{key} must be a full lowercase SHA-256")
    if not data["calibration_disjoint"] or not data["policy_selection_disjoint"]:
        raise ValueError("quality evaluation must be disjoint from calibration and policy search")
    for key in ("record_count", "token_count"):
        if isinstance(data[key], bool) or not isinstance(data[key], int) or data[key] <= 0:
            raise ValueError(f"evaluation_data.{key} must be a positive integer")
    gates = report["predeclared_gates"]
    if not isinstance(gates, Mapping) or not gates or "declared_before_run" not in gates \
            or gates["declared_before_run"] is not True:
        raise ValueError("quality gates must be nonempty and declared before the run")
    profiles = report["profiles"]
    if not isinstance(profiles, Mapping):
        raise ValueError("quality report profiles must be an object")
    missing_profiles = set(REQUIRED_CORE_BASELINES) - set(profiles)
    if require_core_matrix and missing_profiles:
        raise ValueError(f"quality report is missing core baselines {sorted(missing_profiles)}")
    for profile, metrics in profiles.items():
        validate_metric_set(metrics, name=f"profiles.{profile}")
    stratified = report["stratified"]
    if not isinstance(stratified, Mapping) or set(stratified) != {"language", "task", "length"}:
        raise ValueError("stratified results must contain language, task, and length")
    for dimension, values in stratified.items():
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"stratified.{dimension} must be a nonempty object")
        for bucket, metrics in values.items():
            validate_metric_set(metrics, name=f"stratified.{dimension}.{bucket}")
    for section in ("downstream_tasks", "instruction_chat", "long_context",
                    "calibration_convergence"):
        if not isinstance(report[section], Mapping) or not report[section]:
            raise ValueError(f"quality report {section} evidence must be nonempty")
    if not isinstance(report["commands"], list) or not report["commands"] \
            or not all(isinstance(value, str) and value for value in report["commands"]):
        raise ValueError("quality report commands must be a nonempty string list")
    if not isinstance(report["provenance"], Mapping) or not report["provenance"]:
        raise ValueError("quality report provenance must be a nonempty object")


@dataclass
class _Accumulator:
    ce_sum: float = 0.0
    kl_sum: float = 0.0
    agreement: int = 0
    count: int = 0
    kl_values: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.kl_values is None:
            self.kl_values = []

    def add(self, ce: torch.Tensor, kl: torch.Tensor, agreement: torch.Tensor) -> None:
        self.ce_sum += float(ce.double().sum())
        self.kl_sum += float(kl.double().sum())
        self.agreement += int(agreement.sum())
        self.count += int(ce.numel())
        assert self.kl_values is not None
        self.kl_values.append(kl.detach().float().cpu())

    def result(self) -> dict[str, Any]:
        if self.count <= 0 or not self.kl_values:
            raise ValueError("evaluation bucket contains no prediction tokens")
        kl = torch.cat(self.kl_values).double()
        ce = self.ce_sum / self.count
        return {
            "token_count": self.count,
            "cross_entropy": ce,
            "perplexity": math.exp(ce),
            "teacher_kl_mean": self.kl_sum / self.count,
            "teacher_kl_p50": float(torch.quantile(kl, 0.50)),
            "teacher_kl_p95": float(torch.quantile(kl, 0.95)),
            "teacher_kl_p99": float(torch.quantile(kl, 0.99)),
            "top_token_agreement": self.agreement / self.count,
        }


def _length_bucket(tokens: int) -> str:
    if tokens <= 128:
        return "0001-0128"
    if tokens <= 512:
        return "0129-0512"
    if tokens <= 2048:
        return "0513-2048"
    return "2049+"


@torch.inference_mode()
def evaluate_records(student, teacher, tokenizer, records: Iterable[Mapping[str, Any]], *,
                     device: str | torch.device = "cpu", sequence_cap: int = 4096) \
        -> dict[str, Any]:
    """Evaluate student versus teacher on held-out text records.

    Records require ``text`` and may name ``language`` and ``task``. Evaluation
    uses next-token positions only and computes exact full-vocabulary forward KL.
    """
    if sequence_cap < 2:
        raise ValueError("evaluation sequence_cap must be at least two")
    device = torch.device(device)
    student.to(device).eval()
    teacher.to(device).eval()
    totals = _Accumulator()
    buckets: dict[str, dict[str, _Accumulator]] = {
        key: defaultdict(_Accumulator) for key in ("language", "task", "length")
    }
    record_count = 0
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("text"), str) \
                or not record["text"].strip():
            raise ValueError("evaluation records require nonempty text")
        encoded = tokenizer(
            record["text"], return_tensors="pt", truncation=True,
            max_length=sequence_cap, add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        student_logits = student(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        if student_logits.shape != teacher_logits.shape \
                or student_logits.shape[-1] != len(tokenizer):
            raise ValueError("teacher/student/tokenizer vocabulary or logit shape mismatch")
        if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
            raise ValueError("evaluation produced nonfinite logits")
        targets = input_ids[:, 1:]
        ce = F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            targets.reshape(-1), reduction="none")
        log_student = F.log_softmax(student_logits, dim=-1)
        log_teacher = F.log_softmax(teacher_logits, dim=-1)
        teacher_prob = log_teacher.exp()
        kl = (teacher_prob * (log_teacher - log_student)).sum(-1).reshape(-1)
        agreement = (student_logits.argmax(-1) == teacher_logits.argmax(-1)).reshape(-1)
        totals.add(ce, kl, agreement)
        names = {
            "language": str(record.get("language", "unspecified")),
            "task": str(record.get("task", "unspecified")),
            "length": _length_bucket(int(input_ids.shape[1])),
        }
        for dimension, bucket in names.items():
            buckets[dimension][bucket].add(ce, kl, agreement)
        record_count += 1
    if record_count == 0:
        raise ValueError("evaluation retained no records with prediction tokens")
    return {
        "record_count": record_count,
        "metrics": totals.result(),
        "stratified": {
            dimension: {name: accumulator.result()
                        for name, accumulator in sorted(values.items())}
            for dimension, values in buckets.items()
        },
    }


def read_evaluation_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL objects/strings or nonempty plain-text lines without guessing."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        value = raw.strip()
        if not value:
            continue
        if path.suffix.lower() in {".jsonl", ".json"}:
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if isinstance(decoded, str):
                decoded = {"text": decoded}
            if not isinstance(decoded, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object or string")
            record = decoded
        else:
            record = {"text": value}
        records.append(record)
    if not records:
        raise ValueError("evaluation input is empty")
    return records
