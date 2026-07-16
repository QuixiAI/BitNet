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
CAPABILITY_SUITE_SCHEMA = 1
CAPABILITY_REPORT_SCHEMA = 1
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

REQUIRED_CAPABILITY_TASKS = {
    "mmlu_redux": "knowledge",
    "musr": "reasoning",
    "gsm8k": "math",
    "math_500": "math",
    "humaneval_plus": "code",
    "mbpp_plus": "code",
    "ifeval": "instruction",
    "ifbench": "instruction",
    "bfcl_v3_single_turn": "tools",
    "bfcl_v3_multi_turn": "tools",
    "long_context_4k": "long_context",
    "long_context_16k": "long_context",
    "long_context_32k": "long_context",
}
CAPABILITY_MODES = {"w_only", "w_a8"}
TASK_SCORERS = {
    "mmlu_redux": {"multiple_choice"},
    "musr": {"multiple_choice", "exact"},
    "gsm8k": {"numeric"},
    "math_500": {"numeric"},
    "humaneval_plus": {"code_execution"},
    "mbpp_plus": {"code_execution"},
    "ifeval": {"constraint_fraction"},
    "ifbench": {"constraint_fraction"},
    "bfcl_v3_single_turn": {"bfcl_ast", "tool_execution"},
    "bfcl_v3_multi_turn": {"bfcl_ast", "tool_execution"},
    "long_context_4k": {"retrieval_exact", "exact"},
    "long_context_16k": {"retrieval_exact", "exact"},
    "long_context_32k": {"retrieval_exact", "exact"},
}


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


def _immutable_revision(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) not in {40, 64} \
            or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a full immutable 40/64-hex revision")
    return value


def canonical_document_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def validate_capability_suite(suite: Mapping[str, Any]) -> None:
    """Validate the immutable benchmark/scorer identity used by QI-3."""
    required = {"schema", "name", "tasks", "aggregate_method"}
    if not isinstance(suite, Mapping) or set(suite) != required:
        raise ValueError("capability suite has an invalid top-level schema")
    if suite["schema"] != CAPABILITY_SUITE_SCHEMA:
        raise ValueError("unsupported capability suite schema")
    if not isinstance(suite["name"], str) or not suite["name"]:
        raise ValueError("capability suite name must be nonempty")
    if suite["aggregate_method"] != "macro_mean_task_normalized_score":
        raise ValueError("capability suite aggregate method is not pinned")
    tasks = suite["tasks"]
    if not isinstance(tasks, list) or len(tasks) != len(REQUIRED_CAPABILITY_TASKS):
        raise ValueError("capability suite task inventory is incomplete")
    fields = {
        "id", "capability", "dataset", "config", "revision", "split",
        "prompt_template_sha256", "scorer", "scorer_version_sha256",
        "execution_image_digest", "seed", "max_generation_tokens", "backend",
        "deterministic", "context_length",
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping) or set(task) != fields:
            raise ValueError(f"capability suite task {index} has an invalid schema")
        identity = task["id"]
        if identity in by_id or identity not in REQUIRED_CAPABILITY_TASKS:
            raise ValueError(f"capability suite has unknown/duplicate task {identity!r}")
        if task["capability"] != REQUIRED_CAPABILITY_TASKS[identity]:
            raise ValueError(f"{identity}: capability mismatch")
        for key in ("dataset", "revision", "split", "scorer", "backend"):
            if not isinstance(task[key], str) or not task[key]:
                raise ValueError(f"{identity}.{key} must be nonempty")
        _immutable_revision(task["revision"], f"{identity}.revision")
        if task["scorer"] not in TASK_SCORERS[identity]:
            raise ValueError(f"{identity}: scorer is not task-appropriate")
        if task["config"] is not None and not isinstance(task["config"], str):
            raise ValueError(f"{identity}.config must be a string or null")
        _sha256(task["prompt_template_sha256"], f"{identity}.prompt_template_sha256")
        _sha256(task["scorer_version_sha256"], f"{identity}.scorer_version_sha256")
        image = task["execution_image_digest"]
        if task["capability"] in {"code", "tools"}:
            if not isinstance(image, str) or not image.startswith("sha256:"):
                raise ValueError(f"{identity}: code/tool execution image must be pinned")
            _sha256(image[7:], f"{identity}.execution_image_digest")
        elif image is not None:
            raise ValueError(f"{identity}: execution image must be null")
        if isinstance(task["seed"], bool) or not isinstance(task["seed"], int):
            raise ValueError(f"{identity}.seed must be an integer")
        if isinstance(task["max_generation_tokens"], bool) \
                or not isinstance(task["max_generation_tokens"], int) \
                or task["max_generation_tokens"] < 1:
            raise ValueError(f"{identity}.max_generation_tokens must be positive")
        if task["deterministic"] is not True:
            raise ValueError(f"{identity}: deterministic execution is required")
        expected_context = ({"long_context_4k": 4096, "long_context_16k": 16384,
                             "long_context_32k": 32768}.get(identity))
        if task["context_length"] != expected_context:
            raise ValueError(f"{identity}: context length is not pinned correctly")
        by_id[identity] = task
    if set(by_id) != set(REQUIRED_CAPABILITY_TASKS):
        raise ValueError("capability suite is missing required tasks")


def _validate_task_scores(scores: Mapping[str, Any], *, prefix: str) -> None:
    if not isinstance(scores, Mapping) or set(scores) != set(REQUIRED_CAPABILITY_TASKS):
        raise ValueError(f"{prefix} task result inventory is incomplete")
    fields = {"score", "sample_count", "deterministic_parse_failures",
              "llm_fallback_count", "output_sha256"}
    for task, result in scores.items():
        if not isinstance(result, Mapping) or set(result) != fields:
            raise ValueError(f"{prefix}.{task} has an invalid schema")
        score = _finite_number(result["score"], f"{prefix}.{task}.score")
        if not 0 <= score <= 1:
            raise ValueError(f"{prefix}.{task}.score must be in [0,1]")
        count = result["sample_count"]
        failures = result["deterministic_parse_failures"]
        fallbacks = result["llm_fallback_count"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (count, failures, fallbacks)) or count < 1:
            raise ValueError(f"{prefix}.{task} counts are invalid")
        if fallbacks > failures or failures > count:
            raise ValueError(f"{prefix}.{task} fallback counts are inconsistent")
        _sha256(result["output_sha256"], f"{prefix}.{task}.output_sha256")


def _task_macro(scores: Mapping[str, Mapping[str, Any]]) -> float:
    return sum(float(scores[name]["score"]) for name in scores) / len(scores)


def _capability_macros(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for task, capability in REQUIRED_CAPABILITY_TASKS.items():
        grouped[capability].append(float(scores[task]["score"]))
    return {name: sum(values) / len(values) for name, values in sorted(grouped.items())}


def validate_capability_report(report: Mapping[str, Any], suite: Mapping[str, Any],
                               *, quant_spec_sha256: str) -> dict[str, Any]:
    """Fail closed and recompute all QI-3 gates from raw task scores."""
    validate_capability_suite(suite)
    fields = {
        "schema", "suite_sha256", "quant_spec_sha256", "dense_result_sha256",
        "evaluation_data_sha256", "predeclared_gates", "dense", "modes",
        "teacher_kl", "stratified", "rerun", "commands", "provenance",
    }
    if not isinstance(report, Mapping) or set(report) != fields:
        raise ValueError("capability report has an invalid top-level schema")
    if report["schema"] != CAPABILITY_REPORT_SCHEMA:
        raise ValueError("unsupported capability report schema")
    if report["suite_sha256"] != canonical_document_sha256(suite):
        raise ValueError("capability report suite hash mismatch")
    if report["quant_spec_sha256"] != quant_spec_sha256:
        raise ValueError("capability report QuantSpec mismatch")
    for key in ("dense_result_sha256", "evaluation_data_sha256"):
        _sha256(report[key], key)
    gates = report["predeclared_gates"]
    gate_fields = {
        "declared_before_candidate", "dense_result_sha256", "aggregate_retention",
        "per_capability_retention", "maximum_task_regression", "teacher_kl_mean",
        "teacher_kl_p95", "teacher_kl_p99", "rerun_max_score_delta",
    }
    if not isinstance(gates, Mapping) or set(gates) != gate_fields \
            or gates["declared_before_candidate"] is not True \
            or gates["dense_result_sha256"] != report["dense_result_sha256"]:
        raise ValueError("capability gates were not predeclared against this dense result")
    for key in gate_fields - {"declared_before_candidate", "dense_result_sha256"}:
        value = _finite_number(gates[key], f"predeclared_gates.{key}")
        if value < 0:
            raise ValueError(f"predeclared_gates.{key} must be nonnegative")
    dense = report["dense"]
    if not isinstance(dense, Mapping) or set(dense) != {"tasks"}:
        raise ValueError("dense capability results have an invalid schema")
    _validate_task_scores(dense["tasks"], prefix="dense.tasks")
    modes = report["modes"]
    if not isinstance(modes, Mapping) or set(modes) != CAPABILITY_MODES:
        raise ValueError("capability report must name W-only and W+A8 separately")
    for mode, value in modes.items():
        if not isinstance(value, Mapping) or set(value) != {"tasks"}:
            raise ValueError(f"{mode} results have an invalid schema")
        _validate_task_scores(value["tasks"], prefix=f"modes.{mode}.tasks")
    kl = report["teacher_kl"]
    if not isinstance(kl, Mapping) or set(kl) != CAPABILITY_MODES:
        raise ValueError("teacher KL must be reported for both quantization modes")
    for mode, values in kl.items():
        if not isinstance(values, Mapping) or set(values) != {"mean", "p50", "p95", "p99"}:
            raise ValueError(f"teacher_kl.{mode} has an invalid schema")
        numbers = [_finite_number(values[key], f"teacher_kl.{mode}.{key}")
                   for key in ("mean", "p50", "p95", "p99")]
        if min(numbers) < -1e-6 or not numbers[1] <= numbers[2] <= numbers[3]:
            raise ValueError(f"teacher_kl.{mode} is invalid")
    stratified = report["stratified"]
    if not isinstance(stratified, Mapping) or set(stratified) != CAPABILITY_MODES:
        raise ValueError("stratified evidence must cover both modes")
    for mode, dimensions in stratified.items():
        if not isinstance(dimensions, Mapping) or set(dimensions) != {"length", "language"}:
            raise ValueError(f"stratified.{mode} must contain length and language")
        for dimension, buckets in dimensions.items():
            if not isinstance(buckets, Mapping) or not buckets:
                raise ValueError(f"stratified.{mode}.{dimension} must be nonempty")
            for bucket, score in buckets.items():
                if not isinstance(bucket, str) or not bucket or not 0 <= _finite_number(
                        score, f"stratified.{mode}.{dimension}.{bucket}") <= 1:
                    raise ValueError("invalid stratified score")
    rerun = report["rerun"]
    if not isinstance(rerun, Mapping) or set(rerun) != CAPABILITY_MODES:
        raise ValueError("exact rerun evidence must cover both modes")
    decisions: dict[str, Any] = {}
    dense_tasks = dense["tasks"]
    dense_aggregate = _task_macro(dense_tasks)
    dense_caps = _capability_macros(dense_tasks)
    def retention(candidate: float, reference: float) -> float:
        return 1.0 if reference == 0 and candidate == 0 else candidate / max(reference, 1e-12)
    for mode, value in modes.items():
        tasks = value["tasks"]
        repeated = rerun[mode]
        if not isinstance(repeated, Mapping) or set(repeated) != {
                "score_delta", "first_output_sha256", "second_output_sha256"}:
            raise ValueError(f"rerun.{mode} has an invalid schema")
        delta = _finite_number(repeated["score_delta"], f"rerun.{mode}.score_delta")
        if delta < 0:
            raise ValueError(f"rerun.{mode}.score_delta must be nonnegative")
        _sha256(repeated["first_output_sha256"], f"rerun.{mode}.first_output_sha256")
        _sha256(repeated["second_output_sha256"], f"rerun.{mode}.second_output_sha256")
        aggregate = _task_macro(tasks)
        caps = _capability_macros(tasks)
        task_regression = max(float(dense_tasks[name]["score"])
                              - float(tasks[name]["score"]) for name in tasks)
        checks = {
            "aggregate_retention": retention(aggregate, dense_aggregate)
                >= float(gates["aggregate_retention"]),
            "per_capability_retention": all(
                retention(caps[name], dense_caps[name])
                >= float(gates["per_capability_retention"]) for name in caps),
            "maximum_task_regression": task_regression
                <= float(gates["maximum_task_regression"]),
            "teacher_kl_mean": float(kl[mode]["mean"]) <= float(gates["teacher_kl_mean"]),
            "teacher_kl_p95": float(kl[mode]["p95"]) <= float(gates["teacher_kl_p95"]),
            "teacher_kl_p99": float(kl[mode]["p99"]) <= float(gates["teacher_kl_p99"]),
            "rerun_stability": delta <= float(gates["rerun_max_score_delta"]),
        }
        decisions[mode] = {
            "passed": all(checks.values()), "checks": checks,
            "aggregate_score": aggregate,
            "aggregate_retention": retention(aggregate, dense_aggregate),
            "capability_scores": caps, "maximum_task_regression": task_regression,
        }
    if not isinstance(report["commands"], list) or not report["commands"] \
            or not all(isinstance(item, str) and item for item in report["commands"]):
        raise ValueError("capability report commands must be nonempty")
    if not isinstance(report["provenance"], Mapping) or not report["provenance"]:
        raise ValueError("capability report provenance must be nonempty")
    return decisions


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
