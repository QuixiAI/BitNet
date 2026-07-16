"""Deterministic, task-appropriate QI-3 scoring primitives.

Code and tool execution is deliberately outside this module: callers run the
pinned container and pass its boolean verdict here.  The scorer never silently
invokes an LLM; an orchestration layer may record an attributed fallback only
after ``parse_error`` is true.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


@dataclass(frozen=True)
class ScoreResult:
    score: float
    parsed: Any
    parse_error: str | None = None


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return " ".join(value.split())


def _last_number(value: str) -> Decimal:
    matches = re.findall(r"[-+]?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:[eE][-+]?\d+)?", value)
    if not matches:
        raise ValueError("no numeric answer found")
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("invalid numeric answer") from exc


def _canonical_json(value: str) -> Any:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("response is not valid JSON") from exc
    if not isinstance(decoded, (dict, list)):
        raise ValueError("tool response JSON must be an object or array")
    return decoded


def score_prediction(scorer: str, prediction: str, references: Iterable[Any], *,
                     execution_passed: bool | None = None,
                     numeric_abs_tolerance: float = 1e-9) -> ScoreResult:
    """Score one prediction without nondeterministic extraction."""
    if not isinstance(prediction, str):
        raise TypeError("prediction must be text")
    refs = list(references)
    if not refs:
        raise ValueError("at least one reference is required")
    try:
        if scorer in {"exact", "multiple_choice", "retrieval_exact"}:
            parsed = _normalize_text(prediction)
            expected = {_normalize_text(str(value)) for value in refs}
            return ScoreResult(float(parsed in expected), parsed)
        if scorer == "numeric":
            parsed = _last_number(prediction)
            expected = [_last_number(str(value)) for value in refs]
            tolerance = Decimal(str(numeric_abs_tolerance))
            return ScoreResult(float(any(abs(parsed - value) <= tolerance
                                         for value in expected)), str(parsed))
        if scorer in {"json_ast", "bfcl_ast"}:
            parsed = _canonical_json(prediction)
            expected = [value if isinstance(value, (dict, list))
                        else _canonical_json(str(value)) for value in refs]
            return ScoreResult(float(any(parsed == value for value in expected)), parsed)
        if scorer in {"code_execution", "tool_execution"}:
            if execution_passed is None:
                raise ValueError("pinned-container execution verdict is required")
            return ScoreResult(float(execution_passed), bool(execution_passed))
        if scorer == "constraint_fraction":
            if not all(isinstance(value, bool) for value in refs):
                raise ValueError("constraint scorer references must be booleans")
            return ScoreResult(sum(refs) / len(refs), list(refs))
        raise ValueError(f"unknown deterministic scorer {scorer!r}")
    except (ValueError, TypeError) as exc:
        return ScoreResult(0.0, None, str(exc))


def aggregate_scores(values: Iterable[ScoreResult]) -> dict[str, float | int]:
    values = list(values)
    if not values:
        raise ValueError("cannot aggregate an empty score collection")
    score = sum(item.score for item in values) / len(values)
    if not math.isfinite(score):
        raise ValueError("aggregate score is nonfinite")
    return {
        "score": score,
        "sample_count": len(values),
        "deterministic_parse_failures": sum(item.parse_error is not None for item in values),
    }
