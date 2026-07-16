"""Fail-closed QI-1 baseline-matrix state machine.

The expensive model/runtime measurements are deliberately external to this
module.  This module makes their ordering and identity auditable: the dense
teacher is recorded first, gates are then frozen, and only then may candidate
results be attached.  A partially populated file is useful evidence, but it is
never a completed baseline matrix.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import validate_metric_set
from .spec import canonical_json


BASELINE_MATRIX_SCHEMA = 1
ACTIVATION_MODES = ("w_only", "w_a8")
TRAINING_MODES = ("not_applicable", "ptq", "w_only", "w_a8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_hash(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its legal range")
    return result


@dataclass(frozen=True)
class BaselineRow:
    id: str
    physical_type: str
    scale_granularity: str
    producer: str
    training_mode: str
    evaluation_modes: tuple[str, ...]
    source_projection: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.id or not self.physical_type or not self.producer:
            raise ValueError("baseline row identity fields must be nonempty")
        if self.training_mode not in TRAINING_MODES:
            raise ValueError(f"invalid baseline training mode {self.training_mode!r}")
        if not self.evaluation_modes or not set(self.evaluation_modes) <= set(ACTIVATION_MODES):
            raise ValueError("baseline evaluation modes must be w_only and/or w_a8")
        if len(set(self.evaluation_modes)) != len(self.evaluation_modes):
            raise ValueError("baseline evaluation modes must be unique")
        if self.scale_granularity not in {
                "none", "lossless", "row_fp16", "block256_fp16", "group128_fp16"}:
            raise ValueError(f"invalid scale granularity {self.scale_granularity!r}")
        if self.source_projection not in {"dense", "unhealed", "ptq", "qat"}:
            raise ValueError(f"invalid source projection {self.source_projection!r}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evaluation_modes"] = list(self.evaluation_modes)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BaselineRow":
        expected = {field for field in cls.__dataclass_fields__}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("baseline row has an invalid schema")
        return cls(**{**value, "evaluation_modes": tuple(value["evaluation_modes"])})


def required_baseline_rows() -> tuple[BaselineRow, ...]:
    """The complete QI-1 matrix, using unambiguous physical-type names."""
    rows = [
        BaselineRow("dense_teacher", "HF_FP32_DENSE", "none", "transformers",
                    "not_applicable", ("w_only",), "dense"),
        BaselineRow("lossless_full_ternary", "LOSSLESS_FULL_TERNARY_INT2",
                    "lossless", "repository_reference", "ptq",
                    ACTIVATION_MODES, "unhealed"),
        BaselineRow("llamacpp_tq1_0", "GGML_TQ1_0_TERNARY", "block256_fp16",
                    "llama.cpp", "ptq", ("w_only",), "unhealed"),
        BaselineRow("llamacpp_tq2_0", "GGML_TQ2_0_TERNARY", "block256_fp16",
                    "llama.cpp", "ptq", ("w_only",), "unhealed"),
        BaselineRow("llamacpp_iq1_s", "GGML_IQ1_S_IMPORTANCE_QUANTIZED",
                    "block256_fp16", "llama.cpp", "ptq", ("w_only",), "unhealed"),
        BaselineRow("ternary_g128", "TERNARY_G128_INT2_FP16_SCALE",
                    "group128_fp16", "repository_reference", "ptq",
                    ACTIVATION_MODES, "unhealed"),
        BaselineRow("binary_g128", "BONSAI_Q1_0_BINARY_G128_FP16_SCALE",
                    "group128_fp16", "repository_reference", "ptq",
                    ACTIVATION_MODES, "unhealed"),
    ]
    for version in ("V11", "V12"):
        for scale, spelling in (("row_fp16", "ROW_FP16"),
                                ("block256_fp16", "BLOCK256_FP16")):
            rows.append(BaselineRow(
                f"tq1_{version.lower()}_ptq_{'row' if scale == 'row_fp16' else 'block256'}",
                f"TQ1_{version}_J_{spelling}", scale, "schema2_ptq", "ptq",
                ACTIVATION_MODES, "ptq"))
        # Format-v1 QAT currently has trainable row scales.  Block-scale QAT is
        # intentionally absent rather than represented as a fake matrix row.
        for training in ("w_only", "w_a8"):
            rows.append(BaselineRow(
                f"tq1_{version.lower()}_qat_{training}",
                f"TQ1_{version}_J_ROW_FP16", "row_fp16", "schema2_qat",
                training, ACTIVATION_MODES, "qat"))
    return tuple(rows)


_IDENTITY_FIELDS = {
    "source_model", "source_revision", "source_license", "tokenizer",
    "tokenizer_revision", "tokenizer_sha256", "chat_template_sha256",
    "calibration_dataset", "calibration_revision", "calibration_sha256",
    "evaluation_dataset", "evaluation_revision", "evaluation_sha256",
    "runtime_configuration", "runtime_configuration_sha256",
    "repository_commit", "seeds", "tool_versions",
}


def validate_identity(identity: Mapping[str, Any]) -> None:
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
        raise ValueError(f"baseline identity must contain exactly {sorted(_IDENTITY_FIELDS)}")
    for name in ("tokenizer_sha256", "chat_template_sha256", "calibration_sha256",
                 "evaluation_sha256", "runtime_configuration_sha256"):
        if not _is_hash(identity[name]):
            raise ValueError(f"baseline identity {name} must be a SHA-256")
    for name in _IDENTITY_FIELDS - {
            "seeds", "tool_versions", "runtime_configuration", "tokenizer_sha256",
            "chat_template_sha256", "calibration_sha256", "evaluation_sha256",
            "runtime_configuration_sha256"}:
        if not isinstance(identity[name], str) or not identity[name]:
            raise ValueError(f"baseline identity {name} must be a nonempty string")
    if not isinstance(identity["seeds"], Mapping) or not identity["seeds"]:
        raise ValueError("baseline identity seeds must be a nonempty object")
    if not all(isinstance(value, int) and not isinstance(value, bool)
               for value in identity["seeds"].values()):
        raise ValueError("baseline seeds must be integers")
    if not isinstance(identity["tool_versions"], Mapping) or not identity["tool_versions"]:
        raise ValueError("baseline tool_versions must be a nonempty object")
    if not isinstance(identity["runtime_configuration"], Mapping) \
            or not identity["runtime_configuration"]:
        raise ValueError("runtime_configuration must be a nonempty object")
    if _sha256(identity["runtime_configuration"]) != identity["runtime_configuration_sha256"]:
        raise ValueError("runtime configuration hash mismatch")


_STORAGE_FIELDS = {
    "unique_logical_parameters", "low_bit_unique_parameters",
    "high_precision_unique_parameters", "packed_weight_bytes",
    "canonical_artifact_bytes", "final_gguf_bytes", "backend_repack_bytes",
    "resident_language_model_bytes", "model_effective_bpw",
}


def _validate_storage(value: Mapping[str, Any], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _STORAGE_FIELDS:
        raise ValueError(f"{name}.storage has an invalid schema")
    for field in _STORAGE_FIELDS - {"model_effective_bpw"}:
        item = value[field]
        if item is None and field in {"canonical_artifact_bytes", "final_gguf_bytes",
                                     "backend_repack_bytes"}:
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{name}.storage.{field} must be a nonnegative integer or null")
    _finite(value["model_effective_bpw"], f"{name}.storage.model_effective_bpw", minimum=0)
    unique = value["unique_logical_parameters"]
    if unique <= 0 or value["low_bit_unique_parameters"] \
            + value["high_precision_unique_parameters"] != unique:
        raise ValueError(f"{name}.storage parameter accounting does not reconcile")


def _validate_timing(value: Mapping[str, Any], name: str) -> None:
    required = {"samples", "warmups", "iterations", "median_ms", "p20_ms", "p80_ms"}
    if not isinstance(value, Mapping) or not required <= set(value):
        raise ValueError(f"{name} timing has an invalid schema")
    for field in ("samples", "warmups", "iterations"):
        if isinstance(value[field], bool) or not isinstance(value[field], int) \
                or value[field] < (0 if field == "warmups" else 1):
            raise ValueError(f"{name}.{field} is invalid")
    values = [_finite(value[field], f"{name}.{field}", minimum=0)
              for field in ("p20_ms", "median_ms", "p80_ms")]
    if not values[0] <= values[1] <= values[2]:
        raise ValueError(f"{name} timing quantiles are not monotonic")


def validate_row_result(result: Mapping[str, Any], row: BaselineRow, *,
                        identity_sha256: str, gate_sha256: str | None) -> None:
    required = {
        "schema", "row_id", "row_definition_sha256", "identity_sha256",
        "gate_sha256", "artifact_identity", "storage", "quality_by_activation",
        "task_results", "export_parity", "performance", "commands", "provenance",
    }
    if not isinstance(result, Mapping) or set(result) != required:
        raise ValueError(f"baseline result {row.id} has an invalid top-level schema")
    if result["schema"] != 1 or result["row_id"] != row.id:
        raise ValueError(f"baseline result {row.id} identity mismatch")
    if result["row_definition_sha256"] != _sha256(row.to_dict()):
        raise ValueError(f"baseline result {row.id} row-definition mismatch")
    if result["identity_sha256"] != identity_sha256:
        raise ValueError(f"baseline result {row.id} experiment identity mismatch")
    if result["gate_sha256"] != gate_sha256:
        raise ValueError(f"baseline result {row.id} gate binding mismatch")
    if not isinstance(result["artifact_identity"], Mapping) \
            or not result["artifact_identity"]:
        raise ValueError(f"baseline result {row.id} lacks artifact identity")
    _validate_storage(result["storage"], row.id)
    quality = result["quality_by_activation"]
    if not isinstance(quality, Mapping) or set(quality) != set(row.evaluation_modes):
        raise ValueError(f"baseline result {row.id} activation-mode coverage is incomplete")
    for mode, metrics in quality.items():
        validate_metric_set(metrics, name=f"{row.id}.quality_by_activation.{mode}")
    if not isinstance(result["task_results"], Mapping) or not result["task_results"]:
        raise ValueError(f"baseline result {row.id} has no task results")
    parity = result["export_parity"]
    if not isinstance(parity, Mapping) or set(parity) != {"status", "max_abs", "max_rel"}:
        raise ValueError(f"baseline result {row.id} export parity schema is invalid")
    if parity["status"] not in {"not_applicable", "pass", "fail"}:
        raise ValueError(f"baseline result {row.id} export parity status is invalid")
    for field in ("max_abs", "max_rel"):
        if parity[field] is not None:
            _finite(parity[field], f"{row.id}.export_parity.{field}", minimum=0)
    performance = result["performance"]
    if not isinstance(performance, Mapping) or set(performance) != {
            "decode", "prefill", "peak_memory_bytes_by_context"}:
        raise ValueError(f"baseline result {row.id} performance schema is invalid")
    for section in ("decode", "prefill"):
        if not isinstance(performance[section], Mapping) or not performance[section]:
            raise ValueError(f"baseline result {row.id} lacks {section} timings")
        for label, timing in performance[section].items():
            _validate_timing(timing, f"{row.id}.performance.{section}.{label}")
    peaks = performance["peak_memory_bytes_by_context"]
    if not isinstance(peaks, Mapping) or not peaks:
        raise ValueError(f"baseline result {row.id} lacks peak-memory measurements")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in peaks.values()):
        raise ValueError(f"baseline result {row.id} peak-memory values are invalid")
    if not isinstance(result["commands"], Sequence) or isinstance(result["commands"], str) \
            or not result["commands"] or not all(isinstance(item, str) and item
                                                  for item in result["commands"]):
        raise ValueError(f"baseline result {row.id} commands are invalid")
    if not isinstance(result["provenance"], Mapping) or not result["provenance"]:
        raise ValueError(f"baseline result {row.id} provenance is missing")


_GATE_FIELDS = {
    "declared_after_dense_sha256", "maximum_perplexity_ratio",
    "maximum_teacher_kl_p99", "minimum_top_token_agreement",
    "minimum_aggregate_task_retention", "minimum_capability_retention",
    "maximum_decode_latency_ratio", "maximum_prefill_latency_ratio",
    "maximum_model_effective_bpw", "required_export_parity_status",
}


def validate_gates(gates: Mapping[str, Any], dense_sha256: str) -> None:
    if not isinstance(gates, Mapping) or set(gates) != _GATE_FIELDS:
        raise ValueError(f"baseline gates must contain exactly {sorted(_GATE_FIELDS)}")
    if gates["declared_after_dense_sha256"] != dense_sha256:
        raise ValueError("baseline gates are not bound to the recorded dense result")
    for name in _GATE_FIELDS - {"declared_after_dense_sha256",
                                "required_export_parity_status"}:
        minimum = 0
        value = _finite(gates[name], f"gates.{name}", minimum=minimum)
        if name.startswith("minimum_") and value > 1:
            raise ValueError(f"gates.{name} must be in [0,1]")
    if gates["maximum_perplexity_ratio"] < 1:
        raise ValueError("maximum_perplexity_ratio cannot be below one")
    if gates["required_export_parity_status"] not in {"pass", "not_applicable"}:
        raise ValueError("required_export_parity_status must be pass or not_applicable")


def _task_score(value: Any, name: str) -> float:
    if isinstance(value, Mapping):
        if "score" not in value:
            raise ValueError(f"{name} task result has no score")
        value = value["score"]
    return _finite(value, name, minimum=0)


def evaluate_candidate_gates(result: Mapping[str, Any], dense: Mapping[str, Any],
                             gates: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one row without hiding mode- or capability-specific failures."""
    dense_quality = dense["quality_by_activation"]["w_only"]
    checks: dict[str, bool] = {}
    for mode, metrics in result["quality_by_activation"].items():
        prefix = f"{mode}."
        checks[prefix + "perplexity"] = (
            metrics["perplexity"] / dense_quality["perplexity"]
            <= gates["maximum_perplexity_ratio"])
        checks[prefix + "kl_p99"] = (
            metrics["teacher_kl_p99"] <= gates["maximum_teacher_kl_p99"])
        checks[prefix + "top_token"] = (
            metrics["top_token_agreement"] >= gates["minimum_top_token_agreement"])
    dense_tasks = dense["task_results"]
    candidate_tasks = result["task_results"]
    if set(dense_tasks) != set(candidate_tasks) or not dense_tasks:
        raise ValueError("candidate and dense task inventories must match exactly")
    shared = sorted(dense_tasks)
    retentions = {}
    for task in shared:
        candidate_score = _task_score(candidate_tasks[task], f"candidate.{task}")
        dense_score = _task_score(dense_tasks[task], f"dense.{task}")
        retentions[task] = (1.0 if candidate_score == dense_score == 0
                            else candidate_score / max(dense_score, 1e-30))
    checks["tasks.aggregate"] = (sum(retentions.values()) / len(retentions)
                                 >= gates["minimum_aggregate_task_retention"])
    checks["tasks.capability_floor"] = (
        min(retentions.values()) >= gates["minimum_capability_retention"])
    checks["storage.model_bpw"] = (
        result["storage"]["model_effective_bpw"] <= gates["maximum_model_effective_bpw"])
    checks["export.parity"] = (
        result["export_parity"]["status"] == gates["required_export_parity_status"])
    for section in ("decode", "prefill"):
        dense_timings = dense["performance"][section]
        candidate_timings = result["performance"][section]
        if set(dense_timings) != set(candidate_timings):
            raise ValueError(f"candidate and dense {section} timing inventories differ")
        maximum_ratio = max(
            candidate_timings[name]["median_ms"]
            / max(dense_timings[name]["median_ms"], 1e-30)
            for name in dense_timings)
        checks[f"performance.{section}"] = maximum_ratio <= gates[
            f"maximum_{section}_latency_ratio"]
    return {"passed": all(checks.values()), "checks": checks,
            "task_retention": retentions}


class BaselineMatrix:
    """Mutable builder for an immutable, hash-bound experiment record."""

    def __init__(self, document: Mapping[str, Any]):
        self.document = copy.deepcopy(dict(document))
        self.validate()

    @classmethod
    def create(cls, identity: Mapping[str, Any], *,
               rows: Sequence[BaselineRow] | None = None) -> "BaselineMatrix":
        validate_identity(identity)
        definitions = tuple(rows or required_baseline_rows())
        if len({row.id for row in definitions}) != len(definitions):
            raise ValueError("baseline row ids must be unique")
        if not definitions or definitions[0].id != "dense_teacher":
            raise ValueError("the first baseline row must be dense_teacher")
        now = dt.datetime.now(dt.UTC).isoformat()
        return cls({
            "schema": BASELINE_MATRIX_SCHEMA,
            "created_at": now,
            "updated_at": now,
            "identity": dict(identity),
            "identity_sha256": _sha256(identity),
            "row_definitions": [row.to_dict() for row in definitions],
            "row_definitions_sha256": _sha256([row.to_dict() for row in definitions]),
            "dense_result": None,
            "dense_result_sha256": None,
            "gates": None,
            "gate_sha256": None,
            "candidate_results": {},
            "candidate_result_sha256": {},
            "verdicts": {},
            "status": "awaiting_dense",
        })

    @property
    def rows(self) -> dict[str, BaselineRow]:
        return {row.id: row for row in map(BaselineRow.from_dict,
                                           self.document["row_definitions"])}

    def _touch(self) -> None:
        self.document["updated_at"] = dt.datetime.now(dt.UTC).isoformat()

    def record_result(self, result: Mapping[str, Any]) -> None:
        row_id = result.get("row_id") if isinstance(result, Mapping) else None
        if row_id not in self.rows:
            raise ValueError(f"unknown baseline row {row_id!r}")
        row = self.rows[row_id]
        if row.id == "dense_teacher":
            if self.document["dense_result"] is not None or self.document["gates"] is not None:
                raise ValueError("dense result is immutable once recorded")
            validate_row_result(result, row,
                                identity_sha256=self.document["identity_sha256"],
                                gate_sha256=None)
            self.document["dense_result"] = copy.deepcopy(dict(result))
            self.document["dense_result_sha256"] = _sha256(result)
            self.document["status"] = "awaiting_gates"
        else:
            if self.document["gates"] is None:
                raise ValueError("candidate results cannot be opened before gates are declared")
            if row_id in self.document["candidate_results"]:
                raise ValueError(f"candidate result {row_id} is immutable once recorded")
            validate_row_result(result, row,
                                identity_sha256=self.document["identity_sha256"],
                                gate_sha256=self.document["gate_sha256"])
            self.document["candidate_results"][row_id] = copy.deepcopy(dict(result))
            self.document["candidate_result_sha256"][row_id] = _sha256(result)
            self.document["verdicts"][row_id] = evaluate_candidate_gates(
                result, self.document["dense_result"], self.document["gates"])
            required = {item.id for item in self.rows.values()
                        if item.required and item.id != "dense_teacher"}
            present = set(self.document["candidate_results"])
            self.document["status"] = ("complete" if present == required
                                        else "collecting_candidates")
        self._touch()
        self.validate()

    def declare_gates(self, gates: Mapping[str, Any]) -> None:
        if self.document["dense_result"] is None:
            raise ValueError("dense result must be recorded before gates are declared")
        if self.document["gates"] is not None or self.document["candidate_results"]:
            raise ValueError("baseline gates are immutable once declared")
        validate_gates(gates, self.document["dense_result_sha256"])
        self.document["gates"] = copy.deepcopy(dict(gates))
        self.document["gate_sha256"] = _sha256(gates)
        self.document["status"] = "collecting_candidates"
        self._touch()
        self.validate()

    def validate(self) -> None:
        required = {
            "schema", "created_at", "updated_at", "identity", "identity_sha256",
            "row_definitions", "row_definitions_sha256", "dense_result",
            "dense_result_sha256", "gates", "gate_sha256", "candidate_results",
            "candidate_result_sha256", "verdicts", "status",
        }
        if set(self.document) != required or self.document["schema"] != BASELINE_MATRIX_SCHEMA:
            raise ValueError("baseline matrix has an invalid schema")
        validate_identity(self.document["identity"])
        if self.document["identity_sha256"] != _sha256(self.document["identity"]):
            raise ValueError("baseline experiment identity hash mismatch")
        rows = list(map(BaselineRow.from_dict, self.document["row_definitions"]))
        if not rows or len({row.id for row in rows}) != len(rows):
            raise ValueError("baseline row definitions are empty or duplicated")
        if self.document["row_definitions_sha256"] != _sha256(
                self.document["row_definitions"]):
            raise ValueError("baseline row-definition hash mismatch")
        dense = self.document["dense_result"]
        if dense is None:
            if any(self.document[field] is not None for field in (
                    "dense_result_sha256", "gates", "gate_sha256")) \
                    or self.document["candidate_results"]:
                raise ValueError("baseline matrix contains evidence before its dense result")
        else:
            validate_row_result(dense, self.rows["dense_teacher"],
                                identity_sha256=self.document["identity_sha256"],
                                gate_sha256=None)
            if self.document["dense_result_sha256"] != _sha256(dense):
                raise ValueError("dense baseline hash mismatch")
        gates = self.document["gates"]
        if gates is None:
            if self.document["gate_sha256"] is not None \
                    or self.document["candidate_results"]:
                raise ValueError("candidate evidence exists without frozen gates")
        else:
            validate_gates(gates, self.document["dense_result_sha256"])
            if self.document["gate_sha256"] != _sha256(gates):
                raise ValueError("baseline gate hash mismatch")
        results = self.document["candidate_results"]
        hashes = self.document["candidate_result_sha256"]
        verdicts = self.document["verdicts"]
        if not isinstance(results, Mapping) or set(results) != set(hashes) \
                or set(results) != set(verdicts):
            raise ValueError("candidate results, hashes, and verdicts differ")
        for row_id, result in results.items():
            if row_id == "dense_teacher" or row_id not in self.rows:
                raise ValueError(f"invalid candidate row {row_id!r}")
            validate_row_result(result, self.rows[row_id],
                                identity_sha256=self.document["identity_sha256"],
                                gate_sha256=self.document["gate_sha256"])
            if hashes[row_id] != _sha256(result):
                raise ValueError(f"candidate result hash mismatch for {row_id}")
            if verdicts[row_id] != evaluate_candidate_gates(result, dense, gates):
                raise ValueError(f"candidate verdict mismatch for {row_id}")
        legal_status = {"awaiting_dense", "awaiting_gates", "collecting_candidates",
                        "complete"}
        if self.document["status"] not in legal_status:
            raise ValueError("baseline matrix status is invalid")
        required_rows = {row.id for row in rows if row.required and row.id != "dense_teacher"}
        expected_status = ("awaiting_dense" if dense is None else
                           "awaiting_gates" if gates is None else
                           "complete" if set(results) == required_rows else
                           "collecting_candidates")
        if self.document["status"] != expected_status:
            raise ValueError("baseline matrix status does not match its evidence")

    @classmethod
    def load(cls, path: str | Path) -> "BaselineMatrix":
        return cls(json.loads(Path(path).read_text()))

    def write(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.document, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return path


def result_template(matrix: BaselineMatrix, row_id: str) -> dict[str, Any]:
    """Return a schema-complete skeleton without fabricating a measurement."""
    if row_id not in matrix.rows:
        raise KeyError(row_id)
    row = matrix.rows[row_id]
    gate_hash = None if row_id == "dense_teacher" else matrix.document["gate_sha256"]
    if row_id != "dense_teacher" and gate_hash is None:
        raise ValueError("candidate template cannot be opened before gates are declared")
    return {
        "schema": 1,
        "row_id": row_id,
        "row_definition_sha256": _sha256(row.to_dict()),
        "identity_sha256": matrix.document["identity_sha256"],
        "gate_sha256": gate_hash,
        "artifact_identity": {},
        "storage": {field: None for field in sorted(_STORAGE_FIELDS)},
        "quality_by_activation": {mode: {} for mode in row.evaluation_modes},
        "task_results": {},
        "export_parity": {"status": "not_applicable", "max_abs": None,
                          "max_rel": None},
        "performance": {"decode": {}, "prefill": {},
                        "peak_memory_bytes_by_context": {}},
        "commands": [],
        "provenance": {},
    }
