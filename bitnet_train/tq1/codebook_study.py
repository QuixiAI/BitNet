"""Universal-codebook measurement and fail-closed acceptance evidence.

The expensive checkpoint evaluation and kernel measurements remain external to
this module.  Representation metrics are computed here from the exact codebook
and a :class:`~bitnet_train.tq1.solver.PatternCorpus`; the final report validator
then binds those metrics to disjoint construction/held-out model identities,
PTQ/QAT quality comparisons, predeclared gates, and decode/prefill evidence.
Synthetic fixtures can prove this contract, but cannot constitute a completed
universal-codebook study.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .calibration import file_sha256
from .codebook import Codebook
from .evaluation import canonical_document_sha256, validate_metric_set
from .ptq import ternary_universe
from .solver import PatternCorpus


PATTERN_CORPUS_SCHEMA = 1
UNIVERSAL_CODEBOOK_STUDY_SCHEMA = 1
PROJECTION_FAMILIES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
QUALITY_PHASES = ("ptq", "qat")
QUALITY_ACTIVATION_MODES = ("w_only", "w_a8")
KERNEL_WORKLOADS = ("decode", "prefill")

_HASH_FIELDS = {
    "pattern_corpus_sha256", "pattern_corpus_metadata_sha256",
    "solver_config_sha256", "split_manifest_sha256",
}
_DISTORTION_FIELDS = {
    "sensitivity_metric", "pattern_mass", "exact_hit_mass", "exact_hit_rate",
    "frequency_weighted_distortion_sum", "frequency_weighted_distortion_mean",
    "sensitivity_weighted_distortion_sum", "sensitivity_weighted_distortion_mean",
}
_GATE_FIELDS = {
    "declared_before_evaluation", "study_definition_sha256",
    "minimum_aggregate_exact_hit_rate", "minimum_model_exact_hit_rate",
    "minimum_family_exact_hit_rate", "maximum_universe_squared_trit_distance",
    "maximum_aggregate_frequency_weighted_distortion",
    "maximum_model_frequency_weighted_distortion",
    "maximum_family_frequency_weighted_distortion",
    "maximum_aggregate_sensitivity_weighted_distortion",
    "maximum_model_sensitivity_weighted_distortion",
    "maximum_family_sensitivity_weighted_distortion",
    "maximum_quality_cross_entropy_increase",
    "maximum_quality_teacher_kl_mean_increase",
    "maximum_quality_teacher_kl_p99_increase",
    "maximum_quality_top_token_agreement_drop",
    "maximum_quality_task_score_drop", "maximum_decode_latency_ratio",
    "maximum_prefill_latency_ratio",
}


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


def _immutable_revision(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) not in {40, 64} \
            or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a full immutable 40/64-hex revision")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) \
            or (minimum is not None and result < minimum) \
            or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its legal range")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def _validate_model_identity(value: Mapping[str, Any], name: str) -> None:
    fields = {"model_id", "revision", "family"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has an invalid model identity schema")
    for field in ("model_id", "family"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{name}.{field} must be nonempty")
    _immutable_revision(value["revision"], f"{name}.revision")


def _identity_key(value: Mapping[str, Any]) -> tuple[str, str]:
    return str(value["model_id"]), str(value["revision"])


def _identity_label(value: Mapping[str, Any]) -> str:
    return f"{value['model_id']}@{str(value['revision'])[:12]}"


def save_pattern_corpus(path: str | Path, corpus: PatternCorpus, *,
                        metadata: Mapping[str, Any]) -> Path:
    """Write one hashable sensitivity corpus without pickle execution."""
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError("pattern-corpus metadata must be a nonempty object")
    if corpus.diagonal is None and corpus.covariance is None:
        raise ValueError("a study corpus requires diagonal or covariance sensitivity")
    tensors = {"demand": corpus.demand}
    if corpus.diagonal is not None:
        tensors["diagonal"] = corpus.diagonal
    elif corpus.covariance is not None:
        tensors["covariance"] = corpus.covariance
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination), metadata={
        "tq1_pattern_corpus_schema": str(PATTERN_CORPUS_SCHEMA),
        "metadata_json": json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False),
    })
    return destination


def load_pattern_corpus(path: str | Path) -> tuple[PatternCorpus, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    tensors = load_file(str(source), device="cpu")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("tq1_pattern_corpus_schema") != str(PATTERN_CORPUS_SCHEMA):
        raise ValueError("unsupported TQ1 pattern-corpus schema")
    if set(tensors) not in ({"demand", "diagonal"}, {"demand", "covariance"}):
        raise ValueError("a study corpus requires demand and exactly one sensitivity tensor")
    try:
        decoded = json.loads(metadata["metadata_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("pattern corpus has invalid metadata JSON") from exc
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError("pattern-corpus metadata must be a nonempty object")
    return PatternCorpus(
        tensors["demand"], diagonal=tensors.get("diagonal"),
        covariance=tensors.get("covariance")), decoded


def _legal_codewords(codebook: Codebook) -> torch.Tensor:
    physical = torch.arange(codebook.index_count, dtype=torch.int64)
    legal = codebook.legal_index_mask()
    codewords = codebook.decode(physical[legal]).to(torch.int8).contiguous()
    if torch.unique(codewords, dim=0).shape[0] != codewords.shape[0]:
        raise ValueError("legal codebook indices do not decode to unique codewords")
    return codewords


def codebook_coverage(codebook: Codebook, *, chunk_size: int = 128) -> dict[str, Any]:
    """Return exact squared-trit coverage over all 6,561 source patterns."""
    if chunk_size < 1:
        raise ValueError("coverage chunk size must be positive")
    patterns = ternary_universe().to(torch.int16)
    codewords = _legal_codewords(codebook).to(torch.int16)
    minimum = torch.full((patterns.shape[0],), 32767, dtype=torch.int16)
    for start in range(0, codewords.shape[0], chunk_size):
        candidate = codewords[start:start + chunk_size]
        distance = (patterns[:, None] - candidate[None]).square().sum(-1)
        minimum = torch.minimum(minimum, distance.min(1).values.to(torch.int16))
    values, counts = torch.unique(minimum, sorted=True, return_counts=True)
    return {
        "pattern_count": int(patterns.shape[0]),
        "legal_index_count": int(codewords.shape[0]),
        "unique_codeword_count": int(torch.unique(codewords, dim=0).shape[0]),
        "max_squared_trit_distance": int(minimum.max()),
        "squared_trit_distance_histogram": {
            str(int(value)): int(count) for value, count in zip(values, counts)
        },
    }


def distortion_metrics(codebook: Codebook, corpus: PatternCorpus, *,
                       chunk_size: int = 128) -> dict[str, Any]:
    """Measure frequency and sensitivity objectives against legal codewords.

    Frequency distortion is minimum squared Euclidean distance in trit units.
    Sensitivity distortion independently minimizes the corpus' diagonal or
    covariance quadratic objective.  Means are normalized by corpus demand
    mass so corpora with different weighting policies remain comparable.
    """
    if chunk_size < 1:
        raise ValueError("distortion chunk size must be positive")
    if corpus.diagonal is None and corpus.covariance is None:
        raise ValueError("held-out distortion requires diagonal or covariance sensitivity")
    if corpus.covariance is not None:
        asymmetry = float((corpus.covariance
                           - corpus.covariance.transpose(-1, -2)).abs().max())
        if asymmetry > 1e-8:
            raise ValueError(
                f"held-out covariance is not symmetric (max asymmetry {asymmetry})")
        minimum_eigenvalue = float(torch.linalg.eigvalsh(corpus.covariance).min())
        if minimum_eigenvalue < -1e-7:
            raise ValueError(
                "held-out covariance is not positive semidefinite "
                f"(minimum eigenvalue {minimum_eigenvalue})")
    patterns = ternary_universe().double()
    codewords = _legal_codewords(codebook).double()
    frequency_minimum = torch.full((patterns.shape[0],), torch.inf, dtype=torch.float64)
    sensitivity_minimum = torch.full_like(frequency_minimum, torch.inf)
    for start in range(0, codewords.shape[0], chunk_size):
        candidate = codewords[start:start + chunk_size]
        delta = patterns[:, None] - candidate[None]
        frequency = delta.square().sum(-1)
        if corpus.diagonal is not None:
            sensitivity = (delta.square() * corpus.diagonal[:, None]).sum(-1)
        else:
            assert corpus.covariance is not None
            sensitivity = torch.einsum(
                "uci,uij,ucj->uc", delta, corpus.covariance, delta)
        frequency_minimum = torch.minimum(frequency_minimum, frequency.min(1).values)
        sensitivity_minimum = torch.minimum(
            sensitivity_minimum, sensitivity.min(1).values)
    smallest = float(sensitivity_minimum.min())
    if smallest < -1e-9:
        raise ValueError(
            f"sensitivity corpus yields a negative quadratic distance ({smallest})")
    sensitivity_minimum.clamp_min_(0)
    demand = corpus.demand
    mass = float(demand.sum())
    exact_mass = float(demand[frequency_minimum == 0].sum())
    frequency_sum = float((demand * frequency_minimum).sum())
    sensitivity_sum = float((demand * sensitivity_minimum).sum())
    return {
        "sensitivity_metric": (
            "diagonal" if corpus.diagonal is not None else "covariance8"),
        "pattern_mass": mass,
        "exact_hit_mass": exact_mass,
        "exact_hit_rate": exact_mass / mass,
        "frequency_weighted_distortion_sum": frequency_sum,
        "frequency_weighted_distortion_mean": frequency_sum / mass,
        "sensitivity_weighted_distortion_sum": sensitivity_sum,
        "sensitivity_weighted_distortion_mean": sensitivity_sum / mass,
    }


def _validate_distortion_metrics(value: Mapping[str, Any], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _DISTORTION_FIELDS:
        raise ValueError(f"{name} has an invalid distortion schema")
    if value["sensitivity_metric"] not in {"diagonal", "covariance8"}:
        raise ValueError(f"{name}.sensitivity_metric is invalid")
    mass = _finite(value["pattern_mass"], f"{name}.pattern_mass", minimum=0)
    if mass <= 0:
        raise ValueError(f"{name}.pattern_mass must be positive")
    exact_mass = _finite(
        value["exact_hit_mass"], f"{name}.exact_hit_mass", minimum=0,
        maximum=mass)
    rate = _finite(value["exact_hit_rate"], f"{name}.exact_hit_rate",
                   minimum=0, maximum=1)
    if not _close(rate, exact_mass / mass):
        raise ValueError(f"{name}.exact_hit_rate does not reconcile")
    for prefix in ("frequency", "sensitivity"):
        total = _finite(
            value[f"{prefix}_weighted_distortion_sum"],
            f"{name}.{prefix}_weighted_distortion_sum", minimum=0)
        mean = _finite(
            value[f"{prefix}_weighted_distortion_mean"],
            f"{name}.{prefix}_weighted_distortion_mean", minimum=0)
        if not _close(mean, total / mass):
            raise ValueError(f"{name}.{prefix} distortion mean does not reconcile")


def combine_distortion_metrics(values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(values)
    if not items:
        raise ValueError("cannot aggregate an empty distortion inventory")
    for index, value in enumerate(items):
        _validate_distortion_metrics(value, f"distortion[{index}]")
    metrics = {str(value["sensitivity_metric"]) for value in items}
    if len(metrics) != 1:
        raise ValueError("distortion corpora use different sensitivity metrics")
    mass = sum(float(value["pattern_mass"]) for value in items)
    exact = sum(float(value["exact_hit_mass"]) for value in items)
    frequency = sum(float(value["frequency_weighted_distortion_sum"])
                    for value in items)
    sensitivity = sum(float(value["sensitivity_weighted_distortion_sum"])
                      for value in items)
    return {
        "sensitivity_metric": metrics.pop(),
        "pattern_mass": mass,
        "exact_hit_mass": exact,
        "exact_hit_rate": exact / mass,
        "frequency_weighted_distortion_sum": frequency,
        "frequency_weighted_distortion_mean": frequency / mass,
        "sensitivity_weighted_distortion_sum": sensitivity,
        "sensitivity_weighted_distortion_mean": sensitivity / mass,
    }


def _validate_coverage(value: Mapping[str, Any], codebook: Codebook) -> None:
    fields = {
        "pattern_count", "legal_index_count", "unique_codeword_count",
        "max_squared_trit_distance", "squared_trit_distance_histogram",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("universe coverage has an invalid schema")
    for field in fields - {"squared_trit_distance_histogram"}:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"universe_coverage.{field} must be a nonnegative integer")
    histogram = value["squared_trit_distance_histogram"]
    if not isinstance(histogram, Mapping) or not histogram:
        raise ValueError("universe coverage histogram is empty")
    if any(not isinstance(key, str) or not key.isdigit()
           or isinstance(count, bool) or not isinstance(count, int) or count < 1
           for key, count in histogram.items()):
        raise ValueError("universe coverage histogram is invalid")
    if sum(histogram.values()) != 6561 or value["pattern_count"] != 6561:
        raise ValueError("universe coverage does not contain all 6,561 patterns")
    if max(map(int, histogram)) != value["max_squared_trit_distance"]:
        raise ValueError("universe coverage maximum disagrees with its histogram")
    expected = codebook_coverage(codebook)
    if dict(value) != expected:
        raise ValueError("reported universe coverage differs from the exact codebook")


def _validate_quality_row(value: Mapping[str, Any], name: str,
                          *, codebook_sha256: str) -> None:
    fields = {
        "model", "phase", "activation_mode", "reference_artifact_sha256",
        "reference_codebook_sha256", "reference_codebook_scope",
        "candidate_artifact_sha256",
        "candidate_codebook_sha256", "evaluation_data_sha256",
        "reference_metrics", "candidate_metrics", "reference_tasks",
        "candidate_tasks", "deltas",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has an invalid quality-comparison schema")
    _validate_model_identity(value["model"], f"{name}.model")
    if value["phase"] not in QUALITY_PHASES:
        raise ValueError(f"{name}.phase is invalid")
    if value["activation_mode"] not in QUALITY_ACTIVATION_MODES:
        raise ValueError(f"{name}.activation_mode is invalid")
    if value["reference_codebook_scope"] != "model":
        raise ValueError(f"{name} reference must use a model-scoped codebook")
    for field in (
            "reference_artifact_sha256", "reference_codebook_sha256",
            "candidate_artifact_sha256", "candidate_codebook_sha256",
            "evaluation_data_sha256"):
        _sha256(value[field], f"{name}.{field}")
    if value["candidate_codebook_sha256"] != codebook_sha256:
        raise ValueError(f"{name} is not bound to the universal codebook")
    validate_metric_set(value["reference_metrics"], name=f"{name}.reference_metrics")
    validate_metric_set(value["candidate_metrics"], name=f"{name}.candidate_metrics")
    if value["reference_metrics"]["token_count"] \
            != value["candidate_metrics"]["token_count"]:
        raise ValueError(f"{name} reference/candidate token counts differ")
    reference_tasks, candidate_tasks = value["reference_tasks"], value["candidate_tasks"]
    if not isinstance(reference_tasks, Mapping) or not reference_tasks \
            or not isinstance(candidate_tasks, Mapping) \
            or set(reference_tasks) != set(candidate_tasks):
        raise ValueError(f"{name} task inventories are empty or different")
    for task in reference_tasks:
        _finite(reference_tasks[task], f"{name}.reference_tasks.{task}",
                minimum=0, maximum=1)
        _finite(candidate_tasks[task], f"{name}.candidate_tasks.{task}",
                minimum=0, maximum=1)
    delta_fields = {
        "cross_entropy", "perplexity", "teacher_kl_mean", "teacher_kl_p99",
        "top_token_agreement", "aggregate_task_score",
    }
    if not isinstance(value["deltas"], Mapping) or set(value["deltas"]) != delta_fields:
        raise ValueError(f"{name}.deltas has an invalid schema")
    expected = {
        field: (float(value["candidate_metrics"][field])
                - float(value["reference_metrics"][field]))
        for field in delta_fields - {"aggregate_task_score"}
    }
    expected["aggregate_task_score"] = (
        sum(float(candidate_tasks[task]) for task in candidate_tasks) / len(candidate_tasks)
        - sum(float(reference_tasks[task]) for task in reference_tasks) / len(reference_tasks))
    for field, result in expected.items():
        reported = _finite(value["deltas"][field], f"{name}.deltas.{field}")
        if not _close(reported, result):
            raise ValueError(f"{name}.deltas.{field} does not reconcile")


def _timing(value: Mapping[str, Any], name: str) -> None:
    fields = {"p20_ms", "median_ms", "p80_ms"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has an invalid timing schema")
    numbers = [_finite(value[field], f"{name}.{field}", minimum=0)
               for field in ("p20_ms", "median_ms", "p80_ms")]
    if numbers[0] <= 0 or not numbers[0] <= numbers[1] <= numbers[2]:
        raise ValueError(f"{name} timing quantiles are invalid")


def _validate_kernel_row(value: Mapping[str, Any], name: str, *,
                         codebook_sha256: str) -> None:
    fields = {
        "id", "backend", "workload", "shape", "activation_mode", "output_dtype",
        "reference_representation", "candidate_representation", "warmups",
        "iterations", "reference_timing", "candidate_timing", "correctness",
        "device", "toolchain", "command", "run_sha256", "codebook_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has an invalid kernel-performance schema")
    for field in (
            "id", "backend", "activation_mode", "output_dtype",
            "reference_representation", "candidate_representation", "command"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{name}.{field} must be nonempty")
    if value["workload"] not in KERNEL_WORKLOADS:
        raise ValueError(f"{name}.workload is invalid")
    shape = value["shape"]
    if not isinstance(shape, list) or len(shape) != 3 \
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1
                   for item in shape) or shape[2] % 256:
        raise ValueError(f"{name}.shape must be positive [M,N,K] with K divisible by 256")
    for field in ("warmups", "iterations"):
        minimum = 0 if field == "warmups" else 1
        if isinstance(value[field], bool) or not isinstance(value[field], int) \
                or value[field] < minimum:
            raise ValueError(f"{name}.{field} is invalid")
    _timing(value["reference_timing"], f"{name}.reference_timing")
    _timing(value["candidate_timing"], f"{name}.candidate_timing")
    correctness = value["correctness"]
    if not isinstance(correctness, Mapping) or set(correctness) != {
            "atol", "rtol", "max_abs", "max_rel", "passed"}:
        raise ValueError(f"{name}.correctness has an invalid schema")
    for field in ("atol", "rtol", "max_abs", "max_rel"):
        _finite(correctness[field], f"{name}.correctness.{field}", minimum=0)
    if correctness["passed"] is not True:
        raise ValueError(f"{name} did not pass its declared oracle tolerance")
    for field in ("device", "toolchain"):
        if not isinstance(value[field], Mapping) or not value[field]:
            raise ValueError(f"{name}.{field} must be a nonempty object")
    _sha256(value["run_sha256"], f"{name}.run_sha256")
    _sha256(value["codebook_sha256"], f"{name}.codebook_sha256")
    if value["codebook_sha256"] != codebook_sha256:
        raise ValueError(f"{name} is not bound to the universal codebook")


def _definition_document(report: Mapping[str, Any]) -> dict[str, Any]:
    heldout_models = []
    for model in report["heldout"]["models"]:
        heldout_models.append({
            "identity": model["identity"],
            "pattern_corpus_sha256": model["pattern_corpus_sha256"],
            "families": {
                family: evidence["pattern_corpus_sha256"]
                for family, evidence in sorted(model["families"].items())
            },
        })
    quality = [{
        field: row[field] for field in (
            "model", "phase", "activation_mode", "reference_artifact_sha256",
            "reference_codebook_sha256", "reference_codebook_scope",
            "candidate_artifact_sha256",
            "candidate_codebook_sha256", "evaluation_data_sha256")
    } for row in report["quality_comparisons"]]
    kernels = [{
        field: row[field] for field in (
            "id", "backend", "workload", "shape", "activation_mode",
            "output_dtype", "reference_representation", "candidate_representation",
            "warmups", "iterations", "device", "toolchain", "codebook_sha256")
    } for row in report["kernel_performance"]]
    return {
        "schema": UNIVERSAL_CODEBOOK_STUDY_SCHEMA,
        "codebook": report["codebook"],
        "construction": report["construction"],
        "heldout": {
            "split_manifest_sha256": report["heldout"]["split_manifest_sha256"],
            "models": heldout_models,
        },
        "quality_comparisons": quality,
        "kernel_performance": kernels,
    }


def study_definition_sha256(report: Mapping[str, Any]) -> str:
    """Hash only identities and planned cases, never observed outcomes."""
    return canonical_document_sha256(_definition_document(report))


def _validate_gates(value: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != _GATE_FIELDS:
        raise ValueError(f"universal codebook gates must contain exactly {sorted(_GATE_FIELDS)}")
    if value["declared_before_evaluation"] is not True:
        raise ValueError("universal codebook gates were not predeclared")
    if value["study_definition_sha256"] != study_definition_sha256(report):
        raise ValueError("universal codebook gates do not bind this study definition")
    for field in _GATE_FIELDS - {"declared_before_evaluation", "study_definition_sha256"}:
        maximum = 1 if field.startswith("minimum_") else None
        _finite(value[field], f"predeclared_gates.{field}", minimum=0, maximum=maximum)


def _same_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(left[field] == right[field] if isinstance(left[field], str)
               else _close(float(left[field]), float(right[field]))
               for field in left)


def _validate_report_evidence(report: Mapping[str, Any], codebook: Codebook) \
        -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    fields = {
        "schema", "codebook", "construction", "heldout", "universe_coverage",
        "quality_comparisons", "kernel_performance", "predeclared_gates",
        "decision", "commands", "provenance",
    }
    if not isinstance(report, Mapping) or set(report) != fields \
            or report["schema"] != UNIVERSAL_CODEBOOK_STUDY_SCHEMA:
        raise ValueError("universal codebook study has an invalid top-level schema")
    if codebook.scope != "universal":
        raise ValueError("universal-codebook acceptance requires scope=universal")
    if report["codebook"] != codebook.ref().to_dict():
        raise ValueError("study codebook identity differs from the loaded codebook")
    construction = report["construction"]
    if not isinstance(construction, Mapping) or set(construction) != {
            "source_models", *_HASH_FIELDS}:
        raise ValueError("codebook construction identity has an invalid schema")
    source_models = construction["source_models"]
    if not isinstance(source_models, list) or not source_models:
        raise ValueError("codebook construction source model inventory is empty")
    for index, model in enumerate(source_models):
        _validate_model_identity(model, f"construction.source_models[{index}]")
    source_keys = [_identity_key(model) for model in source_models]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("codebook construction source models are duplicated")
    for field in _HASH_FIELDS:
        _sha256(construction[field], f"construction.{field}")
    if codebook.provenance.get("source") != "construction_pattern_corpus":
        raise ValueError("universal codebook provenance does not name its construction corpus")
    for field in (
            "pattern_corpus_sha256", "pattern_corpus_metadata_sha256",
            "solver_config_sha256", "split_manifest_sha256"):
        if codebook.provenance.get(field) != construction[field]:
            raise ValueError(f"universal codebook provenance differs at {field}")
    if codebook.provenance.get("source_models") != source_models:
        raise ValueError("universal codebook provenance differs at source_models")

    heldout = report["heldout"]
    if not isinstance(heldout, Mapping) or set(heldout) != {
            "split_manifest_sha256", "models", "aggregate"}:
        raise ValueError("held-out codebook evidence has an invalid schema")
    _sha256(heldout["split_manifest_sha256"], "heldout.split_manifest_sha256")
    models = heldout["models"]
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("universal codebook study requires at least two held-out models")
    model_metrics: list[Mapping[str, Any]] = []
    heldout_keys = []
    for index, model in enumerate(models):
        name = f"heldout.models[{index}]"
        if not isinstance(model, Mapping) or set(model) != {
                "identity", "pattern_corpus_sha256", "families", "metrics"}:
            raise ValueError(f"{name} has an invalid schema")
        _validate_model_identity(model["identity"], f"{name}.identity")
        heldout_keys.append(_identity_key(model["identity"]))
        _sha256(model["pattern_corpus_sha256"], f"{name}.pattern_corpus_sha256")
        families = model["families"]
        if not isinstance(families, Mapping) or set(families) != set(PROJECTION_FAMILIES):
            raise ValueError(f"{name} must report every primary projection family")
        family_metrics = []
        for family in PROJECTION_FAMILIES:
            evidence = families[family]
            family_name = f"{name}.families.{family}"
            if not isinstance(evidence, Mapping) or set(evidence) != {
                    "pattern_corpus_sha256", "metrics"}:
                raise ValueError(f"{family_name} has an invalid schema")
            _sha256(evidence["pattern_corpus_sha256"],
                    f"{family_name}.pattern_corpus_sha256")
            _validate_distortion_metrics(evidence["metrics"], f"{family_name}.metrics")
            family_metrics.append(evidence["metrics"])
        _validate_distortion_metrics(model["metrics"], f"{name}.metrics")
        expected = combine_distortion_metrics(family_metrics)
        if not _same_metrics(model["metrics"], expected):
            raise ValueError(f"{name}.metrics do not reconcile with its families")
        model_metrics.append(model["metrics"])
    if len(heldout_keys) != len(set(heldout_keys)):
        raise ValueError("held-out model identities are duplicated")
    overlap = set(source_keys) & set(heldout_keys)
    if overlap:
        raise ValueError(f"construction and held-out model checkpoints overlap: {sorted(overlap)}")
    _validate_distortion_metrics(heldout["aggregate"], "heldout.aggregate")
    if not _same_metrics(heldout["aggregate"], combine_distortion_metrics(model_metrics)):
        raise ValueError("held-out aggregate does not reconcile with per-model metrics")
    _validate_coverage(report["universe_coverage"], codebook)

    quality = report["quality_comparisons"]
    if not isinstance(quality, list) or not quality:
        raise ValueError("universal codebook study has no quality comparisons")
    quality_keys = []
    task_inventory: set[str] | None = None
    heldout_identity_by_key = {
        _identity_key(model["identity"]): dict(model["identity"]) for model in models
    }
    for index, row in enumerate(quality):
        _validate_quality_row(
            row, f"quality_comparisons[{index}]", codebook_sha256=codebook.sha256())
        key = (*_identity_key(row["model"]), row["phase"], row["activation_mode"])
        quality_keys.append(key)
        if _identity_key(row["model"]) not in heldout_identity_by_key:
            raise ValueError("quality comparison names a model outside the held-out split")
        if dict(row["model"]) != heldout_identity_by_key[_identity_key(row["model"])]:
            raise ValueError("quality comparison model identity differs from held-out identity")
        current_tasks = set(row["reference_tasks"])
        if task_inventory is None:
            task_inventory = current_tasks
        elif current_tasks != task_inventory:
            raise ValueError("quality comparisons do not use one identical task inventory")
    expected_quality = {
        (*model, phase, mode) for model in heldout_keys
        for phase in QUALITY_PHASES for mode in QUALITY_ACTIVATION_MODES
    }
    if len(quality_keys) != len(set(quality_keys)) or set(quality_keys) != expected_quality:
        raise ValueError("PTQ/QAT W-only/W+A8 quality coverage is incomplete or duplicated")

    kernels = report["kernel_performance"]
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("universal codebook study has no kernel performance rows")
    ids, by_backend = [], {}
    for index, row in enumerate(kernels):
        _validate_kernel_row(
            row, f"kernel_performance[{index}]", codebook_sha256=codebook.sha256())
        ids.append(row["id"])
        by_backend.setdefault(row["backend"], set()).add(row["workload"])
    if len(ids) != len(set(ids)):
        raise ValueError("kernel performance row ids are duplicated")
    if any(workloads != set(KERNEL_WORKLOADS) for workloads in by_backend.values()):
        raise ValueError("each claimed kernel backend requires decode and prefill evidence")
    _validate_gates(report["predeclared_gates"], report)
    if not isinstance(report["commands"], list) or not report["commands"] \
            or not all(isinstance(command, str) and command for command in report["commands"]):
        raise ValueError("universal codebook study commands must be nonempty")
    if not isinstance(report["provenance"], Mapping) or not report["provenance"]:
        raise ValueError("universal codebook study provenance must be nonempty")
    return quality, kernels


def evaluate_universal_codebook_study(report: Mapping[str, Any],
                                      codebook: Codebook) -> dict[str, Any]:
    """Validate evidence and recompute every universal-codebook gate."""
    quality, kernels = _validate_report_evidence(report, codebook)
    gates = report["predeclared_gates"]
    heldout = report["heldout"]
    aggregate = heldout["aggregate"]
    checks: dict[str, bool] = {
        "universe.max_squared_trit_distance": (
            report["universe_coverage"]["max_squared_trit_distance"]
            <= gates["maximum_universe_squared_trit_distance"]),
        "distortion.aggregate.exact_hit_rate": (
            aggregate["exact_hit_rate"] >= gates["minimum_aggregate_exact_hit_rate"]),
        "distortion.aggregate.frequency": (
            aggregate["frequency_weighted_distortion_mean"]
            <= gates["maximum_aggregate_frequency_weighted_distortion"]),
        "distortion.aggregate.sensitivity": (
            aggregate["sensitivity_weighted_distortion_mean"]
            <= gates["maximum_aggregate_sensitivity_weighted_distortion"]),
    }
    for model in heldout["models"]:
        label = _identity_label(model["identity"])
        metrics = model["metrics"]
        checks[f"distortion.model.{label}.exact_hit_rate"] = (
            metrics["exact_hit_rate"] >= gates["minimum_model_exact_hit_rate"])
        checks[f"distortion.model.{label}.frequency"] = (
            metrics["frequency_weighted_distortion_mean"]
            <= gates["maximum_model_frequency_weighted_distortion"])
        checks[f"distortion.model.{label}.sensitivity"] = (
            metrics["sensitivity_weighted_distortion_mean"]
            <= gates["maximum_model_sensitivity_weighted_distortion"])
        for family, evidence in model["families"].items():
            metric = evidence["metrics"]
            prefix = f"distortion.family.{label}.{family}"
            checks[prefix + ".exact_hit_rate"] = (
                metric["exact_hit_rate"] >= gates["minimum_family_exact_hit_rate"])
            checks[prefix + ".frequency"] = (
                metric["frequency_weighted_distortion_mean"]
                <= gates["maximum_family_frequency_weighted_distortion"])
            checks[prefix + ".sensitivity"] = (
                metric["sensitivity_weighted_distortion_mean"]
                <= gates["maximum_family_sensitivity_weighted_distortion"])
    for row in quality:
        label = _identity_label(row["model"])
        prefix = f"quality.{label}.{row['phase']}.{row['activation_mode']}"
        delta = row["deltas"]
        checks[prefix + ".cross_entropy"] = (
            delta["cross_entropy"] <= gates["maximum_quality_cross_entropy_increase"])
        checks[prefix + ".teacher_kl_mean"] = (
            delta["teacher_kl_mean"]
            <= gates["maximum_quality_teacher_kl_mean_increase"])
        checks[prefix + ".teacher_kl_p99"] = (
            delta["teacher_kl_p99"]
            <= gates["maximum_quality_teacher_kl_p99_increase"])
        checks[prefix + ".top_token_agreement"] = (
            -delta["top_token_agreement"]
            <= gates["maximum_quality_top_token_agreement_drop"])
        checks[prefix + ".tasks"] = (
            -delta["aggregate_task_score"]
            <= gates["maximum_quality_task_score_drop"])
    for row in kernels:
        ratio = (float(row["candidate_timing"]["median_ms"])
                 / float(row["reference_timing"]["median_ms"]))
        checks[f"kernel.{row['id']}.latency"] = ratio <= gates[
            f"maximum_{row['workload']}_latency_ratio"]
        checks[f"kernel.{row['id']}.correctness"] = row["correctness"]["passed"] is True
    return {"accepted": all(checks.values()), "checks": dict(sorted(checks.items()))}


def finalize_universal_codebook_study(report: Mapping[str, Any],
                                      codebook: Codebook) -> dict[str, Any]:
    """Return a copy with the mechanically recomputed decision attached."""
    result = copy.deepcopy(dict(report))
    result["decision"] = evaluate_universal_codebook_study(result, codebook)
    validate_universal_codebook_study(result, codebook)
    return result


def validate_universal_codebook_study(report: Mapping[str, Any],
                                      codebook: Codebook) -> dict[str, Any]:
    expected = evaluate_universal_codebook_study(report, codebook)
    decision = report["decision"]
    if not isinstance(decision, Mapping) or dict(decision) != expected:
        raise ValueError("universal codebook decision differs from recomputed gates")
    return expected


def measurement_document(codebook: Codebook, corpus: PatternCorpus, *,
                         corpus_path: str | Path,
                         corpus_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Create the copyable evidence object for one model/family corpus."""
    return {
        "schema": PATTERN_CORPUS_SCHEMA,
        "codebook": codebook.ref().to_dict(),
        "pattern_corpus_sha256": file_sha256(corpus_path),
        "pattern_corpus_metadata_sha256": canonical_document_sha256(corpus_metadata),
        "metrics": distortion_metrics(codebook, corpus),
    }
