"""Deterministic calibration parsing and mergeable activation statistics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import nn


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationRecord:
    input_ids: torch.Tensor
    bucket: str
    source: str
    truncated_tokens: int = 0


def _validate_messages(value: Any, line_number: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"line {line_number}: messages must be a nonempty list")
    out = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(f"line {line_number}: message {index} is not an object")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"line {line_number}: message {index} needs string role/content")
        out.append({"role": role, "content": content})
    return out


def iter_calibration_records(path: str | Path, tokenizer, *, limit: int,
                             sequence_cap: int) -> Iterator[CalibrationRecord]:
    """Parse the exact JSONL/plain-text contract; arrays and malformed JSON fail."""
    if limit < 1 or sequence_cap < 1:
        raise ValueError("limit and sequence_cap must be positive")
    retained = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            bucket, source = "unspecified", "unknown"
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                raise ValueError(f"line {line_number}: top-level JSON arrays are not JSONL records")
            if parsed is not None and not isinstance(parsed, Mapping):
                raise ValueError(f"line {line_number}: JSON record must be an object")
            if isinstance(parsed, Mapping):
                bucket = str(parsed.get("bucket", bucket))
                source = str(parsed.get("source", source))
                if "messages" in parsed:
                    messages = _validate_messages(parsed["messages"], line_number)
                    ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False,
                        return_tensors="pt")
                    ids = ids[0] if ids.ndim == 2 else ids
                else:
                    text = next((parsed.get(key) for key in ("text", "prompt", "content")
                                 if isinstance(parsed.get(key), str)), None)
                    if text is None:
                        raise ValueError(f"line {line_number}: record has no supported text field")
                    ids = tokenizer(text, add_special_tokens=True,
                                    return_tensors="pt").input_ids[0]
            else:
                ids = tokenizer(line, add_special_tokens=True,
                                return_tensors="pt").input_ids[0]
            if ids.numel() == 0:
                continue
            original = int(ids.numel())
            ids = ids[:sequence_cap].to(torch.int64).contiguous()
            yield CalibrationRecord(ids, bucket, source, original - int(ids.numel()))
            retained += 1
            if retained >= limit:
                return
    if retained == 0:
        raise ValueError("calibration input retained no usable records")


@dataclass
class ModuleSums:
    width: int
    modes: frozenset[str]
    token_count: int = 0
    diag_sum: torch.Tensor = field(init=False)
    cov8_sum: torch.Tensor | None = field(init=False, default=None)
    cov256_sum: torch.Tensor | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.width % 256:
            raise ValueError("calibration target width must be divisible by 256")
        self.diag_sum = torch.zeros(self.width, dtype=torch.float64)
        if "covariance8" in self.modes:
            self.cov8_sum = torch.zeros((self.width // 8, 8, 8), dtype=torch.float64)
        if "block256" in self.modes:
            self.cov256_sum = torch.zeros((self.width // 256, 256, 256), dtype=torch.float64)

    def add(self, activation: torch.Tensor) -> None:
        value = activation.detach().float().reshape(-1, self.width).cpu().double()
        if not torch.isfinite(value).all():
            raise ValueError("calibration activation contains NaN or infinity")
        self.token_count += value.shape[0]
        self.diag_sum += value.square().sum(0)
        if self.cov8_sum is not None:
            groups = value.reshape(value.shape[0], -1, 8)
            self.cov8_sum += torch.einsum("tgi,tgj->gij", groups, groups)
        if self.cov256_sum is not None:
            blocks = value.reshape(value.shape[0], -1, 256)
            self.cov256_sum += torch.einsum("tbi,tbj->bij", blocks, blocks)

    def merge(self, other: "ModuleSums") -> None:
        if self.width != other.width or self.modes != other.modes:
            raise ValueError("cannot merge incompatible calibration sums")
        self.token_count += other.token_count
        self.diag_sum += other.diag_sum
        if self.cov8_sum is not None:
            self.cov8_sum += other.cov8_sum
        if self.cov256_sum is not None:
            self.cov256_sum += other.cov256_sum


class CalibrationCollector:
    def __init__(self, modules: Mapping[str, nn.Module], *,
                 modes: Sequence[str] = ("diagonal",)):
        legal = {"diagonal", "covariance8", "block256"}
        if not set(modes) <= legal or not modes:
            raise ValueError(f"calibration modes must be a nonempty subset of {sorted(legal)}")
        self.modules = dict(modules)
        self.modes = frozenset(modes)
        self.sums: dict[str, ModuleSums] = {}
        self.handles = []
        for name, module in self.modules.items():
            width = getattr(module, "in_features", None)
            if width is None and hasattr(module, "weight"):
                width = module.weight.shape[-1]
            if width is None:
                raise ValueError(f"{name}: cannot determine input width")
            self.sums[name] = ModuleSums(int(width), self.modes)
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name: str):
        def capture(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                raise ValueError(f"{name}: target module received no tensor input")
            self.sums[name].add(args[0])
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "CalibrationCollector":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


@torch.no_grad()
def collect_model_statistics(model: nn.Module, modules: Mapping[str, nn.Module],
                             records: Iterable[CalibrationRecord], *, device: str | torch.device,
                             modes: Sequence[str] = ("diagonal",)) \
        -> tuple[dict[str, ModuleSums], dict[str, Any]]:
    was_training = model.training
    model.eval()
    record_count = retained_tokens = truncated_tokens = 0
    buckets: dict[str, int] = {}
    sources: dict[str, int] = {}
    try:
        with CalibrationCollector(modules, modes=modes) as collector:
            for record in records:
                ids = record.input_ids[None].to(device)
                model(input_ids=ids, use_cache=False)
                count = int(record.input_ids.numel())
                record_count += 1
                retained_tokens += count
                truncated_tokens += record.truncated_tokens
                buckets[record.bucket] = buckets.get(record.bucket, 0) + count
                sources[record.source] = sources.get(record.source, 0) + count
        if not record_count:
            raise ValueError("calibration collector received no records")
        return collector.sums, {
            "records": record_count,
            "retained_tokens": retained_tokens,
            "truncated_tokens": truncated_tokens,
            "bucket_tokens": dict(sorted(buckets.items())),
            "source_tokens": dict(sorted(sources.items())),
        }
    finally:
        model.train(was_training)


def normalized_statistics(sums: ModuleSums, *, ridge_factor: float = 1e-5) \
        -> dict[str, torch.Tensor]:
    if not isinstance(ridge_factor, (int, float)) or isinstance(ridge_factor, bool) \
            or not torch.isfinite(torch.tensor(float(ridge_factor))) \
            or ridge_factor < 0:
        raise ValueError("ridge_factor must be finite and nonnegative")
    if sums.token_count <= 0:
        raise ValueError("module collected no tokens")
    diag = sums.diag_sum / sums.token_count
    mean_diag = float(diag.mean())
    if not mean_diag > 0 or not torch.isfinite(diag).all() or torch.any(diag < 0):
        raise ValueError("module has zero or invalid diagonal statistics")
    out = {"diag": (diag / mean_diag).float()}
    if sums.cov8_sum is not None:
        cov = sums.cov8_sum / sums.token_count
        cov = (cov + cov.transpose(-1, -2)) * 0.5
        if torch.any(cov.diagonal(dim1=-2, dim2=-1).sum(-1) <= 0):
            raise ValueError("covariance8 contains a zero-statistic group")
        raw_mean = float(cov.diagonal(dim1=-2, dim2=-1).mean())
        cov += torch.eye(8, dtype=torch.float64)[None] * (ridge_factor * raw_mean)
        cov /= cov.diagonal(dim1=-2, dim2=-1).mean()
        eigen_min = float(torch.linalg.eigvalsh(cov).min())
        if eigen_min < -1e-7:
            raise ValueError(f"covariance8 is not PSD after damping (min={eigen_min})")
        out["cov8"] = cov.float()
    if sums.cov256_sum is not None:
        cov = sums.cov256_sum / sums.token_count
        cov = (cov + cov.transpose(-1, -2)) * 0.5
        if torch.any(cov.diagonal(dim1=-2, dim2=-1).sum(-1) <= 0):
            raise ValueError("covariance256 contains a zero-statistic block")
        raw_mean = float(cov.diagonal(dim1=-2, dim2=-1).mean())
        cov += torch.eye(256, dtype=torch.float64)[None] * (ridge_factor * raw_mean)
        cov /= cov.diagonal(dim1=-2, dim2=-1).mean()
        out["cov256"] = cov.float()
    return out


def save_calibration_artifact(path: str | Path, sums: Mapping[str, ModuleSums], *,
                              metadata: Mapping[str, Any], ridge_factor: float = 1e-5,
                              extra_tensors: Mapping[str, torch.Tensor] | None = None) -> None:
    if not sums:
        raise ValueError("calibration artifact requires at least one target module")
    tensors: dict[str, torch.Tensor] = {}
    counts = {}
    for name, module_sums in sorted(sums.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("calibration target module names must be nonempty strings")
        normalized = normalized_statistics(module_sums, ridge_factor=ridge_factor)
        tensors.update({f"{name}.{key}": value for key, value in normalized.items()})
        tensors[f"{name}.__raw_diag_sum"] = module_sums.diag_sum
        if module_sums.cov8_sum is not None:
            tensors[f"{name}.__raw_cov8_sum"] = module_sums.cov8_sum
        if module_sums.cov256_sum is not None:
            tensors[f"{name}.__raw_cov256_sum"] = module_sums.cov256_sum
        counts[name] = module_sums.token_count
    for name, value in sorted((extra_tensors or {}).items()):
        if name in tensors or not isinstance(name, str) \
                or not name.endswith(".token_frequency"):
            raise ValueError(f"invalid duplicate calibration extra tensor {name!r}")
        value = value.detach().contiguous().cpu()
        if value.ndim != 1 or value.dtype not in {
                torch.int32, torch.int64, torch.float32, torch.float64} \
                or (value.is_floating_point() and (
                    not torch.isfinite(value).all()
                    or not torch.equal(value, value.round()))) \
                or torch.any(value < 0):
            raise ValueError(
                f"calibration extra tensor {name} must be a nonnegative integral vector")
        tensors[name] = value
    damping = {
        name: ridge_factor * float(module_sums.diag_sum.mean() / module_sums.token_count)
        for name, module_sums in sorted(sums.items())
        if module_sums.cov8_sum is not None or module_sums.cov256_sum is not None
    }
    try:
        metadata_json = json.dumps(
            {**metadata,
             "collector_source_sha256": file_sha256(__file__),
             "token_counts": counts,
             "ridge_factor": ridge_factor,
             "ridge_damping_before_normalization": damping,
             "normalization": "mean_diagonal_one"},
            sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration metadata is not finite JSON") from exc
    meta = {"tq1_calibration_schema": "1", "metadata_json": metadata_json}
    save_file(tensors, str(path), metadata=meta)


def _read_calibration_artifact(path: str | Path) \
        -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors import safe_open
    tensors = load_file(str(path), device="cpu")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("tq1_calibration_schema") != "1":
        raise ValueError("unsupported calibration artifact schema")
    try:
        decoded = json.loads(
            metadata["metadata_json"],
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")))
    except (KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("calibration artifact has invalid metadata JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("calibration artifact metadata must be an object")
    return tensors, decoded


def load_calibration_artifact(path: str | Path) \
        -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load and reconcile normalized statistics with their mergeable raw sums."""
    tensors, metadata = _read_calibration_artifact(path)
    _calibration_sums_from_contents(tensors, metadata, verify_normalized=True)
    return tensors, metadata


def _calibration_sums_from_contents(
        tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any], *,
        verify_normalized: bool) -> dict[str, ModuleSums]:
    counts = metadata.get("token_counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("calibration artifact lacks per-module token counts")
    result: dict[str, ModuleSums] = {}
    for name, token_count in sorted(counts.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("calibration artifact has an invalid module name")
        raw_diag = tensors.get(f"{name}.__raw_diag_sum")
        if raw_diag is None or raw_diag.dtype != torch.float64 or raw_diag.ndim != 1:
            raise ValueError(f"{name}: missing float64 raw diagonal sum")
        modes = {"diagonal"}
        if f"{name}.__raw_cov8_sum" in tensors:
            modes.add("covariance8")
        if f"{name}.__raw_cov256_sum" in tensors:
            modes.add("block256")
        sums = ModuleSums(raw_diag.numel(), frozenset(modes))
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            raise ValueError(f"{name}: invalid token count")
        sums.token_count = token_count
        if sums.token_count <= 0:
            raise ValueError(f"{name}: invalid token count")
        sums.diag_sum.copy_(raw_diag)
        if sums.cov8_sum is not None:
            raw = tensors[f"{name}.__raw_cov8_sum"]
            if raw.dtype != torch.float64 or raw.shape != sums.cov8_sum.shape:
                raise ValueError(f"{name}: invalid raw covariance8 sum")
            sums.cov8_sum.copy_(raw)
        if sums.cov256_sum is not None:
            raw = tensors[f"{name}.__raw_cov256_sum"]
            if raw.dtype != torch.float64 or raw.shape != sums.cov256_sum.shape:
                raise ValueError(f"{name}: invalid raw covariance256 sum")
            sums.cov256_sum.copy_(raw)
        if verify_normalized:
            expected = normalized_statistics(
                sums, ridge_factor=float(metadata.get("ridge_factor", 1e-5)))
            for suffix, value in expected.items():
                key = f"{name}.{suffix}"
                if key not in tensors or not torch.equal(tensors[key], value):
                    raise ValueError(f"{name}: normalized {suffix} does not match raw sums")
        result[name] = sums
    raw_prefixes = {key.rsplit(".", 1)[0] for key in tensors if ".__raw_" in key}
    if raw_prefixes != set(result):
        raise ValueError("calibration raw-sum inventory differs from metadata")
    expected_tensors = {
        key
        for name, sums in result.items()
        for key in (
            f"{name}.diag", f"{name}.__raw_diag_sum",
            *((f"{name}.cov8", f"{name}.__raw_cov8_sum")
              if sums.cov8_sum is not None else ()),
            *((f"{name}.cov256", f"{name}.__raw_cov256_sum")
              if sums.cov256_sum is not None else ()),
        )
    }
    for key, value in tensors.items():
        if not key.endswith(".token_frequency"):
            continue
        if value.ndim != 1 or value.dtype not in {
                torch.int32, torch.int64, torch.float32, torch.float64} \
                or (value.is_floating_point() and (
                    not torch.isfinite(value).all()
                    or not torch.equal(value, value.round()))) \
                or torch.any(value < 0):
            raise ValueError(f"{key}: invalid token-frequency tensor")
        expected_tensors.add(key)
    if set(tensors) != expected_tensors:
        raise ValueError("calibration tensor inventory contains missing or unknown fields")
    if metadata.get("normalization") != "mean_diagonal_one":
        raise ValueError("calibration normalization metadata is invalid")
    damping = metadata.get("ridge_damping_before_normalization")
    expected_damping = {
        name: float(metadata["ridge_factor"])
        * float(sums.diag_sum.mean() / sums.token_count)
        for name, sums in sorted(result.items())
        if sums.cov8_sum is not None or sums.cov256_sum is not None
    }
    if not isinstance(damping, dict) or set(damping) != set(expected_damping) \
            or any(not isinstance(damping[name], (int, float))
                   or isinstance(damping[name], bool)
                   or not math.isfinite(float(damping[name]))
                   or float(damping[name]) != value
                   for name, value in expected_damping.items()):
        raise ValueError("calibration ridge-damping metadata does not match raw sums")
    targets = metadata.get("target_modules")
    if targets is not None and (not isinstance(targets, list)
                                or len(targets) != len(set(targets))
                                or set(targets) != set(result)):
        raise ValueError("calibration target-module metadata differs from raw sums")
    return result


def load_calibration_sums(path: str | Path, *, verify_normalized: bool = True) \
        -> tuple[dict[str, ModuleSums], dict[str, Any]]:
    """Reconstruct mergeable raw sums; normalized tensors are never averaged."""
    tensors, metadata = _read_calibration_artifact(path)
    return _calibration_sums_from_contents(
        tensors, metadata, verify_normalized=verify_normalized), metadata


_MERGE_IDENTITY_FIELDS = (
    "model", "model_revision", "tokenizer", "tokenizer_revision",
    "tokenizer_sha256", "chat_template_sha256", "parsing_mode", "chat_template_mode",
    "sequence_cap", "accumulation_dtype", "target_modules", "modes",
    "collector_source_sha256", "ridge_factor",
)


def merge_calibration_artifacts(inputs: Sequence[str | Path], output: str | Path, *,
                                metadata: Mapping[str, Any] | None = None,
                                overwrite: bool = False) -> Path:
    """Merge compatible artifacts by raw FP64 sums and exact token counts."""
    paths = [Path(item).expanduser().resolve() for item in inputs]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("merge requires at least two distinct calibration artifacts")
    destination = Path(output).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    loaded = [load_calibration_sums(path) for path in paths]
    raw_artifacts = [load_calibration_artifact(path)[0] for path in paths]
    sums, first_meta = loaded[0]
    sums = {name: ModuleSums(value.width, value.modes) for name, value in sums.items()}
    for name, value in loaded[0][0].items():
        sums[name].merge(value)
    for other_sums, other_meta in loaded[1:]:
        if set(other_sums) != set(sums):
            raise ValueError("calibration target inventories differ")
        for field in _MERGE_IDENTITY_FIELDS:
            present = field in first_meta or field in other_meta
            if present and first_meta.get(field) != other_meta.get(field):
                raise ValueError(f"calibration merge identity differs at {field}")
        for name in sums:
            sums[name].merge(other_sums[name])

    aggregate: dict[str, Any] = {
        field: first_meta[field] for field in _MERGE_IDENTITY_FIELDS
        if field in first_meta and field not in {"ridge_factor"}
    }
    aggregate.update({
        "merged": True,
        "source_artifacts": [str(path) for path in paths],
        "source_artifact_sha256": [file_sha256(path) for path in paths],
        "source_calibration_sha256": [item[1].get("calibration_file_sha256")
                                      for item in loaded],
        "records": sum(int(item[1].get("records", 0)) for item in loaded),
        "retained_tokens": sum(int(item[1].get("retained_tokens", 0)) for item in loaded),
        "truncated_tokens": sum(int(item[1].get("truncated_tokens", 0)) for item in loaded),
        "source_devices": sorted({str(item[1].get("device", "unknown"))
                                  for item in loaded}),
    })
    buckets: dict[str, int] = {}
    for _, item_metadata in loaded:
        for bucket, count in item_metadata.get("bucket_tokens", {}).items():
            buckets[str(bucket)] = buckets.get(str(bucket), 0) + int(count)
    aggregate["bucket_tokens"] = dict(sorted(buckets.items()))
    source_tokens: dict[str, int] = {}
    for _, item_metadata in loaded:
        for source, count in item_metadata.get("source_tokens", {}).items():
            source_tokens[str(source)] = source_tokens.get(str(source), 0) + int(count)
    aggregate["source_tokens"] = dict(sorted(source_tokens.items()))
    supplied = dict(metadata or {})
    conflict = set(supplied) & set(_MERGE_IDENTITY_FIELDS)
    if conflict:
        raise ValueError(f"merge metadata may not override identity fields {sorted(conflict)}")
    aggregate.update(supplied)
    frequency_keys = {
        key for key in raw_artifacts[0] if key.endswith(".token_frequency")
    }
    if any({key for key in tensors if key.endswith(".token_frequency")}
           != frequency_keys for tensors in raw_artifacts[1:]):
        raise ValueError("calibration token-frequency inventories differ")
    extra_tensors: dict[str, torch.Tensor] = {}
    for key in sorted(frequency_keys):
        first = raw_artifacts[0][key]
        if first.ndim != 1 or first.dtype not in {
                torch.int32, torch.int64, torch.float32, torch.float64}:
            raise ValueError(f"{key}: invalid token-frequency tensor")
        total = torch.zeros_like(first, dtype=torch.int64)
        for tensors in raw_artifacts:
            value = tensors[key]
            if value.shape != first.shape or value.dtype != first.dtype:
                raise ValueError(f"{key}: incompatible token-frequency tensor")
            if value.is_floating_point() and not torch.equal(value, value.round()):
                raise ValueError(f"{key}: token frequencies must be integral")
            if (value < 0).any():
                raise ValueError(f"{key}: token frequencies must be nonnegative")
            total += value.to(torch.int64)
        extra_tensors[key] = total
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_artifact(
        destination, sums, metadata=aggregate,
        ridge_factor=float(first_meta.get("ridge_factor", 1e-5)),
        extra_tensors=extra_tensors)
    return destination
