"""Deterministic calibration parsing and mergeable activation statistics."""

from __future__ import annotations

import hashlib
import json
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
        if not record_count:
            raise ValueError("calibration collector received no records")
        return collector.sums, {
            "records": record_count,
            "retained_tokens": retained_tokens,
            "truncated_tokens": truncated_tokens,
            "bucket_tokens": dict(sorted(buckets.items())),
        }
    finally:
        model.train(was_training)


def normalized_statistics(sums: ModuleSums, *, ridge_factor: float = 1e-5) \
        -> dict[str, torch.Tensor]:
    if sums.token_count <= 0:
        raise ValueError("module collected no tokens")
    diag = sums.diag_sum / sums.token_count
    mean_diag = float(diag.mean())
    if not mean_diag > 0 or not torch.isfinite(diag).all():
        raise ValueError("module has zero or invalid diagonal statistics")
    out = {"diag": (diag / mean_diag).float()}
    if sums.cov8_sum is not None:
        cov = sums.cov8_sum / sums.token_count
        cov = (cov + cov.transpose(-1, -2)) * 0.5
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
        raw_mean = float(cov.diagonal(dim1=-2, dim2=-1).mean())
        cov += torch.eye(256, dtype=torch.float64)[None] * (ridge_factor * raw_mean)
        cov /= cov.diagonal(dim1=-2, dim2=-1).mean()
        out["cov256"] = cov.float()
    return out


def save_calibration_artifact(path: str | Path, sums: Mapping[str, ModuleSums], *,
                              metadata: Mapping[str, Any], ridge_factor: float = 1e-5) -> None:
    tensors: dict[str, torch.Tensor] = {}
    counts = {}
    for name, module_sums in sorted(sums.items()):
        normalized = normalized_statistics(module_sums, ridge_factor=ridge_factor)
        tensors.update({f"{name}.{key}": value for key, value in normalized.items()})
        tensors[f"{name}.__raw_diag_sum"] = module_sums.diag_sum
        if module_sums.cov8_sum is not None:
            tensors[f"{name}.__raw_cov8_sum"] = module_sums.cov8_sum
        if module_sums.cov256_sum is not None:
            tensors[f"{name}.__raw_cov256_sum"] = module_sums.cov256_sum
        counts[name] = module_sums.token_count
    meta = {
        "tq1_calibration_schema": "1",
        "metadata_json": json.dumps({**metadata,
                                      "collector_source_sha256": file_sha256(__file__),
                                      "token_counts": counts,
                                      "ridge_factor": ridge_factor},
                                     sort_keys=True, separators=(",", ":")),
    }
    save_file(tensors, str(path), metadata=meta)


def load_calibration_artifact(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors import safe_open
    tensors = load_file(str(path), device="cpu")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("tq1_calibration_schema") != "1":
        raise ValueError("unsupported calibration artifact schema")
    return tensors, json.loads(metadata["metadata_json"])


def load_calibration_sums(path: str | Path, *, verify_normalized: bool = True) \
        -> tuple[dict[str, ModuleSums], dict[str, Any]]:
    """Reconstruct mergeable raw sums; normalized tensors are never averaged."""
    tensors, metadata = load_calibration_artifact(path)
    counts = metadata.get("token_counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("calibration artifact lacks per-module token counts")
    result: dict[str, ModuleSums] = {}
    for name, token_count in sorted(counts.items()):
        raw_diag = tensors.get(f"{name}.__raw_diag_sum")
        if raw_diag is None or raw_diag.dtype != torch.float64 or raw_diag.ndim != 1:
            raise ValueError(f"{name}: missing float64 raw diagonal sum")
        modes = {"diagonal"}
        if f"{name}.__raw_cov8_sum" in tensors:
            modes.add("covariance8")
        if f"{name}.__raw_cov256_sum" in tensors:
            modes.add("block256")
        sums = ModuleSums(raw_diag.numel(), frozenset(modes))
        sums.token_count = int(token_count)
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
    return result, metadata


_MERGE_IDENTITY_FIELDS = (
    "model", "model_revision", "tokenizer", "tokenizer_revision",
    "tokenizer_sha256", "chat_template_sha256", "parsing_mode",
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
    supplied = dict(metadata or {})
    conflict = set(supplied) & set(_MERGE_IDENTITY_FIELDS)
    if conflict:
        raise ValueError(f"merge metadata may not override identity fields {sorted(conflict)}")
    aggregate.update(supplied)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_artifact(
        destination, sums, metadata=aggregate,
        ridge_factor=float(first_meta.get("ridge_factor", 1e-5)))
    return destination
