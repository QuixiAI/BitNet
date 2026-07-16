"""Calibrated Q4/Q8 KV-cache reference, artifact, and evidence contracts.

This module is the permanent scalar oracle.  It makes no speed claim: the
packed formats reduce cache storage, while attention currently dequantizes to a
floating tensor before the framework attention operator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


KV_CALIBRATION_SCHEMA = 1
KV_LINK_SCHEMA = 1
KV_EVALUATION_SCHEMA = 1
KV_MODES = {"fp16", "q8", "q4"}
ROTATION_STATES = {"pre_rope", "post_rope"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _document_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _full_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


@dataclass(frozen=True)
class KVCalibrationContract:
    model_artifact_sha256: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    layer_count: int
    num_kv_heads: int
    head_dim: int
    kv_dtype: str
    rotation_state: str
    attention_implementation: str
    context_lengths: tuple[int, ...]
    record_count: int
    token_count: int
    source_sha256: tuple[str, ...]
    accumulation_dtype: str = "float64_cpu"
    statistic: str = "key_channel_mean_per_layer_head"
    schema: int = KV_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        _full_sha(self.model_artifact_sha256, "model_artifact_sha256")
        for name in ("model_id", "model_revision", "tokenizer_id",
                     "tokenizer_revision", "attention_implementation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"KV calibration {name} must be nonempty")
        if self.schema != KV_CALIBRATION_SCHEMA:
            raise ValueError("unsupported KV calibration schema")
        if self.layer_count < 1 or self.num_kv_heads < 1 or self.head_dim < 2:
            raise ValueError("KV calibration model dimensions are invalid")
        if self.head_dim % 2:
            raise ValueError("KV head_dim must be even for Q4 packing")
        if self.kv_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("KV calibration dtype is unsupported")
        if self.rotation_state not in ROTATION_STATES:
            raise ValueError("KV calibration rotation state is invalid")
        if self.accumulation_dtype != "float64_cpu" \
                or self.statistic != "key_channel_mean_per_layer_head":
            raise ValueError("KV calibration statistic contract is unsupported")
        if self.record_count < 1 or self.token_count < self.record_count:
            raise ValueError("KV calibration record/token counts are invalid")
        if not self.context_lengths or tuple(sorted(set(self.context_lengths))) \
                != self.context_lengths or any(value < 2 for value in self.context_lengths):
            raise ValueError("KV calibration context lengths must be sorted and unique")
        if not self.source_sha256:
            raise ValueError("KV calibration requires source hashes")
        for index, value in enumerate(self.source_sha256):
            _full_sha(value, f"source_sha256[{index}]")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KVCalibrationContract":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("KV calibration contract has an invalid schema")
        copied = dict(value)
        copied["context_lengths"] = tuple(copied["context_lengths"])
        copied["source_sha256"] = tuple(copied["source_sha256"])
        return cls(**copied)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["context_lengths"] = list(self.context_lengths)
        value["source_sha256"] = list(self.source_sha256)
        return value


@dataclass(frozen=True)
class KVRuntimeIdentity:
    model_artifact_sha256: str
    layer_count: int
    num_kv_heads: int
    head_dim: int
    kv_dtype: str
    rotation_state: str
    attention_implementation: str

    def __post_init__(self) -> None:
        _full_sha(self.model_artifact_sha256, "runtime model_artifact_sha256")


class KVMeanCollector:
    """FP64 CPU collector for explicit ``[batch, kv_head, token, channel]`` keys."""

    def __init__(self, layer_count: int, num_kv_heads: int, head_dim: int):
        if min(layer_count, num_kv_heads, head_dim) < 1:
            raise ValueError("KV collector dimensions must be positive")
        self.shape = (layer_count, num_kv_heads, head_dim)
        self.sums = torch.zeros(self.shape, dtype=torch.float64)
        self.counts = torch.zeros(layer_count, dtype=torch.int64)

    def add(self, layer: int, keys: torch.Tensor,
            token_mask: torch.Tensor | None = None) -> None:
        if not 0 <= layer < self.shape[0] or keys.ndim != 4 \
                or tuple(keys.shape[1::2]) != self.shape[1:]:
            raise ValueError("KV collector expects keys shaped [B,H,T,D]")
        value = keys.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(value).all():
            raise ValueError("KV calibration keys contain NaN or infinity")
        if token_mask is None:
            self.sums[layer] += value.sum(dim=(0, 2))
            self.counts[layer] += value.shape[0] * value.shape[2]
            return
        if token_mask.shape != (value.shape[0], value.shape[2]) \
                or token_mask.dtype != torch.bool:
            raise ValueError("KV token mask must be bool [B,T]")
        expanded = token_mask[:, None, :, None]
        self.sums[layer] += torch.where(expanded, value, 0).sum(dim=(0, 2))
        self.counts[layer] += int(token_mask.sum())

    def means(self, *, expected_token_count: int | None = None) -> torch.Tensor:
        if torch.any(self.counts <= 0) or not torch.equal(
                self.counts, self.counts[:1].expand_as(self.counts)):
            raise ValueError("every KV layer must collect the same nonzero token count")
        if expected_token_count is not None and int(self.counts[0]) != expected_token_count:
            raise ValueError("KV collector token count differs from calibration contract")
        result = self.sums / self.counts[:, None, None]
        if not torch.isfinite(result).all():
            raise ValueError("KV channel means are nonfinite")
        return result.float()


def _link_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def save_kv_calibration(path: str | Path, means: torch.Tensor,
                        contract: KVCalibrationContract, *, overwrite: bool = False) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    link_path = _link_path(path)
    if (path.exists() or link_path.exists()) and not overwrite:
        raise FileExistsError(path)
    expected = (contract.layer_count, contract.num_kv_heads, contract.head_dim)
    means = means.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(means.shape) != expected or not torch.isfinite(means).all():
        raise ValueError("KV means do not match the calibration contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    metadata = {
        "kv_calibration_schema": str(KV_CALIBRATION_SCHEMA),
        "contract_json": _canonical(contract.to_dict()),
    }
    save_file({"key_channel_mean": means}, str(temporary), metadata=metadata)
    temporary.replace(path)
    link = {
        "schema": KV_LINK_SCHEMA,
        "artifact": path.name,
        "artifact_sha256": file_sha256(path),
        "contract_sha256": _document_sha256(contract.to_dict()),
        "model_artifact_sha256": contract.model_artifact_sha256,
    }
    link_path.write_text(json.dumps(link, indent=2, sort_keys=True) + "\n")
    return link


def load_kv_calibration(path: str | Path, expected: KVRuntimeIdentity, *,
                        expected_artifact_sha256: str | None = None) \
        -> tuple[torch.Tensor, KVCalibrationContract, dict[str, Any]]:
    """Load only when every model/layer/head/dtype/rotation identity matches."""
    path = Path(path).expanduser().resolve()
    link = json.loads(_link_path(path).read_text())
    if not isinstance(link, Mapping) or set(link) != {
            "schema", "artifact", "artifact_sha256", "contract_sha256",
            "model_artifact_sha256"} or link["schema"] != KV_LINK_SCHEMA:
        raise ValueError("KV calibration link manifest is invalid")
    if link["artifact"] != path.name or file_sha256(path) != link["artifact_sha256"]:
        raise ValueError("KV calibration artifact hash mismatch")
    if expected_artifact_sha256 is not None \
            and link["artifact_sha256"] != expected_artifact_sha256:
        raise ValueError("KV calibration artifact differs from the requested hash")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
    if metadata.get("kv_calibration_schema") != str(KV_CALIBRATION_SCHEMA) \
            or keys != ["key_channel_mean"]:
        raise ValueError("KV calibration safetensors schema is invalid")
    contract = KVCalibrationContract.from_dict(json.loads(metadata["contract_json"]))
    if _document_sha256(contract.to_dict()) != link["contract_sha256"] \
            or contract.model_artifact_sha256 != link["model_artifact_sha256"]:
        raise ValueError("KV calibration contract/link mismatch")
    identity_fields = (
        "model_artifact_sha256", "layer_count", "num_kv_heads", "head_dim",
        "kv_dtype", "rotation_state", "attention_implementation")
    for name in identity_fields:
        if getattr(contract, name) != getattr(expected, name):
            raise ValueError(f"KV calibration runtime mismatch at {name}")
    means = load_file(str(path), device="cpu")["key_channel_mean"]
    if means.dtype != torch.float32 \
            or tuple(means.shape) != (contract.layer_count, contract.num_kv_heads,
                                      contract.head_dim) \
            or not torch.isfinite(means).all():
        raise ValueError("KV calibration mean tensor is invalid")
    return means, contract, dict(link)


@dataclass(frozen=True)
class PackedKV:
    mode: str
    payload: torch.Tensor
    scales: torch.Tensor | None
    logical_shape: tuple[int, int, int, int]
    centered: bool

    @property
    def physical_bytes(self) -> int:
        return self.payload.numel() * self.payload.element_size() + (
            0 if self.scales is None else self.scales.numel() * self.scales.element_size())


@dataclass(frozen=True)
class PackedKVPair:
    keys: PackedKV
    values: PackedKV
    key_mean: torch.Tensor | None

    @property
    def physical_bytes(self) -> int:
        return self.keys.physical_bytes + self.values.physical_bytes + (
            0 if self.key_mean is None
            else self.key_mean.numel() * self.key_mean.element_size())


def _quantize_tensor(value: torch.Tensor, mode: str, *, centered: bool) -> PackedKV:
    if value.ndim != 4 or value.shape[-1] % 2:
        raise ValueError("KV tensor must be [B,H,T,D] with even D")
    shape = tuple(int(item) for item in value.shape)
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError("KV tensor contains NaN or infinity")
    if mode == "fp16":
        return PackedKV(mode, value.to(torch.float16).contiguous(), None, shape, centered)
    if mode not in {"q8", "q4"}:
        raise ValueError("KV mode must be fp16, q8, or q4")
    qmax = 127 if mode == "q8" else 7
    maximum = value.abs().amax(-1, keepdim=True)
    scales = (maximum / qmax).to(torch.float16)
    denominator = torch.where(scales.float() > 0, scales.float(),
                              torch.ones_like(scales.float()))
    codes = torch.round(value / denominator).clamp(-qmax, qmax).to(torch.int8)
    codes[maximum.expand_as(codes) == 0] = 0
    if mode == "q8":
        payload = codes.contiguous()
    else:
        unsigned = (codes.to(torch.int16) + 7).to(torch.uint8)
        payload = (unsigned[..., 0::2]
                   | torch.bitwise_left_shift(unsigned[..., 1::2], 4)).contiguous()
    return PackedKV(mode, payload, scales.contiguous(), shape, centered)


def _dequantize_tensor(packed: PackedKV) -> torch.Tensor:
    shape = packed.logical_shape
    if len(shape) != 4 or shape[-1] % 2:
        raise ValueError("packed KV logical shape is invalid")
    if packed.mode == "fp16":
        if packed.scales is not None or tuple(packed.payload.shape) != shape \
                or packed.payload.dtype != torch.float16:
            raise ValueError("FP16 KV payload is invalid")
        return packed.payload.float()
    if packed.scales is None or packed.scales.dtype != torch.float16 \
            or tuple(packed.scales.shape) != (*shape[:-1], 1):
        raise ValueError("quantized KV scales are invalid")
    if packed.mode == "q8":
        if packed.payload.dtype != torch.int8 or tuple(packed.payload.shape) != shape:
            raise ValueError("Q8 KV payload is invalid")
        codes = packed.payload
    elif packed.mode == "q4":
        if packed.payload.dtype != torch.uint8 \
                or tuple(packed.payload.shape) != (*shape[:-1], shape[-1] // 2):
            raise ValueError("Q4 KV payload is invalid")
        low = torch.bitwise_and(packed.payload, 0x0F)
        high = torch.bitwise_right_shift(packed.payload, 4)
        unsigned = torch.stack((low, high), -1).reshape(shape)
        if torch.any(unsigned == 15):
            raise ValueError("Q4 KV payload contains a reserved code")
        codes = (unsigned.to(torch.int16) - 7).to(torch.int8)
    else:
        raise ValueError("packed KV mode is invalid")
    return codes.float() * packed.scales.float()


def quantize_kv_pair(keys: torch.Tensor, values: torch.Tensor, mode: str, *,
                     key_mean: torch.Tensor | None = None,
                     center_keys: bool = True) -> PackedKVPair:
    if keys.shape != values.shape:
        raise ValueError("key/value cache shapes differ")
    mean = None
    centered_keys = keys
    if center_keys:
        if key_mean is None or key_mean.shape != (keys.shape[1], keys.shape[3]):
            raise ValueError("centered KV quantization needs [H,D] key means")
        mean = key_mean.detach().float().cpu().contiguous()
        centered_keys = keys.float() - mean.to(keys.device)[None, :, None, :]
    return PackedKVPair(
        _quantize_tensor(centered_keys, mode, centered=center_keys),
        _quantize_tensor(values, mode, centered=False), mean)


def dequantize_kv_pair(pair: PackedKVPair, *, device: torch.device | str | None = None,
                       dtype: torch.dtype = torch.float32) \
        -> tuple[torch.Tensor, torch.Tensor]:
    keys = _dequantize_tensor(pair.keys)
    values = _dequantize_tensor(pair.values)
    if pair.keys.centered:
        if pair.key_mean is None or pair.key_mean.shape != (keys.shape[1], keys.shape[3]):
            raise ValueError("centered KV payload is missing its channel mean")
        keys += pair.key_mean.to(keys.device)[None, :, None, :]
    elif pair.key_mean is not None:
        raise ValueError("uncentered KV payload must not carry a key mean")
    target = torch.device(device) if device is not None else keys.device
    return keys.to(device=target, dtype=dtype), values.to(device=target, dtype=dtype)


def fake_quantize_kv(value: torch.Tensor, bits: int, *,
                     channel_mean: torch.Tensor | None = None) -> torch.Tensor:
    """Optional QAT ablation: hard Q4/Q8 forward and identity STE backward."""
    if bits not in {4, 8}:
        raise ValueError("KV fake quantization supports 4 or 8 bits")
    centered = value
    if channel_mean is not None:
        if channel_mean.shape != (value.shape[1], value.shape[3]):
            raise ValueError("KV fake-quant channel mean shape mismatch")
        centered = value - channel_mean[None, :, None, :].to(value)
    packed = _quantize_tensor(centered, f"q{bits}", centered=channel_mean is not None)
    hard = _dequantize_tensor(packed).to(value)
    if channel_mean is not None:
        hard = hard + channel_mean[None, :, None, :].to(value)
    return value + (hard - value).detach()


def attention_reference(query: torch.Tensor, pair: PackedKVPair, *,
                        attention_mask: torch.Tensor | None = None,
                        scale: float | None = None) -> torch.Tensor:
    keys, values = dequantize_kv_pair(pair, device=query.device, dtype=query.dtype)
    if query.ndim != 4 or query.shape[:2] != keys.shape[:2] \
            or query.shape[-1] != keys.shape[-1]:
        raise ValueError("query and KV shapes are incompatible")
    return F.scaled_dot_product_attention(
        query, keys, values, attn_mask=attention_mask, scale=scale)


def forward_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) \
        -> dict[str, float]:
    if reference_logits.shape != candidate_logits.shape or reference_logits.ndim < 2:
        raise ValueError("KV evaluation logit shapes differ")
    reference = reference_logits.detach().float()
    candidate = candidate_logits.detach().float()
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("KV evaluation logits are nonfinite")
    log_reference = F.log_softmax(reference, -1)
    values = (log_reference.exp()
              * (log_reference - F.log_softmax(candidate, -1))).sum(-1).flatten().double()
    return {"mean": float(values.mean()), "p50": float(torch.quantile(values, 0.50)),
            "p95": float(torch.quantile(values, 0.95)),
            "p99": float(torch.quantile(values, 0.99))}


def validate_kv_evaluation_report(report: Mapping[str, Any], *,
                                  model_artifact_sha256: str,
                                  calibration_artifact_sha256: str) -> None:
    """Validate, but never invent, the production evidence required by QI-4."""
    fields = {"schema", "model_artifact_sha256", "calibration_artifact_sha256",
              "modes", "centering_ablation", "commands", "provenance"}
    if not isinstance(report, Mapping) or set(report) != fields \
            or report["schema"] != KV_EVALUATION_SCHEMA:
        raise ValueError("KV evaluation report has an invalid top-level schema")
    if report["model_artifact_sha256"] != _full_sha(
            model_artifact_sha256, "model_artifact_sha256") \
            or report["calibration_artifact_sha256"] != _full_sha(
                calibration_artifact_sha256, "calibration_artifact_sha256"):
        raise ValueError("KV evaluation artifact identity mismatch")
    modes = report["modes"]
    if not isinstance(modes, Mapping) or set(modes) != KV_MODES:
        raise ValueError("KV evaluation must include FP16, Q8, and Q4")
    mode_fields = {
        "own_generation_kl", "off_policy_kl", "downstream_scores",
        "context_results", "peak_cache_bytes_by_context",
        "decode_latency_ms_by_context", "prefill_latency_ms_by_context",
        "centered_keys",
    }
    context_inventory: set[str] | None = None
    for mode, result in modes.items():
        if not isinstance(result, Mapping) or set(result) != mode_fields:
            raise ValueError(f"KV evaluation {mode} has an invalid schema")
        for population in ("own_generation_kl", "off_policy_kl"):
            values = result[population]
            if not isinstance(values, Mapping) or set(values) != {"mean", "p50", "p95", "p99"}:
                raise ValueError(f"KV evaluation {mode}.{population} is invalid")
            numbers = [float(values[key]) for key in ("mean", "p50", "p95", "p99")]
            if any(not math.isfinite(value) or value < -1e-6 for value in numbers) \
                    or not numbers[1] <= numbers[2] <= numbers[3]:
                raise ValueError(f"KV evaluation {mode}.{population} KL is invalid")
        for section in ("downstream_scores", "context_results"):
            values = result[section]
            if not isinstance(values, Mapping) or not values \
                    or any(not isinstance(name, str) or not name
                           or not isinstance(value, (int, float))
                           or isinstance(value, bool) or not math.isfinite(float(value))
                           for name, value in values.items()):
                raise ValueError(f"KV evaluation {mode}.{section} is invalid")
        contexts = set(result["context_results"])
        if len(contexts) < 2 or any(not name.isdigit() or int(name) < 2
                                    for name in contexts):
            raise ValueError(f"KV evaluation {mode} needs several named context lengths")
        if context_inventory is None:
            context_inventory = contexts
        elif contexts != context_inventory:
            raise ValueError("KV evaluation context inventories differ by mode")
        peaks = result["peak_cache_bytes_by_context"]
        if not isinstance(peaks, Mapping) or set(peaks) != contexts \
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                       for value in peaks.values()):
            raise ValueError(f"KV evaluation {mode} peak cache bytes are invalid")
        for timing_kind in ("decode_latency_ms_by_context",
                            "prefill_latency_ms_by_context"):
            timings = result[timing_kind]
            if not isinstance(timings, Mapping) or set(timings) != contexts:
                raise ValueError(
                    f"KV evaluation {mode}.{timing_kind} context inventory is invalid")
            for context, timing in timings.items():
                if not isinstance(timing, Mapping) \
                        or set(timing) != {"median", "p20", "p80"}:
                    raise ValueError(
                        f"KV evaluation {mode}.{timing_kind}.{context} schema is invalid")
                values = [float(timing[key]) for key in ("p20", "median", "p80")]
                if any(not math.isfinite(value) or value <= 0 for value in values) \
                        or not values[0] <= values[1] <= values[2]:
                    raise ValueError(
                        f"KV evaluation {mode}.{timing_kind}.{context} is invalid")
        if result["centered_keys"] is not (mode == "q4"):
            raise ValueError("only the Q4 primary result uses calibrated key centering")
    for section in ("downstream_scores", "context_results"):
        inventories = [set(modes[mode][section]) for mode in sorted(KV_MODES)]
        if any(inventory != inventories[0] for inventory in inventories[1:]):
            raise ValueError(f"KV evaluation {section} inventories differ by mode")
    if any(abs(float(modes["fp16"][population][key])) > 1e-6
           for population in ("own_generation_kl", "off_policy_kl")
           for key in ("mean", "p50", "p95", "p99")):
        raise ValueError("FP16 KV reference must be compared with itself")
    assert context_inventory is not None
    for context in context_inventory:
        if not modes["q4"]["peak_cache_bytes_by_context"][context] \
                < modes["q8"]["peak_cache_bytes_by_context"][context] \
                < modes["fp16"]["peak_cache_bytes_by_context"][context]:
            raise ValueError(
                "KV cache byte measurements do not reflect Q4/Q8/FP16 storage")
    ablation = report["centering_ablation"]
    if not isinstance(ablation, Mapping) or set(ablation) != {"q4_centered", "q4_uncentered"}:
        raise ValueError("KV evaluation requires a Q4 key-centering ablation")
    for name, values in ablation.items():
        if not isinstance(values, Mapping) or set(values) != {
                "own_generation_kl_mean", "off_policy_kl_mean",
                "peak_cache_bytes_by_context"}:
            raise ValueError(f"KV centering ablation {name} is invalid")
        for key, value in values.items():
            if key == "peak_cache_bytes_by_context":
                if not isinstance(value, Mapping) or set(value) != context_inventory \
                        or any(isinstance(count, bool) or not isinstance(count, int)
                               or count < 1 for count in value.values()):
                    raise ValueError(f"KV centering ablation {name}.{key} is invalid")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"KV centering ablation {name}.{key} is invalid")
    if not isinstance(report["commands"], list) or not report["commands"] \
            or not all(isinstance(value, str) and value for value in report["commands"]):
        raise ValueError("KV evaluation commands must be nonempty")
    if not isinstance(report["provenance"], Mapping) or not report["provenance"]:
        raise ValueError("KV evaluation provenance must be nonempty")
