"""Inference modules backed directly by canonical packed TQ1 artifacts."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .artifact import ArtifactReader, tensor_sha256
from .codebook import Codebook
from .oracle import dequantize_weight, linear_w2a8, linear_w_only, quantize_activation


DENSE_REPACK_LAYOUT_VERSION = "tq1_dense_f32_row_major_v1"
RUNTIME_PERFORMANCE_SCHEMA = 1


def _performance_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


def _timing_triplet(value: Any, name: str) -> None:
    if not isinstance(value, dict) or set(value) != {"p20_ms", "median_ms", "p80_ms"}:
        raise ValueError(f"{name} timing schema is invalid")
    numbers = [float(value[key]) for key in ("p20_ms", "median_ms", "p80_ms")]
    if any(not math.isfinite(item) or item <= 0 for item in numbers) \
            or not numbers[0] <= numbers[1] <= numbers[2]:
        raise ValueError(f"{name} timing values are invalid")


def _positive_performance(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def validate_runtime_performance_report(report: dict[str, Any], *,
                                        model_artifact_sha256: str) -> None:
    """Fail-closed model-level QI-5 performance/energy evidence contract."""
    fields = {"schema", "model_artifact_sha256", "backend", "quant_format",
              "routing_policy", "device", "toolchain", "warmups", "iterations",
              "tg128", "pp512", "ttft", "output_head", "memory", "sustained",
              "energy", "commands", "provenance"}
    if not isinstance(report, dict) or set(report) != fields \
            or report["schema"] != RUNTIME_PERFORMANCE_SCHEMA:
        raise ValueError("runtime performance report has an invalid top-level schema")
    if report["model_artifact_sha256"] != _performance_sha(
            model_artifact_sha256, "model_artifact_sha256"):
        raise ValueError("runtime performance model artifact mismatch")
    for key in ("backend", "quant_format"):
        if not isinstance(report[key], str) or not report[key]:
            raise ValueError(f"runtime performance {key} must be nonempty")
    if not isinstance(report["routing_policy"], dict) or not report["routing_policy"]:
        raise ValueError("runtime routing policy must be recorded")
    for section in ("device", "toolchain", "provenance"):
        if not isinstance(report[section], dict) or not report[section]:
            raise ValueError(f"runtime performance {section} must be nonempty")
    for key in ("warmups", "iterations"):
        if isinstance(report[key], bool) or not isinstance(report[key], int) \
                or report[key] < 1:
            raise ValueError(f"runtime performance {key} must be positive")
    tg = report["tg128"]
    if not isinstance(tg, dict) or set(tg) != {
            "prompt_tokens", "generated_tokens", "timing", "tokens_per_second"} \
            or isinstance(tg["prompt_tokens"], bool) \
            or not isinstance(tg["prompt_tokens"], int) or tg["prompt_tokens"] < 1 \
            or isinstance(tg["generated_tokens"], bool) \
            or not isinstance(tg["generated_tokens"], int) \
            or tg["generated_tokens"] != 128:
        raise ValueError("runtime tg128 evidence is invalid")
    _timing_triplet(tg["timing"], "tg128")
    _positive_performance(tg["tokens_per_second"], "tg128.tokens_per_second")
    pp = report["pp512"]
    if not isinstance(pp, dict) or set(pp) != {
            "prompt_tokens", "timing", "tokens_per_second"} \
            or isinstance(pp["prompt_tokens"], bool) \
            or not isinstance(pp["prompt_tokens"], int) \
            or pp["prompt_tokens"] != 512:
        raise ValueError("runtime pp512 evidence is invalid")
    _timing_triplet(pp["timing"], "pp512")
    _positive_performance(pp["tokens_per_second"], "pp512.tokens_per_second")
    ttft = report["ttft"]
    if not isinstance(ttft, dict) or len(ttft) < 3:
        raise ValueError("runtime TTFT needs at least three prompt lengths")
    lengths = []
    for length, timing in ttft.items():
        try:
            lengths.append(int(length))
        except ValueError as exc:
            raise ValueError("runtime TTFT keys must be prompt lengths") from exc
        _timing_triplet(timing, f"ttft.{length}")
    if min(lengths) < 1 or len(set(lengths)) != len(lengths):
        raise ValueError("runtime TTFT prompt lengths are invalid")
    head = report["output_head"]
    if not isinstance(head, dict) or set(head) != {
            "median_ms", "total_decode_median_ms", "decode_time_share"}:
        raise ValueError("runtime output-head evidence is invalid")
    for key in ("median_ms", "total_decode_median_ms", "decode_time_share"):
        _positive_performance(head[key], f"output_head.{key}")
    expected_share = float(head["median_ms"]) / float(head["total_decode_median_ms"])
    if expected_share > 1 or not math.isclose(
            float(head["decode_time_share"]), expected_share, rel_tol=1e-6):
        raise ValueError("runtime output-head share is inconsistent")
    memory = report["memory"]
    memory_fields = {"canonical_artifact_bytes", "canonical_resident_bytes",
                     "backend_private_repack_bytes", "resident_model_bytes",
                     "peak_bytes_by_context"}
    if not isinstance(memory, dict) or set(memory) != memory_fields \
            or any(isinstance(memory[key], bool) or not isinstance(memory[key], int)
                   or memory[key] < 0 for key in memory_fields - {"peak_bytes_by_context"}):
        raise ValueError("runtime memory evidence is invalid")
    peaks = memory["peak_bytes_by_context"]
    if not isinstance(peaks, dict) or len(peaks) < 2 \
            or any(not str(key).isdigit() or isinstance(value, bool)
                   or not isinstance(value, int) or value < 1 for key, value in peaks.items()):
        raise ValueError("runtime peak context memory is invalid")
    if memory["resident_model_bytes"] \
            < memory["canonical_resident_bytes"] + memory["backend_private_repack_bytes"] \
            or any(value < memory["resident_model_bytes"] for value in peaks.values()):
        raise ValueError("runtime resident/peak memory accounting is inconsistent")
    sustained = report["sustained"]
    if not isinstance(sustained, dict) or set(sustained) != {
            "duration_seconds", "generated_tokens", "window_tokens_per_second",
            "thermal_state"} or _positive_performance(
                sustained["duration_seconds"], "sustained.duration_seconds") < 60 \
            or isinstance(sustained["generated_tokens"], bool) \
            or not isinstance(sustained["generated_tokens"], int) \
            or sustained["generated_tokens"] < 128 \
            or not isinstance(sustained["window_tokens_per_second"], list) \
            or len(sustained["window_tokens_per_second"]) < 3 \
            or any(_positive_performance(value, "sustained.window_tokens_per_second") <= 0
                   for value in sustained["window_tokens_per_second"]) \
            or not isinstance(sustained["thermal_state"], str) \
            or not sustained["thermal_state"]:
        raise ValueError("runtime sustained/thermal evidence is invalid")
    energy = report["energy"]
    if not isinstance(energy, dict) or set(energy) != {
            "measurement_method", "average_power_watts", "duration_seconds",
            "energy_joules", "generated_tokens", "joules_per_token"}:
        raise ValueError("runtime energy evidence is invalid")
    for key in ("average_power_watts", "duration_seconds", "energy_joules",
                "joules_per_token"):
        _positive_performance(energy[key], f"energy.{key}")
    if not isinstance(energy["generated_tokens"], int) or energy["generated_tokens"] < 1 \
            or not isinstance(energy["measurement_method"], str) \
            or not energy["measurement_method"]:
        raise ValueError("runtime energy identity/count is invalid")
    if not math.isclose(
            float(energy["energy_joules"]),
            float(energy["average_power_watts"]) * float(energy["duration_seconds"]),
            rel_tol=0.02) or not math.isclose(
                float(energy["joules_per_token"]),
                float(energy["energy_joules"]) / energy["generated_tokens"],
                rel_tol=1e-6):
        raise ValueError("runtime energy-per-token accounting is inconsistent")
    if not isinstance(report["commands"], list) or not report["commands"] \
            or not all(isinstance(value, str) and value for value in report["commands"]):
        raise ValueError("runtime performance commands must be nonempty")


@dataclass(frozen=True)
class NativeRoutingPolicy:
    """Explicit CPU decode/prefill routing and private-repack memory budget."""

    dense_repack_budget_bytes: int = 0
    short_prefill_min_tokens: int = 128
    long_prefill_min_tokens: int = 512
    output_head_dense_decode: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.dense_repack_budget_bytes, bool) \
                or not isinstance(self.dense_repack_budget_bytes, int) \
                or self.dense_repack_budget_bytes < 0:
            raise ValueError("dense repack budget must be a nonnegative byte count")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (self.short_prefill_min_tokens,
                             self.long_prefill_min_tokens)) \
                or not 2 <= self.short_prefill_min_tokens < self.long_prefill_min_tokens:
            raise ValueError("native short/long prefill thresholds are invalid")
        if not isinstance(self.output_head_dense_decode, bool):
            raise ValueError("output-head dense-decode policy must be boolean")


class _DenseRepackMixin:
    """Exact, lazy, versioned F32 row-major repack for BLAS prefill."""

    def _init_dense_repack(self, policy: NativeRoutingPolicy) -> None:
        self.routing_policy = policy
        self.register_buffer("dense_prefill_weight", None, persistent=False)
        self.repack_report.update({
            "dense_layout_version": DENSE_REPACK_LAYOUT_VERSION,
            "dense_repack_materialized": False,
            "dense_repack_bytes": 0,
            "dense_repack_sha256": None,
            "dense_repack_time_ms": None,
            "dense_repack_peak_temporary_bytes_bound": 0,
            "dense_repack_budget_bytes": policy.dense_repack_budget_bytes,
            "short_prefill_min_tokens": policy.short_prefill_min_tokens,
            "long_prefill_min_tokens": policy.long_prefill_min_tokens,
            "output_head_dense_decode": policy.output_head_dense_decode,
            "route_counts": {"packed_decode": 0, "packed_small_batch": 0,
                             "dense_short_prefill": 0, "dense_long_prefill": 0,
                             "dense_output_head_decode": 0},
            "last_route": None,
        })

    @property
    def _dense_repack_required_bytes(self) -> int:
        return int(self.out_features * self.in_features * 4)

    def _route(self, tokens: int, *, output_head: bool = False) -> str:
        budget_allows = (self.routing_policy.dense_repack_budget_bytes
                         >= self._dense_repack_required_bytes)
        if output_head and tokens == 1 and self.routing_policy.output_head_dense_decode \
                and budget_allows:
            route = "dense_output_head_decode"
        elif tokens >= self.routing_policy.long_prefill_min_tokens and budget_allows:
            route = "dense_long_prefill"
        elif tokens >= self.routing_policy.short_prefill_min_tokens and budget_allows:
            route = "dense_short_prefill"
        elif tokens == 1:
            route = "packed_decode"
        else:
            route = "packed_small_batch"
        self.repack_report["route_counts"][route] += 1
        self.repack_report["last_route"] = route
        return route

    @torch.no_grad()
    def materialize_dense_repack(self) -> torch.Tensor:
        if self.dense_prefill_weight is not None:
            return self.dense_prefill_weight
        required = self._dense_repack_required_bytes
        if required > self.routing_policy.dense_repack_budget_bytes:
            raise MemoryError(
                f"dense TQ1 repack needs {required} bytes but policy permits "
                f"{self.routing_policy.dense_repack_budget_bytes}")
        started = time.perf_counter()
        dense = dequantize_weight(
            self.payload, self.profile, self.codebook,
            row_scales=self.row_scales).to(torch.float32).contiguous()
        if tuple(dense.shape) != (self.out_features, self.in_features) \
                or not torch.isfinite(dense).all():
            raise ValueError("dense private repack failed shape/finite validation")
        self.dense_prefill_weight = dense
        dense_hash = tensor_sha256(dense)
        # The bound accounts for the retained dense F32 tensor plus decoded
        # int8 codewords and int64 indices that can overlap during construction.
        groups = self.out_features * self.in_features // 8
        temporary_bound = required + self.out_features * self.in_features + groups * 8
        self.repack_report.update({
            "dense_repack_materialized": True,
            "dense_repack_bytes": required,
            "dense_repack_sha256": dense_hash,
            "dense_repack_time_ms": (time.perf_counter() - started) * 1e3,
            "dense_repack_peak_temporary_bytes_bound": temporary_bound,
            "resident_repack_bytes": (
                int(self.repack_report["resident_repack_bytes"]) + required),
            "peak_temporary_bytes": max(
                int(self.repack_report["peak_temporary_bytes"]), temporary_bound),
        })
        return dense

    def _dense_linear(self, x2: torch.Tensor) -> torch.Tensor:
        activation = quantize_activation(x2, self.activation_mode).dequantize()
        return F.linear(activation, self.materialize_dense_repack())


class PackedTQ1Linear(nn.Module):
    """Permanent scalar packed backend used for model-level parity.

    Payload and codebook data remain immutable buffers.  The implementation
    intentionally calls the scalar oracle; optimized backends must compare to
    this module before claiming coverage.
    """

    def __init__(self, payload: torch.Tensor, profile: str, codebook: Codebook, *,
                 row_scales: torch.Tensor | None, activation_mode: str,
                 state_dict_name: str):
        super().__init__()
        if payload.ndim != 3:
            raise ValueError("PackedTQ1Linear requires [N,K/256,block_bytes]")
        self.register_buffer("payload", payload.detach().to(torch.uint8).cpu().clone())
        self.register_buffer("row_scales", None if row_scales is None else
                             row_scales.detach().cpu().clone())
        self.profile = profile
        self.codebook = codebook
        self.activation_mode = activation_mode
        self.state_dict_name = state_dict_name
        self.in_features = payload.shape[1] * 256
        self.out_features = payload.shape[0]
        self.payload_sha256 = tensor_sha256(self.payload)
        self.codebook_sha256 = codebook.sha256()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError("packed TQ1 activation width mismatch")
        device, dtype = x.device, x.dtype
        if self.activation_mode == "none":
            result = linear_w_only(
                x, self.payload, self.profile, self.codebook,
                row_scales=self.row_scales, output_dtype=dtype)
        else:
            result = linear_w2a8(
                x, self.payload, self.profile, self.codebook,
                row_scales=self.row_scales, activation_mode=self.activation_mode,
                output_dtype=dtype)
        return result.to(device)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"profile={self.profile}, activation_mode={self.activation_mode}")


class NativeCPUTQ1Linear(_DenseRepackMixin, PackedTQ1Linear):
    """Repo-owned native CPU path over canonical schema-2 payload bytes.

    The only backend-private representation is an expanded int8 codebook plus a
    legal-index bitmap.  Its deterministic hash and memory cost are exposed for
    performance reports; canonical payload and codebook state remain resident.
    """

    def __init__(self, *args, impl: str = "auto",
                 routing_policy: NativeRoutingPolicy | None = None, **kwargs):
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        if self.activation_mode == "none":
            raise ValueError("native CPU TQ1 supports W2A8, not W-only execution")
        physical = torch.arange(self.codebook.index_count, dtype=torch.int64)
        expanded = self.codebook.decode(physical).to(torch.int8).contiguous()
        legal = self.codebook.legal_index_mask().to(torch.uint8).contiguous()
        self.register_buffer("expanded_codebook", expanded)
        self.register_buffer("legal_indices", legal)
        self.impl = impl
        resident = expanded.numel() + legal.numel()
        stream = torch.cat((expanded.view(torch.uint8).reshape(-1), legal)).numpy().tobytes()
        self.repack_report = {
            "layout_version": "expanded_codebook_i8_v1",
            "original_codebook_bytes": len(self.codebook.canonical_bytes()),
            "resident_repack_bytes": resident,
            "peak_temporary_bytes": resident,
            "repack_time_ms": (time.perf_counter() - started) * 1e3,
            "repack_sha256": hashlib.sha256(stream).hexdigest(),
            "canonical_packed_remains_resident": True,
            "implementation": impl,
        }
        self._init_dense_repack(routing_policy or NativeRoutingPolicy())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            raise ValueError("native CPU TQ1 requires CPU activations")
        if x.shape[-1] != self.in_features:
            raise ValueError("packed TQ1 activation width mismatch")
        from bitnet_train.cpu import bitnet_cpu

        original = x.shape[:-1]
        x2 = x.detach().float().reshape(-1, self.in_features)
        route = self._route(x2.shape[0])
        if route.startswith("dense_"):
            return self._dense_linear(x2).to(x.dtype).reshape(
                *original, self.out_features)
        activation = quantize_activation(x2, self.activation_mode)
        payload = self.payload.numpy()
        if self.row_scales is None:
            row_bits = None
            scale_dtype = "f16"
        else:
            scale_dtype = "bf16" if self.row_scales.dtype == torch.bfloat16 else "f16"
            row_bits = self.row_scales.contiguous().view(torch.uint16).numpy()
        codebook = self.expanded_codebook.numpy()
        legal = self.legal_indices.numpy()
        result = torch.from_numpy(bitnet_cpu.gemm_tq1(
            payload, row_bits, codebook, legal, activation.codes.numpy(),
            activation.scales.reshape(x2.shape[0], -1).numpy(), self.profile,
            activation_mode=self.activation_mode,
            row_scale_dtype=scale_dtype, impl=self.impl)).to(x.dtype)
        return result.reshape(*original, self.out_features)


class PackedTQ1Embedding(nn.Module):
    """Canonical packed tensor with lookup and output-linear consumers.

    Lookup decodes only unique requested rows, so repeated prompt tokens share
    work and no dense vocabulary matrix is materialized.  ``linear`` is the
    exact packed scalar output-head path.
    """

    def __init__(self, payload: torch.Tensor, profile: str, codebook: Codebook, *,
                 row_scales: torch.Tensor | None, activation_mode: str,
                 state_dict_name: str, padding_idx: int | None = None,
                 output_dtype: torch.dtype = torch.float32):
        super().__init__()
        if payload.ndim != 3:
            raise ValueError("PackedTQ1Embedding requires [vocab,K/256,block_bytes]")
        self.register_buffer("payload", payload.detach().to(torch.uint8).cpu().clone())
        self.register_buffer("row_scales", None if row_scales is None else
                             row_scales.detach().cpu().clone())
        self.profile = profile
        self.codebook = codebook
        self.activation_mode = activation_mode
        self.state_dict_name = state_dict_name
        self.num_embeddings = int(payload.shape[0])
        self.embedding_dim = int(payload.shape[1] * 256)
        self.in_features = self.embedding_dim
        self.out_features = self.num_embeddings
        self.padding_idx = padding_idx
        self.output_dtype = output_dtype
        self.logical_shape = [self.num_embeddings, self.embedding_dim]
        self.payload_sha256 = tensor_sha256(self.payload)
        self.codebook_sha256 = codebook.sha256()

    def _rows(self, row_ids: torch.Tensor) -> torch.Tensor:
        row_ids = row_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if row_ids.numel() == 0:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        if torch.any(row_ids < 0) or torch.any(row_ids >= self.num_embeddings):
            raise ValueError("packed embedding token id is outside the vocabulary")
        scales = None if self.row_scales is None else self.row_scales[row_ids]
        return dequantize_weight(
            self.payload[row_ids], self.profile, self.codebook, row_scales=scales)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in {
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError("packed embedding input must contain integer token ids")
        flat = input_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        rows = self._rows(unique)
        output = rows[inverse].reshape(*input_ids.shape, self.embedding_dim)
        # Match an ordinary embedding's model dtype/device without storing a
        # second dense representation.
        return output.to(device=input_ids.device, dtype=self.output_dtype)

    def linear(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.embedding_dim:
            raise ValueError("packed output-head hidden width mismatch")
        device, dtype = hidden.device, hidden.dtype
        if self.activation_mode == "none":
            result = linear_w_only(
                hidden, self.payload, self.profile, self.codebook,
                row_scales=self.row_scales, output_dtype=dtype)
        else:
            result = linear_w2a8(
                hidden, self.payload, self.profile, self.codebook,
                row_scales=self.row_scales, activation_mode=self.activation_mode,
                output_dtype=dtype)
        return result.to(device)


class NativeCPUTQ1Embedding(_DenseRepackMixin, PackedTQ1Embedding):
    """Packed row gather plus the native CPU output-head GEMV/GEMM."""

    def __init__(self, *args, impl: str = "auto",
                 routing_policy: NativeRoutingPolicy | None = None, **kwargs):
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        if self.activation_mode == "none":
            raise ValueError("native CPU shared TQ1 supports W2A8, not W-only execution")
        physical = torch.arange(self.codebook.index_count, dtype=torch.int64)
        expanded = self.codebook.decode(physical).to(torch.int8).contiguous()
        legal = self.codebook.legal_index_mask().to(torch.uint8).contiguous()
        self.register_buffer("expanded_codebook", expanded)
        self.register_buffer("legal_indices", legal)
        self.impl = impl
        resident = expanded.numel() + legal.numel()
        stream = torch.cat((expanded.view(torch.uint8).reshape(-1), legal)).numpy().tobytes()
        self.repack_report = {
            "layout_version": "expanded_codebook_i8_v1",
            "original_codebook_bytes": len(self.codebook.canonical_bytes()),
            "resident_repack_bytes": resident,
            "peak_temporary_bytes": resident,
            "repack_time_ms": (time.perf_counter() - started) * 1e3,
            "repack_sha256": hashlib.sha256(stream).hexdigest(),
            "canonical_packed_remains_resident": True,
            "implementation": impl,
        }
        self._init_dense_repack(routing_policy or NativeRoutingPolicy())

    def linear(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.device.type != "cpu":
            raise ValueError("native CPU output head requires CPU activations")
        if hidden.shape[-1] != self.embedding_dim:
            raise ValueError("packed output-head hidden width mismatch")
        from bitnet_train.cpu import bitnet_cpu

        original = hidden.shape[:-1]
        x2 = hidden.detach().float().reshape(-1, self.embedding_dim)
        route = self._route(x2.shape[0], output_head=True)
        if route.startswith("dense_"):
            return self._dense_linear(x2).to(hidden.dtype).reshape(
                *original, self.num_embeddings)
        activation = quantize_activation(x2, self.activation_mode)
        if self.row_scales is None:
            row_bits, scale_dtype = None, "f16"
        else:
            scale_dtype = "bf16" if self.row_scales.dtype == torch.bfloat16 else "f16"
            row_bits = self.row_scales.contiguous().view(torch.uint16).numpy()
        result = torch.from_numpy(bitnet_cpu.gemm_tq1(
            self.payload.numpy(), row_bits, self.expanded_codebook.numpy(),
            self.legal_indices.numpy(), activation.codes.numpy(),
            activation.scales.reshape(x2.shape[0], -1).numpy(), self.profile,
            activation_mode=self.activation_mode, row_scale_dtype=scale_dtype,
            impl=self.impl)).to(hidden.dtype)
        return result.reshape(*original, self.num_embeddings)


class PackedTQ1OutputHead(nn.Module):
    """Parameter-free graph consumer of :class:`PackedTQ1Embedding`."""

    def __init__(self, shared: PackedTQ1Embedding):
        super().__init__()
        object.__setattr__(self, "shared_weight", shared)
        self.in_features = shared.embedding_dim
        self.out_features = shared.num_embeddings
        self.bias = None

    @property
    def weight(self):
        # A packed tensor has no dense Parameter.  Expose the shared module for
        # identity diagnostics while keeping it out of state_dict storage.
        return self.shared_weight

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "weight" and "shared_weight" in self.__dict__:
            if value is not self.shared_weight:
                raise ValueError("cannot retie a packed output head")
            return
        super().__setattr__(name, value)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.shared_weight.linear(hidden)


def _parent_and_attr(model: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    parent_name, _, attribute = module_path.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), attribute


def _unique_storage_bytes(values) -> int:
    storages: dict[tuple[str, int | None, int, int], int] = {}
    for value in values:
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            continue
        storage = value.untyped_storage()
        key = (value.device.type, value.device.index, storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def runtime_memory_report(model: nn.Module, reader: ArtifactReader) -> dict[str, Any]:
    """Exact live tensor-storage accounting for a loaded packed model.

    This intentionally reports tensor allocations, not allocator pool/reserved
    memory.  Context/KV/workspace peaks remain empty until a benchmark records
    them rather than being guessed from a weight-only load.
    """
    model_values = list(model.parameters()) + list(model.buffers())
    codebook_values: list[torch.Tensor] = []
    seen_books: set[int] = set()
    repack_bytes = 0
    for module in model.modules():
        if isinstance(module, (PackedTQ1Linear, PackedTQ1Embedding)):
            identity = id(module.codebook)
            if identity not in seen_books:
                seen_books.add(identity)
                codebook_values.extend(module.codebook.tables.values())
            report = getattr(module, "repack_report", None)
            if report is not None:
                repack_bytes += int(report["resident_repack_bytes"])
    resident = _unique_storage_bytes(model_values + codebook_values)
    sizes = reader.manifest["size_accounting"]
    estimated_stream = (sizes["payload_bytes"] + sizes["row_scale_bytes"]
                        + sizes["codebook_bytes"] + sizes["non_tq1_parameter_bytes"])
    return {
        **sizes,
        "backend_private_repack_bytes": repack_bytes,
        "resident_language_model_bytes": resident,
        "measured_tensor_storage_bytes": resident,
        "estimated_decode_weight_bytes_per_token": estimated_stream,
    }


def load_packed_model(artifact_dir: str | Path, *, activation_mode: str | None = None,
                      dtype: torch.dtype = torch.float32,
                      runtime_backend: str = "scalar_oracle",
                      native_impl: str = "auto",
                      native_routing_policy: NativeRoutingPolicy | None = None):
    """Instantiate the HF architecture and replace every target with packed oracle IO."""
    from transformers import AutoConfig, AutoModelForCausalLM

    reader = ArtifactReader(artifact_dir)
    reader.validate()
    config = AutoConfig.from_pretrained(reader.directory, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    non_tq1 = reader.non_tq1_state_dict(include_aliases=True)
    result = model.load_state_dict(non_tq1, strict=False)
    target_weights = {item["state_dict_name"] for item in reader.manifest["tensors"]}
    unexpected = set(result.unexpected_keys)
    if unexpected:
        raise ValueError(f"artifact has unexpected non-TQ1 state {sorted(unexpected)[:8]}")
    missing_non_targets = set(result.missing_keys) - target_weights
    # Tied output/embed state may be omitted by some HF architectures and is
    # restored by tie_weights; every other missing value is fatal.
    missing_non_targets -= {"lm_head.weight"}
    if missing_non_targets:
        raise ValueError(f"artifact lacks model state {sorted(missing_non_targets)[:8]}")
    registry = reader.registry()
    mode = activation_mode or reader.quant_spec.activation_mode
    if mode not in {"none", "a8_token", "a8_block256"}:
        raise ValueError("invalid packed runtime activation mode")
    if runtime_backend not in {"scalar_oracle", "native_cpu"}:
        raise ValueError("runtime_backend must be scalar_oracle or native_cpu")
    if runtime_backend == "native_cpu" and mode == "none":
        raise ValueError("native_cpu runtime does not support activation_mode=none")
    module_type = PackedTQ1Linear if runtime_backend == "scalar_oracle" else NativeCPUTQ1Linear
    shared_items = [item for item in reader.manifest["tensors"]
                    if item.get("consumer_kind", "linear") == "shared_embedding_head"]
    if len(shared_items) > 1:
        raise ValueError("runtime supports one shared embedding/output tensor")
    if shared_items:
        item = shared_items[0]
        aliases = [name for name, alias in reader.aliases.items()
                   if alias["target"] == item["state_dict_name"]]
        if aliases != ["lm_head.weight"]:
            raise ValueError("shared packed embedding requires exactly the lm_head alias")
        _, payload, scales = reader.tensor(item["state_dict_name"])
        parent, attribute = _parent_and_attr(model, item["module_path"])
        old = getattr(parent, attribute)
        if not isinstance(old, nn.Embedding) or list(old.weight.shape) != item["logical_shape"]:
            raise ValueError("shared packed target is not the configured input Embedding")
        shared_type = (PackedTQ1Embedding if runtime_backend == "scalar_oracle"
                       else NativeCPUTQ1Embedding)
        shared_kwargs = ({"impl": native_impl,
                          "routing_policy": native_routing_policy or NativeRoutingPolicy()}
                         if shared_type is NativeCPUTQ1Embedding else {})
        shared = shared_type(
            payload, item["profile"], registry[item["codebook_id"]],
            row_scales=scales, activation_mode=mode,
            state_dict_name=item["state_dict_name"], padding_idx=old.padding_idx,
            output_dtype=dtype,
            **shared_kwargs)
        setattr(parent, attribute, shared)
        head_parent, head_attribute = _parent_and_attr(model, "lm_head")
        setattr(head_parent, head_attribute, PackedTQ1OutputHead(shared))
    for item in reader.manifest["tensors"]:
        if item.get("consumer_kind", "linear") == "shared_embedding_head":
            continue
        _, payload, scales = reader.tensor(item["state_dict_name"])
        parent, attribute = _parent_and_attr(model, item["module_path"])
        old = getattr(parent, attribute)
        if not isinstance(old, nn.Linear) or old.bias is not None:
            raise ValueError(f"{item['module_path']}: artifact target is not a bias-free Linear")
        if list(old.weight.shape) != item["logical_shape"]:
            raise ValueError(f"{item['module_path']}: artifact/config shape mismatch")
        module_kwargs = ({"impl": native_impl,
                          "routing_policy": native_routing_policy or NativeRoutingPolicy()}
                         if module_type is NativeCPUTQ1Linear else {})
        setattr(parent, attribute, module_type(
            payload, item["profile"], registry[item["codebook_id"]],
            row_scales=scales, activation_mode=mode,
            state_dict_name=item["state_dict_name"], **module_kwargs))
    if not shared_items:
        model.tie_weights()
    reader.verify_model_aliases(model)
    model.eval().requires_grad_(False)
    model.config.quantization_config = {
        "quant_method": "tq1_v",
        "canonical_packed": True,
        "artifact_schema": reader.manifest["artifact_schema"],
        "quant_spec_sha256": reader.manifest["quant_spec_sha256"],
        "activation_mode": mode,
        "runtime_backend": runtime_backend,
    }
    model.tq1_memory_report = runtime_memory_report(model, reader)
    model.tq1_refresh_memory_report = lambda: runtime_memory_report(model, reader)
    return model, reader


@torch.no_grad()
def model_logits_parity(artifact_dir: str | Path, input_ids: torch.Tensor, *,
                        activation_mode: str | None = None,
                        runtime_backend: str = "scalar_oracle") -> dict[str, Any]:
    model, reader = load_packed_model(
        artifact_dir, activation_mode=activation_mode,
        runtime_backend=runtime_backend)
    logits = model(input_ids).logits
    if not torch.isfinite(logits).all():
        raise ValueError("packed scalar runtime produced nonfinite logits")
    return {
        "shape": list(logits.shape),
        "finite": True,
        "quant_spec_sha256": reader.manifest["quant_spec_sha256"],
        "logits_sha256": hashlib.sha256(
            logits.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
        "logits": logits,
    }
