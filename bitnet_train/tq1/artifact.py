"""Canonical schema-2 TQ1 artifact writer and fail-closed reader."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .codebook import Codebook, CodebookRegistry
from .evaluation import validate_quality_report
from .packing import layout, unpack_payload
from .spec import ARTIFACT_SCHEMA, FLOAT_PROFILES, QuantSpec, canonical_json


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    # Artifact dtypes are byte-sized or IEEE little-endian values. All supported
    # production hosts are little-endian; fail instead of silently hashing a
    # host-native big-endian spelling.
    if sys.byteorder != "little" and value.element_size() > 1:
        raise RuntimeError("canonical tensor hashing on big-endian hosts is not implemented")
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _shape_numel(shape: list[int] | tuple[int, ...]) -> int:
    return math.prod(shape)


def _dtype_element_size(name: str) -> int:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported logical dtype {name!r}")
    return torch.empty((), dtype=value).element_size()


@dataclass(frozen=True)
class TensorManifest:
    state_dict_name: str
    module_path: str
    logical_shape: list[int]
    profile: str
    codebook_id: str
    scale_mode: str
    scale_dtype: str | None
    block_bytes: int
    payload_key: str
    payload_sha256: str
    scale_key: str | None
    scale_sha256: str | None
    logical_dtype: str
    logical_kind: str
    consumer_kind: str


class ArtifactBuilder:
    """Accumulate and transactionally write one canonical artifact."""

    def __init__(self, quant_spec: QuantSpec, registry: CodebookRegistry, *,
                 source_model: str, source_revision: str,
                 tokenizer_sha256: str, chat_template_sha256: str,
                 provenance: Mapping[str, Any] | None = None):
        registry.validate_refs(quant_spec.codebooks)
        self.spec = quant_spec
        self.registry = registry
        self.source_model = source_model
        self.source_revision = source_revision
        self.tokenizer_sha256 = tokenizer_sha256
        self.chat_template_sha256 = chat_template_sha256
        self.provenance = dict(provenance or {})
        self.packed: dict[str, torch.Tensor] = {}
        self.non_tq1: dict[str, torch.Tensor] = {}
        self.non_tq1_kinds: dict[str, str] = {}
        self.aliases: dict[str, dict[str, Any]] = {}
        self._source_keys: dict[tuple[Any, ...], str] = {}
        self._raw_source_keys: dict[tuple[Any, ...], str] = {}
        self._target_source_keys: dict[str, tuple[Any, ...]] = {}
        self._source_values: dict[str, torch.Tensor] = {}
        self.tensors: list[TensorManifest] = []

    def _logical_names(self) -> set[str]:
        return ({item.state_dict_name for item in self.tensors}
                | set(self.non_tq1) | set(self.aliases))

    @staticmethod
    def _source_key(tensor: torch.Tensor, *, dtype: torch.dtype,
                    logical_kind: str) -> tuple[Any, ...] | None:
        """Identity for an exact shared storage view before canonical cloning."""
        value = tensor.detach()
        if value.layout != torch.strided or value.device.type == "meta" or value.numel() == 0:
            return None
        storage = value.untyped_storage()
        pointer = storage.data_ptr()
        if pointer == 0:
            return None
        return (
            value.device.type, value.device.index, pointer, storage.nbytes(),
            value.storage_offset(), tuple(value.shape), tuple(value.stride()),
            _dtype_name(value.dtype), value.is_conj(), value.is_neg(),
            _dtype_name(dtype), logical_kind,
        )

    def add_quantized(self, state_dict_name: str, module_path: str,
                      payload: torch.Tensor, *, logical_shape: tuple[int, ...],
                      profile: str, codebook_id: str,
                      row_scales: torch.Tensor | None = None,
                      source_tensor: torch.Tensor | None = None,
                      logical_kind: str = "parameter",
                      consumer_kind: str = "linear") -> None:
        if state_dict_name in self._logical_names():
            raise ValueError(f"duplicate target tensor {state_dict_name}")
        if logical_kind not in {"parameter", "buffer"}:
            raise ValueError("logical_kind must be parameter or buffer")
        if consumer_kind not in {"linear", "shared_embedding_head"}:
            raise ValueError("unknown quantized consumer kind")
        spec = layout(profile)
        if len(logical_shape) not in {2, 3}:
            raise ValueError("logical TQ1 weights must be [N,K] or [E,N,K]")
        if logical_shape[-1] % 256:
            raise ValueError("TQ1 input width must be divisible by 256")
        expected_payload = (*logical_shape[:-1], logical_shape[-1] // 256, spec.block_bytes)
        value = payload.detach().to(torch.uint8).contiguous().cpu()
        if tuple(value.shape) != expected_payload:
            raise ValueError(f"{state_dict_name}: payload shape {tuple(value.shape)} "
                             f"does not equal {expected_payload}")
        book = self.registry[codebook_id]
        expected_encoding = "direct_joint" if "-i-" in profile else \
            "product" if "-p-" in profile else "sign_canonical"
        if book.index_bits != spec.index_bits or book.encoding != expected_encoding:
            raise ValueError("tensor profile is incompatible with its codebook")
        indices, embedded_scales, _ = unpack_payload(value, profile)
        book.validate_indices(indices)
        payload_key = f"{state_dict_name}.__tq1_payload"
        scale_key = scale_hash = scale_dtype = None
        scale_value = None
        if spec.scale_mode == "row":
            if row_scales is None or tuple(row_scales.shape) != logical_shape[:-1]:
                raise ValueError(f"{state_dict_name}: row scale shape must be {logical_shape[:-1]}")
            if row_scales.dtype not in {torch.float16, torch.bfloat16}:
                raise ValueError("row scales must already be rounded to float16 or bfloat16")
            scale_value = row_scales.detach().contiguous().cpu()
            if not torch.isfinite(scale_value).all() or torch.any(scale_value < 0):
                raise ValueError("row scales must be finite and nonnegative")
            scale_key = f"{state_dict_name}.__tq1_scale"
            scale_hash = tensor_sha256(scale_value)
            scale_dtype = "float16" if scale_value.dtype == torch.float16 else "bfloat16"
        elif row_scales is not None or embedded_scales is None:
            raise ValueError("block-scale payload owns its scales")
        elif not torch.isfinite(embedded_scales).all() or torch.any(embedded_scales < 0):
            raise ValueError("embedded scales must be finite and nonnegative")
        self.packed[payload_key] = value
        if scale_key is not None and scale_value is not None:
            self.packed[scale_key] = scale_value
        self.tensors.append(TensorManifest(
            state_dict_name=state_dict_name,
            module_path=module_path,
            logical_shape=list(logical_shape),
            profile=profile,
            codebook_id=codebook_id,
            scale_mode=spec.scale_mode,
            scale_dtype=scale_dtype,
            block_bytes=spec.block_bytes,
            payload_key=payload_key,
            payload_sha256=tensor_sha256(value),
            scale_key=scale_key,
            scale_sha256=scale_hash,
            logical_dtype=_dtype_name(
                source_tensor.dtype if source_tensor is not None else torch.float32),
            logical_kind=logical_kind,
            consumer_kind=consumer_kind,
        ))
        if source_tensor is not None:
            if tuple(source_tensor.shape) != tuple(logical_shape):
                raise ValueError(f"{state_dict_name}: source tensor shape mismatch")
            source_key = self._source_key(
                source_tensor, dtype=source_tensor.dtype, logical_kind=logical_kind)
            if source_key is None:
                raise ValueError(
                    f"{state_dict_name}: a quantized alias target requires addressable storage")
            raw_source_key = source_key[:-2]
            if source_key in self._source_keys or raw_source_key in self._raw_source_keys:
                raise ValueError(f"{state_dict_name}: quantized physical target is already aliased")
            self._source_keys[source_key] = state_dict_name
            self._raw_source_keys[raw_source_key] = state_dict_name
            self._target_source_keys[state_dict_name] = source_key
            self._source_values[state_dict_name] = source_tensor.detach()

    def add_non_tq1(self, state_dict_name: str, tensor: torch.Tensor, *,
                    dtype: torch.dtype | None = None,
                    logical_kind: str = "parameter") -> None:
        if logical_kind not in {"parameter", "buffer"}:
            raise ValueError("logical_kind must be parameter or buffer")
        if state_dict_name in self._logical_names():
            raise ValueError(f"duplicate non-TQ1 tensor {state_dict_name}")
        requested_dtype = tensor.dtype if dtype is None else dtype
        source_key = self._source_key(
            tensor, dtype=requested_dtype, logical_kind=logical_kind)
        raw_source_key = None if source_key is None else source_key[:-2]
        if raw_source_key is not None and raw_source_key in self._raw_source_keys \
                and source_key not in self._source_keys:
            raise ValueError(
                f"{state_dict_name}: tied tensors require identical dtype and logical kind; "
                f"canonical target is {self._raw_source_keys[raw_source_key]}")
        if source_key is not None and source_key in self._source_keys:
            self.add_alias(
                state_dict_name, self._source_keys[source_key], tensor,
                dtype=requested_dtype, logical_kind=logical_kind)
            return
        value = tensor.detach().to(dtype=requested_dtype).contiguous().cpu().clone()
        if not torch.isfinite(value).all():
            raise ValueError(f"{state_dict_name}: non-TQ1 tensor is nonfinite")
        self.non_tq1[state_dict_name] = value
        self.non_tq1_kinds[state_dict_name] = logical_kind
        if source_key is not None:
            self._source_keys[source_key] = state_dict_name
            self._raw_source_keys[raw_source_key] = state_dict_name
            self._target_source_keys[state_dict_name] = source_key
            # Keep the source storage alive until artifact construction ends;
            # otherwise an allocator could reuse its pointer and create a
            # false alias for a later temporary tensor.
            self._source_values[state_dict_name] = tensor.detach()

    def add_alias(self, state_dict_name: str, target: str, tensor: torch.Tensor, *,
                  dtype: torch.dtype | None = None,
                  logical_kind: str = "parameter") -> None:
        """Add a logical name for an exact shared non-TQ1 storage view.

        Alias chains are deliberately not representable: every alias points to
        the one physical canonical tensor.  Requiring the source tensor here
        prevents equal-but-untied values from being silently deduplicated.
        """
        if logical_kind not in {"parameter", "buffer"}:
            raise ValueError("logical_kind must be parameter or buffer")
        if state_dict_name in self._logical_names():
            raise ValueError(f"duplicate alias tensor {state_dict_name}")
        if target in self.aliases:
            raise ValueError("tensor aliases must target physical tensors, not aliases")
        quantized = {item.state_dict_name: item for item in self.tensors}
        if target not in self.non_tq1 and target not in quantized:
            raise ValueError(f"alias target is not a physical tensor: {target}")
        requested_dtype = tensor.dtype if dtype is None else dtype
        source_key = self._source_key(
            tensor, dtype=requested_dtype, logical_kind=logical_kind)
        target_key = self._target_source_keys.get(target)
        if source_key is None or target_key is None or source_key != target_key:
            raise ValueError(
                f"{state_dict_name}: alias value is not the exact shared storage view of {target}")
        if target in self.non_tq1:
            target_shape = tuple(self.non_tq1[target].shape)
            target_dtype = self.non_tq1[target].dtype
            target_kind = self.non_tq1_kinds[target]
        else:
            target_item = quantized[target]
            target_shape = tuple(target_item.logical_shape)
            target_dtype = getattr(torch, target_item.logical_dtype)
            target_kind = target_item.logical_kind
        if tuple(tensor.shape) != target_shape:
            raise ValueError(f"{state_dict_name}: alias shape differs from {target}")
        if requested_dtype != target_dtype:
            raise ValueError(f"{state_dict_name}: alias dtype differs from {target}")
        if logical_kind != target_kind:
            raise ValueError(f"{state_dict_name}: alias kind differs from {target}")
        self.aliases[state_dict_name] = {
            "target": target,
            "shape": list(target_shape),
            "dtype": _dtype_name(target_dtype),
            "kind": logical_kind,
        }

    def _add_codebooks(self) -> None:
        for ref in self.spec.codebooks:
            book = self.registry[ref.id]
            prefix = f"__tq1_codebook.{ref.id}."
            if book.encoding == "sign_canonical":
                masks = book.tables["shapes_masks"]
                self.packed[prefix + "positive_mask"] = masks[:, 0].clone()
                self.packed[prefix + "negative_mask"] = masks[:, 1].clone()
            elif book.encoding == "direct_joint":
                self.packed[prefix + "joint_trits"] = book.tables["joint_trits"]
            else:
                self.packed[prefix + "product_a"] = book.tables["product_a"]
                self.packed[prefix + "product_b"] = book.tables["product_b"]

    def write(self, output_dir: str | Path, *, source_files: str | Path | None = None,
              quantization_report: Mapping[str, Any] | None = None,
              evaluation_report: Mapping[str, Any] | None = None,
              overwrite: bool = False) -> Path:
        destination = Path(output_dir).resolve()
        if destination.exists() and not overwrite:
            raise FileExistsError(f"artifact output already exists: {destination}")
        if not self.tensors:
            raise ValueError("canonical artifact contains no TQ1 tensors")
        self._add_codebooks()
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
        try:
            save_file(self.packed, temp / "tq1_packed.safetensors")
            save_file(self.non_tq1, temp / "non_tq1_model.safetensors")
            if source_files is not None:
                self._copy_model_files(Path(source_files), temp)
            if not (temp / "config.json").is_file():
                raise FileNotFoundError("canonical artifact requires config.json")
            if not any((temp / name).is_file() for name in (
                    "tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
                raise FileNotFoundError("canonical artifact requires tokenizer files")
            report = {
                "quant_spec": self.spec.to_dict(),
                "quant_spec_sha256": self.spec.sha256(),
                **dict(quantization_report or {}),
            }
            if report["quant_spec_sha256"] != self.spec.sha256() \
                    or report["quant_spec"] != self.spec.to_dict():
                raise ValueError("quantization report QuantSpec mismatch")
            (temp / "quantization_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
            if evaluation_report is not None:
                evaluation = {
                    "quant_spec_sha256": self.spec.sha256(),
                    **dict(evaluation_report),
                }
                validate_quality_report(evaluation, self.spec.sha256())
                (temp / "evaluation_report.json").write_text(
                    json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
            manifest = self._manifest(temp, evaluation_report is not None)
            # The manifest accounts for itself. Resolve the decimal-size fixed
            # point before writing it (only integer digit widths can change).
            for _ in range(8):
                rendered = json.dumps(
                    manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                manifest_bytes = len(rendered.encode("utf-8"))
                existing_bytes = sum(path.stat().st_size for path in temp.iterdir()
                                     if path.is_file())
                canonical_bytes = existing_bytes + manifest_bytes
                accounting = manifest["size_accounting"]
                if accounting["manifest_bytes"] == manifest_bytes \
                        and accounting["canonical_artifact_bytes"] == canonical_bytes:
                    break
                accounting["manifest_bytes"] = manifest_bytes
                accounting["canonical_artifact_bytes"] = canonical_bytes
                accounting["artifact_file_bytes"] = canonical_bytes
            else:  # pragma: no cover
                raise RuntimeError("canonical artifact byte accounting did not converge")
            (temp / "tq1_manifest.json").write_bytes(rendered.encode("utf-8"))
            ArtifactReader(temp).validate(require_evaluation=evaluation_report is not None)
            if destination.exists():
                if not overwrite:
                    raise FileExistsError(destination)
                backup = destination.with_name(destination.name + ".replaced")
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(destination, backup)
                try:
                    os.replace(temp, destination)
                except BaseException:
                    os.replace(backup, destination)
                    raise
                shutil.rmtree(backup)
            else:
                os.replace(temp, destination)
            return destination
        except BaseException:
            if temp.exists():
                shutil.rmtree(temp)
            raise

    @staticmethod
    def _copy_model_files(source: Path, destination: Path) -> None:
        names = {
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
            "chat_template.jinja", "added_tokens.json",
        }
        for path in source.iterdir():
            if path.is_file() and path.name in names:
                shutil.copy2(path, destination / path.name)

    def _manifest(self, directory: Path, quality_qualified: bool) -> dict[str, Any]:
        codebooks = []
        for ref in self.spec.codebooks:
            book = self.registry[ref.id]
            codebooks.append({
                **ref.to_dict(),
                "table_shapes": {name: list(value.shape) for name, value in book.tables.items()},
                "legal_index_count": int(book.legal_index_mask().sum()),
                "reserved_index_count": int((~book.legal_index_mask()).sum()),
                "duplicate_equivalence_classes": book.duplicate_equivalence_classes(),
                "provenance": dict(book.provenance),
            })
        packed_path = directory / "tq1_packed.safetensors"
        non_path = directory / "non_tq1_model.safetensors"
        transport_files = {
            path.name: _sha256_file(path)
            for path in sorted(directory.iterdir()) if path.is_file()
            and path.name != "tq1_manifest.json"
        }
        aliases = dict(sorted(self.aliases.items()))
        aliases_sha256 = hashlib.sha256(canonical_json(aliases).encode("utf-8")).hexdigest()
        resolved_policy: dict[str, dict[str, Any]] = {
            item.state_dict_name: {
                "profile": item.profile, "codebook_id": item.codebook_id,
                "storage": "tq1_packed.safetensors",
            }
            for item in self.tensors
        }
        target_patterns = tuple(re.compile(value) for value in self.spec.target_regexes)
        for state_name, value in self.non_tq1.items():
            if not state_name.endswith(".weight"):
                continue
            module_name = state_name.removesuffix(".weight")
            if not any(pattern.fullmatch(module_name) for pattern in target_patterns):
                continue
            profile, codebook_id = self.spec.resolve_profile(module_name)
            if profile not in FLOAT_PROFILES or codebook_id is not None:
                raise ValueError(f"{state_name}: unresolved TQ1 target is stored as non-TQ1")
            expected_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                              "fp32": torch.float32}[profile]
            if value.dtype != expected_dtype:
                raise ValueError(f"{state_name}: floating override dtype disagrees with policy")
            resolved_policy[state_name] = {
                "profile": profile, "codebook_id": None,
                "storage": "non_tq1_model.safetensors",
            }
        return {
            "artifact_schema": ARTIFACT_SCHEMA,
            "spec_revision": self.spec.spec_revision,
            "format_version": self.spec.format_version,
            "ggml_type_registry_revision": self.spec.ggml_type_registry_revision,
            "source_model": self.source_model,
            "source_revision": self.source_revision,
            "tokenizer_sha256": self.tokenizer_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "quant_spec": self.spec.to_dict(),
            "quant_spec_sha256": self.spec.sha256(),
            "codebooks": codebooks,
            "tensors": [asdict(item) for item in sorted(
                self.tensors, key=lambda item: item.state_dict_name)],
            "non_tq1_tensors": {
                name: {"shape": list(value.shape), "dtype": _dtype_name(value.dtype),
                       "kind": self.non_tq1_kinds[name], "sha256": tensor_sha256(value)}
                for name, value in sorted(self.non_tq1.items())
            },
            "tensor_aliases": aliases,
            "tensor_aliases_sha256": aliases_sha256,
            "resolved_tensor_policy": dict(sorted(resolved_policy.items())),
            "files": transport_files,
            "size_accounting": self._size_accounting(directory),
            "known_runtime_compatibility": dict(
                self.provenance.get("known_runtime_compatibility", {})),
            "quality_qualified": quality_qualified,
            "provenance": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "platform": platform.platform(),
                **self.provenance,
            },
            "optional_extensions": ["tensor_aliases", "tensor_aliases_sha256"],
        }

    def _size_accounting(self, directory: Path) -> dict[str, Any]:
        targeted_parameters = sum(_shape_numel(item.logical_shape) for item in self.tensors)
        payload_bytes = sum(_tensor_nbytes(self.packed[item.payload_key])
                            for item in self.tensors)
        row_scale_bytes = sum(
            0 if item.scale_key is None else _tensor_nbytes(self.packed[item.scale_key])
            for item in self.tensors)
        packed_code_bytes = embedded_scale_bytes = affine_bytes = ideal_code_bits = 0
        for item in self.tensors:
            profile_layout = layout(item.profile)
            blocks = _shape_numel(item.logical_shape[:-1]) * item.logical_shape[-1] // 256
            packed_code_bytes += blocks * profile_layout.raw_index_bytes
            # This field is intentionally the entropy-free fixed-width code cost
            # in bits, despite its historical name ending in `_bits`.
            ideal_code_bits += _shape_numel(item.logical_shape) * profile_layout.index_bits // 8
            if profile_layout.scale_mode == "block256":
                embedded_scale_bytes += blocks * 2
            if profile_layout.affine:
                affine_bytes += blocks * 4
        codebook_bytes = sum(
            _tensor_nbytes(value) for name, value in self.packed.items()
            if name.startswith("__tq1_codebook."))
        non_tq1_physical_bytes = sum(_tensor_nbytes(value)
                                     for value in self.non_tq1.values())
        high_precision_parameter_bytes = sum(
            _tensor_nbytes(value) for name, value in self.non_tq1.items()
            if self.non_tq1_kinds[name] == "parameter")
        high_precision_unique_parameters = sum(
            value.numel() for name, value in self.non_tq1.items()
            if self.non_tq1_kinds[name] == "parameter")
        high_precision_buffer_elements = sum(
            value.numel() for name, value in self.non_tq1.items()
            if self.non_tq1_kinds[name] == "buffer")
        alias_parameter_elements = sum(
            _shape_numel(meta["shape"]) for meta in self.aliases.values()
            if meta["kind"] == "parameter")
        alias_reference_bytes = sum(
            _shape_numel(meta["shape"]) * _dtype_element_size(meta["dtype"])
            for meta in self.aliases.values())
        packed_tensor_bytes = sum(_tensor_nbytes(value) for value in self.packed.values())
        packed_file_bytes = (directory / "tq1_packed.safetensors").stat().st_size
        non_tq1_file_bytes = (directory / "non_tq1_model.safetensors").stat().st_size
        container_overhead = (
            packed_file_bytes - packed_tensor_bytes
            + non_tq1_file_bytes - non_tq1_physical_bytes)
        if container_overhead < 0:  # pragma: no cover - safetensors invariant
            raise RuntimeError("safetensors files are smaller than their physical tensors")
        unique_parameters = targeted_parameters + high_precision_unique_parameters
        physical_weight_bytes = (payload_bytes + row_scale_bytes + codebook_bytes
                                 + non_tq1_physical_bytes)
        deployment = self.provenance.get("deployment_accounting", {})
        if not isinstance(deployment, Mapping):
            raise ValueError("deployment_accounting provenance must be an object")
        optional_components = deployment.get("optional_component_bytes", {})
        peak_contexts = deployment.get("peak_memory_bytes_by_context", {})
        if not isinstance(optional_components, Mapping) or not isinstance(peak_contexts, Mapping):
            raise ValueError("deployment component/context byte accounting must be objects")
        existing_bytes = sum(path.stat().st_size for path in directory.iterdir()
                             if path.is_file())
        return {
            "unique_logical_parameters": unique_parameters,
            "logical_parameter_references": unique_parameters + alias_parameter_elements,
            "low_bit_unique_parameters": targeted_parameters,
            "high_precision_unique_parameters": high_precision_unique_parameters,
            "high_precision_buffer_elements": high_precision_buffer_elements,
            "physical_tensor_count": len(self.packed) + len(self.non_tq1),
            "logical_tensor_references": len(self.tensors) + len(self.non_tq1)
            + len(self.aliases),
            "ideal_code_bits": ideal_code_bits,
            "ideal_code_bpw": ideal_code_bits / max(targeted_parameters, 1),
            "packed_code_bytes": packed_code_bytes,
            "embedded_scale_bytes": embedded_scale_bytes,
            "row_scale_bytes": row_scale_bytes,
            "scale_bytes": embedded_scale_bytes + row_scale_bytes,
            "affine_bytes": affine_bytes,
            "codebook_bytes": codebook_bytes,
            "alignment_bytes": 0,
            "payload_bytes": payload_bytes,
            "non_tq1_physical_bytes": non_tq1_physical_bytes,
            "non_tq1_parameter_bytes": high_precision_parameter_bytes,
            "non_tq1_logical_reference_bytes": (
                non_tq1_physical_bytes + alias_reference_bytes),
            "physical_model_storage_bytes": physical_weight_bytes,
            "target_effective_bpw": 8 * (payload_bytes + row_scale_bytes)
            / max(targeted_parameters, 1),
            "model_effective_bpw": 8 * physical_weight_bytes / max(unique_parameters, 1),
            "packed_file_bytes": packed_file_bytes,
            "non_tq1_file_bytes": non_tq1_file_bytes,
            "safetensors_container_overhead_bytes": container_overhead,
            "manifest_bytes": 0,
            "canonical_artifact_bytes": existing_bytes,
            # Backward-compatible spellings retained with corrected semantics.
            "targeted_parameters": targeted_parameters,
            "non_tq1_logical_bytes": non_tq1_physical_bytes + alias_reference_bytes,
            "artifact_file_bytes": existing_bytes,
            # Deployment measurements are nullable until a producer records them.
            "final_gguf_bytes": deployment.get("final_gguf_bytes"),
            "backend_private_repack_bytes": deployment.get("backend_private_repack_bytes"),
            "resident_language_model_bytes": deployment.get("resident_language_model_bytes"),
            "optional_component_bytes": dict(optional_components),
            "estimated_decode_weight_bytes_per_token": (
                payload_bytes + row_scale_bytes + codebook_bytes
                + high_precision_parameter_bytes),
            "measured_decode_weight_bytes_per_token": deployment.get(
                "measured_decode_weight_bytes_per_token"),
            "peak_memory_bytes_by_context": dict(peak_contexts),
        }


class ArtifactReader:
    REQUIRED = {
        "artifact_schema", "spec_revision", "format_version",
        "ggml_type_registry_revision", "source_model", "source_revision",
        "tokenizer_sha256", "chat_template_sha256", "quant_spec",
        "quant_spec_sha256", "codebooks", "tensors", "non_tq1_tensors",
        "resolved_tensor_policy",
        "files", "size_accounting", "known_runtime_compatibility",
        "quality_qualified", "provenance", "optional_extensions",
    }

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        path = self.directory / "tq1_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        self.manifest = json.loads(path.read_text())

    def validate(self, *, require_evaluation: bool = False) -> None:
        missing = self.REQUIRED - set(self.manifest)
        if missing:
            raise ValueError(f"manifest is missing required fields {sorted(missing)}")
        if self.manifest["artifact_schema"] != ARTIFACT_SCHEMA:
            raise ValueError("schema-1 and unknown artifact schemas are not accepted")
        extensions = self.manifest.get("optional_extensions")
        if not isinstance(extensions, list) or not all(isinstance(value, str)
                                                       for value in extensions):
            raise ValueError("optional_extensions must be a string list")
        unknown = set(self.manifest) - self.REQUIRED - set(extensions)
        if unknown:
            raise ValueError(f"manifest has unknown required fields {sorted(unknown)}")
        alias_fields = {"tensor_aliases", "tensor_aliases_sha256"}
        present_alias_fields = alias_fields & set(self.manifest)
        listed_alias_fields = alias_fields & set(extensions)
        if present_alias_fields != alias_fields or listed_alias_fields != alias_fields:
            if present_alias_fields or listed_alias_fields:
                raise ValueError("tensor alias extension must be present and listed as a pair")
        spec = QuantSpec.from_dict(self.manifest["quant_spec"])
        if spec.sha256() != self.manifest["quant_spec_sha256"]:
            raise ValueError("QuantSpec hash mismatch")
        if (self.manifest["spec_revision"] != spec.spec_revision
                or self.manifest["format_version"] != spec.format_version
                or self.manifest["ggml_type_registry_revision"]
                != spec.ggml_type_registry_revision):
            raise ValueError("manifest versions disagree with QuantSpec")
        packed_path = self.directory / "tq1_packed.safetensors"
        non_path = self.directory / "non_tq1_model.safetensors"
        required_paths = (packed_path, non_path, self.directory / "quantization_report.json",
                          self.directory / "config.json")
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        if not any((self.directory / name).is_file() for name in (
                "tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
            raise FileNotFoundError("canonical artifact lacks tokenizer files")
        if not isinstance(self.manifest["quality_qualified"], bool):
            raise ValueError("manifest quality_qualified must be boolean")
        evaluation_path = self.directory / "evaluation_report.json"
        if (require_evaluation or self.manifest["quality_qualified"]) \
                and not evaluation_path.is_file():
            raise FileNotFoundError("quality-qualified artifact lacks evaluation_report.json")
        if evaluation_path.is_file() != self.manifest["quality_qualified"]:
            raise ValueError("evaluation report presence and quality-qualified flag disagree")
        actual_files = {path.name for path in self.directory.iterdir()
                        if path.is_file() and path.name != "tq1_manifest.json"}
        if set(self.manifest["files"]) != actual_files:
            raise ValueError("artifact transport file inventory mismatch")
        for name, expected in self.manifest["files"].items():
            if _sha256_file(self.directory / name) != expected:
                raise ValueError(f"transport hash mismatch for {name}")
        report = json.loads((self.directory / "quantization_report.json").read_text())
        if report.get("quant_spec") != spec.to_dict() \
                or report.get("quant_spec_sha256") != spec.sha256():
            raise ValueError("quantization report QuantSpec mismatch")
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text())
            validate_quality_report(evaluation, spec.sha256())
        registry = self.registry()
        metadata_ids = [item.get("id") for item in self.manifest["codebooks"]]
        if metadata_ids != [item.id for item in spec.codebooks]:
            raise ValueError("manifest codebook order/inventory differs from QuantSpec")
        with safe_open(packed_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if any(key.endswith(".__tq1_indices") for key in keys):
                raise ValueError("schema-1 __tq1_indices key is forbidden")
            expected_tensor_keys: set[str] = set()
            state_names: set[str] = set()
            module_paths: set[str] = set()
            for item in self.manifest["tensors"]:
                expected_item_fields = {field.name for field in fields(TensorManifest)}
                if set(item) != expected_item_fields:
                    raise ValueError(
                        f"invalid quantized tensor manifest schema for "
                        f"{item.get('state_dict_name', '<unknown>')}")
                if item["state_dict_name"] in state_names or item["module_path"] in module_paths:
                    raise ValueError("canonical artifact has duplicate tensor identities")
                state_names.add(item["state_dict_name"])
                module_paths.add(item["module_path"])
                resolved = spec.resolve_profile(item["module_path"])
                if resolved != (item["profile"], item["codebook_id"]):
                    raise ValueError(f"{item['module_path']}: resolved tensor policy mismatch")
                profile_layout = layout(item["profile"])
                logical_shape = tuple(item["logical_shape"])
                if item["logical_dtype"] not in {
                        "float16", "bfloat16", "float32", "float64"}:
                    raise ValueError(f"invalid logical dtype for {item['state_dict_name']}")
                if item["logical_kind"] not in {"parameter", "buffer"}:
                    raise ValueError(f"invalid logical kind for {item['state_dict_name']}")
                if item["consumer_kind"] not in {"linear", "shared_embedding_head"}:
                    raise ValueError(f"invalid consumer kind for {item['state_dict_name']}")
                expected_payload_shape = (*logical_shape[:-1], logical_shape[-1] // 256,
                                          profile_layout.block_bytes)
                expected_tensor_keys.add(item["payload_key"])
                if item["scale_key"] is not None:
                    expected_tensor_keys.add(item["scale_key"])
                payload = handle.get_tensor(item["payload_key"])
                if payload.dtype != torch.uint8 or tuple(payload.shape) != expected_payload_shape:
                    raise ValueError(f"payload shape/dtype mismatch for {item['state_dict_name']}")
                if tensor_sha256(payload) != item["payload_sha256"]:
                    raise ValueError(f"payload hash mismatch for {item['state_dict_name']}")
                if item["scale_key"] is not None:
                    scale = handle.get_tensor(item["scale_key"])
                    expected_dtype = (torch.float16 if item["scale_dtype"] == "float16"
                                      else torch.bfloat16)
                    if scale.dtype != expected_dtype or tuple(scale.shape) != logical_shape[:-1]:
                        raise ValueError(f"scale shape/dtype mismatch for {item['state_dict_name']}")
                    if tensor_sha256(scale) != item["scale_sha256"]:
                        raise ValueError(f"scale hash mismatch for {item['state_dict_name']}")
                indices, embedded, _ = unpack_payload(payload, item["profile"])
                registry[item["codebook_id"]].validate_indices(indices)
                if profile_layout.scale_mode == "block256" and embedded is None:
                    raise ValueError("block-scale tensor has no embedded scales")
            codebook_keys = {key for key in keys if key.startswith("__tq1_codebook.")}
            if keys != expected_tensor_keys | codebook_keys:
                extra = keys - expected_tensor_keys - codebook_keys
                absent = expected_tensor_keys - keys
                raise ValueError(f"packed tensor inventory mismatch: extra={sorted(extra)}, "
                                 f"missing={sorted(absent)}")
        non_tensors = load_file(non_path, device="cpu")
        if set(non_tensors) != set(self.manifest["non_tq1_tensors"]):
            raise ValueError("non-TQ1 tensor inventory mismatch")
        for name, meta in self.manifest["non_tq1_tensors"].items():
            value = non_tensors[name]
            if list(value.shape) != meta["shape"] \
                    or str(value.dtype).removeprefix("torch.") != meta["dtype"]:
                raise ValueError(f"non-TQ1 shape/dtype mismatch for {name}")
            if tensor_sha256(non_tensors[name]) != meta["sha256"]:
                raise ValueError(f"non-TQ1 hash mismatch for {name}")
            kind = meta.get("kind", "parameter")
            if kind not in {"parameter", "buffer"}:
                raise ValueError(f"invalid non-TQ1 logical kind for {name}")

        aliases = self.manifest.get("tensor_aliases", {})
        if not isinstance(aliases, dict):
            raise ValueError("tensor_aliases must be an object")
        if present_alias_fields:
            alias_hash = hashlib.sha256(canonical_json(aliases).encode("utf-8")).hexdigest()
            if alias_hash != self.manifest["tensor_aliases_sha256"]:
                raise ValueError("tensor alias mapping hash mismatch")
        physical_names = set(non_tensors)
        quantized_meta = {
            item["state_dict_name"]: {
                "shape": item["logical_shape"], "dtype": item["logical_dtype"],
                "kind": item["logical_kind"],
            }
            for item in self.manifest["tensors"]
        }
        quantized_names = set(quantized_meta)
        overlap = set(aliases) & (physical_names | quantized_names)
        if overlap:
            raise ValueError(f"tensor aliases collide with physical tensors {sorted(overlap)}")
        for name, alias in aliases.items():
            if not isinstance(name, str) or not isinstance(alias, dict) \
                    or set(alias) != {"target", "shape", "dtype", "kind"}:
                raise ValueError(f"invalid tensor alias entry {name!r}")
            target = alias["target"]
            if target in aliases:
                raise ValueError(f"tensor alias chain or cycle is forbidden: {name} -> {target}")
            if target not in physical_names | quantized_names:
                raise ValueError(f"tensor alias target is missing: {name} -> {target}")
            target_meta = (self.manifest["non_tq1_tensors"][target]
                           if target in physical_names else quantized_meta[target])
            if alias["shape"] != target_meta["shape"]:
                raise ValueError(f"tensor alias shape mismatch for {name}")
            if alias["dtype"] != target_meta["dtype"]:
                raise ValueError(f"tensor alias dtype mismatch for {name}")
            target_kind = target_meta.get("kind", "parameter")
            if alias["kind"] != target_kind:
                raise ValueError(f"tensor alias kind mismatch for {name}")
        llama_head_alias = aliases.get("lm_head.weight")
        if llama_head_alias is not None \
                and llama_head_alias["target"] == "model.embed_tokens.weight":
            model_config = json.loads((self.directory / "config.json").read_text())
            if model_config.get("tie_word_embeddings") is not True:
                raise ValueError(
                    "Llama embedding/head alias requires tie_word_embeddings=true")

        policy = self.manifest["resolved_tensor_policy"]
        if not isinstance(policy, dict) or not policy:
            raise ValueError("resolved tensor policy must be a nonempty object")
        floating_names: set[str] = set()
        for state_name, item in policy.items():
            if not state_name.endswith(".weight") or not isinstance(item, dict) \
                    or set(item) != {"profile", "codebook_id", "storage"}:
                raise ValueError(f"invalid resolved tensor policy entry {state_name}")
            module_name = state_name.removesuffix(".weight")
            expected_profile, expected_book = spec.resolve_profile(module_name)
            if item["profile"] != expected_profile or item["codebook_id"] != expected_book:
                raise ValueError(f"resolved tensor policy disagrees with QuantSpec for {state_name}")
            if expected_profile in FLOAT_PROFILES:
                floating_names.add(state_name)
                if item["storage"] != "non_tq1_model.safetensors" or state_name not in non_tensors:
                    raise ValueError(f"floating override storage mismatch for {state_name}")
                expected_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                                  "fp32": torch.float32}[expected_profile]
                if non_tensors[state_name].dtype != expected_dtype:
                    raise ValueError(f"floating override dtype mismatch for {state_name}")
            elif item["storage"] != "tq1_packed.safetensors" \
                    or state_name not in quantized_names:
                raise ValueError(f"quantized policy storage mismatch for {state_name}")
        if set(policy) != quantized_names | floating_names:
            raise ValueError("resolved tensor policy inventory is inconsistent")
        if present_alias_fields:
            self._validate_size_accounting(non_tensors)

    @property
    def quant_spec(self) -> QuantSpec:
        return QuantSpec.from_dict(self.manifest["quant_spec"])

    def registry(self) -> CodebookRegistry:
        packed = load_file(self.directory / "tq1_packed.safetensors", device="cpu")
        books: dict[str, Codebook] = {}
        for meta in self.manifest["codebooks"]:
            codebook_id = meta["id"]
            prefix = f"__tq1_codebook.{codebook_id}."
            if meta["encoding"] == "sign_canonical":
                tables = {"shapes_masks": torch.stack((
                    packed[prefix + "positive_mask"], packed[prefix + "negative_mask"]), dim=1)}
            elif meta["encoding"] == "direct_joint":
                tables = {"joint_trits": packed[prefix + "joint_trits"]}
            else:
                tables = {"product_a": packed[prefix + "product_a"],
                          "product_b": packed[prefix + "product_b"]}
            book = Codebook(codebook_id, meta["format"], meta["encoding"], meta["scope"],
                            tables, meta.get("provenance", {}))
            if book.sha256() != meta["sha256"]:
                raise ValueError(f"codebook hash mismatch for {codebook_id}")
            books[codebook_id] = book
        registry = CodebookRegistry(books)
        registry.validate_refs(self.quant_spec.codebooks)
        return registry

    def tensor(self, state_dict_name: str) \
            -> tuple[dict[str, Any], torch.Tensor, torch.Tensor | None]:
        alias = self.aliases.get(state_dict_name)
        if alias is not None:
            state_dict_name = alias["target"]
        matches = [item for item in self.manifest["tensors"]
                   if item["state_dict_name"] == state_dict_name]
        if len(matches) != 1:
            raise KeyError(f"canonical artifact has {len(matches)} entries for {state_dict_name}")
        item = matches[0]
        packed = load_file(self.directory / "tq1_packed.safetensors", device="cpu")
        payload = packed[item["payload_key"]]
        scale = None if item["scale_key"] is None else packed[item["scale_key"]]
        return item, payload, scale

    @property
    def aliases(self) -> dict[str, dict[str, Any]]:
        return dict(self.manifest.get("tensor_aliases", {}))

    def non_tq1_state_dict(self, *, include_aliases: bool = True) \
            -> dict[str, torch.Tensor]:
        state = load_file(self.directory / "non_tq1_model.safetensors", device="cpu")
        if include_aliases:
            for name, alias in self.aliases.items():
                # Deliberately assign the same Python tensor object.  Equal
                # clones would defeat the alias contract before model loading.
                if alias["target"] in state:
                    state[name] = state[alias["target"]]
        return state

    def verify_model_aliases(self, model: torch.nn.Module) -> None:
        state = model.state_dict(keep_vars=True)
        quantized = {item["state_dict_name"]: item for item in self.manifest["tensors"]}
        for name, alias in self.aliases.items():
            target = alias["target"]
            if target in quantized:
                if not name.endswith(".weight") or not target.endswith(".weight"):
                    raise ValueError("quantized aliases currently require module weights")
                alias_module_name = name.removesuffix(".weight")
                target_module_name = target.removesuffix(".weight")
                alias_module = model.get_submodule(alias_module_name)
                target_module = model.get_submodule(target_module_name)
                if getattr(alias_module, "shared_weight", None) is not target_module:
                    raise ValueError(
                        f"model did not restore shared packed consumer: {name} -> {target}")
                if list(getattr(target_module, "logical_shape", alias["shape"])) \
                        != alias["shape"]:
                    raise ValueError(f"restored quantized alias metadata differs for {name}")
            else:
                if name not in state or target not in state:
                    raise ValueError(f"model does not expose tensor alias {name} -> {target}")
                if state[name] is not state[target]:
                    raise ValueError(
                        f"model did not restore Python-level tensor tying: {name} -> {target}")
                # Runtime dtype overrides may cast the one physical parameter; the
                # artifact-level alias dtype was already validated before loading.
                if list(state[name].shape) != alias["shape"]:
                    raise ValueError(f"restored model alias metadata differs for {name}")

    def _validate_size_accounting(self,
                                  non_tensors: Mapping[str, torch.Tensor]) -> None:
        accounting = self.manifest["size_accounting"]
        if not isinstance(accounting, dict):
            raise ValueError("size_accounting must be an object")
        packed = load_file(self.directory / "tq1_packed.safetensors", device="cpu")
        tensors = self.manifest["tensors"]
        aliases = self.aliases
        targeted = sum(_shape_numel(item["logical_shape"]) for item in tensors)
        payload_bytes = sum(_tensor_nbytes(packed[item["payload_key"]])
                            for item in tensors)
        row_scale_bytes = sum(
            0 if item["scale_key"] is None else _tensor_nbytes(packed[item["scale_key"]])
            for item in tensors)
        packed_code_bytes = embedded_scale_bytes = affine_bytes = ideal_code_bits = 0
        for item in tensors:
            profile_layout = layout(item["profile"])
            shape = item["logical_shape"]
            blocks = _shape_numel(shape[:-1]) * shape[-1] // 256
            packed_code_bytes += blocks * profile_layout.raw_index_bytes
            ideal_code_bits += _shape_numel(shape) * profile_layout.index_bits // 8
            if profile_layout.scale_mode == "block256":
                embedded_scale_bytes += blocks * 2
            if profile_layout.affine:
                affine_bytes += blocks * 4
        codebook_bytes = sum(
            _tensor_nbytes(value) for name, value in packed.items()
            if name.startswith("__tq1_codebook."))
        non_physical_bytes = sum(_tensor_nbytes(value) for value in non_tensors.values())
        non_meta = self.manifest["non_tq1_tensors"]
        high_parameters = sum(
            value.numel() for name, value in non_tensors.items()
            if non_meta[name].get("kind", "parameter") == "parameter")
        high_parameter_bytes = sum(
            _tensor_nbytes(value) for name, value in non_tensors.items()
            if non_meta[name].get("kind", "parameter") == "parameter")
        buffer_elements = sum(
            value.numel() for name, value in non_tensors.items()
            if non_meta[name].get("kind", "parameter") == "buffer")
        alias_parameter_elements = sum(
            _shape_numel(meta["shape"]) for meta in aliases.values()
            if meta["kind"] == "parameter")
        alias_reference_bytes = sum(
            _shape_numel(meta["shape"]) * _dtype_element_size(meta["dtype"])
            for meta in aliases.values())
        packed_tensor_bytes = sum(_tensor_nbytes(value) for value in packed.values())
        packed_file_bytes = (self.directory / "tq1_packed.safetensors").stat().st_size
        non_file_bytes = (self.directory / "non_tq1_model.safetensors").stat().st_size
        container_overhead = (packed_file_bytes - packed_tensor_bytes
                              + non_file_bytes - non_physical_bytes)
        canonical_bytes = sum(path.stat().st_size for path in self.directory.iterdir()
                              if path.is_file())
        manifest_bytes = (self.directory / "tq1_manifest.json").stat().st_size
        unique_parameters = targeted + high_parameters
        physical_model_bytes = (payload_bytes + row_scale_bytes + codebook_bytes
                                + non_physical_bytes)
        exact = {
            "unique_logical_parameters": unique_parameters,
            "logical_parameter_references": unique_parameters + alias_parameter_elements,
            "low_bit_unique_parameters": targeted,
            "high_precision_unique_parameters": high_parameters,
            "high_precision_buffer_elements": buffer_elements,
            "physical_tensor_count": len(packed) + len(non_tensors),
            "logical_tensor_references": len(tensors) + len(non_tensors) + len(aliases),
            "ideal_code_bits": ideal_code_bits,
            "packed_code_bytes": packed_code_bytes,
            "embedded_scale_bytes": embedded_scale_bytes,
            "row_scale_bytes": row_scale_bytes,
            "scale_bytes": embedded_scale_bytes + row_scale_bytes,
            "affine_bytes": affine_bytes,
            "codebook_bytes": codebook_bytes,
            "alignment_bytes": 0,
            "payload_bytes": payload_bytes,
            "non_tq1_physical_bytes": non_physical_bytes,
            "non_tq1_parameter_bytes": high_parameter_bytes,
            "non_tq1_logical_reference_bytes": non_physical_bytes + alias_reference_bytes,
            "physical_model_storage_bytes": physical_model_bytes,
            "packed_file_bytes": packed_file_bytes,
            "non_tq1_file_bytes": non_file_bytes,
            "safetensors_container_overhead_bytes": container_overhead,
            "manifest_bytes": manifest_bytes,
            "canonical_artifact_bytes": canonical_bytes,
            "targeted_parameters": targeted,
            "non_tq1_logical_bytes": non_physical_bytes + alias_reference_bytes,
            "artifact_file_bytes": canonical_bytes,
            "estimated_decode_weight_bytes_per_token": (
                payload_bytes + row_scale_bytes + codebook_bytes + high_parameter_bytes),
        }
        for name, expected in exact.items():
            value = accounting.get(name)
            if isinstance(value, bool) or value != expected:
                raise ValueError(
                    f"size accounting mismatch for {name}: manifest={value}, actual={expected}")
        floating = {
            "ideal_code_bpw": ideal_code_bits / max(targeted, 1),
            "target_effective_bpw": 8 * (payload_bytes + row_scale_bytes)
            / max(targeted, 1),
            "model_effective_bpw": 8 * physical_model_bytes / max(unique_parameters, 1),
        }
        for name, expected in floating.items():
            value = accounting.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(f"size accounting mismatch for {name}")
        nullable_integer_bytes = {
            "final_gguf_bytes", "backend_private_repack_bytes",
            "resident_language_model_bytes",
        }
        for name in nullable_integer_bytes:
            value = accounting.get(name)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, int) or value < 0):
                raise ValueError(
                    f"size accounting field {name} must be nonnegative integer bytes or null")
        measured_decode = accounting.get("measured_decode_weight_bytes_per_token")
        if measured_decode is not None and (
                isinstance(measured_decode, bool)
                or not isinstance(measured_decode, (int, float)) or measured_decode < 0):
            raise ValueError(
                "measured decode weight bytes must be nonnegative numeric bytes or null")
        for name in ("optional_component_bytes", "peak_memory_bytes_by_context"):
            value = accounting.get(name)
            if not isinstance(value, dict) or any(
                    not isinstance(key, str) or isinstance(count, bool)
                    or not isinstance(count, int) or count < 0
                    for key, count in value.items()):
                raise ValueError(
                    f"size accounting field {name} must map names to integer bytes")
