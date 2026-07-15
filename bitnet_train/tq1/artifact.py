"""Canonical schema-2 TQ1 artifact writer and fail-closed reader."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .codebook import Codebook, CodebookRegistry
from .evaluation import validate_quality_report
from .packing import layout, unpack_payload
from .spec import ARTIFACT_SCHEMA, FLOAT_PROFILES, QuantSpec


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
        self.tensors: list[TensorManifest] = []

    def add_quantized(self, state_dict_name: str, module_path: str,
                      payload: torch.Tensor, *, logical_shape: tuple[int, ...],
                      profile: str, codebook_id: str,
                      row_scales: torch.Tensor | None = None) -> None:
        if any(item.state_dict_name == state_dict_name for item in self.tensors):
            raise ValueError(f"duplicate target tensor {state_dict_name}")
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
        ))

    def add_non_tq1(self, state_dict_name: str, tensor: torch.Tensor) -> None:
        if state_dict_name in self.non_tq1 or state_dict_name in self.packed:
            raise ValueError(f"duplicate non-TQ1 tensor {state_dict_name}")
        value = tensor.detach().contiguous().cpu().clone()
        if not torch.isfinite(value).all():
            raise ValueError(f"{state_dict_name}: non-TQ1 tensor is nonfinite")
        self.non_tq1[state_dict_name] = value

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
            (temp / "tq1_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
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
        targeted_parameters = sum(
            __import__("math").prod(item.logical_shape) for item in self.tensors)
        payload_bytes = sum(self.packed[item.payload_key].numel() for item in self.tensors)
        row_scale_bytes = sum(
            0 if item.scale_key is None else
            self.packed[item.scale_key].numel() * self.packed[item.scale_key].element_size()
            for item in self.tensors)
        non_tq1_bytes = sum(value.numel() * value.element_size()
                            for value in self.non_tq1.values())
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
                name: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
                       "sha256": tensor_sha256(value)}
                for name, value in sorted(self.non_tq1.items())
            },
            "resolved_tensor_policy": dict(sorted(resolved_policy.items())),
            "files": transport_files,
            "size_accounting": {
                "targeted_parameters": targeted_parameters,
                "payload_bytes": payload_bytes,
                "row_scale_bytes": row_scale_bytes,
                "non_tq1_logical_bytes": non_tq1_bytes,
                "target_effective_bpw": 8 * (payload_bytes + row_scale_bytes)
                / max(targeted_parameters, 1),
                "artifact_file_bytes": sum(path.stat().st_size for path in directory.iterdir()
                                           if path.is_file()),
            },
            "known_runtime_compatibility": dict(
                self.provenance.get("known_runtime_compatibility", {})),
            "quality_qualified": quality_qualified,
            "provenance": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "platform": platform.platform(),
                **self.provenance,
            },
            "optional_extensions": [],
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
                if item["state_dict_name"] in state_names or item["module_path"] in module_paths:
                    raise ValueError("canonical artifact has duplicate tensor identities")
                state_names.add(item["state_dict_name"])
                module_paths.add(item["module_path"])
                resolved = spec.resolve_profile(item["module_path"])
                if resolved != (item["profile"], item["codebook_id"]):
                    raise ValueError(f"{item['module_path']}: resolved tensor policy mismatch")
                profile_layout = layout(item["profile"])
                logical_shape = tuple(item["logical_shape"])
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

        policy = self.manifest["resolved_tensor_policy"]
        if not isinstance(policy, dict) or not policy:
            raise ValueError("resolved tensor policy must be a nonempty object")
        quantized_names = {item["state_dict_name"] for item in self.manifest["tensors"]}
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
        matches = [item for item in self.manifest["tensors"]
                   if item["state_dict_name"] == state_dict_name]
        if len(matches) != 1:
            raise KeyError(f"canonical artifact has {len(matches)} entries for {state_dict_name}")
        item = matches[0]
        packed = load_file(self.directory / "tq1_packed.safetensors", device="cpu")
        payload = packed[item["payload_key"]]
        scale = None if item["scale_key"] is None else packed[item["scale_key"]]
        return item, payload, scale
