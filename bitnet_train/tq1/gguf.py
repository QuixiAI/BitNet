"""Exact canonical-artifact to GGUF exporter and byte-level validator."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .artifact import ArtifactReader
from .gguf_io import (
    BOOL, STRING, UINT32, ParsedGGUF, TensorRecord, parse_gguf,
    write_rewritten_gguf)
from .pipeline import bake_debug_checkpoint
from .spec import FLOAT_PROFILES, canonical_json

GGML_TYPES = {
    "tq1_v11-j-b": 43,
    "tq1_v12-j-b": 44,
    "tq1_v11-j-r": 45,
    "tq1_v11-i-r": 45,
    "tq1_v11-p-r": 45,
    "tq1_v12-j-r": 46,
    "tq1_v12-p-r": 46,
    "tq1_v11-j-a4-r": 47,
}


def hf_to_gguf_name(name: str) -> str:
    """Exact primary-Llama mapping; unsupported state names fail closed."""
    fixed = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    if name in fixed:
        return fixed[name]
    import re
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.*)", name)
    if match is None:
        raise KeyError(name)
    layer, suffix = match.groups()
    mapped = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }.get(suffix)
    if mapped is None:
        raise KeyError(name)
    return f"blk.{layer}.{mapped}"


def _row_permutation(row_count: int, head_count: int) -> torch.Tensor:
    if head_count < 1 or row_count % (2 * head_count):
        raise ValueError("q/k rows are incompatible with the declared attention heads")
    return torch.arange(row_count).reshape(
        head_count, 2, row_count // head_count // 2).transpose(1, 2).reshape(-1)


def _permuted_tensor(item: Mapping[str, Any], payload: torch.Tensor,
                     scales: torch.Tensor | None, base: ParsedGGUF) \
        -> tuple[torch.Tensor, torch.Tensor | None]:
    gguf_name = hf_to_gguf_name(item["state_dict_name"])
    if ".attn_q." in gguf_name:
        heads = int(base.metadata.get("llama.attention.head_count", 0))
    elif ".attn_k." in gguf_name:
        heads = int(base.metadata.get(
            "llama.attention.head_count_kv",
            base.metadata.get("llama.attention.head_count", 0)))
    else:
        return payload, scales
    order = _row_permutation(item["logical_shape"][-2], heads)
    return payload[order].contiguous(), None if scales is None else scales[order].contiguous()


def _permuted_dense(state_dict_name: str, value: torch.Tensor,
                    base: ParsedGGUF) -> torch.Tensor:
    gguf_name = hf_to_gguf_name(state_dict_name)
    if ".attn_q." in gguf_name:
        heads = int(base.metadata.get("llama.attention.head_count", 0))
    elif ".attn_k." in gguf_name:
        heads = int(base.metadata.get(
            "llama.attention.head_count_kv",
            base.metadata.get("llama.attention.head_count", 0)))
    else:
        return value.contiguous()
    return value[_row_permutation(value.shape[-2], heads)].contiguous()


def _floating_overrides(reader: ArtifactReader) -> dict[str, tuple[str, str]]:
    return {
        hf_to_gguf_name(state_name): (state_name, item["profile"])
        for state_name, item in reader.manifest["resolved_tensor_policy"].items()
        if item["profile"] in FLOAT_PROFILES
    }


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes(order="C")


def _codebook_records(reader: ArtifactReader) -> list[TensorRecord]:
    records = []
    for reference in reader.quant_spec.codebooks:
        book = reader.registry()[reference.id]
        prefix = f"__tq1_codebook.{reference.id}."
        if book.encoding == "sign_canonical":
            masks = book.tables["shapes_masks"]
            records.extend((
                TensorRecord(prefix + "positive_mask", (masks.shape[0],), 24,
                             _tensor_bytes(masks[:, 0])),
                TensorRecord(prefix + "negative_mask", (masks.shape[0],), 24,
                             _tensor_bytes(masks[:, 1])),
            ))
        elif book.encoding == "direct_joint":
            table = book.tables["joint_trits"]
            records.append(TensorRecord(
                prefix + "joint_trits", (table.shape[1], table.shape[0]), 24,
                _tensor_bytes(table)))
        else:
            for table_name in ("product_a", "product_b"):
                table = book.tables[table_name]
                records.append(TensorRecord(
                    prefix + table_name, (table.shape[1], table.shape[0]), 24,
                    _tensor_bytes(table)))
    return records


def rewrite_base_gguf(artifact_dir: str | Path, base_gguf: str | Path,
                      output_gguf: str | Path) -> dict[str, Any]:
    reader = ArtifactReader(artifact_dir)
    reader.validate()
    base = parse_gguf(base_gguf)
    if base.metadata.get("general.architecture") != "llama":
        raise ValueError("TQ1 revision-1 exporter supports the primary Llama architecture")
    base_tensors = {item.name: item for item in base.tensors}
    target_names = {
        hf_to_gguf_name(item["state_dict_name"]): item
        for item in reader.manifest["tensors"]
    }
    if len(target_names) != len(reader.manifest["tensors"]):
        raise ValueError("multiple canonical targets map to the same GGUF tensor")
    absent = set(target_names) - set(base_tensors)
    if absent:
        raise ValueError(f"ordinary converter omitted TQ1 targets {sorted(absent)}")
    floating = _floating_overrides(reader)
    absent_float = set(floating) - set(base_tensors)
    if absent_float:
        raise ValueError(f"ordinary converter omitted floating overrides {sorted(absent_float)}")
    from safetensors.torch import load_file
    non_tq1 = load_file(
        str(reader.directory / "non_tq1_model.safetensors"), device="cpu")
    records: list[TensorRecord] = []
    policy: dict[str, Any] = {}
    scale_records: list[TensorRecord] = []
    for original in base.tensors:
        item = target_names.get(original.name)
        if item is None:
            if original.name in floating:
                state_name, profile = floating[original.name]
                value = _permuted_dense(state_name, non_tq1[state_name], base)
                logical = tuple(value.shape)
                if original.dimensions != (logical[-1], logical[-2]):
                    raise ValueError(f"{original.name}: floating override shape mismatch")
                dense_type = {"fp32": 0, "fp16": 1, "bf16": 30}[profile]
                records.append(TensorRecord(
                    original.name, original.dimensions, dense_type,
                    _tensor_bytes(value)))
                continue
            records.append(original)
            continue
        _, payload, scales = reader.tensor(item["state_dict_name"])
        payload, scales = _permuted_tensor(item, payload, scales, base)
        logical = tuple(item["logical_shape"])
        if original.dimensions != (logical[-1], logical[-2]):
            raise ValueError(f"{original.name}: converter/artifact logical shape mismatch")
        ggml_type = GGML_TYPES[item["profile"]]
        records.append(TensorRecord(
            original.name, original.dimensions, ggml_type, _tensor_bytes(payload)))
        scale_name = None
        if scales is not None:
            scale_name = original.name.removesuffix(".weight") + ".scale"
            scale_type = 1 if scales.dtype == torch.float16 else 30
            scale_records.append(TensorRecord(
                scale_name, tuple(reversed(scales.shape)), scale_type,
                _tensor_bytes(scales)))
        policy[original.name] = {
            "profile": item["profile"],
            "codebook_id": item["codebook_id"],
            "logical_shape": item["logical_shape"],
            "block_bytes": item["block_bytes"],
            "companion_scale_name": scale_name,
        }
    records.extend(scale_records)
    codebook_records = _codebook_records(reader)
    records.extend(codebook_records)
    spec = reader.quant_spec
    codebook_meta = {item["id"]: item for item in reader.manifest["codebooks"]}
    metadata: dict[str, tuple[Any, int | None]] = {
        "tq1.spec_revision": (spec.spec_revision, STRING),
        "tq1.format_version": (spec.format_version, UINT32),
        "tq1.ggml_type_registry_revision": (spec.ggml_type_registry_revision, UINT32),
        "tq1.quant_spec_json": (spec.canonical_json(), STRING),
        "tq1.quant_spec_sha256": (spec.sha256(), STRING),
        "tq1.tensor_policy_json": (canonical_json(policy), STRING),
        "tq1.codebook.count": (len(spec.codebooks), UINT32),
        "tq1.codebook.ids": ([book.id for book in spec.codebooks], None),
        "tq1.activation_mode": (spec.activation_mode, STRING),
        "tq1.strict_ternary": (
            not any("-a4-" in item["profile"] for item in reader.manifest["tensors"]),
            BOOL),
        "tq1.source_model": (reader.manifest["source_model"], STRING),
        "tq1.source_revision": (reader.manifest["source_revision"], STRING),
    }
    for reference in spec.codebooks:
        item = codebook_meta[reference.id]
        prefix = f"tq1.codebook.{reference.id}."
        metadata.update({
            prefix + "encoding": (reference.encoding, STRING),
            prefix + "index_format": (reference.format, STRING),
            prefix + "sha256": (reference.sha256, STRING),
            prefix + "table_shapes_json": (canonical_json(item["table_shapes"]), STRING),
            prefix + "legal_index_count": (item["legal_index_count"], UINT32),
            prefix + "reserved_index_count": (item["reserved_index_count"], UINT32),
        })
    # Redundant, ordered binding arrays let runtimes bind a model-local
    # codebook without embedding a JSON parser in the tensor backend.  The
    # normative tensor_policy_json remains the canonical policy record.
    binding_names = list(policy)
    metadata.update({
        "tq1.tensor.names": (binding_names, None),
        "tq1.tensor.codebook_ids": (
            [policy[name]["codebook_id"] for name in binding_names], None),
        "tq1.tensor.profiles": (
            [policy[name]["profile"] for name in binding_names], None),
        "tq1.tensor.scale_names": (
            [policy[name]["companion_scale_name"] or "" for name in binding_names], None),
    })
    write_rewritten_gguf(base, output_gguf, records, metadata)
    return {
        "target_tensors": len(target_names),
        "scale_tensors": len(scale_records),
        "codebook_tensors": len(codebook_records),
        "floating_override_tensors": len(floating),
        "tensor_policy": policy,
    }


def validate_tq1_gguf(artifact_dir: str | Path, gguf_path: str | Path) -> dict[str, Any]:
    reader = ArtifactReader(artifact_dir)
    reader.validate()
    gguf = parse_gguf(gguf_path)
    spec = reader.quant_spec
    required_metadata = {
        "tq1.spec_revision": spec.spec_revision,
        "tq1.format_version": spec.format_version,
        "tq1.ggml_type_registry_revision": spec.ggml_type_registry_revision,
        "tq1.quant_spec_json": spec.canonical_json(),
        "tq1.quant_spec_sha256": spec.sha256(),
        "tq1.activation_mode": spec.activation_mode,
        "tq1.source_model": reader.manifest["source_model"],
        "tq1.source_revision": reader.manifest["source_revision"],
    }
    for key, expected in required_metadata.items():
        if gguf.metadata.get(key) != expected:
            raise ValueError(f"GGUF metadata mismatch for {key}")
    tensors = {item.name: item for item in gguf.tensors}
    policy = json.loads(gguf.metadata["tq1.tensor_policy_json"])
    expected_names = {hf_to_gguf_name(item["state_dict_name"])
                      for item in reader.manifest["tensors"]}
    if set(policy) != expected_names:
        raise ValueError("GGUF tensor policy inventory differs from canonical targets")
    binding_names = list(gguf.metadata.get("tq1.tensor.names", ()))
    if (len(binding_names) != len(set(binding_names)) or set(binding_names) != expected_names or
            gguf.metadata.get("tq1.tensor.codebook_ids") !=
            [policy[name]["codebook_id"] for name in binding_names] or
            gguf.metadata.get("tq1.tensor.profiles") !=
            [policy[name]["profile"] for name in binding_names] or
            gguf.metadata.get("tq1.tensor.scale_names") !=
            [policy[name]["companion_scale_name"] or "" for name in binding_names]):
        raise ValueError("GGUF runtime binding arrays differ from tensor policy")
    observed_custom = {item.name for item in gguf.tensors if item.tensor_type in set(GGML_TYPES.values())}
    if observed_custom != expected_names:
        raise ValueError("GGUF custom-type inventory differs from canonical targets")
    base = gguf
    for item in reader.manifest["tensors"]:
        name = hf_to_gguf_name(item["state_dict_name"])
        tensor = tensors[name]
        if tensor.tensor_type != GGML_TYPES[item["profile"]]:
            raise ValueError(f"{name}: GGML type mismatch")
        _, payload, scales = reader.tensor(item["state_dict_name"])
        payload, scales = _permuted_tensor(item, payload, scales, base)
        if tensor.data != _tensor_bytes(payload):
            raise ValueError(f"{name}: packed payload mismatch")
        scale_name = policy[name]["companion_scale_name"]
        if scales is None:
            if scale_name is not None:
                raise ValueError(f"{name}: unexpected row scale policy")
        elif scale_name not in tensors or tensors[scale_name].data != _tensor_bytes(scales):
            raise ValueError(f"{name}: row scale mismatch")
    from safetensors.torch import load_file
    non_tq1 = load_file(
        str(reader.directory / "non_tq1_model.safetensors"), device="cpu")
    for name, (state_name, profile) in _floating_overrides(reader).items():
        expected_type = {"fp32": 0, "fp16": 1, "bf16": 30}[profile]
        if name not in tensors or tensors[name].tensor_type != expected_type:
            raise ValueError(f"{name}: floating override type mismatch")
        value = _permuted_dense(state_name, non_tq1[state_name], gguf)
        if tensors[name].data != _tensor_bytes(value):
            raise ValueError(f"{name}: floating override payload mismatch")
    for record in _codebook_records(reader):
        if record.name not in tensors or tensors[record.name].data != record.data:
            raise ValueError(f"GGUF codebook tensor mismatch for {record.name}")
    digest = hashlib.sha256(Path(gguf_path).read_bytes()).hexdigest()
    return {
        "ok": True,
        "gguf_sha256": digest,
        "quant_spec_sha256": spec.sha256(),
        "target_tensors": len(expected_names),
        "tensor_count": len(tensors),
        "profile_coverage": sorted({item["profile"] for item in reader.manifest["tensors"]}),
    }


def export_tq1_gguf(artifact_dir: str | Path, output_gguf: str | Path, *,
                    converter: str | Path | None = None,
                    python: str = sys.executable, overwrite: bool = False,
                    command: Sequence[str] = ()) -> dict[str, Any]:
    output = Path(output_gguf).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    converter_path = Path(converter) if converter else \
        repository / "3rdparty" / "llama.cpp" / "convert_hf_to_gguf.py"
    if not converter_path.is_file():
        raise FileNotFoundError(converter_path)
    temp_dir = Path(tempfile.mkdtemp(prefix=".tq1-gguf-", dir=output.parent))
    temp_output = temp_dir / output.name
    try:
        baked = bake_debug_checkpoint(artifact_dir, temp_dir / "baked")
        # The baked checkpoint is a private staging input for the ordinary
        # architecture/tokenizer converter. Its debug quantization marker is
        # intentionally not a Transformers/llama.cpp quantizer and would make
        # a correct generic converter reject the directory. Remove it only
        # from this temporary copy; canonical indices/scales below still
        # replace every TQ1 target byte-for-byte.
        baked_config_path = baked / "config.json"
        baked_config = json.loads(baked_config_path.read_text())
        marker = baked_config.pop("quantization_config", None)
        if (not isinstance(marker, dict) or
                marker.get("quant_method") != "tq1_v_debug_baked"):
            raise ValueError("temporary baked checkpoint lacks its TQ1 debug marker")
        baked_config_path.write_text(
            json.dumps(baked_config, indent=2, sort_keys=True) + "\n")
        base = temp_dir / "base-f16.gguf"
        process = subprocess.run(
            [python, str(converter_path), str(baked), "--outfile", str(base),
             "--outtype", "f16"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise RuntimeError("ordinary GGUF conversion failed:\n" + process.stdout[-4000:])
        rewrite = rewrite_base_gguf(artifact_dir, base, temp_output)
        validation = validate_tq1_gguf(artifact_dir, temp_output)
        report = {
            **validation,
            **rewrite,
            "artifact": str(Path(artifact_dir).resolve()),
            "gguf": str(output),
            "converter": str(converter_path.resolve()),
            "converter_sha256": hashlib.sha256(converter_path.read_bytes()).hexdigest(),
            "command": list(command),
            "no_dense_rediscovery": True,
        }
        report_path = output.with_suffix(output.suffix + ".tq1_report.json")
        if output.exists():
            if not overwrite:
                raise FileExistsError(output)
            output.unlink()
        os.replace(temp_output, output)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
