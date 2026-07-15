"""Full-model schema-2 TQ1 calibration/PTQ production pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .artifact import ArtifactBuilder, ArtifactReader, tensor_sha256
from .calibration import (
    collect_model_statistics, file_sha256, iter_calibration_records,
    load_calibration_artifact, save_calibration_artifact)
from .codebook import Codebook, CodebookRegistry, base3_ids
from .packing import layout
from .oracle import dequantize_weight
from .ptq import Importance, PTQConfig, PTQResult, project_weight
from .solver import (
    build_product_codebook, corpus_from_tensors, facility_location_select)
from .spec import FLOAT_PROFILES, QuantSpec


LLAMA_TARGET_REGEXES = (
    r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)",
)
LLAMA_KEEP_FP_REGEXES = (r"lm_head",)


@dataclass(frozen=True)
class LinearInventory:
    target: tuple[str, ...]
    keep_fp: tuple[str, ...]

    def state_dict_targets(self) -> tuple[str, ...]:
        return tuple(name + ".weight" for name in self.target)


def classify_model_linears(model: nn.Module, quant_spec: QuantSpec, *,
                           enforce_llama_count: bool = True) -> LinearInventory:
    target_patterns = [re.compile(value) for value in quant_spec.target_regexes]
    keep_patterns = [re.compile(value) for value in quant_spec.keep_fp_regexes]
    target: list[str] = []
    keep: list[str] = []
    double: list[str] = []
    unmatched: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        target_match = any(pattern.fullmatch(name) for pattern in target_patterns)
        keep_match = any(pattern.fullmatch(name) for pattern in keep_patterns)
        if target_match and keep_match:
            double.append(name)
        elif target_match:
            if module.bias is not None:
                raise ValueError(f"{name}: targeted TQ1 linear must be bias-free")
            if module.weight.ndim != 2 or module.in_features % 256:
                raise ValueError(f"{name}: TQ1 requires [N,K] with K divisible by 256")
            if 127 * module.in_features > 2**31 - 1:
                raise ValueError(f"{name}: W2A8 int32 accumulator can overflow")
            target.append(name)
        elif keep_match:
            keep.append(name)
        else:
            unmatched.append(name)
    if double or unmatched:
        raise ValueError(
            "TQ1 linear inventory is not total/disjoint: "
            f"double={double[:8]}, unmatched={unmatched[:8]}")
    layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if enforce_llama_count and model_type == "llama" and layers is not None \
            and len(target) != 7 * int(layers):
        raise ValueError(
            f"Llama target inventory has {len(target)} linears, expected {7 * int(layers)}")
    return LinearInventory(tuple(sorted(target)), tuple(sorted(keep)))


def scalar_pattern_corpus(model: nn.Module, inventory: LinearInventory, *,
                          weighting: str = "family_equal"):
    """Build the Section 9 corpus using one ordinary initializer scale per row."""
    patterns: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in inventory.target:
            weight = model.get_submodule(name).weight.detach().float().cpu()
            scale = weight.abs().mean(dim=1, keepdim=True)
            denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
            trits = torch.round(weight / denominator).clamp(-1, 1).to(torch.int8)
            patterns[name] = trits.reshape(-1, 8)
    return corpus_from_tensors(patterns, weighting=weighting)


def learn_model_codebook(model: nn.Module, inventory: LinearInventory, *,
                         codebook_id: str, index_format: str,
                         encoding: str = "sign_canonical",
                         weighting: str = "family_equal", lambda_nz: float = 0.0,
                         swap_limit: int = 1, scope: str = "model") -> Codebook:
    corpus = scalar_pattern_corpus(model, inventory, weighting=weighting)
    if encoding == "sign_canonical":
        count = 1024 if index_format == "v11" else 2048
        shapes, report = facility_location_select(
            corpus, select_count=count, lambda_nz=lambda_nz,
            swap_limit=swap_limit, return_trace=True)
        from .codebook import sign_canonical_codebook
        return sign_canonical_codebook(
            codebook_id, index_format, shapes, scope=scope,
            provenance={
                "source": "model", "algorithm": "facility_location",
                "weighting": weighting, "lambda_nz": lambda_nz,
                **report,
            })
    if encoding == "product":
        return build_product_codebook(
            codebook_id=codebook_id, index_format=index_format, corpus=corpus,
            swap_limit=swap_limit, scope=scope)
    raise ValueError("direct IQ1 codebooks must be loaded from the pinned repository asset")


def importance_for_module(statistics: Mapping[str, torch.Tensor], module_name: str,
                          mode: str, width: int) -> Importance:
    if mode == "uniform":
        return Importance("uniform")
    suffix = {"diagonal": "diag", "covariance8": "cov8", "block256": "cov256"}[mode]
    key = f"{module_name}.{suffix}"
    if key not in statistics:
        raise ValueError(f"requested {mode} statistics are missing for {module_name}")
    if mode == "diagonal":
        result = Importance(mode, diag=statistics[key])
    elif mode == "covariance8":
        result = Importance(mode, cov8=statistics[key])
    else:
        result = Importance(mode, cov256=statistics[key])
    result.validate(width)
    return result


def collect_statistics(model: nn.Module, tokenizer, inventory: LinearInventory, *,
                       calibration_file: str | Path, output: str | Path,
                       modes: Sequence[str], sample_count: int,
                       sequence_cap: int, device: str | torch.device,
                       metadata: Mapping[str, Any], ridge_factor: float = 1e-5) -> Path:
    modules = {name: model.get_submodule(name) for name in inventory.target}
    records = iter_calibration_records(
        calibration_file, tokenizer, limit=sample_count, sequence_cap=sequence_cap)
    sums, collection = collect_model_statistics(
        model, modules, records, device=device, modes=modes)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_artifact(output, sums, metadata={
        **metadata,
        "calibration_file_sha256": file_sha256(calibration_file),
        "sample_limit": sample_count,
        "sequence_cap": sequence_cap,
        "modes": list(modes),
        "device": str(device),
        "accumulation_dtype": "float64_cpu",
        "target_modules": list(inventory.target),
        **collection,
    }, ridge_factor=ridge_factor)
    return output


def _ptq_config(spec: QuantSpec, profile: str, *, chunk_groups: int,
                allow_diagonal_fallback: bool) -> PTQConfig:
    return PTQConfig(
        profile=profile,
        scale_dtype=torch.float16 if spec.default_scale_dtype == "float16"
        else torch.bfloat16,
        weight_metric=spec.weight_metric,
        assignment_mode=spec.assignment_mode,
        candidate_count=spec.candidate_count,
        alternating_iterations=spec.alternating_iterations,
        chunk_groups=chunk_groups,
        gptq_feedback=spec.gptq_feedback,
        gptq_damping=spec.gptq_damping,
        allow_diagonal_fallback=allow_diagonal_fallback,
    )


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def executable_source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    repository = directory.parents[1]
    paths = list(sorted(directory.glob("*.py"))) + [
        repository / "bitnet_train" / "cpu" / "src" / "bitnet_cpu.c",
        repository / "bitnet_train" / "cpu" / "bitnet_cpu.py",
        repository / "bitnet_train" / "export" / "compare_gguf.py",
        repository / "bitnet_train" / "export" / "export_gguf.py",
        repository / "train" / "train.py",
    ]
    return {
        str(path.relative_to(repository)): _source_digest(path)
        for path in paths
    }


def repository_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    def run(*args: str) -> str:
        result = subprocess.run(args, cwd=root, check=False, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return result.stdout.strip()
    return {
        "repository_commit": run("git", "rev-parse", "HEAD") or "unknown",
        "dirty_worktree": bool(run("git", "status", "--porcelain")),
        "os": platform.platform(),
        "source_hashes": executable_source_hashes(),
    }


def tokenizer_identity(tokenizer) -> tuple[str, str]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialized = backend.to_str() if backend is not None else repr(tokenizer.get_vocab())
    tokenizer_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    template = getattr(tokenizer, "chat_template", None) or ""
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return tokenizer_hash, template_hash


def _aggregate_reports(reports: Mapping[str, Mapping[str, Any]],
                       tensor_sizes: Mapping[str, int]) -> dict[str, Any]:
    total = sum(tensor_sizes.values())
    weighted_keys = ("rmse", "relative_l2", "weighted_relative_error",
                     "scalar_pattern_exact_hit_rate", "effective_bpw")
    aggregate = {
        key: sum(float(reports[name][key]) * tensor_sizes[name]
                 for name in reports) / max(total, 1)
        for key in weighted_keys
    }
    aggregate.update({
        "target_tensors": len(reports),
        "target_parameters": total,
        "packed_payload_bytes": sum(
            int(reports[name]["physical_payload_bytes"]) for name in reports),
        "row_scale_bytes": sum(int(reports[name]["row_scale_bytes"]) for name in reports),
    })
    return aggregate


def run_full_model_ptq(model: nn.Module, quant_spec: QuantSpec,
                       registry: CodebookRegistry, *, output_dir: str | Path,
                       source_model: str, source_revision: str,
                       tokenizer=None, source_files: str | Path | None = None,
                       statistics: Mapping[str, torch.Tensor] | None = None,
                       calibration_hash: str | None = None,
                       chunk_groups: int = 4096,
                       allow_diagonal_fallback: bool = False,
                       overwrite: bool = False,
                       command: Sequence[str] = (),
                       evaluation_report: Mapping[str, Any] | None = None) -> Path:
    """Quantize every resolved target and transactionally emit schema 2."""
    registry.validate_refs(quant_spec.codebooks)
    inventory = classify_model_linears(model, quant_spec)
    statistics = dict(statistics or {})
    tensor_results: dict[str, PTQResult] = {}
    reports: dict[str, dict[str, Any]] = {}
    sizes: dict[str, int] = {}
    started = time.perf_counter()
    for module_name in inventory.target:
        module: nn.Linear = model.get_submodule(module_name)
        profile, codebook_id = quant_spec.resolve_profile(module_name)
        if profile in FLOAT_PROFILES:
            if codebook_id is not None:
                raise ValueError(f"floating target {module_name} unexpectedly has a codebook")
            continue
        if codebook_id is None:
            raise ValueError(f"TQ1 target {module_name} has no codebook")
        codebook = registry[codebook_id]
        importance = importance_for_module(
            statistics, module_name, quant_spec.importance_mode, module.in_features)
        result = project_weight(
            module.weight, codebook, importance,
            _ptq_config(quant_spec, profile, chunk_groups=chunk_groups,
                        allow_diagonal_fallback=allow_diagonal_fallback))
        state_name = module_name + ".weight"
        result.report.update({
            "module_path": module_name,
            "state_dict_name": state_name,
            "physical_payload_bytes": result.payload.numel(),
            "row_scale_bytes": 0 if result.row_scales is None
            else result.row_scales.numel() * result.row_scales.element_size(),
        })
        tensor_results[state_name] = result
        reports[state_name] = result.report
        sizes[state_name] = module.weight.numel()

    tokenizer_hash, template_hash = (tokenizer_identity(tokenizer) if tokenizer is not None
                                     else ("0" * 64, "0" * 64))
    provenance = {
        **repository_provenance(),
        "command": list(command),
        "calibration_statistics_sha256": calibration_hash,
        "quantizer_elapsed_seconds": time.perf_counter() - started,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "known_runtime_compatibility": {
            "scalar_oracle": {
                "compatible": True, "profiles": "all format-v1",
                "activation_modes": ["none", "a8_token", "a8_block256"],
            },
            "native_cpu": {
                "compatible": quant_spec.activation_mode != "none",
                "profiles": "all format-v1",
                "activation_modes": ["a8_token", "a8_block256"],
            },
            "llama_cpp": {
                "compatible": False,
                "reason": "requires the pinned TQ1 GGML registry/row-scale integration",
            },
        },
    }
    builder = ArtifactBuilder(
        quant_spec, registry, source_model=source_model,
        source_revision=source_revision, tokenizer_sha256=tokenizer_hash,
        chat_template_sha256=template_hash, provenance=provenance)
    for state_name, result in tensor_results.items():
        module_name = state_name.removesuffix(".weight")
        profile, codebook_id = quant_spec.resolve_profile(module_name)
        builder.add_quantized(
            state_name, module_name, result.payload,
            logical_shape=tuple(model.get_submodule(module_name).weight.shape),
            profile=profile, codebook_id=codebook_id or "",
            row_scales=result.row_scales)
    target_state_names = set(tensor_results)
    float_overrides = {
        name + ".weight": profile
        for name in inventory.target
        for profile, _ in (quant_spec.resolve_profile(name),)
        if profile in FLOAT_PROFILES
    }
    for name, value in model.state_dict().items():
        if name not in target_state_names:
            profile = float_overrides.get(name)
            dtype = ({"fp16": torch.float16, "bf16": torch.bfloat16,
                      "fp32": torch.float32}[profile] if profile else value.dtype)
            builder.add_non_tq1(name, value.to(dtype))
    if not tensor_results:
        raise ValueError("resolved mixed policy contains no TQ1 tensors")
    resolved_policy = {
        name + ".weight": {
            "profile": quant_spec.resolve_profile(name)[0],
            "codebook_id": quant_spec.resolve_profile(name)[1],
        }
        for name in inventory.target
    }
    report = {
        "schema": 1,
        "quant_spec": quant_spec.to_dict(),
        "quant_spec_sha256": quant_spec.sha256(),
        "source_model": source_model,
        "source_revision": source_revision,
        "calibration_statistics_sha256": calibration_hash,
        "aggregate": _aggregate_reports(reports, sizes),
        "resolved_tensor_policy": resolved_policy,
        "floating_override_tensors": sorted(float_overrides),
        "tensors": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path = builder.write(
        output_dir, source_files=source_files, quantization_report=report,
        evaluation_report=evaluation_report, overwrite=overwrite)
    ArtifactReader(path).validate(require_evaluation=evaluation_report is not None)
    return path


def load_statistics(path: str | Path | None, quant_spec: QuantSpec) \
        -> tuple[dict[str, torch.Tensor], dict[str, Any], str | None]:
    if path is None:
        if quant_spec.importance_mode != "uniform":
            raise ValueError(f"{quant_spec.importance_mode} PTQ requires --statistics-artifact")
        return {}, {}, None
    tensors, metadata = load_calibration_artifact(path)
    return tensors, metadata, file_sha256(path)


def save_model_source_files(model, tokenizer, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(destination)
    if tokenizer is not None:
        tokenizer.save_pretrained(destination)
    return destination


def load_registry_from_artifact(path: str | Path) -> tuple[QuantSpec, CodebookRegistry]:
    reader = ArtifactReader(path)
    reader.validate()
    return reader.quant_spec, reader.registry()


def bake_debug_checkpoint(artifact_dir: str | Path, output_dir: str | Path, *,
                          overwrite: bool = False) -> Path:
    """Decode a canonical artifact to a clearly labelled HF debug checkpoint."""
    from safetensors.torch import load_file, save_file

    reader = ArtifactReader(artifact_dir)
    reader.validate()
    output = Path(output_dir).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        state = {
            name: value.clone()
            for name, value in load_file(
                str(reader.directory / "non_tq1_model.safetensors"), device="cpu").items()
        }
        registry = reader.registry()
        for item in reader.manifest["tensors"]:
            _, payload, scales = reader.tensor(item["state_dict_name"])
            weight = dequantize_weight(
                payload, item["profile"], registry[item["codebook_id"]],
                row_scales=scales)
            state[item["state_dict_name"]] = weight.to(
                scales.dtype if scales is not None else torch.float16).clone()
        save_file(state, str(temp / "model.safetensors"))
        copied = {
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
            "chat_template.jinja", "added_tokens.json",
        }
        for path in reader.directory.iterdir():
            if path.name in copied and path.is_file():
                shutil.copy2(path, temp / path.name)
        config_path = temp / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError("canonical artifact has no config.json")
        config = json.loads(config_path.read_text())
        config["quantization_config"] = {
            "quant_method": "tq1_v_debug_baked",
            "canonical_packed": False,
            "debug_baked": True,
            "activation_quantization_automatic": False,
            "canonical_artifact": str(reader.directory),
            "quant_spec_sha256": reader.manifest["quant_spec_sha256"],
        }
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        (temp / "bake_report.json").write_text(json.dumps({
            "canonical_packed": False,
            "debug_baked": True,
            "activation_quantization_automatic": False,
            "quant_spec_sha256": reader.manifest["quant_spec_sha256"],
            "tensors": [item["state_dict_name"] for item in reader.manifest["tensors"]],
        }, indent=2, sort_keys=True) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, output)
        return output
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def export_qat_model(model: nn.Module, source_artifact: str | Path,
                     output_dir: str | Path, *, source_files: str | Path | None = None,
                     checkpoint_identity: str,
                     evaluation_report: Mapping[str, Any] | None = None,
                     overwrite: bool = False,
                     command: Sequence[str] = ()) -> Path:
    """Export frozen QAT indices/scales directly, with no dense rediscovery."""
    from .qat import iter_tq1linears

    source = ArtifactReader(source_artifact)
    source.validate()
    spec = source.quant_spec
    registry = source.registry()
    modules = dict(iter_tq1linears(model))
    expected = {item["module_path"] for item in source.manifest["tensors"]}
    if set(modules) != expected:
        raise ValueError(
            f"QAT/source tensor inventory mismatch: model={sorted(modules)}, "
            f"artifact={sorted(expected)}")
    for name, module in modules.items():
        if module.phase != "frozen":
            raise ValueError(f"{name}: QAT indices are not frozen")
        if module.quant_spec_sha256 != spec.sha256():
            raise ValueError(f"{name}: QAT QuantSpec differs from source artifact")
        if module.codebook_sha256 != registry[module.codebook_id].sha256():
            raise ValueError(f"{name}: QAT codebook differs from source artifact")

    builder = ArtifactBuilder(
        spec, registry, source_model=source.manifest["source_model"],
        source_revision=source.manifest["source_revision"],
        tokenizer_sha256=source.manifest["tokenizer_sha256"],
        chat_template_sha256=source.manifest["chat_template_sha256"],
        provenance={
            **repository_provenance(),
            "producer": "frozen_qat_exact_index_export",
            "source_artifact": str(Path(source_artifact).resolve()),
            "source_artifact_quant_spec_sha256": source.manifest["quant_spec_sha256"],
            "checkpoint_identity": checkpoint_identity,
            "command": list(command),
            "calibration_statistics_sha256": source.manifest["provenance"].get(
                "calibration_statistics_sha256"),
            "known_runtime_compatibility": source.manifest.get(
                "known_runtime_compatibility", {}),
        })
    reports: dict[str, Any] = {}
    for item in source.manifest["tensors"]:
        name = item["module_path"]
        module = modules[name]
        payload, scales = module.export_projection()
        # export_projection has already asserted frozen_reference equality.
        builder.add_quantized(
            item["state_dict_name"], name, payload,
            logical_shape=tuple(item["logical_shape"]), profile=item["profile"],
            codebook_id=item["codebook_id"], row_scales=scales)
        reports[item["state_dict_name"]] = module.health()
    state = model.state_dict()
    expected_non_tq1 = set(source.manifest["non_tq1_tensors"])
    absent = expected_non_tq1 - set(state)
    if absent:
        raise ValueError(f"QAT checkpoint lacks non-TQ1 tensors {sorted(absent)[:8]}")
    for name in sorted(expected_non_tq1):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name}: non-TQ1 state is not a tensor")
        builder.add_non_tq1(name, value)
    quantization_report = {
        "schema": 1,
        "producer": "qat",
        "phase": "frozen",
        "checkpoint_identity": checkpoint_identity,
        "quant_spec": spec.to_dict(),
        "quant_spec_sha256": spec.sha256(),
        "source_artifact": str(Path(source_artifact).resolve()),
        "tensors": reports,
    }
    artifact = builder.write(
        output_dir, source_files=source_files or source.directory,
        quantization_report=quantization_report,
        evaluation_report=evaluation_report, overwrite=overwrite)
    ArtifactReader(artifact).validate(require_evaluation=evaluation_report is not None)
    return artifact
