"""Inference modules backed directly by canonical packed TQ1 artifacts."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

from .artifact import ArtifactReader, tensor_sha256
from .codebook import Codebook
from .oracle import linear_w2a8, linear_w_only, quantize_activation


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


class NativeCPUTQ1Linear(PackedTQ1Linear):
    """Repo-owned native CPU path over canonical schema-2 payload bytes.

    The only backend-private representation is an expanded int8 codebook plus a
    legal-index bitmap.  Its deterministic hash and memory cost are exposed for
    performance reports; canonical payload and codebook state remain resident.
    """

    def __init__(self, *args, impl: str = "auto", **kwargs):
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
            "original_codebook_bytes": len(self.codebook.canonical_bytes()),
            "resident_repack_bytes": resident,
            "peak_temporary_bytes": resident,
            "repack_time_ms": (time.perf_counter() - started) * 1e3,
            "repack_sha256": hashlib.sha256(stream).hexdigest(),
            "canonical_packed_remains_resident": True,
            "implementation": impl,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            raise ValueError("native CPU TQ1 requires CPU activations")
        if x.shape[-1] != self.in_features:
            raise ValueError("packed TQ1 activation width mismatch")
        from bitnet_train.cpu import bitnet_cpu

        original = x.shape[:-1]
        x2 = x.detach().float().reshape(-1, self.in_features)
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


def _parent_and_attr(model: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    parent_name, _, attribute = module_path.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), attribute


def load_packed_model(artifact_dir: str | Path, *, activation_mode: str | None = None,
                      dtype: torch.dtype = torch.float32,
                      runtime_backend: str = "scalar_oracle",
                      native_impl: str = "auto"):
    """Instantiate the HF architecture and replace every target with packed oracle IO."""
    from transformers import AutoConfig, AutoModelForCausalLM

    reader = ArtifactReader(artifact_dir)
    reader.validate()
    config = AutoConfig.from_pretrained(reader.directory, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    non_tq1 = load_file(
        str(reader.directory / "non_tq1_model.safetensors"), device="cpu")
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
    for item in reader.manifest["tensors"]:
        _, payload, scales = reader.tensor(item["state_dict_name"])
        parent, attribute = _parent_and_attr(model, item["module_path"])
        old = getattr(parent, attribute)
        if not isinstance(old, nn.Linear) or old.bias is not None:
            raise ValueError(f"{item['module_path']}: artifact target is not a bias-free Linear")
        if list(old.weight.shape) != item["logical_shape"]:
            raise ValueError(f"{item['module_path']}: artifact/config shape mismatch")
        module_kwargs = {"impl": native_impl} if module_type is NativeCPUTQ1Linear else {}
        setattr(parent, attribute, module_type(
            payload, item["profile"], registry[item["codebook_id"]],
            row_scales=scales, activation_mode=mode,
            state_dict_name=item["state_dict_name"], **module_kwargs))
    model.tie_weights()
    model.eval().requires_grad_(False)
    model.config.quantization_config = {
        "quant_method": "tq1_v",
        "canonical_packed": True,
        "artifact_schema": 2,
        "quant_spec_sha256": reader.manifest["quant_spec_sha256"],
        "activation_mode": mode,
        "runtime_backend": runtime_backend,
    }
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
