"""Exact codebook-aware TQ1 QAT modules with hard-forward STE semantics."""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .codebook import Codebook, base3_ids
from .oracle import quantize_activation
from .packing import pack_payload
from .ptq import PTQResult, candidate_table
from .spec import QuantSpec


_PHASES = {"soft": 0, "hard": 1, "frozen": 2}
_PHASE_NAMES = {value: key for key, value in _PHASES.items()}


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clamp_min(torch.finfo(torch.float32).tiny)
    return value + torch.log(-torch.expm1(-value))


class _ExactValueSTE(torch.autograd.Function):
    """Return the first input bit-for-bit; route its gradient to the surrogate."""

    @staticmethod
    def forward(ctx, exact: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
        del ctx, surrogate
        return exact

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        del ctx
        return None, gradient


def a8_ste(x: torch.Tensor, mode: str) -> torch.Tensor:
    quantized = quantize_activation(x, mode).dequantize().to(device=x.device, dtype=x.dtype)
    return _ExactValueSTE.apply(quantized, x)


@dataclass(frozen=True)
class ProjectionState:
    indices: torch.Tensor
    best_distance: torch.Tensor
    second_distance: torch.Tensor
    codewords: torch.Tensor


class TQ1Linear(nn.Module):
    """Bias-free linear whose every forward uses the deployable TQ1 projection."""

    def __init__(self, latent_weight: torch.Tensor, row_scales: torch.Tensor,
                 codebook: Codebook, quant_spec: QuantSpec, *, profile: str,
                 importance_diag: torch.Tensor | None = None,
                 importance_cov8: torch.Tensor | None = None,
                 initial_indices: torch.Tensor | None = None,
                 phase: str = "soft", top_m: int = 8, temperature: float = 1.0,
                 assignment_chunk: int = 2048):
        super().__init__()
        if latent_weight.ndim != 2 or latent_weight.shape[1] % 256:
            raise ValueError("TQ1Linear latent weight must be [N,K], K divisible by 256")
        if tuple(row_scales.shape) != (latent_weight.shape[0],):
            raise ValueError("TQ1Linear requires one initial scale per output row")
        if phase not in _PHASES or "-b" in profile or "-a4-" in profile:
            raise ValueError("format-v1 QAT supports soft/hard/frozen J/I/P row profiles")
        if not 1 <= top_m <= quant_spec.candidate_count:
            raise ValueError("top_m must be in [1,candidate_count]")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if assignment_chunk < 1:
            raise ValueError("assignment_chunk must be positive")
        expected = "direct_joint" if "-i-" in profile else \
            "product" if "-p-" in profile else "sign_canonical"
        if codebook.encoding != expected or codebook.index_format not in profile:
            raise ValueError("QAT profile and codebook disagree")

        # Keep the ordinary ``weight`` state-dict key so a gathered HF checkpoint
        # can still seed the architecture before distributed TQ1 state is restored.
        self.weight = nn.Parameter(latent_weight.detach().float().clone())
        zero_rows = (row_scales == 0).detach().bool()
        positive = row_scales.detach().float().clone()
        positive[zero_rows] = 1.0
        self.scale_parameter = nn.Parameter(_inverse_softplus(positive))
        self.register_buffer("zero_rows", zero_rows)
        self.register_buffer("phase_code", torch.tensor(_PHASES[phase], dtype=torch.int8))
        self.register_buffer("temperature_value", torch.tensor(float(temperature)))
        self.register_buffer("decoded_table", codebook.decode(
            torch.arange(codebook.index_count)).to(torch.int8).clone())
        self.register_buffer("legal_mask", codebook.legal_index_mask().clone())
        candidates = candidate_table(codebook, quant_spec.candidate_count)
        self.register_buffer("candidate_indices", candidates.clone())
        groups = latent_weight.shape[1] // 8
        if initial_indices is None:
            initial_indices = torch.zeros((latent_weight.shape[0], groups), dtype=torch.int64)
        if tuple(initial_indices.shape) != (latent_weight.shape[0], groups):
            raise ValueError("initial index shape mismatch")
        codebook.validate_indices(initial_indices)
        zero_index = torch.nonzero(
            (codebook.decode(torch.arange(codebook.index_count)) == 0).all(-1)
            & codebook.legal_index_mask()).flatten()[0]
        initial_indices = initial_indices.detach().long().clone()
        initial_indices[zero_rows] = int(zero_index)
        self.zero_index = int(zero_index)
        self.register_buffer("indices", initial_indices)
        self.register_buffer("frozen_reference", torch.empty(0, dtype=torch.int64))
        if importance_diag is None:
            importance_diag = torch.ones(latent_weight.shape[1])
        if tuple(importance_diag.shape) != (latent_weight.shape[1],):
            raise ValueError("QAT diagonal importance shape mismatch")
        self.register_buffer("importance_diag", importance_diag.detach().float().reshape(-1, 8))
        if importance_cov8 is not None and tuple(importance_cov8.shape) != (groups, 8, 8):
            raise ValueError("QAT covariance8 shape mismatch")
        self.register_buffer("importance_cov8", None if importance_cov8 is None
                             else importance_cov8.detach().float().clone())
        self.profile = profile
        self.activation_mode = quant_spec.activation_mode
        self.scale_dtype = (torch.float16 if quant_spec.default_scale_dtype == "float16"
                            else torch.bfloat16)
        self.quant_spec_json = quant_spec.canonical_json()
        self.quant_spec_sha256 = quant_spec.sha256()
        self.codebook_id = codebook.id
        self.codebook_sha256 = codebook.sha256()
        self.codebook_encoding = codebook.encoding
        self.top_m = int(top_m)
        self.assignment_chunk = int(assignment_chunk)
        self._cached_versions: tuple[int, int, int] | None = None
        self._last_projection: ProjectionState | None = None
        if phase == "frozen":
            self.freeze_indices()

    @classmethod
    def from_ptq(cls, linear: nn.Linear, result: PTQResult, codebook: Codebook,
                 quant_spec: QuantSpec, *, profile: str,
                 importance_diag: torch.Tensor | None = None,
                 importance_cov8: torch.Tensor | None = None, **kwargs) -> "TQ1Linear":
        if linear.bias is not None:
            raise ValueError("TQ1Linear is bias-free")
        if result.row_scales is None:
            raise ValueError("QAT requires a row-scale PTQ initializer")
        return cls(linear.weight.detach(), result.row_scales, codebook, quant_spec,
                   profile=profile, importance_diag=importance_diag,
                   importance_cov8=importance_cov8,
                   initial_indices=result.indices, **kwargs)

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def latent_weight(self) -> torch.Tensor:
        return self.weight

    @property
    def phase(self) -> str:
        return _PHASE_NAMES[int(self.phase_code)]

    @property
    def temperature(self) -> float:
        return float(self.temperature_value)

    def runtime_scales(self) -> torch.Tensor:
        alpha = F.softplus(self.scale_parameter)
        minimum = torch.finfo(self.scale_dtype).tiny
        rounded = alpha.clamp_min(minimum).to(self.scale_dtype).float()
        if not torch.isfinite(rounded).all():
            raise ValueError("QAT row scale is nonfinite")
        alpha_rt = alpha + (rounded - alpha).detach()
        return torch.where(self.zero_rows, torch.zeros_like(alpha_rt), alpha_rt)

    def set_temperature(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature_value.fill_(value)
        self._cached_versions = None

    def set_phase(self, phase: str) -> None:
        if phase not in _PHASES:
            raise ValueError(f"unknown QAT phase {phase!r}")
        if self.phase == "frozen" and phase != "frozen":
            raise ValueError("frozen QAT cannot be unfrozen in place")
        self.phase_code.fill_(_PHASES[phase])
        self._cached_versions = None
        if phase == "frozen":
            self.freeze_indices()

    def freeze_indices(self) -> None:
        if self.phase != "frozen":
            self.phase_code.fill_(_PHASES["frozen"])
        if self.indices.numel() == 0:
            raise ValueError("cannot freeze without serialized indices")
        self.frozen_reference = self.indices.detach().clone()
        self.latent_weight.requires_grad_(False)
        self._cached_versions = None

    def _distances(self, target: torch.Tensor, candidates: torch.Tensor,
                   group_ids: torch.Tensor) -> torch.Tensor:
        delta = target[:, None] - candidates
        if self.importance_cov8 is None:
            diag = self.importance_diag[group_ids]
            return (delta.square() * diag[:, None]).sum(-1)
        covariance = self.importance_cov8[group_ids]
        return torch.einsum("mci,mij,mcj->mc", delta, covariance, delta)

    def _project(self, alpha: torch.Tensor, *, soft: bool) -> ProjectionState:
        rows, groups = self.out_features, self.in_features // 8
        latent = self.latent_weight.reshape(rows, groups, 8)
        result_indices = torch.empty((rows, groups), dtype=torch.int64,
                                     device=latent.device)
        result_words = torch.empty_like(latent)
        best_all = torch.empty((rows, groups), device=latent.device)
        second_all = torch.empty((rows, groups), device=latent.device)
        flat_target = latent.reshape(-1, 8)
        row_ids = torch.arange(rows, device=latent.device).repeat_interleave(groups)
        group_ids = torch.arange(groups, device=latent.device).repeat(rows)
        normalized = flat_target / alpha[row_ids, None].clamp_min(1e-30)
        initializer = torch.round(normalized).clamp(-1, 1).to(torch.int8)
        choice_table = self.candidate_indices.to(latent.device)
        choices = choice_table[base3_ids(initializer).to(latent.device)]
        decoded_table = self.decoded_table.to(latent.device)
        for start in range(0, flat_target.shape[0], self.assignment_chunk):
            stop = min(start + self.assignment_chunk, flat_target.shape[0])
            selected = choices[start:stop]
            words = decoded_table[selected].to(latent.dtype)
            scaled = words * alpha[row_ids[start:stop], None, None]
            distances = self._distances(flat_target[start:stop], scaled,
                                        group_ids[start:stop])
            ordered = torch.argsort(distances, dim=1, stable=True)
            hard_pos = ordered[:, 0]
            hard_indices = selected.gather(1, hard_pos[:, None]).squeeze(1)
            if soft:
                top = ordered[:, :self.top_m]
                top_dist = distances.gather(1, top)
                probabilities = torch.softmax(-top_dist / self.temperature_value, dim=1)
                top_words = words.gather(1, top[..., None].expand(-1, -1, 8))
                soft_word = (probabilities[..., None] * top_words).sum(1)
                hard_word = decoded_table[hard_indices].to(latent.dtype)
                projected = soft_word + (hard_word - soft_word).detach()
            else:
                projected = decoded_table[hard_indices].to(latent.dtype)
            result_indices.reshape(-1)[start:stop] = hard_indices
            result_words.reshape(-1, 8)[start:stop] = projected
            best_all.reshape(-1)[start:stop] = distances.gather(1, ordered[:, :1]).squeeze(1)
            second_position = 1 if distances.shape[1] > 1 else 0
            second_all.reshape(-1)[start:stop] = distances.gather(
                1, ordered[:, second_position:second_position + 1]).squeeze(1)
        if torch.any(self.zero_rows):
            result_indices[self.zero_rows] = self.zero_index
            result_words[self.zero_rows] = 0
            best_all[self.zero_rows] = 0
            second_all[self.zero_rows] = 0
        return ProjectionState(result_indices, best_all, second_all, result_words)

    def projection(self, alpha: torch.Tensor) -> ProjectionState:
        if self.phase == "frozen":
            if self.frozen_reference.numel() == 0 or not torch.equal(
                    self.indices, self.frozen_reference):
                raise RuntimeError("frozen TQ1 indices changed")
            words = self.decoded_table.to(self.indices.device)[self.indices].to(
                self.latent_weight.dtype)
            zeros = torch.zeros_like(self.indices, dtype=self.latent_weight.dtype)
            return ProjectionState(self.indices, zeros, zeros, words)
        if self.phase == "soft":
            state = self._project(alpha, soft=True)
        else:
            versions = (self.latent_weight._version, self.scale_parameter._version,
                        int(self.phase_code))
            if not self.training and self._cached_versions == versions \
                    and self._last_projection is not None:
                return self._last_projection
            state = self._project(alpha, soft=False)
            if not self.training:
                self._cached_versions = versions
        with torch.no_grad():
            self.indices.copy_(state.indices.detach())
        self._last_projection = state
        return state

    def projected_weight(self) -> torch.Tensor:
        alpha = self.runtime_scales()
        state = self.projection(alpha)
        hard_words = self.decoded_table.to(state.indices.device)[state.indices].to(
            self.latent_weight.dtype)
        hard = alpha[:, None, None] * hard_words
        if self.phase == "frozen":
            return hard.reshape_as(self.latent_weight)
        surrogate_words = state.codewords if self.phase == "soft" else hard_words.detach()
        surrogate = (self.latent_weight.reshape_as(hard)
                     + alpha[:, None, None] * surrogate_words)
        projected = _ExactValueSTE.apply(hard.detach(), surrogate)
        return projected.reshape_as(self.latent_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = a8_ste(x, self.activation_mode) if self.activation_mode != "none" else x
        return F.linear(activation, self.projected_weight().to(activation.dtype))

    def margin_loss(self, margin: float) -> torch.Tensor:
        if self._last_projection is None:
            alpha = self.runtime_scales()
            self.projection(alpha)
        state = self._last_projection
        assert state is not None
        return torch.relu(torch.as_tensor(margin, device=state.best_distance.device)
                          - (state.second_distance - state.best_distance)).mean()

    @torch.no_grad()
    def health(self, previous_indices: torch.Tensor | None = None) -> dict[str, float]:
        alpha = self.runtime_scales()
        state = self.projection(alpha)
        counts = torch.bincount(state.indices.flatten(), minlength=self.decoded_table.shape[0]).float()
        probabilities = counts[counts > 0] / counts.sum()
        hard_words = self.decoded_table[state.indices]
        metrics = {
            "index_flip_rate": (0.0 if previous_indices is None else
                                float((state.indices.cpu() != previous_indices.cpu()).float().mean())),
            "margin_p05": float(torch.quantile(
                (state.second_distance - state.best_distance).float().flatten(), 0.05)),
            "codebook_entropy": float(-(probabilities * probabilities.log2()).sum()),
            "codebook_perplexity": float(torch.exp(-(probabilities * probabilities.log()).sum())),
            "dead_codewords": float((counts[self.legal_mask] == 0).sum()),
            "zero_codeword_rate": float((hard_words == 0).all(-1).float().mean()),
            "frac_neg": float((hard_words == -1).float().mean()),
            "frac_zero": float((hard_words == 0).float().mean()),
            "frac_pos": float((hard_words == 1).float().mean()),
            "scale_min": float(alpha.min()),
            "scale_median": float(alpha.median()),
            "scale_max": float(alpha.max()),
        }
        return metrics

    @torch.no_grad()
    def export_projection(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.runtime_scales()
        state = self.projection(alpha)
        payload = pack_payload(state.indices.cpu(), self.profile)
        return payload, alpha.to(self.scale_dtype).cpu()

    def get_extra_state(self) -> torch.Tensor:
        state = {
            "profile": self.profile,
            "phase": self.phase,
            "temperature": self.temperature,
            "top_m": self.top_m,
            "quant_spec_json": self.quant_spec_json,
            "quant_spec_sha256": self.quant_spec_sha256,
            "codebook_id": self.codebook_id,
            "codebook_sha256": self.codebook_sha256,
            "codebook_encoding": self.codebook_encoding,
        }
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return torch.tensor(list(payload), dtype=torch.uint8)

    def set_extra_state(self, state: torch.Tensor | dict[str, Any]) -> None:
        # Dict support reads early development checkpoints; new checkpoints are
        # safetensors-compatible uint8 canonical JSON.
        if isinstance(state, torch.Tensor):
            if state.dtype != torch.uint8 or state.ndim != 1:
                raise ValueError("TQ1 extra state must be a uint8 JSON vector")
            state = json.loads(bytes(state.cpu().tolist()).decode("utf-8"))
        if state["quant_spec_sha256"] != self.quant_spec_sha256:
            raise ValueError("QAT checkpoint QuantSpec mismatch")
        if state["codebook_sha256"] != self.codebook_sha256:
            raise ValueError("QAT checkpoint codebook mismatch")
        if state["profile"] != self.profile:
            raise ValueError("QAT checkpoint tensor profile mismatch")
        self.top_m = int(state["top_m"])
        self.temperature_value.fill_(float(state["temperature"]))
        self.phase_code.fill_(_PHASES[state["phase"]])
        self.latent_weight.requires_grad_(state["phase"] != "frozen")


class TQ1Experts(nn.Module):
    """Generic expert-leading wrapper; each expert owns independent rows/scales."""

    def __init__(self, experts: list[TQ1Linear]):
        super().__init__()
        if not experts:
            raise ValueError("TQ1Experts requires at least one expert")
        shape = (experts[0].in_features, experts[0].out_features)
        if any((expert.in_features, expert.out_features) != shape for expert in experts):
            raise ValueError("all fused experts must share their logical shape")
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        if tuple(expert_ids.shape) != tuple(x.shape[:-1]):
            raise ValueError("expert_ids must match the activation leading shape")
        flat_x = x.reshape(-1, x.shape[-1])
        flat_ids = expert_ids.reshape(-1)
        out = torch.empty((flat_x.shape[0], self.experts[0].out_features),
                          device=x.device, dtype=x.dtype)
        for expert_id, expert in enumerate(self.experts):
            positions = torch.nonzero(flat_ids == expert_id).flatten()
            if positions.numel():
                out[positions] = expert(flat_x[positions])
        if torch.any((flat_ids < 0) | (flat_ids >= len(self.experts))):
            raise ValueError("expert id outside range")
        return out.reshape(*x.shape[:-1], -1)


def iter_tq1linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, TQ1Linear):
            yield name, module
