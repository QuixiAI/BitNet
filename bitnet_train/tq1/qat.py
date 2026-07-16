"""Exact codebook-aware TQ1 QAT modules with hard-forward STE semantics."""

from __future__ import annotations

import math
import json
import weakref
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .codebook import Codebook, base3_ids
from .oracle import quantize_activation
from .packing import pack_payload
from .ptq import PTQResult, candidate_table
from .spec import TQ1_PROFILES, QuantSpec


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
        if latent_weight.dtype not in {torch.float16, torch.bfloat16, torch.float32} \
                or not torch.isfinite(latent_weight).all():
            raise ValueError("TQ1Linear latent weight has an unsupported dtype or value")
        if tuple(row_scales.shape) != (latent_weight.shape[0],):
            raise ValueError("TQ1Linear requires one initial scale per output row")
        if not row_scales.is_floating_point() or not torch.isfinite(row_scales).all() \
                or torch.any(row_scales < 0):
            raise ValueError("TQ1Linear initial row scales must be finite and nonnegative")
        source_zero_rows = (latent_weight == 0).all(1)
        if torch.any(source_zero_rows != (row_scales == 0)):
            raise ValueError("TQ1Linear zero-row and zero-scale inventories disagree")
        if phase not in _PHASES or "-b" in profile or "-a4-" in profile:
            raise ValueError("format-v1 QAT supports soft/hard/frozen J/I/P row profiles")
        declared_profiles = {quant_spec.default_profile} | {
            rule.profile for rule in quant_spec.tensor_overrides
            if rule.profile in TQ1_PROFILES
        }
        if profile not in declared_profiles:
            raise ValueError("QAT profile is not declared by the canonical QuantSpec")
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
        try:
            registered_codebook = quant_spec.codebook(codebook.id)
        except KeyError as exc:
            raise ValueError("QAT codebook is absent from the canonical QuantSpec") from exc
        if registered_codebook != codebook.ref():
            raise ValueError("QAT codebook identity differs from the canonical QuantSpec")

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
        candidates = (candidate_table(codebook, quant_spec.candidate_count)
                      if quant_spec.assignment_mode == "shortlist"
                      else torch.empty((0, 0), dtype=torch.int64))
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
        if quant_spec.importance_mode == "uniform":
            if importance_cov8 is not None:
                raise ValueError("uniform QAT cannot carry covariance importance")
            if importance_diag is None:
                importance_diag = torch.ones(latent_weight.shape[1])
        elif quant_spec.importance_mode == "covariance8":
            if importance_cov8 is None:
                raise ValueError("covariance8 QAT requires covariance statistics")
            if importance_diag is None:
                importance_diag = importance_cov8.diagonal(
                    dim1=-2, dim2=-1).reshape(-1)
        elif importance_diag is None:
            raise ValueError(
                f"{quant_spec.importance_mode} QAT requires diagonal statistics")
        if tuple(importance_diag.shape) != (latent_weight.shape[1],):
            raise ValueError("QAT diagonal importance shape mismatch")
        if not torch.isfinite(importance_diag).all() or torch.any(importance_diag < 0):
            raise ValueError("QAT diagonal importance is invalid")
        self.register_buffer("importance_diag", importance_diag.detach().float().reshape(-1, 8))
        if importance_cov8 is not None and tuple(importance_cov8.shape) != (groups, 8, 8):
            raise ValueError("QAT covariance8 shape mismatch")
        if importance_cov8 is not None and (not torch.isfinite(importance_cov8).all()
                                            or not torch.allclose(
                                                importance_cov8, importance_cov8.transpose(-1, -2),
                                                atol=1e-6, rtol=1e-6)):
            raise ValueError("QAT covariance8 importance is invalid")
        if importance_cov8 is not None:
            minimum = float(torch.linalg.eigvalsh(
                importance_cov8.detach().float().cpu()).min())
            if minimum < -1e-6:
                raise ValueError("QAT covariance8 importance is not positive semidefinite")
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
        self.assignment_mode = quant_spec.assignment_mode
        self.weight_metric = quant_spec.weight_metric
        self.importance_mode = quant_spec.importance_mode
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
                   group_ids: torch.Tensor,
                   iq1_factor: torch.Tensor | None = None) -> torch.Tensor:
        delta = target[:, None] - candidates
        if self.importance_cov8 is None:
            diag = self.importance_diag[group_ids]
            if iq1_factor is not None:
                diag = diag * iq1_factor
            return (delta.square() * diag[:, None]).sum(-1)
        covariance = self.importance_cov8[group_ids]
        if iq1_factor is not None:
            root = torch.sqrt(iq1_factor)
            covariance = covariance * root[:, :, None] * root[:, None, :]
        return torch.einsum("mci,mij,mcj->mc", delta, covariance, delta)

    def _iq1_factor(self, latent: torch.Tensor) -> torch.Tensor | None:
        if self.weight_metric == "uniform":
            return None
        blocks = latent.float().reshape(self.out_features, -1, 256)
        sigma2 = 2.0 * blocks.square().mean(-1, keepdim=True)
        return torch.sqrt(sigma2 + blocks.square()).reshape(-1, 8)

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
        decoded_table = self.decoded_table.to(latent.device)
        iq1_factor = self._iq1_factor(latent)
        legal_indices = (torch.nonzero(self.legal_mask).flatten().to(latent.device)
                         if self.assignment_mode == "exhaustive" else None)
        choice_table = (self.candidate_indices.to(latent.device)
                        if self.assignment_mode == "shortlist" else None)
        initializer_ids = (base3_ids(initializer).to(latent.device)
                           if choice_table is not None else None)
        for start in range(0, flat_target.shape[0], self.assignment_chunk):
            stop = min(start + self.assignment_chunk, flat_target.shape[0])
            if legal_indices is not None:
                selected = legal_indices[None].expand(stop - start, -1)
            else:
                assert choice_table is not None and initializer_ids is not None
                selected = choice_table[initializer_ids[start:stop]]
                selected = torch.sort(selected, dim=1).values
            words = decoded_table[selected].to(latent.dtype)
            scaled = words * alpha[row_ids[start:stop], None, None]
            distances = self._distances(flat_target[start:stop], scaled,
                                        group_ids[start:stop],
                                        None if iq1_factor is None
                                        else iq1_factor[start:stop])
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
    def health(self, previous_indices: torch.Tensor | None = None) -> dict[str, Any]:
        alpha = self.runtime_scales()
        state = self.projection(alpha)
        diagnostic_state = self._project(alpha, soft=False) if self.phase == "frozen" else state
        counts = torch.bincount(
            state.indices.flatten(), minlength=self.decoded_table.shape[0]).float()
        probabilities = counts[counts > 0] / counts.sum()
        hard_words = self.decoded_table[state.indices]
        source_alpha = F.softplus(self.scale_parameter)
        source_alpha = torch.where(
            self.zero_rows, torch.zeros_like(source_alpha), source_alpha)
        divisor = torch.where(alpha > 0, alpha, torch.ones_like(alpha))[:, None, None]
        scalar = torch.round(
            self.latent_weight.reshape_as(hard_words) / divisor).clamp(-1, 1).to(torch.int8)
        scalar[self.zero_rows] = 0
        changed = (hard_words != scalar).sum(-1)
        margin = (diagnostic_state.second_distance
                  - diagnostic_state.best_distance).float().flatten()
        delta = (alpha[:, None, None] * hard_words.to(self.latent_weight.dtype)
                 - self.latent_weight.reshape_as(hard_words)).float()
        latent = self.latent_weight.detach().float().reshape_as(delta)
        iq1_factor = self._iq1_factor(latent)
        if iq1_factor is not None:
            iq1_factor = iq1_factor.reshape_as(delta)
        if self.importance_cov8 is None:
            metric = self.importance_diag.float()[None]
            if iq1_factor is not None:
                metric = metric * iq1_factor
            weighted_error = float((delta.square() * metric).sum())
            weighted_source = float((latent.square() * metric).sum())
        else:
            covariance = self.importance_cov8.float()
            if iq1_factor is None:
                weighted_error = float(torch.einsum(
                    "rgi,gij,rgj->", delta, covariance, delta))
                weighted_source = float(torch.einsum(
                    "rgi,gij,rgj->", latent, covariance, latent))
            else:
                root = torch.sqrt(iq1_factor)
                covariance = (covariance[None] * root[..., :, None]
                              * root[..., None, :])
                weighted_error = float(torch.einsum(
                    "rgi,rgij,rgj->", delta, covariance, delta))
                weighted_source = float(torch.einsum(
                    "rgi,rgij,rgj->", latent, covariance, latent))
        previous_counts = None
        index_flip_count = newly_activated = retired = 0
        if previous_indices is not None:
            if previous_indices.dtype == torch.bool \
                    or previous_indices.is_floating_point() \
                    or previous_indices.is_complex() or previous_indices.is_quantized:
                raise ValueError("previous QAT indices must use an integer tensor dtype")
            previous = previous_indices.detach().to(
                device=state.indices.device, dtype=torch.int64)
            if tuple(previous.shape) != tuple(state.indices.shape):
                raise ValueError("previous QAT index shape mismatch")
            if previous.numel() and (int(previous.min()) < 0
                                     or int(previous.max()) >= self.decoded_table.shape[0]
                                     or torch.any(~self.legal_mask[previous])):
                raise ValueError("previous QAT indices contain an illegal codeword")
            previous_counts = torch.bincount(
                previous.flatten(), minlength=self.decoded_table.shape[0])
            index_flip_count = int((state.indices != previous).sum())
            newly_activated = int(((counts > 0) & (previous_counts == 0)
                                   & self.legal_mask).sum())
            retired = int(((counts == 0) & (previous_counts > 0)
                           & self.legal_mask).sum())
        lane_fractions = []
        for lane in range(8):
            values = hard_words[..., lane]
            lane_fractions.append({
                "negative": float((values == -1).float().mean()),
                "zero": float((values == 0).float().mean()),
                "positive": float((values == 1).float().mean()),
            })
        tiny = torch.finfo(self.scale_dtype).tiny
        metrics = {
            "phase": self.phase,
            "group_count": state.indices.numel(),
            "weight_count": self.latent_weight.numel(),
            "index_flip_count": index_flip_count,
            "index_flip_rate": index_flip_count / max(state.indices.numel(), 1),
            "frozen_assignment_disagreement_rate": (
                float((diagnostic_state.indices != state.indices).float().mean())
                if self.phase == "frozen" else 0.0),
            "mean_changed_trits_per_group": float(changed.float().mean()),
            "changed_trits_histogram": {
                str(value): int((changed == value).sum()) for value in range(9)
            },
            "scalar_pattern_exact_hit_rate": float((changed == 0).float().mean()),
            "margin_min": float(margin.min()),
            "margin_mean": float(margin.mean()),
            "margin_p05": float(torch.quantile(margin, 0.05)),
            "margin_p50": float(torch.quantile(margin, 0.50)),
            "margin_p95": float(torch.quantile(margin, 0.95)),
            "codebook_entropy": float(-(probabilities * probabilities.log2()).sum()),
            "codebook_perplexity": float(torch.exp(-(probabilities * probabilities.log()).sum())),
            "dead_codewords": int((counts[self.legal_mask] == 0).sum()),
            "newly_activated_codewords": newly_activated,
            "retired_codewords": retired,
            "zero_codeword_rate": float((hard_words == 0).all(-1).float().mean()),
            "frac_neg": float((hard_words == -1).float().mean()),
            "frac_zero": float((hard_words == 0).float().mean()),
            "frac_pos": float((hard_words == 1).float().mean()),
            "lane_trit_fractions": lane_fractions,
            "source_scale_min": float(source_alpha.min()),
            "source_scale_median": float(source_alpha.median()),
            "source_scale_max": float(source_alpha.max()),
            "scale_min": float(alpha.min()),
            "scale_median": float(alpha.median()),
            "scale_max": float(alpha.max()),
            "scale_underflow_count": int(
                ((~self.zero_rows) & (source_alpha < tiny)).sum()),
            "weighted_projection_error": weighted_error,
            "weighted_source_energy": weighted_source,
            "weighted_projection_relative_error": math.sqrt(
                weighted_error / max(weighted_source, 1e-30)),
        }
        return metrics

    @torch.no_grad()
    def export_projection(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.runtime_scales()
        state = self.projection(alpha)
        payload = pack_payload(state.indices.cpu(), self.profile)
        return payload, alpha.to(self.scale_dtype).cpu()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        immutable = (
            "zero_rows", "decoded_table", "legal_mask", "candidate_indices",
            "importance_diag", "importance_cov8",
        )
        for name in immutable:
            expected = getattr(self, name)
            key = prefix + name
            if expected is None:
                if key in state_dict:
                    error_msgs.append(f"{key}: unexpected immutable QAT buffer")
                continue
            if key in state_dict:
                observed = state_dict[key]
                if observed.dtype != expected.dtype \
                        or tuple(observed.shape) != tuple(expected.shape) \
                        or not torch.equal(observed.detach().cpu(), expected.detach().cpu()):
                    error_msgs.append(
                        f"{key}: immutable QAT buffer differs from canonical initialization")
        for name in ("indices", "frozen_reference"):
            key = prefix + name
            if key in state_dict and state_dict[key].dtype != torch.int64:
                error_msgs.append(f"{key}: serialized QAT indices must be int64")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def get_extra_state(self) -> torch.Tensor:
        state = {
            "schema": 1,
            "profile": self.profile,
            "phase": self.phase,
            "temperature": self.temperature,
            "top_m": self.top_m,
            "quant_spec_json": self.quant_spec_json,
            "quant_spec_sha256": self.quant_spec_sha256,
            "codebook_id": self.codebook_id,
            "codebook_sha256": self.codebook_sha256,
            "codebook_encoding": self.codebook_encoding,
            "assignment_mode": self.assignment_mode,
            "weight_metric": self.weight_metric,
            "importance_mode": self.importance_mode,
        }
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return torch.tensor(list(payload), dtype=torch.uint8)

    def set_extra_state(self, state: torch.Tensor | dict[str, Any]) -> None:
        # New checkpoints use a safetensors-compatible uint8 canonical JSON
        # vector. Dict input remains useful to PyTorch callers but obeys the
        # same explicit schema.
        if isinstance(state, torch.Tensor):
            if state.dtype != torch.uint8 or state.ndim != 1:
                raise ValueError("TQ1 extra state must be a uint8 JSON vector")
            try:
                state = json.loads(bytes(state.cpu().tolist()).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("TQ1 extra state is not canonical JSON") from exc
        required = {
            "schema", "profile", "phase", "temperature", "top_m",
            "quant_spec_json", "quant_spec_sha256", "codebook_id",
            "codebook_sha256", "codebook_encoding", "assignment_mode",
            "weight_metric", "importance_mode",
        }
        if not isinstance(state, dict) or set(state) != required or state["schema"] != 1:
            raise ValueError("TQ1 extra state has an unsupported schema")
        if state["quant_spec_sha256"] != self.quant_spec_sha256:
            raise ValueError("QAT checkpoint QuantSpec mismatch")
        if state["quant_spec_json"] != self.quant_spec_json:
            raise ValueError("QAT checkpoint canonical QuantSpec JSON mismatch")
        if state["codebook_sha256"] != self.codebook_sha256:
            raise ValueError("QAT checkpoint codebook mismatch")
        if state["codebook_id"] != self.codebook_id \
                or state["codebook_encoding"] != self.codebook_encoding:
            raise ValueError("QAT checkpoint codebook identity mismatch")
        if state["profile"] != self.profile:
            raise ValueError("QAT checkpoint tensor profile mismatch")
        for name in ("assignment_mode", "weight_metric", "importance_mode"):
            if state.get(name, getattr(self, name)) != getattr(self, name):
                raise ValueError(f"QAT checkpoint {name} mismatch")
        top_m = state["top_m"]
        temperature = state["temperature"]
        phase = state["phase"]
        if isinstance(top_m, bool) or not isinstance(top_m, int) \
                or top_m < 1 or top_m != self.top_m:
            raise ValueError("QAT checkpoint top_m differs from canonical initialization")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
                or not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("QAT checkpoint temperature is invalid")
        if phase not in _PHASES:
            raise ValueError("QAT checkpoint phase is invalid")
        if self.phase != phase or self.temperature != float(temperature):
            raise ValueError("QAT checkpoint duplicated phase/temperature state disagrees")
        if phase == "frozen":
            if tuple(self.frozen_reference.shape) != tuple(self.indices.shape) \
                    or not torch.equal(self.frozen_reference, self.indices):
                raise ValueError("QAT checkpoint frozen indices are inconsistent")
        elif self.frozen_reference.numel() != 0:
            raise ValueError("non-frozen QAT checkpoint carries a frozen reference")
        self.top_m = top_m
        self.temperature_value.fill_(float(temperature))
        self.phase_code.fill_(_PHASES[phase])
        self.latent_weight.requires_grad_(state["phase"] != "frozen")


def aggregate_qat_health(tensors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Reconcile per-tensor QAT health into one group-weighted model summary."""
    if not tensors:
        raise ValueError("QAT health aggregation requires at least one tensor")
    required = {
        "phase", "group_count", "weight_count", "index_flip_count",
        "frozen_assignment_disagreement_rate", "changed_trits_histogram",
        "margin_min", "margin_p05", "margin_p50", "margin_p95",
        "codebook_entropy", "codebook_perplexity", "dead_codewords",
        "newly_activated_codewords", "retired_codewords", "zero_codeword_rate",
        "frac_neg", "frac_zero", "frac_pos", "lane_trit_fractions",
        "source_scale_min", "source_scale_median", "source_scale_max",
        "scale_min", "scale_median", "scale_max", "scale_underflow_count",
        "weighted_projection_error", "weighted_source_energy",
    }
    for name, metrics in tensors.items():
        missing = required - set(metrics)
        if missing:
            raise ValueError(f"{name}: QAT health is missing {sorted(missing)}")
        if not isinstance(metrics["group_count"], int) or metrics["group_count"] < 1:
            raise ValueError(f"{name}: QAT health group count is invalid")
        histogram = metrics["changed_trits_histogram"]
        if not isinstance(histogram, Mapping) or set(histogram) != {
                str(value) for value in range(9)} \
                or sum(int(value) for value in histogram.values()) != metrics["group_count"]:
            raise ValueError(f"{name}: changed-trit histogram does not reconcile")
        lanes = metrics["lane_trit_fractions"]
        if not isinstance(lanes, list) or len(lanes) != 8:
            raise ValueError(f"{name}: lane-trit inventory is invalid")
    total_groups = sum(int(value["group_count"]) for value in tensors.values())

    def weighted(key: str) -> float:
        return sum(float(value[key]) * int(value["group_count"])
                   for value in tensors.values()) / total_groups

    histogram = {
        str(changed): sum(int(value["changed_trits_histogram"][str(changed)])
                          for value in tensors.values())
        for changed in range(9)
    }
    lane_fractions = []
    for lane in range(8):
        lane_fractions.append({
            trit: sum(float(value["lane_trit_fractions"][lane][trit])
                      * int(value["group_count"])
                      for value in tensors.values()) / total_groups
            for trit in ("negative", "zero", "positive")
        })
    weighted_error = sum(float(value["weighted_projection_error"])
                         for value in tensors.values())
    weighted_source = sum(float(value["weighted_source_energy"])
                          for value in tensors.values())
    index_flips = sum(int(value["index_flip_count"]) for value in tensors.values())
    return {
        "tensor_count": len(tensors),
        "group_count": total_groups,
        "weight_count": sum(int(value["weight_count"]) for value in tensors.values()),
        "phases": sorted({str(value["phase"]) for value in tensors.values()}),
        "index_flip_count": index_flips,
        "index_flip_rate": index_flips / total_groups,
        "frozen_assignment_disagreement_rate": weighted(
            "frozen_assignment_disagreement_rate"),
        "mean_changed_trits_per_group": sum(
            changed * histogram[str(changed)] for changed in range(9)) / total_groups,
        "changed_trits_histogram": histogram,
        "scalar_pattern_exact_hit_rate": histogram["0"] / total_groups,
        "margin_min": min(float(value["margin_min"]) for value in tensors.values()),
        # The aggregate p05 is deliberately conservative across tensors; p50/p95
        # are explicitly group-weighted summaries of per-tensor quantiles.
        "margin_p05": min(float(value["margin_p05"]) for value in tensors.values()),
        "margin_p50_tensor_weighted": weighted("margin_p50"),
        "margin_p95_tensor_weighted": weighted("margin_p95"),
        "codebook_entropy_tensor_weighted": weighted("codebook_entropy"),
        "codebook_perplexity_tensor_weighted": weighted("codebook_perplexity"),
        "dead_codewords": sum(int(value["dead_codewords"])
                              for value in tensors.values()),
        "newly_activated_codewords": sum(int(value["newly_activated_codewords"])
                                         for value in tensors.values()),
        "retired_codewords": sum(int(value["retired_codewords"])
                                 for value in tensors.values()),
        "zero_codeword_rate": weighted("zero_codeword_rate"),
        "frac_neg": weighted("frac_neg"),
        "frac_zero": weighted("frac_zero"),
        "frac_pos": weighted("frac_pos"),
        "lane_trit_fractions": lane_fractions,
        "source_scale_min": min(float(value["source_scale_min"])
                                for value in tensors.values()),
        "source_scale_median_tensor_weighted": weighted("source_scale_median"),
        "source_scale_max": max(float(value["source_scale_max"])
                                for value in tensors.values()),
        "scale_min": min(float(value["scale_min"]) for value in tensors.values()),
        "scale_median_tensor_weighted": weighted("scale_median"),
        "scale_max": max(float(value["scale_max"]) for value in tensors.values()),
        "scale_underflow_count": sum(int(value["scale_underflow_count"])
                                     for value in tensors.values()),
        "weighted_projection_error": weighted_error,
        "weighted_source_energy": weighted_source,
        "weighted_projection_relative_error": math.sqrt(
            weighted_error / max(weighted_source, 1e-30)),
    }


class TQ1Embedding(TQ1Linear):
    """One TQ1 latent/projection shared by token lookup and the output head.

    The module deliberately lives at the input-embedding module path, so its
    latent remains ``model.embed_tokens.weight`` in a training checkpoint.
    :class:`TQ1OutputHead` owns no parameters or buffers and delegates its
    linear consumer back to this exact object.
    """

    def __init__(self, *args, padding_idx: int | None = None,
                 max_norm: float | None = None, norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False, sparse: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if sparse:
            raise ValueError("TQ1 embedding QAT does not support sparse gradients")
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = float(norm_type)
        self.scale_grad_by_freq = bool(scale_grad_by_freq)
        self.sparse = False
        self.consumer_family = "shared_embedding_output_head"

    @classmethod
    def from_ptq(cls, embedding: nn.Embedding, result: PTQResult,
                 codebook: Codebook, quant_spec: QuantSpec, *, profile: str,
                 importance_diag: torch.Tensor | None = None,
                 importance_cov8: torch.Tensor | None = None,
                 **kwargs) -> "TQ1Embedding":
        if result.row_scales is None:
            raise ValueError("shared embedding/head QAT requires row scales")
        return cls(
            embedding.weight.detach(), result.row_scales, codebook, quant_spec,
            profile=profile, importance_diag=importance_diag,
            importance_cov8=importance_cov8, initial_indices=result.indices,
            padding_idx=embedding.padding_idx, max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse, **kwargs)

    @property
    def num_embeddings(self) -> int:
        return self.out_features

    @property
    def embedding_dim(self) -> int:
        return self.in_features

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in {
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError("TQ1 embedding input must contain integer token ids")
        weight = self.projected_weight()
        return F.embedding(
            input_ids, weight, self.padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse)

    def linear(self, hidden: torch.Tensor) -> torch.Tensor:
        activation = (a8_ste(hidden, self.activation_mode)
                      if self.activation_mode != "none" else hidden)
        return F.linear(activation, self.projected_weight().to(activation.dtype))

    @torch.no_grad()
    def shared_health(self, previous_indices: torch.Tensor | None = None) \
            -> dict[str, Any]:
        metrics: dict[str, Any] = self.health(previous_indices)
        metrics["consumer_family"] = self.consumer_family
        metrics["embedding_rows"] = self.num_embeddings
        metrics["output_head_columns"] = self.embedding_dim
        return metrics


class TQ1OutputHead(nn.Module):
    """Parameter-free output consumer for a :class:`TQ1Embedding`."""

    def __init__(self, shared: TQ1Embedding):
        super().__init__()
        if not isinstance(shared, TQ1Embedding):
            raise TypeError("TQ1OutputHead requires a TQ1Embedding")
        object.__setattr__(self, "_shared_ref", weakref.ref(shared))
        self.in_features = shared.embedding_dim
        self.out_features = shared.num_embeddings
        self.bias = None

    @property
    def shared_weight(self) -> TQ1Embedding:
        value = self._shared_ref()
        if value is None:  # pragma: no cover - model ownership keeps it alive
            raise RuntimeError("shared TQ1 embedding was destroyed")
        return value

    @property
    def weight(self) -> nn.Parameter:
        return self.shared_weight.weight

    def __setattr__(self, name: str, value: Any) -> None:
        # Hugging Face may call tie_weights() during save.  Accept only the
        # already-shared object and do not register a duplicate Parameter.
        if name == "weight" and "_shared_ref" in self.__dict__:
            if value is not self.shared_weight.weight:
                raise ValueError("cannot retie a TQ1 output head to a different weight")
            return
        super().__setattr__(name, value)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.shared_weight.linear(hidden)


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
