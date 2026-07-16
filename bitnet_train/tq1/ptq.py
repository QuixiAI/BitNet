"""Exact rounded-scale TQ1 PTQ projection for every format-v1 profile."""

from __future__ import annotations

import math
import resource
import sys
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import torch

from .codebook import Codebook, base3_ids
from .oracle import dequantize_weight
from .packing import layout, pack_payload


@dataclass(frozen=True)
class Importance:
    mode: str = "uniform"
    diag: torch.Tensor | None = None
    cov8: torch.Tensor | None = None
    cov256: torch.Tensor | None = None

    def validate(self, width: int) -> None:
        if self.mode == "uniform":
            return
        if self.mode == "diagonal":
            if self.diag is None or tuple(self.diag.shape) != (width,):
                raise ValueError(f"diagonal importance must have shape [{width}]")
        elif self.mode == "covariance8":
            if self.cov8 is None or tuple(self.cov8.shape) != (width // 8, 8, 8):
                raise ValueError(f"covariance8 must have shape [{width // 8},8,8]")
        elif self.mode == "block256":
            if self.cov256 is None or tuple(self.cov256.shape) != (width // 256, 256, 256):
                raise ValueError(f"cov256 must have shape [{width // 256},256,256]")
        else:
            raise ValueError(f"unsupported importance mode {self.mode!r}")
        for tensor in (self.diag, self.cov8, self.cov256):
            if tensor is not None and not torch.isfinite(tensor).all():
                raise ValueError("importance statistics contain NaN or infinity")
        if self.cov8 is not None:
            covariance = self.cov8.detach().float().cpu()
            if not torch.allclose(
                    covariance, covariance.transpose(-1, -2),
                    atol=1e-6, rtol=1e-6):
                raise ValueError("covariance8 importance is not symmetric")
            minimum = float(torch.linalg.eigvalsh(covariance).min())
            if minimum < -1e-6:
                raise ValueError(
                    f"covariance8 importance is not positive semidefinite (min={minimum})")
        if self.cov256 is not None:
            covariance = self.cov256.detach().float().cpu()
            if not torch.allclose(
                    covariance, covariance.transpose(-1, -2),
                    atol=1e-6, rtol=1e-6):
                raise ValueError("block256 importance is not symmetric")
            if torch.any(covariance.diagonal(dim1=-2, dim2=-1) < 0):
                raise ValueError("block256 importance has a negative diagonal")


@dataclass(frozen=True)
class PTQConfig:
    profile: str
    scale_dtype: torch.dtype = torch.float16
    weight_metric: str = "iq1"
    assignment_mode: str = "shortlist"
    candidate_count: int = 32
    alternating_iterations: int = 3
    chunk_groups: int = 4096
    gptq_feedback: bool = False
    gptq_damping: float = 0.01
    allow_diagonal_fallback: bool = False

    def validate(self) -> None:
        spec = layout(self.profile)
        if self.scale_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("runtime scale dtype must be float16 or bfloat16")
        if spec.scale_mode == "block256" and self.scale_dtype != torch.float16:
            raise ValueError("embedded block scales are float16")
        if self.weight_metric not in {"uniform", "iq1"}:
            raise ValueError("weight_metric must be uniform or iq1")
        if self.assignment_mode not in {"exhaustive", "shortlist"}:
            raise ValueError("assignment_mode must be exhaustive or shortlist")
        if self.candidate_count < 1 or self.chunk_groups < 1:
            raise ValueError("candidate_count and chunk_groups must be positive")
        if not 2 <= self.alternating_iterations <= 4:
            raise ValueError("alternating_iterations must be in [2,4]")
        if self.gptq_feedback and "-a4-" in self.profile:
            raise ValueError("format-v1 GPTQ does not support A4")
        if not math.isfinite(self.gptq_damping) or self.gptq_damping < 0:
            raise ValueError("GPTQ damping must be finite and nonnegative")


@dataclass
class PTQResult:
    payload: torch.Tensor
    row_scales: torch.Tensor | None
    indices: torch.Tensor
    affine_nibbles: torch.Tensor | None
    dequantized: torch.Tensor
    report: dict[str, Any] = field(default_factory=dict)


def ternary_universe() -> torch.Tensor:
    value = torch.arange(6561, dtype=torch.int64)
    lanes = []
    for _ in range(8):
        lanes.append((value % 3 - 1).to(torch.int8))
        value //= 3
    return torch.stack(lanes, dim=1)


_CANDIDATE_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def candidate_table(codebook: Codebook, candidate_count: int) -> torch.Tensor:
    """Pattern id -> unique legal indices, shell distance then numerical index."""
    key = (codebook.sha256(), candidate_count)
    if key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[key]
    legal = torch.nonzero(codebook.legal_index_mask()).flatten()
    count = min(candidate_count, legal.numel())
    patterns = ternary_universe().float()
    codewords = codebook.decode(legal).float()
    distances = (patterns.square().sum(1, keepdim=True)
                 + codewords.square().sum(1)[None]
                 - 2 * patterns @ codewords.T).round().to(torch.int16)
    # legal is numerical-index ordered; stable sorting supplies the tie rule.
    order = torch.argsort(distances, dim=1, stable=True)[:, :count]
    table = legal[order].contiguous()
    _CANDIDATE_CACHE[key] = table
    return table


def _rounded_scale(value: float | torch.Tensor, dtype: torch.dtype, *, nonzero: bool) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("scale is negative or nonfinite")
    if not nonzero or number == 0:
        return 0.0
    return float(torch.tensor(max(number, torch.finfo(dtype).tiny), dtype=dtype).float())


def _scale_underflows(value: float | torch.Tensor, dtype: torch.dtype) -> bool:
    number = float(value)
    return 0 < number < torch.finfo(dtype).tiny


def _process_peak_rss_bytes() -> int:
    """Return the process high-water RSS using the platform's ru_maxrss units."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the other supported Unix CI hosts report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _metric_for_row(weight: torch.Tensor, importance: Importance, weight_metric: str) \
        -> tuple[torch.Tensor | None, torch.Tensor | None]:
    width = weight.numel()
    if importance.mode == "uniform":
        diag = torch.ones(width, dtype=torch.float32)
        cov = None
    elif importance.mode == "diagonal":
        diag = importance.diag.detach().float().cpu().clone()
        cov = None
    elif importance.mode == "covariance8":
        cov = importance.cov8.detach().float().cpu().clone()
        diag = cov.diagonal(dim1=-2, dim2=-1).reshape(-1).clone()
    else:
        cov256 = importance.cov256.detach().float().cpu()
        diag = cov256.diagonal(dim1=-2, dim2=-1).reshape(-1).clone()
        cov = torch.stack([
            cov256[b, start:start + 8, start:start + 8]
            for b in range(cov256.shape[0]) for start in range(0, 256, 8)
        ])
    if weight_metric == "iq1":
        blocks = weight.reshape(-1, 256)
        sigma2 = 2.0 * blocks.square().mean(1, keepdim=True)
        factor = torch.sqrt(sigma2 + blocks.square()).reshape(-1)
        diag *= factor
        if cov is not None:
            root = torch.sqrt(factor).reshape(-1, 8)
            cov *= root[:, :, None] * root[:, None, :]
    if torch.any(diag < 0) or not torch.isfinite(diag).all():
        raise ValueError("effective importance is invalid")
    return diag.reshape(-1, 8), cov


def _errors(target: torch.Tensor, candidates: torch.Tensor,
            diag: torch.Tensor, cov: torch.Tensor | None) -> torch.Tensor:
    delta = target[:, None, :] - candidates
    if cov is None:
        return (delta.square() * diag[:, None, :]).sum(-1)
    return torch.einsum("gci,gij,gcj->gc", delta, cov, delta)


def _assign_groups(target: torch.Tensor, scale: float, codebook: Codebook,
                   diag: torch.Tensor, cov: torch.Tensor | None,
                   config: PTQConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale <= 0:
        zero = torch.nonzero((codebook.decode(torch.arange(codebook.index_count)) == 0)
                             .all(dim=-1) & codebook.legal_index_mask()).flatten()[0]
        indices = torch.full((target.shape[0],), int(zero), dtype=torch.int64)
        decoded = codebook.decode(indices).float()
        return indices, decoded, _errors(target, decoded[:, None], diag, cov)[:, 0]
    initializer = torch.round(target / scale).clamp(-1, 1).to(torch.int8)
    if config.assignment_mode == "exhaustive":
        choices = torch.nonzero(codebook.legal_index_mask()).flatten()[None].expand(
            target.shape[0], -1)
    else:
        choices = candidate_table(codebook, config.candidate_count)[base3_ids(initializer)]
    # Candidate-table shell order determines membership only. The objective's
    # normative tie break is the lowest legal numerical index.
    choices = torch.sort(choices, dim=1).values
    result_indices = torch.empty(target.shape[0], dtype=torch.int64)
    result_decoded = torch.empty_like(target)
    result_errors = torch.empty(target.shape[0], dtype=torch.float32)
    for start in range(0, target.shape[0], config.chunk_groups):
        stop = min(start + config.chunk_groups, target.shape[0])
        selected = choices[start:stop]
        candidates = codebook.decode(selected).float() * scale
        error = _errors(target[start:stop], candidates, diag[start:stop],
                        None if cov is None else cov[start:stop])
        best = error.argmin(1)
        result_indices[start:stop] = selected.gather(1, best[:, None]).squeeze(1)
        result_decoded[start:stop] = codebook.decode(result_indices[start:stop]).float()
        result_errors[start:stop] = error.gather(1, best[:, None]).squeeze(1)
    return result_indices, result_decoded, result_errors


def _assign_a4(target: torch.Tensor, scale: float, codebook: Codebook,
               diag: torch.Tensor, cov: torch.Tensor | None, config: PTQConfig) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.shape[0] % 4:
        raise ValueError("A4 assignment requires four groups per subblock")
    initializer = torch.round(target / max(scale, 1e-30)).clamp(-1, 1).to(torch.int8)
    if config.assignment_mode == "exhaustive":
        choices = torch.nonzero(codebook.legal_index_mask()).flatten()[None].expand(
            target.shape[0], -1)
    else:
        choices = candidate_table(codebook, config.candidate_count)[base3_ids(initializer)]
    choices = torch.sort(choices, dim=1).values
    rho = torch.tensor([6 / 8, 7 / 8, 1.0, 9 / 8], dtype=torch.float32)
    mu = torch.tensor([0.0, 1 / 8, -1 / 8], dtype=torch.float32)
    out_indices = torch.empty(target.shape[0], dtype=torch.int64)
    out_decoded = torch.empty_like(target)
    out_errors = torch.empty(target.shape[0], dtype=torch.float32)
    nibbles = torch.empty(target.shape[0] // 4, dtype=torch.uint8)
    for subblock in range(target.shape[0] // 4):
        groups = slice(4 * subblock, 4 * subblock + 4)
        best_key = None
        best = None
        for mu_id in range(3):
            for rho_id in range(4):
                nibble = rho_id | (mu_id << 2)
                selected = choices[groups]
                normalized = rho[rho_id] * (codebook.decode(selected).float() + mu[mu_id])
                error = _errors(target[groups], normalized * scale, diag[groups],
                                None if cov is None else cov[groups])
                positions = error.argmin(1)
                indices = selected.gather(1, positions[:, None]).squeeze(1)
                errors = error.gather(1, positions[:, None]).squeeze(1)
                key = (float(errors.sum()), nibble, *indices.tolist())
                if best_key is None or key < best_key:
                    best_key = key
                    best = (indices, rho[rho_id]
                            * (codebook.decode(indices).float() + mu[mu_id]), errors, nibble)
        assert best is not None
        out_indices[groups], out_decoded[groups], out_errors[groups], nibble = best
        nibbles[subblock] = nibble
    return out_indices, out_decoded, out_errors, nibbles


def _refit(weight: torch.Tensor, decoded: torch.Tensor, diag: torch.Tensor,
           cov: torch.Tensor | None) -> float | None:
    if cov is None:
        numerator = (diag * weight * decoded).sum()
        denominator = (diag * decoded.square()).sum()
    else:
        numerator = torch.einsum("gi,gij,gj->", decoded, cov, weight)
        denominator = torch.einsum("gi,gij,gj->", decoded, cov, decoded)
    if float(numerator) <= 0 or float(denominator) <= 0:
        return None
    return float(numerator / denominator)


def _objective(weight: torch.Tensor, decoded: torch.Tensor, scale: float,
               diag: torch.Tensor, cov: torch.Tensor | None) -> float:
    return float(_errors(weight, decoded[:, None] * scale, diag, cov).sum())


def _effective_cov256(weight: torch.Tensor, importance: Importance,
                      weight_metric: str) -> torch.Tensor:
    if importance.mode != "block256" or importance.cov256 is None:
        raise ValueError("GPTQ feedback requires block256 covariance statistics")
    covariance = importance.cov256.detach().float().cpu().clone()
    if weight_metric == "iq1":
        blocks = weight.reshape(-1, 256)
        sigma2 = 2.0 * blocks.square().mean(1, keepdim=True)
        factor = torch.sqrt(sigma2 + blocks.square())
        root = torch.sqrt(factor)
        covariance *= root[:, :, None] * root[:, None, :]
    return covariance


def _full_block_objective(weight: torch.Tensor, decoded: torch.Tensor, scale: float,
                          covariance: torch.Tensor) -> float:
    delta = weight.reshape(-1, 256) - scale * decoded.reshape(-1, 256)
    return float(torch.einsum("bi,bij,bj->", delta, covariance, delta))


def _full_block_refit(weight: torch.Tensor, decoded: torch.Tensor,
                      covariance: torch.Tensor) -> float | None:
    w = weight.reshape(-1, 256)
    c = decoded.reshape(-1, 256)
    numerator = torch.einsum("bi,bij,bj->", c, covariance, w)
    denominator = torch.einsum("bi,bij,bj->", c, covariance, c)
    if float(numerator) <= 0 or float(denominator) <= 0:
        return None
    return float(numerator / denominator)


def _prepare_gptq_metric(covariance: torch.Tensor, config: PTQConfig) \
        -> tuple[torch.Tensor, list[list[torch.Tensor]], dict[str, Any]]:
    if covariance.ndim != 3 or tuple(covariance.shape[1:]) != (256, 256):
        raise ValueError("GPTQ covariance must be [blocks,256,256]")
    if not torch.isfinite(covariance).all() or not torch.allclose(
            covariance, covariance.transpose(-1, -2), atol=1e-6, rtol=1e-6):
        raise ValueError("GPTQ covariance is nonfinite or nonsymmetric")
    prepared = covariance.detach().float().cpu().clone()
    transfers: list[list[torch.Tensor]] = []
    damping_values: list[float] = []
    failure_locations: list[dict[str, int | str]] = []
    fallback_blocks: list[int] = []
    eye = torch.eye(256)
    for block in range(prepared.shape[0]):
        damping = config.gptq_damping * float(prepared[block].diagonal().mean())
        if not math.isfinite(damping) or damping < 0:
            raise ValueError(f"GPTQ damping is invalid for block {block}")
        damping_values.append(damping)
        prepared[block] += eye * damping

        failure: dict[str, int | str] | None = None
        _, info = torch.linalg.cholesky_ex(prepared[block], check_errors=False)
        if int(info):
            failure = {"block": block, "stage": "block", "info": int(info)}
        else:
            for group in range(32):
                begin, end = group * 8, group * 8 + 8
                _, info = torch.linalg.cholesky_ex(
                    prepared[block, begin:end, begin:end], check_errors=False)
                if int(info):
                    failure = {
                        "block": block, "group": group,
                        "stage": "group", "info": int(info),
                    }
                    break
        if failure is not None:
            failure_locations.append(failure)
            if not config.allow_diagonal_fallback:
                position = (f"block {block}" if failure["stage"] == "block" else
                            f"block {block}, group {failure['group']}")
                raise ValueError(f"GPTQ Cholesky failed for {position}")
            fallback_blocks.append(block)
            prepared[block] = torch.diag(
                prepared[block].diagonal().clamp_min(1e-12))

        block_transfers = []
        for group in range(32):
            begin, end = group * 8, group * 8 + 8
            if end == 256:
                block_transfers.append(torch.empty((8, 0), dtype=torch.float32))
                continue
            hgg = prepared[block, begin:end, begin:end]
            factor = torch.linalg.cholesky(hgg)
            block_transfers.append(torch.cholesky_solve(
                prepared[block, begin:end, end:], factor))
        transfers.append(block_transfers)
    return prepared, transfers, {
        "group_order": "increasing_k",
        "block_size": 256,
        "damping_factor": config.gptq_damping,
        "block_damping_values": damping_values,
        "factorization_failures": len(failure_locations),
        "factorization_failure_locations": failure_locations,
        "factorization_fallbacks": len(fallback_blocks),
        "diagonal_fallback_blocks": fallback_blocks,
    }


def _gptq_sweep(weight: torch.Tensor, covariance: torch.Tensor,
                transfers: list[list[torch.Tensor]], scale: float,
                codebook: Codebook, config: PTQConfig) -> torch.Tensor:
    """One deterministic increasing-K feedback sweep at an exact runtime scale."""
    blocks = weight.reshape(-1, 256)
    if len(transfers) != blocks.shape[0]:
        raise ValueError("GPTQ transfer inventory differs from the weight blocks")
    indices = torch.empty((blocks.shape[0], 32), dtype=torch.int64)
    for block in range(blocks.shape[0]):
        target = blocks[block].clone()
        for group in range(32):
            begin, end = group * 8, group * 8 + 8
            hgg = covariance[block, begin:end, begin:end]
            diag = hgg.diagonal()[None]
            chosen, decoded, _ = _assign_groups(
                target[begin:end][None], scale, codebook, diag, hgg[None], config)
            indices[block, group] = chosen[0]
            if end < 256:
                error = target[begin:end] - scale * decoded[0]
                target[end:] -= error @ transfers[block][group]
    return indices.reshape(-1)


def _apply_gptq(weight: torch.Tensor, covariance: torch.Tensor, ordinary_scale: float,
                ordinary_source_scale: float, ordinary_indices: torch.Tensor,
                codebook: Codebook, config: PTQConfig) \
        -> tuple[float, torch.Tensor, dict[str, Any]]:
    covariance, transfers, factorization_report = _prepare_gptq_metric(
        covariance, config)
    ordinary_decoded = codebook.decode(ordinary_indices).float()
    candidates: list[tuple[float, int, float, float, torch.Tensor]] = [(
        _full_block_objective(weight, ordinary_decoded, ordinary_scale, covariance),
        0, ordinary_scale, ordinary_source_scale, ordinary_indices,
    )]
    sweep_scales = [ordinary_scale]
    underflow_events = 0
    first_indices = _gptq_sweep(
        weight, covariance, transfers, ordinary_scale, codebook, config)
    first_decoded = codebook.decode(first_indices).float()
    refit = _full_block_refit(weight, first_decoded, covariance)
    if refit is not None:
        underflow_events += int(_scale_underflows(refit, config.scale_dtype))
        first_scale = _rounded_scale(refit, config.scale_dtype, nonzero=True)
        candidates.append((_full_block_objective(weight, first_decoded, first_scale, covariance),
                           1, first_scale, refit, first_indices))
        sweep_scales.append(first_scale)
        if first_scale != ordinary_scale:
            second_indices = _gptq_sweep(
                weight, covariance, transfers, first_scale, codebook, config)
            second_decoded = codebook.decode(second_indices).float()
            second_refit = _full_block_refit(weight, second_decoded, covariance)
            if second_refit is not None:
                underflow_events += int(_scale_underflows(second_refit, config.scale_dtype))
            second_source_scale = second_refit if second_refit is not None else refit
            second_scale = (_rounded_scale(second_source_scale, config.scale_dtype,
                                           nonzero=True))
            candidates.append((
                _full_block_objective(weight, second_decoded, second_scale, covariance),
                2, second_scale, second_source_scale, second_indices,
            ))
            sweep_scales.append(second_scale)
    # Objective, then candidate zero, then earliest sweep.
    best = min(candidates, key=lambda item: (item[0], item[1]))
    return best[2], best[4], {
        "ordinary_objective": candidates[0][0],
        "selected_objective": best[0],
        "selected_candidate": best[1],
        "selected_source_scale": best[3],
        "selected_scale_underflow": _scale_underflows(best[3], config.scale_dtype),
        "rounding_underflow_events": underflow_events,
        "sweep_scales": sweep_scales,
        **factorization_report,
    }


def _solve_unit(weight: torch.Tensor, diag: torch.Tensor, cov: torch.Tensor | None,
                codebook: Codebook, config: PTQConfig) \
        -> tuple[float, torch.Tensor, torch.Tensor | None, list[float], dict[str, Any]]:
    nonzero = bool(torch.any(weight != 0))
    if not nonzero:
        zero = torch.nonzero((codebook.decode(torch.arange(codebook.index_count)) == 0)
                             .all(-1) & codebook.legal_index_mask()).flatten()[0]
        return 0.0, torch.full((weight.shape[0],), int(zero), dtype=torch.int64), \
            (torch.zeros(weight.shape[0] // 4, dtype=torch.uint8)
             if "-a4-" in config.profile else None), [0.0], {
                 "source_scale": 0.0,
                 "selected_scale_underflow": False,
                 "rounding_underflow_events": 0,
                 "rejected_refits": 0,
             }
    underflow_events = 0

    def rounded(value: float) -> float:
        nonlocal underflow_events
        underflow_events += int(_scale_underflows(value, config.scale_dtype))
        return _rounded_scale(value, config.scale_dtype, nonzero=True)

    flat_diag = diag.reshape(-1)
    source_alpha = float((flat_diag * weight.abs().reshape(-1)).sum()
                         / flat_diag.sum().clamp_min(1e-30))
    alpha = rounded(source_alpha)
    trace = []
    rejected = 0
    indices = affine = None
    decoded = None
    for _ in range(config.alternating_iterations):
        if "-a4-" in config.profile:
            indices, decoded, _, affine = _assign_a4(weight, alpha, codebook, diag, cov, config)
        else:
            indices, decoded, _ = _assign_groups(weight, alpha, codebook, diag, cov, config)
        trace.append(_objective(weight, decoded, alpha, diag, cov))
        candidate = _refit(weight, decoded, diag, cov)
        if candidate is None:
            rejected += 1
        else:
            source_alpha = candidate
            alpha = rounded(candidate)
    assert indices is not None and decoded is not None
    # Required final rounded-scale comparison.
    candidates = []
    if "-a4-" in config.profile:
        idx_a, dec_a, _, aff_a = _assign_a4(weight, alpha, codebook, diag, cov, config)
    else:
        idx_a, dec_a, _ = _assign_groups(weight, alpha, codebook, diag, cov, config)
        aff_a = None
    candidates.append((_objective(weight, dec_a, alpha, diag, cov), 0, alpha,
                       source_alpha, idx_a, aff_a, dec_a))
    refit = _refit(weight, dec_a, diag, cov)
    if refit is not None:
        alpha_b = rounded(refit)
        if alpha_b == alpha:
            # The analytic source changed but its exact runtime representation did not.
            objective, ordinal, runtime, _, idx_a, aff_a, dec_a = candidates[0]
            candidates[0] = (objective, ordinal, runtime, refit, idx_a, aff_a, dec_a)
        else:
            if "-a4-" in config.profile:
                idx_b, dec_b, _, aff_b = _assign_a4(
                    weight, alpha_b, codebook, diag, cov, config)
            else:
                idx_b, dec_b, _ = _assign_groups(
                    weight, alpha_b, codebook, diag, cov, config)
                aff_b = None
            candidates.append((_objective(weight, dec_b, alpha_b, diag, cov), 1,
                               alpha_b, refit, idx_b, aff_b, dec_b))
    best = min(candidates, key=lambda item: (item[0], item[1]))
    trace.append(best[0])
    if best[2] <= 0:
        raise ValueError("nonzero scale unit has no valid positive rounded refit")
    return best[2], best[4], best[5], trace, {
        "source_scale": best[3],
        "selected_scale_underflow": _scale_underflows(best[3], config.scale_dtype),
        "rounding_underflow_events": underflow_events,
        "rejected_refits": rejected,
    }


def project_weight(weight: torch.Tensor, codebook: Codebook, importance: Importance,
                   config: PTQConfig) -> PTQResult:
    """Project a latent [N,K] matrix directly into its declared TQ1 profile."""
    config.validate()
    started = time.perf_counter()
    peak_memory_baseline = _process_peak_rss_bytes()
    if weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("PTQ source dtype must be float16, bfloat16, or float32")
    source = weight.detach().float().cpu()
    if source.ndim != 2 or source.shape[0] < 1 or source.shape[1] < 256 \
            or source.shape[1] % 256:
        raise ValueError("PTQ source must be [N,K] with K divisible by 256")
    if not torch.isfinite(source).all():
        raise ValueError("PTQ source contains NaN or infinity")
    importance.validate(source.shape[1])
    spec = layout(config.profile)
    expected_encoding = "direct_joint" if "-i-" in config.profile else \
        "product" if "-p-" in config.profile else "sign_canonical"
    if codebook.index_bits != spec.index_bits or codebook.encoding != expected_encoding:
        raise ValueError("PTQ codebook is incompatible with the requested profile")
    rows, width = source.shape
    indices = torch.empty((rows, width // 8), dtype=torch.int64)
    row_scales = torch.empty(rows, dtype=config.scale_dtype) if spec.scale_mode == "row" else None
    block_scales = (torch.empty((rows, width // 256), dtype=torch.float16)
                    if spec.scale_mode == "block256" else None)
    affine = (torch.empty((rows, width // 32), dtype=torch.uint8) if spec.affine else None)
    traces: list[list[float]] = []
    gptq_reports: list[dict[str, Any]] = []
    source_scales: list[float] = []
    rounded_scales: list[float] = []
    rejected_refits = underflow_count = rounding_underflow_events = 0
    zero_rows = int((source == 0).all(dim=1).sum())
    for row in range(rows):
        diag, cov = _metric_for_row(source[row], importance, config.weight_metric)
        groups = source[row].reshape(-1, 8)
        if spec.scale_mode == "row":
            scale, idx, aff, trace, solve_report = _solve_unit(
                groups, diag, cov, codebook, config)
            source_scale = float(solve_report["source_scale"])
            if config.gptq_feedback:
                effective_cov = _effective_cov256(source[row], importance,
                                                   config.weight_metric)
                scale, idx, gptq_report = _apply_gptq(
                    groups, effective_cov, scale, source_scale, idx, codebook, config)
                gptq_reports.append({"row": row, **gptq_report})
                source_scale = float(gptq_report["selected_source_scale"])
                rounding_underflow_events += int(
                    gptq_report["rounding_underflow_events"])
            row_scales[row] = scale
            indices[row] = idx
            if affine is not None and aff is not None:
                affine[row] = aff
            source_scales.append(source_scale)
            rounded_scales.append(scale)
            underflow_count += int(_scale_underflows(source_scale, config.scale_dtype))
            rounding_underflow_events += int(solve_report["rounding_underflow_events"])
            rejected_refits += int(solve_report["rejected_refits"])
            traces.append(trace)
        else:
            row_trace = []
            for block in range(width // 256):
                group_slice = slice(block * 32, (block + 1) * 32)
                scale, idx, _, trace, solve_report = _solve_unit(
                    groups[group_slice], diag[group_slice],
                    None if cov is None else cov[group_slice], codebook, config)
                source_scale = float(solve_report["source_scale"])
                if config.gptq_feedback:
                    effective_cov = _effective_cov256(source[row], importance,
                                                       config.weight_metric)[block:block + 1]
                    scale, idx, gptq_report = _apply_gptq(
                        groups[group_slice], effective_cov, scale, source_scale,
                        idx, codebook, config)
                    gptq_reports.append({"row": row, "block": block, **gptq_report})
                    source_scale = float(gptq_report["selected_source_scale"])
                    rounding_underflow_events += int(
                        gptq_report["rounding_underflow_events"])
                block_scales[row, block] = scale
                indices[row, group_slice] = idx
                source_scales.append(source_scale)
                rounded_scales.append(scale)
                underflow_count += int(_scale_underflows(source_scale, config.scale_dtype))
                rounding_underflow_events += int(
                    solve_report["rounding_underflow_events"])
                rejected_refits += int(solve_report["rejected_refits"])
                row_trace.extend(trace)
            traces.append(row_trace)
    affine_blocks = affine.reshape(rows, width // 256, 8) if affine is not None else None
    payload = pack_payload(indices, config.profile, block_scales=block_scales,
                           affine_nibbles=affine_blocks)
    dequantized = dequantize_weight(payload, config.profile, codebook,
                                    row_scales=row_scales)
    delta = dequantized - source
    norm = float(source.norm())
    audit = _candidate_audit(
        source, indices, row_scales, block_scales, codebook, importance, config,
        affine_nibbles=affine)
    decoded_codes = codebook.decode(indices).reshape_as(source)
    scalar = torch.empty_like(decoded_codes)
    if row_scales is not None:
        divisor = torch.where(row_scales.float() > 0, row_scales.float(),
                              torch.ones_like(row_scales.float()))[:, None]
        scalar = torch.round(source / divisor).clamp(-1, 1).to(torch.int8)
    else:
        divisor = torch.where(block_scales.float() > 0, block_scales.float(),
                              torch.ones_like(block_scales.float()))
        scalar = torch.round(source.reshape(rows, -1, 256) / divisor[..., None]) \
            .clamp(-1, 1).to(torch.int8).reshape_as(source)
    changed = (decoded_codes != scalar).reshape(-1, 8).sum(1)
    weighted_error, weighted_source = _weighted_norms(source, delta, importance,
                                                       config.weight_metric)
    usage = _codeword_usage(indices, codebook)
    source_scale_range = _scale_range(source_scales)
    rounded_scale_range = _scale_range(rounded_scales)
    factorization_fallbacks = sum(
        int(value["factorization_fallbacks"]) for value in gptq_reports)
    fallback_scale_units = sum(
        int(value["factorization_fallbacks"] > 0) for value in gptq_reports)
    zero_scale_units = sum(int(value == 0) for value in rounded_scales)
    elapsed_seconds = time.perf_counter() - started
    peak_memory_bytes = max(peak_memory_baseline, _process_peak_rss_bytes())
    report = {
        "profile": config.profile,
        "shape": [rows, width],
        "logical_shape": [rows, width],
        "codebook_id": codebook.id,
        "codebook_sha256": codebook.sha256(),
        "scale_dtype": str(config.scale_dtype).removeprefix("torch."),
        "source_scale_range": source_scale_range,
        "rounded_scale_range": rounded_scale_range,
        "raw_bpw": spec.index_bits / 8 + (0.125 if spec.affine else 0),
        "effective_bpw": payload.numel() * 8 / source.numel()
                         + (0 if row_scales is None else row_scales.numel() * 16 / source.numel()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(delta.norm()) / max(norm, 1e-30),
        "max_abs_error": float(delta.abs().max()),
        "weighted_relative_error": math.sqrt(weighted_error / max(weighted_source, 1e-30)),
        "scalar_pattern_exact_hit_rate": float((changed == 0).float().mean()),
        "mean_changed_trits_per_group": float(changed.float().mean()),
        "changed_trits_histogram": {
            str(value): int((changed == value).sum()) for value in range(9)
        },
        "underflow_count": underflow_count,
        "rounding_underflow_events": rounding_underflow_events,
        "zero_rows": zero_rows,
        "zero_scale_units": zero_scale_units,
        "rejected_refits": rejected_refits,
        "fallback_count": factorization_fallbacks,
        "factorization_fallbacks": factorization_fallbacks,
        "fallback_scale_units": fallback_scale_units,
        "iteration_objectives": traces,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_baseline_bytes": peak_memory_baseline,
        "peak_memory_delta_bytes": max(0, peak_memory_bytes - peak_memory_baseline),
        "peak_memory_scope": "process_high_water_rss",
        "peak_memory_measurement": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "index_entropy": usage["entropy_bits"],
        "codeword_entropy_bits": usage["entropy_bits"],
        "dead_codewords": usage["dead_codewords"],
        "top_codeword_usages": usage["top_usages"],
        "gptq_feedback": config.gptq_feedback,
        "gptq": gptq_reports,
        "candidate_oracle": audit,
        "scale_min": rounded_scale_range["min"],
        "scale_median": rounded_scale_range["median"],
        "scale_max": rounded_scale_range["max"],
    }
    return PTQResult(payload, row_scales, indices, affine_blocks, dequantized, report)


def _scale_range(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("PTQ scale inventory is empty")
    tensor = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(tensor).all() or torch.any(tensor < 0):
        raise ValueError("PTQ scale inventory is invalid")
    return {
        "min": float(tensor.min()),
        "median": float(tensor.median()),
        "max": float(tensor.max()),
    }


def _codeword_usage(indices: torch.Tensor, codebook: Codebook, *, top_k: int = 10) \
        -> dict[str, Any]:
    counts = torch.bincount(indices.reshape(-1), minlength=codebook.index_count).double()
    legal = codebook.legal_index_mask()
    if counts.numel() != legal.numel() or torch.any(counts[~legal] != 0):
        raise ValueError("PTQ result contains an illegal codebook index")
    legal_counts = counts[legal]
    probabilities = legal_counts[legal_counts > 0] / counts.sum()
    entropy = float(-(probabilities * probabilities.log2()).sum())
    ordered = sorted(
        (int(index) for index in torch.nonzero(legal).flatten()),
        key=lambda index: (-float(counts[index]), index))
    top = [
        {
            "index": index,
            "count": int(counts[index]),
            "fraction": float(counts[index] / counts.sum()),
        }
        for index in ordered[:top_k] if counts[index] > 0
    ]
    return {
        "entropy_bits": entropy,
        "dead_codewords": int((legal_counts == 0).sum()),
        "top_usages": top,
    }


def _weighted_norms(source: torch.Tensor, delta: torch.Tensor, importance: Importance,
                    weight_metric: str) -> tuple[float, float]:
    error = energy = 0.0
    for row in range(source.shape[0]):
        if importance.mode == "block256" and importance.cov256 is not None:
            covariance = _effective_cov256(source[row], importance, weight_metric)
            d = delta[row].reshape(-1, 256)
            w = source[row].reshape(-1, 256)
            error += float(torch.einsum("bi,bij,bj->", d, covariance, d))
            energy += float(torch.einsum("bi,bij,bj->", w, covariance, w))
        else:
            diag, cov = _metric_for_row(source[row], importance, weight_metric)
            d = delta[row].reshape(-1, 8)
            w = source[row].reshape(-1, 8)
            error += _objective(d, torch.zeros_like(d), 0.0, diag, cov)
            energy += _objective(w, torch.zeros_like(w), 0.0, diag, cov)
    return error, energy


def _candidate_audit(source: torch.Tensor, selected_indices: torch.Tensor,
                     row_scales: torch.Tensor | None, block_scales: torch.Tensor | None,
                     codebook: Codebook, importance: Importance, config: PTQConfig,
                     sample_count: int = 256,
                     affine_nibbles: torch.Tensor | None = None) -> dict[str, Any]:
    affine = "-a4-" in config.profile
    if affine:
        expected = (selected_indices.shape[0], selected_indices.shape[1] // 4)
        if affine_nibbles is None or tuple(affine_nibbles.shape) != expected:
            raise ValueError("A4 candidate audit requires the complete affine inventory")
        total = affine_nibbles.numel()
        sample_count = min(sample_count, 32)
        sample_unit = "affine_subblock32"
    else:
        if affine_nibbles is not None:
            raise ValueError("strict candidate audit received unexpected affine nibbles")
        total = selected_indices.numel()
        sample_unit = "codeword_group8"
    if config.assignment_mode == "exhaustive":
        return {
            "comparison_performed": False,
            "reason": "assignment path is already the exhaustive oracle",
            "sample_unit": sample_unit,
            "population_count": total,
            "sample_count": 0,
            "mismatch_count": 0,
            "mismatch_rate": 0.0,
            "mean_excess_objective": 0.0,
            "max_excess_objective": 0.0,
        }
    count = min(sample_count, total)
    positions = torch.linspace(0, total - 1, count, dtype=torch.float64).round().long().unique()
    shortlist = replace(config, assignment_mode="shortlist", gptq_feedback=False)
    exhaustive = replace(config, assignment_mode="exhaustive", gptq_feedback=False)
    mismatches = 0
    excess = []
    groups_per_row = selected_indices.shape[1]
    metric_cache: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for position in positions.tolist():
        if affine:
            row, subblock = divmod(position, groups_per_row // 4)
            group = subblock * 4
            group_slice = slice(group, group + 4)
        else:
            row, group = divmod(position, groups_per_row)
            group_slice = slice(group, group + 1)
        if row not in metric_cache:
            metric_cache[row] = _metric_for_row(source[row], importance, config.weight_metric)
        diag, cov = metric_cache[row]
        scale = (float(row_scales[row]) if row_scales is not None
                 else float(block_scales[row, group // 32]))
        target = source[row].reshape(-1, 8)[group_slice]
        diag_group = diag[group_slice]
        cov_group = None if cov is None else cov[group_slice]
        if affine:
            short_idx, _, short_error, short_nibble = _assign_a4(
                target, scale, codebook, diag_group, cov_group, shortlist)
            oracle_idx, _, oracle_error, oracle_nibble = _assign_a4(
                target, scale, codebook, diag_group, cov_group, exhaustive)
            mismatches += int(
                int(short_nibble[0]) != int(oracle_nibble[0])
                or not torch.equal(short_idx, oracle_idx))
            objective_excess = float(short_error.sum() - oracle_error.sum())
        else:
            short_idx, _, short_error = _assign_groups(
                target, scale, codebook, diag_group, cov_group, shortlist)
            oracle_idx, _, oracle_error = _assign_groups(
                target, scale, codebook, diag_group, cov_group, exhaustive)
            mismatches += int(short_idx.item() != oracle_idx.item())
            objective_excess = float(short_error[0] - oracle_error[0])
        excess.append(max(0.0, objective_excess))
    return {
        "comparison_performed": True,
        "sample_unit": sample_unit,
        "population_count": total,
        "sample_count": len(positions),
        "mismatch_count": mismatches,
        "mismatch_rate": mismatches / max(len(positions), 1),
        "mean_excess_objective": sum(excess) / max(len(excess), 1),
        "max_excess_objective": max(excess, default=0.0),
    }
