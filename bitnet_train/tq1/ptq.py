"""Exact rounded-scale TQ1 PTQ projection for every format-v1 profile."""

from __future__ import annotations

import math
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


def _gptq_sweep(weight: torch.Tensor, covariance: torch.Tensor, scale: float,
                codebook: Codebook, config: PTQConfig) -> tuple[torch.Tensor, int]:
    """One deterministic increasing-K feedback sweep at an exact runtime scale."""
    blocks = weight.reshape(-1, 256)
    indices = torch.empty((blocks.shape[0], 32), dtype=torch.int64)
    fallbacks = 0
    for block in range(blocks.shape[0]):
        h = covariance[block].clone()
        damping = config.gptq_damping * float(h.diagonal().mean())
        h += torch.eye(256) * damping
        try:
            torch.linalg.cholesky(h)
        except torch.linalg.LinAlgError:
            if not config.allow_diagonal_fallback:
                raise ValueError(f"GPTQ Cholesky failed for block {block}")
            fallbacks += 1
            h = torch.diag(torch.diagonal(h).clamp_min(1e-12))
        target = blocks[block].clone()
        for group in range(32):
            begin, end = group * 8, group * 8 + 8
            hgg = h[begin:end, begin:end]
            diag = hgg.diagonal()[None]
            chosen, decoded, _ = _assign_groups(
                target[begin:end][None], scale, codebook, diag, hgg[None], config)
            indices[block, group] = chosen[0]
            if end < 256:
                error = target[begin:end] - scale * decoded[0]
                try:
                    transfer = torch.linalg.solve(hgg, h[begin:end, end:])
                except torch.linalg.LinAlgError:
                    if not config.allow_diagonal_fallback:
                        raise ValueError(f"GPTQ group solve failed at block {block}, group {group}")
                    fallbacks += 1
                    transfer = h[begin:end, end:] / hgg.diagonal().clamp_min(1e-12)[:, None]
                target[end:] -= error @ transfer
    return indices.reshape(-1), fallbacks


def _apply_gptq(weight: torch.Tensor, covariance: torch.Tensor, ordinary_scale: float,
                ordinary_indices: torch.Tensor, codebook: Codebook, config: PTQConfig) \
        -> tuple[float, torch.Tensor, dict[str, Any]]:
    ordinary_decoded = codebook.decode(ordinary_indices).float()
    candidates: list[tuple[float, int, float, torch.Tensor]] = [(
        _full_block_objective(weight, ordinary_decoded, ordinary_scale, covariance),
        0, ordinary_scale, ordinary_indices,
    )]
    sweep_scales = [ordinary_scale]
    fallbacks = 0
    first_indices, count = _gptq_sweep(weight, covariance, ordinary_scale, codebook, config)
    fallbacks += count
    first_decoded = codebook.decode(first_indices).float()
    refit = _full_block_refit(weight, first_decoded, covariance)
    if refit is not None:
        first_scale = _rounded_scale(refit, config.scale_dtype, nonzero=True)
        candidates.append((_full_block_objective(weight, first_decoded, first_scale, covariance),
                           1, first_scale, first_indices))
        sweep_scales.append(first_scale)
        if first_scale != ordinary_scale:
            second_indices, count = _gptq_sweep(
                weight, covariance, first_scale, codebook, config)
            fallbacks += count
            second_decoded = codebook.decode(second_indices).float()
            second_refit = _full_block_refit(weight, second_decoded, covariance)
            second_scale = (_rounded_scale(second_refit, config.scale_dtype, nonzero=True)
                            if second_refit is not None else first_scale)
            candidates.append((
                _full_block_objective(weight, second_decoded, second_scale, covariance),
                2, second_scale, second_indices,
            ))
            sweep_scales.append(second_scale)
    # Objective, then candidate zero, then earliest sweep.
    best = min(candidates, key=lambda item: (item[0], item[1]))
    return best[2], best[3], {
        "ordinary_objective": candidates[0][0],
        "selected_objective": best[0],
        "selected_candidate": best[1],
        "sweep_scales": sweep_scales,
        "factorization_fallbacks": fallbacks,
    }


def _solve_unit(weight: torch.Tensor, diag: torch.Tensor, cov: torch.Tensor | None,
                codebook: Codebook, config: PTQConfig) \
        -> tuple[float, torch.Tensor, torch.Tensor | None, list[float], int]:
    nonzero = bool(torch.any(weight != 0))
    if not nonzero:
        zero = torch.nonzero((codebook.decode(torch.arange(codebook.index_count)) == 0)
                             .all(-1) & codebook.legal_index_mask()).flatten()[0]
        return 0.0, torch.full((weight.shape[0],), int(zero), dtype=torch.int64), \
            (torch.zeros(weight.shape[0] // 4, dtype=torch.uint8)
             if "-a4-" in config.profile else None), [0.0], 0
    flat_diag = diag.reshape(-1)
    alpha = float((flat_diag * weight.abs().reshape(-1)).sum()
                  / flat_diag.sum().clamp_min(1e-30))
    alpha = _rounded_scale(alpha, config.scale_dtype, nonzero=True)
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
            alpha = _rounded_scale(candidate, config.scale_dtype, nonzero=True)
    assert indices is not None and decoded is not None
    # Required final rounded-scale comparison.
    candidates = []
    if "-a4-" in config.profile:
        idx_a, dec_a, _, aff_a = _assign_a4(weight, alpha, codebook, diag, cov, config)
    else:
        idx_a, dec_a, _ = _assign_groups(weight, alpha, codebook, diag, cov, config)
        aff_a = None
    candidates.append((_objective(weight, dec_a, alpha, diag, cov), alpha, idx_a, aff_a, dec_a))
    refit = _refit(weight, dec_a, diag, cov)
    if refit is not None:
        alpha_b = _rounded_scale(refit, config.scale_dtype, nonzero=True)
        if "-a4-" in config.profile:
            idx_b, dec_b, _, aff_b = _assign_a4(weight, alpha_b, codebook, diag, cov, config)
        else:
            idx_b, dec_b, _ = _assign_groups(weight, alpha_b, codebook, diag, cov, config)
            aff_b = None
        candidates.append((_objective(weight, dec_b, alpha_b, diag, cov),
                           alpha_b, idx_b, aff_b, dec_b))
    candidates.sort(key=lambda item: item[0])
    best = candidates[0]
    trace.append(best[0])
    if best[1] <= 0:
        raise ValueError("nonzero scale unit has no valid positive rounded refit")
    return best[1], best[2], best[3], trace, rejected


def project_weight(weight: torch.Tensor, codebook: Codebook, importance: Importance,
                   config: PTQConfig) -> PTQResult:
    """Project a latent [N,K] matrix directly into its declared TQ1 profile."""
    config.validate()
    source = weight.detach().float().cpu()
    if source.ndim != 2 or source.shape[1] % 256:
        raise ValueError("PTQ source must be [N,K] with K divisible by 256")
    if not torch.isfinite(source).all():
        raise ValueError("PTQ source contains NaN or infinity")
    importance.validate(source.shape[1])
    spec = layout(config.profile)
    expected_encoding = "direct_joint" if "-i-" in config.profile else \
        "product" if "-p-" in config.profile else "sign_canonical"
    if codebook.index_bits != spec.index_bits or codebook.encoding != expected_encoding:
        raise ValueError("PTQ codebook is incompatible with the requested profile")
    started = time.perf_counter()
    rows, width = source.shape
    indices = torch.empty((rows, width // 8), dtype=torch.int64)
    row_scales = torch.empty(rows, dtype=config.scale_dtype) if spec.scale_mode == "row" else None
    block_scales = (torch.empty((rows, width // 256), dtype=torch.float16)
                    if spec.scale_mode == "block256" else None)
    affine = (torch.empty((rows, width // 32), dtype=torch.uint8) if spec.affine else None)
    traces: list[list[float]] = []
    gptq_reports: list[dict[str, Any]] = []
    rejected_refits = zero_rows = 0
    for row in range(rows):
        diag, cov = _metric_for_row(source[row], importance, config.weight_metric)
        groups = source[row].reshape(-1, 8)
        if spec.scale_mode == "row":
            scale, idx, aff, trace, rejected = _solve_unit(
                groups, diag, cov, codebook, config)
            if config.gptq_feedback:
                effective_cov = _effective_cov256(source[row], importance,
                                                   config.weight_metric)
                scale, idx, gptq_report = _apply_gptq(
                    groups, effective_cov, scale, idx, codebook, config)
                gptq_reports.append({"row": row, **gptq_report})
            row_scales[row] = scale
            indices[row] = idx
            if affine is not None and aff is not None:
                affine[row] = aff
            zero_rows += int(scale == 0)
            rejected_refits += rejected
            traces.append(trace)
        else:
            row_trace = []
            for block in range(width // 256):
                group_slice = slice(block * 32, (block + 1) * 32)
                scale, idx, _, trace, rejected = _solve_unit(
                    groups[group_slice], diag[group_slice],
                    None if cov is None else cov[group_slice], codebook, config)
                if config.gptq_feedback:
                    effective_cov = _effective_cov256(source[row], importance,
                                                       config.weight_metric)[block:block + 1]
                    scale, idx, gptq_report = _apply_gptq(
                        groups[group_slice], effective_cov, scale, idx, codebook, config)
                    gptq_reports.append({"row": row, "block": block, **gptq_report})
                block_scales[row, block] = scale
                indices[row, group_slice] = idx
                rejected_refits += rejected
                row_trace.extend(trace)
            traces.append(row_trace)
    affine_blocks = affine.reshape(rows, width // 256, 8) if affine is not None else None
    payload = pack_payload(indices, config.profile, block_scales=block_scales,
                           affine_nibbles=affine_blocks)
    dequantized = dequantize_weight(payload, config.profile, codebook,
                                    row_scales=row_scales)
    delta = dequantized - source
    norm = float(source.norm())
    audit = (_candidate_audit(source, indices, row_scales, block_scales, codebook,
                              importance, config)
             if config.assignment_mode == "shortlist" and not spec.affine else None)
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
    report = {
        "profile": config.profile,
        "shape": [rows, width],
        "codebook_id": codebook.id,
        "codebook_sha256": codebook.sha256(),
        "scale_dtype": str(config.scale_dtype).removeprefix("torch."),
        "raw_bpw": spec.index_bits / 8 + (0.125 if spec.affine else 0),
        "effective_bpw": payload.numel() * 8 / source.numel()
                         + (0 if row_scales is None else row_scales.numel() * 16 / source.numel()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(delta.norm()) / max(norm, 1e-30),
        "max_abs_error": float(delta.abs().max()),
        "weighted_relative_error": math.sqrt(weighted_error / max(weighted_source, 1e-30)),
        "scalar_pattern_exact_hit_rate": float((changed == 0).float().mean()),
        "changed_trits_histogram": {
            str(value): int((changed == value).sum()) for value in range(9)
        },
        "zero_rows": zero_rows,
        "rejected_refits": rejected_refits,
        "iteration_objectives": traces,
        "elapsed_seconds": time.perf_counter() - started,
        "index_entropy": _index_entropy(indices),
        "dead_codewords": int(codebook.legal_index_mask().sum()
                              - torch.unique(indices).numel()),
        "gptq_feedback": config.gptq_feedback,
        "gptq": gptq_reports,
        "candidate_oracle": audit,
        "scale_min": (float(row_scales.float().min()) if row_scales is not None
                      else float(block_scales.float().min())),
        "scale_max": (float(row_scales.float().max()) if row_scales is not None
                      else float(block_scales.float().max())),
    }
    return PTQResult(payload, row_scales, indices, affine_blocks, dequantized, report)


def _index_entropy(indices: torch.Tensor) -> float:
    counts = torch.bincount(indices.reshape(-1), minlength=int(indices.max()) + 1).double()
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * probabilities.log2()).sum())


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
                     sample_count: int = 256) -> dict[str, Any]:
    total = selected_indices.numel()
    count = min(sample_count, total)
    positions = torch.linspace(0, total - 1, count, dtype=torch.float64).round().long().unique()
    exhaustive = replace(config, assignment_mode="exhaustive", gptq_feedback=False)
    mismatches = 0
    excess = []
    groups_per_row = selected_indices.shape[1]
    metric_cache: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for position in positions.tolist():
        row, group = divmod(position, groups_per_row)
        if row not in metric_cache:
            metric_cache[row] = _metric_for_row(source[row], importance, config.weight_metric)
        diag, cov = metric_cache[row]
        scale = (float(row_scales[row]) if row_scales is not None
                 else float(block_scales[row, group // 32]))
        target = source[row, group * 8:group * 8 + 8][None]
        diag_group = diag[group:group + 1]
        cov_group = None if cov is None else cov[group:group + 1]
        oracle_idx, _, oracle_error = _assign_groups(
            target, scale, codebook, diag_group, cov_group, exhaustive)
        selected = selected_indices[row, group:group + 1]
        selected_word = codebook.decode(selected).float()
        selected_error = _errors(target, selected_word[:, None] * scale,
                                 diag_group, cov_group)[0, 0]
        mismatches += int(selected.item() != oracle_idx.item())
        excess.append(max(0.0, float(selected_error - oracle_error[0])))
    return {
        "sample_count": len(positions),
        "mismatch_count": mismatches,
        "mismatch_rate": mismatches / max(len(positions), 1),
        "mean_excess_objective": sum(excess) / max(len(excess), 1),
        "max_excess_objective": max(excess, default=0.0),
    }
