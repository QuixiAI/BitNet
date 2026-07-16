"""Cost-gated, lossless-greedy speculative drafting reference for QI-6."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


SPECULATIVE_COST_SCHEMA = 1
SPECULATIVE_SERVICE_SCHEMA = 1


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _full_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) \
            or (float(value) < 0 if allow_zero else float(value) <= 0):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
    return float(value)


def _timing(value: Any, name: str) -> float:
    if not isinstance(value, Mapping) or set(value) != {"p20_ms", "median_ms", "p80_ms"}:
        raise ValueError(f"{name} timing schema is invalid")
    numbers = [_positive(value[key], f"{name}.{key}")
               for key in ("p20_ms", "median_ms", "p80_ms")]
    if not numbers[0] <= numbers[1] <= numbers[2]:
        raise ValueError(f"{name} timing quantiles are not monotonic")
    return numbers[1]


@dataclass(frozen=True)
class SpeculativeGateDecision:
    eligible: bool
    measurement_sha256: str
    model_artifact_sha256: str
    backend: str
    workload: str
    block_size: int | None
    projected_speedup: float
    minimum_projected_speedup: float
    maximum_resident_drafter_bytes: int
    maximum_workload_regression: float
    reason: str


def evaluate_speculative_cost(measurement: Mapping[str, Any]) -> SpeculativeGateDecision:
    """Select a block only when measured total cost predicts positive return."""
    fields = {"schema", "model_artifact_sha256", "backend", "workload",
              "baseline_target_decode", "candidates", "predeclared_gates",
              "device", "toolchain", "commands", "provenance"}
    if not isinstance(measurement, Mapping) or set(measurement) != fields \
            or measurement["schema"] != SPECULATIVE_COST_SCHEMA:
        raise ValueError("speculative cost measurement has an invalid schema")
    model_sha = _full_sha(measurement["model_artifact_sha256"],
                          "model_artifact_sha256")
    for key in ("backend", "workload"):
        if not isinstance(measurement[key], str) or not measurement[key]:
            raise ValueError(f"speculative cost {key} must be nonempty")
    baseline = _timing(measurement["baseline_target_decode"], "baseline_target_decode")
    gates = measurement["predeclared_gates"]
    gate_fields = {"declared_before_drafter", "minimum_projected_speedup",
                   "maximum_resident_drafter_bytes", "maximum_workload_regression"}
    if not isinstance(gates, Mapping) or set(gates) != gate_fields \
            or gates["declared_before_drafter"] is not True:
        raise ValueError("speculative gates were not predeclared")
    minimum_speedup = _positive(gates["minimum_projected_speedup"],
                                "minimum_projected_speedup")
    if minimum_speedup <= 1:
        raise ValueError("minimum projected speedup must require positive return")
    max_bytes = gates["maximum_resident_drafter_bytes"]
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("maximum resident drafter bytes is invalid")
    max_regression = _positive(gates["maximum_workload_regression"],
                               "maximum_workload_regression", allow_zero=True)
    candidates = measurement["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("speculative cost measurement has no block candidates")
    for section in ("device", "toolchain", "provenance"):
        if not isinstance(measurement[section], Mapping) or not measurement[section]:
            raise ValueError(f"speculative cost {section} must be nonempty")
    if not isinstance(measurement["commands"], list) or not measurement["commands"] \
            or not all(isinstance(value, str) and value for value in measurement["commands"]):
        raise ValueError("speculative cost commands must be nonempty")
    valid: list[tuple[float, int]] = []
    block_inventory: set[int] = set()
    candidate_fields = {"block_size", "draft_block", "target_verification",
                        "scheduler_overhead", "acceptance_histogram",
                        "resident_drafter_bytes", "prompt_cache_reused",
                        "reprefill_tokens", "workload_ms_per_token"}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or set(candidate) != candidate_fields:
            raise ValueError(f"speculative candidate {index} has an invalid schema")
        block = candidate["block_size"]
        if isinstance(block, bool) or not isinstance(block, int) or not 2 <= block <= 8:
            raise ValueError("speculative block candidates must be between two and eight")
        if block in block_inventory:
            raise ValueError("speculative block candidates must be unique")
        block_inventory.add(block)
        draft_ms = _timing(candidate["draft_block"], f"candidates[{index}].draft_block")
        verify_ms = _timing(
            candidate["target_verification"], f"candidates[{index}].target_verification")
        scheduler_ms = _timing(
            candidate["scheduler_overhead"], f"candidates[{index}].scheduler_overhead")
        histogram = candidate["acceptance_histogram"]
        if not isinstance(histogram, list) or len(histogram) != block + 1 \
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in histogram) or sum(histogram) < 1:
            raise ValueError("speculative acceptance histogram is invalid")
        accepted = sum(count * value for value, count in enumerate(histogram)) \
            / sum(histogram)
        emitted = accepted + 1.0  # lossless verifier emits one target/bonus token
        projected = (draft_ms + verify_ms + scheduler_ms) / emitted
        resident = candidate["resident_drafter_bytes"]
        if isinstance(resident, bool) or not isinstance(resident, int) or resident < 1:
            raise ValueError("speculative drafter residency is invalid")
        cache_reused = candidate["prompt_cache_reused"]
        if not isinstance(cache_reused, bool):
            raise ValueError("speculative prompt-cache reuse must be boolean")
        reprefill = candidate["reprefill_tokens"]
        if isinstance(reprefill, bool) or not isinstance(reprefill, int) or reprefill < 0:
            raise ValueError("speculative re-prefill tokens must be a nonnegative integer")
        workloads = candidate["workload_ms_per_token"]
        if not isinstance(workloads, Mapping) or "multi_turn" not in workloads \
                or not workloads:
            raise ValueError("speculative candidate must report multi-turn service cost")
        regressions = []
        for name, pair in workloads.items():
            if not isinstance(name, str) or not name or not isinstance(pair, Mapping) \
                    or set(pair) != {"baseline", "candidate"}:
                raise ValueError("speculative workload cost schema is invalid")
            before = _positive(pair["baseline"], f"{name}.baseline")
            after = _positive(pair["candidate"], f"{name}.candidate")
            regressions.append(after / before - 1.0)
        speedup = baseline / projected
        if cache_reused and reprefill == 0 \
                and speedup >= minimum_speedup and resident <= max_bytes \
                and max(regressions) <= max_regression:
            valid.append((projected, block))
    if 4 not in block_inventory:
        raise ValueError("speculative cost sweep must include block size four")
    digest = _canonical_sha256(measurement)
    if not valid:
        return SpeculativeGateDecision(
            False, digest, model_sha, measurement["backend"], measurement["workload"],
            None, 1.0, minimum_speedup, max_bytes, max_regression,
            "no measured block satisfies total-cost, memory, and service gates")
    projected, block = min(valid)
    return SpeculativeGateDecision(
        True, digest, model_sha, measurement["backend"], measurement["workload"],
        block, baseline / projected, minimum_speedup, max_bytes, max_regression,
        "measured total-cost model predicts positive return")


class BlockParallelDrafter(nn.Module):
    """Small multi-tap drafter with parallel bases and sequential correction."""

    def __init__(self, hidden_size: int, vocab_size: int, tap_layers: Sequence[int],
                 bottleneck: int, gate: SpeculativeGateDecision):
        super().__init__()
        if not gate.eligible or gate.block_size is None:
            raise ValueError("drafter construction is blocked by the measured cost gate")
        if hidden_size < 2 or vocab_size < 2 or bottleneck < 2:
            raise ValueError("drafter dimensions are invalid")
        if len(tap_layers) < 2 or len(set(tap_layers)) != len(tap_layers) \
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in tap_layers):
            raise ValueError("drafter requires several unique normalized target taps")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tap_layers = tuple(int(value) for value in tap_layers)
        self.bottleneck = bottleneck
        self.block_size = gate.block_size
        self.gate_measurement_sha256 = gate.measurement_sha256
        self.tap_norms = nn.ModuleList(nn.RMSNorm(hidden_size) for _ in tap_layers)
        self.context_projection = nn.Linear(hidden_size * len(tap_layers), bottleneck)
        self.slot_embedding = nn.Embedding(self.block_size, bottleneck)
        self.start_embedding = nn.Embedding(1, bottleneck)
        self.token_embedding = nn.Embedding(vocab_size, bottleneck)
        self.block_parallel_head = nn.Linear(bottleneck, vocab_size)
        self.sequential_context = nn.Linear(bottleneck, bottleneck, bias=False)
        self.sequential_token = nn.Linear(bottleneck, bottleneck, bias=False)
        self.sequential_state = nn.Linear(bottleneck, bottleneck, bias=False)
        self.sequential_head = nn.Linear(bottleneck, vocab_size, bias=False)
        self.survival_head = nn.Linear(bottleneck, 1)

    def forward(self, hidden_taps: Sequence[torch.Tensor], *,
                teacher_tokens: torch.Tensor | None = None) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(hidden_taps) != len(self.tap_layers):
            raise ValueError("drafter hidden tap inventory mismatch")
        normalized = []
        batch = None
        for norm, hidden in zip(self.tap_norms, hidden_taps):
            if hidden.ndim not in {2, 3} or hidden.shape[-1] != self.hidden_size:
                raise ValueError("drafter taps must be [B,H] or [B,T,H]")
            last = hidden[:, -1] if hidden.ndim == 3 else hidden
            if batch is None:
                batch = last.shape[0]
            if last.shape[0] != batch:
                raise ValueError("drafter tap batch sizes differ")
            normalized.append(norm(last))
        assert batch is not None
        context = self.context_projection(torch.cat(normalized, -1))
        slots = torch.arange(self.block_size, device=context.device)
        bases = torch.tanh(context[:, None, :] + self.slot_embedding(slots)[None])
        parallel_logits = self.block_parallel_head(bases)
        if teacher_tokens is not None and (teacher_tokens.shape != (batch, self.block_size)
                                           or teacher_tokens.dtype != torch.int64
                                           or torch.any(teacher_tokens < 0)
                                           or torch.any(teacher_tokens >= self.vocab_size)):
            raise ValueError("teacher tokens must be int64 [B,block]")
        previous_token = torch.zeros(batch, dtype=torch.int64, device=context.device)
        previous_state = self.start_embedding(previous_token)
        logits, survival, generated = [], [], []
        for position in range(self.block_size):
            state = torch.tanh(
                self.sequential_context(bases[:, position])
                + self.sequential_token(self.token_embedding(previous_token))
                + self.sequential_state(previous_state))
            current = parallel_logits[:, position] + self.sequential_head(state)
            token = (teacher_tokens[:, position] if teacher_tokens is not None
                     else current.argmax(-1))
            logits.append(current)
            survival.append(self.survival_head(state).squeeze(-1))
            generated.append(token)
            previous_state, previous_token = state, token
        return torch.stack(logits, 1), torch.stack(survival, 1), torch.stack(generated, 1)


def survival_weighted_distillation(
        draft_logits: torch.Tensor, survival_logits: torch.Tensor,
        teacher_logits: torch.Tensor, accepted_prefix: torch.Tensor, *,
        temperature: float = 1.0, survival_loss_weight: float = 0.1) \
        -> dict[str, torch.Tensor]:
    if draft_logits.shape != teacher_logits.shape or draft_logits.ndim != 3 \
            or survival_logits.shape != draft_logits.shape[:2] \
            or accepted_prefix.shape != draft_logits.shape[:2] \
            or accepted_prefix.dtype != torch.bool:
        raise ValueError("survival distillation tensor shapes/dtypes are invalid")
    if temperature <= 0 or survival_loss_weight < 0:
        raise ValueError("survival distillation hyperparameters are invalid")
    # Once a token is rejected, later tokens are unreachable in that round.
    if torch.any(accepted_prefix[:, 1:] & ~accepted_prefix[:, :-1]):
        raise ValueError("accepted_prefix must be monotonic")
    reachable = torch.cat((
        torch.ones_like(accepted_prefix[:, :1]), accepted_prefix[:, :-1]), 1)
    log_student = F.log_softmax(draft_logits.float() / temperature, -1)
    log_teacher = F.log_softmax(teacher_logits.detach().float() / temperature, -1)
    kl = (log_teacher.exp() * (log_teacher - log_student)).sum(-1) * temperature**2
    kd = (kl * reachable).sum() / reachable.sum().clamp_min(1)
    survival = F.binary_cross_entropy_with_logits(
        survival_logits.float(), accepted_prefix.float())
    total = kd + survival_loss_weight * survival
    return {"loss": total, "distillation": kd, "survival": survival,
            "reachable_fraction": reachable.float().mean()}


def verify_greedy_lossless(target_logits: torch.Tensor, draft_tokens: torch.Tensor) \
        -> list[list[int]]:
    """Accept a greedy draft prefix, then emit the target mismatch/bonus token."""
    if target_logits.ndim != 3 or draft_tokens.ndim != 2 \
            or target_logits.shape[0] != draft_tokens.shape[0] \
            or target_logits.shape[1] != draft_tokens.shape[1] + 1 \
            or draft_tokens.dtype != torch.int64:
        raise ValueError("greedy verification requires logits [B,block+1,V] and drafts [B,block]")
    if not torch.isfinite(target_logits).all() or torch.any(draft_tokens < 0) \
            or torch.any(draft_tokens >= target_logits.shape[-1]):
        raise ValueError("greedy verification logits/tokens are invalid")
    target = target_logits.detach().float().argmax(-1).cpu()
    drafts = draft_tokens.detach().cpu()
    outputs = []
    for target_row, draft_row in zip(target, drafts):
        emitted = []
        for position, token in enumerate(draft_row.tolist()):
            if token != int(target_row[position]):
                emitted.append(int(target_row[position]))
                break
            emitted.append(token)
        else:
            emitted.append(int(target_row[len(draft_row)]))
        outputs.append(emitted)
    return outputs


def _q8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = weight.detach().float()
    maximum = value.abs().amax(-1)
    scales = (maximum / 127).to(torch.float16)
    denominator = torch.where(scales.float() > 0, scales.float(),
                              torch.ones_like(scales.float()))
    codes = torch.round(value / denominator[..., None]).clamp(-127, 127).to(torch.int8)
    codes[maximum == 0] = 0
    return codes, scales


class PackedQ8Linear(nn.Module):
    def __init__(self, source: nn.Linear):
        super().__init__()
        codes, scales = _q8(source.weight)
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", None if source.bias is None
                             else source.bias.detach().to(torch.float16))
        self.in_features, self.out_features = source.in_features, source.out_features

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.codes.float() * self.scales.float()[:, None]
        bias = None if self.bias is None else self.bias.to(value.dtype)
        return F.linear(value, weight.to(value.dtype), bias)


class PackedQ8Embedding(nn.Module):
    def __init__(self, source: nn.Embedding):
        super().__init__()
        codes, scales = _q8(source.weight)
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.num_embeddings, self.embedding_dim = source.num_embeddings, source.embedding_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        weight = self.codes.float() * self.scales.float()[:, None]
        return F.embedding(ids, weight)


def quantize_drafter_q8(drafter: BlockParallelDrafter) \
        -> tuple[BlockParallelDrafter, dict[str, Any]]:
    quantized = copy.deepcopy(drafter).eval()
    quantized_parameters = original_parameters = 0
    resident_bytes = 0

    def replace(module: nn.Module) -> None:
        nonlocal quantized_parameters, original_parameters, resident_bytes
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                original_parameters += child.weight.numel()
                quantized_parameters += child.weight.numel()
                replacement = PackedQ8Linear(child)
                resident_bytes += sum(value.numel() * value.element_size()
                                      for value in replacement.buffers())
                setattr(module, name, replacement)
            elif isinstance(child, nn.Embedding):
                original_parameters += child.weight.numel()
                quantized_parameters += child.weight.numel()
                replacement = PackedQ8Embedding(child)
                resident_bytes += sum(value.numel() * value.element_size()
                                      for value in replacement.buffers())
                setattr(module, name, replacement)
            else:
                replace(child)
    replace(quantized)
    remaining = sum(value.numel() for value in quantized.parameters())
    original_total = sum(value.numel() for value in drafter.parameters())
    remaining_bytes = sum(value.numel() * value.element_size()
                          for value in quantized.parameters())
    total_bytes = remaining_bytes + sum(
        value.numel() * value.element_size() for value in quantized.buffers())
    return quantized, {
        "format": "q8_row_fp16_scale", "quantized_parameters": quantized_parameters,
        "original_parameters": original_total,
        "remaining_float_parameters": remaining,
        "remaining_float_parameter_bytes": remaining_bytes,
        "quantized_parameter_fraction": quantized_parameters / max(original_total, 1),
        "resident_quantized_module_bytes": resident_bytes,
        "total_resident_tensor_bytes": total_bytes,
    }


@torch.inference_mode()
def drafter_parity(reference: BlockParallelDrafter, quantized: BlockParallelDrafter,
                   hidden_taps: Sequence[torch.Tensor], *,
                   teacher_tokens: torch.Tensor | None = None) -> dict[str, float]:
    ref_logits, ref_survival, _ = reference(hidden_taps, teacher_tokens=teacher_tokens)
    got_logits, got_survival, _ = quantized(hidden_taps, teacher_tokens=teacher_tokens)
    absolute = (got_logits - ref_logits).abs()
    denominator = float(ref_logits.abs().max()) + 1e-9
    return {
        "max_abs_error": float(absolute.max()),
        "max_rel_error": float(absolute.max()) / denominator,
        "top_token_agreement": float(
            (got_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean()),
        "survival_max_abs_error": float((got_survival - ref_survival).abs().max()),
    }


def validate_speculative_service_report(report: Mapping[str, Any], decision: SpeculativeGateDecision,
                                        *, drafter_artifact_sha256: str) -> None:
    fields = {"schema", "cost_measurement_sha256", "model_artifact_sha256",
              "drafter_artifact_sha256", "backend", "workload", "block_size",
              "prompt_cache_reused", "resident_drafter_bytes", "concurrency",
              "acceptance_histogram", "workloads", "quantized_parity",
              "commands", "provenance"}
    if not decision.eligible or not isinstance(report, Mapping) or set(report) != fields \
            or report["schema"] != SPECULATIVE_SERVICE_SCHEMA:
        raise ValueError("speculative service report is invalid or not cost-gated")
    if report["cost_measurement_sha256"] != decision.measurement_sha256 \
            or report["model_artifact_sha256"] != decision.model_artifact_sha256 \
            or report["drafter_artifact_sha256"] != _full_sha(
                drafter_artifact_sha256, "drafter_artifact_sha256") \
            or report["backend"] != decision.backend or report["workload"] != decision.workload \
            or report["block_size"] != decision.block_size:
        raise ValueError("speculative service artifact/gate identity mismatch")
    if report["prompt_cache_reused"] is not True:
        raise ValueError("speculative service must reuse the prompt cache")
    if isinstance(report["resident_drafter_bytes"], bool) \
            or not isinstance(report["resident_drafter_bytes"], int) \
            or not 1 <= report["resident_drafter_bytes"] \
            <= decision.maximum_resident_drafter_bytes:
        raise ValueError("speculative resident drafter bytes are invalid")
    concurrency = report["concurrency"]
    if not isinstance(concurrency, list) or len(concurrency) < 2 \
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                   for value in concurrency) or len(set(concurrency)) != len(concurrency):
        raise ValueError("speculative service needs multiple concurrency levels")
    histogram = report["acceptance_histogram"]
    if not isinstance(histogram, list) or len(histogram) != decision.block_size + 1 \
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in histogram) or sum(histogram) < 1:
        raise ValueError("speculative service acceptance distribution is invalid")
    workloads = report["workloads"]
    if not isinstance(workloads, Mapping) or "multi_turn" not in workloads or len(workloads) < 2:
        raise ValueError("speculative service needs multi-turn and another workload")
    for name, values in workloads.items():
        if not isinstance(values, Mapping) or set(values) != {
                "baseline", "candidate", "speedup", "reprefill_tokens"}:
            raise ValueError(f"speculative workload {name} schema is invalid")
        baseline = _timing(values["baseline"], f"workloads.{name}.baseline")
        candidate = _timing(values["candidate"], f"workloads.{name}.candidate")
        speedup = _positive(values["speedup"], f"workloads.{name}.speedup")
        reprefill = values["reprefill_tokens"]
        if not math.isclose(speedup, baseline / candidate, rel_tol=1e-6) \
                or isinstance(reprefill, bool) or not isinstance(reprefill, int) \
                or reprefill != 0 \
                or candidate / baseline - 1.0 > decision.maximum_workload_regression:
            raise ValueError(f"speculative workload {name} accounting is invalid")
    parity = report["quantized_parity"]
    if not isinstance(parity, Mapping) or set(parity) != {
            "max_abs_error", "max_rel_error", "top_token_agreement",
            "survival_max_abs_error", "thresholds"}:
        raise ValueError("speculative quantized parity schema is invalid")
    thresholds = parity["thresholds"]
    for name in ("max_abs_error", "max_rel_error", "top_token_agreement",
                 "survival_max_abs_error"):
        _positive(parity[name], f"quantized_parity.{name}", allow_zero=True)
    if float(parity["top_token_agreement"]) > 1:
        raise ValueError("speculative quantized top-token agreement is invalid")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
            "max_abs_error", "max_rel_error", "minimum_top_token_agreement",
            "survival_max_abs_error"} \
            or not 0 <= _positive(
                thresholds["minimum_top_token_agreement"],
                "minimum_top_token_agreement", allow_zero=True) <= 1 \
            or _positive(thresholds["max_abs_error"], "max_abs_error", allow_zero=True) < 0 \
            or _positive(thresholds["max_rel_error"], "max_rel_error", allow_zero=True) < 0 \
            or _positive(thresholds["survival_max_abs_error"],
                         "survival_max_abs_error", allow_zero=True) < 0 \
            or float(parity["max_abs_error"]) > float(thresholds["max_abs_error"]) \
            or float(parity["max_rel_error"]) > float(thresholds["max_rel_error"]) \
            or float(parity["survival_max_abs_error"]) \
            > float(thresholds["survival_max_abs_error"]) \
            or float(parity["top_token_agreement"]) \
            < float(thresholds["minimum_top_token_agreement"]):
        raise ValueError("speculative quantized drafter parity failed")
    if not isinstance(report["commands"], list) or not report["commands"] \
            or not all(isinstance(value, str) and value for value in report["commands"]):
        raise ValueError("speculative service commands must be nonempty")
    if not isinstance(report["provenance"], Mapping) or not report["provenance"]:
        raise ValueError("speculative service provenance must be nonempty")
