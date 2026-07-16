import copy

import pytest
import torch

from bitnet_train.tq1.speculative import (
    BlockParallelDrafter, drafter_parity, evaluate_speculative_cost,
    quantize_drafter_q8, survival_weighted_distillation,
    validate_speculative_service_report, verify_greedy_lossless)


def _timing(value):
    return {"p20_ms": value * 0.9, "median_ms": value, "p80_ms": value * 1.1}


def _measurement(candidate_ms=1.0):
    candidates = []
    for block in (2, 4, 6):
        candidates.append({
            "block_size": block, "draft_block": _timing(candidate_ms),
            "target_verification": _timing(1.0),
            "scheduler_overhead": _timing(0.1),
            "acceptance_histogram": [0] * block + [100],
            "resident_drafter_bytes": 1000, "prompt_cache_reused": True,
            "reprefill_tokens": 0,
            "workload_ms_per_token": {
                "code": {"baseline": 2.0, "candidate": 1.0},
                "multi_turn": {"baseline": 2.0, "candidate": 1.5}},
        })
    return {
        "schema": 1, "model_artifact_sha256": "a" * 64,
        "backend": "cpu", "workload": "mixed-chat",
        "baseline_target_decode": _timing(2.0), "candidates": candidates,
        "predeclared_gates": {"declared_before_drafter": True,
                              "minimum_projected_speedup": 1.05,
                              "maximum_resident_drafter_bytes": 2000,
                              "maximum_workload_regression": 0.0},
        "device": {"model": "unit"}, "toolchain": {"torch": "unit"},
        "commands": ["measure-spec"], "provenance": {"git": "abc"},
    }


def test_cost_gate_blocks_negative_return_and_selects_measured_block():
    positive = evaluate_speculative_cost(_measurement())
    assert positive.eligible and positive.block_size in {2, 4, 6}
    negative = evaluate_speculative_cost(_measurement(candidate_ms=20.0))
    assert not negative.eligible and negative.block_size is None
    with pytest.raises(ValueError, match="cost gate"):
        BlockParallelDrafter(8, 16, (1, 3), 4, negative)
    malformed = _measurement()
    malformed["candidates"][0]["prompt_cache_reused"] = False
    malformed["candidates"][0]["workload_ms_per_token"] = {"invalid": 1}
    with pytest.raises(ValueError, match="multi-turn"):
        evaluate_speculative_cost(malformed)


def test_drafter_normalized_taps_survival_loss_q8_parity_and_lossless_verify():
    torch.manual_seed(7)
    gate = evaluate_speculative_cost(_measurement())
    model = BlockParallelDrafter(8, 17, (1, 3, 5), 6, gate)
    taps = [torch.randn(2, 4, 8) for _ in range(3)]
    teacher_tokens = torch.randint(0, 17, (2, gate.block_size))
    logits, survival, generated = model(taps, teacher_tokens=teacher_tokens)
    assert logits.shape == (2, gate.block_size, 17)
    assert survival.shape == generated.shape == (2, gate.block_size)
    accepted = torch.tensor([[True] * gate.block_size,
                             [True] + [False] * (gate.block_size - 1)])
    loss = survival_weighted_distillation(
        logits, survival, torch.randn_like(logits), accepted)
    loss["loss"].backward()
    assert model.survival_head.weight.grad is not None

    q8, accounting = quantize_drafter_q8(model)
    parity = drafter_parity(model.eval(), q8, taps, teacher_tokens=teacher_tokens)
    assert accounting["quantized_parameter_fraction"] > 0.9
    assert parity["top_token_agreement"] >= 0.8

    target = torch.full((1, gate.block_size + 1, 5), -10.0)
    target_tokens = (torch.arange(gate.block_size + 1) + 1) % 5
    target[0, torch.arange(gate.block_size + 1), target_tokens] = 10.0
    draft = target_tokens[:gate.block_size].clone()[None]
    assert verify_greedy_lossless(target, draft)[0] == target_tokens.tolist()
    draft[0, 1] = 4
    assert verify_greedy_lossless(target, draft)[0] == [1, 2]


def test_service_evidence_is_bound_to_cost_gate_and_quantized_parity():
    decision = evaluate_speculative_cost(_measurement())
    timing = _timing(2.0)
    candidate = _timing(1.0)
    report = {
        "schema": 1, "cost_measurement_sha256": decision.measurement_sha256,
        "model_artifact_sha256": decision.model_artifact_sha256,
        "drafter_artifact_sha256": "b" * 64, "backend": decision.backend,
        "workload": decision.workload, "block_size": decision.block_size,
        "prompt_cache_reused": True, "resident_drafter_bytes": 1000,
        "concurrency": [1, 4],
        "acceptance_histogram": [1] * (decision.block_size + 1),
        "workloads": {
            "code": {"baseline": timing, "candidate": candidate,
                     "speedup": 2.0, "reprefill_tokens": 0},
            "multi_turn": {"baseline": timing, "candidate": candidate,
                           "speedup": 2.0, "reprefill_tokens": 0},
        },
        "quantized_parity": {"max_abs_error": 0.01, "max_rel_error": 0.01,
                             "top_token_agreement": 0.99,
                             "survival_max_abs_error": 0.01,
                             "thresholds": {"max_abs_error": 0.02,
                                            "max_rel_error": 0.02,
                                            "survival_max_abs_error": 0.02,
                                            "minimum_top_token_agreement": 0.98}},
        "commands": ["service-bench"], "provenance": {"git": "abc"},
    }
    validate_speculative_service_report(
        report, decision, drafter_artifact_sha256="b" * 64)
    broken = copy.deepcopy(report)
    broken["workloads"]["multi_turn"]["reprefill_tokens"] = 1
    with pytest.raises(ValueError, match="accounting"):
        validate_speculative_service_report(
            broken, decision, drafter_artifact_sha256="b" * 64)
    broken = copy.deepcopy(report)
    broken["quantized_parity"]["survival_max_abs_error"] = 0.03
    with pytest.raises(ValueError, match="parity failed"):
        validate_speculative_service_report(
            broken, decision, drafter_artifact_sha256="b" * 64)
