import json

import pytest
import torch

from bitnet_train.tq1.kv_cache import (
    KVCalibrationContract, KVMeanCollector, KVRuntimeIdentity,
    attention_reference, dequantize_kv_pair, fake_quantize_kv, file_sha256,
    forward_kl, load_kv_calibration, quantize_kv_pair, save_kv_calibration,
    validate_kv_evaluation_report)


def _contract(token_count=6):
    return KVCalibrationContract(
        model_artifact_sha256="a" * 64, model_id="unit", model_revision="rev",
        tokenizer_id="unit-tokenizer", tokenizer_revision="tok-rev",
        layer_count=2, num_kv_heads=3, head_dim=8, kv_dtype="float16",
        rotation_state="post_rope", attention_implementation="sdpa",
        context_lengths=(4, 8), record_count=2, token_count=token_count,
        source_sha256=("b" * 64,))


def _identity(**updates):
    values = dict(model_artifact_sha256="a" * 64, layer_count=2,
                  num_kv_heads=3, head_dim=8, kv_dtype="float16",
                  rotation_state="post_rope", attention_implementation="sdpa")
    values.update(updates)
    return KVRuntimeIdentity(**values)


def test_kv_calibration_is_linked_hashed_and_fail_closed(tmp_path):
    torch.manual_seed(1)
    keys = torch.randn(2, 3, 3, 8, dtype=torch.float16)
    collector = KVMeanCollector(2, 3, 8)
    collector.add(0, keys)
    collector.add(1, keys + 1)
    expected = torch.stack((keys.float().mean((0, 2)),
                            (keys + 1).float().mean((0, 2))))
    torch.testing.assert_close(collector.means(expected_token_count=6), expected)
    path = tmp_path / "kv_mean.safetensors"
    link = save_kv_calibration(path, collector.means(), _contract())
    assert link["artifact_sha256"] == file_sha256(path)
    means, contract, loaded_link = load_kv_calibration(
        path, _identity(), expected_artifact_sha256=link["artifact_sha256"])
    torch.testing.assert_close(means, expected)
    assert contract.rotation_state == "post_rope" and loaded_link == link
    with pytest.raises(ValueError, match="rotation_state"):
        load_kv_calibration(path, _identity(rotation_state="pre_rope"))
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_kv_calibration(path, _identity())


@pytest.mark.parametrize("mode,max_error", [("fp16", 0.002), ("q8", 0.03), ("q4", 0.55)])
def test_kv_scalar_formats_roundtrip_and_attention(mode, max_error):
    torch.manual_seed(9)
    keys = torch.randn(2, 3, 5, 8)
    values = torch.randn_like(keys)
    mean = torch.randn(3, 8) * 0.2
    pair = quantize_kv_pair(
        keys, values, mode, key_mean=mean if mode == "q4" else None,
        center_keys=mode == "q4")
    got_keys, got_values = dequantize_kv_pair(pair)
    assert float((got_keys - keys).abs().max()) <= max_error
    assert float((got_values - values).abs().max()) <= max_error
    query = torch.randn(2, 3, 2, 8)
    got = attention_reference(query, pair)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, got_keys, got_values)
    torch.testing.assert_close(got, expected)
    dense_bytes = 2 * keys.numel() * 2
    if mode == "q4":
        assert pair.physical_bytes < dense_bytes


def test_kv_fake_quant_uses_hard_forward_and_ste_backward():
    torch.manual_seed(2)
    value = torch.randn(1, 2, 3, 8, requires_grad=True)
    mean = torch.randn(2, 8)
    got = fake_quantize_kv(value, 4, channel_mean=mean)
    expected, _ = dequantize_kv_pair(quantize_kv_pair(
        value.detach(), value.detach(), "q4", key_mean=mean, center_keys=True))
    torch.testing.assert_close(got, expected)
    got.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_kv_forward_kl_and_evidence_contract():
    torch.manual_seed(4)
    logits = torch.randn(2, 5, 11)
    assert forward_kl(logits, logits)["p99"] == pytest.approx(0, abs=1e-7)
    kl = {"mean": 0.01, "p50": 0.005, "p95": 0.03, "p99": 0.05}
    zero_kl = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    timings = {
        "4096": {"p20": 1.0, "median": 1.2, "p80": 1.4},
        "16384": {"p20": 2.0, "median": 2.2, "p80": 2.4},
    }
    mode = lambda centered, cache_bytes, values=kl: {
        "own_generation_kl": values, "off_policy_kl": values,
        "downstream_scores": {"gsm8k": 0.5},
        "context_results": {"4096": 0.5, "16384": 0.45},
        "peak_cache_bytes_by_context": {
            "4096": cache_bytes, "16384": cache_bytes * 4},
        "decode_latency_ms_by_context": timings,
        "prefill_latency_ms_by_context": timings,
        "centered_keys": centered,
    }
    report = {
        "schema": 1, "model_artifact_sha256": "a" * 64,
        "calibration_artifact_sha256": "c" * 64,
        "modes": {"fp16": mode(False, 300, zero_kl), "q8": mode(False, 200),
                  "q4": mode(True, 100)},
        "centering_ablation": {
            "q4_centered": {"own_generation_kl_mean": 0.01,
                            "off_policy_kl_mean": 0.02,
                            "peak_cache_bytes_by_context": {
                                "4096": 100, "16384": 400}},
            "q4_uncentered": {"own_generation_kl_mean": 0.02,
                              "off_policy_kl_mean": 0.03,
                              "peak_cache_bytes_by_context": {
                                  "4096": 96, "16384": 384}},
        },
        "commands": ["evaluate-kv"], "provenance": {"git": "abc"},
    }
    validate_kv_evaluation_report(
        report, model_artifact_sha256="a" * 64,
        calibration_artifact_sha256="c" * 64)
    report["modes"].pop("q8")
    with pytest.raises(ValueError, match="FP16, Q8, and Q4"):
        validate_kv_evaluation_report(
            report, model_artifact_sha256="a" * 64,
            calibration_artifact_sha256="c" * 64)
