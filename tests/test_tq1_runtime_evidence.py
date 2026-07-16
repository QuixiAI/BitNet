import pytest

from bitnet_train.tq1.runtime import validate_runtime_performance_report


def _timing(value):
    return {"p20_ms": value * 0.9, "median_ms": value, "p80_ms": value * 1.1}


def _report():
    return {
        "schema": 1, "model_artifact_sha256": "a" * 64,
        "backend": "native_cpu", "quant_format": "tq1_v12-j-r+a8_token",
        "routing_policy": {"short": 32, "long": 256},
        "device": {"model": "unit"}, "toolchain": {"compiler": "unit"},
        "warmups": 5, "iterations": 20,
        "tg128": {"prompt_tokens": 32, "generated_tokens": 128,
                  "timing": _timing(1000), "tokens_per_second": 128},
        "pp512": {"prompt_tokens": 512, "timing": _timing(100),
                  "tokens_per_second": 5120},
        "ttft": {"128": _timing(20), "512": _timing(80), "2048": _timing(300)},
        "output_head": {"median_ms": 2.0, "total_decode_median_ms": 10.0,
                        "decode_time_share": 0.2},
        "memory": {"canonical_artifact_bytes": 100, "canonical_resident_bytes": 100,
                   "backend_private_repack_bytes": 200, "resident_model_bytes": 300,
                   "peak_bytes_by_context": {"512": 350, "4096": 500}},
        "sustained": {"duration_seconds": 60, "generated_tokens": 600,
                      "window_tokens_per_second": [10, 9.9, 9.8],
                      "thermal_state": "nominal"},
        "energy": {"measurement_method": "external meter",
                   "average_power_watts": 10.0, "duration_seconds": 60.0,
                   "energy_joules": 600.0, "generated_tokens": 600,
                   "joules_per_token": 1.0},
        "commands": ["benchmark-model"], "provenance": {"git": "abc"},
    }


def test_runtime_performance_contract_requires_energy_per_token_and_named_workloads():
    report = _report()
    validate_runtime_performance_report(report, model_artifact_sha256="a" * 64)
    report["energy"]["joules_per_token"] = 2.0
    with pytest.raises(ValueError, match="energy-per-token"):
        validate_runtime_performance_report(report, model_artifact_sha256="a" * 64)


def test_runtime_performance_contract_reconciles_head_and_resident_memory():
    report = _report()
    report["output_head"]["median_ms"] = 11.0
    report["output_head"]["decode_time_share"] = 1.1
    with pytest.raises(ValueError, match="output-head share"):
        validate_runtime_performance_report(report, model_artifact_sha256="a" * 64)
    report = _report()
    report["memory"]["resident_model_bytes"] = 299
    with pytest.raises(ValueError, match="memory accounting"):
        validate_runtime_performance_report(report, model_artifact_sha256="a" * 64)
