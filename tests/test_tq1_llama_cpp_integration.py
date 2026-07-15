from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "quant" / "llama_cpp"


def test_revision_locked_llama_patch_manifest_is_self_consistent():
    manifest = json.loads((INTEGRATION / "integration.json").read_text())
    patch = INTEGRATION / manifest["patch"]
    assert manifest["source"]["revision"] == (
        "a5822222909b785f23ddc74ce3c8f85bd0e38562")
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest["patch_sha256"]
    assert manifest["type_ids"] == {
        "GGML_TYPE_TQ1_V11": 43,
        "GGML_TYPE_TQ1_V11_J_A4_R": 47,
        "GGML_TYPE_TQ1_V11_R": 45,
        "GGML_TYPE_TQ1_V12": 44,
        "GGML_TYPE_TQ1_V12_R": 46,
    }

    source = patch.read_text()
    for path in (
        "ggml/include/ggml.h",
        "ggml/src/ggml-common.h",
        "ggml/src/ggml-cpu/ggml-cpu.c",
        "gguf-py/gguf/constants.py",
        "src/llama-graph.cpp",
        "src/llama-model-loader.cpp",
        "src/llama-model.cpp",
    ):
        assert f"diff --git a/{path} b/{path}" in source
    assert "llama_tq1_sha256(quant_spec_json) != quant_spec_sha256" in source
    assert "ggml_cpu_validate_tq1_tensor" in source


def test_llama_patch_helper_requires_an_explicit_target():
    result = subprocess.run(
        [sys.executable, str(INTEGRATION / "apply_and_test.py")],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False)
    assert result.returncode != 0
    assert "--target" in result.stdout
