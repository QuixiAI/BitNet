"""Canonical TQ1_V quantization contracts.

The package is deliberately independent of the CLI in :mod:`quant.quant` so
training, artifact export, GGUF conversion, and runtime tests all import the
same numerics and schema implementation.
"""

from .spec import (
    ARTIFACT_SCHEMA,
    FORMAT_VERSION,
    GGML_TYPE_REGISTRY_REVISION,
    SPEC_REVISION,
    CodebookRef,
    QuantSpec,
    TensorRule,
    canonical_json,
)

__all__ = [
    "ARTIFACT_SCHEMA",
    "FORMAT_VERSION",
    "GGML_TYPE_REGISTRY_REVISION",
    "SPEC_REVISION",
    "CodebookRef",
    "QuantSpec",
    "TensorRule",
    "canonical_json",
]
