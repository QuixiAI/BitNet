"""Serializable, hash-stable QuantSpec from quant/quant_spec.md section 5."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields
from typing import Any, Mapping

SPEC_REVISION = "1.0.0"
ARTIFACT_SCHEMA = 2
FORMAT_VERSION = 1
GGML_TYPE_REGISTRY_REVISION = 1
IQ1_CODEBOOK_SHA256 = "1edfeb295366968940d5d4397dc046110f851acb59de9407fdf0c06982adaa72"

_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")
_HASH_RE = re.compile(r"[0-9a-f]{64}")

FLOAT_PROFILES = frozenset({"fp32", "bf16", "fp16"})
TQ1_PROFILES = frozenset({
    "tq1_v11-j-r",
    "tq1_v12-j-r",
    "tq1_v11-i-r",
    "tq1_v11-p-r",
    "tq1_v12-p-r",
    "tq1_v11-j-a4-r",
    "tq1_v11-j-b",
    "tq1_v12-j-b",
})
ALL_PROFILES = FLOAT_PROFILES | TQ1_PROFILES


def _float_text(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical JSON forbids NaN and infinity")
    if value == 0.0:
        return "0"
    if value.is_integer():
        return str(int(value))
    text = repr(value).lower()
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e", 1)
    sign = ""
    if exponent.startswith(("+", "-")):
        if exponent[0] == "-":
            sign = "-"
        exponent = exponent[1:]
    exponent = exponent.lstrip("0") or "0"
    return f"{mantissa}e{sign}{exponent}"


def canonical_json(value: Any) -> str:
    """Return the exact canonical JSON spelling required by the specification."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float_text(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return "{" + ",".join(
            f"{canonical_json(key)}:{canonical_json(value[key])}"
            for key in sorted(value)
        ) + "}"
    raise TypeError(f"unsupported canonical JSON value {type(value).__name__}")


def _reject_unknown(cls, raw: Mapping[str, Any]) -> None:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw) - allowed
    missing = {
        field.name for field in fields(cls)
        if field.default is field.default_factory  # never true; handled by ctor
    }
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    del missing


def _encoding_for_profile(profile: str) -> str | None:
    if profile in FLOAT_PROFILES:
        return None
    if "-j-" in profile or "-j-a4-" in profile:
        return "sign_canonical"
    if "-i-" in profile:
        return "direct_joint"
    if "-p-" in profile:
        return "product"
    raise ValueError(f"unknown profile {profile!r}")


def _format_for_profile(profile: str) -> str | None:
    if profile in FLOAT_PROFILES:
        return None
    return "v11" if profile.startswith("tq1_v11-") else "v12"


@dataclass(frozen=True)
class CodebookRef:
    id: str
    format: str
    encoding: str
    scope: str
    sha256: str

    def __post_init__(self) -> None:
        if _ID_RE.fullmatch(self.id) is None:
            raise ValueError(f"invalid codebook id {self.id!r}")
        if self.format not in {"v11", "v12"}:
            raise ValueError(f"invalid codebook format {self.format!r}")
        if self.encoding not in {"sign_canonical", "direct_joint", "product"}:
            raise ValueError(f"invalid codebook encoding {self.encoding!r}")
        if self.scope not in {"universal", "model", "iq1"}:
            raise ValueError(f"invalid codebook scope {self.scope!r}")
        if _HASH_RE.fullmatch(self.sha256) is None:
            raise ValueError("codebook sha256 must be 64 lowercase hex characters")
        if self.encoding == "direct_joint" and self.format != "v11":
            raise ValueError("the format-v1 IQ1 direct grid is V11 only")
        if self.scope == "iq1" and (
                self.encoding != "direct_joint" or self.format != "v11"
                or self.sha256 != IQ1_CODEBOOK_SHA256):
            raise ValueError("iq1 scope is reserved for the pinned direct-joint grid")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CodebookRef":
        _reject_unknown(cls, raw)
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class TensorRule:
    match: str
    profile: str
    codebook_id: str | None

    def __post_init__(self) -> None:
        try:
            re.compile(self.match)
        except re.error as exc:
            raise ValueError(f"invalid tensor-rule regex {self.match!r}: {exc}") from exc
        if self.profile not in ALL_PROFILES:
            raise ValueError(f"invalid tensor profile {self.profile!r}")
        if self.profile in FLOAT_PROFILES:
            if self.codebook_id is not None:
                raise ValueError("floating-point tensor rules must use codebook_id=null")
        elif self.codebook_id is None or _ID_RE.fullmatch(self.codebook_id) is None:
            raise ValueError("TQ1 tensor rules require a valid codebook id")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TensorRule":
        _reject_unknown(cls, raw)
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class QuantSpec:
    spec_revision: str
    artifact_schema: int
    format_version: int
    ggml_type_registry_revision: int
    default_profile: str
    codebooks: tuple[CodebookRef, ...]
    default_codebook_id: str
    default_scale_mode: str
    default_scale_dtype: str
    activation_mode: str
    importance_mode: str
    weight_metric: str
    candidate_count: int
    assignment_mode: str
    alternating_iterations: int
    gptq_feedback: bool
    gptq_block_size: int
    gptq_damping: float
    qat_projection: str
    default_affine_mode: str
    target_regexes: tuple[str, ...]
    keep_fp_regexes: tuple[str, ...]
    tensor_overrides: tuple[TensorRule, ...]
    shared_embedding_head: bool = False
    shared_head_importance: float = 0.75
    shared_embedding_importance: float = 0.25

    def __post_init__(self) -> None:
        if self.spec_revision != SPEC_REVISION:
            raise ValueError(f"unsupported spec revision {self.spec_revision!r}")
        if self.artifact_schema != ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported artifact schema {self.artifact_schema}")
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"unsupported format version {self.format_version}")
        if self.ggml_type_registry_revision != GGML_TYPE_REGISTRY_REVISION:
            raise ValueError("unsupported GGML type-registry revision")
        if self.default_profile not in TQ1_PROFILES:
            raise ValueError("default_profile must be a complete TQ1 profile")
        if not self.codebooks:
            raise ValueError("at least one codebook is required")
        by_id = {book.id: book for book in self.codebooks}
        if len(by_id) != len(self.codebooks):
            raise ValueError("codebook ids must be unique")
        if self.default_codebook_id not in by_id:
            raise ValueError("default_codebook_id is absent from the registry")
        self._validate_profile_codebook(self.default_profile, by_id[self.default_codebook_id])
        if self.default_scale_mode not in {"row", "block256"}:
            raise ValueError("default_scale_mode must be row or block256")
        expected_mode = "block256" if self.default_profile.endswith("-b") else "row"
        if self.default_scale_mode != expected_mode:
            raise ValueError("default scale mode disagrees with default profile")
        if self.default_scale_dtype not in {"float16", "bfloat16"}:
            raise ValueError("invalid runtime scale dtype")
        if self.default_scale_mode == "block256" and self.default_scale_dtype != "float16":
            raise ValueError("format-v1 embedded block scales are float16")
        if self.activation_mode not in {"a8_token", "a8_block256", "none"}:
            raise ValueError("invalid activation mode")
        if self.importance_mode not in {"uniform", "diagonal", "covariance8", "block256"}:
            raise ValueError("invalid importance mode")
        if self.weight_metric not in {"uniform", "iq1"}:
            raise ValueError("invalid weight metric")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if self.assignment_mode not in {"exhaustive", "shortlist"}:
            raise ValueError("invalid assignment mode")
        if self.alternating_iterations < 2 or self.alternating_iterations > 4:
            raise ValueError("alternating_iterations must be in [2, 4]")
        if self.gptq_block_size != 256:
            raise ValueError("format-v1 GPTQ block size is 256")
        if not math.isfinite(self.gptq_damping) or self.gptq_damping < 0:
            raise ValueError("gptq_damping must be finite and nonnegative")
        if self.gptq_feedback and self.importance_mode != "block256":
            raise ValueError("GPTQ feedback requires block256 statistics")
        if self.gptq_feedback and "-a4-" in self.default_profile:
            raise ValueError("GPTQ feedback does not support A4")
        if self.qat_projection not in {"none", "soft", "hard", "frozen"}:
            raise ValueError("invalid QAT projection")
        if self.default_affine_mode not in {"none", "rho_mu_a4"}:
            raise ValueError("invalid affine mode")
        if "-a4-" in self.default_profile and self.default_affine_mode != "rho_mu_a4":
            raise ValueError("A4 profile requires rho_mu_a4")
        if "-a4-" not in self.default_profile and self.default_affine_mode != "none":
            raise ValueError("rho_mu_a4 is only legal for A4 profiles")
        if not self.target_regexes or not self.keep_fp_regexes:
            raise ValueError("target and keep-fp regex inventories must both be explicit")
        for pattern in self.target_regexes + self.keep_fp_regexes:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid inventory regex {pattern!r}: {exc}") from exc
        for rule in self.tensor_overrides:
            if rule.profile in TQ1_PROFILES:
                book = by_id.get(rule.codebook_id or "")
                if book is None:
                    raise ValueError(f"tensor rule references unknown codebook {rule.codebook_id!r}")
                self._validate_profile_codebook(rule.profile, book)
        resolved_tq1_profiles = {
            self.default_profile,
            *(rule.profile for rule in self.tensor_overrides
              if rule.profile in TQ1_PROFILES),
        }
        if self.gptq_feedback and any("-a4-" in profile
                                      for profile in resolved_tq1_profiles):
            raise ValueError("GPTQ feedback cannot be combined with an A4 tensor override")
        if self.qat_projection != "none" and any(
                profile.endswith("-b") or "-a4-" in profile
                for profile in resolved_tq1_profiles):
            raise ValueError(
                "format-v1 QAT supports only J/I/P external-row-scale profiles")
        if not isinstance(self.shared_embedding_head, bool):
            raise ValueError("shared_embedding_head must be boolean")
        for name, value in (("shared_head_importance", self.shared_head_importance),
                            ("shared_embedding_importance",
                             self.shared_embedding_importance)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.shared_embedding_head and self.shared_head_importance \
                + self.shared_embedding_importance <= 0:
            raise ValueError("shared embedding/head importance weights cannot both be zero")

    @staticmethod
    def _validate_profile_codebook(profile: str, book: CodebookRef) -> None:
        expected_encoding = _encoding_for_profile(profile)
        expected_format = _format_for_profile(profile)
        if book.encoding != expected_encoding or book.format != expected_format:
            raise ValueError(
                f"profile {profile!r} is incompatible with codebook {book.id!r} "
                f"({book.format}/{book.encoding})"
            )
        if expected_encoding == "direct_joint" \
                and (book.scope != "iq1" or book.sha256 != IQ1_CODEBOOK_SHA256):
            raise ValueError("format-v1 direct-joint profiles require the pinned IQ1 grid")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "QuantSpec":
        _reject_unknown(cls, raw)
        data = dict(raw)
        data["codebooks"] = tuple(
            item if isinstance(item, CodebookRef) else CodebookRef.from_dict(item)
            for item in data.get("codebooks", ())
        )
        data["tensor_overrides"] = tuple(
            item if isinstance(item, TensorRule) else TensorRule.from_dict(item)
            for item in data.get("tensor_overrides", ())
        )
        for name in ("target_regexes", "keep_fp_regexes"):
            data[name] = tuple(data.get(name, ()))
        return cls(**data)

    @classmethod
    def core(
        cls,
        *,
        default_profile: str,
        codebook: CodebookRef,
        target_regexes: tuple[str, ...],
        keep_fp_regexes: tuple[str, ...],
        activation_mode: str = "a8_token",
        importance_mode: str = "diagonal",
    ) -> "QuantSpec":
        """Construct the normative core defaults without hiding them in callers."""
        return cls(
            spec_revision=SPEC_REVISION,
            artifact_schema=ARTIFACT_SCHEMA,
            format_version=FORMAT_VERSION,
            ggml_type_registry_revision=GGML_TYPE_REGISTRY_REVISION,
            default_profile=default_profile,
            codebooks=(codebook,),
            default_codebook_id=codebook.id,
            default_scale_mode="block256" if default_profile.endswith("-b") else "row",
            default_scale_dtype="float16",
            activation_mode=activation_mode,
            importance_mode=importance_mode,
            weight_metric="iq1",
            candidate_count=32,
            assignment_mode="shortlist",
            alternating_iterations=3,
            gptq_feedback=False,
            gptq_block_size=256,
            gptq_damping=0.01,
            qat_projection="none",
            default_affine_mode="rho_mu_a4" if "-a4-" in default_profile else "none",
            target_regexes=target_regexes,
            keep_fp_regexes=keep_fp_regexes,
            tensor_overrides=(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            field.name: (
                [item.to_dict() for item in value]
                if field.name in {"codebooks", "tensor_overrides"}
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for field in fields(self)
            for value in (getattr(self, field.name),)
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def codebook(self, codebook_id: str) -> CodebookRef:
        for book in self.codebooks:
            if book.id == codebook_id:
                return book
        raise KeyError(codebook_id)

    def resolve_profile(self, module_path: str) -> tuple[str, str | None]:
        matches = [rule for rule in self.tensor_overrides if re.fullmatch(rule.match, module_path)]
        if len(matches) > 1:
            raise ValueError(f"{module_path}: multiple tensor overrides matched")
        if matches:
            return matches[0].profile, matches[0].codebook_id
        return self.default_profile, self.default_codebook_id
