"""Canonical command-line producer behind ``quant/quant.py``."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from .artifact import ArtifactReader
from .calibration import file_sha256
from .codebook import CodebookRegistry, load_iq1_reference, sign_canonical_codebook
from .codebook_study import load_pattern_corpus
from .evaluation import canonical_document_sha256
from .pipeline import (
    LLAMA_KEEP_FP_REGEXES, LLAMA_SHARED_EMBEDDING_REGEX, LLAMA_TARGET_REGEXES,
    bake_debug_checkpoint,
    classify_model_linears, collect_statistics, learn_model_codebook,
    load_statistics, run_full_model_ptq, save_model_source_files)
from .solver import build_joint_codebook, build_product_codebook, canonical_shapes
from .spec import QuantSpec

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cached_revision(model_id: str, revision: str) -> str | None:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + model_id.replace("/", "--"))
    ref = cache / "refs" / revision
    if ref.is_file():
        return ref.read_text().strip()
    if _COMMIT_RE.fullmatch(revision) and (cache / "snapshots" / revision).is_dir():
        return revision
    snapshots = cache / "snapshots"
    values = sorted(path.name for path in snapshots.iterdir()) if snapshots.is_dir() else []
    return values[0] if len(values) == 1 else None


def resolve_revision(model_id: str, revision: str, *, local_files_only: bool) -> str:
    path = Path(model_id).expanduser()
    if path.exists():
        digest = __import__("hashlib").sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file())
        for item in files:
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            digest.update(item.read_bytes())
        return digest.hexdigest()
    if _COMMIT_RE.fullmatch(revision):
        return revision
    cached = _cached_revision(model_id, revision)
    if cached:
        return cached
    if local_files_only:
        raise ValueError(
            f"cannot resolve {model_id}@{revision} to an immutable cached revision")
    from huggingface_hub import model_info
    resolved = str(model_info(model_id, revision=revision).sha)
    if not _COMMIT_RE.fullmatch(resolved):
        raise ValueError("Hugging Face did not return an immutable source revision")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TQ1_V schema-2 calibration, PTQ, and canonical artifact producer")
    parser.add_argument("--config", default=None,
                        help="YAML/JSON defaults; unknown keys are fatal")
    parser.add_argument("--spec", default=None, help="complete QuantSpec JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--load-dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--profile", default="tq1_v12-j-r", choices=[
        "tq1_v11-j-r", "tq1_v12-j-r", "tq1_v11-i-r",
        "tq1_v11-p-r", "tq1_v12-p-r", "tq1_v11-j-a4-r",
        "tq1_v11-j-b", "tq1_v12-j-b"])
    parser.add_argument("--activation-mode", default="a8_token",
                        choices=["a8_token", "a8_block256", "none"])
    parser.add_argument("--scale-dtype", default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--importance-mode", default="diagonal",
                        choices=["uniform", "diagonal", "covariance8", "block256"])
    parser.add_argument("--weight-metric", default="iq1", choices=["uniform", "iq1"])
    parser.add_argument("--assignment-mode", default="shortlist",
                        choices=["shortlist", "exhaustive"])
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--alternating-iterations", type=int, default=3)
    parser.add_argument("--chunk-groups", type=int, default=4096)
    parser.add_argument("--gptq-feedback", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--gptq-damping", type=float, default=0.01)
    parser.add_argument("--allow-diagonal-fallback", action="store_true")
    parser.add_argument("--target-regex", action="append", default=None)
    parser.add_argument("--keep-fp-regex", action="append", default=None)
    parser.add_argument("--quantize-tied-embedding-head", action="store_true",
                        help="project model.embed_tokens once and share it with lm_head")
    parser.add_argument("--shared-head-importance", type=float, default=0.75)
    parser.add_argument("--shared-embedding-importance", type=float, default=0.25)

    parser.add_argument("--codebook-source", default="learned",
                        choices=["learned", "universal", "iq1", "artifact"])
    parser.add_argument("--codebook-id", default=None)
    parser.add_argument("--codebook-artifact", default=None,
                        help="schema-2 artifact carrying the exact codebook registry")
    parser.add_argument("--codebook-corpus", default=None,
                        help="construction-role sensitivity corpus for a universal table")
    parser.add_argument("--codebook-weighting", default="family_equal",
                        choices=["parameter", "tensor_equal", "family_equal"])
    parser.add_argument("--codebook-lambda-nz", type=float, default=0.0)
    parser.add_argument("--codebook-swap-limit", type=int, default=1)
    parser.add_argument("--iq1-reference-dir", default="~/llama.cpp")

    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-seq-len", type=int, default=1024)
    parser.add_argument("--statistics-artifact", default=None)
    parser.add_argument("--statistics-output", default=None)
    parser.add_argument("--ridge-factor", type=float, default=1e-5)
    parser.add_argument("--baked-debug-output", default=None)
    parser.add_argument("--evaluation-report", default=None,
                        help="precomputed quality report JSON; omit for unqualified artifacts")
    return parser


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    if known.config:
        raw = yaml.safe_load(Path(known.config).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("--config root must be an object")
        destinations = {action.dest for action in parser._actions}
        normalized = {key.replace("-", "_"): value for key, value in raw.items()}
        unknown = set(normalized) - destinations
        if unknown:
            raise ValueError(f"--config has unknown fields {sorted(unknown)}")
        parser.set_defaults(**normalized)
    args = parser.parse_args(argv)
    if not args.output:
        parser.error("--output is required (directly or through --config)")
    return args


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.spec:
        if not args.codebook_artifact:
            raise ValueError("--spec requires --codebook-artifact")
        if args.codebook_corpus:
            raise ValueError("--codebook-corpus cannot be combined with --spec")
        return
    if args.codebook_source == "artifact" and not args.codebook_artifact:
        raise ValueError("--codebook-source artifact requires --codebook-artifact")
    if args.codebook_source == "iq1" and args.profile != "tq1_v11-i-r":
        raise ValueError("IQ1 source requires --profile tq1_v11-i-r")
    if _encoding(args.profile) == "direct_joint" \
            and args.codebook_source not in {"iq1", "artifact"}:
        raise ValueError("direct-joint profiles require --codebook-source iq1 or artifact")
    if args.codebook_source == "universal" and not args.codebook_corpus:
        raise ValueError("--codebook-source universal requires --codebook-corpus")
    if args.codebook_corpus and args.codebook_source != "universal":
        raise ValueError("--codebook-corpus is only legal for a universal codebook")


def _encoding(profile: str) -> str:
    if "-i-" in profile:
        return "direct_joint"
    if "-p-" in profile:
        return "product"
    return "sign_canonical"


def _format(profile: str) -> str:
    return "v11" if profile.startswith("tq1_v11-") else "v12"


def _load_artifact_registry(path: str | Path) -> CodebookRegistry:
    reader = ArtifactReader(path)
    reader.validate()
    return reader.registry()


def _placeholder_book(index_format: str):
    shapes = canonical_shapes()
    count = 1024 if index_format == "v11" else 2048
    zero = shapes[(shapes == 0).all(1)]
    nonzero = shapes[~(shapes == 0).all(1)][:count - 1]
    return sign_canonical_codebook(
        "placeholder", index_format, torch.cat((zero, nonzero)), scope="model")


def _build_spec(args, book) -> QuantSpec:
    target_regexes = tuple(args.target_regex or LLAMA_TARGET_REGEXES)
    if args.quantize_tied_embedding_head \
            and LLAMA_SHARED_EMBEDDING_REGEX not in target_regexes:
        target_regexes += (LLAMA_SHARED_EMBEDDING_REGEX,)
    spec = QuantSpec.core(
        default_profile=args.profile, codebook=book.ref(),
        target_regexes=target_regexes,
        keep_fp_regexes=tuple(args.keep_fp_regex or LLAMA_KEEP_FP_REGEXES),
        activation_mode=args.activation_mode, importance_mode=args.importance_mode)
    return replace(
        spec, default_scale_dtype=args.scale_dtype,
        weight_metric=args.weight_metric, candidate_count=args.candidate_count,
        assignment_mode=args.assignment_mode,
        alternating_iterations=args.alternating_iterations,
        gptq_feedback=args.gptq_feedback, gptq_damping=args.gptq_damping,
        shared_embedding_head=args.quantize_tied_embedding_head,
        shared_head_importance=args.shared_head_importance,
        shared_embedding_importance=args.shared_embedding_importance)


def _load_model(args, revision: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": args.local_files_only,
        "trust_remote_code": False,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=_dtype(args.load_dtype), **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **kwargs)
    return model, tokenizer


def _prepare_statistics(args, model, tokenizer, inventory, spec: QuantSpec,
                        output: Path, revision: str, *,
                        statistics_path: str | None = None):
    path = statistics_path or args.statistics_artifact
    if path is None and spec.importance_mode != "uniform":
        if not args.calibration_file:
            raise ValueError(
                f"{spec.importance_mode} importance requires --statistics-artifact "
                "or --calibration-file")
        path = args.statistics_output or str(output) + ".calibration.safetensors"
        modes = ["diagonal"]
        if spec.importance_mode == "covariance8":
            modes.append("covariance8")
        elif spec.importance_mode == "block256":
            modes.append("block256")
        device = _device(args.device)
        model.to(device)
        collect_statistics(
            model, tokenizer, inventory, calibration_file=args.calibration_file,
            output=path, modes=modes,
            sample_count=args.calibration_samples,
            sequence_cap=args.calibration_seq_len, device=device,
            metadata={
                "model": args.model, "model_revision": revision,
                "tokenizer": args.model, "tokenizer_revision": revision,
            }, ridge_factor=args.ridge_factor)
        model.to("cpu")
        if device.type == "mps":
            torch.mps.empty_cache()
    statistics, metadata, digest = load_statistics(path, spec)
    expected = set(inventory.statistics_targets())
    recorded = set(metadata.get("target_modules", expected))
    if path and recorded != expected:
        raise ValueError("calibration artifact target inventory differs from the model")
    return path, statistics, metadata, digest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_arguments(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    revision = resolve_revision(
        args.model, args.revision, local_files_only=args.local_files_only)
    print(f"[source] {args.model}@{revision}")
    model, tokenizer = _load_model(args, revision)
    prepared_statistics = None

    if args.spec:
        spec = QuantSpec.from_dict(json.loads(Path(args.spec).read_text()))
        registry = _load_artifact_registry(args.codebook_artifact)
        registry.validate_refs(spec.codebooks)
    else:
        # Inventory is needed before learning a model-scoped table.  Use a
        # temporary structurally valid spec and replace it once the final table
        # hash is known.
        if args.codebook_source == "artifact":
            registry = _load_artifact_registry(args.codebook_artifact)
            codebook_id = args.codebook_id or next(iter(book.id for book in registry.refs()))
            book = registry[codebook_id]
            if book.encoding != _encoding(args.profile) or book.index_format != _format(args.profile):
                raise ValueError("loaded codebook is incompatible with --profile")
            registry = CodebookRegistry({book.id: book})
        elif args.codebook_source == "iq1":
            book = load_iq1_reference(
                args.codebook_id or "iq1s_grid",
                reference_dir=args.iq1_reference_dir)
            registry = CodebookRegistry({book.id: book})
        else:
            # A uniform placeholder exists only long enough to classify linears.
            placeholder = _placeholder_book(_format(args.profile))
            temporary = QuantSpec.core(
                default_profile=f"tq1_{_format(args.profile)}-j-r",
                codebook=placeholder.ref(),
                target_regexes=(tuple(args.target_regex or LLAMA_TARGET_REGEXES)
                                + ((LLAMA_SHARED_EMBEDDING_REGEX,)
                                   if args.quantize_tied_embedding_head
                                   and LLAMA_SHARED_EMBEDDING_REGEX not in tuple(
                                       args.target_regex or LLAMA_TARGET_REGEXES)
                                   else ())),
                keep_fp_regexes=tuple(args.keep_fp_regex or LLAMA_KEEP_FP_REGEXES),
                activation_mode=args.activation_mode,
                importance_mode=args.importance_mode)
            temporary = replace(
                temporary,
                shared_embedding_head=args.quantize_tied_embedding_head,
                shared_head_importance=args.shared_head_importance,
                shared_embedding_importance=args.shared_embedding_importance)
            inventory = classify_model_linears(model, temporary)
            if args.codebook_source == "universal":
                construction_corpus, corpus_metadata = load_pattern_corpus(
                    args.codebook_corpus)
                if corpus_metadata.get("role") != "construction":
                    raise ValueError("universal codebook corpus must have role=construction")
                if args.codebook_weighting != "family_equal" \
                        or corpus_metadata.get("normalize_each") is not True \
                        or corpus_metadata.get("source_weightings") != ["parameter"]:
                    raise ValueError(
                        "format-v1 universal construction requires parameter weighting "
                        "within each family and a family-equal (--normalize-each) merge")
                source_models = corpus_metadata.get("source_model_identities")
                split_manifest_sha256 = corpus_metadata.get("split_manifest_sha256")
                codebook_importance_mode = corpus_metadata.get("sensitivity_metric")
                if codebook_importance_mode not in {"diagonal", "covariance8"}:
                    raise ValueError(
                        "universal construction corpus lacks a sensitivity metric")
                if not isinstance(source_models, list) or not source_models:
                    raise ValueError("universal construction corpus lacks source model identities")
                if any(not isinstance(item, dict)
                       or set(item) != {"model_id", "revision", "family"}
                       or not isinstance(item["model_id"], str) or not item["model_id"]
                       or not isinstance(item["family"], str) or not item["family"]
                       or not isinstance(item["revision"], str)
                       or _COMMIT_RE.fullmatch(item["revision"]) is None
                       for item in source_models):
                    raise ValueError("universal construction source identities are invalid")
                source_keys = [(item["model_id"], item["revision"])
                               for item in source_models]
                if len(source_keys) != len(set(source_keys)):
                    raise ValueError("universal construction source identities are duplicated")
                if not isinstance(split_manifest_sha256, str) \
                        or _COMMIT_RE.fullmatch(split_manifest_sha256) is None \
                        or len(split_manifest_sha256) != 64:
                    raise ValueError("universal construction corpus lacks a split SHA-256")
                if _encoding(args.profile) == "product":
                    book = build_product_codebook(
                        args.codebook_id or f"universal_{_format(args.profile)}p",
                        _format(args.profile), construction_corpus,
                        scope="universal", swap_limit=args.codebook_swap_limit)
                else:
                    book = build_joint_codebook(
                        args.codebook_id or f"universal_{_format(args.profile)}j",
                        _format(args.profile), construction_corpus,
                        scope="universal", lambda_nz=args.codebook_lambda_nz,
                        swap_limit=args.codebook_swap_limit)
                solver_config = {
                    "encoding": _encoding(args.profile),
                    "index_format": _format(args.profile),
                    "lambda_nz": args.codebook_lambda_nz,
                    "swap_limit": args.codebook_swap_limit,
                    "importance_mode": codebook_importance_mode,
                }
                book = replace(book, provenance={
                    **book.provenance,
                    "source": "construction_pattern_corpus",
                    "pattern_corpus_sha256": file_sha256(args.codebook_corpus),
                    "pattern_corpus_metadata_sha256": canonical_document_sha256(
                        corpus_metadata),
                    "solver_config_sha256": canonical_document_sha256(solver_config),
                    "solver_config": solver_config,
                    "source_models": source_models,
                    "split_manifest_sha256": split_manifest_sha256,
                    "weighting": "family_equal",
                    "importance_mode": codebook_importance_mode,
                })
            else:
                prepared_statistics = _prepare_statistics(
                    args, model, tokenizer, inventory, temporary, output, revision)
                book = learn_model_codebook(
                    model, inventory,
                    codebook_id=args.codebook_id or f"llama32_{_format(args.profile)}"
                    + ("p" if _encoding(args.profile) == "product" else "j"),
                    index_format=_format(args.profile), encoding=_encoding(args.profile),
                    weighting=args.codebook_weighting,
                    lambda_nz=args.codebook_lambda_nz,
                    swap_limit=args.codebook_swap_limit,
                    statistics=prepared_statistics[1],
                    importance_mode=args.importance_mode)
                solver_config = {
                    "encoding": _encoding(args.profile),
                    "index_format": _format(args.profile),
                    "weighting": args.codebook_weighting,
                    "importance_mode": (
                        "diagonal" if args.importance_mode == "block256"
                        else args.importance_mode),
                    "lambda_nz": args.codebook_lambda_nz,
                    "swap_limit": args.codebook_swap_limit,
                }
                book = replace(book, provenance={
                    **book.provenance,
                    "source": "model",
                    "source_model": args.model,
                    "source_revision": revision,
                    "calibration_statistics_sha256": prepared_statistics[3],
                    "calibration_metadata_sha256": (
                        canonical_document_sha256(prepared_statistics[2])
                        if prepared_statistics[2] else None),
                    "solver_config_sha256": canonical_document_sha256(solver_config),
                    "solver_config": solver_config,
                })
            registry = CodebookRegistry({book.id: book})
        spec = _build_spec(args, book)

    inventory = classify_model_linears(model, spec)
    print(f"[inventory] {len(inventory.target)} TQ1 linears, "
          f"{len(inventory.shared_tied)} shared tensors, {len(inventory.keep_fp)} FP linears")
    if prepared_statistics is None:
        prepared_statistics = _prepare_statistics(
            args, model, tokenizer, inventory, spec, output, revision)
    statistics_path, statistics, statistics_meta, statistics_hash = prepared_statistics
    if statistics_path and set(statistics_meta.get(
            "target_modules", inventory.statistics_targets())) \
            != set(inventory.statistics_targets()):
        raise ValueError("prepared calibration inventory differs from final QuantSpec")
    evaluation = (json.loads(Path(args.evaluation_report).read_text())
                  if args.evaluation_report else None)
    with tempfile.TemporaryDirectory(prefix="tq1-source-") as temporary:
        source_files = save_model_source_files(model, tokenizer, temporary)
        artifact = run_full_model_ptq(
            model, spec, registry, output_dir=output,
            source_model=args.model, source_revision=revision,
            tokenizer=tokenizer, source_files=source_files,
            statistics=statistics, calibration_hash=statistics_hash,
            chunk_groups=args.chunk_groups,
            allow_diagonal_fallback=args.allow_diagonal_fallback,
            overwrite=args.overwrite,
            command=("quant/quant.py", *(argv if argv is not None else sys.argv[1:])),
            evaluation_report=evaluation)
    if args.baked_debug_output:
        bake_debug_checkpoint(artifact, args.baked_debug_output,
                              overwrite=args.overwrite)
    print(json.dumps({
        "artifact": str(artifact), "artifact_schema": 2,
        "quant_spec_sha256": spec.sha256(),
        "statistics_artifact": statistics_path,
        "quality_qualified": evaluation is not None,
    }, indent=2, sort_keys=True))
    return 0
