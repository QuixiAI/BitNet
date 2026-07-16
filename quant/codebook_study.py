#!/usr/bin/env python3
"""Measure and validate the QuantSpec universal-codebook acceptance study."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.artifact import ArtifactReader  # noqa: E402
from bitnet_train.tq1.calibration import (  # noqa: E402
    file_sha256, load_calibration_artifact)
from bitnet_train.tq1.cli import _dtype, resolve_revision  # noqa: E402
from bitnet_train.tq1.codebook import sign_canonical_codebook  # noqa: E402
from bitnet_train.tq1.codebook_study import (  # noqa: E402
    PROJECTION_FAMILIES, codebook_coverage, combine_distortion_metrics,
    finalize_universal_codebook_study, load_pattern_corpus,
    measurement_document, save_pattern_corpus, study_definition_sha256,
    validate_universal_codebook_study)
from bitnet_train.tq1.evaluation import canonical_document_sha256  # noqa: E402
from bitnet_train.tq1.pipeline import (  # noqa: E402
    LLAMA_KEEP_FP_REGEXES, LLAMA_TARGET_REGEXES, classify_model_linears,
    scalar_pattern_family_corpora)
from bitnet_train.tq1.solver import (  # noqa: E402
    canonical_shapes, combine_pattern_corpora)
from bitnet_train.tq1.spec import QuantSpec  # noqa: E402


def _reader(path: str | Path) -> ArtifactReader:
    reader = ArtifactReader(path)
    reader.validate()
    return reader


def _write(path: str | Path, value) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return destination


def _sha256(value, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


def _extract_corpora(args) -> Path:
    from transformers import AutoModelForCausalLM

    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    revision = resolve_revision(
        args.model, args.revision, local_files_only=args.local_files_only)
    split_manifest_sha256 = _sha256(
        args.split_manifest_sha256, "--split-manifest-sha256")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, local_files_only=args.local_files_only,
        trust_remote_code=False, dtype=_dtype(args.load_dtype))
    shapes = canonical_shapes()
    zero = shapes[(shapes == 0).all(1)]
    nonzero = shapes[~(shapes == 0).all(1)][:1023]
    placeholder = sign_canonical_codebook(
        "study_placeholder", "v11", torch.cat((zero, nonzero)))
    spec = QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=placeholder.ref(),
        target_regexes=LLAMA_TARGET_REGEXES,
        keep_fp_regexes=LLAMA_KEEP_FP_REGEXES,
        activation_mode="none", importance_mode=args.importance_mode)
    inventory = classify_model_linears(model, spec)
    statistics, statistics_metadata = load_calibration_artifact(args.statistics)
    for field, expected in (("model", args.model), ("model_revision", revision)):
        if field in statistics_metadata and statistics_metadata[field] != expected:
            raise ValueError(
                f"calibration {field} {statistics_metadata[field]!r} != {expected!r}")
    recorded_targets = statistics_metadata.get("target_modules")
    if recorded_targets is not None and set(recorded_targets) != set(inventory.target):
        raise ValueError("calibration target inventory differs from held-out model targets")
    corpora = scalar_pattern_family_corpora(
        model, inventory, statistics, importance_mode=args.importance_mode,
        weighting=args.weighting)
    del model

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.", dir=destination.parent))
    try:
        statistics_sha256 = file_sha256(args.statistics)
        files = {}
        for family in PROJECTION_FAMILIES:
            source_tensors = [name for name in inventory.target
                              if name.endswith(family) or f".{family}." in name]
            path = temporary / f"{family}.safetensors"
            save_pattern_corpus(path, corpora[family], metadata={
                "schema": 1,
                "role": args.role,
                "model_identity": {
                    "model_id": args.model, "revision": revision,
                    "family": args.model_family,
                },
                "projection_family": family,
                "source_tensors": source_tensors,
                "statistics_sha256": statistics_sha256,
                "statistics_metadata_sha256": canonical_document_sha256(
                    statistics_metadata),
                "split_manifest_sha256": split_manifest_sha256,
                "sensitivity_metric": args.importance_mode,
                "weighting": args.weighting,
            })
            files[path.name] = file_sha256(path)
        manifest = {
            "schema": 1,
            "role": args.role,
            "model_identity": {
                "model_id": args.model, "revision": revision,
                "family": args.model_family,
            },
            "statistics_sha256": statistics_sha256,
            "statistics_metadata_sha256": canonical_document_sha256(
                statistics_metadata),
            "split_manifest_sha256": split_manifest_sha256,
            "sensitivity_metric": args.importance_mode,
            "weighting": args.weighting,
            "target_modules": list(inventory.target),
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _measure_model_directory(directory: str | Path, codebook) -> dict:
    root = Path(directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {
        "schema", "role", "model_identity", "statistics_sha256",
        "statistics_metadata_sha256", "split_manifest_sha256",
        "sensitivity_metric", "weighting", "target_modules", "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != required \
            or manifest["schema"] != 1 or manifest["role"] != "heldout":
        raise ValueError("held-out corpus directory manifest has an invalid schema or role")
    identity = manifest["model_identity"]
    if not isinstance(identity, dict) or set(identity) != {
            "model_id", "revision", "family"} \
            or any(not isinstance(identity[field], str) or not identity[field]
                   for field in identity):
        raise ValueError("held-out corpus directory has an invalid model identity")
    if len(identity["revision"]) not in {40, 64} \
            or any(character not in "0123456789abcdef"
                   for character in identity["revision"]):
        raise ValueError("held-out corpus model revision is not immutable")
    for field in ("statistics_sha256", "statistics_metadata_sha256",
                  "split_manifest_sha256"):
        _sha256(manifest[field], f"manifest.{field}")
    if manifest["sensitivity_metric"] not in {"diagonal", "covariance8"} \
            or manifest["weighting"] not in {
                "parameter", "tensor_equal", "family_equal"}:
        raise ValueError("held-out corpus metric or weighting is invalid")
    if not isinstance(manifest["target_modules"], list) \
            or not manifest["target_modules"] \
            or not all(isinstance(value, str) and value
                       for value in manifest["target_modules"]):
        raise ValueError("held-out corpus target-module inventory is invalid")
    expected_files = {f"{family}.safetensors" for family in PROJECTION_FAMILIES}
    actual_files = {path.name for path in root.iterdir()
                    if path.is_file() and path.name != "manifest.json"}
    if not isinstance(manifest["files"], dict) \
            or set(manifest["files"]) != expected_files \
            or actual_files != expected_files:
        raise ValueError("held-out corpus directory family-file inventory is incomplete")
    families = {}
    for family in PROJECTION_FAMILIES:
        path = root / f"{family}.safetensors"
        digest = file_sha256(path)
        if manifest["files"][path.name] != digest:
            raise ValueError(f"held-out corpus transport hash mismatch for {family}")
        corpus, metadata = load_pattern_corpus(path)
        if metadata.get("role") != "heldout" \
                or metadata.get("model_identity") != manifest["model_identity"] \
                or metadata.get("projection_family") != family \
                or metadata.get("split_manifest_sha256") \
                != manifest["split_manifest_sha256"]:
            raise ValueError(f"held-out corpus metadata mismatch for {family}")
        families[family] = {
            "pattern_corpus_sha256": digest,
            "metrics": measurement_document(
                codebook, corpus, corpus_path=path,
                corpus_metadata=metadata)["metrics"],
        }
    return {
        "identity": manifest["model_identity"],
        "pattern_corpus_sha256": file_sha256(manifest_path),
        "families": families,
        "metrics": combine_distortion_metrics(
            evidence["metrics"] for evidence in families.values()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("coverage", "measure", "measure-model", "finalize", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--artifact", required=True)
        command.add_argument("--codebook-id", required=True)
        if name in {"finalize", "validate"}:
            command.add_argument("--report", required=True)
        if name in {"coverage", "measure", "measure-model", "finalize"}:
            command.add_argument("--output", required=True)
        if name == "measure":
            command.add_argument("--corpus", required=True)
        if name == "measure-model":
            command.add_argument("--corpus-directory", required=True)
    definition = sub.add_parser("definition-hash")
    definition.add_argument("--report", required=True)
    extract = sub.add_parser("extract-corpora")
    extract.add_argument("--model", required=True)
    extract.add_argument("--revision", default="main")
    extract.add_argument("--statistics", required=True)
    extract.add_argument("--split-manifest-sha256", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--model-family", default="llama")
    extract.add_argument("--role", default="heldout",
                         choices=["construction", "heldout"])
    extract.add_argument("--importance-mode", default="diagonal",
                         choices=["diagonal", "covariance8"])
    extract.add_argument("--weighting", default="parameter",
                         choices=["parameter", "tensor_equal", "family_equal"])
    extract.add_argument("--load-dtype", default="float32",
                         choices=["float32", "float16", "bfloat16"])
    extract.add_argument("--local-files-only", action="store_true")
    merge = sub.add_parser("merge-corpora")
    merge.add_argument("--corpus", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--role", required=True,
                       choices=["construction", "heldout"])
    merge.add_argument("--normalize-each", action="store_true",
                       help="give every input model/family corpus total mass one")
    args = parser.parse_args(argv)

    if args.command == "merge-corpora":
        sources = [Path(path).expanduser().resolve() for path in args.corpus]
        if len(sources) < 2 or len(set(sources)) != len(sources):
            raise ValueError("merge-corpora requires at least two distinct inputs")
        loaded = [load_pattern_corpus(path) for path in sources]
        if any(item[1].get("role") != args.role for item in loaded):
            raise ValueError("merged corpus inputs do not match the declared split role")
        sensitivity_metrics = {item[1].get("sensitivity_metric") for item in loaded}
        if len(sensitivity_metrics) != 1 \
                or not sensitivity_metrics <= {"diagonal", "covariance8"}:
            raise ValueError("merged corpora do not share one sensitivity metric")
        sensitivity_metric = sensitivity_metrics.pop()
        split_hashes = {item[1].get("split_manifest_sha256") for item in loaded}
        if len(split_hashes) != 1 or None in split_hashes:
            raise ValueError("merged corpora do not share one split-manifest SHA-256")
        split_manifest_sha256 = _sha256(
            split_hashes.pop(), "source split_manifest_sha256")
        identities = []
        source_weightings = []
        for _, metadata in loaded:
            nested = metadata.get("source_model_identities")
            if nested is None:
                nested = [metadata.get("model_identity")]
            if not isinstance(nested, list) or any(not isinstance(value, dict)
                                                   for value in nested):
                raise ValueError("merged corpus source model identities are missing")
            identities.extend(nested)
            nested_weightings = metadata.get("source_weightings")
            if nested_weightings is None:
                nested_weightings = [metadata.get("weighting")]
            if not isinstance(nested_weightings, list) \
                    or any(value not in {"parameter", "tensor_equal", "family_equal"}
                           for value in nested_weightings):
                raise ValueError("merged corpus source weightings are missing or invalid")
            source_weightings.extend(nested_weightings)
        unique_identities = []
        seen_identities = set()
        for identity in identities:
            identity_hash = canonical_document_sha256(identity)
            if identity_hash not in seen_identities:
                unique_identities.append(identity)
                seen_identities.add(identity_hash)
        corpus = combine_pattern_corpora(
            [item[0] for item in loaded], normalize_each=args.normalize_each)
        destination = save_pattern_corpus(args.output, corpus, metadata={
            "schema": 1,
            "role": args.role,
            "normalize_each": args.normalize_each,
            "sensitivity_metric": sensitivity_metric,
            "split_manifest_sha256": split_manifest_sha256,
            "source_model_identities": unique_identities,
            "source_weightings": sorted(set(source_weightings)),
            "source_corpus_sha256": [file_sha256(path) for path in sources],
            "source_metadata_sha256": [
                canonical_document_sha256(item[1]) for item in loaded],
        })
        print(json.dumps({"output": str(destination),
                          "sha256": file_sha256(destination)},
                         indent=2, sort_keys=True))
        return 0
    if args.command == "extract-corpora":
        destination = _extract_corpora(args)
        print(json.dumps({"output": str(destination)}, indent=2, sort_keys=True))
        return 0
    if args.command == "definition-hash":
        report = json.loads(Path(args.report).read_text())
        print(json.dumps({"study_definition_sha256": study_definition_sha256(report)},
                         sort_keys=True))
        return 0

    reader = _reader(args.artifact)
    codebook = reader.registry()[args.codebook_id]
    if args.command == "coverage":
        result = codebook_coverage(codebook)
        destination = _write(args.output, result)
    elif args.command == "measure":
        corpus, metadata = load_pattern_corpus(args.corpus)
        result = measurement_document(
            codebook, corpus, corpus_path=args.corpus, corpus_metadata=metadata)
        destination = _write(args.output, result)
    elif args.command == "measure-model":
        result = _measure_model_directory(args.corpus_directory, codebook)
        destination = _write(args.output, result)
    elif args.command == "finalize":
        report = json.loads(Path(args.report).read_text())
        result = finalize_universal_codebook_study(report, codebook)
        destination = _write(args.output, result)
    else:
        report = json.loads(Path(args.report).read_text())
        decision = validate_universal_codebook_study(report, codebook)
        print(json.dumps({"valid": True, "decision": decision},
                         indent=2, sort_keys=True))
        return 0
    print(json.dumps({"output": str(destination), "codebook_sha256": codebook.sha256()},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
