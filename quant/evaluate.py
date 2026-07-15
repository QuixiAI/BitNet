#!/usr/bin/env python3
"""Evaluate one canonical TQ1 artifact against its dense teacher on held-out text."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.artifact import ArtifactReader  # noqa: E402
from bitnet_train.tq1.cli import resolve_revision  # noqa: E402
from bitnet_train.tq1.evaluation import (  # noqa: E402
    evaluate_records, file_sha256, read_evaluation_records)
from bitnet_train.tq1.runtime import load_packed_model  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="held-out full-vocabulary CE/KL/agreement for a TQ1 artifact")
    parser.add_argument("--release-artifact", required=True,
                        help="artifact whose quality matrix this component will enter")
    student_source = parser.add_mutually_exclusive_group(required=True)
    student_source.add_argument("--student-artifact")
    student_source.add_argument("--student-model")
    parser.add_argument("--student-revision", default=None)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-backend", default="scalar_oracle",
                        choices=["scalar_oracle", "native_cpu"])
    parser.add_argument("--activation-mode", default=None,
                        choices=["a8_token", "a8_block256", "none"])
    parser.add_argument("--sequence-cap", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--calibration-disjoint", action="store_true")
    parser.add_argument("--policy-selection-disjoint", action="store_true")
    args = parser.parse_args(argv)
    if not args.calibration_disjoint or not args.policy_selection_disjoint:
        parser.error("both held-out disjointness flags are required")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    reader = ArtifactReader(args.release_artifact)
    reader.validate()
    revision = resolve_revision(
        args.teacher, args.teacher_revision,
        local_files_only=args.local_files_only)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        reader.directory, local_files_only=True, use_fast=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, revision=revision, local_files_only=args.local_files_only,
        trust_remote_code=False, dtype=torch.float32)
    if args.student_artifact:
        student, student_reader = load_packed_model(
            args.student_artifact, activation_mode=args.activation_mode,
            runtime_backend=args.runtime_backend)
        student_identity = {
            "kind": "canonical_tq1_artifact",
            "path": str(Path(args.student_artifact).resolve()),
            "quant_spec_sha256": student_reader.manifest["quant_spec_sha256"],
        }
    else:
        student_revision = resolve_revision(
            args.student_model, args.student_revision or "main",
            local_files_only=args.local_files_only)
        student = AutoModelForCausalLM.from_pretrained(
            args.student_model, revision=student_revision,
            local_files_only=args.local_files_only, trust_remote_code=False,
            dtype=torch.float32)
        student_identity = {
            "kind": "huggingface_model", "model": args.student_model,
            "revision": student_revision,
        }
    records = read_evaluation_records(args.data)
    result = evaluate_records(
        student, teacher, tokenizer, records, device=args.device,
        sequence_cap=args.sequence_cap)
    payload = {
        "schema": 1,
        "profile": args.profile_label,
        "quant_spec_sha256": reader.manifest["quant_spec_sha256"],
        "evaluation_data": {
            "dataset": args.dataset,
            "revision": args.dataset_revision,
            "split": args.split,
            "sha256": file_sha256(args.data),
            "tokenizer_sha256": reader.manifest["tokenizer_sha256"],
            "record_count": result["record_count"],
            "token_count": result["metrics"]["token_count"],
            "calibration_disjoint": True,
            "policy_selection_disjoint": True,
        },
        "metrics": result["metrics"],
        "stratified": result["stratified"],
        "runtime": {
            "backend": (args.runtime_backend if args.student_artifact else "transformers"),
            "activation_mode": (args.activation_mode or reader.quant_spec.activation_mode
                                if args.student_artifact else "model_native"),
        },
        "student": student_identity,
        "teacher": {"model": args.teacher, "revision": revision},
        "command": ["quant/evaluate.py", *(argv if argv is not None else sys.argv[1:])],
        "provenance": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
