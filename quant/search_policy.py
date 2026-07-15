#!/usr/bin/env python
"""Whole-model greedy mixed-format policy search over a canonical artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.artifact import ArtifactReader  # noqa: E402
from bitnet_train.tq1.policy import (  # noqa: E402
    PolicyTensor, greedy_policy_search, make_move_groups, policy_to_spec)
from bitnet_train.tq1.spec import canonical_json  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_codebooks(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        profile, separator, codebook = value.partition("=")
        if not separator or not profile or not codebook or profile in result:
            raise ValueError(f"invalid or duplicate --profile-codebook {value!r}")
        result[profile] = codebook
    return result


def _parse_objective(stdout: str) -> tuple[float, dict]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("policy evaluator produced no JSON result")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict) or not isinstance(payload.get("objective"), (int, float)):
        raise ValueError("policy evaluator's final line must contain numeric objective")
    return float(payload["objective"]), payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True,
                        help="starting schema-2 artifact with every needed codebook")
    parser.add_argument("--promotion-edges", required=True,
                        help="JSON file mapping each profile to ordered promotions")
    parser.add_argument("--byte-budget", required=True, type=int)
    parser.add_argument("--policy-split-sha256", required=True)
    parser.add_argument("--granularity", choices=["tensor", "family", "layer"],
                        default="tensor")
    parser.add_argument("--max-trials", type=int, default=256)
    parser.add_argument("--profile-codebook", action="append", default=[],
                        help="PROFILE=ID when multiple compatible codebooks exist")
    parser.add_argument("--output-spec", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--evaluator-command", nargs=argparse.REMAINDER, required=True,
                        help="argv containing {policy}; final stdout line is JSON objective")
    args = parser.parse_args(argv)
    if not args.evaluator_command or not any("{policy}" in value
                                             for value in args.evaluator_command):
        parser.error("--evaluator-command must contain a {policy} placeholder")
    outputs = (Path(args.output_spec).resolve(), Path(args.output_report).resolve())
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("policy output exists; pass --overwrite")

    reader = ArtifactReader(args.artifact)
    reader.validate()
    quantized = {item["state_dict_name"]: item for item in reader.manifest["tensors"]}
    non_tq1 = reader.manifest["non_tq1_tensors"]
    starting = {
        name: item["profile"]
        for name, item in reader.manifest["resolved_tensor_policy"].items()
    }
    tensors = []
    for name in sorted(starting):
        shape = (quantized[name]["logical_shape"] if name in quantized
                 else non_tq1[name]["shape"])
        tensors.append(PolicyTensor(name, tuple(shape)))
    edges = json.loads(Path(args.promotion_edges).read_text())
    if not isinstance(edges, dict) or not all(
            isinstance(key, str) and isinstance(value, list) and
            all(isinstance(item, str) for item in value)
            for key, value in edges.items()):
        raise ValueError("promotion edges must be a string-to-string-list object")
    evaluations: dict[str, dict] = {}

    def evaluate(policy):
        policy_json = canonical_json(policy)
        digest = hashlib.sha256(policy_json.encode()).hexdigest()
        if digest in evaluations:
            return evaluations[digest]["objective"]
        with tempfile.TemporaryDirectory(prefix="tq1-policy-") as temporary:
            policy_path = Path(temporary) / "policy.json"
            policy_path.write_text(policy_json + "\n")
            command = [value.replace("{policy}", str(policy_path))
                       for value in args.evaluator_command]
            environment = dict(os.environ, TQ1_POLICY_JSON=str(policy_path),
                               TQ1_POLICY_SPLIT_SHA256=args.policy_split_sha256)
            process = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=environment, timeout=args.timeout, check=False)
            if process.returncode:
                raise RuntimeError(
                    f"policy evaluator failed ({process.returncode}):\n{process.stdout[-4000:]}")
            objective, payload = _parse_objective(process.stdout)
        evaluations[digest] = {**payload, "objective": objective,
                               "policy_sha256": digest}
        return objective

    result = greedy_policy_search(
        tensors, starting, edges, byte_budget=args.byte_budget,
        evaluator=evaluate, policy_split_sha256=args.policy_split_sha256,
        max_trials=args.max_trials,
        move_groups=make_move_groups(tensors, args.granularity))
    final_spec = policy_to_spec(
        reader.quant_spec, result.policy,
        profile_codebooks=_profile_codebooks(args.profile_codebook))
    report = {
        "schema": 1,
        "source_artifact": str(Path(args.artifact).resolve()),
        "source_manifest_sha256": _sha256(reader.directory / "tq1_manifest.json"),
        "policy_split_sha256": args.policy_split_sha256,
        "granularity": args.granularity,
        "byte_budget": args.byte_budget,
        "max_trials": args.max_trials,
        "evaluator_command": args.evaluator_command,
        "starting_policy": starting,
        "final_policy": dict(result.policy),
        "final_objective": result.objective,
        "final_total_bytes": result.total_bytes,
        "trials": list(result.trials),
        "evaluator_results": evaluations,
        "quant_spec": final_spec.to_dict(),
        "quant_spec_sha256": final_spec.sha256(),
    }
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(final_spec.canonical_json() + "\n")
    outputs[1].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_spec": str(outputs[0]), "output_report": str(outputs[1]),
        "objective": result.objective, "total_bytes": result.total_bytes,
        "quant_spec_sha256": final_spec.sha256(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
