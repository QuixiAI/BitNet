#!/usr/bin/env python3
"""Assemble independently measured baselines into a quality-qualified report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.artifact import ArtifactReader  # noqa: E402
from bitnet_train.tq1.evaluation import validate_quality_report  # noqa: E402


def _assignment(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("profile must be LABEL=report.json")
    return label, Path(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="assemble and validate the complete TQ1 quality matrix")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--profile", action="append", type=_assignment, required=True,
                        help="repeat LABEL=evaluation-component.json")
    parser.add_argument("--release-profile", required=True,
                        help="profile whose stratified buckets represent this release")
    parser.add_argument("--evidence", required=True,
                        help="JSON containing predeclared_gates/downstream_tasks/"
                             "instruction_chat/long_context/calibration_convergence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    reader = ArtifactReader(args.artifact)
    reader.validate()
    spec_hash = reader.manifest["quant_spec_sha256"]
    components = {}
    identity = None
    commands = []
    provenance = {}
    for label, path in args.profile:
        if label in components:
            raise ValueError(f"duplicate profile label {label!r}")
        component = json.loads(path.read_text())
        if component.get("schema") != 1 or component.get("profile") != label:
            raise ValueError(f"{path}: component schema/profile mismatch")
        if component.get("quant_spec_sha256") != spec_hash:
            raise ValueError(f"{path}: QuantSpec mismatch")
        current = component.get("evaluation_data")
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError("profile components used different held-out data")
        components[label] = component
        commands.extend(component.get("command", []))
        provenance[label] = component.get("provenance", {})
    if args.release_profile not in components:
        raise ValueError("release profile has no evaluation component")
    evidence = json.loads(Path(args.evidence).read_text())
    evidence_keys = {
        "predeclared_gates", "downstream_tasks", "instruction_chat",
        "long_context", "calibration_convergence",
    }
    if set(evidence) != evidence_keys:
        raise ValueError(f"evidence JSON must contain exactly {sorted(evidence_keys)}")
    report = {
        "schema": 1,
        "quant_spec_sha256": spec_hash,
        "evaluation_data": identity,
        "profiles": {label: component["metrics"]
                     for label, component in sorted(components.items())},
        "stratified": components[args.release_profile]["stratified"],
        "commands": commands,
        "provenance": provenance,
        **evidence,
    }
    validate_quality_report(report, spec_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "profiles": sorted(components)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
