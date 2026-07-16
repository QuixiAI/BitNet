#!/usr/bin/env python
"""Validate a pinned QI-3 suite and recompute capability quality gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.evaluation import (  # noqa: E402
    canonical_document_sha256, validate_capability_report,
    validate_capability_suite)


def _read(path: str) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--report")
    parser.add_argument("--quant-spec-sha256")
    args = parser.parse_args(argv)
    suite = _read(args.suite)
    validate_capability_suite(suite)
    result = {"suite_sha256": canonical_document_sha256(suite), "valid": True}
    if args.report:
        if not args.quant_spec_sha256:
            parser.error("--quant-spec-sha256 is required with --report")
        result["gate_decisions"] = validate_capability_report(
            _read(args.report), suite, quant_spec_sha256=args.quant_spec_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
