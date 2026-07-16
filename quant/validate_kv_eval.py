#!/usr/bin/env python
"""Validate a QI-4 KV evaluation report against exact artifact hashes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.kv_cache import validate_kv_evaluation_report  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--calibration-artifact-sha256", required=True)
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text())
    validate_kv_evaluation_report(
        report, model_artifact_sha256=args.model_artifact_sha256,
        calibration_artifact_sha256=args.calibration_artifact_sha256)
    print(json.dumps({"valid": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
