#!/usr/bin/env python
"""Merge schema-1 TQ1 statistics by raw sums and counts (never normalized means)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.calibration import (  # noqa: E402
    file_sha256, load_calibration_artifact, merge_calibration_artifacts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="two or more statistics safetensors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = merge_calibration_artifacts(
        args.inputs, args.output, overwrite=args.overwrite,
        metadata={"merge_command": ["quant/merge_calibration.py",
                                    *(argv if argv is not None else sys.argv[1:])]})
    _, metadata = load_calibration_artifact(result)
    print(json.dumps({
        "output": str(result), "sha256": file_sha256(result),
        "records": metadata["records"],
        "retained_tokens": metadata["retained_tokens"],
        "source_artifacts": metadata["source_artifacts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
