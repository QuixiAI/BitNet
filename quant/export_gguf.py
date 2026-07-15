#!/usr/bin/env python
"""Exact schema-2 TQ1 artifact to GGUF exporter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.gguf import export_tq1_gguf  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--converter", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = export_tq1_gguf(
        args.artifact, args.output, converter=args.converter,
        overwrite=args.overwrite,
        command=("quant/export_gguf.py", *(argv if argv is not None else sys.argv[1:])))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
