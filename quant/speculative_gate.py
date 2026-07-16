#!/usr/bin/env python
"""Evaluate the measured QI-6 cost gate before constructing a drafter."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.speculative import evaluate_speculative_cost  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", required=True)
    args = parser.parse_args(argv)
    value = json.loads(Path(args.measurement).read_text())
    print(json.dumps(asdict(evaluate_speculative_cost(value)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
