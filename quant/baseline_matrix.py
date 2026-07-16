#!/usr/bin/env python3
"""Create and advance the hash-bound QI-1 baseline matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.baseline import (  # noqa: E402
    BaselineMatrix, required_baseline_rows, result_template)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--identity", required=True)
    create.add_argument("--output", required=True)
    template = sub.add_parser("template")
    template.add_argument("--matrix", required=True)
    template.add_argument("--row", required=True)
    template.add_argument("--output", required=True)
    record = sub.add_parser("record")
    record.add_argument("--matrix", required=True)
    record.add_argument("--result", required=True)
    gates = sub.add_parser("declare-gates")
    gates.add_argument("--matrix", required=True)
    gates.add_argument("--gates", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--matrix", required=True)
    args = parser.parse_args(argv)

    if args.command == "create":
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        identity = json.loads(Path(args.identity).read_text())
        matrix = BaselineMatrix.create(identity, rows=required_baseline_rows())
        matrix.write(output)
    elif args.command == "template":
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        matrix = BaselineMatrix.load(args.matrix)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(
            result_template(matrix, args.row), indent=2, sort_keys=True) + "\n")
    elif args.command == "record":
        matrix = BaselineMatrix.load(args.matrix)
        matrix.record_result(json.loads(Path(args.result).read_text()))
        matrix.write(args.matrix)
    elif args.command == "declare-gates":
        matrix = BaselineMatrix.load(args.matrix)
        matrix.declare_gates(json.loads(Path(args.gates).read_text()))
        matrix.write(args.matrix)
    else:
        matrix = BaselineMatrix.load(args.matrix)
        matrix.validate()
    print(json.dumps({
        "matrix": str(Path(getattr(args, "matrix", getattr(args, "output", ""))).resolve()),
        "status": matrix.document["status"],
        "recorded_rows": (["dense_teacher"] if matrix.document["dense_result"] else [])
        + sorted(matrix.document["candidate_results"]),
        "gate_sha256": matrix.document["gate_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
