#!/usr/bin/env python3
"""A/B the exact A8 rounding kernel against a separately built baseline.

The two dylibs/so files are loaded in one process so ctypes overhead, input
vectors, sample ordering, and thermal conditions are shared.  This harness is
intentionally separate from ``bench_kernels.py`` because its baseline is a
historical source build, not another symbol in the current library.
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


_F32 = np.ctypeslib.ndpointer(np.float32, flags="C")
_I8 = np.ctypeslib.ndpointer(np.int8, flags="C")


def _load(path: Path):
    library = C.CDLL(str(path.resolve()))
    function = library.bn_act_quant_int8
    function.restype = C.c_float
    function.argtypes = [_F32, C.c_int64, _I8]
    return library, function


def _percentile(values: list[float], q: int) -> float:
    return float(np.percentile(np.asarray(values), q, method="linear"))


def _run(function, value: np.ndarray, output: np.ndarray) -> float:
    return float(function(value, value.size, output))


def _identity() -> dict[str, str]:
    def command(*args: str) -> str:
        try:
            return subprocess.run(
                args, check=True, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {
        "device": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "compiler": command("cc", "--version").splitlines()[0],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-lib", type=Path, required=True)
    parser.add_argument("--candidate-lib", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--widths", default="512,2048,8192")
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--target-elements", type=int, default=2_000_000)
    parser.add_argument("--minimum-calls", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    widths = tuple(int(value) for value in args.widths.split(","))
    if any(value <= 0 for value in widths) or args.warmups < 1 \
            or args.samples < 3 or args.target_elements < 1 \
            or args.minimum_calls < 1:
        raise ValueError("widths and timing counts must be positive")

    libraries = []
    functions = {}
    for label, path in ((args.baseline_label, args.baseline_lib),
                        (args.candidate_label, args.candidate_lib)):
        library, function = _load(path)
        libraries.append(library)  # retain the handle for the complete run
        functions[label] = function

    rows: list[dict[str, object]] = [{
        "kind": "metadata",
        "benchmark": "bn_act_quant_int8_rounding_ab",
        "baseline_revision": args.baseline_revision,
        "baseline_library": str(args.baseline_lib.resolve()),
        "candidate_library": str(args.candidate_lib.resolve()),
        "widths": widths,
        "warmups": args.warmups,
        "samples": args.samples,
        "target_elements": args.target_elements,
        "minimum_calls": args.minimum_calls,
        "seed": args.seed,
        "effective_bytes_per_element": 5,
        **_identity(),
    }]

    fixture = np.asarray([1.6625983, -1.0669429, 0.0], dtype=np.float32)
    fixture_scale = np.float32(np.max(np.abs(fixture)) / np.float32(127.0))
    fixture_reference = np.rint(fixture / fixture_scale).clip(-127, 127).astype(np.int8)
    candidate_correct = True
    for label, function in functions.items():
        output = np.empty(fixture.size, dtype=np.int8)
        scale = _run(function, fixture, output)
        matches = bool(np.array_equal(output, fixture_reference))
        candidate_correct &= label != args.candidate_label or matches
        rows.append({
            "kind": "correctness",
            "implementation": label,
            "fixture": fixture.tolist(),
            "codes": output.tolist(),
            "reference_codes": fixture_reference.tolist(),
            "scale": scale,
            "matches_reference": matches,
            "max_code_error": int(np.max(np.abs(
                output.astype(np.int16) - fixture_reference.astype(np.int16)))),
        })

    rng = np.random.default_rng(args.seed)
    for width in widths:
        value = np.ascontiguousarray(rng.standard_normal(width), dtype=np.float32)
        scale = np.float32(np.max(np.abs(value)) / np.float32(127.0))
        reference = np.rint(value / scale).clip(-127, 127).astype(np.int8)
        outputs = {label: np.empty(width, dtype=np.int8) for label in functions}
        for label, function in functions.items():
            for _ in range(args.warmups):
                _run(function, value, outputs[label])
        calls = max(args.minimum_calls, args.target_elements // width)
        samples = {label: [] for label in functions}
        labels = list(functions)
        for sample in range(args.samples):
            order = labels if sample % 2 == 0 else labels[::-1]
            for label in order:
                start = time.perf_counter_ns()
                for _ in range(calls):
                    _run(functions[label], value, outputs[label])
                samples[label].append(
                    (time.perf_counter_ns() - start) / calls / 1e6)
        baseline_median = statistics.median(samples[args.baseline_label])
        for label in labels:
            values = samples[label]
            median = statistics.median(values)
            matches = bool(np.array_equal(outputs[label], reference))
            candidate_correct &= label != args.candidate_label or matches
            rows.append({
                "kind": "timing",
                "width": width,
                "implementation": label,
                "calls_per_sample": calls,
                "p20_ms": _percentile(values, 20),
                "median_ms": median,
                "p80_ms": _percentile(values, 80),
                "cv": statistics.pstdev(values) / statistics.mean(values),
                "baseline_over_candidate": baseline_median / median,
                "effective_read_write_GBps": width * 5 / (median / 1e3) / 1e9,
                "matches_numpy_divide": matches,
                "max_code_error": int(np.max(np.abs(
                    outputs[label].astype(np.int16) - reference.astype(np.int16)))),
            })

    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not candidate_correct:
        raise SystemExit("candidate failed the exact NumPy-division rounding oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
