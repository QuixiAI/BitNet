#!/usr/bin/env python3
"""Focused QI-5 A/B for packed CPU routes versus a cached dense-F32 repack."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.codebook import sign_canonical_codebook  # noqa: E402
from bitnet_train.tq1.oracle import dequantize_weight, linear_w2a8  # noqa: E402
from bitnet_train.tq1.packing import pack_payload  # noqa: E402
from bitnet_train.tq1.runtime import (  # noqa: E402
    NativeCPUTQ1Embedding, NativeCPUTQ1Linear, NativeRoutingPolicy)
from bitnet_train.tq1.solver import canonical_shapes  # noqa: E402


def _command(*args: str) -> str:
    try:
        return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()
    except Exception:
        return "unavailable"


def _timing(fn, warmups: int, iterations: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        values.append((time.perf_counter() - started) * 1e3)
    values.sort()
    mean = statistics.fmean(values)
    return {
        "median_ms": statistics.median(values),
        "p20_ms": values[max(0, math.ceil(0.2 * len(values)) - 1)],
        "p80_ms": values[min(len(values) - 1, math.ceil(0.8 * len(values)) - 1)],
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def _codebook(profile: str):
    shapes = canonical_shapes()
    count = 1024 if "v11" in profile else 2048
    values = torch.cat((shapes[(shapes == 0).all(1)],
                        shapes[~(shapes == 0).all(1)][:count - 1]))
    return sign_canonical_codebook(
        f"runtime_route_{count}", "v11" if count == 1024 else "v12", values)


def _oracle_chunks(x: torch.Tensor, payload: torch.Tensor, scales: torch.Tensor,
                   profile: str, book, chunk_rows: int = 256) -> torch.Tensor:
    values = []
    for start in range(0, payload.shape[0], chunk_rows):
        end = min(start + chunk_rows, payload.shape[0])
        values.append(linear_w2a8(
            x, payload[start:end], profile, book, row_scales=scales[start:end],
            activation_mode="a8_token"))
    return torch.cat(values, -1)


def _embedding_lookup_cases(module: NativeCPUTQ1Embedding, payload: torch.Tensor,
                            scales: torch.Tensor, profile: str, book, *,
                            warmups: int, iterations: int) -> list[dict]:
    """Measure packed unique-row gather without materializing the vocabulary."""
    vocab = int(payload.shape[0])
    patterns = {
        "repeated_32_u1": torch.full((1, 32), 7, dtype=torch.int64),
        "prompt_128_u16": (torch.arange(128, dtype=torch.int64) % 16)[None],
        "prompt_512_u256": (torch.arange(512, dtype=torch.int64) % 256)[None],
        "ragged_3x17": (torch.arange(51, dtype=torch.int64).reshape(3, 17) * 7919) % vocab,
    }
    patterns["ragged_3x17"][-1, -1] = vocab - 1
    rows = []
    for name, ids in patterns.items():
        unique, inverse = torch.unique(ids.reshape(-1), sorted=True, return_inverse=True)
        expected_rows = dequantize_weight(
            payload[unique], profile, book, row_scales=scales[unique])
        expected = expected_rows[inverse].reshape(*ids.shape, payload.shape[1] * 256)
        got = module(ids)
        maximum = float((got - expected).abs().max())
        if maximum != 0:
            raise RuntimeError(f"packed embedding lookup differs from its scalar oracle: {maximum}")
        rows.append({
            "name": name, "token_shape": list(ids.shape),
            "token_count": ids.numel(), "unique_token_rows": unique.numel(),
            "includes_last_vocabulary_row": bool(torch.any(ids == vocab - 1)),
            "correctness_tolerance": {"atol": 0.0, "rtol": 0.0},
            "max_abs_error": maximum,
            **_timing(lambda ids=ids: module(ids), warmups, iterations),
        })
    return rows


def _case(profile: str, n: int, k: int, m: int, *, output_head: bool,
          warmups: int, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    book = _codebook(profile)
    legal = np.flatnonzero(book.legal_index_mask().numpy())
    indices = torch.from_numpy(rng.choice(
        legal, size=(n, k // 8), replace=True).astype(np.int64))
    payload = pack_payload(indices, profile)
    scales = torch.from_numpy(
        (rng.random(n, dtype=np.float32) * 0.1).astype(np.float16))
    x = torch.from_numpy(rng.standard_normal((m, k), dtype=np.float32))
    dense_bytes = n * k * 4
    baseline_policy = NativeRoutingPolicy(
        dense_repack_budget_bytes=0, short_prefill_min_tokens=2,
        long_prefill_min_tokens=64)
    candidate_policy = NativeRoutingPolicy(
        dense_repack_budget_bytes=dense_bytes, short_prefill_min_tokens=2,
        long_prefill_min_tokens=64, output_head_dense_decode=output_head)
    cls = NativeCPUTQ1Embedding if output_head else NativeCPUTQ1Linear
    common = dict(row_scales=scales, activation_mode="a8_token",
                  state_dict_name="bench.weight", impl="auto")
    baseline = cls(payload, profile, book, routing_policy=baseline_policy, **common)
    candidate = cls(payload, profile, book, routing_policy=candidate_policy, **common)
    candidate.materialize_dense_repack()
    baseline_fn = (lambda: baseline.linear(x)) if output_head else (lambda: baseline(x))
    candidate_fn = (lambda: candidate.linear(x)) if output_head else (lambda: candidate(x))
    expected = _oracle_chunks(x, payload, scales, profile, book)
    baseline_out, candidate_out = baseline_fn(), candidate_fn()
    baseline_abs = float((baseline_out - expected).abs().max())
    candidate_abs = float((candidate_out - expected).abs().max())
    denominator = float(expected.abs().max()) + 1e-9
    baseline_timing = _timing(baseline_fn, warmups, iterations)
    candidate_timing = _timing(candidate_fn, warmups, iterations)
    row = {
        "schema": 1, "profile": profile, "shape": {"M": m, "N": n, "K": k},
        "workload": "output_head_decode" if output_head else (
            "decode" if m == 1 else "long_prefill" if m >= 64 else "short_prefill"),
        "activation_mode": "a8_token", "output_dtype": "float32",
        "routing_policy": {
            "dense_repack_budget_bytes": dense_bytes,
            "short_prefill_min_tokens": 2,
            "long_prefill_min_tokens": 64,
            "output_head_dense_decode": output_head,
        },
        "correctness_tolerance": {"atol": 2e-5, "rtol": 2e-5},
        "baseline_max_abs_error": baseline_abs,
        "baseline_max_rel_error": baseline_abs / denominator,
        "candidate_max_abs_error": candidate_abs,
        "candidate_max_rel_error": candidate_abs / denominator,
        "baseline": {"route": baseline.repack_report["last_route"],
                     **baseline_timing},
        "candidate": {"route": candidate.repack_report["last_route"],
                      **candidate_timing},
        "speedup": baseline_timing["median_ms"] / candidate_timing["median_ms"],
        "canonical_payload_bytes": payload.numel(),
        "resident_dense_repack_bytes": candidate.repack_report["dense_repack_bytes"],
        "repack_time_ms": candidate.repack_report["dense_repack_time_ms"],
        "repack_sha256": candidate.repack_report["dense_repack_sha256"],
        "repack_layout_version": candidate.repack_report["dense_layout_version"],
        "canonical_packed_remains_resident": True,
    }
    row["candidate_first_use_estimated_ms"] = (
        row["repack_time_ms"] + candidate_timing["median_ms"])
    row["first_use_speedup"] = (
        baseline_timing["median_ms"] / row["candidate_first_use_estimated_ms"])
    if output_head:
        row["embedding_lookup"] = _embedding_lookup_cases(
            baseline, payload, scales, profile, book,
            warmups=warmups, iterations=iterations)
    if max(baseline_abs, candidate_abs) > 2e-5 + 2e-5 * float(expected.abs().max()):
        raise RuntimeError(f"runtime route correctness gate failed: {row}")
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--include-head", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args(argv)
    if args.warmups < 1 or args.iterations < 3:
        raise ValueError("benchmark needs at least one warmup and three iterations")
    cases = []
    if not args.head_only:
        for profile in ("tq1_v11-j-r", "tq1_v12-j-r"):
            for n, k in ((512, 2048), (2048, 2048), (8192, 2048),
                         (2048, 8192), (513, 2048)):
                cases.append((profile, n, k, 32, False))
                cases.append((profile, n, k, 64, False))
                cases.append((profile, n, k, 128, False))
    if args.include_head or args.head_only:
        cases.append(("tq1_v12-j-r", 128256, 2048, 1, True))
    rows = []
    for index, case in enumerate(cases):
        print(f"{case[0]} M{case[3]} N{case[1]} K{case[2]}", flush=True)
        rows.append(_case(*case[:4], output_head=case[4], warmups=args.warmups,
                          iterations=args.iterations, seed=args.seed + index))
    meta = {
        "schema": 1, "date": dt.date.today().isoformat(),
        "command": " ".join(sys.argv), "warmups": args.warmups,
        "iterations": args.iterations, "seed": args.seed,
        "device": platform.machine(), "processor": platform.processor(),
        "macos": _command("sw_vers"), "clang": _command("clang", "--version"),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "git": _command("git", "rev-parse", "--short", "HEAD") + "-dirty",
        "energy": "not measured by this kernel-focused A/B",
    }
    document = {"metadata": meta, "results": rows}
    output = (Path(args.output) if args.output else
              Path(__file__).resolve().parent / "results" / dt.date.today().isoformat()
              / "tq1-runtime-routes" / "run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows),
                      "max_abs_error": max(max(row["baseline_max_abs_error"],
                                               row["candidate_max_abs_error"])
                                           for row in rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
