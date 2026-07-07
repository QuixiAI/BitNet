#!/usr/bin/env python
"""Heal scaling study (train_plan §11.6): tokens-to-recovery curves from the
metrics.jsonl of A1/A3 (or any two) runs — the forecasting basis that turns
"PoC 1-3B / production 10-150B" into a number per track. Reads the eval blocks
(val_ce_primary / ppl / kl_tf vs tokens), fits how many tokens each run needs to
reach a target PPL (or KL_tf), and reports the ratio.

  python train/scaling_report.py --runs runs/a1-t2 runs/a3-t2 --labels A1 A3 \
      --target-ppl 20 --out scaling_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _curve(run_dir, key):
    mpath = Path(run_dir) / "metrics.jsonl"
    pts = []
    for line in mpath.read_text().splitlines():
        m = json.loads(line)
        if key in m and "tokens" not in m and "step" in m:
            pass
        if key in m:
            tok = m.get("tokens")
            if tok is not None:
                pts.append((tok, m[key]))
    return sorted(pts)


def tokens_to_target(curve, target, decreasing=True) -> float | None:
    """First token count at which the metric crosses the target (linear interp)."""
    for (t0, v0), (t1, v1) in zip(curve, curve[1:]):
        hit = (v1 <= target) if decreasing else (v1 >= target)
        prev = (v0 > target) if decreasing else (v0 < target)
        if hit and prev and v1 != v0:
            frac = (v0 - target) / (v0 - v1) if decreasing else (target - v0) / (v1 - v0)
            return t0 + frac * (t1 - t0)
        if hit:
            return t1
    return None


def build_report(runs, labels, target_ppl, target_kl) -> str:
    lines = ["# Heal scaling study (train_plan §11.6)", ""]
    key, tgt = ("ppl_w_a8", target_ppl) if target_ppl else ("kl_tf", target_kl)
    lines.append(f"Target: {key} = {tgt}")
    lines.append("")
    lines.append("| run | evals | final tokens | final " + key + " | tokens-to-target |")
    lines.append("|---|---|---|---|---|")
    ttts = {}
    for run, label in zip(runs, labels):
        try:
            curve = _curve(run, key)
        except FileNotFoundError:
            lines.append(f"| {label} | _no metrics.jsonl_ | | | |")
            continue
        if not curve:
            lines.append(f"| {label} | _no {key} points_ | | | |")
            continue
        ttt = tokens_to_target(curve, tgt, decreasing=True)
        ttts[label] = ttt
        lines.append(f"| {label} | {len(curve)} | {curve[-1][0]:,} | {curve[-1][1]:.3f} "
                     f"| {('%.2e' % ttt) if ttt else 'not reached'} |")
    if len(ttts) == 2 and all(v for v in ttts.values()):
        a, b = list(ttts.values())
        lines += ["", f"**Scaling ratio ({labels[1]}/{labels[0]}): "
                  f"{b / a:.2f}×** tokens-to-target — the §11.6 forecasting basis "
                  "for G-track / production budgets."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--target-ppl", type=float, default=None)
    ap.add_argument("--target-kl", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.target_ppl and not args.target_kl:
        raise SystemExit("pass --target-ppl or --target-kl")
    labels = args.labels or [Path(r).name for r in args.runs]
    report = build_report(args.runs, labels, args.target_ppl, args.target_kl)
    if args.out:
        Path(args.out).write_text(report)
        print(f"[scaling] wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
