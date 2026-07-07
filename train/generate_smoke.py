#!/usr/bin/env python
"""Fixed-prompt generation smoke (train_plan §10.5): a small set of prompts run
through the converted student in each eval mode + the frozen packed inference
path, outputs recorded for eyeballing and diffing across checkpoints. Not a
metric — a qualitative canary (repetition/collapse show here before PPL moves).

  python train/generate_smoke.py --ckpt <dir> --profile train/profiles/a1.yaml \
      --prompts train/prompts_smoke.txt --modes w_a8,w_only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train.bitlinear import BitLinear, set_eval_mode  # noqa: E402
from bitnet_train.conversion import load_converted, load_profile  # noqa: E402

DEFAULT_PROMPTS = [
    "The capital of France is",
    "In a shocking turn of events,",
    "def fibonacci(n):",
    "The three laws of thermodynamics are",
    "Once upon a time, there was a",
]


@torch.no_grad()
def generate(model, tok, prompts, device, max_new_tokens=48, greedy=True) -> list[dict]:
    model.eval()
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        gen = model.generate(ids, do_sample=not greedy, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id or 0,
                             temperature=1.0 if not greedy else None)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        out.append({"prompt": p, "completion": text})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--prompts", default=None, help="one prompt/line (else built-in set)")
    ap.add_argument("--modes", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--sample", action="store_true", help="temperature-1 sampling")
    ap.add_argument("--frozen", action="store_true",
                    help="also run the packed inference path (BitLinear.freeze)")
    ap.add_argument("--backend", default="reference")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    profile = load_profile(args.profile)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model, _ = load_converted(args.ckpt, profile, backend=args.backend)
    model.to(device)
    prompts = ([l for l in Path(args.prompts).read_text().splitlines() if l.strip()]
               if args.prompts else DEFAULT_PROMPTS)
    modes = args.modes.split(",") if args.modes else profile.eval_modes

    report = {"ckpt": str(args.ckpt), "runs": {}}
    for mode in modes:
        set_eval_mode(model, mode)
        report["runs"][mode] = generate(model, tok, prompts, device,
                                        args.max_new_tokens, greedy=not args.sample)
        print(f"\n=== mode {mode} ===")
        for r in report["runs"][mode]:
            print(f"  {r['prompt']!r} -> {r['completion']!r}")
    if args.frozen:
        if device != "mps" and getattr(device, "type", device) != "mps":
            print("\n[frozen] packed inference path is MPS-only; skipping on "
                  f"{device}")
            args.frozen = False
    if args.frozen:
        set_eval_mode(model, profile.eval_modes[0])
        for m in model.modules():
            if isinstance(m, BitLinear):
                m.freeze()
        report["runs"]["frozen_packed"] = generate(model, tok, prompts, device,
                                                   args.max_new_tokens, greedy=not args.sample)
        print("\n=== frozen packed path ===")
        for r in report["runs"]["frozen_packed"]:
            print(f"  {r['prompt']!r} -> {r['completion']!r}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
