#!/usr/bin/env python
"""Downstream zero-shot eval via lm-eval-harness (train_plan §10.5): ARC,
HellaSwag, PIQA, WinoGrande, BoolQ. Only after LM loss is healthy — "don't
over-index on benchmarks before PPL recovers". Wraps the converted BitNet model
as an lm-eval model so a single eval mode's accuracy is measured.

lm-eval is an optional heavy dep (importorskip in the test); this driver errors
with a clear install hint if it is absent.

  python train/eval_downstream.py --ckpt <dir> --profile train/profiles/a1.yaml \
      --tasks arc_easy,hellaswag,piqa --mode w_a8 --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train.bitlinear import set_eval_mode  # noqa: E402
from bitnet_train.conversion import load_converted, load_profile  # noqa: E402

DEFAULT_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq"


def build_lm(model, tokenizer, device, batch_size=8):
    """Wrap a HF model as an lm-eval HFLM. The converted model IS an
    AutoModelForCausalLM (wrapped, swapped linears), so HFLM drives it unchanged."""
    try:
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise SystemExit("lm-eval not installed: pip install lm-eval") from e
    return HFLM(pretrained=model, tokenizer=tokenizer, device=str(device),
                batch_size=batch_size)


def run(ckpt, profile, tasks, mode, device, limit=None, batch_size=8, backend="reference"):
    from transformers import AutoTokenizer
    import lm_eval

    tok = AutoTokenizer.from_pretrained(ckpt)
    model, _ = load_converted(ckpt, profile, backend=backend)
    model.to(device).eval()
    set_eval_mode(model, mode)
    lm = build_lm(model, tok, device, batch_size)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks.split(","), limit=limit,
                                  bootstrap_iters=0)
    return {t: {k: v for k, v in m.items() if isinstance(v, (int, float))}
            for t, m in res["results"].items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--mode", default=None, help="eval mode (default: profile's first)")
    ap.add_argument("--limit", type=int, default=None, help="docs per task (smoke)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--backend", default="reference")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    profile = load_profile(args.profile)
    mode = args.mode or profile.eval_modes[0]
    results = run(args.ckpt, profile, args.tasks, mode, device, args.limit,
                  args.batch_size, args.backend)
    print(json.dumps({"mode": mode, "results": results}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"mode": mode, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
