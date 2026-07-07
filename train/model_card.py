#!/usr/bin/env python
"""Model-card generator (train_plan §13.3 release checklist / moe_train_plan §0.5).

Assembles the honest-framing card from run artifacts so the stakeholder language
is the path of least resistance, not an afterthought. Pulls what it can from the
conversion report, provenance, parity report, and eval JSONs; leaves TODO
markers for numbers a human must sign off (dense-baseline gap, heal tokens).

  python train/model_card.py --run runs/a1-t2 --conversion runs/a1-init/conversion_report.json \
      --parity runs/a1-t2/parity_report.json --out runs/a1-t2/MODEL_CARD.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else {}


TEMPLATE = """# {name}

**A Llama/Gemma/Qwen-shaped 1.58-bit heal — NOT a reproduction of Microsoft's
BitNet b1.58 2B4T** (different architecture, FFN, norms; not a 4T-token
from-scratch regime). In the lineage of the quantize-and-heal work
(e.g. HF1BitLLM/Llama3-8B-1.58-100B-tokens).

## Base & license
- Base model: `{base_model}`
- License: {license} (inherit the base's; confirm before release)
- Provenance: quantizer `{quantizer_hash}`, config `{config_hash}`, git `{git_rev}`

## What is ternary
- Ternarized: {n_ternarized} BitLinear + {n_expert_stacks} fused expert stacks
- Ternarized parameter fraction (of stored params): **{param_frac:.1%}**
- Ternarized-FLOPs fraction (per-token linear compute): **{flop_frac:.1%}**
- Kept full precision: {kept_fp}
- Packed size vs naive bits-per-param estimate: TODO (from export)

## Training
- Heal token count: {heal_tokens}
- Teacher: `{teacher}` (KD: {kd})
- Recipe: AdamW(0.9,0.95) clip 1.0, wd 0.1 on 2-D latents; {precision}

## Measured quality (state eval mode + runtime for every number)
- Eval mode / runtime the numbers below were measured in: {eval_mode}
- Validation PPL (healed): {ppl}
- Dense-baseline gap on the standard evals: TODO (fill from eval_downstream)
- KL-to-teacher trajectory: {kl_tf}

## Export & parity
- Export route(s): {routes}
- Tensor parity: {parity_status}
- Runtime PPL parity: {ppl_parity}

## Known limitations
- {limitations}
- FP-embedding quality floor is structural (embeddings/PLE stay FP).
"""


def build_card(args) -> str:
    conv = _load(args.conversion)
    prov = _load(args.provenance)
    parity = _load(args.parity)
    t0 = _load(args.t0_report)
    metrics = []
    if args.run:
        mpath = Path(args.run) / "metrics.jsonl"
        if mpath.exists():
            metrics = [json.loads(l) for l in mpath.read_text().splitlines()]
    evals = [m for m in metrics if "val_ce_primary" in m]
    last = evals[-1] if evals else {}

    return TEMPLATE.format(
        name=args.name,
        base_model=conv.get("base_model") or prov.get("base_model", "TODO"),
        license=args.license or "TODO",
        quantizer_hash=prov.get("quantizer_hash", "?"),
        config_hash=prov.get("config_hash", "?"),
        git_rev=prov.get("git_rev", "?"),
        n_ternarized=conv.get("n_ternarized", "?"),
        n_expert_stacks=conv.get("n_expert_stacks", 0),
        param_frac=conv.get("ternary_param_fraction", 0.0),
        flop_frac=conv.get("ternary_flop_fraction", 0.0),
        kept_fp=", ".join(conv.get("kept_fp", [])[:6]) or "TODO",
        heal_tokens=(f"{metrics[-1]['tokens']:,}" if metrics else "TODO"),
        teacher=prov.get("args", {}).get("teacher") or args.teacher or "TODO",
        kd=prov.get("kd", "TODO"),
        precision=prov.get("args", {}).get("latent_dtype", "fp32") + " latents",
        eval_mode=(",".join(t0.get("eval_modes", [])) or "TODO"),
        ppl=(f"{last.get('ppl_' + (args.eval_mode or 'w_a8'), '?')}" if last else "TODO"),
        kl_tf=(f"{last.get('kl_tf', '?')}" if last else "TODO"),
        routes=", ".join(conv.get("export_route", [])) or args.routes or "TODO",
        parity_status=("PASS" if parity.get("ok") else parity.get("ok", "TODO")),
        ppl_parity=json.dumps(parity.get("ppl_parity", {})) if parity else "TODO",
        limitations=args.limitations or "text-only (if Gemma); long-context caveats "
        "if seq-4096 training was skipped",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Model-1.58")
    ap.add_argument("--run", default=None, help="run dir with metrics.jsonl")
    ap.add_argument("--conversion", default=None)
    ap.add_argument("--provenance", default=None)
    ap.add_argument("--parity", default=None)
    ap.add_argument("--t0-report", default=None)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--license", default=None)
    ap.add_argument("--routes", default=None)
    ap.add_argument("--eval-mode", default=None)
    ap.add_argument("--limitations", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    card = build_card(args)
    if args.out:
        Path(args.out).write_text(card)
        print(f"[model_card] wrote {args.out}")
    else:
        print(card)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
