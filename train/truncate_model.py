#!/usr/bin/env python
"""Slice a truncated smoke model out of a real checkpoint (moe_train_plan §5.4:
"a short single-node smoke run on a truncated model (e.g. 4 layers x 16 experts
sliced from the checkpoint) validates bf16-latents/fp32-master/8-bit-Adam
mechanics before the first multi-node job").

Streams the safetensors shards — never materializes the full model — keeping
only layers < --layers, slicing 3-D expert stacks (dim 0) and the router rows
to --experts, and rewriting the config accordingly.

  python train/truncate_model.py --ckpt <hf-dir> --out runs/q15b-smoke \
      --layers 4 --experts 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# on-disk MoE layouts vary by version: 3-D fused stack (experts.gate_up_proj) or
# per-expert 2-D (experts.<N>.gate_proj.weight). Handle both.
_EXPERT_STACK_RE = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)$")
_EXPERT_IDX_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")
_ROUTER_RE = re.compile(r"\.mlp\.gate\.weight$")


def truncate(ckpt: str | Path, out: str | Path, layers: int,
             experts: int | None = None) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig

    ckpt, out = Path(ckpt), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(ckpt.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors under {ckpt}")

    kept: dict[str, torch.Tensor] = {}
    dropped = 0
    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            for name in sf.keys():
                m = _LAYER_RE.search(name)
                if m and int(m.group(1)) >= layers:
                    dropped += 1
                    continue
                if experts is not None:
                    em = _EXPERT_IDX_RE.search(name)
                    if em and int(em.group(1)) >= experts:   # drop per-expert 2-D tensor
                        dropped += 1
                        continue
                t = sf.get_tensor(name)
                if experts is not None:
                    if _EXPERT_STACK_RE.search(name):        # slice the 3-D fused stack
                        t = t[:experts].contiguous()
                    elif _ROUTER_RE.search(name):            # router rows -> selected experts
                        t = t[:experts].contiguous()
                kept[name] = t
    save_file(kept, str(out / "model.safetensors"),
              metadata={"format": "pt"})

    cfg = AutoConfig.from_pretrained(ckpt)
    cfg.num_hidden_layers = layers
    if experts is not None and hasattr(cfg, "num_experts"):
        # transformers 5 serializes the count as num_local_experts; num_experts is
        # a read alias that does NOT round-trip through save_pretrained
        for attr in ("num_experts", "num_local_experts"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, experts)
        cfg.num_experts_per_tok = min(cfg.num_experts_per_tok, experts)
    if hasattr(cfg, "mlp_only_layers"):
        cfg.mlp_only_layers = [l for l in (cfg.mlp_only_layers or []) if l < layers]
    cfg.save_pretrained(out)
    for tok_file in ckpt.glob("tokenizer*"):
        (out / tok_file.name).write_bytes(tok_file.read_bytes())

    report = {"layers": layers, "experts": experts, "kept_tensors": len(kept),
              "dropped_tensors": dropped,
              "params": sum(t.numel() for t in kept.values())}
    (out / "truncate_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--experts", type=int, default=None)
    args = ap.parse_args()
    rep = truncate(args.ckpt, args.out, args.layers, args.experts)
    print(f"[truncate] {rep['kept_tensors']} tensors kept ({rep['params'] / 1e6:.1f}M "
          f"params), {rep['dropped_tensors']} dropped -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
