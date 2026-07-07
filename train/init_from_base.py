#!/usr/bin/env python
"""Convert + T0 validation + damage map (train_plan §7.0 file #5, §11.1).

Pipeline: load the dense base → (optional) dense calibration PPL → profile-driven
conversion → save converted checkpoint (fp32 latents) + conversion_report.json +
provenance.json → field-by-field config diff HARD FAIL (rope_scaling above all) →
optional damage decomposition: eval-only passes for A1d (ternary W only), A1b
(A8 only), A2 (both), then module-family subsets (each family quantized W+A8
while everything else runs dense-FP) and optional coarse per-layer passes.

Usage:
  python train/init_from_base.py --profile train/profiles/a1.yaml --out runs/a1-init \
      [--data-dir train/data/llama3 --damage-map] [--backend reference] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train import provenance  # noqa: E402
from bitnet_train.bitlinear import iter_bitlinears, quant_toggled, ternary_health  # noqa: E402
from bitnet_train.conversion import convert, diff_config, load_profile  # noqa: E402
from bitnet_train.data import calibration_windows, load_manifest  # noqa: E402
from eval_ppl import evaluate_ppl  # noqa: E402


def _family_sets(model) -> dict[str, set[str]]:
    """Module-family name sets for the damage decomposition (§11.1):
    qkv / o / gate_up / down, derived from the converted names."""
    fams = {"qkv": set(), "o": set(), "gate_up": set(), "down": set()}
    for name, _ in iter_bitlinears(model):
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("q_proj", "k_proj", "v_proj"):
            fams["qkv"].add(name)
        elif leaf == "o_proj":
            fams["o"].add(name)
        elif leaf in ("gate_proj", "up_proj"):
            fams["gate_up"].add(name)
        elif leaf == "down_proj":
            fams["down"].add(name)
    return {k: v for k, v in fams.items() if v}


def damage_map(model, windows, device, per_layer_chunks: int = 0) -> dict:
    all_names = {n for n, _ in iter_bitlinears(model)}
    out = {}

    def run(tag, **toggle):
        with quant_toggled(model, **toggle):
            r = evaluate_ppl(model, windows, device, mode=None)
        out[tag] = {"ce": r["ce"], "ppl": r["ppl"]}
        print(f"[damage] {tag:24s} ce={r['ce']:.4f} ppl={r['ppl']:.2f}")

    run("dense_fp", names=all_names, act_quant=False, weight_ternary=False)
    run("A1d_ternary_only", names=all_names, act_quant=False, weight_ternary=True)
    run("A1b_a8_only", names=all_names, act_quant=True, weight_ternary=False)
    run("A2_full", names=all_names, act_quant=True, weight_ternary=True)

    for fam, names in _family_sets(model).items():
        # family runs W+A8 (its normal forward); the complement runs dense-FP
        run(f"family_{fam}", names=all_names - names,
            act_quant=False, weight_ternary=False)

    if per_layer_chunks > 0:
        by_layer = defaultdict(set)
        for name in all_names:
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                by_layer[int(m.group(1))].add(name)
        layers = sorted(by_layer)
        step = max(1, len(layers) // per_layer_chunks)
        for i in range(0, len(layers), step):
            chunk = layers[i:i + step]
            names = set().union(*(by_layer[l] for l in chunk))
            run(f"layers_{chunk[0]}-{chunk[-1]}", names=all_names - names,
                act_quant=False, weight_ternary=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="reference", choices=["reference", "metal"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-dir", default=None, help="shards for calibration eval")
    ap.add_argument("--calib-windows", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--damage-map", action="store_true")
    ap.add_argument("--per-layer-chunks", type=int, default=0)
    ap.add_argument("--skip-base-eval", action="store_true")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    profile = load_profile(args.profile)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] loading dense base {profile.base_model}")
    dense_cfg = AutoConfig.from_pretrained(profile.base_model)
    model = AutoModelForCausalLM.from_pretrained(profile.base_model,
                                                 torch_dtype=torch.float32)

    windows = None
    report = {"profile": profile.name, "base_model": profile.base_model}
    if args.data_dir:
        windows = calibration_windows(args.data_dir, args.calib_windows,
                                      split="val", seq_len=args.seq_len)
        if not args.skip_base_eval:
            model.to(device)
            r = evaluate_ppl(model, windows, device, mode=None)
            report["dense_baseline"] = {"ce": r["ce"], "ppl": r["ppl"]}
            print(f"[init] dense baseline ce={r['ce']:.4f} ppl={r['ppl']:.2f}")

    print("[init] converting")
    conv = convert(model, profile, backend=args.backend)
    print(f"[init] {conv.n_ternarized} linears + {conv.n_expert_stacks} expert "
          f"stacks ternarized; {conv.n_kept_fp} kept FP; "
          f"param_frac={conv.ternary_param_fraction:.3f} "
          f"flop_frac={conv.ternary_flop_fraction:.3f}")

    # config preservation: hard requirement (train_plan §2.2)
    diffs = diff_config(dense_cfg, model.config)
    if diffs:
        print(f"[init] FATAL config drift: {json.dumps(diffs, indent=2, default=str)}")
        return 1

    model.save_pretrained(out_dir)
    try:
        AutoTokenizer.from_pretrained(profile.base_model).save_pretrained(out_dir)
    except OSError as e:                       # tokenizer optional for random bases
        print(f"[init] tokenizer not saved: {e}")
    conv.to_json(out_dir / "conversion_report.json")
    meta = provenance.build_meta(
        profile_path=args.profile, model_config=model.config,
        data_manifest=load_manifest(args.data_dir) if args.data_dir else None,
        extra={"backend": args.backend, "base_model": profile.base_model})
    (out_dir / "provenance.json").write_text(json.dumps(meta, indent=2))

    # re-verify the saved config against the original, catching save-side drift
    saved_cfg = AutoConfig.from_pretrained(out_dir)
    diffs = diff_config(dense_cfg, saved_cfg)
    if diffs:
        print(f"[init] FATAL saved-config drift: {json.dumps(diffs, default=str)}")
        return 1
    print("[init] config preservation: clean")

    if windows is not None:
        model.to(device)
        for mode in profile.eval_modes:
            r = evaluate_ppl(model, windows, device, mode=mode)
            report[f"converted_{mode}"] = {"ce": r["ce"], "ppl": r["ppl"]}
            print(f"[init] converted mode={mode:7s} ce={r['ce']:.4f} ppl={r['ppl']:.2f}")
        if args.damage_map:
            report["damage_map"] = damage_map(model, windows, device,
                                              args.per_layer_chunks)
        report["ternary_health_head"] = dict(list(ternary_health(model).items())[:4])

    (out_dir / "t0_report.json").write_text(json.dumps(report, indent=2))
    print(f"[init] wrote {out_dir}/t0_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
