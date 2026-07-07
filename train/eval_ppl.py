#!/usr/bin/env python
"""Fixed-calibration-set PPL + KL_tf per eval mode (train_plan §7.0 file #6, §8.4).

The calibration set is the first N val windows in manifest order — deterministic,
and every report embeds the manifest hash so numbers are only ever compared on
the same frozen corpus. CE is computed row-chunked (never a full (T, V) fp32
tensor). Eval modes are the §8.4 / moe §7.5 matrix, applied via
bitlinear.set_eval_mode on the SAME trained model.

Usage:
  python train/eval_ppl.py --ckpt <dir> --profile train/profiles/a1.yaml \
      --data-dir train/data/llama3 --modes w_a8,w_only [--teacher <hf-id>] \
      [--calib-windows 64] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train import provenance  # noqa: E402
from bitnet_train.bitlinear import set_eval_mode  # noqa: E402
from bitnet_train.conversion import load_converted, load_profile  # noqa: E402
from bitnet_train.data import calibration_windows, load_manifest, manifest_hash  # noqa: E402

ROW_CHUNK = 128


@torch.no_grad()
def window_ce(model, ids: torch.Tensor, row_chunk: int = ROW_CHUNK) -> tuple[float, int]:
    """Sum of next-token CE (nats) and token count for one (B, T) batch,
    row-chunked so logits are cast to fp32 only a slice at a time."""
    out = model(ids)
    logits = out.logits[:, :-1, :]
    targets = ids[:, 1:]
    total, count = 0.0, 0
    for s in range(0, targets.shape[1], row_chunk):
        sl = logits[:, s:s + row_chunk, :].float()
        tg = targets[:, s:s + row_chunk]
        total += float(F.cross_entropy(sl.reshape(-1, sl.shape[-1]), tg.reshape(-1),
                                       reduction="sum"))
        count += tg.numel()
    return total, count


@torch.no_grad()
def evaluate_ppl(model, windows: torch.Tensor, device, batch_size: int = 1,
                 mode: str | None = None) -> dict:
    """PPL over the calibration windows in one eval mode (None = leave as-is)."""
    if mode is not None:
        set_eval_mode(model, mode)
    model.eval()
    total, count = 0.0, 0
    for i in range(0, windows.shape[0], batch_size):
        ids = windows[i:i + batch_size].to(device)
        t, c = window_ce(model, ids)
        total += t
        count += c
    ce = total / max(count, 1)
    return {"mode": mode, "ce": ce, "ppl": float(torch.tensor(ce).exp()),
            "tokens": count}


@torch.no_grad()
def kl_tf(student, teacher, windows: torch.Tensor, device, tau: float = 1.0,
          batch_size: int = 1, row_chunk: int = ROW_CHUNK) -> float:
    """Teacher-forced KL(teacher ‖ student) per token on the calibration set —
    the truest healing gauge (train_plan §10.1). Row-chunked."""
    student.eval()
    teacher.eval()
    total, count = 0.0, 0
    for i in range(0, windows.shape[0], batch_size):
        ids = windows[i:i + batch_size].to(device)
        s_logits = student(ids).logits
        t_logits = teacher(ids).logits
        T = ids.shape[1]
        for s in range(0, T, row_chunk):
            sl = s_logits[:, s:s + row_chunk, :].float() / tau
            tl = t_logits[:, s:s + row_chunk, :].float() / tau
            p_t = F.softmax(tl, dim=-1)
            kl = (p_t * (F.log_softmax(tl, -1) - F.log_softmax(sl, -1))).sum(-1)
            total += float(kl.sum())
            count += kl.numel()
    return total / max(count, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="converted checkpoint dir")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--modes", default=None, help="comma list; default = profile")
    ap.add_argument("--teacher", default=None, help="HF id/dir for KL_tf (optional)")
    ap.add_argument("--calib-windows", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--backend", default="reference", choices=["reference", "metal"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="write report JSON here")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    profile = load_profile(args.profile)
    modes = (args.modes.split(",") if args.modes else profile.eval_modes)

    windows = calibration_windows(args.data_dir, args.calib_windows,
                                  split=args.split, seq_len=args.seq_len)
    model, _ = load_converted(args.ckpt, profile, backend=args.backend)
    model.to(device)

    report = {
        "ckpt": str(args.ckpt),
        "calib_windows": int(windows.shape[0]),
        "seq_len": int(windows.shape[1]),
        "meta": provenance.build_meta(profile_path=args.profile,
                                      model_config=model.config,
                                      data_manifest=load_manifest(args.data_dir)),
        "results": [],
    }
    for mode in modes:
        r = evaluate_ppl(model, windows, device, mode=mode)
        report["results"].append(r)
        print(f"[eval_ppl] mode={mode:7s} ce={r['ce']:.4f} ppl={r['ppl']:.2f}")

    if args.teacher:
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.bfloat16).to(device)
        set_eval_mode(model, modes[0])
        report["kl_tf"] = kl_tf(model, teacher, windows, device, tau=args.tau)
        print(f"[eval_ppl] KL_tf(mode={modes[0]}) = {report['kl_tf']:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
