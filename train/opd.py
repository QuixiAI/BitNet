#!/usr/bin/env python
"""OPD driver: T3.4 measurement gate, then T3.5 polish (train_plan §6.2, §11.5).

  --measure  T3.4: sample rollouts from the healed student, score with the
             teacher, compute KL_op / entropy / length-repetition-EOS stats,
             compare to KL_tf. ZERO weight updates — OPD earns its compute or
             doesn't run (large gap -> proceed; small -> shorten/skip).
  (default)  T3.5: the §6.3 GKD loop (E3 support-set reverse KL, λ mixed
             batches), exit criterion checked by the caller (gap -> floor AND
             generation evals up at fixed KL_tf).

HARD RULE (§6.2): this file must not run before A1 T2. It is written now per an
explicit build-everything override; the ecosystem re-check at T3.4 kickoff still
applies (this is the pinned §6.3 spec, not the last word).

  python train/opd.py --measure --ckpt <healed> --profile train/profiles/a1.yaml \
      --prompts <file> --teacher <hf-id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bitnet_train.bitlinear import set_eval_mode  # noqa: E402
from bitnet_train.conversion import load_converted, load_profile  # noqa: E402
from bitnet_train.distill import TeacherWrapper  # noqa: E402
from bitnet_train.opd import rollout_metrics  # noqa: E402
from bitnet_train.opd.gkd_loop import (  # noqa: E402
    OPDConfig, _teacher_logprob_fn, opd_step, sample_rollouts)


def load_prompts(path: str, tokenizer, n: int, device) -> list[torch.Tensor]:
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()][:n]
    return [tokenizer(l, return_tensors="pt").input_ids.to(device) for l in lines]


@torch.no_grad()
def measure(student, teacher, prompts, cfg: OPDConfig, device) -> dict:
    """T3.4: the decision packet, zero updates. KL_op via the E3 estimator on
    student rollouts; KL_tf on the same prompts teacher-forced."""
    import torch.nn.functional as F

    from bitnet_train.opd import estimators

    kl_ops, kl_tfs, ents, tok_lists, lengths = [], [], [], [], []
    for p in prompts:
        ids, gen_mask = sample_rollouts(student, p, cfg)
        out = student(ids, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, :-1, :].reshape(-1, out.hidden_states[-1].shape[-1])
        head_w = student.get_output_embeddings().weight
        mask = gen_mask[:, 1:].reshape(-1)
        tfn, teacher_logp = _teacher_logprob_fn(teacher, ids, cfg.temperature)
        teacher_topk = teacher_logp.topk(cfg.teacher_k, dim=-1).indices
        sampled = ids[:, 1:].reshape(-1)
        support = estimators.build_e3_support(hidden, head_w, teacher_topk, sampled,
                                              student_k=cfg.student_k)
        kl = estimators.support_reverse_kl(
            hidden[mask], head_w, support[mask],
            lambda idx: teacher_logp[mask].gather(1, idx), tail_mode=cfg.tail_mode)
        kl_ops.append(float(kl))

        # KL_tf on the same prompt teacher-forced (student CE-free forward KL gauge)
        s_logits = student(p).logits.float()
        t_logits = teacher.model(p).logits.float()
        p_t = F.softmax(t_logits, -1)
        kl_tf = (p_t * (F.log_softmax(t_logits, -1) - F.log_softmax(s_logits, -1))).sum(-1)
        kl_tfs.append(float(kl_tf.mean()))

        gen = ids[0, p.shape[1]:].tolist()
        tok_lists.append(gen)
        lengths.append(len(gen))
        s_lp = F.log_softmax(student(ids).logits[0, p.shape[1] - 1:-1].float(), -1)
        ents.append(rollout_metrics.rollout_entropy(s_lp))

    kl_op = sum(kl_ops) / len(kl_ops)
    kl_tf = sum(kl_tfs) / len(kl_tfs)
    ent = sum(ents) / len(ents)
    return rollout_metrics.summarize(kl_op, kl_tf, ent, tok_lists, lengths,
                                     cfg.max_new_tokens)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--prompts", required=True, help="one prompt per line")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--measure", action="store_true", help="T3.4 gate (no updates)")
    ap.add_argument("--estimator", default="e3", choices=["e0", "e2", "e3"])
    ap.add_argument("--tail-mode", default="renorm", choices=["renorm", "other"])
    ap.add_argument("--lam", type=float, default=1.0, help="on-policy batch fraction")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gap-threshold", type=float, default=0.05,
                    help="T3.4: proceed to T3.5 only if on-policy gap exceeds this")
    ap.add_argument("--backend", default="reference")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    profile = load_profile(args.profile)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    student, _ = load_converted(args.ckpt, profile, backend=args.backend)
    student.to(device)
    set_eval_mode(student, profile.eval_modes[0])       # rollout policy = training policy
    teacher = TeacherWrapper(args.teacher or profile.teacher, device)
    prompts = load_prompts(args.prompts, tok, args.n_prompts, device)
    cfg = OPDConfig(estimator=args.estimator, tail_mode=args.tail_mode, lam=args.lam,
                    max_new_tokens=args.max_new_tokens)

    if args.measure:
        report = measure(student, teacher, prompts, cfg, device)
        report["decision"] = ("proceed" if report["on_policy_gap"] > args.gap_threshold
                              else "skip_or_shorten")
        print(f"[opd/T3.4] KL_op={report['kl_op']:.4f} KL_tf={report['kl_tf']:.4f} "
              f"gap={report['on_policy_gap']:.4f} -> {report['decision']}")
        print(f"[opd/T3.4] entropy={report['entropy']:.3f} distinct-2={report['distinct_2']:.3f} "
              f"rep-3={report['repetition_3']:.3f} early-EOS={report['early_eos_rate']:.3f}")
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2))
        return 0

    # T3.5 polish
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    student.train()
    for step in range(args.steps):
        p = prompts[step % len(prompts)]
        on_policy = (torch.rand(()) < cfg.lam).item()
        loss, m = opd_step(student, teacher, p, cfg, on_policy=on_policy)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"[opd/T3.5] step {step} loss {m['opd_loss']:.4f} "
                  f"{'on' if on_policy else 'tf'}-policy tok={m['gen_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
