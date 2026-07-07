"""On-policy rollout metrics (train_plan §10.3, §6.2): the T3.4 measurement
gate's inputs and the T3.5 collapse monitors. All computed on STUDENT rollouts
scored by the teacher.

  KL_op          = E_{x~student rollouts}[ D_KL(p_S ‖ p_T) ]   (E3 estimator)
  on-policy gap  = KL_op - KL_tf   (exposure bias, quantified)
Collapse channels: rollout entropy, distinct-n / repetition, early-EOS rate,
length distribution vs teacher, win-rate vs a reference checkpoint.
"""

from __future__ import annotations

from collections import Counter

import torch


def rollout_entropy(logprobs: torch.Tensor) -> float:
    """Mean per-token entropy of the student's sampling distribution over a
    rollout batch. logprobs: (N, V) or a list of them."""
    p = logprobs.exp()
    return float(-(p * logprobs).sum(-1).mean())


def distinct_n(token_lists: list[list[int]], n: int = 2) -> float:
    """distinct-n: unique n-grams / total n-grams across rollouts (repetition
    canary; low = degenerate looping)."""
    total, uniq = 0, set()
    for toks in token_lists:
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / max(total, 1)


def repetition_rate(token_lists: list[list[int]], n: int = 3) -> float:
    """Fraction of n-grams that repeat within a sequence (the mode-seeking
    reverse-KL failure)."""
    rep, total = 0, 0
    for toks in token_lists:
        c = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
        total += sum(c.values())
        rep += sum(v - 1 for v in c.values() if v > 1)
    return rep / max(total, 1)


def early_eos_rate(lengths: list[int], max_len: int, frac: float = 0.25) -> float:
    """Fraction of rollouts that stopped before frac*max_len (collapse to short
    generations)."""
    thr = frac * max_len
    return sum(l < thr for l in lengths) / max(len(lengths), 1)


def length_stats(lengths: list[int]) -> dict:
    t = torch.tensor(lengths, dtype=torch.float)
    return {"len_mean": float(t.mean()), "len_std": float(t.std(unbiased=False)),
            "len_min": int(t.min()), "len_max": int(t.max())}


def on_policy_gap(kl_op: float, kl_tf: float) -> float:
    """Exposure bias, quantified (§10.3). Large gap -> T3.5 proceeds (§6.2)."""
    return kl_op - kl_tf


def summarize(kl_op, kl_tf, entropy, token_lists, lengths, max_len) -> dict:
    """The T3.4 decision packet + the T3.5 collapse channels in one dict."""
    return {
        "kl_op": kl_op,
        "kl_tf": kl_tf,
        "on_policy_gap": on_policy_gap(kl_op, kl_tf),
        "entropy": entropy,
        "distinct_2": distinct_n(token_lists, 2),
        "distinct_3": distinct_n(token_lists, 3),
        "repetition_3": repetition_rate(token_lists, 3),
        "early_eos_rate": early_eos_rate(lengths, max_len),
        **length_stats(lengths),
    }
