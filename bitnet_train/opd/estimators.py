"""On-policy-distillation reverse-KL estimators (train_plan §6.3.1, E0-E4).

OPD minimizes per-token REVERSE KL on the STUDENT's state distribution:
  D_KL(p_S ‖ p_T) = Σ_v p_S(v) · (log p_S(v) - log p_T(v))
at states y<t the student itself generated. The estimator variants differ in
which support they sum over — and the plan's headline result (§6.3.1) is that a
teacher-top-k-only support (E2) is BIASED: it misses exactly the tokens where
the student puts mass and the teacher doesn't (the error tokens OPD exists to
correct). So E3 (student-top-k ∪ teacher-top-k ∪ sampled token) is the baseline.

All estimators are chunked-safe: they take student HIDDEN states + head weight
and a TEACHER logprob oracle (a callable returning logprobs at requested vocab
indices), so a full (T, V) student or teacher logit tensor is never
materialized (§6.3's inherited chunked mandate). Reverse KL is directly
differentiable at visited states (E2/E3 are NOT score-function estimators —
that's E1); the gradient flows through p_S into the student.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _student_logprobs_full(hidden, head_w, vchunk):
    """Streaming log-softmax pieces: returns (lse (T,), a closure giving logprob
    columns for a set of vocab indices) — never a (T, V) tensor at once."""
    h = hidden
    V = head_w.shape[0]
    with torch.no_grad():
        m = torch.full((h.shape[0],), -torch.inf, device=h.device)
        s = torch.zeros(h.shape[0], device=h.device)
        for v0 in range(0, V, vchunk):
            lg = h.float() @ head_w[v0:v0 + vchunk].float().t()
            nm = torch.maximum(m, lg.max(-1).values)
            s = s * torch.exp(m - nm) + torch.exp(lg - nm.unsqueeze(-1)).sum(-1)
            m = nm
        lse = m + s.log()
    return lse


def full_reverse_kl(hidden, head_w, teacher_logprob_full, vchunk=8192):
    """E0 — exact full-vocab reverse KL (small-vocab unit-test / oracle only;
    memory-heavy). teacher_logprob_full(idx) -> (T, |idx|) teacher logprobs.
    Differentiable through the student (lse is a detached constant, exact by
    Σ_v p_S = 1)."""
    T, V = hidden.shape[0], head_w.shape[0]
    lse = _student_logprobs_full(hidden, head_w, vchunk)
    loss = hidden.new_zeros(())
    for v0 in range(0, V, vchunk):
        idx = torch.arange(v0, min(v0 + vchunk, V), device=hidden.device)
        logp_s = (hidden.float() @ head_w[v0:v0 + vchunk].float().t()) - lse.unsqueeze(-1)
        p_s = logp_s.exp()
        logp_t = teacher_logprob_full(idx)
        loss = loss + (p_s * (logp_s - logp_t)).sum()
    return loss / T


def support_reverse_kl(hidden, head_w, support_idx, teacher_logprob, *,
                       tail_mode: str = "renorm", vchunk: int = 8192):
    """E3 baseline — truncated reverse KL over an explicit per-token support set
    S (union of student-top-k, teacher-top-k, the sampled token). Differentiable
    through the student. teacher_logprob(support_idx) -> (T, K) teacher logprobs
    at those indices (an "other" bucket handles the tail per tail_mode).

    tail_mode:
      'renorm'  — renormalize p_S over S (the tail is dropped; simplest, biased
                  toward the support but the support already contains the
                  student's own mass by construction of S).
      'other'   — an explicit tail outcome carrying (1 - Σ_S p_S) student mass
                  and (1 - Σ_S p_T) teacher mass; the config-recorded choice that
                  'changes gradients' (mirrors kd_kl_topk's other-bucket).
    support_idx: (T, K) long. Returns scalar mean reverse KL.
    """
    T = hidden.shape[0]
    lse = _student_logprobs_full(hidden, head_w, vchunk)         # detached normalizer
    # student logprobs at the support (differentiable via the gathered logits)
    w_s = head_w[support_idx]                                     # (T, K, H)
    z_s = torch.einsum("tkh,th->tk", w_s.float(), hidden.float())
    logp_s = z_s - lse.unsqueeze(-1)
    p_s = logp_s.exp()
    logp_t = teacher_logprob(support_idx).to(hidden.dtype).float()

    if tail_mode == "renorm":
        Z = p_s.sum(-1, keepdim=True).clamp_min(1e-20)
        pr = p_s / Z
        loss = (pr * (pr.clamp_min(1e-30).log() - logp_t)).sum(-1)
    elif tail_mode == "other":
        S_s = p_s.sum(-1).clamp(max=1.0)
        tail_s = (1.0 - S_s).clamp_min(0.0)
        S_t = logp_t.exp().sum(-1).clamp(max=1.0)
        tail_t = (1.0 - S_t).clamp_min(1e-20)
        loss = (p_s * (logp_s - logp_t)).sum(-1)
        loss = loss + tail_s * (tail_s.clamp_min(1e-30).log() - tail_t.log())
    else:
        raise ValueError(f"tail_mode must be 'renorm' or 'other', got {tail_mode!r}")
    return loss.mean()


def teacher_topk_reverse_kl(hidden, head_w, teacher_topk_idx, teacher_logprob, *,
                            tail_mode="renorm", vchunk=8192):
    """E2 — reverse KL supported ONLY on the teacher's top-k. Provided so the
    §6.3.1 support-bias can be MEASURED (ablation O5), NOT as a baseline: it
    misses tokens where the student errs. Same signature/return as E3."""
    return support_reverse_kl(hidden, head_w, teacher_topk_idx, teacher_logprob,
                              tail_mode=tail_mode, vchunk=vchunk)


def build_e3_support(hidden, head_w, teacher_topk_idx, sampled_idx, *,
                     student_k: int = 8, vchunk: int = 8192) -> torch.Tensor:
    """The E3 support set per token: student-top-k ∪ teacher-top-k ∪ sampled
    token, deduplicated and padded to a fixed width (repeats are harmless — the
    renorm/other math is set-based). Returns (T, K_union) long.

    The student's full distribution is materialized locally in-process (chunked
    argmax), so E3 costs merely gathering teacher logprobs at the union too
    (§6.3.1: 'E3 is nearly free')."""
    T, V = hidden.shape[0], head_w.shape[0]
    with torch.no_grad():
        top_vals = torch.full((T, student_k), -torch.inf, device=hidden.device)
        top_idx = torch.zeros((T, student_k), dtype=torch.long, device=hidden.device)
        for v0 in range(0, V, vchunk):
            zs = hidden.float() @ head_w[v0:v0 + vchunk].float().t()
            cat_v = torch.cat([top_vals, zs], dim=1)
            cat_i = torch.cat([top_idx,
                               torch.arange(v0, min(v0 + vchunk, V),
                                            device=hidden.device).expand(T, -1)], dim=1)
            top_vals, sel = cat_v.topk(student_k, dim=1)
            top_idx = cat_i.gather(1, sel)
        parts = [top_idx, teacher_topk_idx.to(hidden.device),
                 sampled_idx.reshape(T, 1).to(hidden.device)]
        union = torch.cat(parts, dim=1)
        # dedup per row by sorting; keep width fixed (repeats fold out in the math)
        return union
