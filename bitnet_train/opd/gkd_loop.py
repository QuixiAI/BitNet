"""GKD-style on-policy distillation loop (train_plan §6.3), self-contained —
NO external TRL/verl dependency (the repo takes no external deps; TRL's GKD is
the reference SPEC, not an import). The plan's §6.4 names TRL as the minimal-code
path, but its GKD trainer materializes full logits (disqualifying at 262K vocab,
§6.4) and would need subclassing for the E3 chunked support-set loss anyway — so
we implement the pinned §6.3 algorithm directly.

Given prompt x from the frozen prompt set:
  sample y ~ student(·|x), temperature 1.0, up to L tokens        (rollout policy
    = training policy: sampling goes through the fake-quant forward, §6.3)
  at every position t: per-token loss = D_KL(p_S ‖ p_T) at (x, y<t)  [E3 support]
  optionally mix fraction (1-λ) teacher-forced batches (GKD's λ, O3)
No reward model, no discounting, no reference-policy KL.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from bitnet_train.opd import estimators


@dataclass
class OPDConfig:
    estimator: str = "e3"        # e0 | e2 | e3 (baseline e3; e2 for the O5 bias ablation)
    tail_mode: str = "renorm"    # renorm | other
    student_k: int = 8
    teacher_k: int = 16
    lam: float = 1.0             # fraction of ON-policy batches (GKD λ; O3 sweep)
    max_new_tokens: int = 64
    temperature: float = 1.0
    vchunk: int = 8192


@torch.no_grad()
def sample_rollouts(student, prompt_ids, cfg: OPDConfig):
    """Generate y ~ student through the fake-quant forward (BitLinear is stateless
    so KV-cache generate works unmodified, §6.3). Returns (full_ids, gen_mask)
    where gen_mask marks the generated positions (loss applies there only)."""
    out = student.generate(prompt_ids, do_sample=True,
                           temperature=cfg.temperature, top_k=0, top_p=1.0,
                           max_new_tokens=cfg.max_new_tokens,
                           pad_token_id=getattr(student.config, "eos_token_id", 0))
    gen_mask = torch.zeros_like(out, dtype=torch.bool)
    gen_mask[:, prompt_ids.shape[1]:] = True
    return out, gen_mask


def _teacher_logprob_fn(teacher, ids, tau):
    """Returns callable idx(T,K)->(T,K) teacher logprobs at the visited states,
    computed once per rollout (teacher scores, never generates, §6.7). Full-vocab
    log-softmax is done teacher-side in one pass; the gather is cheap."""
    with torch.no_grad():
        logits = teacher.model(ids).logits[:, :-1, :].float() / tau
        logp = F.log_softmax(logits, dim=-1).reshape(-1, logits.shape[-1])
    T, V = logp.shape

    def fn(idx):
        return logp.gather(1, idx.reshape(T, -1).clamp(0, V - 1))
    return fn, logp


def opd_step(student, teacher, prompt_ids, cfg: OPDConfig, *, on_policy=True):
    """One OPD micro-step -> (loss, metrics dict). on_policy=False is the GKD
    teacher-forced mixed batch (uses prompt_ids directly as the sequence)."""
    if on_policy:
        ids, gen_mask = sample_rollouts(student, prompt_ids, cfg)
    else:
        ids = prompt_ids
        gen_mask = torch.ones_like(ids, dtype=torch.bool)
        gen_mask[:, 0] = False

    out = student(ids, output_hidden_states=True)
    hidden = out.hidden_states[-1][:, :-1, :]
    H = hidden.shape[-1]
    flat_h = hidden.reshape(-1, H)
    head_w = student.get_output_embeddings().weight
    mask = gen_mask[:, 1:].reshape(-1)

    tfn, teacher_logp = _teacher_logprob_fn(teacher, ids, cfg.temperature)

    with torch.no_grad():
        teacher_topk = teacher_logp.topk(cfg.teacher_k, dim=-1).indices
        sampled = ids[:, 1:].reshape(-1)

    if cfg.estimator == "e0":
        def tfull(idx):
            return teacher_logp.index_select(1, idx)
        loss_all = estimators.full_reverse_kl(flat_h[mask], head_w,
                                              lambda idx: teacher_logp[mask][:, idx],
                                              vchunk=cfg.vchunk)
        loss = loss_all
    else:
        if cfg.estimator == "e3":
            support = estimators.build_e3_support(
                flat_h, head_w, teacher_topk, sampled,
                student_k=cfg.student_k, vchunk=cfg.vchunk)
        elif cfg.estimator == "e2":
            support = teacher_topk
        else:
            raise ValueError(f"unknown estimator {cfg.estimator!r}")
        sup_m = support[mask]
        loss = estimators.support_reverse_kl(
            flat_h[mask], head_w, sup_m,
            lambda idx: teacher_logp[mask].gather(1, idx),
            tail_mode=cfg.tail_mode, vchunk=cfg.vchunk)

    metrics = {"opd_loss": float(loss.detach()), "gen_tokens": int(mask.sum()),
               "on_policy": on_policy}
    return loss, metrics
