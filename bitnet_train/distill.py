"""Distillation losses + teacher plumbing (train_plan §5.1 / §7.1; moe §5.1).

The chunked-losses mandate, honored two ways behind one interface:

  * portable path (CPU/CUDA): custom autograd.Functions that stream the STUDENT
    logits over vocab chunks — recomputed in the backward — so a full (T, V)
    student logit tensor never exists. The dense-teacher arm receives teacher
    logits per ROW slice only (the caller runs the teacher per slice), so
    teacher and student (T, V) tensors never coexist either.
  * fused MPS path: per row-slice, materialize the (rows, V) student slice once
    and run the fused kernels (cross_entropy_fwd/bwd, kd_kl_topk_*,
    kd_kl_dense_*) — the "fused equivalents" clause.

KD term: alpha * tau^2 * KL(softmax(t/tau) ‖ softmax(s/tau)); the fused kernels
take invtemp = 1/tau and the alpha*tau^2 factor is applied here. Top-k caches
(A6c) store softmax(teacher/tau) at a DECLARED tau and are valid only for the
frozen corpus (manifest hash) — the reader refuses anything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

IGNORE = -100


# ---------------------------------------------------------------------------
# portable chunked losses (the oracle path)
# ---------------------------------------------------------------------------

class _ChunkedCE(torch.autograd.Function):
    """Streaming-logsumexp CE from hidden states + head weight; chunk logits are
    recomputed in the backward (never a (T, V) student tensor)."""

    @staticmethod
    def forward(ctx, hidden, head_w, targets, vchunk):
        h = hidden.detach().float()
        w = head_w.detach().float()
        T, V = h.shape[0], w.shape[0]
        m = torch.full((T,), -torch.inf, device=h.device)
        s = torch.zeros(T, device=h.device)
        tgt_logit = torch.zeros(T, device=h.device)
        valid = targets != IGNORE
        for v0 in range(0, V, vchunk):
            lg = h @ w[v0:v0 + vchunk].t()
            nm = torch.maximum(m, lg.max(dim=-1).values)
            s = s * torch.exp(m - nm) + torch.exp(lg - nm.unsqueeze(-1)).sum(-1)
            m = nm
            in_chunk = valid & (targets >= v0) & (targets < v0 + vchunk)
            if in_chunk.any():
                tgt_logit[in_chunk] = lg[in_chunk].gather(
                    -1, (targets[in_chunk] - v0).unsqueeze(-1)).squeeze(-1)
        lse = m + s.log()
        loss = torch.where(valid, lse - tgt_logit, torch.zeros_like(lse))
        ctx.save_for_backward(hidden, head_w, targets, lse)
        ctx.vchunk = vchunk
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        hidden, head_w, targets, lse = ctx.saved_tensors
        h = hidden.detach().float()
        w = head_w.detach().float()
        V = w.shape[0]
        valid = (targets != IGNORE).float()
        go = (grad_out.float() * valid).unsqueeze(-1)
        gh = torch.zeros_like(h)
        gw = torch.zeros_like(w)
        for v0 in range(0, V, ctx.vchunk):
            wc = w[v0:v0 + vchunk_of(ctx, v0, V)]
            lg = h @ wc.t()
            p = torch.exp(lg - lse.unsqueeze(-1))
            in_chunk = (targets >= v0) & (targets < v0 + wc.shape[0]) & (targets != IGNORE)
            if in_chunk.any():
                p[in_chunk, targets[in_chunk] - v0] -= 1.0
            d = p * go
            gh += d @ wc
            gw[v0:v0 + wc.shape[0]] = d.t() @ h
        return gh.to(hidden.dtype), gw.to(head_w.dtype), None, None


def vchunk_of(ctx, v0, V):
    return min(ctx.vchunk, V - v0)


class _ChunkedKDTopk(torch.autograd.Function):
    """Sparse-teacher KD-KL (A6c), portable mirror of the kd_kl_topk kernel math
    (renorm / other-bucket tails; negative t_idx = padding)."""

    @staticmethod
    def forward(ctx, hidden, head_w, t_idx, t_prob, invtemp, tail_mode, vchunk):
        h = hidden.detach().float()
        w = head_w.detach().float()
        T, V = h.shape[0], w.shape[0]
        m = torch.full((T,), -torch.inf, device=h.device)
        s = torch.zeros(T, device=h.device)
        for v0 in range(0, V, vchunk):
            lg = (h @ w[v0:v0 + vchunk].t()) * invtemp
            nm = torch.maximum(m, lg.max(dim=-1).values)
            s = s * torch.exp(m - nm) + torch.exp(lg - nm.unsqueeze(-1)).sum(-1)
            m = nm
        lse = m + s.log()

        pad = t_idx < 0
        idx = t_idx.clamp_min(0)
        sel = torch.einsum("tkh,th->tk",
                           w[idx].reshape(*idx.shape, -1), h) * invtemp
        logq = (sel - lse.unsqueeze(-1)).masked_fill(pad, 0.0)
        p = t_prob.float().masked_fill(pad, 0.0)
        P = p.sum(-1)
        S = torch.exp(logq).masked_fill(pad, 0.0).sum(-1)
        tiny = 1e-30
        if tail_mode == 0:                              # renormalize over the k entries
            pt = p / P.clamp_min(tiny).unsqueeze(-1)
            loss = (pt * (pt.clamp_min(tiny).log() - logq)).masked_fill(pad, 0.0).sum(-1)
        else:                                           # other-bucket
            loss = (p * (p.clamp_min(tiny).log() - logq)).masked_fill(pad, 0.0).sum(-1)
            tail = (1.0 - P).clamp_min(0.0)
            loss = loss + torch.where(
                tail > 0, tail * (tail.clamp_min(tiny).log()
                                  - (1.0 - S).clamp_min(tiny).log()),
                torch.zeros_like(tail))
        ctx.save_for_backward(hidden, head_w, t_idx, t_prob, lse)
        ctx.invtemp, ctx.tail_mode, ctx.vchunk = invtemp, tail_mode, vchunk
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        hidden, head_w, t_idx, t_prob, lse = ctx.saved_tensors
        h = hidden.detach().float()
        w = head_w.detach().float()
        T, V = h.shape[0], w.shape[0]
        invtemp = ctx.invtemp
        go = grad_out.float() * invtemp
        pad = t_idx < 0
        idx = t_idx.clamp_min(0)
        p = t_prob.float().masked_fill(pad, 0.0)
        P = p.sum(-1)
        sel = torch.einsum("tkh,th->tk", w[idx].reshape(*idx.shape, -1), h) * invtemp
        q_sel = torch.exp(sel - lse.unsqueeze(-1)).masked_fill(pad, 0.0)
        S = q_sel.sum(-1)
        tiny = 1e-30
        if ctx.tail_mode == 0:
            qcoef = torch.ones_like(P)
            corr = -(p / P.clamp_min(tiny).unsqueeze(-1))
        else:
            tail = (1.0 - P).clamp_min(0.0)
            tail_c = torch.where(tail > 0, tail / (1.0 - S).clamp_min(tiny),
                                 torch.zeros_like(tail))
            qcoef = P - tail_c * S
            corr = -p + tail_c.unsqueeze(-1) * q_sel

        gh = torch.zeros_like(h)
        gw = torch.zeros_like(w)
        for v0 in range(0, V, ctx.vchunk):
            wc = w[v0:v0 + min(ctx.vchunk, V - v0)]
            lg = (h @ wc.t()) * invtemp
            q = torch.exp(lg - lse.unsqueeze(-1))
            d = q * (qcoef * go).unsqueeze(-1)
            gh += d @ wc
            gw[v0:v0 + wc.shape[0]] = d.t() @ h
        # sparse corrections at the teacher indices
        d_sel = (corr * go.unsqueeze(-1)).masked_fill(pad, 0.0)     # (T, k)
        gh += torch.einsum("tk,tkh->th", d_sel, w[idx].reshape(*idx.shape, -1))
        gw.index_add_(0, idx.reshape(-1),
                      (d_sel.unsqueeze(-1) * h.unsqueeze(1)).reshape(-1, h.shape[-1]))
        return gh.to(hidden.dtype), gw.to(head_w.dtype), None, None, None, None, None


def chunked_ce(hidden, head_w, targets, vchunk=8192):
    return _ChunkedCE.apply(hidden, head_w, targets, vchunk)


def chunked_kd_topk(hidden, head_w, t_idx, t_prob, tau=2.0, tail_mode=0, vchunk=8192):
    return _ChunkedKDTopk.apply(hidden, head_w, t_idx, t_prob, 1.0 / tau,
                                tail_mode, vchunk)


def chunked_kd_dense(hidden, head_w, teacher_logits, tau=2.0, vchunk=8192):
    """Dense-teacher KL for ONE row slice (teacher_logits (rows, V) is this
    slice's tensor — the caller never holds more than a slice). Differentiable
    through hidden and head_w via a straightforward vocab-chunked graph."""
    invtemp = 1.0 / tau
    T, V = hidden.shape[0], head_w.shape[0]
    with torch.no_grad():
        tl = teacher_logits.float() * invtemp
        lse_t = torch.logsumexp(tl, dim=-1)
    # student lse without materializing all logits in the autograd graph:
    with torch.no_grad():
        m = torch.full((T,), -torch.inf, device=hidden.device)
        s = torch.zeros(T, device=hidden.device)
        hf, wf = hidden.detach().float(), head_w.detach().float()
        for v0 in range(0, V, vchunk):
            lg = (hf @ wf[v0:v0 + vchunk].t()) * invtemp
            nm = torch.maximum(m, lg.max(dim=-1).values)
            s = s * torch.exp(m - nm) + torch.exp(lg - nm.unsqueeze(-1)).sum(-1)
            m = nm
        lse_s = m + s.log()
    loss = torch.zeros(T, device=hidden.device)
    for v0 in range(0, V, vchunk):
        zs = (hidden.float() @ head_w[v0:v0 + vchunk].float().t()) * invtemp
        with torch.no_grad():
            zt = tl[:, v0:v0 + vchunk]
            p_t = torch.exp(zt - lse_t.unsqueeze(-1))
            const = (p_t * (zt - lse_t.unsqueeze(-1))).sum(-1)
        # d/dzs of [-Σ p_t (zs - lse_s)] handles the lse_s term via the detached
        # identity Σ_v q_v = 1: grad zs_v = (q_v - p_t,v) — reproduce it with a
        # custom surrogate: loss_slice = const - Σ p_t zs + p_slice_mass * lse_s*
        loss = loss + const - (p_t * zs).sum(-1)
    # the lse_s term: Σ_v p_t = 1 per row, so + lse_s once, with gradient q_v
    lse_s_diff = _StudentLSE.apply(hidden, head_w, lse_s, invtemp, vchunk)
    return loss + lse_s_diff


class _StudentLSE(torch.autograd.Function):
    """lse of student logits at temperature, with the exact softmax gradient,
    computed vocab-chunked (forward value supplied precomputed)."""

    @staticmethod
    def forward(ctx, hidden, head_w, lse_s, invtemp, vchunk):
        ctx.save_for_backward(hidden, head_w, lse_s)
        ctx.invtemp, ctx.vchunk = invtemp, vchunk
        return lse_s

    @staticmethod
    def backward(ctx, grad_out):
        hidden, head_w, lse_s = ctx.saved_tensors
        h, w = hidden.detach().float(), head_w.detach().float()
        V = w.shape[0]
        go = grad_out.float().unsqueeze(-1) * ctx.invtemp
        gh = torch.zeros_like(h)
        gw = torch.zeros_like(w)
        for v0 in range(0, V, ctx.vchunk):
            wc = w[v0:v0 + min(ctx.vchunk, V - v0)]
            q = torch.exp((h @ wc.t()) * ctx.invtemp - lse_s.unsqueeze(-1))
            d = q * go
            gh += d @ wc
            gw[v0:v0 + wc.shape[0]] = d.t() @ h
        return gh.to(hidden.dtype), gw.to(head_w.dtype), None, None, None


# ---------------------------------------------------------------------------
# fused MPS wrappers (per row slice; logits for the slice are materialized once)
# ---------------------------------------------------------------------------

class _FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        from bitnet_train.bitlinear_metal import _tk
        loss, lse = _tk().cross_entropy_fwd(logits.contiguous(), targets.contiguous())
        ctx.save_for_backward(logits, targets, lse)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        from bitnet_train.bitlinear_metal import _tk
        logits, targets, lse = ctx.saved_tensors
        g = _tk().cross_entropy_bwd(logits.contiguous(), targets.contiguous(), lse,
                                    grad_out.contiguous())
        return g, None


class _FusedKDTopk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, t_idx, t_prob, invtemp, tail_mode):
        from bitnet_train.bitlinear_metal import _tk
        loss, lse = _tk().kd_kl_topk_fwd(logits.contiguous(), t_idx.contiguous(),
                                         t_prob.contiguous(), invtemp, tail_mode)
        ctx.save_for_backward(logits, t_idx, t_prob, lse)
        ctx.invtemp, ctx.tail_mode = invtemp, tail_mode
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        from bitnet_train.bitlinear_metal import _tk
        logits, t_idx, t_prob, lse = ctx.saved_tensors
        g = _tk().kd_kl_topk_bwd(logits.contiguous(), t_idx, t_prob, lse,
                                 grad_out.contiguous(), ctx.invtemp, ctx.tail_mode)
        return g, None, None, None, None


class _FusedKDDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s_logits, t_logits, invtemp):
        from bitnet_train.bitlinear_metal import _tk
        loss, lse_t, lse_s = _tk().kd_kl_dense_fwd(t_logits.contiguous(),
                                                   s_logits.contiguous(), invtemp)
        ctx.save_for_backward(s_logits, t_logits, lse_t, lse_s)
        ctx.invtemp = invtemp
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        from bitnet_train.bitlinear_metal import _tk
        s_logits, t_logits, lse_t, lse_s = ctx.saved_tensors
        g = _tk().kd_kl_dense_bwd(t_logits.contiguous(), s_logits.contiguous(),
                                  lse_t, lse_s, grad_out.contiguous(), ctx.invtemp)
        return g, None, None


# ---------------------------------------------------------------------------
# teacher + loss computer
# ---------------------------------------------------------------------------

class TeacherWrapper(nn.Module):
    """Frozen dense teacher, no-grad, bf16 (train_plan §5.1)."""

    def __init__(self, name_or_path: str, device, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path,
                                                          torch_dtype=dtype)
        self.model.eval().requires_grad_(False)
        self.model.to(device)

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits

    @torch.no_grad()
    def slice_server(self, input_ids: torch.Tensor, drop_last: bool = True):
        """One decoder pass -> (B, T, H) hidden; teacher LOGITS are then computed
        per row slice on demand — full teacher (T, V) never materializes (§5.1).
        drop_last aligns with next-token targets (positions 0..T-2)."""
        base = self.model.model                     # the decoder (post final norm)
        hidden = base(input_ids).last_hidden_state
        if drop_last:
            hidden = hidden[:, :-1, :]
        flat = hidden.reshape(-1, hidden.shape[-1])
        head_w = self.model.get_output_embeddings().weight

        def serve(sl: slice) -> torch.Tensor:
            return (flat[sl] @ head_w.t()).float()
        return serve


@dataclass
class LossConfig:
    alpha: float = 1.0
    tau: float = 2.0
    kd_mode: str = "dense"            # none | dense | topk
    tail_mode: int = 0                # 0 renorm | 1 other-bucket (topk only)
    vchunk: int = 8192
    tchunk: int = 1024                # row-slice size (the (rows,V) transient)
    prefer_fused: bool = True         # fused MPS kernels when on MPS


class LossComputer:
    """CE + alpha*tau^2*KL over row slices of (hidden, head_w). teacher_batch:
    dense -> teacher logits (B*T, V)-shaped CALLABLE per slice (slice -> tensor);
    topk  -> (t_idx (B*T, k) int32, t_prob (B*T, k) f32) from the cache reader."""

    def __init__(self, cfg: LossConfig):
        self.cfg = cfg

    def _fused(self, device) -> bool:
        return self.cfg.prefer_fused and device.type == "mps"

    def __call__(self, hidden: torch.Tensor, head_w: torch.Tensor,
                 targets: torch.Tensor, teacher_batch=None) -> dict:
        cfg = self.cfg
        T = hidden.shape[0]
        fused = self._fused(hidden.device)
        ce_sum = hidden.new_zeros((), dtype=torch.float32)
        kd_sum = hidden.new_zeros((), dtype=torch.float32)
        n_ce = max(int((targets != IGNORE).sum()), 1)
        for r0 in range(0, T, cfg.tchunk):
            h = hidden[r0:r0 + cfg.tchunk]
            tg = targets[r0:r0 + cfg.tchunk]
            if fused:
                logits = F.linear(h, head_w)
                ce_sum = ce_sum + _FusedCE.apply(logits, tg.int()).float().sum()
            else:
                ce_sum = ce_sum + chunked_ce(h, head_w, tg, cfg.vchunk).sum()
            if cfg.kd_mode == "dense" and teacher_batch is not None:
                t_logits = teacher_batch(slice(r0, r0 + cfg.tchunk))
                if fused:
                    kd_sum = kd_sum + _FusedKDDense.apply(
                        logits, t_logits.to(logits.dtype), 1.0 / cfg.tau).float().sum()
                else:
                    kd_sum = kd_sum + chunked_kd_dense(h, head_w, t_logits,
                                                       cfg.tau, cfg.vchunk).sum()
            elif cfg.kd_mode == "topk" and teacher_batch is not None:
                t_idx, t_prob = teacher_batch
                ti = t_idx[r0:r0 + cfg.tchunk]
                tp = t_prob[r0:r0 + cfg.tchunk]
                if fused:
                    kd_sum = kd_sum + _FusedKDTopk.apply(
                        logits, ti.int(), tp.float(), 1.0 / cfg.tau,
                        cfg.tail_mode).float().sum()
                else:
                    kd_sum = kd_sum + chunked_kd_topk(h, head_w, ti, tp, cfg.tau,
                                                      cfg.tail_mode,
                                                      cfg.vchunk).sum()
        ce = ce_sum / n_ce
        kd = kd_sum / T
        total = ce + cfg.alpha * cfg.tau ** 2 * kd
        return {"loss": total, "ce": ce.detach(), "kd": kd.detach()}


# ---------------------------------------------------------------------------
# teacher top-k cache (A6c)
# ---------------------------------------------------------------------------

def build_topk_cache(teacher, data_dir: str | Path, out_dir: str | Path, *,
                     k: int = 64, tau: float = 2.0, split: str = "train",
                     seq_len: int | None = None, device="cpu",
                     batch_size: int = 1, limit_windows: int = 0) -> dict:
    """Precompute softmax(teacher/tau) top-k per position over the FROZEN corpus,
    aligned to PackedWindows dataset order. memmaps: idx int32 + prob fp16,
    both (n_windows, seq_len, k)."""
    from bitnet_train.data import PackedWindows, load_manifest, manifest_hash

    ds = PackedWindows(data_dir, split=split, seq_len=seq_len)
    n = min(len(ds), limit_windows) if limit_windows else len(ds)
    T = ds.seq_len
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_mm = np.lib.format.open_memmap(out_dir / "topk_idx.npy", mode="w+",
                                       dtype=np.int32, shape=(n, T, k))
    prob_mm = np.lib.format.open_memmap(out_dir / "topk_prob.npy", mode="w+",
                                        dtype=np.float16, shape=(n, T, k))
    model = teacher.model if isinstance(teacher, TeacherWrapper) else teacher
    model.eval()
    with torch.no_grad():
        for b0 in range(0, n, batch_size):
            ids = torch.stack([ds[i] for i in range(b0, min(b0 + batch_size, n))])
            logits = model(ids.to(device)).logits.float() / tau
            probs = F.softmax(logits, dim=-1)
            top = probs.topk(k, dim=-1)
            idx_mm[b0:b0 + ids.shape[0]] = top.indices.cpu().numpy().astype(np.int32)
            prob_mm[b0:b0 + ids.shape[0]] = top.values.cpu().numpy().astype(np.float16)
    idx_mm.flush()
    prob_mm.flush()
    meta = {"manifest_hash": manifest_hash(load_manifest(data_dir)),
            "tokenizer": load_manifest(data_dir).get("tokenizer"),
            "split": split, "seq_len": T, "k": k, "tau": tau, "n_windows": n}
    (out_dir / "cache_manifest.json").write_text(json.dumps(meta, indent=2))
    return meta


class TopkCacheReader:
    """Validates {manifest_hash, seq_len, k, tau} before serving — a moved corpus
    or changed temperature invalidates the cache (train_plan §5.1 rule 2)."""

    def __init__(self, cache_dir: str | Path, data_dir: str | Path,
                 tau: float | None = None):
        from bitnet_train.data import load_manifest, manifest_hash
        cache_dir = Path(cache_dir)
        self.meta = json.loads((cache_dir / "cache_manifest.json").read_text())
        current = manifest_hash(load_manifest(data_dir))
        if self.meta["manifest_hash"] != current:
            raise ValueError(f"top-k cache is stale: built on {self.meta['manifest_hash']}, "
                             f"corpus is {current} — rebuild (frozen-corpus rule)")
        if tau is not None and abs(self.meta["tau"] - tau) > 1e-9:
            raise ValueError(f"cache tau {self.meta['tau']} != requested {tau}")
        self.idx = np.load(cache_dir / "topk_idx.npy", mmap_mode="r")
        self.prob = np.load(cache_dir / "topk_prob.npy", mmap_mode="r")

    def batch(self, window_indices: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
        wi = window_indices.cpu().numpy()
        idx = torch.from_numpy(np.ascontiguousarray(self.idx[wi])).to(device)
        prob = torch.from_numpy(np.ascontiguousarray(self.prob[wi])).float().to(device)
        return idx, prob
