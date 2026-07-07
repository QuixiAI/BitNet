"""T0.5 — layerwise reconstruction init (train_plan §11.3).

GPTQ/AdaRound/BRECQ-family: per decoder block, briefly optimize the ternary
LATENT weights (and optional per-channel input scales) to minimize the
fake-quant-vs-dense BLOCK-OUTPUT error on a small calibration set — a better
heal starting point than raw conversion. Run only if the damage map is
weight-dominant or T2 underdelivers (NOT baseline: real complexity, redundant
if the KD heal works on budget).

Method (block-local, teacher-forced): capture each dense block's (input, output)
on calibration windows; then for the CONVERTED block, minimize
||block_q(input) - dense_output||^2 over the block's BitLinear latents by SGD
through the STE (the fake-quant forward is differentiable to the latent). No
labels, no full-model backward — one block at a time, cheap and parallelizable.
The per-channel input scale is the A7 smoothing fold reused as a learnable
diagonal (function-preserving), off by default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

from bitnet_train.bitlinear import BitExperts, BitLinear, iter_bitlinears


def _block_list(model: nn.Module):
    out = []
    for name, mod in model.named_modules():
        if re.fullmatch(r"model\.layers\.\d+", name):
            out.append((name, mod))
    return out


@dataclass
class ReconReport:
    per_block: dict          # name -> {err_before, err_after, steps}

    @property
    def mean_reduction(self) -> float:
        rs = [1.0 - b["err_after"] / max(b["err_before"], 1e-12)
              for b in self.per_block.values()]
        return sum(rs) / max(len(rs), 1)


_KW = ("attention_mask", "position_ids", "position_embeddings")


@torch.no_grad()
def _capture_io(dense_model, windows, device, batch_size, block):
    """Teacher-forced (input hidden, block kwargs, dense output) for one dense
    block over the calibration windows — the DENSE model feeds it, so upstream
    blocks are exact (BRECQ's teacher-forced protocol). The block's own call
    kwargs (rotary position_embeddings, mask) are captured and replayed so the
    reconstructed block runs bit-identically minus the ternary weights."""
    ins, kwargs, outs = [], [], []

    def pre_hook(_m, args, kw):
        ins.append((args[0] if args else kw["hidden_states"]).detach().cpu())
        kwargs.append({k: kw[k] for k in _KW if k in kw})

    def out_hook(_m, _i, out):
        o = out[0] if isinstance(out, tuple) else out
        outs.append(o.detach().cpu())

    h1 = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = block.register_forward_hook(out_hook)
    for i in range(0, windows.shape[0], batch_size):
        dense_model(windows[i:i + batch_size].to(device))
    h1.remove()
    h2.remove()
    return ins, kwargs, outs


def reconstruct(dense_model: nn.Module, conv_model: nn.Module,
                windows: torch.Tensor, device, *, steps: int = 200, lr: float = 1e-3,
                batch_size: int = 1) -> ReconReport:
    """Optimize each converted block's ternary latents to match the dense block's
    output. dense_model and conv_model must be the SAME architecture (conv_model =
    convert(copy of dense)). conv_model is updated in place; returns the report."""
    dense_model.eval().to(device)
    conv_model.to(device)
    dense_blocks = _block_list(dense_model)
    conv_blocks = _block_list(conv_model)
    assert len(dense_blocks) == len(conv_blocks)
    report = {}

    def to_dev(v):
        if torch.is_tensor(v):
            return v.to(device)
        if isinstance(v, tuple):
            return tuple(to_dev(t) for t in v)
        return v

    for bi, (bname, cblock) in enumerate(conv_blocks):
        ins, kwargs, outs = _capture_io(dense_model, windows, device, batch_size,
                                        dense_blocks[bi][1])
        params = [m.weight for _, m in iter_bitlinears(cblock)]
        params += [p for m in cblock.modules() if isinstance(m, BitExperts)
                   for p in (m.gate_up_proj, m.down_proj)]
        if not params:
            continue
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=lr)

        def block_forward(x, kw):
            y = cblock(x, **{k: to_dev(v) for k, v in kw.items()})
            return y[0] if isinstance(y, tuple) else y

        cblock.train()
        err0 = err1 = None
        n = len(ins)
        for s in range(steps):
            j = s % n
            x = ins[j].to(device)
            tgt = outs[j].to(device)
            y = block_forward(x, kwargs[j])
            loss = torch.nn.functional.mse_loss(y.float(), tgt.float())
            if s == 0:
                err0 = float(loss.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            err1 = float(loss.detach())
        cblock.eval()
        for p in params:
            p.requires_grad_(True)                       # stay trainable for the heal
        report[bname] = {"err_before": err0, "err_after": err1, "steps": steps}
    return ReconReport(report)
