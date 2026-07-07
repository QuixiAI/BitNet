"""The cold-expert decay-mask test (moe_train_plan §5.6 — REQUIRED before any
nonzero expert weight decay; until this is green the trainer forces expert
wd = 0):

    route zero tokens to expert j -> optimizer.step()
      -> assert expert j's latents unchanged (modulo allowed global bookkeeping)
    route tokens to expert j -> optimizer.step()
      -> assert decay applied exactly now

Run against the REAL optimizer wrapping the trainer builds (build_param_groups
-> AdamW with masked groups at wd 0 -> ColdExpertDecayMasker + RouterHooks),
not a toy AdamW — zero-vs-absent gradients are exactly where a mask silently
breaks. (FSDP flat-param wrapping gets its own variant when the multi-GPU
config lands; this is the single-process wrapping the smoke runs use.)
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("transformers")

from bitnet_train.bitlinear import iter_bitexperts  # noqa: E402
from bitnet_train.conversion import build_param_groups, convert, load_profile  # noqa: E402
from bitnet_train.moe_metrics import RouterHooks  # noqa: E402
from bitnet_train.optim import ColdExpertDecayMasker  # noqa: E402

PROFILES = Path(__file__).resolve().parents[1] / "train" / "profiles"
LR, WD = 1e-2, 0.1


@pytest.fixture()
def rig():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        num_experts=8, num_experts_per_tok=2, vocab_size=512, mlp_only_layers=[],
        tie_word_embeddings=True, head_dim=32))
    prof = load_profile(PROFILES / "ci_tiny_moe.yaml")
    convert(model, prof, backend="reference")
    groups = build_param_groups(model, prof, LR, weight_decay=WD)
    for g in groups:
        if g.get("decay_masked"):
            g["weight_decay"] = 0.0            # exactly what train.py does (§5.2)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    masker = ColdExpertDecayMasker(model, opt, intended_wd=WD)
    experts = next(m for _, m in iter_bitexperts(model))
    return model, opt, masker, experts


def _zero_grads(opt):
    for g in opt.param_groups:
        for p in g["params"]:
            p.grad = torch.zeros_like(p)


def test_unrouted_expert_untouched_then_decayed_exactly(rig):
    model, opt, masker, experts = rig
    j = 3
    before = experts.gate_up_proj.detach().clone()

    # phase 1: zero tokens routed anywhere; a step must not move ANY latent
    _zero_grads(opt)
    opt.step()
    masker.step({})                                       # nothing routed
    torch.testing.assert_close(experts.gate_up_proj.detach(), before, rtol=0, atol=0)

    # phase 2: expert j routed (grads still zero -> the ONLY change is the decay)
    _zero_grads(opt)
    opt.step()
    masker.step({id(experts): [j]})
    after = experts.gate_up_proj.detach()
    torch.testing.assert_close(after[j], before[j] * (1.0 - LR * WD), rtol=0, atol=0)
    others = [e for e in range(experts.num_experts) if e != j]
    torch.testing.assert_close(after[others], before[others], rtol=0, atol=0)


def test_real_wrapping_backward_routes_and_masks(rig):
    """A real forward/backward: RouterHooks captures the routed union; after the
    step, unrouted experts' slices are exactly untouched (zero grad + masked
    decay), routed ones moved."""
    model, opt, masker, experts = rig
    hooks = RouterHooks(model).attach()
    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():                                 # experts 6/7 can never win
        model.model.layers[0].mlp.gate.weight[6:].fill_(-100.0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if "experts" in n}

    out = model(ids, labels=ids)
    out.loss.backward()
    routed = hooks.routed_and_reset()
    routed_ids = set(routed[id(experts)])
    assert routed_ids and len(routed_ids) < experts.num_experts, \
        "test needs a partial routing pattern; enlarge batch if this fires"
    opt.step()
    masker.step(routed)
    hooks.detach()

    gu = experts.gate_up_proj.detach()
    for e in range(experts.num_experts):
        if e in routed_ids:
            assert not torch.equal(gu[e], before["model.layers.0.mlp.experts.gate_up_proj"][e]), \
                f"routed expert {e} did not move"
        else:
            torch.testing.assert_close(
                gu[e], before["model.layers.0.mlp.experts.gate_up_proj"][e],
                rtol=0, atol=0)


def test_masker_rejects_nonzero_group_wd(rig):
    """The §5.2 ordering guard: a masked group carrying optimizer-level decay is
    a configuration bug (it would double-decay hot experts and erode cold ones)."""
    model, opt, masker, experts = rig
    opt.param_groups[0]["weight_decay"] = 0.05            # sabotage
    with pytest.raises(ValueError, match="weight_decay=0"):
        ColdExpertDecayMasker(model, opt, intended_wd=WD)
