"""Profile-driven model conversion (train_plan §2.1 / §7.0 file #3).

Per-track behavior is data, not code: an ArchProfile (YAML) names which
nn.Linear modules become BitLinear and which stay FP; this module walks the
actual module tree and classifies EVERY nn.Linear — "enumerate, don't assume".
An unmatched or doubly-matched Linear is a hard error, which is exactly what
defeats the Qwen3 trap (`mlp.gate` router vs `mlp.experts.N.gate_proj`):
fullmatch means an unanchored 'gate' regex loudly over-matches instead of
silently ternarizing the router.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch import nn

from bitnet_train.bitlinear_metal import BitLinear

TERNARIZE, KEEP_FP = "ternarize", "keep_fp"


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

@dataclass
class ParamGroupSpec:
    name: str
    match: str                       # fullmatch regex over the PARAMETER name
    lr_scale: float = 1.0
    weight_decay: float = 0.0
    frozen: bool = False
    decay_masked: bool = False       # Q-track cold-expert masking (moe §4.3)


@dataclass
class ArchProfile:
    name: str
    base_model: str
    teacher: str
    target_linear_regexes: list[str]
    keep_fp_regexes: list[str]
    expert_stack_regexes: list[str] = field(default_factory=list)
    # ^ transformers-5 fused MoE experts (3-D gate_up_proj/down_proj) — swapped to
    #   BitExperts by module, since no per-expert nn.Linear exists to regex-match.
    freeze_fp_params: bool = False
    quant: dict = field(default_factory=lambda: {"granularity": "tensor", "group_k": 32})
    export_route: list[str] = field(default_factory=list)
    eval_modes: list[str] = field(default_factory=lambda: ["w_a8", "w_only"])
    data: dict = field(default_factory=dict)          # {tokenizer, shard_set}
    param_groups: list[ParamGroupSpec] | None = None  # None = A-track default
    decay_masking: bool = False
    aux_loss: bool = False                            # Q-track: output_router_logits
    fp8_exclude_regexes: list[str] = field(default_factory=lambda: [r".*\.mlp\.gate"])
    extras: dict = field(default_factory=dict)        # seq_len_schedule / lr_grid / budgets

    @property
    def granularity(self) -> str:
        return self.quant.get("granularity", "tensor")

    @property
    def group_k(self) -> int:
        return int(self.quant.get("group_k") or 32)


def load_profile(path: str | Path) -> ArchProfile:
    raw = yaml.safe_load(Path(path).read_text())
    groups = raw.pop("param_groups", None)
    if groups is not None:
        groups = [ParamGroupSpec(**g) for g in groups]
    known = {f for f in ArchProfile.__dataclass_fields__ if f != "extras"}
    extras = {k: raw.pop(k) for k in list(raw) if k not in known}
    prof = ArchProfile(param_groups=groups, extras=extras,
                       name=raw.pop("name", Path(path).stem), **raw)
    if prof.granularity not in ("tensor", "group"):
        raise ValueError(f"profile {prof.name}: bad granularity {prof.granularity!r}")
    return prof


def profile_hash(path: str | Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# classification + conversion
# ---------------------------------------------------------------------------

def classify_linears(model: nn.Module, profile: ArchProfile) -> dict[str, str]:
    """Every nn.Linear (that is not already a BitLinear) must fullmatch exactly
    one class. Unmatched or doubly-matched => hard error with the offending names."""
    tern = [re.compile(p) for p in profile.target_linear_regexes]
    keep = [re.compile(p) for p in profile.keep_fp_regexes]
    out, unmatched, double = {}, [], []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or isinstance(mod, BitLinear):
            continue
        is_t = any(p.fullmatch(name) for p in tern)
        is_k = any(p.fullmatch(name) for p in keep)
        if is_t and is_k:
            double.append(name)
        elif is_t:
            out[name] = TERNARIZE
        elif is_k:
            out[name] = KEEP_FP
        else:
            unmatched.append(name)
    if double or unmatched:
        raise ValueError(
            f"profile {profile.name}: enumerate-don't-assume violation.\n"
            f"  doubly-matched: {double[:8]}{'...' if len(double) > 8 else ''}\n"
            f"  unmatched:      {unmatched[:8]}{'...' if len(unmatched) > 8 else ''}")
    return out


@dataclass
class ConversionReport:
    profile: str
    n_ternarized: int
    n_kept_fp: int
    ternarized: list[str]
    kept_fp: list[str]
    family_counts: dict[str, int]
    ternary_param_fraction: float          # of ALL model parameters
    ternary_flop_fraction: float           # of per-token Linear FLOPs (incl. lm_head)
    n_expert_stacks: int = 0
    expert_stacks: list[str] = field(default_factory=list)

    def to_json(self, path: str | Path):
        Path(path).write_text(json.dumps(self.__dict__, indent=2))


def _family(name: str) -> str:
    """model.layers.5.mlp.experts.17.up_proj -> mlp.experts.*.up_proj"""
    parts = [("*" if p.isdigit() else p) for p in name.split(".")]
    fam = ".".join(parts)
    return fam.removeprefix("model.layers.*.")


def convert(model: nn.Module, profile: ArchProfile,
            backend: str = "reference") -> ConversionReport:
    """In-place swap of the profile's target linears for BitLinear (fp32 latent)
    and fused expert stacks for BitExperts. Never touches model.config (config
    preservation is a hard requirement)."""
    from bitnet_train.bitlinear import BitExperts

    classes = classify_linears(model, profile)
    tern_names = [n for n, c in classes.items() if c == TERNARIZE]
    kept = [n for n, c in classes.items() if c == KEEP_FP]

    tern_params = 0
    expert_pats = [re.compile(p) for p in profile.expert_stack_regexes]
    expert_names = [n for n, m in model.named_modules()
                    if any(p.fullmatch(n) for p in expert_pats)
                    and not isinstance(m, BitExperts)]
    for name in expert_names:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        attr = name.rsplit(".", 1)[-1]
        old = model.get_submodule(name)
        new = BitExperts(old, granularity=profile.granularity, group_k=profile.group_k)
        setattr(parent, attr, new)
        tern_params += new.gate_up_proj.numel() + new.down_proj.numel()

    for name in tern_names:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        attr = name.rsplit(".", 1)[-1]
        old: nn.Linear = model.get_submodule(name)
        if old.bias is not None:
            raise ValueError(f"{name}: BitLinear is bias-free but this Linear has bias")
        new = BitLinear(old.in_features, old.out_features, group_k=profile.group_k,
                        backend=backend, granularity=profile.granularity)
        with torch.no_grad():
            new.weight.copy_(old.weight.float())
        new.weight.requires_grad_(old.weight.requires_grad)
        setattr(parent, attr, new)
        tern_params += new.weight.numel()

    if profile.freeze_fp_params:
        tern_ids = {id(m.weight) for m in model.modules() if isinstance(m, BitLinear)}
        tern_ids |= {id(p) for m in model.modules() if isinstance(m, BitExperts)
                     for p in m.parameters()}
        for p in model.parameters():
            if id(p) not in tern_ids:
                p.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    lin_flops = tern_params + sum(
        m.weight.numel() for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and not isinstance(m, BitLinear))
    fam: dict[str, int] = {}
    for n in tern_names + expert_names:
        fam[_family(n)] = fam.get(_family(n), 0) + 1
    return ConversionReport(
        profile=profile.name, n_ternarized=len(tern_names), n_kept_fp=len(kept),
        n_expert_stacks=len(expert_names),
        ternarized=tern_names, kept_fp=kept, expert_stacks=expert_names,
        family_counts=fam,
        ternary_param_fraction=tern_params / max(total_params, 1),
        ternary_flop_fraction=tern_params / max(lin_flops, 1))


def load_converted(ckpt_dir: str | Path, profile: ArchProfile,
                   backend: str = "reference", dtype=torch.float32):
    """Load a converted checkpoint: BitLinear's state dict is key/shape-identical to
    a bias-free nn.Linear, so from_pretrained restores the latents into plain
    Linears and convert() re-wraps them (weights are copied into fp32 latents)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=dtype)
    report = convert(model, profile, backend=backend)
    return model, report


# ---------------------------------------------------------------------------
# optimizer param groups (train_plan §5.2 / moe_train_plan §5.2)
# ---------------------------------------------------------------------------

def build_param_groups(model: nn.Module, profile: ArchProfile, base_lr: float,
                       weight_decay: float = 0.1) -> list[dict]:
    """A-track default (profile.param_groups is None): wd on 2-D BitLinear latents
    only, wd 0 on everything else. Q-track: the profile's ordered ParamGroupSpec
    list, first-fullmatch wins; frozen groups get requires_grad_(False) and are
    excluded. Groups carry name/decay_masked/param_names for masking + tests."""
    from bitnet_train.bitlinear import BitExperts

    if profile.param_groups is None:
        bit_ids = {id(m.weight) for m in model.modules() if isinstance(m, BitLinear)}
        bit_ids |= {id(p) for m in model.modules() if isinstance(m, BitExperts)
                    for p in m.parameters()}
        latents = [p for p in model.parameters() if id(p) in bit_ids and p.requires_grad]
        rest = [p for p in model.parameters()
                if id(p) not in bit_ids and p.requires_grad]
        return [
            {"params": latents, "lr": base_lr, "weight_decay": weight_decay,
             "name": "bitlinear_latents", "decay_masked": False},
            {"params": rest, "lr": base_lr, "weight_decay": 0.0,
             "name": "other", "decay_masked": False},
        ]

    specs = [(re.compile(g.match), g) for g in profile.param_groups]
    buckets: dict[str, list] = {g.name: [] for g in profile.param_groups}
    names: dict[str, list[str]] = {g.name: [] for g in profile.param_groups}
    for pname, p in model.named_parameters():
        for pat, g in specs:
            if pat.fullmatch(pname):
                if g.frozen:
                    p.requires_grad_(False)
                else:
                    buckets[g.name].append(p)
                    names[g.name].append(pname)
                break
        else:
            raise ValueError(f"profile {profile.name}: parameter {pname!r} matched "
                             "no param_group (add a catch-all)")
    groups = []
    for g in profile.param_groups:
        if g.frozen or not buckets[g.name]:
            continue
        groups.append({"params": buckets[g.name], "lr": base_lr * g.lr_scale,
                       "weight_decay": g.weight_decay, "name": g.name,
                       "decay_masked": g.decay_masked,
                       "param_names": names[g.name]})
    return groups


# ---------------------------------------------------------------------------
# config preservation (train_plan §2.2: rope_scaling above all)
# ---------------------------------------------------------------------------

def diff_config(cfg_a, cfg_b, ignore: tuple[str, ...] = ("_name_or_path",
                                                         "transformers_version",
                                                         "torch_dtype", "dtype")) -> dict:
    """Field-by-field diff of two HF configs (as dicts). Returns {} when clean."""
    da = cfg_a.to_dict() if hasattr(cfg_a, "to_dict") else dict(cfg_a)
    db = cfg_b.to_dict() if hasattr(cfg_b, "to_dict") else dict(cfg_b)
    diffs = {}
    for k in sorted(set(da) | set(db)):
        if any(fnmatch.fnmatch(k, pat) for pat in ignore):
            continue
        if da.get(k, "<missing>") != db.get(k, "<missing>"):
            diffs[k] = {"a": da.get(k, "<missing>"), "b": db.get(k, "<missing>")}
    return diffs
