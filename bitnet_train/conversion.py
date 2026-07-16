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

    @property
    def quant_scheme(self) -> str:
        return str(self.quant.get("scheme", "scalar_ternary"))


def load_profile(path: str | Path) -> ArchProfile:
    raw = yaml.safe_load(Path(path).read_text())
    groups = raw.pop("param_groups", None)
    if groups is not None:
        groups = [ParamGroupSpec(**g) for g in groups]
    known = {f for f in ArchProfile.__dataclass_fields__ if f != "extras"}
    extras = {k: raw.pop(k) for k in list(raw) if k not in known}
    prof = ArchProfile(param_groups=groups, extras=extras,
                       name=raw.pop("name", Path(path).stem), **raw)
    if prof.quant_scheme == "scalar_ternary":
        allowed = {"scheme", "granularity", "group_k"}
        unknown = set(prof.quant) - allowed
        if unknown:
            raise ValueError(f"profile {prof.name}: unknown scalar quant keys {sorted(unknown)}")
        if prof.granularity not in ("tensor", "group"):
            raise ValueError(f"profile {prof.name}: bad granularity {prof.granularity!r}")
    elif prof.quant_scheme == "tq1_v":
        allowed = {
            "scheme", "spec", "artifact", "default_profile", "default_codebook_id",
            "activation_mode", "importance_stats", "qat_projection", "candidate_count",
            "top_m", "temperature_start", "temperature_end", "soft_tokens", "hard_tokens",
            "freeze_eval_every_tokens", "freeze_indices_at_tokens", "freeze_max_tokens",
            "margin", "lambda_margin", "tensor_overrides",
            "assignment_chunk", "lambda_hidden", "hidden_layers",
            "freeze_flip_threshold", "freeze_margin_threshold", "freeze_sustain_evals",
            "freeze_trend_tolerance",
            "shared_embedding_head", "shared_head_importance",
            "shared_embedding_importance",
        }
        unknown = set(prof.quant) - allowed
        if unknown:
            raise ValueError(f"profile {prof.name}: unknown TQ1 quant keys {sorted(unknown)}")
        required = {
            "artifact", "default_profile", "default_codebook_id", "qat_projection",
            "soft_tokens", "hard_tokens", "freeze_eval_every_tokens",
        }
        missing = required - set(prof.quant)
        if missing:
            raise ValueError(f"profile {prof.name}: missing TQ1 quant keys {sorted(missing)}")
        if prof.quant["qat_projection"] not in {"soft", "hard", "frozen"}:
            raise ValueError("training profile qat_projection must be soft, hard, or frozen")
    else:
        raise ValueError(f"profile {prof.name}: unknown quant scheme {prof.quant_scheme!r}")
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
            backend: str = "reference", tq1_artifact: str | Path | None = None) -> ConversionReport:
    """In-place swap of the profile's target linears for BitLinear (fp32 latent)
    and fused expert stacks for BitExperts. Never touches model.config (config
    preservation is a hard requirement)."""
    from bitnet_train.bitlinear import BitExperts

    if profile.quant_scheme == "tq1_v":
        return _convert_tq1(model, profile, tq1_artifact)

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
                   backend: str = "reference", dtype=torch.float32,
                   tq1_artifact: str | Path | None = None):
    """Load a converted checkpoint: BitLinear's state dict is key/shape-identical to
    a bias-free nn.Linear, so from_pretrained restores the latents into plain
    Linears and convert() re-wraps them (weights are copied into fp32 latents)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=dtype)
    report = convert(model, profile, backend=backend, tq1_artifact=tq1_artifact)
    return model, report


def _convert_tq1(model: nn.Module, profile: ArchProfile,
                  artifact_path: str | Path | None) -> ConversionReport:
    from bitnet_train.tq1.artifact import ArtifactReader
    from bitnet_train.tq1.calibration import load_calibration_artifact
    from bitnet_train.tq1.packing import unpack_payload
    from bitnet_train.tq1.qat import TQ1Embedding, TQ1Linear, TQ1OutputHead
    from bitnet_train.tq1.spec import FLOAT_PROFILES

    artifact_path = artifact_path or profile.quant.get("artifact")
    if not artifact_path:
        raise ValueError("TQ1 conversion requires a canonical artifact")
    reader = ArtifactReader(artifact_path)
    reader.validate()
    quant_spec = reader.quant_spec
    if profile.quant.get("default_profile") != quant_spec.default_profile:
        raise ValueError("profile default TQ1 format differs from canonical artifact")
    if profile.quant.get("default_codebook_id") != quant_spec.default_codebook_id:
        raise ValueError("profile default codebook differs from canonical artifact")
    if "shared_embedding_head" in profile.quant \
            and bool(profile.quant["shared_embedding_head"]) \
            != quant_spec.shared_embedding_head:
        raise ValueError("profile shared embedding/head policy differs from canonical artifact")
    for name in ("shared_head_importance", "shared_embedding_importance"):
        if name in profile.quant and float(profile.quant[name]) != getattr(quant_spec, name):
            raise ValueError(f"profile {name} differs from canonical artifact")
    configured_spec = profile.quant.get("spec")
    if configured_spec:
        raw = json.loads(Path(configured_spec).read_text())
        from bitnet_train.tq1.spec import QuantSpec
        expected = QuantSpec.from_dict(raw)
        if expected.sha256() != quant_spec.sha256():
            raise ValueError("profile QuantSpec differs from canonical artifact")
    registry = reader.registry()
    statistics = {}
    if profile.quant.get("importance_stats"):
        statistics, _ = load_calibration_artifact(profile.quant["importance_stats"])

    classes = classify_linears(model, profile)
    configured_targets = sorted(name for name, kind in classes.items() if kind == TERNARIZE)
    target_names = sorted(
        name for name in configured_targets
        if quant_spec.resolve_profile(name)[0] not in FLOAT_PROFILES)
    float_overrides = sorted(set(configured_targets) - set(target_names))
    kept = sorted([name for name, kind in classes.items() if kind == KEEP_FP]
                  + float_overrides)
    linear_items = [item for item in reader.manifest["tensors"]
                    if item.get("consumer_kind", "linear") == "linear"]
    shared_items = [item for item in reader.manifest["tensors"]
                    if item.get("consumer_kind", "linear") == "shared_embedding_head"]
    artifact_names = sorted(item["module_path"] for item in linear_items)
    if artifact_names != target_names:
        raise ValueError(f"TQ1 artifact/profile inventory mismatch: expected={target_names}, "
                         f"artifact={artifact_names}")
    if bool(shared_items) != quant_spec.shared_embedding_head or len(shared_items) > 1:
        raise ValueError("TQ1 artifact shared embedding/head inventory is inconsistent")
    tern_params = 0
    for name in target_names:
        old: nn.Linear = model.get_submodule(name)
        if old.bias is not None:
            raise ValueError(f"{name}: TQ1Linear is bias-free")
        item, payload, scales = reader.tensor(name + ".weight")
        if scales is None:
            raise ValueError(f"{name}: QAT requires a row-scale artifact profile")
        indices, _, _ = unpack_payload(payload, item["profile"])
        codebook = registry[item["codebook_id"]]
        diag = statistics.get(name + ".diag")
        cov8 = statistics.get(name + ".cov8")
        new = TQ1Linear(
            old.weight.detach(), scales, codebook, quant_spec,
            profile=item["profile"], importance_diag=diag,
            importance_cov8=cov8, initial_indices=indices,
            phase=profile.quant.get("qat_projection", "soft"),
            top_m=int(profile.quant.get("top_m", 8)),
            temperature=float(profile.quant.get("temperature_start", 1.0)),
            assignment_chunk=int(profile.quant.get("assignment_chunk", 2048)),
        )
        new.weight.requires_grad_(old.weight.requires_grad)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], new)
        tern_params += new.weight.numel()
    shared_names: list[str] = []
    if shared_items:
        item = shared_items[0]
        name = item["module_path"]
        old = model.get_submodule(name)
        if not isinstance(old, nn.Embedding):
            raise ValueError(f"{name}: shared TQ1 target is not an Embedding")
        aliases = [alias_name for alias_name, alias in reader.aliases.items()
                   if alias["target"] == item["state_dict_name"]]
        if aliases != ["lm_head.weight"] or old.weight is not model.lm_head.weight:
            raise ValueError("shared TQ1 artifact/model does not preserve tied head identity")
        _, payload, scales = reader.tensor(item["state_dict_name"])
        if scales is None:
            raise ValueError("shared embedding/head QAT requires row scales")
        indices, _, _ = unpack_payload(payload, item["profile"])
        from bitnet_train.tq1.pipeline import shared_importance_for_module
        shared_importance, _ = shared_importance_for_module(
            statistics, name, quant_spec, old.embedding_dim)
        if shared_importance.mode == "block256":
            raise ValueError("shared QAT supports diagonal/covariance8 importance, not cov256")
        shared = TQ1Embedding(
            old.weight.detach(), scales, registry[item["codebook_id"]], quant_spec,
            profile=item["profile"], importance_diag=shared_importance.diag,
            importance_cov8=shared_importance.cov8, initial_indices=indices,
            phase=profile.quant.get("qat_projection", "soft"),
            top_m=int(profile.quant.get("top_m", 8)),
            temperature=float(profile.quant.get("temperature_start", 1.0)),
            assignment_chunk=int(profile.quant.get("assignment_chunk", 2048)),
            padding_idx=old.padding_idx, max_norm=old.max_norm,
            norm_type=old.norm_type, scale_grad_by_freq=old.scale_grad_by_freq,
            sparse=old.sparse)
        shared.weight.requires_grad_(old.weight.requires_grad)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], shared)
        model.lm_head = TQ1OutputHead(shared)
        tern_params += shared.weight.numel()
        shared_names.append(name)
        reader.verify_model_aliases(model)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    lin_flops = tern_params + sum(
        module.weight.numel() for module_name, module in model.named_modules()
        if isinstance(module, nn.Linear))
    family_counts: dict[str, int] = {}
    for name in target_names + shared_names:
        family_counts[_family(name)] = family_counts.get(_family(name), 0) + 1
    return ConversionReport(
        profile=profile.name, n_ternarized=len(target_names) + len(shared_names),
        n_kept_fp=len(kept) - len(shared_names),
        ternarized=target_names + shared_names,
        kept_fp=[name for name in kept if name != "lm_head"],
        family_counts=family_counts,
        ternary_param_fraction=tern_params / max(total_params, 1),
        ternary_flop_fraction=tern_params / max(lin_flops, 1),
    )


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
        tq1_scale_ids: set[int] = set()
        if profile.quant_scheme == "tq1_v":
            from bitnet_train.tq1.qat import TQ1Linear
            tq1_modules = [module for module in model.modules() if isinstance(module, TQ1Linear)]
            bit_ids |= {id(module.weight) for module in tq1_modules}
            tq1_scale_ids = {id(module.scale_parameter) for module in tq1_modules}
        latents = [p for p in model.parameters() if id(p) in bit_ids and p.requires_grad]
        rest = [p for p in model.parameters()
                if id(p) not in bit_ids and id(p) not in tq1_scale_ids and p.requires_grad]
        groups = [
            {"params": latents, "lr": base_lr, "weight_decay": weight_decay,
             "name": "bitlinear_latents", "decay_masked": False},
        ]
        if tq1_scale_ids:
            groups.append({
                "params": [p for p in model.parameters()
                           if id(p) in tq1_scale_ids and p.requires_grad],
                "lr": base_lr * 0.1, "weight_decay": 0.0,
                "name": "tq1_row_scales", "decay_masked": False,
            })
        groups.append(
            {"params": rest, "lr": base_lr * (0.1 if tq1_scale_ids else 1.0),
             "weight_decay": 0.0,
             "name": "other", "decay_masked": False},
        )
        return groups

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
