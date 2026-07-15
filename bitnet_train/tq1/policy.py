"""Deterministic whole-model mixed-format promotion search."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Mapping, Sequence

from .packing import PROFILE_LAYOUTS
from .spec import FLOAT_PROFILES, QuantSpec, TensorRule, canonical_json


@dataclass(frozen=True)
class PolicyTensor:
    name: str
    logical_shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        result = 1
        for dim in self.logical_shape:
            result *= dim
        return result

    def bytes_for(self, profile: str) -> int:
        if profile in {"fp16", "bf16"}:
            return self.elements * 2
        if profile == "fp32":
            return self.elements * 4
        spec = PROFILE_LAYOUTS[profile]
        rows = 1
        for dim in self.logical_shape[:-1]:
            rows *= dim
        width = self.logical_shape[-1]
        payload = rows * (width // 256) * spec.block_bytes
        return payload + (rows * 2 if spec.scale_mode == "row" else 0)


@dataclass(frozen=True)
class PolicySearchResult:
    policy: Mapping[str, str]
    objective: float
    total_bytes: int
    trials: tuple[dict, ...]


def greedy_policy_search(tensors: Sequence[PolicyTensor], starting_policy: Mapping[str, str],
                         promotion_edges: Mapping[str, Sequence[str]], *,
                         byte_budget: int, evaluator: Callable[[Mapping[str, str]], float],
                         policy_split_sha256: str, max_trials: int,
                         move_groups: Mapping[str, Sequence[str]] | None = None) \
        -> PolicySearchResult:
    by_name = {tensor.name: tensor for tensor in tensors}
    if len(by_name) != len(tensors) or set(by_name) != set(starting_policy):
        raise ValueError("tensor inventory and starting policy must match exactly")
    if byte_budget <= 0 or max_trials <= 0:
        raise ValueError("byte budget and max_trials must be positive")
    if re.fullmatch(r"[0-9a-f]{64}", policy_split_sha256) is None:
        raise ValueError("policy split SHA-256 must be 64 lowercase hex characters")
    groups = ({name: (name,) for name in sorted(by_name)} if move_groups is None else
              {name: tuple(members) for name, members in move_groups.items()})
    flattened = [name for members in groups.values() for name in members]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(by_name) \
            or any(not members for members in groups.values()):
        raise ValueError("move groups must be a disjoint total tensor partition")

    def size(policy):
        return sum(by_name[name].bytes_for(profile) for name, profile in policy.items())

    policy = dict(starting_policy)
    current_bytes = size(policy)
    if current_bytes > byte_budget:
        raise ValueError("starting policy exceeds byte budget")
    current_objective = float(evaluator(policy))
    if not math.isfinite(current_objective):
        raise ValueError("policy evaluator returned a nonfinite objective")
    trials: list[dict] = [{
        "trial": 0, "move": None, "objective": current_objective,
        "total_bytes": current_bytes, "accepted": True,
        "policy_split_sha256": policy_split_sha256,
        "policy_sha256": __import__("hashlib").sha256(
            canonical_json(policy).encode()).hexdigest(),
    }]
    trial_id = 1
    while trial_id <= max_trials:
        candidates = []
        for group_name, names in sorted(groups.items()):
            before_profiles = {policy[name] for name in names}
            if len(before_profiles) != 1:
                continue
            before = next(iter(before_profiles))
            for after in promotion_edges.get(before, ()):
                candidate = dict(policy)
                for name in names:
                    candidate[name] = after
                candidate_bytes = size(candidate)
                delta = candidate_bytes - current_bytes
                if delta <= 0:
                    raise ValueError(f"promotion {before}->{after} is not byte-increasing")
                if candidate_bytes > byte_budget:
                    continue
                objective = float(evaluator(candidate))
                if not math.isfinite(objective):
                    raise ValueError("policy evaluator returned a nonfinite objective")
                improvement = current_objective - objective
                record = {
                    "trial": trial_id, "move": {
                        "group": group_name, "tensors": list(names),
                        "from": before, "to": after},
                    "objective": objective, "total_bytes": candidate_bytes,
                    "byte_delta": delta, "improvement": improvement,
                    "improvement_per_byte": improvement / delta,
                    "accepted": False, "policy_split_sha256": policy_split_sha256,
                    "policy_sha256": __import__("hashlib").sha256(
                        canonical_json(candidate).encode()).hexdigest(),
                }
                trials.append(record); trial_id += 1
                if improvement > 0:
                    canonical_policy = tuple(sorted(candidate.items()))
                    candidates.append((-(improvement / delta), candidate_bytes,
                                       canonical_policy, objective, record, candidate))
                if trial_id > max_trials:
                    break
            if trial_id > max_trials:
                break
        if not candidates:
            break
        _, candidate_bytes, _, objective, record, candidate = min(candidates)
        record["accepted"] = True
        policy = candidate
        current_bytes = candidate_bytes
        current_objective = objective
    return PolicySearchResult(policy, current_objective, current_bytes, tuple(trials))


def make_move_groups(tensors: Sequence[PolicyTensor], granularity: str) \
        -> dict[str, tuple[str, ...]]:
    if granularity not in {"tensor", "family", "layer"}:
        raise ValueError("policy move granularity must be tensor, family, or layer")
    result: dict[str, list[str]] = {}
    for tensor in tensors:
        if granularity == "tensor":
            key = tensor.name
        elif granularity == "layer":
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", tensor.name)
            if match is None:
                raise ValueError(f"cannot determine layer for {tensor.name}")
            key = "layer." + match.group(1)
        else:
            match = re.search(
                r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)(?:\.weight)?$",
                tensor.name)
            if match is None:
                raise ValueError(f"cannot determine projection family for {tensor.name}")
            key = match.group(1)
        result.setdefault(key, []).append(tensor.name)
    return {key: tuple(sorted(values)) for key, values in sorted(result.items())}


def policy_to_spec(spec: QuantSpec, policy: Mapping[str, str], *,
                   profile_codebooks: Mapping[str, str] | None = None) -> QuantSpec:
    """Materialize a complete resolved policy as exact, disjoint tensor rules."""
    configured = dict(profile_codebooks or {})
    rules = []
    for state_name, profile in sorted(policy.items()):
        module_name = state_name.removesuffix(".weight")
        if profile in FLOAT_PROFILES:
            codebook_id = None
        else:
            if profile not in PROFILE_LAYOUTS:
                raise ValueError(f"unsupported policy profile {profile!r}")
            expected_format = "v11" if profile.startswith("tq1_v11-") else "v12"
            expected_encoding = ("direct_joint" if "-i-" in profile else
                                 "product" if "-p-" in profile else "sign_canonical")
            compatible = [book.id for book in spec.codebooks
                          if book.format == expected_format and
                          book.encoding == expected_encoding]
            codebook_id = configured.get(profile)
            if codebook_id is None:
                if len(compatible) != 1:
                    raise ValueError(
                        f"profile {profile} needs an explicit codebook mapping; "
                        f"compatible={compatible}")
                codebook_id = compatible[0]
            if codebook_id not in compatible:
                raise ValueError(f"codebook {codebook_id!r} is incompatible with {profile}")
        rules.append(TensorRule(re.escape(module_name), profile, codebook_id))
    from dataclasses import replace
    return replace(spec, tensor_overrides=tuple(rules))
