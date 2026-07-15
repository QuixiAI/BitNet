from dataclasses import replace

import torch

from bitnet_train.tq1.codebook import sign_canonical_codebook
from bitnet_train.tq1.policy import (
    PolicyTensor, greedy_policy_search, make_move_groups, policy_to_spec)
from bitnet_train.tq1.solver import canonical_shapes
from bitnet_train.tq1.spec import CodebookRef, QuantSpec


def test_mixed_policy_search_uses_whole_model_gain_per_byte_and_is_deterministic():
    tensors = [PolicyTensor("a", (8, 256)), PolicyTensor("b", (8, 256))]
    start = {"a": "tq1_v11-j-r", "b": "tq1_v11-j-r"}

    def evaluate(policy):
        # b is much more sensitive, so it must be promoted first.
        return 10 - (1 if policy["a"] == "tq1_v12-j-r" else 0) \
            - (4 if policy["b"] == "tq1_v12-j-r" else 0)

    one_v12 = tensors[0].bytes_for("tq1_v12-j-r") + tensors[1].bytes_for("tq1_v11-j-r")
    result = greedy_policy_search(
        tensors, start, {"tq1_v11-j-r": ("tq1_v12-j-r",)},
        byte_budget=one_v12, evaluator=evaluate,
        policy_split_sha256="a" * 64, max_trials=10)
    assert result.policy == {"a": "tq1_v11-j-r", "b": "tq1_v12-j-r"}
    assert result.objective == 6
    assert [trial["move"]["tensors"] for trial in result.trials[1:3]] == [["a"], ["b"]]
    assert result.trials[2]["accepted"]


def test_family_moves_are_atomic_and_policy_materializes_exact_rules():
    tensors = [
        PolicyTensor("model.layers.0.self_attn.q_proj.weight", (8, 256)),
        PolicyTensor("model.layers.1.self_attn.q_proj.weight", (8, 256)),
        PolicyTensor("model.layers.0.mlp.down_proj.weight", (8, 256)),
    ]
    start = {item.name: "tq1_v11-j-r" for item in tensors}
    groups = make_move_groups(tensors, "family")
    assert groups["q_proj"] == tuple(item.name for item in tensors[:2])

    def evaluate(policy):
        return float(sum(profile == "tq1_v11-j-r" for profile in policy.values()))

    budget = sum(item.bytes_for("tq1_v12-j-r") for item in tensors)
    result = greedy_policy_search(
        tensors, start, {"tq1_v11-j-r": ("tq1_v12-j-r",)},
        byte_budget=budget, evaluator=evaluate,
        policy_split_sha256="b" * 64, max_trials=8, move_groups=groups)
    assert result.policy[tensors[0].name] == result.policy[tensors[1].name]
    accepted = [trial for trial in result.trials if trial["accepted"] and trial["move"]]
    assert accepted[0]["move"]["tensors"] in [list(groups["q_proj"]),
                                                list(groups["down_proj"])]

    shapes = canonical_shapes()
    v11 = sign_canonical_codebook("v11", "v11", torch.cat((
        shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)][:1023])))
    # A structural V12 ref is sufficient for policy resolution in this unit test.
    v12_ref = CodebookRef("v12", "v12", "sign_canonical", "model", "c" * 64)
    spec = QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=v11.ref(),
        target_regexes=(r"model\.layers\.\d+\..*",), keep_fp_regexes=("lm_head",),
        importance_mode="uniform")
    spec = replace(spec, codebooks=(v11.ref(), v12_ref))
    materialized = policy_to_spec(spec, {
        tensors[0].name: "tq1_v12-j-r",
        tensors[1].name: "bf16",
        tensors[2].name: "tq1_v11-j-r",
    })
    assert materialized.resolve_profile(
        "model.layers.0.self_attn.q_proj") == ("tq1_v12-j-r", "v12")
    assert materialized.resolve_profile(
        "model.layers.1.self_attn.q_proj") == ("bf16", None)
