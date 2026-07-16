from __future__ import annotations

import pytest
import torch

from bitnet_train.conversion import ArchProfile, build_param_groups, convert
from bitnet_train.tq1.artifact import ArtifactBuilder
from bitnet_train.tq1.codebook import CodebookRegistry, base3_ids, sign_canonical_codebook
from bitnet_train.tq1.ptq import Importance, PTQConfig, project_weight, ternary_universe
from bitnet_train.tq1.qat import TQ1Linear
from bitnet_train.tq1.spec import QuantSpec


def _book():
    universe = ternary_universe()
    nz = universe != 0
    first = nz.long().argmax(1)
    negative = nz.any(1) & (universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(negative, -1, 1).to(torch.int8)[:, None]
    shapes = universe[torch.unique(base3_ids(canonical), sorted=True)]
    shapes = torch.cat((shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)]))
    return sign_canonical_codebook("tiny", "v11", shapes[:1024])


def test_profile_driven_tq1_conversion_uses_canonical_artifact(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(23)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=256, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
        tie_word_embeddings=True,
    )).float()
    targets = (r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
               r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj")
    book = _book()
    spec = QuantSpec.core(default_profile="tq1_v11-j-r", codebook=book.ref(),
                          target_regexes=targets, keep_fp_regexes=("lm_head",),
                          importance_mode="uniform")
    registry = CodebookRegistry({book.id: book})
    builder = ArtifactBuilder(
        spec, registry, source_model="tiny", source_revision="a" * 40,
        tokenizer_sha256="b" * 64, chat_template_sha256="c" * 64)
    source_files = tmp_path / "source_files"
    source_files.mkdir()
    model.config.save_pretrained(source_files)
    (source_files / "tokenizer_config.json").write_text("{}")
    target_state_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(
                __import__("re").fullmatch(pattern, name) for pattern in targets):
            result = project_weight(
                module.weight, book, Importance(),
                PTQConfig("tq1_v11-j-r", weight_metric="uniform",
                          assignment_mode="shortlist", candidate_count=8,
                          alternating_iterations=2),
            )
            state_name = name + ".weight"
            target_state_names.add(state_name)
            builder.add_quantized(
                state_name, name, result.payload, logical_shape=tuple(module.weight.shape),
                profile="tq1_v11-j-r", codebook_id=book.id,
                row_scales=result.row_scales)
    for name, value in model.state_dict().items():
        if name not in target_state_names:
            builder.add_non_tq1(name, value)
    artifact = builder.write(tmp_path / "artifact", source_files=source_files,
                             quantization_report={"ok": True})
    profile = ArchProfile(
        name="tiny_tq1", base_model="tiny", teacher="tiny",
        target_linear_regexes=list(targets), keep_fp_regexes=["lm_head"],
        quant={
            "scheme": "tq1_v", "artifact": str(artifact),
            "default_profile": "tq1_v11-j-r", "default_codebook_id": book.id,
            "qat_projection": "hard", "top_m": 4,
        },
    )
    mismatched = ArchProfile(
        name="bad_tq1", base_model="tiny", teacher="tiny",
        target_linear_regexes=list(targets), keep_fp_regexes=["lm_head"],
        quant={**profile.quant, "candidate_count": spec.candidate_count + 1})
    with pytest.raises(ValueError, match="candidate_count differs"):
        convert(model, mismatched, tq1_artifact=artifact)
    report = convert(model, profile, tq1_artifact=artifact)
    assert report.n_ternarized == 7
    assert sum(isinstance(module, TQ1Linear) for module in model.modules()) == 7
    output = model(torch.randint(0, 512, (1, 8))).logits
    assert output.shape == (1, 8, 512) and torch.isfinite(output).all()
    groups = build_param_groups(model, profile, 1e-4)
    assert [group["name"] for group in groups] == [
        "bitlinear_latents", "tq1_row_scales", "other"]
    assert groups[1]["lr"] == groups[2]["lr"] == 1e-5
