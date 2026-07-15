from dataclasses import replace

import torch
from torch import nn

from bitnet_train.tq1.artifact import ArtifactBuilder, ArtifactReader
from bitnet_train.tq1.codebook import CodebookRegistry, sign_canonical_codebook
from bitnet_train.tq1.packing import pack_payload
from bitnet_train.tq1.pipeline import export_qat_model
from bitnet_train.tq1.qat import TQ1Linear
from bitnet_train.tq1.solver import canonical_shapes
from bitnet_train.tq1.spec import QuantSpec


class _Model(nn.Module):
    def __init__(self, tq1):
        super().__init__()
        self.x = tq1
        self.lm_head = nn.Linear(2, 2, bias=False)


def test_frozen_qat_export_is_index_and_scale_exact(tmp_path):
    shapes = canonical_shapes()
    book = sign_canonical_codebook("export", "v11", torch.cat((
        shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)][:1023])))
    spec = replace(QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=("x",), keep_fp_regexes=("lm_head",),
        importance_mode="uniform"), candidate_count=4)
    registry = CodebookRegistry({book.id: book})
    indices = torch.zeros((2, 32), dtype=torch.int64)
    scales = torch.tensor([0.5, 0.75], dtype=torch.float16)
    source = tmp_path / "source"
    builder = ArtifactBuilder(
        spec, registry, source_model="tiny", source_revision="0" * 40,
        tokenizer_sha256="1" * 64, chat_template_sha256="2" * 64)
    builder.add_quantized(
        "x.weight", "x", pack_payload(indices, "tq1_v11-j-r"),
        logical_shape=(2, 256), profile="tq1_v11-j-r",
        codebook_id=book.id, row_scales=scales)
    builder.add_non_tq1("lm_head.weight", torch.eye(2))
    source_files = tmp_path / "source_files"
    source_files.mkdir()
    (source_files / "config.json").write_text("{}")
    (source_files / "tokenizer_config.json").write_text("{}")
    builder.write(source, source_files=source_files, quantization_report={})

    module = TQ1Linear(
        torch.zeros(2, 256), scales, book, spec, profile="tq1_v11-j-r",
        initial_indices=indices, phase="hard", top_m=4)
    module.freeze_indices()
    model = _Model(module)
    output = export_qat_model(
        model, source, tmp_path / "qat", checkpoint_identity="step-10")
    reader = ArtifactReader(output)
    reader.validate()
    _, got_payload, got_scales = reader.tensor("x.weight")
    expected_payload, expected_scales = module.export_projection()
    assert torch.equal(got_payload, expected_payload)
    assert torch.equal(got_scales, expected_scales)
