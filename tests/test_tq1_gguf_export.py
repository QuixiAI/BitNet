from dataclasses import replace

import pytest
import torch

from bitnet_train.tq1.artifact import ArtifactBuilder
from bitnet_train.tq1.codebook import (
    CodebookRegistry, base3_ids, direct_joint_codebook, product_codebook,
    sign_canonical_codebook)
from bitnet_train.export.compare_gguf import tq1_tensor_parity
from bitnet_train.tq1.gguf import rewrite_base_gguf, validate_tq1_gguf
from bitnet_train.tq1.gguf_io import (
    ParsedGGUF, TensorRecord, UINT32, encode_metadata, parse_gguf,
    write_rewritten_gguf)
from bitnet_train.tq1.packing import pack_payload
from bitnet_train.tq1.ptq import ternary_universe
from bitnet_train.tq1.solver import canonical_shapes
from bitnet_train.tq1.spec import QuantSpec, TensorRule


def test_exact_gguf_rewrite_preserves_payload_scale_and_codebook(tmp_path):
    shapes = canonical_shapes()
    book = sign_canonical_codebook("ggufbook", "v11", torch.cat((
        shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)][:1023])))
    module_path = "model.layers.0.self_attn.q_proj"
    spec = replace(QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=(r"model\.layers\.\d+\.self_attn\.q_proj",),
        keep_fp_regexes=("lm_head",), importance_mode="uniform"),
        candidate_count=4)
    source_files = tmp_path / "source_files"
    source_files.mkdir()
    (source_files / "config.json").write_text('{"tie_word_embeddings": true}')
    (source_files / "tokenizer_config.json").write_text("{}")
    artifact = tmp_path / "artifact"
    builder = ArtifactBuilder(
        spec, CodebookRegistry({book.id: book}), source_model="tiny",
        source_revision="a" * 40, tokenizer_sha256="b" * 64,
        chat_template_sha256="c" * 64)
    indices = torch.arange(64).reshape(2, 32)
    payload = pack_payload(indices, "tq1_v11-j-r")
    scales = torch.tensor([0.5, 0.25], dtype=torch.float16)
    builder.add_quantized(
        module_path + ".weight", module_path, payload,
        logical_shape=(2, 256), profile="tq1_v11-j-r",
        codebook_id=book.id, row_scales=scales)
    tied = torch.ones(2, 2)
    builder.add_non_tq1("model.embed_tokens.weight", tied)
    builder.add_non_tq1("lm_head.weight", tied)
    builder.write(artifact, source_files=source_files, quantization_report={})

    metadata_values = {
        "general.architecture": "llama", "general.alignment": 32,
        "llama.attention.head_count": 1, "llama.attention.head_count_kv": 1,
    }
    raw_metadata = b"".join((
        encode_metadata("general.architecture", "llama"),
        encode_metadata("general.alignment", 32, UINT32),
        encode_metadata("llama.attention.head_count", 1, UINT32),
        encode_metadata("llama.attention.head_count_kv", 1, UINT32),
    ))
    base = ParsedGGUF(3, metadata_values, {}, raw_metadata, (), 32)
    base_path = tmp_path / "base.gguf"
    write_rewritten_gguf(base, base_path, (
        TensorRecord("blk.0.attn_q.weight", (256, 2), 1, bytes(2 * 256 * 2)),
        TensorRecord("token_embd.weight", (2, 2), 1, bytes(2 * 2 * 2)),
    ), {})
    output = tmp_path / "tq1.gguf"
    report = rewrite_base_gguf(artifact, base_path, output)
    checked = validate_tq1_gguf(artifact, output)
    assert checked["final_gguf_bytes"] == output.stat().st_size
    assert checked["size_accounting"]["final_gguf_bytes"] == output.stat().st_size
    assert report["target_tensors"] == checked["target_tensors"] == 1
    assert checked["ok"] is True
    written_names = {item.name for item in parse_gguf(output).tensors}
    assert "token_embd.weight" in written_names and "output.weight" not in written_names
    rows, ok = tq1_tensor_parity(artifact, output)
    assert ok and len(rows) == 1 and rows[0].status == "exact"

    damaged = tmp_path / "damaged.gguf"
    data = bytearray(output.read_bytes())
    data[-1] ^= 1
    damaged.write_bytes(data)
    rows, ok = tq1_tensor_parity(artifact, damaged)
    assert not ok and rows[-1].status == "mismatch"


def test_quantized_tied_embedding_head_exports_one_low_bit_gguf_tensor(tmp_path):
    shapes = canonical_shapes()
    book = sign_canonical_codebook("sharedgguf", "v11", torch.cat((
        shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)][:1023])))
    spec = replace(QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=(r"model\.embed_tokens",), keep_fp_regexes=("lm_head",),
        importance_mode="uniform"), candidate_count=4, shared_embedding_head=True)
    source = tmp_path / "shared_source"
    source.mkdir()
    (source / "config.json").write_text('{"tie_word_embeddings": true}')
    (source / "tokenizer_config.json").write_text("{}")
    builder = ArtifactBuilder(
        spec, CodebookRegistry({book.id: book}), source_model="tiny",
        source_revision="a" * 40, tokenizer_sha256="b" * 64,
        chat_template_sha256="c" * 64)
    indices = torch.arange(32).reshape(1, 32).repeat(3, 1)
    payload = pack_payload(indices, "tq1_v11-j-r")
    scales = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float16)
    shared_weight = torch.ones(3, 256)
    builder.add_quantized(
        "model.embed_tokens.weight", "model.embed_tokens", payload,
        logical_shape=(3, 256), profile="tq1_v11-j-r", codebook_id=book.id,
        row_scales=scales, source_tensor=shared_weight,
        consumer_kind="shared_embedding_head")
    builder.add_alias("lm_head.weight", "model.embed_tokens.weight", shared_weight)
    artifact = builder.write(
        tmp_path / "shared_artifact", source_files=source, quantization_report={})

    values = {"general.architecture": "llama", "general.alignment": 32,
              "llama.attention.head_count": 1, "llama.attention.head_count_kv": 1}
    raw = b"".join((
        encode_metadata("general.architecture", "llama"),
        encode_metadata("general.alignment", 32, UINT32),
        encode_metadata("llama.attention.head_count", 1, UINT32),
        encode_metadata("llama.attention.head_count_kv", 1, UINT32)))
    base = ParsedGGUF(3, values, {}, raw, (), 32)
    base_path = tmp_path / "shared_base.gguf"
    write_rewritten_gguf(base, base_path, (
        TensorRecord("token_embd.weight", (256, 3), 1, bytes(3 * 256 * 2)),), {})
    output = tmp_path / "shared_tq1.gguf"
    rewrite_base_gguf(artifact, base_path, output)
    checked = validate_tq1_gguf(artifact, output)
    assert checked["ok"] is True
    written = {item.name: item for item in parse_gguf(output).tensors}
    assert written["token_embd.weight"].tensor_type == 45
    assert written["token_embd.weight"].data == payload.numpy().tobytes()
    assert "output.weight" not in written
    rows, ok = tq1_tensor_parity(artifact, output)
    assert ok and [row.gguf_name for row in rows] == ["token_embd.weight"]


def test_mixed_bf16_override_is_written_exactly_with_qk_permutation(tmp_path):
    shapes = canonical_shapes()
    book = sign_canonical_codebook("mixedgguf", "v11", torch.cat((
        shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)][:1023])))
    spec = replace(QuantSpec.core(
        default_profile="tq1_v11-j-r", codebook=book.ref(),
        target_regexes=(r"model\.layers\.0\.self_attn\.(q|k)_proj",),
        keep_fp_regexes=("lm_head",), importance_mode="uniform"),
        candidate_count=4, tensor_overrides=(TensorRule(
            r"model\.layers\.0\.self_attn\.q_proj", "bf16", None),))
    source = tmp_path / "mixed_source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "tokenizer_config.json").write_text("{}")
    artifact = tmp_path / "mixed_artifact"
    builder = ArtifactBuilder(
        spec, CodebookRegistry({book.id: book}), source_model="tiny",
        source_revision="d" * 40, tokenizer_sha256="e" * 64,
        chat_template_sha256="f" * 64)
    indices = torch.arange(128).reshape(4, 32)
    builder.add_quantized(
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj",
        pack_payload(indices, "tq1_v11-j-r"), logical_shape=(4, 256),
        profile="tq1_v11-j-r", codebook_id=book.id,
        row_scales=torch.ones(4, dtype=torch.float16))
    q = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256).to(torch.bfloat16)
    builder.add_non_tq1("model.layers.0.self_attn.q_proj.weight", q)
    builder.add_non_tq1("lm_head.weight", torch.ones(2, 2))
    builder.write(artifact, source_files=source, quantization_report={})

    values = {
        "general.architecture": "llama", "general.alignment": 32,
        "llama.attention.head_count": 1, "llama.attention.head_count_kv": 1}
    raw = b"".join((
        encode_metadata("general.architecture", "llama"),
        encode_metadata("general.alignment", 32, UINT32),
        encode_metadata("llama.attention.head_count", 1, UINT32),
        encode_metadata("llama.attention.head_count_kv", 1, UINT32)))
    base = ParsedGGUF(3, values, {}, raw, (), 32)
    base_path = tmp_path / "mixed_base.gguf"
    write_rewritten_gguf(base, base_path, (
        TensorRecord("blk.0.attn_q.weight", (256, 4), 1, bytes(256 * 4 * 2)),
        TensorRecord("blk.0.attn_k.weight", (256, 4), 1, bytes(256 * 4 * 2)),
        TensorRecord("output.weight", (2, 2), 1, bytes(8))), {})
    output = tmp_path / "mixed.gguf"
    rewrite_base_gguf(artifact, base_path, output)
    checked = validate_tq1_gguf(artifact, output)
    assert checked["ok"] is True
    written = {item.name: item for item in parse_gguf(output).tensors}
    assert written["blk.0.attn_q.weight"].tensor_type == 30
    order = torch.tensor([0, 2, 1, 3])
    assert written["blk.0.attn_q.weight"].data == \
        q[order].contiguous().view(torch.uint8).numpy().tobytes()


def _four_lane_universe():
    value = torch.arange(3 ** 4, dtype=torch.int64)
    lanes = []
    for _ in range(4):
        lanes.append((value % 3 - 1).to(torch.int8))
        value //= 3
    return torch.stack(lanes, 1)


def _profile_book(profile):
    fmt = "v11" if "v11" in profile else "v12"
    if "-i-" in profile:
        universe = ternary_universe()
        zero = universe[(universe == 0).all(1)]
        nonzero = universe[~(universe == 0).all(1)][:2047]
        return direct_joint_codebook(
            "gguf_i", torch.cat((nonzero[:1029], zero, nonzero[1029:])),
            scope="loaded")
    if "-p-" in profile:
        universe = _four_lane_universe()
        nonzero = universe != 0
        first = nonzero.long().argmax(1)
        negative = nonzero.any(1) & (
            universe.gather(1, first[:, None]).squeeze(1) < 0)
        canonical = universe * torch.where(
            negative, -1, 1).to(torch.int8)[:, None]
        representatives = universe[torch.unique(base3_ids(canonical), sorted=True)]
        representatives = torch.cat((
            representatives[(representatives == 0).all(1)],
            representatives[~(representatives == 0).all(1)]))
        a = representatives[:32]
        if fmt == "v11":
            b = representatives[:32]
        else:
            selected = torch.cat((representatives, -representatives[1:24]))
            zero = selected[(selected == 0).all(1)]
            rest = selected[~(selected == 0).all(1)]
            b = torch.cat((zero, rest[torch.argsort(base3_ids(rest))]))
        return product_codebook(f"gguf_{fmt}_p", fmt, a, b)
    shapes = canonical_shapes()
    count = 1024 if fmt == "v11" else 2048
    return sign_canonical_codebook(f"gguf_{fmt}_j", fmt, torch.cat((
        shapes[(shapes == 0).all(1)],
        shapes[~(shapes == 0).all(1)][:count - 1])))


@pytest.mark.parametrize(("profile", "ggml_type", "has_row_scale"), [
    ("tq1_v11-j-r", 45, True),
    ("tq1_v12-j-r", 46, True),
    ("tq1_v11-i-r", 45, True),
    ("tq1_v11-p-r", 45, True),
    ("tq1_v12-p-r", 46, True),
    ("tq1_v11-j-a4-r", 47, True),
    ("tq1_v11-j-b", 43, False),
    ("tq1_v12-j-b", 44, False),
])
def test_exact_gguf_export_covers_every_format_v1_profile(
        tmp_path, profile, ggml_type, has_row_scale):
    book = _profile_book(profile)
    module_path = "model.layers.0.self_attn.o_proj"
    spec = replace(QuantSpec.core(
        default_profile=profile, codebook=book.ref(),
        target_regexes=(r"model\.layers\.0\.self_attn\.o_proj",),
        keep_fp_regexes=("lm_head",), importance_mode="uniform"),
        candidate_count=4)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "tokenizer_config.json").write_text("{}")
    builder = ArtifactBuilder(
        spec, CodebookRegistry({book.id: book}), source_model="tiny",
        source_revision="9" * 40, tokenizer_sha256="8" * 64,
        chat_template_sha256="7" * 64)
    indices = torch.zeros((2, 32), dtype=torch.int64)
    row_scales = None
    pack_options = {}
    if profile.endswith("-b"):
        pack_options["block_scales"] = torch.tensor(
            [[0.5], [0.25]], dtype=torch.float16)
    else:
        row_scales = torch.tensor([0.5, 0.25], dtype=torch.float16)
    if "-a4-" in profile:
        pack_options["affine_nibbles"] = torch.arange(
            8, dtype=torch.uint8).reshape(1, 1, 8).repeat(2, 1, 1)
    payload = pack_payload(indices, profile, **pack_options)
    builder.add_quantized(
        module_path + ".weight", module_path, payload,
        logical_shape=(2, 256), profile=profile, codebook_id=book.id,
        row_scales=row_scales)
    builder.add_non_tq1("lm_head.weight", torch.ones(2, 2))
    artifact = builder.write(
        tmp_path / "artifact", source_files=source, quantization_report={})

    values = {
        "general.architecture": "llama", "general.alignment": 32,
        "llama.attention.head_count": 1, "llama.attention.head_count_kv": 1}
    raw = b"".join((
        encode_metadata("general.architecture", "llama"),
        encode_metadata("general.alignment", 32, UINT32),
        encode_metadata("llama.attention.head_count", 1, UINT32),
        encode_metadata("llama.attention.head_count_kv", 1, UINT32)))
    base = ParsedGGUF(3, values, {}, raw, (), 32)
    base_path = tmp_path / "base.gguf"
    write_rewritten_gguf(base, base_path, (
        TensorRecord("blk.0.attn_output.weight", (256, 2), 1, bytes(1024)),
        TensorRecord("output.weight", (2, 2), 1, bytes(8))), {})
    output = tmp_path / "output.gguf"
    rewrite_base_gguf(artifact, base_path, output)
    assert validate_tq1_gguf(artifact, output)["ok"] is True
    parsed = parse_gguf(output)
    written = {item.name: item for item in parsed.tensors}
    weight = written["blk.0.attn_output.weight"]
    assert weight.tensor_type == ggml_type
    assert weight.data == payload.contiguous().numpy().tobytes()
    scale_name = "blk.0.attn_output.scale"
    assert (scale_name in written) is has_row_scale
    if has_row_scale:
        assert written[scale_name].data == \
            row_scales.contiguous().view(torch.uint8).numpy().tobytes()
    assert parsed.metadata["tq1.strict_ternary"] is ("-a4-" not in profile)
