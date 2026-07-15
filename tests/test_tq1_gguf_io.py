import numpy as np

from bitnet_train.tq1.gguf_io import (
    ParsedGGUF, TensorRecord, UINT32, encode_metadata, parse_gguf,
    write_rewritten_gguf)


def test_minimal_gguf_rewrite_preserves_base_and_custom_tensors(tmp_path):
    raw_metadata = encode_metadata("general.architecture", "llama") \
        + encode_metadata("general.alignment", 32, UINT32)
    base = ParsedGGUF(
        version=3,
        metadata={"general.architecture": "llama", "general.alignment": 32},
        metadata_types={"general.architecture": 8, "general.alignment": UINT32},
        raw_metadata=raw_metadata,
        tensors=(), alignment=32)
    f32 = np.arange(8, dtype=np.float32).tobytes()
    tq1 = bytes(range(44))
    path = tmp_path / "model.gguf"
    write_rewritten_gguf(base, path, (
        TensorRecord("norm.weight", (8,), 0, f32),
        TensorRecord("blk.0.attn_q.weight", (256, 1), 45, tq1),
    ), {"tq1.spec_revision": ("1.0.0", None)})
    parsed = parse_gguf(path)
    assert parsed.metadata["general.architecture"] == "llama"
    assert parsed.metadata["tq1.spec_revision"] == "1.0.0"
    assert [(tensor.name, tensor.tensor_type, tensor.data) for tensor in parsed.tensors] == [
        ("norm.weight", 0, f32),
        ("blk.0.attn_q.weight", 45, tq1),
    ]
