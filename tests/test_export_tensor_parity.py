"""Export baking + TQ2_0 decode/encode parity — llama.cpp-free (train_plan §8.2:
exact code recovery from baked ternary; the decode layout is source-transcribed)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train import quant  # noqa: E402
from bitnet_train.export.compare_gguf import (  # noqa: E402
    TensorParityRow, decode_tq2_0, encode_tq2_0_ref, _is_ternary,
    parity_rows_ok)

rng = np.random.default_rng(0)


def test_parity_summary_fails_closed():
    exact = TensorParityRow("a", "a", "TQ2_0", "exact")
    assert parity_rows_ok([exact])
    for status in ("mismatch", "skipped", "unmapped"):
        assert not parity_rows_ok([TensorParityRow("a", "a", "TQ2_0", status)])
    assert not parity_rows_ok([])
    assert not parity_rows_ok([TensorParityRow(
        "a", "a", "TQ2_0", "exact", within_f16_bound=False)])


def _pack_tq2_0(codes: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Inverse of decode_tq2_0, mirroring quantize_row_tq2_0_ref's loop exactly."""
    N, K = codes.shape
    nb = K // 256
    blocks = np.zeros((N, nb, 66), np.uint8)
    cb = (codes.reshape(N, nb, 256) + 1).astype(np.uint8)
    for j in (0, 1):
        for n in range(4):
            blocks[:, :, j * 32:(j + 1) * 32] |= \
                (cb[:, :, j * 128 + n * 32:j * 128 + (n + 1) * 32] & 3) << (2 * n)
    blocks[:, :, 64:66] = d.astype(np.float16).reshape(N, nb, 1).view(np.uint8)
    return blocks.reshape(-1)


def test_tq2_0_pack_decode_roundtrip():
    N, K = 8, 512
    codes = rng.integers(-1, 2, (N, K)).astype(np.int8)
    d = np.abs(rng.standard_normal((N, K // 256))).astype(np.float32) + 0.01
    raw = _pack_tq2_0(codes, d)
    got_codes, got_d = decode_tq2_0(raw, N, K)
    np.testing.assert_array_equal(got_codes, codes)
    np.testing.assert_array_equal(got_d, d.astype(np.float16).astype(np.float32))


def test_baked_values_requantize_exactly():
    """The §8.2 argument, executed: bake per-tensor ternary -> TQ2_0's absmax
    re-quantization recovers identical codes and the identical f16 scale."""
    w = torch.randn(16, 512) * 0.03
    codes, scale = quant.ternary_codes(w, "tensor")
    baked = quant.dequant_codes(codes, scale).numpy()
    assert _is_ternary(baked)
    ref_codes, ref_d = encode_tq2_0_ref(baked)
    # zero-blocks aside, codes match the bake and every block scale is f16(s)
    np.testing.assert_array_equal(ref_codes, codes.numpy())
    s16 = float(scale.to(torch.float16).float())
    nz = ref_d[np.abs(baked).reshape(16, -1, 256).max(2) > 0]
    assert np.all(nz == np.float16(s16))
    # and the full circle: pack -> decode -> dequant == baked, error exactly 0
    raw = _pack_tq2_0(ref_codes, ref_d)
    got_codes, got_d = decode_tq2_0(raw, 16, 512)
    deq = got_codes.astype(np.float32) * np.repeat(got_d, 256, axis=1)
    np.testing.assert_array_equal(deq, baked)


def test_group_baking_breaks_tq2_parity():
    """The counter-case that justifies the per-tensor baseline: group-32 baked
    values put multiple magnitudes inside one 256-block, so one absmax scale
    cannot represent them — codes flip by construction (plan decision D2).
    Engineered 4x scale spread between groups (uniform randn groups land close
    enough in scale that round() hides the effect)."""
    w = torch.randn(4, 512) * 0.03
    w.reshape(4, 16, 32)[:, ::2, :] *= 4.0             # alternate groups 4x hotter
    codes, scale = quant.ternary_codes(w, "group", 32)
    baked = quant.dequant_codes(codes, scale, 32).numpy()
    ref_codes, _ = encode_tq2_0_ref(baked)
    assert (ref_codes != codes.numpy()).mean() > 0.05


def _pack_i2s(codes: np.ndarray, scale: float) -> np.ndarray:
    """Inverse of decode_i2s, mirroring the fork's quantize_i2_s packing loop:
    n/4 code bytes then a trailing f32 scale. code map {-1:0, 0:1, +1:2}."""
    n = codes.size
    q8 = np.where(codes == 0, 1, np.where(codes > 0, 2, 0)).astype(np.uint8).reshape(-1)
    nblk = n // 128
    bytes_out = np.zeros(nblk * 32, np.uint8)
    for g in range(4):
        seg = q8.reshape(nblk, 128)[:, g * 32:(g + 1) * 32]
        bytes_out.reshape(nblk, 32)[:] |= (seg << (6 - 2 * g)).astype(np.uint8)
    return np.concatenate([bytes_out, np.frombuffer(
        np.float32(scale).tobytes(), np.uint8)])


def test_i2s_pack_decode_roundtrip():
    from bitnet_train.export.compare_gguf import decode_i2s
    N, K = 8, 512                                       # n % 128 == 0
    codes = rng.integers(-1, 2, (N, K)).astype(np.int8)
    raw = _pack_i2s(codes, 0.037)
    got, s = decode_i2s(raw, N, K)
    np.testing.assert_array_equal(got, codes)
    assert abs(s - np.float32(0.037)) < 1e-9


def test_i2s_is_preserve_regime_for_baked():
    """Baked {-s,0,+s} -> I2_S re-quantization recovers codes AND scale exactly:
    the scale is the tensor absmax, which for baked ternary IS s (no F16 block
    rounding — the scale is per-tensor f32)."""
    from bitnet_train.export.compare_gguf import decode_i2s, encode_i2s_ref
    w = torch.randn(4, 512) * 0.03
    codes, scale = quant.ternary_codes(w, "tensor")
    baked = quant.dequant_codes(codes, scale).numpy()
    ref_codes, ref_s = encode_i2s_ref(baked)
    np.testing.assert_array_equal(ref_codes, codes.numpy())
    assert abs(ref_s - float(scale.to(torch.float16).float())) < 1e-6
    # full circle: pack -> decode -> dequant == baked exactly
    raw = _pack_i2s(ref_codes, ref_s)
    got, s = decode_i2s(raw, 4, 512)
    np.testing.assert_array_equal(got.astype(np.float32) * s, baked)


def test_bake_checkpoint_tiny(tmp_path):
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    from bitnet_train.conversion import convert, load_profile
    from bitnet_train.export.export_gguf import bake_checkpoint

    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
                      tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg)
    prof = load_profile(Path(__file__).resolve().parents[1] / "train" / "profiles"
                        / "ci_tiny.yaml")
    convert(model, prof, backend="reference")
    rep = bake_checkpoint(model, prof, tmp_path / "baked")
    assert len(rep.tensors) == 14
    assert (tmp_path / "baked" / "bake_report.json").exists()

    from bitnet_train.export.compare_gguf import load_baked_tensors
    baked = load_baked_tensors(tmp_path / "baked")
    tern = [n for n, v in baked.items() if v.ndim == 2 and _is_ternary(v)]
    assert len(tern) >= 14                             # every target came out ternary
