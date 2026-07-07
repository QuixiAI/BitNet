"""quant.py formulas vs train_plan §4 (with the canonical f16-scale delta)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train import quant  # noqa: E402

torch.manual_seed(0)


def test_weight_quant_pertensor_formula():
    w = torch.randn(64, 96) * 0.04
    got = quant.weight_quant(w, "tensor")
    s = w.abs().mean().clamp_min(1e-5)
    ref = (w / s).round().clamp(-1, 1) * s.to(torch.float16).float()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    vals = torch.unique((got / s.to(torch.float16).float()).round())
    assert set(vals.tolist()).issubset({-1.0, 0.0, 1.0})


def test_weight_quant_group_matches_pergroup():
    w = torch.randn(32, 128) * 0.05
    torch.testing.assert_close(quant.weight_quant(w, "group", 32),
                               quant.weight_quant_pergroup(w, 32), rtol=0, atol=0)


def test_activation_quant_formula():
    x = torch.randn(8, 64) * 3
    got = quant.activation_quant(x)
    s = x.abs().amax(-1, keepdim=True) / 127.0
    ref = (x / s).round().clamp(-127, 127) * s
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-7)
    assert quant.activation_quant(torch.zeros(2, 8)).abs().sum() == 0


@pytest.mark.parametrize("granularity", ["tensor", "group"])
def test_ternary_codes_roundtrip(granularity):
    w = torch.randn(48, 128) * 0.04
    codes, scale = quant.ternary_codes(w, granularity, 32)
    assert codes.dtype == torch.int8
    assert set(torch.unique(codes).tolist()).issubset({-1, 0, 1})
    deq = quant.dequant_codes(codes, scale, 32)
    torch.testing.assert_close(deq.to(w.dtype), quant.weight_quant(w, granularity, 32),
                               rtol=0, atol=0)


def test_lambda_ramp_endpoints():
    w = torch.randn(32, 64) * 0.05
    torch.testing.assert_close(quant.lambda_ramp(w, 0.0), w, rtol=0, atol=0)
    torch.testing.assert_close(quant.lambda_ramp(w, 1.0),
                               quant.weight_quant(w, "tensor"), rtol=0, atol=0)
    mid = quant.lambda_ramp(w, 0.5)
    torch.testing.assert_close(mid, 0.5 * w + 0.5 * quant.weight_quant(w, "tensor"),
                               rtol=0, atol=1e-7)


def test_quantizer_hash_stable():
    a, b = quant.quantizer_hash(), quant.quantizer_hash()
    assert a == b and len(a) == 16
