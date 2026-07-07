"""BitLinear reference-backend forward + eval-mode/act_quant flag (CPU, no MPS)."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train import quant  # noqa: E402
from bitnet_train.bitlinear import BitLinear, set_eval_mode  # noqa: E402

torch.manual_seed(0)


@pytest.mark.parametrize("granularity", ["tensor", "group"])
def test_forward_matches_manual_ste(granularity):
    lin = BitLinear(64, 32, backend="reference", granularity=granularity)
    x = torch.randn(4, 64)
    y = lin(x)
    w_q = quant.weight_quant(lin.weight, granularity, lin.group_k)
    x_q = quant.activation_quant(x)
    torch.testing.assert_close(y, F.linear(x_q, w_q), rtol=1e-5, atol=1e-6)


def test_act_quant_flag_w_only():
    lin = BitLinear(64, 32, backend="reference", granularity="tensor")
    x = torch.randn(4, 64)
    y_a8 = lin(x)
    lin.act_quant = False
    y_wonly = lin(x)
    w_q = quant.weight_quant(lin.weight, "tensor")
    torch.testing.assert_close(y_wonly, F.linear(x, w_q), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(y_a8, y_wonly)


def test_set_eval_mode_flips_all_layers():
    model = torch.nn.Sequential(BitLinear(32, 32, backend="reference"),
                                BitLinear(32, 32, backend="reference"))
    set_eval_mode(model, "w_only")
    assert all(not m.act_quant for m in model)
    set_eval_mode(model, "w_a8")
    assert all(m.act_quant for m in model)


def test_leading_dims_and_repr():
    lin = BitLinear(64, 48, backend="reference")
    y = lin(torch.randn(2, 3, 64))
    assert y.shape == (2, 3, 48)
    assert "bias=False" in repr(lin)
