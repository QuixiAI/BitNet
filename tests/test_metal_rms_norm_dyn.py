"""rms_norm at model widths: the templated path (1024) and the dynamic streaming
path (2048 = A1/Qwen3 hidden, 3072 = A3 hidden) must both match the torch reference."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bitnet_train" / "metal"))

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _ref(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


@pytest.mark.parametrize("D", [1024, 2048, 3072, 8192])
def test_rms_norm_widths(D):
    import tk_torch as tk
    torch.manual_seed(0)
    x = (torch.randn(64, D) * 2).to(torch.bfloat16).to("mps")
    w = (torch.randn(D) * 0.5 + 1).to(torch.bfloat16).to("mps")
    y = tk.rms_norm(x, w, 1e-5)
    torch.mps.synchronize()
    torch.testing.assert_close(y.float(), _ref(x, w, 1e-5).float(), rtol=2e-2, atol=1e-2)
