"""TQ2_0 in the vendored Metal stack (no external runtime): quantize_tq2_0 must
be BYTE-EXACT against the numpy transcription of ggml's quantize_row_tq2_0_ref
(the oracle export/compare_gguf and the T0 parity gate already trust), and the
tq2_0 dequant struct must feed qdequant/qgemv/qgemm correctly."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

from bitnet_train import quant  # noqa: E402
from bitnet_train.export.compare_gguf import decode_tq2_0, encode_tq2_0_ref  # noqa: E402
from test_export_tensor_parity import _pack_tq2_0  # noqa: E402


def _tk():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bitnet_train" / "metal"))
    import tk_torch
    return tk_torch


def _oracle_bytes(Wf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes, d = encode_tq2_0_ref(Wf)
    return _pack_tq2_0(codes, d), codes, d


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantize_matches_ggml_oracle_bytes(dtype):
    tk = _tk()
    torch.manual_seed(0)
    N, K = 64, 512
    W = (torch.randn(N, K) * 0.05).to(dtype).to("mps")
    wq, w_deq = tk.quantize_tq2_0(W)
    torch.mps.synchronize()
    assert wq.shape == (N, K // 256, 66) and w_deq.shape == (N, K)

    raw_ref, codes, d = _oracle_bytes(W.float().cpu().numpy())
    np.testing.assert_array_equal(wq.cpu().numpy().reshape(-1), raw_ref)   # byte-exact
    deq_ref = codes.astype(np.float32) * np.repeat(d, 256, axis=1)
    np.testing.assert_allclose(w_deq.float().cpu().numpy(), deq_ref,
                               rtol=1 / 128, atol=1e-6)                    # bf16 of exact


def test_baked_ternary_packs_exactly():
    """The §8.2 preserve regime, now fully in-repo: per-tensor-baked {-s,0,+s}
    -> on-device TQ2_0 pack == the exporter's re-quantization, bit-for-bit."""
    tk = _tk()
    torch.manual_seed(1)
    W = torch.randn(32, 512) * 0.03
    codes, scale = quant.ternary_codes(W, "tensor")
    baked = quant.dequant_codes(codes, scale)
    wq, _ = tk.quantize_tq2_0(baked.to("mps"))
    torch.mps.synchronize()
    raw_ref, ref_codes, _ = _oracle_bytes(baked.numpy())
    np.testing.assert_array_equal(wq.cpu().numpy().reshape(-1), raw_ref)
    np.testing.assert_array_equal(ref_codes, codes.numpy())


def test_expert_stack_batched_matches_per_expert():
    tk = _tk()
    torch.manual_seed(2)
    E, N, K = 3, 32, 256
    W = (torch.randn(E, N, K) * 0.05).to("mps")
    wq3, wd3 = tk.quantize_tq2_0(W)
    torch.mps.synchronize()
    assert wq3.shape == (E, N, 1, 66)
    for e in range(E):
        wq2, wd2 = tk.quantize_tq2_0(W[e].contiguous())
        torch.mps.synchronize()
        assert torch.equal(wq3[e], wq2)
        torch.testing.assert_close(wd3[e], wd2, rtol=0, atol=0)


def test_dequantize_roundtrip():
    tk = _tk()
    torch.manual_seed(3)
    W = (torch.randn(48, 512) * 0.05).to("mps")
    wq, w_deq = tk.quantize_tq2_0(W)
    w = tk.dequantize_tq2_0(wq)
    torch.mps.synchronize()
    # fp16 dense out vs the numpy decode of the same bytes: exact
    got_codes, got_d = decode_tq2_0(wq.cpu().numpy().reshape(-1), 48, 512)
    ref = got_codes.astype(np.float32) * np.repeat(got_d, 256, axis=1)
    np.testing.assert_array_equal(w.float().cpu().numpy(), ref)
    torch.testing.assert_close(w.float(), w_deq.float().to("mps"), rtol=1 / 128, atol=1e-6)


def test_qgemv_and_qgemm_match_dense():
    tk = _tk()
    torch.manual_seed(4)
    N, K = 64, 512
    W = (torch.randn(N, K) * 0.05).to("mps")
    wq, w_deq = tk.quantize_tq2_0(W)
    wd = w_deq.float()

    x1 = (torch.randn(K, 1) * 0.7).half().to("mps")
    y = tk.qgemv(wq, x1, "tq2_0")
    torch.mps.synchronize()
    ref = wd @ x1.float()
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=1e-2)  # fp16 dot @ K=512

    for M in (32, 64):                       # fragment route and dequant+GEMM route
        x = (torch.randn(K, M) * 0.7).half().to("mps")
        y = tk.qgemm(wq, x, "tq2_0")
        torch.mps.synchronize()
        ref = wd @ x.float()
        torch.testing.assert_close(y.float(), ref, rtol=2e-3, atol=2e-2)
