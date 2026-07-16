import pytest
import torch

from bitnet_train.tq1.shared_fallback import (
    PackedSharedFallbackEmbedding, PackedSharedFallbackOutputHead,
    SharedFallbackEmbedding,
    SharedFallbackOutputHead, SharedFallbackSpec, dequantize_shared_fallback,
    pack_shared_fallback, quantize_shared_fallback, unpack_shared_fallback)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_shared_fallback_pack_lookup_head_and_single_gradient(bits):
    torch.manual_seed(bits)
    spec = SharedFallbackSpec(bits)
    latent = torch.randn(19, 128)
    codes, scales = quantize_shared_fallback(latent, spec)
    packed = pack_shared_fallback(codes, scales, spec)
    assert torch.equal(unpack_shared_fallback(packed), codes)
    expected = dequantize_shared_fallback(codes, scales, spec)
    inference = PackedSharedFallbackEmbedding(packed)
    inference_head = PackedSharedFallbackOutputHead(inference)
    ids = torch.tensor([[5, 2, 5], [18, 0, 2]])
    torch.testing.assert_close(inference(ids), expected[ids])
    hidden = torch.randn(2, 128)
    torch.testing.assert_close(inference_head(hidden), hidden @ expected.T)
    assert list(inference_head.parameters()) == []
    assert inference_head.weight is inference

    shared = SharedFallbackEmbedding(latent, spec)
    head = SharedFallbackOutputHead(shared)
    loss = shared(ids).sum() + head(hidden).sum()
    loss.backward()
    assert shared.weight.grad is not None and torch.count_nonzero(shared.weight.grad)
    assert list(head.parameters()) == []
    assert head.weight is shared.weight


def test_shared_fallback_rejects_reserved_packed_code():
    spec = SharedFallbackSpec(2)
    codes = torch.zeros(1, 128, dtype=torch.int8)
    scales = torch.ones(1, 1, dtype=torch.float16)
    packed = pack_shared_fallback(codes, scales, spec)
    packed.payload[0, 0] = 0xFF
    with pytest.raises(ValueError, match="reserved"):
        unpack_shared_fallback(packed)
