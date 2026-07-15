import torch

from bitnet_train.tq1.codebook import sign_canonical_codebook
from bitnet_train.tq1.oracle import linear_w2a8
from bitnet_train.tq1.packing import pack_payload
from bitnet_train.tq1.runtime import PackedTQ1Linear
from bitnet_train.tq1.solver import canonical_shapes


def test_packed_scalar_module_matches_oracle_and_modes():
    shapes = canonical_shapes()
    book = sign_canonical_codebook("runtime", "v11", torch.cat((
        shapes[(shapes == 0).all(1)], shapes[~(shapes == 0).all(1)][:1023])))
    torch.manual_seed(91)
    indices = torch.randint(0, 2048, (5, 64))
    indices[indices == 1024] = 0
    payload = pack_payload(indices, "tq1_v11-j-r")
    scales = torch.rand(5, dtype=torch.float16)
    x = torch.randn(2, 3, 512)
    module = PackedTQ1Linear(
        payload, "tq1_v11-j-r", book, row_scales=scales,
        activation_mode="a8_block256", state_dict_name="x.weight")
    got = module(x)
    expected = linear_w2a8(
        x, payload, "tq1_v11-j-r", book, row_scales=scales,
        activation_mode="a8_block256")
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    assert module.payload_sha256 and module.codebook_sha256 == book.sha256()
