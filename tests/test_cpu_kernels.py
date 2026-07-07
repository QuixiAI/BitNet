"""K-track CPU engine kernels vs numpy oracles (moe_train_plan §7.3 discipline:
scalar reference is the permanent oracle; NEON diffs against scalar before trust)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bitnet_train" / "cpu"))

import bitnet_cpu as bn  # noqa: E402

rng = np.random.default_rng(0)


# ---- numpy packers (same oracles as the Metal tests) ----

def pack_bitnet(W, per_tensor=False):
    W = np.ascontiguousarray(W, np.float32)
    N, K = W.shape
    nb = K // 32
    Wb = W.reshape(N, nb, 32)
    if per_tensor:
        scale = np.full((N, nb), max(np.abs(W).mean(), 1e-5), np.float32)
    else:
        scale = np.maximum(np.abs(Wb).mean(axis=2), 1e-5).astype(np.float32)
    q = np.clip(np.rint(Wb / scale[..., None]), -1, 1).astype(np.int32)
    code = (q + 1).astype(np.uint32).reshape(N, nb, 8, 4)
    out = np.zeros((N, nb, 10), np.uint8)
    out[:, :, 0:2] = scale.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:10] = (code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4)
                       | (code[..., 3] << 6)).astype(np.uint8)
    deq = (scale.astype(np.float16).astype(np.float32)[..., None]
           * (q.astype(np.float32))).reshape(N, K)
    return out, deq


def ref_act_quant(x):
    s = np.abs(x).max() / 127.0
    q = np.clip(np.rint(x / s), -127, 127) if s > 0 else np.zeros_like(x)
    return q.astype(np.int8), np.float32(s)


def test_act_quant():
    x = rng.standard_normal(512).astype(np.float32)
    xq, s = bn.act_quant_int8(x)
    q_ref, s_ref = ref_act_quant(x)
    assert np.isclose(s, s_ref)
    np.testing.assert_array_equal(xq, q_ref)


@pytest.mark.parametrize("pt", [False, True])
@pytest.mark.parametrize("impl", ["scalar", "neon"])
def test_gemv_w2a8(pt, impl):
    N, K = 96, 256
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    wq, w_deq = pack_bitnet(W, per_tensor=pt)
    x = rng.standard_normal(K).astype(np.float32)
    xq, s = bn.act_quant_int8(x)
    y = bn.gemv_w2a8(wq, xq, s, pt=pt, impl=impl)
    ref = (w_deq @ (xq.astype(np.float32) * s)).astype(np.float32)
    np.testing.assert_allclose(y, ref, rtol=2e-4, atol=1e-4)


def test_gemv_w2a8_neon_equals_scalar():
    N, K = 64, 512
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    wq, _ = pack_bitnet(W)
    xq = rng.integers(-127, 128, K).astype(np.int8)
    a = bn.gemv_w2a8(wq, xq, 0.013, impl="scalar")
    b = bn.gemv_w2a8(wq, xq, 0.013, impl="neon")
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_route_topk():
    E, k = 128, 8
    logits = rng.standard_normal(E).astype(np.float32) * 3
    ids, w = bn.route_topk(logits, k)
    p = np.exp(logits - logits.max()); p /= p.sum()
    ref_ids = np.argsort(-p)[:k]
    assert set(ids.tolist()) == set(ref_ids.tolist())
    ref_w = p[ids] / p[ids].sum()
    np.testing.assert_allclose(w, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("pt", [False, True])
def test_expert_ffn(pt):
    H, I = 128, 96
    Wg = rng.standard_normal((I, H)).astype(np.float32) * 0.1
    Wu = rng.standard_normal((I, H)).astype(np.float32) * 0.1
    Wd = rng.standard_normal((H, I)).astype(np.float32) * 0.1
    gq, g_deq = pack_bitnet(Wg, pt); uq, u_deq = pack_bitnet(Wu, pt)
    dq, d_deq = pack_bitnet(Wd, pt)
    x = rng.standard_normal(H).astype(np.float32) * 0.5
    y = bn.expert_ffn_w2a8(x, gq, uq, dq, w_r=0.7, pt=pt)

    xq, xs = ref_act_quant(x)
    xf = xq.astype(np.float32) * xs
    g = g_deq @ xf; u = u_deq @ xf
    h = (g / (1 + np.exp(-g))) * u
    hq, hs = ref_act_quant(h)
    ref = 0.7 * (d_deq @ (hq.astype(np.float32) * hs))
    np.testing.assert_allclose(y, ref, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("pt", [False, True])
def test_expert_ffn_tl1_matches_format_a(pt):
    """The bakeoff-winner FFN (formats_bakeoff.md follow-up) must equal the
    format-A fused FFN — same packed source, TL1 tiles vs 2-bit blocks. I,H %16."""
    H, I = 128, 96
    Wg = rng.standard_normal((I, H)).astype(np.float32) * 0.1
    Wu = rng.standard_normal((I, H)).astype(np.float32) * 0.1
    Wd = rng.standard_normal((H, I)).astype(np.float32) * 0.1
    gq, _ = pack_bitnet(Wg, pt); uq, _ = pack_bitnet(Wu, pt); dq, _ = pack_bitnet(Wd, pt)
    x = rng.standard_normal(H).astype(np.float32) * 0.5
    ref = bn.expert_ffn_w2a8(x, gq, uq, dq, w_r=0.7, pt=pt)
    y = bn.expert_ffn_tl1(x, bn.pack_tl1(gq), bn.pack_tl1(uq), bn.pack_tl1(dq),
                          w_r=0.7, pt=pt)
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


def test_moe_ffn_matches_expert_loop():
    H, I, E, k = 128, 96, 8, 2
    stacks = []
    for rows, cols in ((I, H), (I, H), (H, I)):
        Ws = rng.standard_normal((E, rows, cols)).astype(np.float32) * 0.1
        packed = np.stack([pack_bitnet(Ws[e])[0] for e in range(E)])
        stacks.append((Ws, packed))
    (_, gq), (_, uq), (_, dq) = stacks
    x = rng.standard_normal(H).astype(np.float32) * 0.5
    logits = rng.standard_normal(E).astype(np.float32)
    ids, w = bn.route_topk(logits, k)
    y = bn.moe_ffn_w2a8(x, gq, uq, dq, ids, w)
    ref = np.zeros(H, np.float32)
    for j in range(k):
        e = int(ids[j])
        bn.expert_ffn_w2a8(x, gq[e], uq[e], dq[e], w_r=float(w[j]), out=ref)
    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("impl", ["scalar", "neon"])
def test_gemv_fp8(impl):
    N, K = 64, 256
    W = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    scale = np.abs(W).max(axis=1) / 448.0
    codes = np.zeros((N, K), np.uint8)
    # encode via the library's own LUT inverse: nearest-code search (test-only, small)
    lut = np.array([_e4m3(b) for b in range(256)], np.float32)
    for n in range(N):
        codes[n] = np.abs(W[n][:, None] / scale[n] - lut[None, :]).argmin(axis=1)
    x = rng.standard_normal(K).astype(np.float32)
    y = bn.gemv_fp8(codes, scale.astype(np.float32), x, impl=impl)
    ref = (lut[codes] * scale[:, None]) @ x
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


def _e4m3(b):
    s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 7
    if e == 0:
        v = m * 2.0 ** -9
    elif e == 15 and m == 7:
        v = 0.0
    else:
        v = (1 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


def test_attn_decode_kv8():
    T, Hq, Hkv, D = 40, 8, 2, 64
    q = rng.standard_normal((Hq, D)).astype(np.float32)
    K_ = rng.standard_normal((T, Hkv, D)).astype(np.float32)
    V = rng.standard_normal((T, Hkv, D)).astype(np.float32)
    ks = (np.abs(K_).max(axis=2) / 127.0).astype(np.float32)
    vs = (np.abs(V).max(axis=2) / 127.0).astype(np.float32)
    kc = np.clip(np.rint(K_ / ks[..., None]), -127, 127).astype(np.int8)
    vc = np.clip(np.rint(V / vs[..., None]), -127, 127).astype(np.int8)
    y = bn.attn_decode_kv8(q, kc, ks, vc, vs)

    kd = kc.astype(np.float32) * ks[..., None]
    vd = vc.astype(np.float32) * vs[..., None]
    rep = Hq // Hkv
    for h in range(Hq):
        hk = h // rep
        sc = (kd[:, hk] @ q[h]) / np.sqrt(D)
        p = np.exp(sc - sc.max()); p /= p.sum()
        ref = p @ vd[:, hk]
        np.testing.assert_allclose(y[h], ref, rtol=2e-4, atol=2e-5)


def test_unpack_and_prefill():
    N, K, M = 96, 256, 33
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    wq, w_deq = pack_bitnet(W)
    np.testing.assert_allclose(bn.unpack_ternary_f32(wq), w_deq, rtol=0, atol=0)
    X = rng.standard_normal((M, K)).astype(np.float32)
    np.testing.assert_allclose(bn.prefill_ternary(wq, X), X @ w_deq.T, rtol=1e-5, atol=1e-5)


# ---- decode glue ----

def test_rms_norm():
    R, D = 8, 64
    x = rng.standard_normal((R, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    y = bn.rms_norm(x, w, eps=1e-6)
    ref = x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-6) * w
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-6)


def test_rope_neox():
    H, D, pos, theta = 4, 64, 37, 1e6
    x = rng.standard_normal((H, D)).astype(np.float32)
    y = bn.rope_neox(x.copy(), pos, theta)
    # HF apply_rotary_pos_emb: q*cos + rotate_half(q)*sin, inv_freq = theta^(-2i/D)
    inv = theta ** (-np.arange(0, D, 2) / D)
    ang = pos * inv
    cos = np.concatenate([np.cos(ang), np.cos(ang)])
    sin = np.concatenate([np.sin(ang), np.sin(ang)])
    rot = np.concatenate([-x[:, D // 2:], x[:, :D // 2]], axis=1)
    ref = x * cos + rot * sin
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)


def test_kv_quant_append_roundtrip():
    T, Hq, Hkv, D = 12, 8, 2, 64
    kc = np.zeros((T, Hkv, D), np.int8); vc = np.zeros((T, Hkv, D), np.int8)
    ks = np.zeros((T, Hkv), np.float32); vs = np.zeros((T, Hkv), np.float32)
    K_ = rng.standard_normal((T, Hkv, D)).astype(np.float32)
    V = rng.standard_normal((T, Hkv, D)).astype(np.float32)
    for t in range(T):
        bn.kv_quant_append(K_[t], V[t], t, kc, ks, vc, vs)
    # per-(token, head) absmax convention, matching the attn kernel's read side
    ref_ks = np.abs(K_).max(axis=2) / 127.0
    np.testing.assert_allclose(ks, ref_ks, rtol=1e-6, atol=1e-7)
    ref_kc = np.clip(np.rint(K_ / ref_ks[..., None]), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(kc, ref_kc)
    # and the written cache must feed the attention kernel unchanged
    q = rng.standard_normal((Hq, D)).astype(np.float32)
    y = bn.attn_decode_kv8(q, kc, ks, vc, vs)
    kd = kc.astype(np.float32) * ks[..., None]
    vd = vc.astype(np.float32) * vs[..., None]
    rep = Hq // Hkv
    for h in range(Hq):
        hk = h // rep
        sc = (kd[:, hk] @ q[h]) / np.sqrt(D)
        p = np.exp(sc - sc.max()); p /= p.sum()
        np.testing.assert_allclose(y[h], p @ vd[:, hk], rtol=2e-4, atol=2e-5)


# ---- head GEMVs ----

@pytest.mark.parametrize("impl", ["scalar", "neon"])
def test_gemv_q8(impl):
    N, K = 64, 256
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    wq, w_deq = bn.pack_q8(W)
    x = rng.standard_normal(K).astype(np.float32)
    xq, s = bn.act_quant_int8(x)
    y = bn.gemv_q8(wq, xq, s, impl=impl)
    ref = w_deq @ (xq.astype(np.float32) * s)
    np.testing.assert_allclose(y, ref, rtol=2e-4, atol=1e-4)


@pytest.mark.parametrize("impl", ["scalar", "neon"])
def test_gemv_bf16(impl):
    N, K = 64, 259                                   # odd K exercises the NEON tail
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    wb = (W.view(np.uint32) >> 16).astype(np.uint16)  # truncate-to-bf16 codes
    w_deq = (wb.astype(np.uint32) << 16).view(np.float32)
    x = rng.standard_normal(K).astype(np.float32)
    y = bn.gemv_bf16(wb, x, impl=impl)
    np.testing.assert_allclose(y, w_deq @ x, rtol=1e-5, atol=1e-5)


# ---- packing bake-off formats ----

@pytest.mark.parametrize("pt", [False, True])
def test_gemv_b3(pt):
    N, K = 96, 256
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    wq, _ = pack_bitnet(W, per_tensor=pt)
    wb = bn.pack_b3(wq)
    assert wb.nbytes == N * (K // 32) * 9
    xq = rng.integers(-127, 128, K).astype(np.int8)
    ref = bn.gemv_w2a8(wq, xq, 0.017, pt=pt, impl="scalar")
    y = bn.gemv_b3(wb, xq, 0.017, pt=pt)
    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("pt", [False, True])
@pytest.mark.parametrize("impl", ["scalar", "neon"])
def test_gemv_tl1(pt, impl):
    N, K = 96, 256
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    wq, _ = pack_bitnet(W, per_tensor=pt)
    wt = bn.pack_tl1(wq)
    assert wt.nbytes == wq.nbytes                     # same 2.5 b/w traffic as format A
    xq = rng.integers(-127, 128, K).astype(np.int8)
    ref = bn.gemv_w2a8(wq, xq, 0.017, pt=pt, impl="scalar")
    y = bn.gemv_tl1(wt, xq, 0.017, pt=pt, impl=impl)
    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)
