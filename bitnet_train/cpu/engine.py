"""K-track CPU decode engine (moe_train_plan §7.3, §8.2 Q-K0): assemble a baked
checkpoint into the bn_* kernels and run a full batch-1 decode loop, so decode
tok/s and TTFT can be measured on real weights (the Q-K0 gate's instrument).

This is the ENGINE project (§0.3): the model artifact never depends on it. It
loads the exported/baked HF checkpoint (attention/embeddings FP, experts
ternary), packs experts into the chosen format (bitnet A / TL1 C), and steps a
Qwen3-MoE-shaped decode: embed -> [RMSNorm, QK-norm, RoPE, int8-KV attention,
router, ternary MoE FFN] x L -> final norm -> head. Attention/embeddings run in
numpy fp32 here (the FP8 pack is a K-track detail); the ternary EXPERT path —
the memory story — goes through the kernels.

Scope: correctness + timing on small models; the multicore dispatch and FP8
packing are named-but-separate digs (§7.3). Requires numpy + a baked HF dir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from bitnet_train.cpu import bitnet_cpu as bn


def pack_bitnet_np(W, per_tensor=True):
    W = np.ascontiguousarray(W, np.float32)
    N, K = W.shape
    nb = K // 32
    Wb = W.reshape(N, nb, 32)
    scale = (np.full((N, nb), max(np.abs(W).mean(), 1e-5), np.float32) if per_tensor
             else np.maximum(np.abs(Wb).mean(axis=2), 1e-5).astype(np.float32))
    q = np.clip(np.rint(Wb / scale[..., None]), -1, 1).astype(np.int32)
    code = (q + 1).astype(np.uint32).reshape(N, nb, 8, 4)
    out = np.zeros((N, nb, 10), np.uint8)
    out[:, :, 0:2] = scale.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:10] = (code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4)
                       | (code[..., 3] << 6)).astype(np.uint8)
    return out


@dataclass
class PackedExpert:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray


class CPUEngine:
    """Batch-1 Qwen3-MoE decode over ternary experts. fmt: 'bitnet' | 'tl1'."""

    def __init__(self, hf_dir, fmt: str = "bitnet", pt: bool = True):
        from transformers import AutoConfig, AutoModelForCausalLM
        bn._lib.bn_init()
        self.cfg = AutoConfig.from_pretrained(hf_dir)
        model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype="float32")
        self.fmt, self.pt = fmt, pt
        self._load(model)

    def _pack(self, W):
        a = pack_bitnet_np(W, per_tensor=self.pt)
        return bn.pack_tl1(a) if self.fmt == "tl1" else a

    def _load(self, model):
        import torch
        sd = model.state_dict()
        g = lambda k: sd[k].float().numpy()
        c = self.cfg
        self.embed = g("model.embed_tokens.weight")
        self.L = c.num_hidden_layers
        self.H = c.hidden_size
        self.n_head = c.num_attention_heads
        self.n_kv = c.num_key_value_heads
        self.hd = getattr(c, "head_dim", self.H // self.n_head)
        self.theta = float(getattr(c, "rope_theta", 1e6))
        self.eps = float(getattr(c, "rms_norm_eps", 1e-6))
        self.k = c.num_experts_per_tok
        self.E = c.num_experts
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            # transformers-5 fused experts, already (out, in) like a Linear weight:
            # gate_up_proj (E, 2I, H), down_proj (E, H, I) — no transpose needed.
            gu = g(p + "mlp.experts.gate_up_proj")
            dn = g(p + "mlp.experts.down_proj")
            I = gu.shape[1] // 2
            packed = [PackedExpert(self._pack(gu[e, :I]), self._pack(gu[e, I:]),
                                   self._pack(dn[e])) for e in range(self.E)]
            layers = {
                "in_norm": g(p + "input_layernorm.weight"),
                "post_norm": g(p + "post_attention_layernorm.weight"),
                "q": g(p + "self_attn.q_proj.weight"),
                "k": g(p + "self_attn.k_proj.weight"),
                "v": g(p + "self_attn.v_proj.weight"),
                "o": g(p + "self_attn.o_proj.weight"),
                "q_norm": g(p + "self_attn.q_norm.weight") if p + "self_attn.q_norm.weight" in sd else None,
                "k_norm": g(p + "self_attn.k_norm.weight") if p + "self_attn.k_norm.weight" in sd else None,
                "router": g(p + "mlp.gate.weight"),
                "experts": packed, "I": I,
            }
            self.layers.append(layers)
        self.final_norm = g("model.norm.weight")
        self.head = g("lm_head.weight") if "lm_head.weight" in sd else self.embed

    def _expert_ffn(self, x, pe: PackedExpert, w_r, out):
        if self.fmt == "tl1":
            bn.expert_ffn_tl1(x, pe.gate, pe.up, pe.down, w_r=w_r, pt=self.pt, out=out)
        else:
            bn.expert_ffn_w2a8(x, pe.gate, pe.up, pe.down, w_r=w_r, pt=self.pt, out=out)

    def step(self, tok_id: int, pos: int, kv):
        """One decode step; kv is the per-layer int8 KV cache state. Returns logits."""
        h = self.embed[tok_id].copy()
        for li, L in enumerate(self.layers):
            xn = bn.rms_norm(h[None, :], L["in_norm"], self.eps)[0]
            q = (L["q"] @ xn).reshape(self.n_head, self.hd)
            kk = (L["k"] @ xn).reshape(self.n_kv, self.hd)
            vv = (L["v"] @ xn).reshape(self.n_kv, self.hd)
            if L["q_norm"] is not None:
                q = bn.rms_norm(q, L["q_norm"], self.eps)
                kk = bn.rms_norm(kk, L["k_norm"], self.eps)
            bn.rope_neox(q, pos, self.theta)
            bn.rope_neox(kk, pos, self.theta)
            kc, ks, vc, vs = kv[li]
            bn.kv_quant_append(kk, vv, pos, kc, ks, vc, vs)
            attn = bn.attn_decode_kv8(q, kc[:pos + 1], ks[:pos + 1],
                                      vc[:pos + 1], vs[:pos + 1])
            h = h + L["o"] @ attn.reshape(-1)
            xn = bn.rms_norm(h[None, :], L["post_norm"], self.eps)[0]
            logits = L["router"] @ xn
            ids, w = bn.route_topk(logits, self.k)
            ff = np.zeros(self.H, np.float32)
            for j in range(self.k):
                self._expert_ffn(xn, L["experts"][ids[j]], float(w[j]), ff)
            h = h + ff
        h = bn.rms_norm(h[None, :], self.final_norm, self.eps)[0]
        return self.head @ h

    def new_kv(self, max_len: int):
        return [(np.zeros((max_len, self.n_kv, self.hd), np.int8),
                 np.zeros((max_len, self.n_kv), np.float32),
                 np.zeros((max_len, self.n_kv, self.hd), np.int8),
                 np.zeros((max_len, self.n_kv), np.float32)) for _ in range(self.L)]

    def generate(self, prompt_ids, max_new_tokens=32, greedy=True):
        kv = self.new_kv(len(prompt_ids) + max_new_tokens)
        pos = 0
        for t in prompt_ids[:-1]:
            self.step(int(t), pos, kv)
            pos += 1
        cur = int(prompt_ids[-1])
        out = []
        for _ in range(max_new_tokens):
            logits = self.step(cur, pos, kv)
            pos += 1
            cur = int(logits.argmax()) if greedy else int(
                np.random.choice(len(logits), p=_softmax(logits)))
            out.append(cur)
        return out


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def bench(engine: CPUEngine, prompt_len=64, gen=32) -> dict:
    """Q-K0 numbers: TTFT (prefill of prompt_len) and decode tok/s."""
    ids = list(np.random.randint(0, engine.head.shape[0], prompt_len))
    kv = engine.new_kv(prompt_len + gen)
    t0 = time.perf_counter()
    for pos, t in enumerate(ids):
        engine.step(int(t), pos, kv)
    ttft = time.perf_counter() - t0
    cur, pos = ids[-1], prompt_len
    t0 = time.perf_counter()
    for _ in range(gen):
        logits = engine.step(cur, pos, kv)
        cur = int(logits.argmax())
        pos += 1
    dt = time.perf_counter() - t0
    return {"format": engine.fmt, "ttft_s": ttft, "prompt_len": prompt_len,
            "decode_tok_s": gen / dt, "decode_ms_per_tok": dt / gen * 1e3}
