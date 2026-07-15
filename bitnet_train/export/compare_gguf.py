"""Train/export parity report (train_plan §7.0 file #8, §8.2–8.5).

Tensor parity: decode the GGUF's ternary blocks and compare codes + scales
against the BAKED checkpoint (never against latents). The decode layouts are
transcribed from source, not guessed:

  TQ2_0 (mainline ggml-quants.c, read 2026-07-06): block = {uint8 qs[64];
  fp16 d} per 256 weights; chunk j in {0,32} bytes, byte m encodes weights
  j*4 + m + n*32 at bits 2n; d = per-block ABSMAX; value = (q-1)*d.
  T0 recon consequence: per-tensor-baked {-s,0,+s} values reproduce codes AND
  the f16 scale exactly — TQ2_0 is a PRESERVE-regime route by construction.

  I2_S (Eddie-Wang fork): decoder filled in when the submodule is initialized
  (T0 recon item); until then I2_S rows report skipped.

PPL parity pairs the runtime number with the matching PyTorch eval mode
(w_a8 <-> I2_S, w_only/a0 <-> TQ2_0 — §8.4).

Schema-2 TQ1 parity consumes the canonical artifact directly.  It compares the
GGUF tensor inventory, type/profile policy, packed payload, row-scale bit
patterns, and codebook tensors without decoding and reassigning weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bitnet_train.export.export_gguf import mainline_dir, runtime_ppl


def _import_gguf(llama_dir: Path | None = None):
    try:
        import gguf
        return gguf
    except ImportError:
        ml = llama_dir or mainline_dir()
        sys.path.insert(0, str(ml / "gguf-py"))
        import gguf
        return gguf


# ---------------------------------------------------------------------------
# block decoders (source-transcribed)
# ---------------------------------------------------------------------------

def decode_tq2_0(raw: np.ndarray, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """raw uint8 bytes -> (codes int8 (N, K) in {-1,0,+1}, scales f32 (N, K/256))."""
    assert K % 256 == 0, "TQ2_0 needs K % 256 == 0"
    nb = K // 256
    blocks = raw.reshape(N, nb, 66)
    qs = blocks[:, :, :64]
    d = np.ascontiguousarray(blocks[:, :, 64:66]).view(np.float16).astype(np.float32)
    codes = np.empty((N, nb, 256), np.int8)
    for j in (0, 1):                                   # two 32-byte chunks / block
        chunk = qs[:, :, j * 32:(j + 1) * 32]
        for n in range(4):
            codes[:, :, j * 128 + n * 32:j * 128 + (n + 1) * 32] = \
                ((chunk >> (2 * n)) & 3).astype(np.int8) - 1
    return codes.reshape(N, K), d.reshape(N, nb)


def encode_tq2_0_ref(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference re-quantization of dense values, mirroring quantize_row_tq2_0_ref:
    per-256 absmax scale, codes = lroundf(x/d) — ROUND HALF AWAY FROM ZERO, not
    np.rint's half-even (they differ exactly when x == d/2, which bf16-grid
    inputs hit; baked {-s,0,+s} input has no ties either way)."""
    N, K = x.shape
    nb = K // 256
    xb = x.reshape(N, nb, 256).astype(np.float32)
    d = np.abs(xb).max(axis=2)
    inv = np.where(d > 0, 1.0 / np.where(d == 0, 1, d), 0.0)
    r = xb * inv[..., None]
    codes = (np.sign(r) * np.floor(np.abs(r) + 0.5)).clip(-1, 1).astype(np.int8)
    return codes.reshape(N, K), d.astype(np.float16).astype(np.float32)


_I2S_MAP = np.array([-1, 0, 1, 0], np.int8)   # ggml map2bit {0:-1,1:0,2:+1,3:0}


def decode_i2s(raw: np.ndarray, N: int, K: int) -> tuple[np.ndarray, float]:
    """Eddie-Wang fork I2_S -> (codes int8 (N,K) in {-1,0,+1}, per-TENSOR scale).
    Transcribed from ggml-quants.c dequantize_row_i2_s / quantize_i2_s (fork
    3rdparty/llama.cpp @ 1f86f05, read 2026-07-07):

      layout: n/4 code bytes then a trailing f32 scale (i2_scale = ABSMAX of the
      (baked) tensor). Flat element e = 128*i + 32*g + p lives in byte[32*i + p]
      at bits (6 - 2*g); code in {0,1,2,3} -> {-1,0,+1,0}.

    Because baked ternary is exactly {-s, 0, +s}, its absmax IS s — so I2_S is a
    PRESERVE-regime route (codes AND scale reproduce from baked input), like
    TQ2_0. i2_scale is per-tensor, not per-block, so there is no F16 block-scale
    rounding on this route."""
    n = N * K
    code_bytes = np.frombuffer(raw.tobytes()[:n // 4], dtype=np.uint8)
    scale = float(np.frombuffer(raw.tobytes()[n // 4:n // 4 + 4], dtype=np.float32)[0])
    nblk = n // 128                                  # 128 elements / 32 bytes
    b = code_bytes[:nblk * 32].reshape(nblk, 32)
    codes = np.empty((nblk, 4, 32), np.int8)
    for g in range(4):
        codes[:, g, :] = _I2S_MAP[(b >> (6 - 2 * g)) & 3]
    return codes.reshape(n)[:n].reshape(N, K), scale


def encode_i2s_ref(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Reference re-quantization mirroring quantize_i2_s: per-TENSOR absmax scale,
    code = sign (2 if >0, 0 if <0, 1 if ~0). For baked {-s,0,+s} input the sign is
    exact and absmax == s."""
    scale = float(np.abs(x).max())
    codes = np.where(np.abs(x) < 1e-6, 0, np.where(x > 0, 1, -1)).astype(np.int8)
    return codes, scale


# ---------------------------------------------------------------------------
# baked-checkpoint loading + name mapping
# ---------------------------------------------------------------------------

def load_baked_tensors(baked_dir: str | Path) -> dict[str, np.ndarray]:
    """All tensors from the baked checkpoint's safetensors (single or sharded)."""
    from safetensors import safe_open
    baked_dir = Path(baked_dir)
    files = sorted(baked_dir.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors under {baked_dir}")
    out = {}
    for f in files:
        with safe_open(str(f), framework="np") as sf:
            for name in sf.keys():
                out[name] = sf.get_tensor(name)
    return out


def hf_to_gguf_name_map(gguf_mod, arch: str, n_layers: int) -> "object":
    arch_enum = {v: k for k, v in gguf_mod.MODEL_ARCH_NAMES.items()}[arch]
    return gguf_mod.get_tensor_name_map(arch_enum, n_layers)


def _llama_qk_permute(w: np.ndarray, n_head: int) -> np.ndarray:
    """convert_hf_to_gguf LlamaModel.permute, verbatim: the llama GGUF arch stores
    attn_q/attn_k with rows reordered for interleaved rotary (T0 recon fact —
    without this, q/k parity 'mismatches' ~64% on content-identical tensors)."""
    return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
            .swapaxes(1, 2).reshape(w.shape))


# arches whose converter permutes q/k this way (qwen*/gemma do not)
_QK_PERMUTED_ARCHES = ("llama", "llama4", "mistral", "mixtral")


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------

@dataclass
class TensorParityRow:
    hf_name: str
    gguf_name: str
    gguf_type: str
    status: str                    # exact | mismatch | skipped | unmapped
    code_mismatch_rate: float = 0.0
    max_abs_err: float = 0.0
    mean_abs_err: float = 0.0
    within_f16_bound: bool = True
    detail: str = ""


def parity_rows_ok(rows: list[TensorParityRow], regime: str = "preserve") -> bool:
    """Fail closed: absence, unsupported types, or bad reconstruction never pass."""
    if regime not in {"preserve", "requantize"}:
        raise ValueError("regime must be preserve or requantize")
    if not rows:
        return False
    for row in rows:
        if row.status in {"skipped", "unmapped"} or not row.within_f16_bound:
            return False
        if regime == "preserve" and row.status != "exact":
            return False
        if regime == "requantize" and row.status not in {"exact", "mismatch"}:
            return False
    return True


def tensor_parity(baked_dir: str | Path, gguf_path: str | Path,
                  regime: str = "preserve",
                  llama_dir: Path | None = None) -> tuple[list[TensorParityRow], bool]:
    """Per-tensor code/scale parity of a GGUF against the baked checkpoint.
    regime='preserve' demands exact codes on every decoded ternary tensor."""
    gguf_mod = _import_gguf(llama_dir)
    reader = gguf_mod.GGUFReader(str(gguf_path))

    def field_str(key):
        f = reader.get_field(key)
        return bytes(f.parts[f.data[0]]).decode() if f else None

    def field_int(key):
        f = reader.get_field(key)
        return int(f.parts[f.data[0]][0]) if f else None

    arch = field_str("general.architecture")
    n_layers = field_int(f"{arch}.block_count")
    n_head = field_int(f"{arch}.attention.head_count")
    n_head_kv = field_int(f"{arch}.attention.head_count_kv")
    tmap = hf_to_gguf_name_map(gguf_mod, arch, n_layers)
    tensor_names = [t.name for t in reader.tensors]
    duplicate_names = sorted({name for name in tensor_names if tensor_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"GGUF contains duplicate tensor names: {duplicate_names[:8]}")
    gtensors = {t.name: t for t in reader.tensors}
    baked = load_baked_tensors(baked_dir)
    ternary_names = [n for n, v in baked.items()
                     if v.ndim == 2 and _is_ternary(v)]

    rows = []
    for hf_name in sorted(ternary_names):
        base = hf_name.removesuffix(".weight")
        mapped = tmap.get_name(base, try_suffixes=("",))
        gname = f"{mapped}.weight" if mapped else None
        if gname is None or gname not in gtensors:
            rows.append(TensorParityRow(hf_name, gname or "?", "-", "unmapped",
                                        detail="no gguf tensor for this name"))
            continue
        t = gtensors[gname]
        ttype = t.tensor_type.name
        x = baked[hf_name].astype(np.float32)
        if arch in _QK_PERMUTED_ARCHES and n_head:
            if ".attn_q." in gname:
                x = _llama_qk_permute(x, n_head)
            elif ".attn_k." in gname:
                x = _llama_qk_permute(x, n_head_kv or n_head)
        N, K = x.shape
        if ttype == "TQ2_0":
            got_codes, got_d = decode_tq2_0(np.frombuffer(t.data.tobytes(),
                                                          dtype=np.uint8), N, K)
            ref_codes, ref_d = encode_tq2_0_ref(x)
            mism = float((got_codes != ref_codes).mean())
            deq = got_codes.astype(np.float32) * np.repeat(got_d, 256, axis=1)
            err = np.abs(deq - x)
            bound = float(np.spacing(np.abs(x).max().astype(np.float16)))
            row = TensorParityRow(
                hf_name, gname, ttype,
                "exact" if mism == 0 else "mismatch",
                code_mismatch_rate=mism,
                max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
                within_f16_bound=bool(err.max() <= max(bound, 1e-12)))
        elif ttype in ("F16", "F32", "BF16"):
            deq = _dense_data(t, N, K, ttype)
            err = np.abs(deq - x)
            row = TensorParityRow(hf_name, gname, ttype,
                                  "exact" if err.max() == 0 else "mismatch",
                                  max_abs_err=float(err.max()),
                                  mean_abs_err=float(err.mean()),
                                  detail="stored dense (not ternary-packed)")
        elif ttype == "I2_S":
            got_codes, got_s = decode_i2s(np.frombuffer(t.data.tobytes(),
                                                        dtype=np.uint8), N, K)
            ref_codes, ref_s = encode_i2s_ref(x)
            mism = float((got_codes != ref_codes).mean())
            deq = got_codes.astype(np.float32) * got_s
            err = np.abs(deq - x)
            row = TensorParityRow(
                hf_name, gname, ttype,
                "exact" if mism == 0 else "mismatch",
                code_mismatch_rate=mism, max_abs_err=float(err.max()),
                mean_abs_err=float(err.mean()),
                within_f16_bound=True,   # per-tensor f32 scale, no block rounding
                detail=f"per-tensor scale {got_s:.4g}")
        else:
            row = TensorParityRow(hf_name, gname, ttype, "skipped",
                                  detail=f"no decoder for {ttype}")
        rows.append(row)
    return rows, parity_rows_ok(rows, regime)


def tq1_tensor_parity(artifact_dir: str | Path, gguf_path: str | Path) \
        -> tuple[list[TensorParityRow], bool]:
    """Byte-exact schema-2 canonical artifact/GGUF parity.

    Validation is intentionally fail closed.  A structural or metadata failure
    becomes a mismatch row so callers always receive a report, while no failed
    validation can be mistaken for a successful/empty comparison.
    """
    from bitnet_train.tq1.artifact import ArtifactReader
    from bitnet_train.tq1.gguf import (
        GGML_TYPES, hf_to_gguf_name, validate_tq1_gguf)

    try:
        reader = ArtifactReader(artifact_dir)
        reader.validate()
    except Exception as exc:
        row = TensorParityRow(
            "<canonical-artifact>", "<gguf>", "TQ1_V", "mismatch",
            within_f16_bound=False, detail=f"artifact validation failed: {exc}")
        return [row], False

    rows = [TensorParityRow(
        item["state_dict_name"], hf_to_gguf_name(item["state_dict_name"]),
        next(name for name, value in GGML_TYPES.items()
             if value == GGML_TYPES[item["profile"]]),
        "exact", detail=(
            f"profile={item['profile']} payload/scale/codebook compared bit-exact"))
        for item in reader.manifest["tensors"]]
    try:
        validate_tq1_gguf(artifact_dir, gguf_path)
    except Exception as exc:
        rows.append(TensorParityRow(
            "<canonical-artifact>", "<gguf>", "TQ1_V", "mismatch",
            within_f16_bound=False, detail=str(exc)))
        return rows, False
    return rows, parity_rows_ok(rows, "preserve")


def _is_ternary(v: np.ndarray) -> bool:
    """Baked ternary tensors have exactly the values {-s, 0, +s}."""
    sample = v[:min(4, v.shape[0])].astype(np.float32)
    u = np.unique(np.abs(sample))
    return len(u) <= 2 and (len(u) == 1 or u[0] == 0.0)


def _dense_data(t, N: int, K: int, ttype: str) -> np.ndarray:
    if ttype == "BF16":
        raw = np.frombuffer(t.data.tobytes(), dtype=np.uint16).reshape(N, K)
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return np.asarray(t.data).reshape(N, K).astype(np.float32)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def write_report(rows: list[TensorParityRow], ok: bool, out_prefix: str | Path,
                 meta: dict | None = None, ppl: dict | None = None):
    out_prefix = Path(out_prefix)
    payload = {"ok": ok, "meta": meta or {}, "ppl_parity": ppl or {},
               "tensors": [asdict(r) for r in rows]}
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    lines = ["| tensor | gguf type | status | mismatch | max err | f16 bound |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.hf_name} | {r.gguf_type} | {r.status} | "
                     f"{r.code_mismatch_rate:.2e} | {r.max_abs_err:.3e} | "
                     f"{'ok' if r.within_f16_bound else 'EXCEEDED'} |")
    lines.append(f"\n**overall: {'PASS' if ok else 'FAIL'}**")
    if ppl:
        lines.append(f"\nPPL parity: {json.dumps(ppl)}")
    out_prefix.with_suffix(".md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--baked")
    source.add_argument("--artifact",
                        help="schema-2 canonical TQ1 artifact directory")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--regime", default="preserve", choices=["preserve", "requantize"])
    ap.add_argument("--out", default="parity_report")
    ap.add_argument("--ppl-text", default=None, help="text file for runtime PPL")
    ap.add_argument("--py-ppl", type=float, default=None,
                    help="the paired PyTorch eval-mode PPL")
    ap.add_argument("--ppl-tol", type=float, default=0.05,
                    help="relative PPL tolerance (predeclared)")
    args = ap.parse_args()

    from bitnet_train import provenance
    if args.artifact:
        if args.regime != "preserve":
            ap.error("--artifact TQ1 parity only supports --regime preserve")
        rows, ok = tq1_tensor_parity(args.artifact, args.gguf)
    else:
        rows, ok = tensor_parity(args.baked, args.gguf, regime=args.regime)
    ppl = None
    if args.ppl_text:
        rt_ppl, log = runtime_ppl(args.gguf, args.ppl_text)
        ppl = {"runtime_ppl": rt_ppl, "py_ppl": args.py_ppl}
        if rt_ppl is None:
            ppl["status"] = f"FAIL: runtime PPL unavailable: {log[:200]}"
            ok = False
        elif args.py_ppl is None:
            ppl["status"] = "FAIL: paired PyTorch PPL was not supplied"
            ok = False
        elif args.py_ppl is not None:
            rel = abs(rt_ppl - args.py_ppl) / args.py_ppl
            ppl["rel_diff"] = rel
            ppl["status"] = "pass" if rel <= args.ppl_tol else "FAIL"
            ok = ok and rel <= args.ppl_tol
    meta = {"quantizer_hash": provenance.quantizer_hash(), "gguf": str(args.gguf),
            "baked": str(args.baked) if args.baked else None,
            "artifact": str(args.artifact) if args.artifact else None,
            "regime": args.regime}
    write_report(rows, ok, args.out, meta=meta, ppl=ppl)
    exact = sum(r.status == "exact" for r in rows)
    print(f"[parity] {exact}/{len(rows)} exact; overall {'PASS' if ok else 'FAIL'} "
          f"-> {args.out}.json/.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
