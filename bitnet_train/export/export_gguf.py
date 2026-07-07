"""Baked-ternary export (train_plan §7.0 file #7, §8.2).

bake_checkpoint writes w_baked = weight_quant(w_latent) in fp32 — every ternary
value exactly {-s, 0, +s} on the f16-rounded scale grid — so a route quantizer
that sees clean input has no reason to flip codes. Latents never leave the
training checkpoint. The route exporters are subprocess wrappers that return a
STRUCTURED SKIP (not a failure) when their tooling is absent, which is what lets
the CI micro-loop run llama.cpp-free.

Routes (train_plan §8.1):
  I2_S   — utils/convert-hf-to-gguf-bitnet.py + fork llama-quantize (3rdparty submodule)
  TQ2_0  — mainline convert_hf_to_gguf.py + llama-quantize TQ2_0
           (mainline dir: $LLAMA_MAINLINE, default ~/llama.cpp)
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import torch

from bitnet_train import quant
from bitnet_train.bitlinear import iter_bitexperts, iter_bitlinears

REPO = Path(__file__).resolve().parents[2]
FORK_DIR = REPO / "3rdparty" / "llama.cpp"


def mainline_dir() -> Path:
    return Path(os.environ.get("LLAMA_MAINLINE", str(Path.home() / "llama.cpp")))


# ---------------------------------------------------------------------------
# bake
# ---------------------------------------------------------------------------

@dataclass
class BakeReport:
    quantizer_version: str
    quantizer_hash: str
    granularity: str
    group_k: int
    tensors: dict[str, dict] = field(default_factory=dict)

    def to_json(self, path: str | Path):
        Path(path).write_text(json.dumps(self.__dict__, indent=2))


@torch.no_grad()
def bake_checkpoint(model, profile, out_dir: str | Path,
                    tokenizer_src: str | Path | None = None) -> BakeReport:
    """Replace every ternary latent with its baked {-s,0,+s} fp32 value in place,
    save_pretrained, and record per-tensor scale/code counts. The model instance
    is CONSUMED (latents overwritten) — load a fresh copy for training."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = BakeReport(quantizer_version=quant.QUANTIZER_VERSION,
                     quantizer_hash=quant.quantizer_hash(),
                     granularity=profile.granularity, group_k=profile.group_k)

    def bake_one(name: str, w: torch.Tensor):
        codes, scale = quant.ternary_codes(w, profile.granularity, profile.group_k)
        baked = quant.dequant_codes(codes, scale, profile.group_k).to(w.dtype)
        w.copy_(baked)
        rep.tensors[name] = {
            "scale": (float(scale) if scale.dim() == 0
                      else [float(scale.min()), float(scale.max())]),
            "n_neg": int((codes == -1).sum()), "n_zero": int((codes == 0).sum()),
            "n_pos": int((codes == 1).sum()), "shape": list(w.shape),
        }

    for name, mod in iter_bitlinears(model):
        bake_one(f"{name}.weight", mod.weight)
        mod._wcache = (-1, None, None)                 # invalidate quant caches
        mod._packed = None
    for name, mod in iter_bitexperts(model):
        for slice_name, w in mod.expert_slices():
            bake_one(f"{name}.{slice_name}", w)

    model = model.float()
    model.save_pretrained(out_dir)                     # fp32: dtype cannot flip codes
    if tokenizer_src is not None:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(str(tokenizer_src)).save_pretrained(out_dir)
    rep.to_json(out_dir / "bake_report.json")
    return rep


# ---------------------------------------------------------------------------
# route exporters
# ---------------------------------------------------------------------------

@dataclass
class ExportResult:
    route: str
    ok: bool
    skipped: bool = False
    reason: str = ""
    gguf: str = ""
    log_tail: str = ""


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       cwd=str(cwd) if cwd else None)
    log = (p.stdout or "") + (p.stderr or "")
    return p.returncode, log


def export_tq2(baked_dir: str | Path, out_gguf: str | Path,
               llama_dir: str | Path | None = None,
               tensor_type_overrides: list[str] = (),
               python: str = "python3") -> ExportResult:
    """Mainline route: convert_hf_to_gguf.py --outtype f32 -> llama-quantize TQ2_0.
    tensor_type_overrides: raw --tensor-type args (e.g. 'ffn_down_exps=tq2_0')."""
    ml = Path(llama_dir) if llama_dir else mainline_dir()
    conv = ml / "convert_hf_to_gguf.py"
    quantize = ml / "build" / "bin" / "llama-quantize"
    if not conv.exists():
        return ExportResult("tq2_upstream", ok=False, skipped=True,
                            reason=f"mainline converter not found at {conv}")
    if not quantize.exists():
        return ExportResult("tq2_upstream", ok=False, skipped=True,
                            reason=f"llama-quantize not built at {quantize}")
    out_gguf = Path(out_gguf)
    f32_gguf = out_gguf.with_suffix(".f32.gguf")
    rc, log = _run([python, conv, str(baked_dir), "--outfile", f32_gguf,
                    "--outtype", "f32"])
    if rc != 0:
        return ExportResult("tq2_upstream", ok=False,
                            reason="convert_hf_to_gguf failed", log_tail=log[-2000:])
    cmd = [quantize]
    for ov in tensor_type_overrides:
        cmd += ["--tensor-type", ov]
    cmd += [f32_gguf, out_gguf, "TQ2_0"]
    rc, log = _run(cmd)
    if rc != 0:
        return ExportResult("tq2_upstream", ok=False, reason="llama-quantize failed",
                            log_tail=log[-2000:])
    return ExportResult("tq2_upstream", ok=True, gguf=str(out_gguf),
                        log_tail=log[-800:])


def export_i2s(baked_dir: str | Path, out_gguf: str | Path,
               fork_dir: str | Path = FORK_DIR,
               python: str = "python3") -> ExportResult:
    """Fork route (the Llama-1.58 precedent path): utils/convert-hf-to-gguf-bitnet.py
    --outtype f32 -> fork llama-quantize ... I2_S 1."""
    fork = Path(fork_dir)
    conv = REPO / "utils" / "convert-hf-to-gguf-bitnet.py"
    quantize = None
    for cand in (fork / "build" / "bin" / "llama-quantize",
                 fork / "build" / "bin" / "quantize"):
        if cand.exists():
            quantize = cand
            break
    if not (fork / "ggml").exists() and not (fork / "src").exists():
        return ExportResult("i2s_bitnet_cpp", ok=False, skipped=True,
                            reason=f"fork submodule not initialized at {fork} "
                                   "(git submodule update --init --recursive)")
    if quantize is None:
        # the fork build needs a model-specific bitnet-lut-kernels.h from the
        # repo's TL1/TL2 codegen (docs/codegen.md); setup_env.py runs the codegen
        # + build for a chosen model at T0. The parity DECODER (compare_gguf.
        # decode_i2s) does not need the binary and is always available.
        return ExportResult("i2s_bitnet_cpp", ok=False, skipped=True,
                            reason=f"fork llama-quantize not built under {fork}/build "
                                   "(run setup_env.py for the target model — it "
                                   "codegens bitnet-lut-kernels.h then builds)")
    out_gguf = Path(out_gguf)
    f32_gguf = out_gguf.with_suffix(".f32.gguf")
    rc, log = _run([python, conv, str(baked_dir), "--outfile", f32_gguf,
                    "--outtype", "f32"])
    if rc != 0:
        return ExportResult("i2s_bitnet_cpp", ok=False,
                            reason="convert-hf-to-gguf-bitnet failed",
                            log_tail=log[-2000:])
    rc, log = _run([quantize, f32_gguf, out_gguf, "I2_S", "1"])
    if rc != 0:
        return ExportResult("i2s_bitnet_cpp", ok=False, reason="fork quantize failed",
                            log_tail=log[-2000:])
    return ExportResult("i2s_bitnet_cpp", ok=True, gguf=str(out_gguf),
                        log_tail=log[-800:])


def runtime_ppl(gguf: str | Path, text_file: str | Path,
                llama_dir: str | Path | None = None, ctx: int = 512,
                ngl: int = 0, extra: list[str] = ()) -> tuple[float | None, str]:
    """PPL from llama-perplexity; returns (ppl, log_tail). None on failure/skip.
    ngl=0 (CPU) is the safe default: STOCK llama.cpp has no Metal TQ2_0 kernels
    and default GPU offload crashes in graph compute on Apple (T0 recon fact,
    2026-07-06). Our local mainline (~/llama.cpp) carries TQ2_0 Metal kernels
    (mul_mv/mul_mm/*_id/get_rows, test-backend-ops green vs CPU, 2026-07-07) —
    pass ngl=99 on that build for GPU eval."""
    ml = Path(llama_dir) if llama_dir else mainline_dir()
    binp = ml / "build" / "bin" / "llama-perplexity"
    if not binp.exists():
        return None, f"llama-perplexity not built at {binp}"
    rc, log = _run([binp, "-m", gguf, "-f", text_file, "-c", str(ctx),
                    "-ngl", str(ngl), *extra])
    if rc != 0:
        return None, log[-2000:]
    import re
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", log)
    return (float(m.group(1)) if m else None), log[-800:]
