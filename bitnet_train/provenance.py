"""Provenance hashes + RNG capture (train_plan §5.6: checkpoint metadata MUST
record quantizer version/hash, RNG seeds, config hash, data-manifest hash —
parity reports and ablations are only meaningful with provenance pinned).
Closes the AUM gap (no hashes/RNG in its resumable state)."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from bitnet_train.quant import QUANTIZER_VERSION, quantizer_hash  # noqa: F401
from bitnet_train.conversion import profile_hash  # noqa: F401


def config_hash(cfg) -> str:
    """Order-independent hash of an HF config (or plain dict)."""
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    d.pop("transformers_version", None)
    d.pop("_name_or_path", None)
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def rng_capture() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        try:
            state["mps"] = torch.mps.get_rng_state()
        except (AttributeError, RuntimeError):
            pass
    return state


def rng_restore(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def build_meta(*, profile_path: str | Path | None = None, model_config=None,
               data_manifest: dict | None = None, seeds: dict | None = None,
               extra: dict | None = None) -> dict:
    """The checkpoint/report metadata block. Every hash that is derivable is
    re-derived at --resume and mismatches are hard errors."""
    from bitnet_train.data import manifest_hash as _mh
    meta = {
        "quantizer_version": QUANTIZER_VERSION,
        "quantizer_hash": quantizer_hash(),
        "git_rev": git_rev(),
    }
    if profile_path is not None:
        meta["profile_hash"] = profile_hash(profile_path)
        meta["profile_path"] = str(profile_path)
    if model_config is not None:
        meta["config_hash"] = config_hash(model_config)
    if data_manifest is not None:
        meta["manifest_hash"] = _mh(data_manifest)
    if seeds:
        meta["seeds"] = seeds
    if extra:
        meta.update(extra)
    return meta
