"""Token-shard dataset (adapted from AUM train/train.py PackedWindows, dtype-widened).

Flat EOS-separated token shards + manifest.json, non-overlapping seq_len windows
over lazy per-shard memmaps, window-level shuffle, drop_last. Deltas from AUM:
the on-disk dtype comes from the MANIFEST (uint32 — 128K/152K/262K vocabs do not
fit uint16; a uint16 pipeline silently wraps half the vocab, train_plan §5.5),
plus size/dtype asserts and the canonical manifest hash that checkpoints and
teacher caches pin (§5.6).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_manifest(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text())


def manifest_hash(manifest: dict) -> str:
    """Canonical hash over everything that defines the frozen corpus — tokenizer,
    vocab, dtype, eos, seq_len, and each split's shard names/sizes/sha256s.
    Timestamps and paths are excluded on purpose (a moved corpus hashes the same;
    a re-tokenized or re-sharded one does not)."""
    core = {
        "tokenizer": manifest.get("tokenizer"),
        "vocab_size": manifest.get("vocab_size"),
        "dtype": manifest.get("dtype"),
        "eos_id": manifest.get("eos_id"),
        "seq_len": manifest.get("seq_len"),
        "splits": {
            split: [{"name": s["name"], "n_tokens": s["n_tokens"],
                     "sha256": s.get("sha256")}
                    for s in info["shards"]]
            for split, info in manifest.get("splits", {}).items()
        },
    }
    blob = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class PackedWindows(Dataset):
    """Non-overlapping seq_len windows over the flat token stream; per-shard tails
    are dropped; lazy per-worker memmaps (worker-safe)."""

    def __init__(self, data_dir: str | Path, split: str = "train",
                 seq_len: int | None = None):
        self.dir = Path(data_dir)
        man = load_manifest(self.dir)
        self.manifest = man
        self.seq_len = int(seq_len or man["seq_len"])
        self.dtype = np.dtype(man["dtype"])
        if man.get("vocab_size") and man["vocab_size"] > np.iinfo(self.dtype).max + 1:
            raise ValueError(f"vocab {man['vocab_size']} does not fit manifest dtype "
                             f"{man['dtype']} — re-tokenize (train_plan §5.5)")
        if split not in man["splits"]:
            raise KeyError(f"split {split!r} not in manifest ({list(man['splits'])})")
        self.paths: list[Path] = []
        self.index: list[tuple[int, int]] = []
        for s in man["splits"][split]["shards"]:
            p = self.dir / s["name"]
            expect = s["n_tokens"] * self.dtype.itemsize
            if p.stat().st_size != expect:
                raise ValueError(f"{p.name}: size {p.stat().st_size} != manifest "
                                 f"n_tokens*{self.dtype.itemsize} ({expect})")
            si = len(self.paths)
            self.paths.append(p)
            for w in range(s["n_tokens"] // self.seq_len):
                self.index.append((si, w))
        self._maps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        shard, w = self.index[i]
        mm = self._maps.get(shard)
        if mm is None:
            mm = self._maps[shard] = np.memmap(self.paths[shard], dtype=self.dtype,
                                               mode="r")
        a = np.asarray(mm[w * self.seq_len:(w + 1) * self.seq_len], dtype=np.int64)
        return torch.from_numpy(a)


class IndexedWindows(Dataset):
    """Wraps PackedWindows to also yield the DATASET index — the key that aligns
    a batch with its teacher top-k cache rows (caches are stored in dataset
    order; the loader shuffles)."""

    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i: int):
        return torch.tensor(i, dtype=torch.int64), self.ds[i]


def make_loader(ds: Dataset, batch_size: int, seed: int,
                num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(seed),
                      drop_last=True, num_workers=num_workers)


def cycle(loader: DataLoader):
    while True:
        yield from loader


def calibration_windows(data_dir: str | Path, n: int, split: str = "val",
                        seq_len: int | None = None) -> torch.Tensor:
    """The FIXED calibration set: the first n windows of the split in manifest
    order (deterministic — train_plan §10.1's 'fixed calibration set')."""
    ds = PackedWindows(data_dir, split=split, seq_len=seq_len)
    n = min(n, len(ds))
    if n == 0:
        raise ValueError(f"no calibration windows in {data_dir}:{split}")
    return torch.stack([ds[i] for i in range(n)])
