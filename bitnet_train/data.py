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
from torch.utils.data import DataLoader, Dataset, Sampler


def load_manifest(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text())


def manifest_hash(manifest: dict) -> str:
    """Canonical hash over everything that defines the frozen corpus — tokenizer,
    vocab, dtype, eos, seq_len, and each split's shard names/sizes/sha256s.
    Timestamps and paths are excluded on purpose (a moved corpus hashes the same;
    a re-tokenized or re-sharded one does not)."""
    core = {
        "schema": manifest.get("schema"),
        "tokenizer": manifest.get("tokenizer"),
        "tokenizer_revision": manifest.get("tokenizer_revision"),
        "chat_template_sha256": manifest.get("chat_template_sha256"),
        "vocab_size": manifest.get("vocab_size"),
        "dtype": manifest.get("dtype"),
        "eos_id": manifest.get("eos_id"),
        "seq_len": manifest.get("seq_len"),
        "loss_mask": manifest.get("loss_mask"),
        "splits": {
            split: [{"name": s["name"], "n_tokens": s["n_tokens"],
                     "sha256": s.get("sha256"),
                     "loss_mask_name": s.get("loss_mask_name"),
                     "loss_mask_sha256": s.get("loss_mask_sha256")}
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
        self.mask_paths: list[Path | None] = []
        self.index: list[tuple[int, int]] = []
        for s in man["splits"][split]["shards"]:
            p = self.dir / s["name"]
            expect = s["n_tokens"] * self.dtype.itemsize
            if p.stat().st_size != expect:
                raise ValueError(f"{p.name}: size {p.stat().st_size} != manifest "
                                 f"n_tokens*{self.dtype.itemsize} ({expect})")
            si = len(self.paths)
            self.paths.append(p)
            mask_name = s.get("loss_mask_name")
            mask_path = self.dir / mask_name if mask_name else None
            if mask_path is not None and mask_path.stat().st_size != s["n_tokens"]:
                raise ValueError(f"{mask_path.name}: loss-mask size must equal n_tokens")
            self.mask_paths.append(mask_path)
            for w in range(s["n_tokens"] // self.seq_len):
                self.index.append((si, w))
        self._maps: dict[int, np.memmap] = {}
        self._mask_maps: dict[int, np.memmap] = {}
        has_masks = [path is not None for path in self.mask_paths]
        if any(has_masks) and not all(has_masks):
            raise ValueError("a split may not mix masked and unmasked shards")
        self.has_loss_mask = bool(has_masks and has_masks[0])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        shard, w = self.index[i]
        mm = self._maps.get(shard)
        if mm is None:
            mm = self._maps[shard] = np.memmap(self.paths[shard], dtype=self.dtype,
                                               mode="r")
        a = np.array(mm[w * self.seq_len:(w + 1) * self.seq_len],
                     dtype=np.int64, copy=True)
        tokens = torch.from_numpy(a)
        if not self.has_loss_mask:
            return tokens
        mask = self._mask_maps.get(shard)
        if mask is None:
            mask = self._mask_maps[shard] = np.memmap(
                self.mask_paths[shard], dtype=np.uint8, mode="r")
        m = np.array(mask[w * self.seq_len:(w + 1) * self.seq_len],
                     dtype=np.uint8, copy=True)
        return tokens, torch.from_numpy(m).bool()


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


class EpochRandomSampler(Sampler[int]):
    """Random permutation addressed by (seed, epoch), independent of prior RNG use."""

    def __init__(self, data_source: Dataset, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("data epoch must be nonnegative")
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def make_loader(ds: Dataset, batch_size: int, seed: int,
                num_workers: int = 0) -> DataLoader:
    sampler = EpochRandomSampler(ds, seed)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                      generator=torch.Generator().manual_seed(seed),
                      drop_last=True, num_workers=num_workers)


class ResumableDataStream:
    """Infinite loader stream with an exact checkpointable local-batch cursor.

    ``loader`` may already be wrapped by Accelerate. Its local length defines an
    epoch, while ``set_epoch`` propagates through BatchSamplerShard to the
    underlying EpochRandomSampler. Resume skips at most one epoch, never every
    batch since step zero.
    """

    SCHEMA = 1

    def __init__(self, loader: DataLoader, *, seed: int,
                 state: dict | None = None, consumed_batches: int = 0):
        self.loader = loader
        self.seed = int(seed)
        self.batches_per_epoch = len(loader)
        if self.batches_per_epoch <= 0:
            raise ValueError("training loader has no complete batches")
        if state is not None:
            required = {"schema", "seed", "batches_per_epoch", "consumed_batches",
                        "epoch", "offset"}
            if set(state) != required or state["schema"] != self.SCHEMA:
                raise ValueError("unsupported or malformed data-stream checkpoint")
            if int(state["seed"]) != self.seed:
                raise ValueError("data-stream seed differs from checkpoint")
            if int(state["batches_per_epoch"]) != self.batches_per_epoch:
                raise ValueError("data-stream epoch length differs from checkpoint")
            consumed_batches = int(state["consumed_batches"])
            expected_epoch, expected_offset = divmod(
                consumed_batches, self.batches_per_epoch)
            if (int(state["epoch"]), int(state["offset"])) != \
                    (expected_epoch, expected_offset):
                raise ValueError("data-stream cursor is internally inconsistent")
        if consumed_batches < 0:
            raise ValueError("consumed data batches must be nonnegative")
        self.consumed_batches = int(consumed_batches)
        self.epoch, self.offset = divmod(
            self.consumed_batches, self.batches_per_epoch)
        self._iterator = None

    def _start_epoch(self) -> None:
        if hasattr(self.loader, "set_epoch"):
            self.loader.set_epoch(self.epoch)
        else:
            batch_sampler = getattr(self.loader, "batch_sampler", None)
            sampler = getattr(batch_sampler, "sampler", None)
            if not hasattr(sampler, "set_epoch"):
                raise ValueError("loader does not expose an epoch-addressable sampler")
            sampler.set_epoch(self.epoch)
        self._iterator = iter(self.loader)
        for _ in range(self.offset):
            try:
                next(self._iterator)
            except StopIteration as exc:
                raise ValueError("saved data-stream offset exceeds the loader epoch") from exc

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._start_epoch()
        try:
            batch = next(self._iterator)
        except StopIteration:
            # The cursor is advanced immediately after every yielded batch, so
            # consuming the final batch has already moved it to (epoch + 1, 0).
            # Advancing here as well would silently skip an epoch on the next
            # call.
            if self.offset != 0:
                raise RuntimeError("loader ended before its recorded epoch length")
            self._start_epoch()
            batch = next(self._iterator)
        self.consumed_batches += 1
        self.epoch, self.offset = divmod(
            self.consumed_batches, self.batches_per_epoch)
        return batch

    def state_dict(self) -> dict[str, int]:
        return {
            "schema": self.SCHEMA, "seed": self.seed,
            "batches_per_epoch": self.batches_per_epoch,
            "consumed_batches": self.consumed_batches,
            "epoch": self.epoch, "offset": self.offset,
        }


def cycle(loader: DataLoader):
    while True:
        yield from loader


def calibration_windows(data_dir: str | Path, n: int, split: str = "val",
                        seq_len: int | None = None):
    """The FIXED calibration set: the first n windows of the split in manifest
    order (deterministic — train_plan §10.1's 'fixed calibration set')."""
    ds = PackedWindows(data_dir, split=split, seq_len=seq_len)
    n = min(n, len(ds))
    if n == 0:
        raise ValueError(f"no calibration windows in {data_dir}:{split}")
    values = [ds[i] for i in range(n)]
    if ds.has_loss_mask:
        return torch.stack([value[0] for value in values]), \
            torch.stack([value[1] for value in values])
    return torch.stack(values)
