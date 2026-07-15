"""Deterministic instruction-data rendering with assistant-only loss masks.

The mask is aligned one-for-one with the rendered token stream.  Native
``assistant_masks`` from a generation-aware chat template are preferred.  For
templates without ``{% generation %}`` blocks, the strict fallback renders
every conversation prefix, proves that the template is prefix-stable, and
marks the assistant content through that turn's terminator (but not its role
header).  Ambiguous or non-prefix templates fail instead of silently training
on the wrong tokens.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Normalize common chat records into the tokenizer's role/content form."""
    raw = record.get("messages", record.get("conversations"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("instruction record must contain a nonempty messages list")
    role_aliases = {
        "human": "user", "user": "user", "system": "system",
        "assistant": "assistant", "gpt": "assistant", "bot": "assistant",
        "tool": "tool", "function": "tool",
    }
    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"message {index} is not an object")
        role = item.get("role", item.get("from"))
        content = item.get("content", item.get("value"))
        if not isinstance(role, str) or role.lower() not in role_aliases:
            raise ValueError(f"message {index} has an unsupported role {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError(f"message {index} content must be a nonempty string")
        messages.append({"role": role_aliases[role.lower()], "content": content})
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("conversation contains no assistant response")
    return messages


def _native_assistant_mask(tokenizer, messages: list[dict[str, str]]) \
        -> tuple[list[int], list[int]] | None:
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_assistant_tokens_mask=True)
    except (TypeError, ValueError):
        return None
    if not isinstance(rendered, Mapping):
        return None
    ids = rendered.get("input_ids")
    mask = rendered.get("assistant_masks", rendered.get("assistant_tokens_mask"))
    if ids is None or mask is None:
        return None
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    if isinstance(mask, torch.Tensor):
        mask = mask.reshape(-1).tolist()
    ids, mask = list(ids), [int(value) for value in mask]
    if len(ids) != len(mask) or not any(mask):
        return None
    if any(value not in (0, 1) for value in mask):
        raise ValueError("tokenizer returned a non-binary assistant mask")
    return ids, mask


def _tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.reshape(-1, 2).tolist()
    if len(ids) != len(offsets):
        raise ValueError("tokenizer returned misaligned offsets")
    return list(ids), [(int(begin), int(end)) for begin, end in offsets]


def _prefix_assistant_mask(tokenizer, messages: list[dict[str, str]]) \
        -> tuple[list[int], list[int]]:
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    if not isinstance(full, str) or not full:
        raise ValueError("chat template rendered an empty/non-string conversation")
    spans: list[tuple[int, int]] = []
    previous = ""
    for index, message in enumerate(messages):
        current = tokenizer.apply_chat_template(
            messages[:index + 1], tokenize=False, add_generation_prompt=False)
        if not isinstance(current, str) or not current.startswith(previous) \
                or not full.startswith(current):
            raise ValueError(
                "chat template is not prefix-stable; add generation blocks so the "
                "tokenizer can return a native assistant mask")
        appended_begin = len(previous)
        if message["role"] == "assistant":
            content = message["content"]
            content_begin = current.find(content, appended_begin)
            if content_begin < 0 or current.find(content, content_begin + 1) >= 0:
                raise ValueError(
                    f"assistant content at message {index} is ambiguous after rendering")
            # Include the template's assistant turn terminator.  The next role
            # header is appended by the next prefix and is therefore excluded.
            spans.append((content_begin, len(current)))
        previous = current
    if previous != full:
        raise ValueError("full chat render differs from its final prefix render")
    ids, offsets = _tokenize_with_offsets(tokenizer, full)
    mask = [0] * len(ids)
    for token, (begin, end) in enumerate(offsets):
        if begin == end:  # synthetic/special tokens with no source span
            continue
        if any(begin < span_end and end > span_begin for span_begin, span_end in spans):
            mask[token] = 1
    if not any(mask):
        raise ValueError("assistant mask contains no tokens after chat-template rendering")
    return ids, mask


def render_assistant_mask(tokenizer, messages: list[dict[str, str]], *,
                          prefer_native: bool = True) -> tuple[np.ndarray, np.ndarray, str]:
    """Return uint32 ids, uint8 mask, and the mask construction method."""
    native = _native_assistant_mask(tokenizer, messages) if prefer_native else None
    if native is None:
        ids, mask = _prefix_assistant_mask(tokenizer, messages)
        method = "prefix_offsets_v1"
    else:
        ids, mask = native
        method = "tokenizer_generation_mask"
    values = np.asarray(ids, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or values.min() < 0 \
            or values.max() > np.iinfo(np.uint32).max:
        raise ValueError("rendered token ids do not fit a nonempty uint32 stream")
    masks = np.asarray(mask, dtype=np.uint8)
    if masks.shape != values.shape or not np.isin(masks, (0, 1)).all():
        raise ValueError("assistant mask is not aligned binary data")
    return values.astype(np.uint32), masks, method


@dataclass
class MaskedShardWriter:
    directory: Path
    split: str
    shard_size: int

    def __post_init__(self) -> None:
        if self.shard_size < 1:
            raise ValueError("shard_size must be positive")
        self.tokens: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.buffered = 0
        self.total = 0
        self.assistant_tokens = 0
        self.shards: list[dict[str, Any]] = []

    def add(self, tokens: np.ndarray, mask: np.ndarray) -> None:
        if tokens.dtype != np.uint32 or mask.dtype != np.uint8 \
                or tokens.ndim != 1 or mask.shape != tokens.shape:
            raise ValueError("masked writer requires aligned uint32/uint8 vectors")
        self.tokens.append(tokens)
        self.masks.append(mask)
        self.buffered += int(tokens.size)
        self.total += int(tokens.size)
        self.assistant_tokens += int(mask.sum())
        if self.buffered >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffered:
            return
        tokens = np.concatenate(self.tokens).astype(np.uint32, copy=False)
        masks = np.concatenate(self.masks).astype(np.uint8, copy=False)
        index = len(self.shards)
        token_name = f"{self.split}_{index:05d}.bin"
        mask_name = f"{self.split}_{index:05d}.loss_mask.bin"
        token_bytes, mask_bytes = tokens.tobytes(), masks.tobytes()
        (self.directory / token_name).write_bytes(token_bytes)
        (self.directory / mask_name).write_bytes(mask_bytes)
        self.shards.append({
            "name": token_name,
            "n_tokens": int(tokens.size),
            "sha256": sha256_bytes(token_bytes),
            "loss_mask_name": mask_name,
            "loss_mask_sha256": sha256_bytes(mask_bytes),
            "assistant_tokens": int(masks.sum()),
        })
        self.tokens.clear()
        self.masks.clear()
        self.buffered = 0

    def close(self) -> None:
        self.flush()


def build_masked_shards(records: Iterable[Mapping[str, Any]], tokenizer, directory: str | Path,
                        *, seq_len: int, shard_size: int, val_fraction: float = 0.01,
                        limit: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render records into deterministic train/val flat shards and a manifest body."""
    if seq_len < 2 or not 0 <= val_fraction < 1:
        raise ValueError("seq_len must be >=2 and val_fraction must be in [0,1)")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    writers = {"train": MaskedShardWriter(directory, "train", shard_size)}
    if val_fraction:
        writers["val"] = MaskedShardWriter(directory, "val", shard_size)
    every = max(2, round(1 / val_fraction)) if val_fraction else 0
    retained = rejected = truncated = 0
    methods: dict[str, int] = {}
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("instruction tokenizer has no eos token")
    for source_index, record in enumerate(records):
        if limit and retained >= limit:
            break
        try:
            messages = normalize_messages(record)
            tokens, mask, method = render_assistant_mask(tokenizer, messages)
        except ValueError:
            rejected += 1
            continue
        if tokens.size > seq_len:
            # Preserve the beginning deterministically.  Reject examples whose
            # truncation removes every supervised token.
            tokens, mask = tokens[:seq_len], mask[:seq_len]
            truncated += 1
            if not mask.any():
                rejected += 1
                continue
        delimiter = np.asarray([eos], dtype=np.uint32)
        tokens = np.concatenate((tokens, delimiter))
        mask = np.concatenate((mask, np.zeros(1, dtype=np.uint8)))
        split = "val" if every and source_index % every == 0 else "train"
        writers[split].add(tokens, mask)
        retained += 1
        methods[method] = methods.get(method, 0) + 1
    if not retained or not writers["train"].total:
        raise ValueError("instruction source retained no training conversations")
    for writer in writers.values():
        writer.close()
    splits = {
        split: {
            "shards": writer.shards,
            "total_tokens": writer.total,
            "assistant_tokens": writer.assistant_tokens,
            "approx_sequences": writer.total // seq_len,
        }
        for split, writer in writers.items()
    }
    stats = {
        "retained_records": retained,
        "rejected_records": rejected,
        "truncated_records": truncated,
        "mask_methods": dict(sorted(methods.items())),
    }
    return splits, stats


def write_masked_manifest(directory: str | Path, *, tokenizer_id: str,
                          tokenizer_revision: str, tokenizer, seq_len: int,
                          source: Mapping[str, Any], splits: Mapping[str, Any],
                          statistics: Mapping[str, Any]) -> dict[str, Any]:
    directory = Path(directory)
    template = tokenizer.chat_template or ""
    manifest = {
        "schema": 2,
        "tokenizer": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": max(int(tokenizer.vocab_size), len(tokenizer)),
        "dtype": "uint32",
        "eos_id": int(tokenizer.eos_token_id),
        "seq_len": int(seq_len),
        "chat_template_sha256": sha256_bytes(template.encode("utf-8")),
        "loss_mask": {
            "dtype": "uint8",
            "meaning": "1=assistant content/turn terminator contributes to next-token loss",
            "alignment": "one byte per token",
        },
        "source": dict(source),
        "statistics": dict(statistics),
        "splits": dict(splits),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return manifest
