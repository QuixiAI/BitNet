"""Deterministic capability-balanced, assistant-token-quota instruction mixture."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .instruct_data import (
    MaskedShardWriter, normalize_messages, render_assistant_mask, sha256_bytes)
from .spec import canonical_json


MIXTURE_MANIFEST_SCHEMA = 3
CAPABILITY_BUCKETS = {
    "instruction", "tools", "math", "code", "multilingual",
    "long_context", "chat",
}


@dataclass(frozen=True)
class MixtureSource:
    name: str
    dataset: str
    revision: str
    license: str
    split: str
    config: str | None
    capability: str
    language: str
    assistant_token_quota: float
    context_token_target: float
    id_field: str = "id"

    def __post_init__(self) -> None:
        for name in ("name", "dataset", "revision", "license", "split",
                     "capability", "language", "id_field"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"mixture source {name} must be nonempty")
        if len(self.revision) not in {40, 64} \
                or any(character not in "0123456789abcdef" for character in self.revision):
            raise ValueError("mixture source revision must be a full immutable 40/64-hex id")
        if self.capability not in CAPABILITY_BUCKETS:
            raise ValueError(f"unknown mixture capability {self.capability!r}")
        for name in ("assistant_token_quota", "context_token_target"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"mixture source {name} must be in (0,1]")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MixtureSource":
        if not isinstance(value, Mapping):
            raise ValueError("mixture source must be an object")
        expected = set(cls.__dataclass_fields__)
        unknown = set(value) - expected
        if unknown:
            raise ValueError("mixture source has unknown fields")
        missing = {name for name, field in cls.__dataclass_fields__.items()
                   if field.default is MISSING and field.default_factory is MISSING} - set(value)
        if missing:
            raise ValueError(f"mixture source is missing {sorted(missing)}")
        return cls(**value)


def validate_mixture_sources(sources: Iterable[MixtureSource]) -> tuple[MixtureSource, ...]:
    values = tuple(sources)
    if not values or len({source.name for source in values}) != len(values):
        raise ValueError("mixture source names must be nonempty and unique")
    if {source.capability for source in values} != CAPABILITY_BUCKETS:
        missing = CAPABILITY_BUCKETS - {source.capability for source in values}
        raise ValueError(f"mixture must cover all capability buckets; missing {sorted(missing)}")
    for field in ("assistant_token_quota", "context_token_target"):
        total = sum(getattr(source, field) for source in values)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"mixture {field} values must sum to one (got {total})")
    return values


def _record_identity(source: MixtureSource, record: Mapping[str, Any],
                     messages: list[dict[str, str]]) -> tuple[str, str]:
    digest = hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()
    raw = record.get(source.id_field)
    selected_id = str(raw) if isinstance(raw, (str, int)) else digest
    if not selected_id:
        selected_id = digest
    return selected_id, digest


class _CandidateStream:
    def __init__(self, source: MixtureSource, records: Iterable[Mapping[str, Any]],
                 tokenizer, *, seq_len: int, seen: set[str]):
        self.source = source
        self.records: Iterator[Mapping[str, Any]] = iter(records)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seen = seen
        self.rejected = self.duplicates = self.truncated = 0

    def next(self) -> tuple[np.ndarray, np.ndarray, str, str, str, str, str] | None:
        for record in self.records:
            try:
                if not isinstance(record, Mapping):
                    raise ValueError("mixture record is not an object")
                messages = normalize_messages(record)
                selected_id, digest = _record_identity(self.source, record, messages)
                if digest in self.seen:
                    self.duplicates += 1
                    continue
                tokens, mask, method = render_assistant_mask(self.tokenizer, messages)
                if tokens.size > self.seq_len:
                    tokens, mask = tokens[:self.seq_len], mask[:self.seq_len]
                    self.truncated += 1
                if not mask.any():
                    raise ValueError("mixture truncation removed every assistant token")
                bucket = str(record.get("bucket", self.source.capability))
                language = str(record.get("language", self.source.language))
                if not bucket or not language:
                    raise ValueError("mixture record bucket/language must be nonempty")
            except ValueError:
                self.rejected += 1
                continue
            self.seen.add(digest)
            return tokens, mask, selected_id, digest, method, bucket, language
        return None


def _quantiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p50": 0.0, "p95": 0.0, "max": 0, "mean": 0.0}
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()), "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)), "max": int(array.max()),
        "mean": float(array.mean()),
    }


def build_quota_mixture(
        sources: Iterable[MixtureSource], records: Mapping[str, Iterable[Mapping[str, Any]]],
        tokenizer, directory: str | Path, *, seq_len: int, shard_size: int,
        assistant_token_budget: int, val_fraction: float = 0.01,
        quota_tolerance: float = 0.03, tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
        mixture_spec_sha256: str | None = None) -> dict[str, Any]:
    """Build a deterministic mixture whose scheduling unit is assistant tokens.

    The scheduler repeatedly chooses the source with the lowest delivered / target
    assistant-token ratio.  Overshoot is bounded by one selected conversation;
    the manifest records both target and observed assistant/context shares.
    """
    specs = validate_mixture_sources(sources)
    if set(records) != {source.name for source in specs}:
        raise ValueError("mixture record streams differ from source declarations")
    if seq_len < 2 or shard_size < 1 or assistant_token_budget < 1:
        raise ValueError("mixture sequence/shard/budget values must be positive")
    if not 0 <= val_fraction < 1 or not 0 <= quota_tolerance < 1:
        raise ValueError("mixture fractions are invalid")
    if tokenizer.eos_token_id is None:
        raise ValueError("mixture tokenizer has no EOS token")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    writers = {"train": MaskedShardWriter(directory, "train", shard_size)}
    if val_fraction:
        writers["val"] = MaskedShardWriter(directory, "val", shard_size)
    seen: set[str] = set()
    stream_objects = {
        source.name: _CandidateStream(
            source, records[source.name], tokenizer, seq_len=seq_len, seen=seen)
        for source in specs
    }
    active = set(stream_objects)
    delivered = {source.name: 0 for source in specs}
    context = {source.name: 0 for source in specs}
    selected_ids = {source.name: [] for source in specs}
    selected_hashes = {source.name: [] for source in specs}
    lengths = {source.name: [] for source in specs}
    assistant_lengths = {source.name: [] for source in specs}
    source_buckets = {source.name: {} for source in specs}
    bucket_counts: dict[str, dict[str, int]] = {}
    language_counts: dict[str, dict[str, int]] = {}
    methods: dict[str, int] = {}
    records_count = 0
    total_assistant = 0
    while total_assistant < assistant_token_budget:
        available = [source for source in specs if source.name in active]
        if not available:
            raise ValueError("mixture sources exhausted before the assistant-token budget")
        source = min(
            available,
            key=lambda item: (
                delivered[item.name] / max(item.assistant_token_quota, 1e-30),
                item.name))
        candidate = stream_objects[source.name].next()
        if candidate is None:
            active.remove(source.name)
            continue
        tokens, mask, selected_id, digest, method, bucket, language = candidate
        eos = np.asarray([tokenizer.eos_token_id], dtype=np.uint32)
        tokens = np.concatenate((tokens, eos))
        mask = np.concatenate((mask, np.zeros(1, dtype=np.uint8)))
        assistant = int(mask.sum())
        split_hash = int(hashlib.sha256(
            f"{source.name}:{selected_id}".encode()).hexdigest()[:16], 16)
        split = ("val" if val_fraction
                 and split_hash / float(16 ** 16) < val_fraction else "train")
        writers[split].add(tokens, mask)
        delivered[source.name] += assistant
        context[source.name] += int(tokens.size)
        selected_ids[source.name].append(selected_id)
        selected_hashes[source.name].append(digest)
        lengths[source.name].append(int(tokens.size))
        assistant_lengths[source.name].append(assistant)
        methods[method] = methods.get(method, 0) + 1
        source_buckets[source.name][bucket] = source_buckets[source.name].get(bucket, 0) + 1
        for table, label in ((bucket_counts, bucket), (language_counts, language)):
            current = table.setdefault(label, {
                "records": 0, "assistant_tokens": 0, "context_tokens": 0})
            current["records"] += 1
            current["assistant_tokens"] += assistant
            current["context_tokens"] += int(tokens.size)
        total_assistant += assistant
        records_count += 1
    if not writers["train"].total:
        raise ValueError("mixture hash split retained no training records")
    for writer in writers.values():
        writer.close()
    total_context = sum(context.values())
    largest_record_share = max(
        (max(value, default=0) / total_assistant
         for value in assistant_lengths.values()),
        default=0.0)
    permitted = max(quota_tolerance, largest_record_share)
    observed_assistant = {
        name: count / total_assistant for name, count in delivered.items()
    }
    violations = {
        source.name: abs(observed_assistant[source.name] - source.assistant_token_quota)
        for source in specs
        if abs(observed_assistant[source.name] - source.assistant_token_quota) > permitted
    }
    if violations:
        raise ValueError(f"assistant-token quota tolerance exceeded: {violations}")
    template = getattr(tokenizer, "chat_template", None) or ""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_serialized = (backend.to_str() if backend is not None else json.dumps(
        tokenizer.get_vocab(), sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    source_reports = {}
    for source in specs:
        stream = stream_objects[source.name]
        source_reports[source.name] = {
            **asdict(source),
            "selected_record_count": len(selected_ids[source.name]),
            "selected_ids": selected_ids[source.name],
            "selected_ids_sha256": hashlib.sha256(canonical_json(
                selected_ids[source.name]).encode()).hexdigest(),
            "selected_record_sha256": selected_hashes[source.name],
            "assistant_tokens": delivered[source.name],
            "assistant_token_share": observed_assistant[source.name],
            "assistant_token_share_delta": (
                observed_assistant[source.name] - source.assistant_token_quota),
            "context_tokens": context[source.name],
            "context_token_share": context[source.name] / max(total_context, 1),
            "context_token_share_delta": (
                context[source.name] / max(total_context, 1)
                - source.context_token_target),
            "length_statistics": _quantiles(lengths[source.name]),
            "rejected_records": stream.rejected,
            "deduplicated_records": stream.duplicates,
            "truncated_records": stream.truncated,
            "selected_bucket_counts": dict(sorted(source_buckets[source.name].items())),
        }
    manifest = {
        "schema": MIXTURE_MANIFEST_SCHEMA,
        "dtype": "uint32",
        "loss_mask_dtype": "uint8",
        "loss_mask": {
            "dtype": "uint8",
            "meaning": "1=assistant content/turn terminator contributes to next-token loss",
            "alignment": "one byte per token",
        },
        "seq_len": seq_len,
        "eos_id": int(tokenizer.eos_token_id),
        "vocab_size": max(int(tokenizer.vocab_size), len(tokenizer)),
        "assistant_token_budget": assistant_token_budget,
        "assistant_tokens_selected": total_assistant,
        "context_tokens_selected": total_context,
        "record_count": records_count,
        "quota_tolerance": quota_tolerance,
        "effective_quota_tolerance": permitted,
        "quota_unit": "supervised_assistant_tokens",
        "context_share_role": "recorded_activation_distribution_not_scheduler_unit",
        "deduplication": {
            "method": "sha256_canonical_normalized_messages_global",
            "unique_records": len(seen),
            "duplicates_removed": sum(
                stream.duplicates for stream in stream_objects.values()),
        },
        "tokenizer_sha256": sha256_bytes(tokenizer_serialized.encode()),
        "chat_template_sha256": sha256_bytes(template.encode()),
        "tokenizer": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "mixture_spec_sha256": mixture_spec_sha256,
        "mask_methods": dict(sorted(methods.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "sources": source_reports,
        "splits": {
            split: {
                "shards": writer.shards,
                "total_tokens": writer.total,
                "assistant_tokens": writer.assistant_tokens,
                "approx_sequences": writer.total // seq_len,
            }
            for split, writer in writers.items()
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return manifest
