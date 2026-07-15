#!/usr/bin/env python
"""Build chat-template-exact SmolTalk/local instruction shards with loss masks.

Example:
  .venv/bin/python train/prepare_instruct_data.py \
    --source HuggingFaceTB/smoltalk --config all \
    --tokenizer unsloth/Llama-3.2-1B-Instruct \
    --output train/data/llama32_instruct --revision <immutable-revision>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.data import manifest_hash  # noqa: E402
from bitnet_train.tq1.instruct_data import (  # noqa: E402
    build_masked_shards, write_masked_manifest)


def _local_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield value


def _dataset_records(args):
    from datasets import load_dataset
    dataset = load_dataset(
        args.source, args.config, split=args.split, revision=args.revision,
        streaming=True)
    if args.shuffle_buffer:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    yield from dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render instruction conversations and assistant-only masks")
    parser.add_argument("--source", default="HuggingFaceTB/smoltalk",
                        help="HF dataset id or a local JSONL file")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", required=True,
                        help="immutable dataset revision (or a local-file SHA-256 label)")
    parser.add_argument("--tokenizer", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--shard-size-tokens", type=int, default=25_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.tokenizer_revision,
            local_files_only=args.local_files_only, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("assistant mask construction requires a fast tokenizer")
        local = Path(args.source).is_file()
        records = _local_records(Path(args.source)) if local else _dataset_records(args)
        splits, statistics = build_masked_shards(
            records, tokenizer, output, seq_len=args.seq_len,
            shard_size=args.shard_size_tokens, val_fraction=args.val_fraction,
            limit=args.limit_records)
        source = {
            "id": str(Path(args.source).resolve()) if local else args.source,
            "config": args.config,
            "split": args.split,
            "revision": args.revision,
            "shuffle_seed": args.seed,
            "shuffle_buffer": args.shuffle_buffer,
        }
        manifest = write_masked_manifest(
            output, tokenizer_id=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision, tokenizer=tokenizer,
            seq_len=args.seq_len, source=source, splits=splits,
            statistics=statistics)
        print(json.dumps({
            "output": str(output), "manifest_hash": manifest_hash(manifest),
            **statistics,
        }, indent=2, sort_keys=True))
        return 0
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
