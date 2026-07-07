#!/usr/bin/env python
"""Tokenize a heal corpus into packed uint32 token shards (train_plan §5.5).

Adapted from AUM train/prepare_data.py with the plan's deltas:

  * DTYPE = uint32 — the heal tokenizers (LLaMA-3 128,256 / Qwen3 ~152K /
    Gemma 262,144) do NOT fit uint16; a uint16 pipeline silently wraps half the
    vocab and trains on garbage (train_plan §5.5, moe_train_plan §5.5).
  * tokenizer = any AutoTokenizer, taken from --tokenizer or the track profile's
    data.tokenizer (per-family, non-negotiable; existing corpora with other
    tokenizers are RE-TOKENIZED, never remapped).
  * every shard records a streamed sha256 and the manifest prints/pins the
    canonical manifest_hash — the thing checkpoints and teacher caches validate
    against (§5.6; caches are invalid if data moves).

Everything else keeps AUM's shape: streaming HF sources with per-source token
budgets, EOS-separated flat shards, global val stripe, --append top-ups,
--self-test with no external deps.

    python train/prepare_data.py --profile train/profiles/a1.yaml \
        --out-dir train/data/llama3
    python train/prepare_data.py --source 'corpus/*.jsonl' --tokenizer meta-llama/Llama-3.2-1B \
        --out-dir train/data/mine --val-fraction 0.01
    python train/prepare_data.py --self-test

A shard reads back as:  np.memmap(path, dtype=np.uint32, mode="r") -> 1-D token stream.
"""

import argparse
import glob
import gzip
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DTYPE = np.uint32          # 128K/152K/262K vocabs; uint16 silently wraps (§5.5)
_MAXID = np.iinfo(DTYPE).max


# --------------------------------------------------------------------------- documents
def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def iter_local(paths, text_column):
    for path in paths:
        is_jsonl = ".jsonl" in path or ".ndjson" in path
        with _open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield _extract_text(json.loads(line), text_column) if is_jsonl else line


_PREFERRED_CONFIGS = ("eng_Latn", "en", "eng", "english", "default", "sample-10BT")
_TEXT_KEYS = ("text", "content", "markdown", "raw_content", "document", "chapter")
_LANG_KEYS = ("language", "lang", "language_code", "language_script")


def _require_datasets():
    try:
        import datasets  # noqa: F401
        return datasets
    except ImportError as e:  # pragma: no cover
        raise SystemExit("datasets is required for HF sources: pip install datasets") from e


def _enable_hf_transfer():
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass


def _lang_ok(example, language):
    if language in (None, "", "all"):
        return True
    for k in _LANG_KEYS:
        v = example.get(k)
        if isinstance(v, str) and v:
            return v.lower().replace("-", "_").startswith(language.lower())
    return True


def _load_stream(name, split="train", config=None, skip=0):
    hf = _require_datasets()

    def _load(config):
        try:
            ds = hf.load_dataset(name, config, split=split, streaming=True)
        except ValueError as e:
            if "split" not in str(e).lower():
                raise
            avail = hf.get_dataset_split_names(name, config)
            pick = next((s for s in avail if "train" in s), avail[0])
            print(f"  [{name}] split {split!r} missing; using {pick!r} (available: {avail})")
            ds = hf.load_dataset(name, config, split=pick, streaming=True)
        return ds.skip(skip) if skip else ds

    if config is not None:
        return _load(config), config
    try:
        return _load(None), None
    except ValueError as e:
        if "Config name is missing" not in str(e) and "BuilderConfig" not in str(e):
            raise
    configs = hf.get_dataset_config_names(name)
    chosen = next((p for p in _PREFERRED_CONFIGS if p in configs), None)
    if chosen is None:
        raise SystemExit(
            f"[{name}] has {len(configs)} configs and none matched the preferences "
            f"{_PREFERRED_CONFIGS}. Pin one explicitly as '{name}:<config>' "
            f"(e.g. from: {configs[:8]}{' ...' if len(configs) > 8 else ''}).")
    print(f"  [{name}] {len(configs)} configs; using {chosen!r}")
    return _load(chosen), chosen


def _extract_text(example, text_column=None):
    if text_column and isinstance(example.get(text_column), str):
        return example[text_column]
    for k in _TEXT_KEYS:
        if isinstance(example.get(k), str):
            return example[k]
    strings = [v for v in example.values() if isinstance(v, str)]
    return max(strings, key=len) if strings else ""


def iter_hf(source, split, text_column, limit=0, config=None, language="en", skip=0):
    ds, _ = _load_stream(source, split, config, skip=skip)
    n = skipped = 0
    for ex in ds:
        if not _lang_ok(ex, language):
            skipped += 1
            continue
        yield _extract_text(ex, text_column)
        n += 1
        if limit and n >= limit:
            break
    if skipped:
        print(f"  [{source}] skipped {skipped:,} non-{language} documents (language filter)")


def iter_documents(source, split, text_column, limit, language="en", skip=0):
    matches = glob.glob(source)
    if matches:
        it = iter_local(sorted(matches), text_column)
    else:
        name, _, config = source.partition(":")
        it = iter_hf(name, split, text_column, limit, config=config or None,
                     language=language, skip=skip)
    for i, doc in enumerate(it):
        if limit and i >= limit:
            return
        if doc:
            yield doc


def read_dataset_list(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            name, _, config = line.partition(":")
            entries.append((name, config or None))
    return entries


# --------------------------------------------------------------------------- sharding
class ShardWriter:
    """Accumulate token arrays and flush flat uint32 shards of ~shard_size tokens
    each, streaming a sha256 per shard (the manifest-hash input)."""

    def __init__(self, out_dir, split, shard_size):
        self.out_dir, self.split, self.shard_size = out_dir, split, shard_size
        self.buf, self.buf_len, self.shards, self.total = [], 0, [], 0

    def add(self, arr):
        self.buf.append(arr)
        self.buf_len += arr.size
        self.total += arr.size
        if self.buf_len >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.buf_len:
            return
        arr = np.concatenate(self.buf).astype(DTYPE)
        name = f"{self.split}_{len(self.shards):05d}.bin"
        raw = arr.tobytes()
        with open(os.path.join(self.out_dir, name), "wb") as f:
            f.write(raw)
        self.shards.append({"name": name, "n_tokens": int(arr.size),
                            "sha256": hashlib.sha256(raw).hexdigest()[:16]})
        self.buf, self.buf_len = [], 0

    def close(self):
        self._flush()


def pack(doc_iter, encode_fn, eos_id, writer_for, batch_docs, token_budget=0, progress=None):
    idx_buf, txt_buf, buf_chars, n_docs, n_tokens = [], [], 0, 0, 0
    max_buf_chars = 16_000_000

    def flush():
        nonlocal n_docs, n_tokens, buf_chars
        if not txt_buf:
            return
        before = n_tokens
        for i, ids in zip(idx_buf, encode_fn(txt_buf)):
            if not len(ids):
                continue
            a = np.asarray(ids, dtype=np.int64)
            if int(a.max()) >= _MAXID:
                raise ValueError(f"token id {int(a.max())} does not fit {DTYPE}")
            a = np.append(a, eos_id)
            writer_for(i).add(a.astype(DTYPE))
            n_docs += 1
            n_tokens += a.size
        idx_buf.clear(); txt_buf.clear(); buf_chars = 0
        if progress is not None:
            progress(n_tokens - before)

    for i, doc in enumerate(doc_iter):
        idx_buf.append(i); txt_buf.append(doc); buf_chars += len(doc)
        if len(txt_buf) >= batch_docs or buf_chars >= max_buf_chars:
            flush()
            if token_budget and n_tokens >= token_budget:
                break
    flush()
    return n_docs, n_tokens


def write_manifest(out_dir, tokenizer, vocab_size, eos_id, seq_len, source, writers):
    splits = {}
    for split, w in writers.items():
        splits[split] = {"shards": w.shards, "total_tokens": w.total,
                         "approx_sequences": w.total // seq_len if seq_len else None}
    manifest = {
        "tokenizer": tokenizer, "vocab_size": vocab_size, "dtype": "uint32",
        "eos_id": int(eos_id), "seq_len": seq_len, "source": source,
        "splits": splits, "created_unix": int(time.time()),
        "read_hint": "np.memmap(<out_dir>/<shard>, dtype=np.uint32, mode='r') -> 1-D token stream",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# --------------------------------------------------------------------------- driver
def _setup(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = args.vocab_size or max(tok.vocab_size, len(tok))
    if vocab_size > _MAXID:
        raise SystemExit(f"vocab {vocab_size} exceeds uint32?!")
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    if eos_id is None:
        raise SystemExit(f"{args.tokenizer} has neither eos nor pad token")

    existing = None
    if getattr(args, "append", False):
        mpath = os.path.join(args.out_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise SystemExit(f"--append: no manifest.json in {args.out_dir}")
        existing = json.load(open(mpath))
        if existing["vocab_size"] != vocab_size or existing["eos_id"] != eos_id:
            raise SystemExit("--append: tokenizer/vocab/eos mismatch with the existing manifest")

    def encode_fn(texts):
        return tok(texts, add_special_tokens=False)["input_ids"]

    os.makedirs(args.out_dir, exist_ok=True)
    splits = {"train"} | ({"val"} if args.val_fraction > 0 else set()) \
        | (set(existing["splits"]) if existing else set())
    writers = {s: ShardWriter(args.out_dir, s, args.shard_size_tokens) for s in sorted(splits)}
    if existing:
        for s, w in writers.items():
            prior = existing["splits"].get(s)
            if prior:
                w.shards = list(prior["shards"])
                w.total = prior["total_tokens"]
    if args.val_fraction > 0:
        every = max(2, round(1.0 / args.val_fraction))
        counter = itertools.count()                      # GLOBAL stripe
        writer_for = lambda _i: writers["val" if next(counter) % every == 0 else "train"]
    else:
        writer_for = lambda _i: writers["train"]
    return vocab_size, eos_id, encode_fn, writers, writer_for, existing


def _finish(args, vocab_size, eos_id, writers, source, n_docs):
    from bitnet_train.data import manifest_hash

    for w in writers.values():
        w.close()
    m = write_manifest(args.out_dir, args.tokenizer, vocab_size, eos_id, args.seq_len,
                       source, writers)
    print(f"tokenized {n_docs:,} docs -> {args.out_dir}")
    for split, s in m["splits"].items():
        print(f"  {split:5s}: {s['total_tokens']:>14,} tokens  "
              f"~{s['approx_sequences']:>10,} x {args.seq_len}-seqs  ({len(s['shards'])} shards)")
    print(f"  manifest_hash: {manifest_hash(m)}  (pinned by checkpoints + teacher caches)")


def run(args):
    vocab_size, eos_id, encode_fn, writers, writer_for, existing = _setup(args)
    budget = args.chunks_per_dataset * args.seq_len if args.chunks_per_dataset else 0
    docs = iter_documents(args.source, args.split, args.text_column, args.limit_docs,
                          language=args.language, skip=args.skip_docs)
    n_docs, n_tok = pack(docs, encode_fn, eos_id, writer_for, args.batch_docs,
                         token_budget=budget)
    source = args.source
    if existing is not None:
        source = existing["source"] if isinstance(existing["source"], dict) else \
            {"base": existing["source"]}
        source.setdefault("appended", []).append(
            {"source": args.source, "documents": n_docs, "tokens": n_tok,
             "skip_docs": args.skip_docs})
    _finish(args, vocab_size, eos_id, writers, source, n_docs)


def run_mix(args):
    names = read_dataset_list(args.datasets_file)
    if not names:
        raise SystemExit(f"no datasets listed in {args.datasets_file}")
    if args.append:
        raise SystemExit("--append works with --source (single-source top-ups), not mix mode")
    vocab_size, eos_id, encode_fn, writers, writer_for, _ = _setup(args)
    budget = args.chunks_per_dataset * args.seq_len
    print(f"corpus: {args.chunks_per_dataset:,} x {args.seq_len}-token chunks "
          f"(~{budget / 1e6:.0f}M tokens) from each of {len(names)} datasets "
          f"({args.datasets_file}) -> {args.out_dir}")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    per_source, n_total = {}, 0
    for si, (name, config) in enumerate(names):
        label = f"{name}:{config}" if config else name
        print(f"[{si + 1}/{len(names)}] {label}")
        before = {s: w.total for s, w in writers.items()}
        docs = iter_hf(name, args.split, args.text_column,
                       limit=args.samples_per_dataset, config=config, language=args.language)
        bar = tqdm(total=budget, unit="tok", unit_scale=True, desc=label,
                   dynamic_ncols=True) if tqdm is not None else None
        n, n_tok = pack((d for d in docs if d), encode_fn, eos_id, writer_for, args.batch_docs,
                        token_budget=budget, progress=bar.update if bar else None)
        if bar:
            bar.close()
        tokens = {s: writers[s].total - before[s] for s in writers}
        per_source[label] = {"documents": n, "tokens": tokens}
        n_total += n
        short = f"  (stream ended below the {budget / 1e6:.0f}M-token budget)" \
            if budget and n_tok < budget else ""
        print(f"  {n:,} docs, {sum(tokens.values()):,} tokens{short}")

    source = {"datasets_file": args.datasets_file, "chunks_per_dataset": args.chunks_per_dataset,
              "token_budget_per_dataset": budget, "per_source": per_source}
    _finish(args, vocab_size, eos_id, writers, source, n_total)


def self_test():
    """Exercise pack/shard/manifest + hash stability with synthetic data (numpy only)."""
    import shutil
    import zlib

    from bitnet_train.data import PackedWindows, manifest_hash

    vocab, eos = 128_256, 0                              # a uint32-requiring vocab
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_selftest_shards")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    docs = [("token sample document number %d " % d) * (3 + d % 7) for d in range(500)]
    enc = lambda texts: [[1 + zlib.crc32(w.encode()) % (vocab - 1) for w in t.split()] for t in texts]

    writers = {"train": ShardWriter(tmp, "train", shard_size=4000),
               "val": ShardWriter(tmp, "val", shard_size=4000)}
    every = 20
    n, n_tok = pack(iter(docs), enc, eos, lambda i: writers["val" if i % every == 0 else "train"],
                    batch_docs=64)
    for w in writers.values():
        w.close()
    m = write_manifest(tmp, "self-test-fake", vocab, eos, seq_len=128, source="synthetic",
                       writers=writers)

    for split, w in writers.items():
        counted = sum(s["n_tokens"] for s in w.shards)
        assert counted == w.total, (split, counted, w.total)
        stream = np.concatenate([np.memmap(os.path.join(tmp, s["name"]), dtype=DTYPE, mode="r")
                                 for s in w.shards]) if w.shards else np.array([], DTYPE)
        assert stream.size == w.total
        assert stream.max(initial=0) < vocab and (stream == eos).any()
        assert stream.max(initial=0) > np.iinfo(np.uint16).max  # ids REQUIRE uint32
    assert m["splits"]["train"]["total_tokens"] > m["splits"]["val"]["total_tokens"] > 0
    assert n == 500 and n_tok == sum(w.total for w in writers.values())

    # the loader round-trip: dtype from manifest, int64 out, deterministic hash
    ds = PackedWindows(tmp, split="train", seq_len=128)
    win = ds[0]
    assert win.dtype == torch_int64() and win.shape == (128,)
    h1 = manifest_hash(m)
    m2 = json.load(open(os.path.join(tmp, "manifest.json")))
    m2["created_unix"] = 0                               # timestamps excluded from the hash
    assert manifest_hash(m2) == h1
    shutil.rmtree(tmp)
    print(f"self-test OK: {n} docs -> uint32 shards, ids beyond uint16 range, "
          f"loader round-trip, manifest_hash {h1} timestamp-stable")


def torch_int64():
    import torch
    return torch.int64


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Tokenize a heal corpus into packed uint32 shards.")
    ap.add_argument("--source", default=None,
                    help="a single HF dataset id OR a local path/glob (.txt/.jsonl[.gz]); "
                         "omit to use --datasets-file")
    ap.add_argument("--datasets-file", default=os.path.join(here, "datasets"))
    ap.add_argument("--profile", default=None,
                    help="track profile yaml; supplies the tokenizer (data.tokenizer)")
    ap.add_argument("--chunks-per-dataset", type=int, default=15_000,
                    help="TOKEN budget per dataset in seq-len chunks (0 = no budget)")
    ap.add_argument("--samples-per-dataset", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(here, "data"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-column", default=None)
    ap.add_argument("--language", default="en",
                    help="language-metadata filter prefix ('all' disables). NOTE: Qwen3 is "
                         "heavily multilingual — a pure-English heal corpus is a mild domain "
                         "shift; record it in the track report (moe_train_plan §5.5)")
    ap.add_argument("--tokenizer", default=None, help="AutoTokenizer id (or via --profile)")
    ap.add_argument("--vocab-size", type=int, default=0, help="0 = from the tokenizer")
    ap.add_argument("--shard-size-tokens", type=int, default=100_000_000)
    ap.add_argument("--batch-docs", type=int, default=1000)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--skip-docs", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.tokenizer is None:
        if args.profile is None:
            raise SystemExit("pass --tokenizer or --profile")
        from bitnet_train.conversion import load_profile
        args.tokenizer = load_profile(args.profile).data["tokenizer"]
    _enable_hf_transfer()
    if args.source:
        run(args)
    else:
        run_mix(args)


if __name__ == "__main__":
    main()
