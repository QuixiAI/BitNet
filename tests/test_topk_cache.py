"""Teacher top-k cache build/read + frozen-corpus hash validation (A6c rules)."""

import json
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from bitnet_train.data import PackedWindows  # noqa: E402
from bitnet_train.distill import TopkCacheReader, build_topk_cache  # noqa: E402

pytest.importorskip("transformers")


@pytest.fixture()
def shard_dir(tmp_path):
    from prepare_data import ShardWriter, pack, write_manifest
    d = tmp_path / "shards"
    d.mkdir()
    docs = [("doc %d words here " % i) * 20 for i in range(64)]
    enc = lambda ts: [[1 + zlib.crc32(w.encode()) % 500 for w in t.split()] for t in ts]
    w = {"train": ShardWriter(str(d), "train", 100_000)}
    pack(iter(docs), enc, 0, lambda i: w["train"], batch_docs=16)
    w["train"].close()
    write_manifest(str(d), "fake", 512, 0, seq_len=32, source="synthetic", writers=w)
    return d


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=1, vocab_size=512))


def test_cache_roundtrip_and_alignment(shard_dir, tmp_path):
    model = _tiny_model()
    cache = tmp_path / "cache"
    meta = build_topk_cache(model, shard_dir, cache, k=8, tau=2.0, seq_len=32,
                            limit_windows=6, batch_size=2)
    assert meta["n_windows"] == 6 and meta["k"] == 8

    reader = TopkCacheReader(cache, shard_dir, tau=2.0)
    ds = PackedWindows(shard_dir, seq_len=32)
    idx, prob = reader.batch(torch.tensor([3]), "cpu")
    with torch.no_grad():
        logits = model(ds[3].unsqueeze(0)).logits.float() / 2.0
        ref = torch.softmax(logits, -1).topk(8, -1)
    torch.testing.assert_close(idx[0], ref.indices[0].int())
    torch.testing.assert_close(prob[0], ref.values[0].half().float(),
                               rtol=1e-3, atol=1e-4)


def test_cache_rejects_moved_corpus_and_wrong_tau(shard_dir, tmp_path):
    model = _tiny_model()
    cache = tmp_path / "cache"
    build_topk_cache(model, shard_dir, cache, k=4, tau=2.0, seq_len=32,
                     limit_windows=2)
    with pytest.raises(ValueError, match="tau"):
        TopkCacheReader(cache, shard_dir, tau=1.0)

    # mutate the corpus -> manifest hash changes -> reader refuses
    man = json.loads((shard_dir / "manifest.json").read_text())
    man["splits"]["train"]["shards"][0]["sha256"] = "0" * 16
    (shard_dir / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="stale"):
        TopkCacheReader(cache, shard_dir, tau=2.0)
