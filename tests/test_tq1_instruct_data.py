import json

import numpy as np
import torch

from bitnet_train.data import PackedWindows, manifest_hash
from bitnet_train.tq1.instruct_data import (
    build_masked_shards, normalize_messages, render_assistant_mask,
    write_masked_manifest)


class _Tokenizer:
    eos_token_id = 0
    vocab_size = 256
    chat_template = "prefix-stable-test-template"
    is_fast = True

    def __len__(self):
        return self.vocab_size

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False,
                            return_dict=False, return_assistant_tokens_mask=False):
        del add_generation_prompt
        text = "<B>" + "".join(
            f"<{item['role']}>{item['content']}<E>" for item in messages)
        if not tokenize:
            return text
        ids = [ord(char) for char in text]
        if return_dict:
            # Exercise the strict prefix/offset fallback by omitting native masks.
            return {"input_ids": ids}
        return ids

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def test_message_normalization_and_exact_assistant_spans():
    messages = normalize_messages({"conversations": [
        {"from": "human", "value": "question"},
        {"from": "gpt", "value": "answer"},
        {"from": "human", "value": "again"},
        {"from": "assistant", "value": "done"},
    ]})
    ids, mask, method = render_assistant_mask(_Tokenizer(), messages)
    text = "".join(chr(int(value)) for value in ids)
    selected = "".join(char for char, keep in zip(text, mask) if keep)
    assert method == "prefix_offsets_v1"
    assert selected == "answer<E>done<E>"


def test_masked_shards_roundtrip_through_packed_windows(tmp_path):
    records = [
        {"messages": [{"role": "user", "content": f"q{i}"},
                      {"role": "assistant", "content": f"a{i}"}]}
        for i in range(5)
    ]
    tokenizer = _Tokenizer()
    splits, stats = build_masked_shards(
        records, tokenizer, tmp_path, seq_len=64, shard_size=80,
        val_fraction=0.0)
    manifest = write_masked_manifest(
        tmp_path, tokenizer_id="tiny", tokenizer_revision="abc", tokenizer=tokenizer,
        seq_len=32, source={"id": "test", "revision": "1"}, splits=splits,
        statistics=stats)
    assert manifest_hash(manifest) == manifest_hash(json.loads(
        (tmp_path / "manifest.json").read_text()))
    ds = PackedWindows(tmp_path, split="train", seq_len=32)
    tokens, mask = ds[0]
    assert tokens.dtype == torch.int64
    assert tuple(tokens.shape) == tuple(mask.shape) == (32,)
    assert mask.dtype == torch.bool and bool(mask.any())
    shard = manifest["splits"]["train"]["shards"][0]
    assert len((tmp_path / shard["loss_mask_name"]).read_bytes()) == shard["n_tokens"]
    assert np.fromfile(tmp_path / shard["loss_mask_name"], dtype=np.uint8).max() == 1
