import json
import hashlib

import numpy as np
import pytest

from bitnet_train.tq1.mixture import (
    CAPABILITY_BUCKETS, MIXTURE_MANIFEST_SCHEMA, MixtureSource,
    build_quota_mixture)


class _Tokenizer:
    eos_token_id = 0
    vocab_size = 256
    chat_template = "mixture-template-v1"

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return {chr(index): index for index in range(256)}

    def apply_chat_template(self, messages, *, tokenize,
                            add_generation_prompt=False, return_dict=False,
                            return_assistant_tokens_mask=False):
        del add_generation_prompt, return_assistant_tokens_mask
        text = "".join(f"<{item['role']}>{item['content']}|" for item in messages)
        ids = [ord(char) for char in text]
        if return_dict:
            return {"input_ids": ids}
        return ids if tokenize else text

    def __call__(self, text, *, add_special_tokens=False,
                 return_offsets_mapping=False):
        del add_special_tokens
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1)
                                        for index in range(len(text))]
        return result


def test_assistant_token_quota_mixture_is_reproducible_and_complete(tmp_path):
    capabilities = sorted(CAPABILITY_BUCKETS)
    share = 1 / len(capabilities)
    sources = [MixtureSource(
        name=name, dataset=f"local/{name}",
        revision=hashlib.sha256(name.encode()).hexdigest(),
        license="Apache-2.0", split="train", config=None, capability=name,
        language="en" if name != "multilingual" else "multi",
        assistant_token_quota=share, context_token_target=share)
        for name in capabilities]
    records = {
        source.name: [
            {"id": f"{source.name}-{index}", "messages": [
                {"role": "user", "content": f"question-{source.name}-{index}"},
                {"role": "assistant", "content": "answer"},
            ]}
            for index in range(8)
        ] for source in sources
    }
    manifest = build_quota_mixture(
        sources, records, _Tokenizer(), tmp_path, seq_len=128, shard_size=256,
        assistant_token_budget=140, val_fraction=0.0, quota_tolerance=0.05,
        tokenizer_id="unit", tokenizer_revision="a" * 40,
        mixture_spec_sha256="b" * 64)
    assert manifest["schema"] == MIXTURE_MANIFEST_SCHEMA
    assert manifest["quota_unit"] == "supervised_assistant_tokens"
    assert manifest["loss_mask"]["dtype"] == "uint8"
    assert set(manifest["sources"]) == set(capabilities)
    assert manifest["deduplication"]["unique_records"] == manifest["record_count"]
    assert manifest["deduplication"]["duplicates_removed"] == 0
    assert sum(item["assistant_tokens"] for item in manifest["sources"].values()) \
        == manifest["assistant_tokens_selected"]
    assert sum(item["context_tokens"] for item in manifest["sources"].values()) \
        == manifest["context_tokens_selected"]
    assert sum(item["records"] for item in manifest["bucket_counts"].values()) \
        == manifest["record_count"]
    assert sum(item["assistant_tokens"]
               for item in manifest["language_counts"].values()) \
        == manifest["assistant_tokens_selected"]
    for report in manifest["sources"].values():
        assert report["selected_ids"]
        assert report["license"] == "Apache-2.0"
        assert report["length_statistics"]["max"] <= 129
        assert report["assistant_token_share"] == pytest.approx(share, abs=0.05)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    shard = manifest["splits"]["train"]["shards"][0]
    assert np.fromfile(tmp_path / shard["loss_mask_name"], dtype=np.uint8).any()


def test_mixture_globally_deduplicates_normalized_conversations(tmp_path):
    capabilities = sorted(CAPABILITY_BUCKETS)
    share = 1 / len(capabilities)
    sources = [MixtureSource(
        name=name, dataset=name, revision="a" * 40, license="test",
        split="train", config=None, capability=name, language="en",
        assistant_token_quota=share, context_token_target=share)
        for name in capabilities]
    duplicate = {"messages": [{"role": "user", "content": "same"},
                               {"role": "assistant", "content": "same"}]}
    records = {source.name: [duplicate] + [
        {"messages": [{"role": "user", "content": f"{source.name}-{index}"},
                      {"role": "assistant", "content": "ok"}]}
        for index in range(20)] for source in sources}
    manifest = build_quota_mixture(
        sources, records, _Tokenizer(), tmp_path, seq_len=128, shard_size=1024,
        assistant_token_budget=140, val_fraction=0.0, quota_tolerance=0.08)
    assert sum(item["deduplicated_records"]
               for item in manifest["sources"].values()) >= 1
    assert manifest["deduplication"]["duplicates_removed"] >= 1
