#!/usr/bin/env python3
"""Build a deterministic tiny schema-2 TQ1 Llama GGUF for loader testing."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import torch


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from bitnet_train.tq1.codebook import (  # noqa: E402
    CodebookRegistry, sign_canonical_codebook)
from bitnet_train.tq1.gguf import export_tq1_gguf  # noqa: E402
from bitnet_train.tq1.pipeline import (  # noqa: E402
    LLAMA_KEEP_FP_REGEXES, LLAMA_TARGET_REGEXES, run_full_model_ptq)
from bitnet_train.tq1.solver import canonical_shapes  # noqa: E402
from bitnet_train.tq1.spec import QuantSpec  # noqa: E402


def _write_sentencepiece_tokenizer(directory: Path) -> int:
    import sentencepiece as spm

    corpus = directory / "fixture_corpus.txt"
    corpus.write_text("\n".join(
        f"tiny deterministic llama fixture token row {i} alpha beta gamma delta"
        for i in range(128)) + "\n")
    prefix = directory / "fixture_spm"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=64,
        model_type="unigram",
        character_coverage=1.0,
        hard_vocab_limit=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        minloglevel=2,
    )
    tokenizer_model = directory / "tokenizer.model"
    (directory / "fixture_spm.model").replace(tokenizer_model)
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 64,
        "pad_token": "<pad>",
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": "<unk>",
    }
    (directory / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n")
    (directory / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<s>", "eos_token": "</s>",
        "pad_token": "<pad>", "unk_token": "<unk>",
    }, indent=2, sort_keys=True) + "\n")
    return processor.vocab_size()


def build_fixture(output: Path, converter: Path) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    vocab_size = _write_sentencepiece_tokenizer(source)
    torch.manual_seed(20260715)
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        vocab_size=vocab_size,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).float().eval()
    model.config.architectures = ["LlamaForCausalLM"]
    model.config.save_pretrained(source)

    shapes = canonical_shapes()
    zero = shapes[(shapes == 0).all(1)]
    nonzero = shapes[~(shapes == 0).all(1)][:1023]
    codebook = sign_canonical_codebook(
        "tiny_loader_v11j", "v11", torch.cat((zero, nonzero)), scope="model")
    spec = QuantSpec.core(
        default_profile="tq1_v11-j-r",
        codebook=codebook.ref(),
        target_regexes=LLAMA_TARGET_REGEXES,
        keep_fp_regexes=LLAMA_KEEP_FP_REGEXES,
        activation_mode="a8_token",
        importance_mode="uniform",
    )
    spec = replace(
        spec, weight_metric="uniform", candidate_count=4,
        alternating_iterations=2)
    artifact = run_full_model_ptq(
        model,
        spec,
        CodebookRegistry({codebook.id: codebook}),
        output_dir=output / "artifact",
        source_model="bitnet/tiny-loader-fixture",
        source_revision="fixture-20260715",
        source_files=source,
        chunk_groups=256,
        command=("make_tiny_fixture.py",),
    )
    gguf = output / "tiny-tq1-v11-r.gguf"
    report = export_tq1_gguf(
        artifact,
        gguf,
        converter=converter,
        python=sys.executable,
        command=("make_tiny_fixture.py",),
    )
    summary = {
        "artifact": str(artifact),
        "gguf": str(gguf),
        "gguf_sha256": report["gguf_sha256"],
        "quant_spec_sha256": report["quant_spec_sha256"],
        "target_tensors": report["target_tensors"],
        "vocab_size": vocab_size,
    }
    (output / "fixture.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return gguf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    converter = args.converter.expanduser().resolve()
    if not converter.is_file():
        raise FileNotFoundError(converter)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    gguf = build_fixture(output, converter)
    print(gguf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
