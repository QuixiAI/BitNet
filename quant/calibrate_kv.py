#!/usr/bin/env python
"""Build a linked KV key-channel-mean artifact from captured key tensors.

The input is a safetensors file containing exactly ``layer.<N>.key`` tensors in
explicit ``[batch, kv_head, token, channel]`` layout and, optionally, a shared
boolean ``token_mask`` tensor shaped ``[batch, token]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.tq1.kv_cache import (  # noqa: E402
    KVCalibrationContract, KVMeanCollector, file_sha256, save_kv_calibration)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured-keys", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--kv-dtype", choices=("float16", "bfloat16", "float32"),
                        required=True)
    parser.add_argument("--rotation-state", choices=("pre_rope", "post_rope"),
                        required=True)
    parser.add_argument("--attention-implementation", required=True)
    parser.add_argument("--context-length", action="append", type=int, required=True)
    parser.add_argument("--record-count", type=int, required=True)
    parser.add_argument("--source-sha256", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    capture_path = Path(args.captured_keys).expanduser().resolve()
    tensors = load_file(str(capture_path), device="cpu")
    expected_keys = {f"layer.{index}.key" for index in range(args.layer_count)}
    allowed_keys = expected_keys | {"token_mask"}
    if set(tensors) != allowed_keys and set(tensors) != expected_keys:
        raise ValueError("captured KV tensor inventory is incomplete or has unknown tensors")
    mask = tensors.get("token_mask")
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError("captured token_mask must be boolean")
    collector = KVMeanCollector(args.layer_count, args.num_kv_heads, args.head_dim)
    observed_dtype = None
    for layer in range(args.layer_count):
        value = tensors[f"layer.{layer}.key"]
        dtype = str(value.dtype).removeprefix("torch.")
        if observed_dtype is None:
            observed_dtype = dtype
        if dtype != observed_dtype or dtype != args.kv_dtype:
            raise ValueError("captured KV dtype differs from the declared contract")
        collector.add(layer, value, mask)
    token_count = int(collector.counts[0])
    sources = tuple(args.source_sha256 or [file_sha256(capture_path)])
    contract = KVCalibrationContract(
        model_artifact_sha256=args.model_artifact_sha256,
        model_id=args.model_id, model_revision=args.model_revision,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        layer_count=args.layer_count, num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim, kv_dtype=args.kv_dtype,
        rotation_state=args.rotation_state,
        attention_implementation=args.attention_implementation,
        context_lengths=tuple(sorted(set(args.context_length))),
        record_count=args.record_count, token_count=token_count,
        source_sha256=sources)
    link = save_kv_calibration(
        args.output, collector.means(expected_token_count=token_count), contract,
        overwrite=args.overwrite)
    print(json.dumps({"output": str(Path(args.output).resolve()), **link},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
