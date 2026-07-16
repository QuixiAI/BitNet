#!/usr/bin/env python
"""Export an export-qualified frozen TQ1 training checkpoint to schema 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitnet_train.conversion import load_converted, load_profile  # noqa: E402
from bitnet_train.tq1.pipeline import export_qat_model  # noqa: E402


def checkpoint_identity(directory: Path) -> str:
    digest = hashlib.sha256()
    names = ("model.safetensors", "model.safetensors.index.json",
             "pytorch_model.bin", "pytorch_model.bin.index.json", "trainer_state.pt")
    found = False
    for name in names:
        path = directory / name
        if path.is_file():
            found = True
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    for path in sorted(directory.glob("model-*.safetensors")):
        found = True
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    if not found:
        raise FileNotFoundError(f"no HF checkpoint weights in {directory}")
    return digest.hexdigest()


def _restore_exact_state(model, directory: Path) -> None:
    from transformers.modeling_utils import load_sharded_checkpoint
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        result = load_sharded_checkpoint(model, directory, strict=False, prefer_safe=True)
        unexpected = set(result.unexpected_keys)
        if unexpected:
            raise ValueError(f"unexpected TQ1 checkpoint keys {sorted(unexpected)[:8]}")
        return
    safe = directory / "model.safetensors"
    if safe.is_file():
        from safetensors.torch import load_file
        state = load_file(str(safe), device="cpu")
    else:
        state = torch.load(directory / "pytorch_model.bin", map_location="cpu",
                           weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected TQ1 checkpoint keys {result.unexpected_keys[:8]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-report", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    state_path = checkpoint / "trainer_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    trainer_state = torch.load(state_path, map_location="cpu", weights_only=False)
    controller = trainer_state.get("training_state", {}).get("tq1_controller")
    if not controller or controller.get("schema") != 2 \
            or controller.get("domain") != "global_tokens" \
            or controller.get("phase") != "frozen" \
            or not controller.get("export_qualified"):
        raise ValueError(
            "checkpoint lacks a token-domain, frozen, export-qualified controller")
    profile = load_profile(args.profile)
    model, _ = load_converted(
        checkpoint, profile, backend="reference", dtype=torch.float32,
        tq1_artifact=args.source_artifact)
    _restore_exact_state(model, checkpoint)
    evaluation = (json.loads(Path(args.evaluation_report).read_text())
                  if args.evaluation_report else None)
    identity = checkpoint_identity(checkpoint)
    output = export_qat_model(
        model, args.source_artifact, args.output, source_files=checkpoint,
        checkpoint_identity=identity, evaluation_report=evaluation,
        overwrite=args.overwrite,
        command=("train/export_tq1.py", *(argv if argv is not None else sys.argv[1:])))
    print(json.dumps({
        "artifact": str(output), "checkpoint_sha256": identity,
        "quality_qualified": evaluation is not None,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
