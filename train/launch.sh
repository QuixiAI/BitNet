#!/usr/bin/env bash
# Launch the BitNet healing trainer (adapted from ~/AUM/train/launch.sh).
# CUDA -> accelerate launch across all GPUs, bf16; MPS/Mac -> single process.
# Everything after the config passes through to train/train.py (--set KEY=VALUE wins last).
#
#   train/launch.sh train/configs/a1/t1.yaml --init runs/a1-init --set lr=2e-4
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:?usage: train/launch.sh <config.yaml> [train.py args...]}"
shift

PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then PY=python3; fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
  echo "launch: CUDA x${NGPU}, accelerate + bf16"
  exec "$PY" -m accelerate.commands.launch --num_processes "$NGPU" \
      train/train.py --config "$CONFIG" --mixed-precision bf16 "$@"
else
  echo "launch: single process ($("$PY" -c 'import torch; print("mps" if torch.backends.mps.is_available() else "cpu")'))"
  exec "$PY" train/train.py --config "$CONFIG" "$@"
fi
