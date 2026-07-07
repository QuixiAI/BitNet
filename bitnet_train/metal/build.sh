#!/usr/bin/env bash
# Standalone build/verify for the vendored BitNet-training Metal kernels.
# Mirrors QuixiCore-Metal's build discipline: compile the .metal sources into a metallib
# with `xcrun metal` (no MLX, no CMake), independent of the PyTorch JIT path.
#
# Usage:
#   ./build.sh            # compile all kernels -> bitnet.metallib, list functions
#   ./build.sh check      # also print the Metal toolchain versions
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
include="${root}/include/metal"
out="${root}/tk_torch/bitnet.metallib"

if [ "${1:-}" = "check" ]; then
  echo "== Metal toolchain =="
  xcrun --find metal || { echo "Metal toolchain missing: xcodebuild -downloadComponent MetalToolchain"; exit 2; }
  echo
fi

echo "== Compiling kernels -> ${out} =="
# shellcheck disable=SC2046
find "${root}/kernels" -name '*.metal' -print0 \
  | xargs -0 xcrun metal -std=metal3.1 -O2 -I "${include}" -o "${out}"
echo "built $(du -h "${out}" | cut -f1) metallib"
echo
echo "== Functions in metallib (BitNet + training subset) =="
xcrun metal-nm "${out}" 2>/dev/null \
  | awk '{print $NF}' \
  | grep -iE 'w2a8|bitnet|quant.*int8|rms_norm|cross_entropy|adamw|matmul|swiglu' \
  | sort -u \
  || echo "(metal-nm not available; metallib built OK)"
