#!/bin/sh
# Build libbitnet_cpu.dylib (macOS) / .so. NEON paths compile automatically on arm64.
set -e
cd "$(dirname "$0")"
case "$(uname)" in
  Darwin) OUT=libbitnet_cpu.dylib ;;
  *)      OUT=libbitnet_cpu.so ;;
esac
cc -O3 -std=c11 -Wall -shared -fPIC -o "$OUT" src/bitnet_cpu.c -lm
echo "built $OUT"
