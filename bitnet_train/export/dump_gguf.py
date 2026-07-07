#!/usr/bin/env python
"""GGUF tensor-list dump (train_plan §3.1 / §11.1 recon instrument): print every
tensor's name, ggml type, shape, and the architecture string — the ground truth
for what a converter/runtime accepts. The plan's A-track T0 action item is to
dump the released Llama3-8B-1.58 GGUF and match it; this is that tool.

  python -m bitnet_train.export.dump_gguf model.gguf
  python -m bitnet_train.export.dump_gguf model.gguf --types      # type histogram
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from bitnet_train.export.compare_gguf import _import_gguf


def dump(path: str, show_types: bool = False) -> dict:
    gguf = _import_gguf()
    r = gguf.GGUFReader(path)

    def field_str(key):
        f = r.get_field(key)
        return bytes(f.parts[f.data[0]]).decode() if f else None

    arch = field_str("general.architecture")
    tensors = [(t.name, t.tensor_type.name, tuple(int(s) for s in t.shape))
               for t in r.tensors]
    types = Counter(t[1] for t in tensors)
    print(f"architecture: {arch}   tensors: {len(tensors)}")
    if show_types:
        for ty, n in types.most_common():
            print(f"  {ty:10s} x{n}")
    else:
        for name, ty, shape in tensors:
            print(f"  {name:48s} {ty:10s} {shape}")
    return {"arch": arch, "n_tensors": len(tensors), "types": dict(types),
            "tensors": tensors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--types", action="store_true", help="type histogram only")
    args = ap.parse_args()
    dump(args.gguf, args.types)
    return 0


if __name__ == "__main__":
    sys.exit(main())
