"""Canonical TQ1 codebooks, identity streams, and legal-index validation."""

from __future__ import annotations

import hashlib
import re
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch

from .spec import CodebookRef, FORMAT_VERSION

IQ1_CODEBOOK_SHA256 = "1edfeb295366968940d5d4397dc046110f851acb59de9407fdf0c06982adaa72"
IQ1_REFERENCE_REVISION = "a5822222909b785f23ddc74ce3c8f85bd0e38562"
_POW3_4 = torch.tensor([1, 3, 9, 27], dtype=torch.int64)
_POW3_8 = torch.tensor([1, 3, 9, 27, 81, 243, 729, 2187], dtype=torch.int64)


def base3_ids(trits: torch.Tensor) -> torch.Tensor:
    if trits.shape[-1] not in {4, 8}:
        raise ValueError("base-3 encoding supports four or eight lanes")
    powers = _POW3_4 if trits.shape[-1] == 4 else _POW3_8
    return ((trits.to(torch.int64).cpu() + 1) * powers).sum(dim=-1)


def masks_to_trits(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim != 2 or masks.shape[1] != 2 or masks.dtype != torch.uint8:
        raise ValueError("shape masks must be uint8 [count,2]")
    bits = torch.arange(8, dtype=torch.int64)
    positive = ((masks[:, 0].to(torch.int64)[:, None] >> bits) & 1).to(torch.int8)
    negative = ((masks[:, 1].to(torch.int64)[:, None] >> bits) & 1).to(torch.int8)
    if torch.any(positive & negative):
        raise ValueError("positive and negative masks overlap")
    return positive - negative


def trits_to_masks(trits: torch.Tensor) -> torch.Tensor:
    if trits.ndim != 2 or trits.shape[1] != 8:
        raise ValueError("joint trits must have shape [count,8]")
    if not torch.all((trits >= -1) & (trits <= 1)):
        raise ValueError("codebooks may contain only -1, 0, +1")
    bits = (1 << torch.arange(8, dtype=torch.int64))[None]
    positive = ((trits.cpu() == 1).to(torch.int64) * bits).sum(1).to(torch.uint8)
    negative = ((trits.cpu() == -1).to(torch.int64) * bits).sum(1).to(torch.uint8)
    return torch.stack((positive, negative), dim=1)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes(order="C")


def canonical_codebook_bytes(index_format: str, encoding: str,
                             tables: Mapping[str, torch.Tensor]) -> bytes:
    expected = {
        "sign_canonical": ("shapes_masks",),
        "direct_joint": ("joint_trits",),
        "product": ("product_a", "product_b"),
    }
    if encoding not in expected:
        raise ValueError(f"unsupported codebook encoding {encoding!r}")
    if tuple(tables) != expected[encoding]:
        raise ValueError(f"{encoding} tables must be ordered as {expected[encoding]}")
    out = bytearray(b"TQ1_CODEBOOK\0")
    out += struct.pack("<I", FORMAT_VERSION)
    enc = encoding.encode("utf-8")
    fmt = index_format.encode("utf-8")
    out += struct.pack("<H", len(enc)) + enc
    out += struct.pack("<H", len(fmt)) + fmt
    out += struct.pack("<H", len(tables))
    for name, tensor in tables.items():
        if tensor.dtype == torch.uint8:
            dtype_code = 1
        elif tensor.dtype == torch.int8:
            dtype_code = 2
        else:
            raise ValueError(f"{name}: codebook table must be uint8 or int8")
        encoded_name = name.encode("utf-8")
        payload = _tensor_bytes(tensor)
        out += struct.pack("<H", len(encoded_name)) + encoded_name
        out += struct.pack("<BB", dtype_code, tensor.ndim)
        out += b"".join(struct.pack("<I", int(dim)) for dim in tensor.shape)
        out += struct.pack("<Q", len(payload)) + payload
    return bytes(out)


@dataclass(frozen=True)
class Codebook:
    id: str
    index_format: str
    encoding: str
    scope: str
    tables: Mapping[str, torch.Tensor]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        copied = {name: value.detach().contiguous().cpu().clone()
                  for name, value in self.tables.items()}
        object.__setattr__(self, "tables", copied)
        self.validate()

    @property
    def index_bits(self) -> int:
        return 11 if self.index_format == "v11" else 12

    @property
    def index_count(self) -> int:
        return 1 << self.index_bits

    def canonical_bytes(self) -> bytes:
        return canonical_codebook_bytes(self.index_format, self.encoding, self.tables)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def ref(self) -> CodebookRef:
        return CodebookRef(self.id, self.index_format, self.encoding,
                           self.scope, self.sha256())

    def validate(self) -> None:
        if self.index_format not in {"v11", "v12"}:
            raise ValueError("index format must be v11 or v12")
        if self.encoding == "sign_canonical":
            self._validate_joint()
        elif self.encoding == "direct_joint":
            self._validate_direct()
        elif self.encoding == "product":
            self._validate_product()
        else:
            raise ValueError(f"unsupported encoding {self.encoding!r}")

    def _validate_joint(self) -> None:
        if tuple(self.tables) != ("shapes_masks",):
            raise ValueError("joint codebook requires only shapes_masks")
        masks = self.tables["shapes_masks"]
        count = 1024 if self.index_format == "v11" else 2048
        if masks.dtype != torch.uint8 or tuple(masks.shape) != (count, 2):
            raise ValueError(f"shapes_masks must be uint8 [{count},2]")
        trits = masks_to_trits(masks)
        if torch.any(trits[0] != 0):
            raise ValueError("joint shape zero must be the all-zero vector")
        if torch.unique(masks, dim=0).shape[0] != count:
            raise ValueError("joint codebook contains duplicate shapes")
        nonzero = trits[1:]
        first = (nonzero != 0).to(torch.int64).argmax(dim=1)
        if torch.any(nonzero.gather(1, first[:, None]).squeeze(1) != 1):
            raise ValueError("joint shapes are not sign-canonical")
        ids = base3_ids(trits)
        if torch.any(ids[2:] <= ids[1:-1]):
            raise ValueError("joint shapes must be ordered by increasing base-3 id")

    def _validate_direct(self) -> None:
        if self.index_format != "v11" or tuple(self.tables) != ("joint_trits",):
            raise ValueError("direct_joint format-v1 codebooks are V11 joint_trits")
        trits = self.tables["joint_trits"]
        if trits.dtype != torch.int8 or tuple(trits.shape) != (2048, 8):
            raise ValueError("joint_trits must be int8 [2048,8]")
        if not torch.all((trits >= -1) & (trits <= 1)):
            raise ValueError("joint_trits contains a non-trit")
        if torch.unique(trits, dim=0).shape[0] != 2048:
            raise ValueError("direct joint codebook contains duplicate rows")
        zeros = torch.nonzero((trits == 0).all(dim=1)).flatten().tolist()
        if zeros != [1029]:
            raise ValueError(f"IQ1 zero row must be index 1029, got {zeros}")
        if self.scope == "iq1" and self.sha256() != IQ1_CODEBOOK_SHA256:
            raise ValueError("IQ1 codebook does not match the pinned canonical hash")

    def _validate_product(self) -> None:
        if tuple(self.tables) != ("product_a", "product_b"):
            raise ValueError("product codebook requires ordered A and B tables")
        a, b = self.tables["product_a"], self.tables["product_b"]
        b_count = 32 if self.index_format == "v11" else 64
        if a.dtype != torch.int8 or tuple(a.shape) != (32, 4):
            raise ValueError("product_a must be int8 [32,4]")
        if b.dtype != torch.int8 or tuple(b.shape) != (b_count, 4):
            raise ValueError(f"product_b must be int8 [{b_count},4]")
        for name, table in (("product_a", a), ("product_b", b)):
            if not torch.all((table >= -1) & (table <= 1)):
                raise ValueError(f"{name} contains a non-trit")
            if torch.any(table[0] != 0):
                raise ValueError(f"{name}[0] must be zero")
            if torch.unique(table, dim=0).shape[0] != table.shape[0]:
                raise ValueError(f"{name} contains duplicate rows")
            ids = base3_ids(table)
            if torch.any(ids[2:] <= ids[1:-1]):
                raise ValueError(f"{name} must be ordered by increasing base-3 id")
        self._validate_sign_pairs(a, expected_pairs=0, name="product_a")
        self._validate_sign_pairs(b, expected_pairs=0 if self.index_format == "v11" else 23,
                                  name="product_b")
        expected_unique = 2047 if self.index_format == "v11" else 4049
        if torch.unique(base3_ids(self.decode(torch.arange(self.index_count)))).numel() \
                != expected_unique:
            raise ValueError("product codebook has the wrong expanded unique count")

    @staticmethod
    def _validate_sign_pairs(table: torch.Tensor, expected_pairs: int, name: str) -> None:
        ids = set(base3_ids(table).tolist())
        neg = base3_ids(-table)
        pairs = sum(int(int(value) in ids) for value in neg[1:].tolist()) // 2
        if pairs != expected_pairs:
            raise ValueError(f"{name} has {pairs} nonzero sign pairs, expected {expected_pairs}")

    def legal_index_mask(self) -> torch.Tensor:
        if self.encoding == "direct_joint":
            return torch.ones(self.index_count, dtype=torch.bool)
        decoded = self.decode(torch.arange(self.index_count))
        ids = base3_ids(decoded)
        legal = torch.zeros(self.index_count, dtype=torch.bool)
        first: dict[int, int] = {}
        for index, pattern_id in enumerate(ids.tolist()):
            if pattern_id not in first:
                first[pattern_id] = index
                legal[index] = True
        return legal

    def duplicate_equivalence_classes(self) -> dict[int, list[int]]:
        ids = base3_ids(self.decode(torch.arange(self.index_count))).tolist()
        groups: dict[int, list[int]] = {}
        for index, pattern_id in enumerate(ids):
            groups.setdefault(pattern_id, []).append(index)
        return {members[0]: members for members in groups.values() if len(members) > 1}

    def validate_indices(self, indices: torch.Tensor) -> None:
        values = indices.to(torch.int64).cpu()
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= self.index_count):
            raise ValueError("codebook index is outside the physical range")
        legal = self.legal_index_mask()
        if values.numel() and not torch.all(legal[values]):
            bad = torch.unique(values[~legal[values]]).tolist()
            raise ValueError(f"payload contains reserved indices {bad[:8]}")

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        values = indices.to(torch.int64).cpu()
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= self.index_count):
            raise ValueError("codebook index is outside the physical range")
        if self.encoding == "direct_joint":
            return self.tables["joint_trits"][values]
        if self.encoding == "sign_canonical":
            count = 1024 if self.index_format == "v11" else 2048
            shapes = masks_to_trits(self.tables["shapes_masks"])
            sign = torch.where((values & count) != 0, -1, 1).to(torch.int8)
            return shapes[values & (count - 1)] * sign[..., None]
        a, b = self.tables["product_a"], self.tables["product_b"]
        a_id = values & 31
        if self.index_format == "v11":
            b_id = (values >> 5) & 31
            sign_bit = 1 << 10
        else:
            b_id = (values >> 5) & 63
            sign_bit = 1 << 11
        sign = torch.where((values & sign_bit) != 0, -1, 1).to(torch.int8)
        return torch.cat((a[a_id], b[b_id]), dim=-1) * sign[..., None]


class CodebookRegistry:
    def __init__(self, codebooks: Mapping[str, Codebook]):
        self._codebooks = dict(codebooks)
        if set(self._codebooks) != {book.id for book in self._codebooks.values()}:
            raise ValueError("registry mapping keys must equal codebook ids")
        hashes = [book.sha256() for book in self._codebooks.values()]
        if len(hashes) != len(set(hashes)):
            raise ValueError("registry aliases the same codebook under multiple ids")

    def __getitem__(self, codebook_id: str) -> Codebook:
        return self._codebooks[codebook_id]

    def refs(self) -> tuple[CodebookRef, ...]:
        return tuple(book.ref() for book in self._codebooks.values())

    def validate_refs(self, refs: tuple[CodebookRef, ...]) -> None:
        actual = self.refs()
        if actual != refs:
            raise ValueError("loaded codebook registry does not match QuantSpec")


def sign_canonical_codebook(codebook_id: str, index_format: str,
                            shapes: torch.Tensor, *, scope: str = "model",
                            provenance: Mapping[str, object] | None = None) -> Codebook:
    """Create a J codebook while enforcing the canonical serialized row order."""
    value = shapes.detach().to(torch.int8).cpu()
    count = 1024 if index_format == "v11" else 2048
    if tuple(value.shape) != (count, 8):
        raise ValueError(f"{index_format} needs {count} sign-canonical shapes")
    zero = torch.nonzero((value == 0).all(dim=1)).flatten()
    if zero.numel() != 1:
        raise ValueError("J codebook input must contain zero exactly once")
    nonzero = value[~(value == 0).all(dim=1)]
    order = torch.argsort(base3_ids(nonzero), stable=True)
    ordered = torch.cat((torch.zeros((1, 8), dtype=torch.int8), nonzero[order]), dim=0)
    return Codebook(codebook_id, index_format, "sign_canonical", scope,
                    {"shapes_masks": trits_to_masks(ordered)}, provenance or {})


def direct_joint_codebook(codebook_id: str, trits: torch.Tensor, *,
                          scope: str = "iq1",
                          provenance: Mapping[str, object] | None = None) -> Codebook:
    return Codebook(codebook_id, "v11", "direct_joint", scope,
                    {"joint_trits": trits.detach().to(torch.int8).cpu()}, provenance or {})


def load_iq1_reference(codebook_id: str = "iq1s_grid", *,
                       reference_dir: str | Path = "~/llama.cpp") -> Codebook:
    """Transcribe the pinned read-only IQ1 grid into a self-contained codebook.

    The resulting canonical artifact embeds the table; runtimes never consult
    this reference checkout.  A revision mismatch is fatal because an identical
    symbol name is not sufficient provenance.
    """
    root = Path(reference_dir).expanduser().resolve()
    source = root / "ggml" / "src" / "ggml-common.h"
    if not source.is_file():
        raise FileNotFoundError(source)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
        text=True, stdout=subprocess.PIPE).stdout.strip()
    if revision != IQ1_REFERENCE_REVISION:
        raise ValueError(
            f"IQ1 reference revision {revision} != pinned {IQ1_REFERENCE_REVISION}")
    match = re.search(
        r"GGML_TABLE_BEGIN\(uint64_t,\s*iq1s_grid,\s*NGRID_IQ1S\)"
        r"(.*?)GGML_TABLE_END\(\)", source.read_text(), re.DOTALL)
    if match is None:
        raise ValueError(f"could not locate iq1s_grid in {source}")
    packed = [int(token, 16) for token in re.findall(r"0x[0-9a-fA-F]+", match.group(1))]
    if len(packed) != 2048:
        raise ValueError(f"expected 2048 IQ1 grid entries, found {len(packed)}")
    grid = torch.empty((2048, 8), dtype=torch.int8)
    for row, value in enumerate(packed):
        for lane in range(8):
            byte = (value >> (8 * lane)) & 0xff
            grid[row, lane] = byte - 256 if byte >= 128 else byte
    return direct_joint_codebook(
        codebook_id, grid, scope="iq1", provenance={
            "source": "llama.cpp/ggml/src/ggml-common.h:iq1s_grid",
            "revision": revision,
            "canonical_sha256": IQ1_CODEBOOK_SHA256,
        })


def product_codebook(codebook_id: str, index_format: str, product_a: torch.Tensor,
                     product_b: torch.Tensor, *, scope: str = "model",
                     provenance: Mapping[str, object] | None = None) -> Codebook:
    return Codebook(codebook_id, index_format, "product", scope, {
        "product_a": product_a.detach().to(torch.int8).cpu(),
        "product_b": product_b.detach().to(torch.int8).cpu(),
    }, provenance or {})
