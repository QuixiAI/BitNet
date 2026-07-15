"""Small fail-closed GGUF v3 reader/writer used for exact TQ1 rewriting.

It deliberately supports only primitive tensor types plus TQ1 registry revision
1.  The ordinary HF converter supplies the base GGUF, so this module never
needs to understand or regenerate architecture/tokenizer metadata.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, \
    UINT64, INT64, FLOAT64 = range(13)


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dimensions: tuple[int, ...]       # GGUF ne[] order (K,N,...)
    tensor_type: int
    data: bytes


@dataclass(frozen=True)
class ParsedGGUF:
    version: int
    metadata: Mapping[str, Any]
    metadata_types: Mapping[str, int]
    raw_metadata: bytes
    tensors: tuple[TensorRecord, ...]
    alignment: int


class _Cursor:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def take(self, count: int) -> bytes:
        end = self.offset + count
        if end > len(self.data):
            raise ValueError("truncated GGUF")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def unpack(self, spelling: str):
        size = struct.calcsize("<" + spelling)
        return struct.unpack("<" + spelling, self.take(size))[0]

    def string(self) -> str:
        size = self.unpack("Q")
        return self.take(size).decode("utf-8")


_SCALARS = {
    UINT8: "B", INT8: "b", UINT16: "H", INT16: "h", UINT32: "I", INT32: "i",
    FLOAT32: "f", BOOL: "?", UINT64: "Q", INT64: "q", FLOAT64: "d",
}


def _read_value(cursor: _Cursor, value_type: int):
    if value_type in _SCALARS:
        return cursor.unpack(_SCALARS[value_type])
    if value_type == STRING:
        return cursor.string()
    if value_type == ARRAY:
        element_type = cursor.unpack("I")
        count = cursor.unpack("Q")
        return [_read_value(cursor, element_type) for _ in range(count)]
    raise ValueError(f"unsupported GGUF metadata value type {value_type}")


def _tensor_nbytes(tensor_type: int, dimensions: tuple[int, ...]) -> int:
    elements = math.prod(dimensions)
    primitive = {0: 4, 1: 2, 24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 30: 2}
    if tensor_type in primitive:
        return elements * primitive[tensor_type]
    tq1 = {
        43: (256, 46),  # TQ1_V11 generic block scale
        44: (256, 50),  # TQ1_V12 generic block scale
        45: (256, 44),  # TQ1_V11_R
        46: (256, 48),  # TQ1_V12_R
        47: (256, 48),  # TQ1_V11_J_A4_R
    }
    if tensor_type in tq1:
        block, size = tq1[tensor_type]
        if not dimensions or dimensions[0] % block or elements % block:
            raise ValueError("TQ1 GGUF tensor has an invalid logical block shape")
        return elements // block * size
    raise ValueError(f"unsupported GGUF tensor type {tensor_type}")


def parse_gguf(path: str | Path) -> ParsedGGUF:
    raw = Path(path).read_bytes()
    cursor = _Cursor(raw)
    if cursor.unpack("I") != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = cursor.unpack("I")
    if version != GGUF_VERSION:
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count, metadata_count = cursor.unpack("Q"), cursor.unpack("Q")
    metadata_begin = cursor.offset
    metadata: dict[str, Any] = {}
    metadata_types: dict[str, int] = {}
    for _ in range(metadata_count):
        key = cursor.string()
        if key in metadata:
            raise ValueError(f"duplicate GGUF metadata key {key!r}")
        value_type = cursor.unpack("I")
        metadata[key] = _read_value(cursor, value_type)
        metadata_types[key] = value_type
    metadata_end = cursor.offset
    infos = []
    names = set()
    for _ in range(tensor_count):
        name = cursor.string()
        if name in names:
            raise ValueError(f"duplicate GGUF tensor {name!r}")
        names.add(name)
        rank = cursor.unpack("I")
        dimensions = tuple(cursor.unpack("Q") for _ in range(rank))
        tensor_type = cursor.unpack("I")
        relative_offset = cursor.unpack("Q")
        infos.append((name, dimensions, tensor_type, relative_offset))
    alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("GGUF alignment must be a positive power of two")
    data_begin = align(cursor.offset, alignment)
    tensors = []
    previous_end = 0
    for name, dimensions, tensor_type, relative_offset in infos:
        if relative_offset % alignment or relative_offset < previous_end:
            raise ValueError(f"invalid GGUF tensor offset for {name}")
        size = _tensor_nbytes(tensor_type, dimensions)
        begin, end = data_begin + relative_offset, data_begin + relative_offset + size
        if end > len(raw):
            raise ValueError(f"truncated GGUF tensor {name}")
        tensors.append(TensorRecord(name, dimensions, tensor_type, raw[begin:end]))
        previous_end = relative_offset + size
    return ParsedGGUF(
        version, metadata, metadata_types, raw[metadata_begin:metadata_end],
        tuple(tensors), alignment)


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _infer_array_type(values: list[Any]) -> int:
    if not values:
        raise ValueError("GGUF arrays may not be empty")
    if all(isinstance(value, str) for value in values):
        return STRING
    if all(isinstance(value, bool) for value in values):
        return BOOL
    if all(isinstance(value, int) and value >= 0 for value in values):
        return UINT64
    if all(isinstance(value, int) for value in values):
        return INT64
    if all(isinstance(value, (int, float)) for value in values):
        return FLOAT64
    raise TypeError("unsupported heterogeneous GGUF array")


def encode_metadata(key: str, value: Any, value_type: int | None = None) -> bytes:
    if value_type is None:
        if isinstance(value, bool):
            value_type = BOOL
        elif isinstance(value, str):
            value_type = STRING
        elif isinstance(value, int):
            value_type = UINT64 if value >= 0 else INT64
        elif isinstance(value, float):
            value_type = FLOAT64
        elif isinstance(value, (list, tuple)):
            value_type = ARRAY
        else:
            raise TypeError(f"unsupported GGUF metadata type {type(value).__name__}")
    out = bytearray(_pack_string(key) + struct.pack("<I", value_type))
    if value_type in _SCALARS:
        out += struct.pack("<" + _SCALARS[value_type], value)
    elif value_type == STRING:
        out += _pack_string(value)
    elif value_type == ARRAY:
        values = list(value)
        element_type = _infer_array_type(values)
        out += struct.pack("<IQ", element_type, len(values))
        for item in values:
            if element_type == STRING:
                out += _pack_string(item)
            else:
                out += struct.pack("<" + _SCALARS[element_type], item)
    else:
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")
    return bytes(out)


def write_rewritten_gguf(base: ParsedGGUF, output: str | Path,
                         tensors: Iterable[TensorRecord],
                         metadata: Mapping[str, tuple[Any, int | None]]) -> None:
    records = tuple(tensors)
    names = [item.name for item in records]
    if len(names) != len(set(names)):
        raise ValueError("rewritten GGUF tensor names are not unique")
    duplicate_metadata = set(metadata) & set(base.metadata)
    if duplicate_metadata:
        raise ValueError(f"new GGUF metadata duplicates base keys {sorted(duplicate_metadata)}")
    extra_metadata = b"".join(
        encode_metadata(key, value, value_type)
        for key, (value, value_type) in metadata.items())
    offsets = []
    position = 0
    for record in records:
        expected = _tensor_nbytes(record.tensor_type, record.dimensions)
        if len(record.data) != expected:
            raise ValueError(
                f"{record.name}: {len(record.data)} data bytes != expected {expected}")
        position = align(position, base.alignment)
        offsets.append(position)
        position += len(record.data)
    header = struct.pack(
        "<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(records),
        len(base.metadata) + len(metadata))
    infos = bytearray()
    for record, offset in zip(records, offsets):
        infos += _pack_string(record.name)
        infos += struct.pack("<I", len(record.dimensions))
        infos += b"".join(struct.pack("<Q", value) for value in record.dimensions)
        infos += struct.pack("<IQ", record.tensor_type, offset)
    prefix = header + base.raw_metadata + extra_metadata + bytes(infos)
    payload = bytearray(prefix)
    payload += bytes(align(len(payload), base.alignment) - len(payload))
    data_begin = len(payload)
    for record, offset in zip(records, offsets):
        want = data_begin + offset
        payload += bytes(want - len(payload))
        payload += record.data
    Path(output).write_bytes(payload)
