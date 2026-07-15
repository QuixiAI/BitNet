#!/usr/bin/env python3
"""Canonical schema-2 TQ1_V producer for Llama-3.2-1B-Instruct.

The executable entry point delegates to :mod:`bitnet_train.tq1.cli`, which
implements the normalized QuantSpec, calibration, learned/loaded codebooks,
all format-v1 PTQ profiles, and transactional canonical artifacts required by
``quant_spec.md``.  The older schema-1 classes remain below solely as a stable
research-oracle API for historical tests; the CLI cannot emit schema 1.

Example::

    .venv/bin/python quant/quant.py \
      --output runs/Llama-3.2-1B-Instruct-TQ1-V12 \
      --calibration-file data/calibration.jsonl \
      --device mps

Use ``--importance-mode uniform`` for an explicit no-calibration experiment;
the production default requires a calibration file or statistics artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SCHEMA_VERSION = "1"
DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
GROUP_SIZE = 8
PATTERN_COUNT = 3**GROUP_SIZE
PAYLOAD_GROUPS = 32                       # 32 x 8 = 256 weights

DEFAULT_TARGET_REGEXES = (
    r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
    r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj",
)
DEFAULT_KEEP_FP_REGEXES = (r"lm_head",)


# ---------------------------------------------------------------------------
# Format and codebook
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TQ1Format:
    name: str
    index_bits: int
    shape_bits: int

    @property
    def n_shapes(self) -> int:
        return 1 << self.shape_bits

    @property
    def n_indices(self) -> int:
        return 1 << self.index_bits

    @property
    def high_bits(self) -> int:
        return self.index_bits - 8

    @property
    def payload_bytes(self) -> int:
        return PAYLOAD_GROUPS + PAYLOAD_GROUPS * self.high_bits // 8

    @property
    def raw_bpw(self) -> float:
        return self.index_bits / GROUP_SIZE

    def validate(self) -> None:
        if self.index_bits != self.shape_bits + 1:
            raise ValueError("one index bit must be the global sign")
        if self.index_bits not in (11, 12):
            raise ValueError("only TQ1_V11 and TQ1_V12 are public formats")
        if PAYLOAD_GROUPS * self.high_bits % 8:
            raise ValueError("high index bits do not fill complete bytes")


FORMATS = {
    "v11": TQ1Format("TQ1_V11", index_bits=11, shape_bits=10),
    "v12": TQ1Format("TQ1_V12", index_bits=12, shape_bits=11),
}


def _format(value: str | TQ1Format) -> TQ1Format:
    if isinstance(value, TQ1Format):
        value.validate()
        return value
    key = value.lower().replace("tq1_", "").replace("tq1-", "")
    try:
        spec = FORMATS[key]
    except KeyError as exc:
        raise ValueError(f"format must be one of {sorted(FORMATS)}, got {value!r}") from exc
    spec.validate()
    return spec


@lru_cache(maxsize=1)
def ternary_universe() -> torch.Tensor:
    """All 3^8 vectors in base-3 integer order, as int8 [6561, 8]."""
    value = torch.arange(PATTERN_COUNT, dtype=torch.int64)
    lanes = []
    for _ in range(GROUP_SIZE):
        lanes.append((value % 3 - 1).to(torch.int8))
        value = torch.div(value, 3, rounding_mode="floor")
    return torch.stack(lanes, dim=1)


def encode_ternary(vectors: torch.Tensor) -> torch.Tensor:
    """Encode (..., 8) values in {-1,0,+1} as base-3 IDs in [0, 6560]."""
    if vectors.shape[-1] != GROUP_SIZE:
        raise ValueError(f"last dimension must be {GROUP_SIZE}, got {vectors.shape}")
    powers = torch.tensor([3**i for i in range(GROUP_SIZE)],
                          dtype=torch.int64, device=vectors.device)
    return ((vectors.to(torch.int64) + 1) * powers).sum(dim=-1)


def canonicalize_sign(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orient each nonzero vector so its first nonzero trit is positive.

    Returns ``(canonical, negative)``.  ``negative`` is the global sign bit of
    the original vector.  Zero is canonical and never receives the sign bit.
    """
    if vectors.shape[-1] != GROUP_SIZE:
        raise ValueError(f"last dimension must be {GROUP_SIZE}, got {vectors.shape}")
    flat = vectors.reshape(-1, GROUP_SIZE)
    nz = flat.ne(0)
    has_nz = nz.any(dim=1)
    first_lane = nz.to(torch.int64).argmax(dim=1)
    first = flat.gather(1, first_lane[:, None]).squeeze(1)
    negative = has_nz & first.lt(0)
    orient = torch.where(negative, -torch.ones_like(first), torch.ones_like(first))
    canonical = flat * orient[:, None]
    return canonical.reshape_as(vectors), negative.reshape(vectors.shape[:-1])


@lru_cache(maxsize=1)
def canonical_universe() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the 3,281 sign-canonical shapes and full-pattern -> shape map."""
    full = ternary_universe()
    canonical, _ = canonicalize_sign(full)
    canonical_ids = encode_ternary(canonical)
    unique_ids = torch.unique(canonical_ids, sorted=True)
    shapes = ternary_universe()[unique_ids]
    row_for_id = torch.full((PATTERN_COUNT,), -1, dtype=torch.int64)
    row_for_id[unique_ids] = torch.arange(unique_ids.numel(), dtype=torch.int64)
    full_to_shape = row_for_id[canonical_ids]
    if shapes.shape != (3281, GROUP_SIZE) or (full_to_shape < 0).any():
        raise AssertionError("sign-canonical ternary universe construction failed")
    return shapes, full_to_shape


@dataclass
class TQ1Codebook:
    """A sign-canonical or direct-joint ternary codebook.

    ``sign_canonical`` stores 2^shape_bits shapes and interprets the high index
    bit as a global sign.  ``joint`` stores all 2^index_bits codewords directly;
    this is used for the exact llama.cpp IQ1-grid baseline.
    """

    spec: TQ1Format
    shapes: torch.Tensor                    # canonical shapes or direct codewords
    construction: dict[str, object] = field(default_factory=dict)
    encoding: str = "sign_canonical"

    def __post_init__(self) -> None:
        self.spec.validate()
        self.shapes = self.shapes.detach().to(device="cpu", dtype=torch.int8).contiguous()
        if self.encoding not in ("sign_canonical", "joint"):
            raise ValueError("codebook encoding must be sign_canonical or joint")
        expected_rows = (self.spec.n_shapes if self.encoding == "sign_canonical"
                         else self.spec.n_indices)
        if self.shapes.shape != (expected_rows, GROUP_SIZE):
            raise ValueError(
                f"{self.spec.name}/{self.encoding} needs "
                f"{(expected_rows, GROUP_SIZE)} rows, "
                f"got {tuple(self.shapes.shape)}")
        if not torch.isin(self.shapes, torch.tensor([-1, 0, 1], dtype=torch.int8)).all():
            raise ValueError("codebook contains a non-ternary value")
        if self.encoding == "sign_canonical":
            canonical, _ = canonicalize_sign(self.shapes)
            if not torch.equal(canonical, self.shapes):
                raise ValueError("codebook shapes are not sign-canonical")
            if self.shapes[0].count_nonzero():
                raise ValueError("shape 0 must be the all-zero reserved shape")
            if torch.unique(encode_ternary(self.shapes)).numel() != self.spec.n_shapes:
                raise ValueError("codebook contains duplicate canonical shapes")
        elif torch.unique(encode_ternary(self.shapes)).numel() != self.spec.n_indices:
            raise ValueError("joint codebook contains duplicate codewords")

    @property
    def negative_zero_index(self) -> int | None:
        return self.spec.n_shapes if self.encoding == "sign_canonical" else None

    def decode(self, indices: torch.Tensor, *, dtype: torch.dtype | None = None,
               device: torch.device | str | None = None) -> torch.Tensor:
        """Decode arbitrary-shaped indices to ``indices.shape + (8,)`` trits."""
        target = torch.device(device) if device is not None else indices.device
        idx = indices.to(device=target, dtype=torch.int64)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.spec.n_indices):
            raise ValueError(f"index outside {self.spec.name} range")
        shapes = self.shapes.to(target)
        if self.encoding == "joint":
            out = shapes[idx]
        else:
            shape_id = torch.bitwise_and(idx, self.spec.n_shapes - 1)
            sign = torch.where(torch.bitwise_and(idx, self.spec.n_shapes).ne(0), -1, 1)
            out = shapes[shape_id] * sign[..., None]
        return out.to(dtype=dtype) if dtype is not None else out

    def expanded(self, *, dtype: torch.dtype = torch.float32,
                 device: torch.device | str = "cpu") -> torch.Tensor:
        ids = torch.arange(self.spec.n_indices, dtype=torch.int64, device=device)
        return self.decode(ids, dtype=dtype, device=device)

    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(SCHEMA_VERSION.encode())
        h.update(self.spec.name.encode())
        h.update(self.encoding.encode())
        h.update(self.shapes.numpy().tobytes())
        return h.hexdigest()[:16]


def _squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Integer squared Euclidean distance without a (A,B,8) temporary."""
    af, bf = a.float(), b.float()
    return (af.square().sum(1, keepdim=True)
            + bf.square().sum(1).unsqueeze(0)
            - 2.0 * af @ bf.t()).round().clamp_min_(0).to(torch.int16)


def learn_sign_canonical_codebook(
    spec: str | TQ1Format,
    pattern_counts: torch.Tensor | None = None,
    *,
    frequency_fraction: float = 0.75,
) -> TQ1Codebook:
    """Build the deterministic sign-canonical codebook proposed in quant.md.

    Required anchors are zero, all support-1/support-2 shapes, and all dense
    shapes.  Empirically frequent model patterns fill ``frequency_fraction`` of
    the table.  Weighted farthest-first coverage fills the rest.  This is a
    practical deterministic approximation to the document's discrete facility
    location objective; the artifact stores the resulting model-owned table.
    """
    spec = _format(spec)
    if not 0.0 <= frequency_fraction <= 1.0:
        raise ValueError("frequency_fraction must be in [0, 1]")
    shapes, full_to_shape = canonical_universe()
    n_universe = shapes.shape[0]

    if pattern_counts is None:
        full_counts = torch.zeros(PATTERN_COUNT, dtype=torch.float64)
    else:
        full_counts = torch.as_tensor(pattern_counts, dtype=torch.float64).flatten()
        if full_counts.numel() != PATTERN_COUNT:
            raise ValueError(f"pattern_counts must have {PATTERN_COUNT} entries")
        if not torch.isfinite(full_counts).all() or (full_counts < 0).any():
            raise ValueError("pattern_counts must be finite and nonnegative")
    canonical_counts = torch.zeros(n_universe, dtype=torch.float64)
    canonical_counts.scatter_add_(0, full_to_shape, full_counts)

    nnz = shapes.ne(0).sum(dim=1)
    zero_row = int((nnz == 0).nonzero(as_tuple=False)[0])
    anchor_rows = ((nnz <= 2) | (nnz == GROUP_SIZE)).nonzero(as_tuple=False).flatten()
    anchor_rows = sorted((int(i) for i in anchor_rows),
                         key=lambda i: int(encode_ternary(shapes[i])))
    selected = [zero_row] + [i for i in anchor_rows if i != zero_row]
    selected_set = set(selected)

    freq_target = max(len(selected), int(round(spec.n_shapes * frequency_fraction)))
    if float(canonical_counts.sum()) > 0:
        frequency_order = sorted(
            range(n_universe),
            key=lambda i: (-float(canonical_counts[i]),
                           int(encode_ternary(shapes[i]))),
        )
        for row in frequency_order:
            if len(selected) >= min(freq_target, spec.n_shapes):
                break
            if row not in selected_set:
                selected.append(row)
                selected_set.add(row)

    # Initialize coverage against the (potentially large) selected prefix.
    selected_tensor = shapes[torch.tensor(selected, dtype=torch.int64)]
    min_distance = torch.full((n_universe,), 32767, dtype=torch.int16)
    for start in range(0, selected_tensor.shape[0], 256):
        d = _squared_distances(shapes, selected_tensor[start:start + 256])
        min_distance = torch.minimum(min_distance, d.min(dim=1).values)

    log_frequency = torch.log1p(canonical_counts)
    if float(log_frequency.max()) > 0:
        log_frequency /= log_frequency.max()
    while len(selected) < spec.n_shapes:
        # Coverage is primary; frequency only breaks/weights equal shells.
        score = min_distance.float() * (1.0 + 0.25 * log_frequency.float())
        if selected:
            score[torch.tensor(selected, dtype=torch.int64)] = -1.0
        row = int(score.argmax())
        selected.append(row)
        selected_set.add(row)
        d = ((shapes.to(torch.int16) - shapes[row].to(torch.int16))
             .square().sum(dim=1).to(torch.int16))
        min_distance = torch.minimum(min_distance, d)

    chosen = shapes[torch.tensor(selected, dtype=torch.int64)]
    book = TQ1Codebook(
        spec,
        chosen,
        construction={
            "algorithm": "anchors+frequency+weighted-farthest-first",
            "frequency_fraction": frequency_fraction,
            "anchor_count": len(anchor_rows),
            "observed_patterns": int((full_counts > 0).sum()),
            "observations": float(full_counts.sum()),
        },
    )
    book.construction["coverage"] = codebook_coverage(book)
    return book


def codebook_coverage(codebook: TQ1Codebook) -> dict[str, object]:
    """Exact coverage histogram over all 6,561 ternary vectors."""
    full = ternary_universe()
    codewords = codebook.expanded(dtype=torch.int8)
    # The sign-set zero is reserved and duplicates index zero; drop it.
    if codebook.negative_zero_index is not None:
        keep = torch.ones(codebook.spec.n_indices, dtype=torch.bool)
        keep[codebook.negative_zero_index] = False
        codewords = codewords[keep]
    minima = torch.full((PATTERN_COUNT,), 32767, dtype=torch.int16)
    for start in range(0, codewords.shape[0], 256):
        d = _squared_distances(full, codewords[start:start + 256])
        minima = torch.minimum(minima, d.min(dim=1).values)
    values, counts = torch.unique(minima, return_counts=True)
    return {
        "max_squared_trit_distance": int(minima.max()),
        "histogram": {str(int(v)): int(n) for v, n in zip(values, counts)},
    }


# ---------------------------------------------------------------------------
# Physical index packing (quant.md sections 3.1 and 3.2)
# ---------------------------------------------------------------------------


def _pack_high_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack [rows, blocks, 32] low-order bit fields into a dense byte tail."""
    rows, blocks, groups = values.shape
    if groups != PAYLOAD_GROUPS or PAYLOAD_GROUPS * bits % 8:
        raise ValueError("invalid high-bit field shape")
    nbytes = PAYLOAD_GROUPS * bits // 8
    out = torch.zeros((rows, blocks, nbytes), dtype=torch.int64)
    v = values.to(torch.int64)
    mask = (1 << bits) - 1
    if v.numel() and (int(v.min()) < 0 or int(v.max()) > mask):
        raise ValueError("high-bit field overflow")
    for group in range(PAYLOAD_GROUPS):
        bit = group * bits
        byte, shift = divmod(bit, 8)
        out[:, :, byte] |= (v[:, :, group] << shift) & 0xFF
        if shift + bits > 8:
            out[:, :, byte + 1] |= v[:, :, group] >> (8 - shift)
    return out.to(torch.uint8)


def _unpack_high_bits(data: torch.Tensor, bits: int) -> torch.Tensor:
    rows, blocks, nbytes = data.shape
    expected = PAYLOAD_GROUPS * bits // 8
    if nbytes != expected:
        raise ValueError(f"expected {expected} high-bit bytes, got {nbytes}")
    src = data.to(torch.int64)
    out = torch.empty((rows, blocks, PAYLOAD_GROUPS), dtype=torch.int64)
    mask = (1 << bits) - 1
    for group in range(PAYLOAD_GROUPS):
        bit = group * bits
        byte, shift = divmod(bit, 8)
        value = src[:, :, byte] >> shift
        if shift + bits > 8:
            value |= src[:, :, byte + 1] << (8 - shift)
        out[:, :, group] = value & mask
    return out


def pack_indices(indices: torch.Tensor, spec: str | TQ1Format) -> torch.Tensor:
    """Pack uint indices as [rows, blocks, 44|48] design-layout payloads.

    Each 256-weight block stores 32 low index bytes followed by the densely
    packed three (V11) or four (V12) high bits.  Row scales are external.
    """
    spec = _format(spec)
    idx = torch.as_tensor(indices, device="cpu", dtype=torch.int64)
    if idx.ndim != 2 or idx.shape[1] % PAYLOAD_GROUPS:
        raise ValueError("indices must be [rows, groups] with groups divisible by 32")
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= spec.n_indices):
        raise ValueError(f"index outside {spec.name} range")
    rows, groups = idx.shape
    blocks = groups // PAYLOAD_GROUPS
    block_idx = idx.reshape(rows, blocks, PAYLOAD_GROUPS)
    low = torch.bitwise_and(block_idx, 0xFF).to(torch.uint8)
    high = _pack_high_bits(torch.bitwise_right_shift(block_idx, 8), spec.high_bits)
    packed = torch.cat([low, high], dim=-1).contiguous()
    if packed.shape != (rows, blocks, spec.payload_bytes):
        raise AssertionError("packed payload has the wrong physical size")
    return packed


def unpack_indices(packed: torch.Tensor, spec: str | TQ1Format) -> torch.Tensor:
    """Inverse of :func:`pack_indices`, returning int64 [rows, groups]."""
    spec = _format(spec)
    src = torch.as_tensor(packed, device="cpu", dtype=torch.uint8)
    if src.ndim != 3 or src.shape[2] != spec.payload_bytes:
        raise ValueError(
            f"packed payload must be [rows, blocks, {spec.payload_bytes}]")
    low = src[:, :, :PAYLOAD_GROUPS].to(torch.int64)
    high = _unpack_high_bits(src[:, :, PAYLOAD_GROUPS:], spec.high_bits)
    return (low | (high << 8)).reshape(src.shape[0], -1)


# ---------------------------------------------------------------------------
# Importance-aware alternating projection
# ---------------------------------------------------------------------------


def build_candidate_table(codebook: TQ1Codebook, candidate_count: int = 32,
                          *, chunk: int = 256) -> torch.Tensor:
    """Nearest codeword candidates for every possible scalar ternary pattern."""
    valid_ids = torch.arange(codebook.spec.n_indices, dtype=torch.int64)
    if codebook.negative_zero_index is not None:
        valid_ids = valid_ids[valid_ids != codebook.negative_zero_index]
    count = min(int(candidate_count), valid_ids.numel())
    if count <= 0:
        raise ValueError("candidate_count must be positive")
    codewords = codebook.decode(valid_ids, dtype=torch.float32)
    c2 = codewords.square().sum(dim=1)
    full = ternary_universe().float()
    table = torch.empty((PATTERN_COUNT, count), dtype=torch.int16)
    for start in range(0, PATTERN_COUNT, chunk):
        patterns = full[start:start + chunk]
        distances = (patterns.square().sum(dim=1, keepdim=True) + c2[None, :]
                     - 2.0 * patterns @ codewords.t()).clamp_min_(0)
        nearest = distances.topk(count, dim=1, largest=False, sorted=True).indices
        table[start:start + patterns.shape[0]] = valid_ids[nearest].to(torch.int16)
    return table


@dataclass
class ProjectionMetrics:
    rows: int
    columns: int
    groups: int
    rmse: float
    max_abs_error: float
    max_rel_error: float
    relative_l2_error: float
    weighted_relative_l2_error: float
    scalar_pattern_exact_rate: float
    changed_trit_fraction: float
    scale_min: float
    scale_max: float
    payload_bytes: int
    effective_bpw_with_row_scale: float


@dataclass
class ProjectionResult:
    indices: torch.Tensor                  # uint16 [rows, K/8], CPU
    packed: torch.Tensor                   # uint8 [rows, K/256, 44|48], CPU
    scales: torch.Tensor                   # fp16/bf16 [rows], CPU
    dequantized: torch.Tensor              # fp32 [rows, K], CPU
    metrics: ProjectionMetrics


class TQ1Projector:
    """Reusable projection oracle for one fixed TQ1_V codebook."""

    def __init__(self, codebook: TQ1Codebook, *, candidate_count: int = 32,
                 device: torch.device | str = "cpu", chunk_groups: int = 16384):
        self.codebook = codebook
        self.device = torch.device(device)
        self.chunk_groups = int(chunk_groups)
        if self.chunk_groups <= 0:
            raise ValueError("chunk_groups must be positive")
        self.candidate_table_cpu = build_candidate_table(codebook, candidate_count)

    @torch.no_grad()
    def project(
        self,
        weight: torch.Tensor,
        *,
        activation_importance: torch.Tensor | None = None,
        metric: str = "iq1",
        iterations: int = 3,
        scale_dtype: torch.dtype = torch.float16,
    ) -> ProjectionResult:
        """Project a 2-D latent matrix onto the exact packed representation.

        ``metric='iq1'`` uses ``sqrt(sigma^2 + w^2)`` and multiplies it by
        activation second moments when supplied.  ``activation`` uses only the
        second moments; ``uniform`` is ordinary squared error.
        """
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2-D, got {tuple(weight.shape)}")
        rows, columns = weight.shape
        if columns % (GROUP_SIZE * PAYLOAD_GROUPS):
            raise ValueError(
                f"input width {columns} must be divisible by 256 for TQ1_V blocks")
        if metric not in ("uniform", "activation", "iq1"):
            raise ValueError("metric must be uniform, activation, or iq1")
        if iterations < 1:
            raise ValueError("iterations must be at least one")
        if scale_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("row scales must be float16 or bfloat16")

        device = self.device
        w = weight.detach().to(device=device, dtype=torch.float32).contiguous()
        groups_per_row = columns // GROUP_SIZE
        wg = w.reshape(rows, groups_per_row, GROUP_SIZE)

        if metric == "iq1":
            # Match llama.cpp's IQ1 reference: sigma^2 is shared by one
            # QK_K=256 block, not by the whole row.  The subsequent row-scale
            # refit is the TQ1_V-specific part.
            blocks = w.reshape(rows, columns // (GROUP_SIZE * PAYLOAD_GROUPS),
                               GROUP_SIZE * PAYLOAD_GROUPS)
            sigma2 = 2.0 * blocks.square().mean(dim=2, keepdim=True)
            h = torch.sqrt(sigma2 + blocks.square()).reshape_as(wg)
        else:
            h = torch.ones_like(wg)
        if activation_importance is not None and metric != "uniform":
            imp = torch.as_tensor(activation_importance, dtype=torch.float32,
                                  device=device).flatten()
            if imp.numel() != columns:
                raise ValueError(
                    f"activation importance has {imp.numel()} lanes, expected {columns}")
            if not torch.isfinite(imp).all() or (imp < 0).any():
                raise ValueError("activation importance must be finite and nonnegative")
            # Only relative sensitivity matters; keep its mean at one.
            imp = imp / imp.mean().clamp_min(1e-12)
            h.mul_(imp.reshape(1, groups_per_row, GROUP_SIZE))
        h.clamp_min_(1e-12)

        # Weighted absmean is the BitNet row-scale initializer.
        h2 = h.reshape(rows, columns)
        alpha = ((h2 * w.abs()).sum(dim=1) / h2.sum(dim=1).clamp_min(1e-20))
        alpha.clamp_min_(1e-12)

        candidates = self.candidate_table_cpu.to(device=device, dtype=torch.int64)
        codewords = self.codebook.expanded(dtype=torch.float32, device=device)
        powers = torch.tensor([3**i for i in range(GROUP_SIZE)], dtype=torch.int64,
                              device=device)

        def assign(scales: torch.Tensor) -> torch.Tensor:
            flat_w = wg.reshape(-1, GROUP_SIZE)
            flat_h = h.reshape(-1, GROUP_SIZE)
            result = torch.empty(flat_w.shape[0], dtype=torch.int32, device=device)
            for start in range(0, flat_w.shape[0], self.chunk_groups):
                end = min(start + self.chunk_groups, flat_w.shape[0])
                row = torch.div(torch.arange(start, end, device=device), groups_per_row,
                                rounding_mode="floor")
                scale = scales[row]
                source = (flat_w[start:end] / scale[:, None]).round().clamp_(-1, 1)
                pattern_id = ((source.to(torch.int64) + 1) * powers).sum(dim=1)
                candidate_ids = candidates[pattern_id]
                c = codewords[candidate_ids]
                residual = flat_w[start:end, None, :] - scale[:, None, None] * c
                error = (residual.square() * flat_h[start:end, None, :]).sum(dim=-1)
                best = error.argmin(dim=1)
                result[start:end] = candidate_ids[
                    torch.arange(end - start, device=device), best].to(torch.int32)
            return result.reshape(rows, groups_per_row)

        indices = None
        for _ in range(iterations):
            indices = assign(alpha)
            selected = codewords[indices.to(torch.int64)]
            numerator = (h * wg * selected).sum(dim=(1, 2))
            denominator = (h * selected.square()).sum(dim=(1, 2))
            refit = numerator / denominator.clamp_min(1e-20)
            alpha = torch.where(denominator > 0, refit, alpha).clamp_min_(1e-12)

        # Runtime scales are rounded before the final reassignment, exactly as
        # recommended in quant.md section 6.7.
        rounded_scale = alpha.to(scale_dtype).to(torch.float32)
        min_normal = torch.tensor(torch.finfo(scale_dtype).tiny, device=device)
        rounded_scale = rounded_scale.clamp_min(min_normal)
        indices = assign(rounded_scale)
        selected = codewords[indices.to(torch.int64)]
        dequantized = (selected * rounded_scale[:, None, None]).reshape(rows, columns)

        scalar = (w / rounded_scale[:, None]).round().clamp_(-1, 1).reshape_as(selected)
        difference = dequantized - w
        weighted_error = (h * difference.reshape_as(h).square()).sum()
        weighted_base = (h * wg.square()).sum().clamp_min(1e-30)
        max_rel = (difference.abs() / w.abs().clamp_min(1e-8)).max()

        indices_cpu = indices.to(device="cpu", dtype=torch.uint16)
        packed = pack_indices(indices_cpu, self.codebook.spec)
        unpacked = unpack_indices(packed, self.codebook.spec)
        if not torch.equal(unpacked, indices_cpu.to(torch.int64)):
            raise AssertionError("packed index round-trip failed")
        scale_cpu = rounded_scale.to(device="cpu", dtype=scale_dtype)
        decoded_cpu = self.codebook.decode(unpacked, dtype=torch.float32)
        roundtrip = (decoded_cpu * scale_cpu.float()[:, None, None]).reshape(rows, columns)
        dequantized_cpu = dequantized.to(device="cpu", dtype=torch.float32)
        if not torch.equal(roundtrip, dequantized_cpu):
            raise AssertionError("packed artifact does not exactly decode to baked weights")

        payload_bytes = packed.numel() + scale_cpu.numel() * scale_cpu.element_size()
        metrics = ProjectionMetrics(
            rows=rows,
            columns=columns,
            groups=rows * groups_per_row,
            rmse=float(difference.square().mean().sqrt()),
            max_abs_error=float(difference.abs().max()),
            max_rel_error=float(max_rel),
            relative_l2_error=float(difference.norm() / w.norm().clamp_min(1e-30)),
            weighted_relative_l2_error=float(torch.sqrt(weighted_error / weighted_base)),
            scalar_pattern_exact_rate=float((selected == scalar).all(dim=-1).float().mean()),
            changed_trit_fraction=float((selected != scalar).float().mean()),
            scale_min=float(rounded_scale.min()),
            scale_max=float(rounded_scale.max()),
            payload_bytes=payload_bytes,
            effective_bpw_with_row_scale=payload_bytes * 8.0 / weight.numel(),
        )
        return ProjectionResult(indices_cpu, packed, scale_cpu, dequantized_cpu, metrics)


# ---------------------------------------------------------------------------
# Model inventory, pattern statistics, and calibration importance
# ---------------------------------------------------------------------------


def classify_linears(
    model: nn.Module,
    target_regexes: Sequence[str] = DEFAULT_TARGET_REGEXES,
    keep_fp_regexes: Sequence[str] = DEFAULT_KEEP_FP_REGEXES,
    *,
    allow_unmatched: bool = False,
) -> tuple[list[tuple[str, nn.Linear]], list[str], list[str]]:
    """Enumerate every Linear and hard-fail on ambiguous/unmatched modules."""
    targets = [re.compile(p) for p in target_regexes]
    keeps = [re.compile(p) for p in keep_fp_regexes]
    selected: list[tuple[str, nn.Linear]] = []
    kept: list[str] = []
    unmatched: list[str] = []
    double: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        is_target = any(p.fullmatch(name) for p in targets)
        is_keep = any(p.fullmatch(name) for p in keeps)
        if is_target and is_keep:
            double.append(name)
        elif is_target:
            if module.bias is not None:
                raise ValueError(f"target {name} has a bias; BitNet projections must be bias-free")
            if module.in_features % (GROUP_SIZE * PAYLOAD_GROUPS):
                raise ValueError(
                    f"target {name} input width {module.in_features} is not block-256 aligned")
            selected.append((name, module))
        elif is_keep:
            kept.append(name)
        else:
            unmatched.append(name)
    if double or (unmatched and not allow_unmatched):
        raise ValueError(
            "enumerate-don't-assume violation:\n"
            f"  doubly matched: {double}\n"
            f"  unmatched: {unmatched}\n"
            "Use explicit --target-regex/--keep-fp-regex entries; "
            "--allow-unmatched-linears is an opt-out.")
    expected = getattr(model.config, "num_hidden_layers", None)
    if (getattr(model.config, "model_type", None) == "llama" and expected is not None
            and tuple(target_regexes) == DEFAULT_TARGET_REGEXES
            and len(selected) != int(expected) * 7):
        raise ValueError(
            f"Llama inventory expected {int(expected) * 7} block projections, "
            f"found {len(selected)}")
    return selected, kept, unmatched


@torch.no_grad()
def collect_pattern_counts(
    linears: Sequence[tuple[str, nn.Linear]],
    *,
    rows_per_linear: int = 256,
    row_chunk: int = 256,
) -> torch.Tensor:
    """Frequency table from ordinary per-row absmean ternarization.

    Sampling is deterministic and evenly spaced.  ``rows_per_linear=0`` scans
    every row.
    """
    counts = torch.zeros(PATTERN_COUNT, dtype=torch.float64)
    powers = torch.tensor([3**i for i in range(GROUP_SIZE)], dtype=torch.int64)
    for _, module in linears:
        weight = module.weight.detach()
        nrows = weight.shape[0]
        if rows_per_linear > 0 and nrows > rows_per_linear:
            row_ids = torch.linspace(0, nrows - 1, rows_per_linear).round().to(torch.int64)
            row_ids = torch.unique(row_ids, sorted=True)
        else:
            row_ids = torch.arange(nrows, dtype=torch.int64)
        for start in range(0, row_ids.numel(), row_chunk):
            ids = row_ids[start:start + row_chunk].to(weight.device)
            rows = weight.index_select(0, ids).float().cpu()
            scale = rows.abs().mean(dim=1, keepdim=True).clamp_min_(1e-12)
            codes = (rows / scale).round().clamp_(-1, 1).to(torch.int64)
            groups = codes.reshape(-1, GROUP_SIZE)
            pattern_ids = ((groups + 1) * powers).sum(dim=1)
            counts += torch.bincount(pattern_ids, minlength=PATTERN_COUNT).double()
    return counts


def _calibration_texts(path: str | Path, tokenizer, limit: int) -> list[str]:
    texts: list[str] = []
    with Path(path).open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            text: str | None = None
            if raw.startswith("{"):
                item = json.loads(raw)
                if isinstance(item.get("messages"), list):
                    text = tokenizer.apply_chat_template(
                        item["messages"], tokenize=False, add_generation_prompt=False)
                else:
                    for key in ("text", "prompt", "content"):
                        if isinstance(item.get(key), str):
                            text = item[key]
                            break
            else:
                text = raw
            if text:
                texts.append(text)
            if len(texts) >= limit:
                break
    if not texts:
        raise ValueError(f"no usable calibration texts in {path}")
    return texts


@torch.no_grad()
def collect_activation_importance(
    model: nn.Module,
    tokenizer,
    linears: Sequence[tuple[str, nn.Linear]],
    texts: Sequence[str],
    *,
    device: torch.device | str,
    seq_len: int = 512,
) -> dict[str, torch.Tensor]:
    """Collect diagonal input covariance E[x^2] with forward-pre hooks."""
    device = torch.device(device)
    sums = {name: torch.zeros(module.in_features, dtype=torch.float32, device=device)
            for name, module in linears}
    counts = {name: 0 for name, _ in linears}
    hooks = []

    def make_hook(name: str):
        def hook(_module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            sums[name].add_(x.square().sum(dim=0))
            counts[name] += x.shape[0]
        return hook

    for name, module in linears:
        hooks.append(module.register_forward_pre_hook(make_hook(name)))
    original_device = next(model.parameters()).device
    model.to(device).eval()
    try:
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=seq_len)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model(**encoded, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
        if original_device != device:
            model.to(original_device)
    result = {}
    for name, _ in linears:
        if counts[name] == 0:
            raise AssertionError(f"calibration hook for {name} never fired")
        value = (sums[name] / counts[name]).float().cpu()
        result[name] = value / value.mean().clamp_min(1e-12)
    return result


# ---------------------------------------------------------------------------
# Reference W2A8 evaluation wrapper
# ---------------------------------------------------------------------------


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token absmax int8 fake quantization used by the BitNet train stack."""
    scale = x.detach().float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / 127.0
    return ((x.float() / scale).round().clamp(-127, 127) * scale).to(x.dtype)


class TQ1ReferenceLinear(nn.Linear):
    """Dense reference execution of packed TQ1 weights with BitNet A8 inputs.

    It is an oracle, not a speed path.  The state-dict keys and shapes remain
    identical to ``nn.Linear``, so the baked checkpoint stays standard HF.
    """

    @classmethod
    def from_linear(cls, source: nn.Linear) -> "TQ1ReferenceLinear":
        out = cls(source.in_features, source.out_features, bias=False,
                  device=source.weight.device, dtype=source.weight.dtype)
        out.weight = source.weight
        out.weight.requires_grad_(source.weight.requires_grad)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = x + (activation_quant(x) - x).detach()
        return F.linear(xq.to(self.weight.dtype), self.weight).to(x.dtype)


def _parent_and_attr(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    if "." not in name:
        return model, name
    parent_name, attr = name.rsplit(".", 1)
    return model.get_submodule(parent_name), attr


def enable_reference_w2a8(model: nn.Module, target_names: Iterable[str]) -> nn.Module:
    """Replace selected baked linears with the state-compatible W2A8 oracle."""
    for name in target_names:
        module = model.get_submodule(name)
        if isinstance(module, TQ1ReferenceLinear):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear")
        parent, attr = _parent_and_attr(model, name)
        setattr(parent, attr, TQ1ReferenceLinear.from_linear(module))
    return model


# ---------------------------------------------------------------------------
# Artifact I/O and end-to-end conversion
# ---------------------------------------------------------------------------


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name!r}") from exc


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def _iq1_reference_provenance(path: str | Path) -> dict[str, object]:
    """Record the read-only llama.cpp IQ1 authority used by this oracle."""
    root = Path(path).expanduser().resolve()
    common = root / "ggml" / "src" / "ggml-common.h"
    quants = root / "ggml" / "src" / "ggml-quants.c"
    result: dict[str, object] = {
        "path": str(root),
        "available": common.is_file() and quants.is_file(),
        "files": [str(common), str(quants)],
    }
    if result["available"]:
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True).stdout.strip()
            result["git_commit"] = head
        except (OSError, subprocess.CalledProcessError):
            result["git_commit"] = None
    return result


def load_iq1_grid_codebook(path: str | Path = "~/llama.cpp") -> TQ1Codebook:
    """Load llama.cpp's exact 2,048-entry ``iq1s_grid`` as a V11 baseline.

    The C table is a uint64 array whose in-memory little-endian bytes are the
    eight int8 trits.  No source is vendored or modified; the artifact records
    the reference checkout and commit.
    """
    root = Path(path).expanduser().resolve()
    source = root / "ggml" / "src" / "ggml-common.h"
    if not source.is_file():
        raise FileNotFoundError(f"IQ1 grid source not found at {source}")
    text = source.read_text()
    match = re.search(
        r"GGML_TABLE_BEGIN\(uint64_t,\s*iq1s_grid,\s*NGRID_IQ1S\)"
        r"(.*?)GGML_TABLE_END\(\)",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"could not locate iq1s_grid in {source}")
    packed = [int(token, 16) for token in re.findall(r"0x[0-9a-fA-F]+", match.group(1))]
    if len(packed) != FORMATS["v11"].n_indices:
        raise ValueError(f"expected 2,048 IQ1 entries, found {len(packed)}")
    grid = torch.empty((len(packed), GROUP_SIZE), dtype=torch.int8)
    for row, value in enumerate(packed):
        for lane in range(GROUP_SIZE):
            byte = (value >> (8 * lane)) & 0xFF
            grid[row, lane] = byte - 256 if byte >= 128 else byte
    provenance = _iq1_reference_provenance(root)
    return TQ1Codebook(
        FORMATS["v11"],
        grid,
        construction={
            "algorithm": "llama.cpp-iq1s-grid-baseline",
            "reference": provenance,
        },
        encoding="joint",
    )


def _resolved_revision(model_id: str, revision: str, local_only: bool) -> str:
    if Path(model_id).exists() or local_only:
        return revision
    try:
        from huggingface_hub import model_info
        return str(model_info(model_id, revision=revision).sha)
    except Exception as exc:  # model loading below will provide the authoritative error
        print(f"[source] WARNING: could not resolve revision {revision!r}: {exc}")
        return revision


def _load_codebook(path: str | Path, spec: TQ1Format) -> TQ1Codebook:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        encoding = metadata.get("codebook_encoding", "sign_canonical")
        key = ("__tq1_v_codebook_shapes" if encoding == "sign_canonical"
               else "__tq1_v_codebook")
        table = handle.get_tensor(key)
    return TQ1Codebook(
        spec, table,
        construction={"algorithm": "loaded", "source": str(path)},
        encoding=encoding)


def decode_artifact_weight(packed: torch.Tensor, scales: torch.Tensor,
                           codebook: TQ1Codebook) -> torch.Tensor:
    indices = unpack_indices(packed, codebook.spec)
    trits = codebook.decode(indices, dtype=torch.float32)
    return (trits * scales.float()[:, None, None]).reshape(scales.shape[0], -1)


def quantize_model(
    model: nn.Module,
    linears: Sequence[tuple[str, nn.Linear]],
    codebook: TQ1Codebook,
    *,
    device: torch.device | str,
    candidate_count: int,
    chunk_groups: int,
    metric: str,
    iterations: int,
    scale_dtype: torch.dtype,
    activation_importance: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, object]]]:
    """Quantize in place and return packed tensors plus per-tensor reports."""
    projector = TQ1Projector(codebook, candidate_count=candidate_count,
                             device=device, chunk_groups=chunk_groups)
    codebook_key = ("__tq1_v_codebook_shapes"
                    if codebook.encoding == "sign_canonical"
                    else "__tq1_v_codebook")
    packed_tensors: dict[str, torch.Tensor] = {codebook_key: codebook.shapes}
    reports: dict[str, dict[str, object]] = {}
    started = time.monotonic()
    for number, (name, module) in enumerate(linears, start=1):
        t0 = time.monotonic()
        importance = activation_importance.get(name) if activation_importance else None
        result = projector.project(
            module.weight,
            activation_importance=importance,
            metric=metric,
            iterations=iterations,
            scale_dtype=scale_dtype,
        )
        # FP32 baking (the default) preserves a rounded half scale exactly.  A
        # half/bfloat load is also exact when its dtype matches scale_dtype.
        module.weight.data.copy_(result.dequantized.to(
            device=module.weight.device, dtype=module.weight.dtype))
        packed_tensors[f"{name}.weight.__tq1_indices"] = result.packed
        packed_tensors[f"{name}.weight.__tq1_scale"] = result.scales
        report = asdict(result.metrics)
        report["shape"] = list(module.weight.shape)
        report["elapsed_seconds"] = time.monotonic() - t0
        reports[f"{name}.weight"] = report
        print(
            f"[quant] {number:3d}/{len(linears)} {name:58s} "
            f"rel_l2={result.metrics.relative_l2_error:.5f} "
            f"changed={result.metrics.changed_trit_fraction:.3%} "
            f"bpw={result.metrics.effective_bpw_with_row_scale:.4f} "
            f"{report['elapsed_seconds']:.1f}s",
            flush=True,
        )
    print(f"[quant] projected {len(linears)} tensors in {time.monotonic() - started:.1f}s")
    return packed_tensors, reports


def _aggregate_report(tensor_reports: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    weights = [int(r["rows"]) * int(r["columns"]) for r in tensor_reports.values()]
    total = max(sum(weights), 1)
    payload = sum(int(r["payload_bytes"]) for r in tensor_reports.values())
    return {
        "ternary_parameters": sum(weights),
        "packed_payload_bytes": payload,
        "effective_bpw_with_row_scales": payload * 8.0 / total,
        "weighted_mean_relative_l2_error": sum(
            n * float(r["relative_l2_error"])
            for n, r in zip(weights, tensor_reports.values())
        ) / total,
        "max_abs_error": max((float(r["max_abs_error"]) for r in tensor_reports.values()),
                             default=0.0),
    }


def _save_artifacts(
    output: Path,
    model,
    tokenizer,
    packed_tensors: Mapping[str, torch.Tensor],
    report: dict[str, object],
    *,
    max_shard_size: str,
) -> None:
    from safetensors.torch import save_file
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(output)
    packed_path = output / f"{str(report['format']).lower()}.safetensors"
    save_file(
        dict(packed_tensors),
        str(packed_path),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "format": str(report["format"]),
            "codebook_hash": str(report["codebook_hash"]),
            "codebook_encoding": str(report["codebook_encoding"]),
            "source_model": str(report["source_model"]),
            "source_revision": str(report["source_revision"]),
        },
    )
    report["packed_artifact"] = packed_path.name
    (output / "quantization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Importance-aware strict TQ1_V quantizer for Llama-3.2-1B-Instruct")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HF model ID or local checkpoint (default: {DEFAULT_MODEL})")
    parser.add_argument("--revision", default="main",
                        help="HF revision; resolved to an immutable commit in the report")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--format", default="v12", choices=sorted(FORMATS))
    parser.add_argument("--device", default="auto",
                        help="projection device: auto, cpu, mps, cuda, ...")
    parser.add_argument("--load-dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="FP32 is the correctness-first baked-checkpoint default")
    parser.add_argument("--scale-dtype", default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--metric", default="iq1",
                        choices=["uniform", "activation", "iq1"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=32,
                        help="nearby codewords evaluated per eight-weight group")
    parser.add_argument("--chunk-groups", type=int, default=16384,
                        help="projection work chunk; reduce on memory pressure")
    parser.add_argument("--codebook", default=None,
                        help="reuse the codebook from an existing TQ1 safetensors artifact")
    parser.add_argument("--codebook-source", default="learned",
                        choices=["learned", "universal", "iq1"],
                        help="iq1 is the exact ~/llama.cpp grid and requires V11")
    parser.add_argument("--codebook-rows-per-linear", type=int, default=256,
                        help="deterministic rows sampled for frequencies; 0 scans all")
    parser.add_argument("--frequency-fraction", type=float, default=0.75)
    parser.add_argument("--calibration-file", default=None,
                        help="optional text/JSONL for activation second moments")
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--calibration-seq-len", type=int, default=512)
    parser.add_argument("--calibration-device", default=None,
                        help="defaults to --device")
    parser.add_argument("--target-regex", action="append", default=None,
                        help="fullmatch regex; repeat to replace Llama defaults")
    parser.add_argument("--keep-fp-regex", action="append", default=None,
                        help="fullmatch regex; repeat to replace the lm_head default")
    parser.add_argument("--allow-unmatched-linears", action="store_true")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--iq1-reference-dir", default="~/llama.cpp",
                        help="read-only llama.cpp IQ1 authority recorded in provenance")
    parser.add_argument("--verify-prompt", default=None,
                        help="optional W2A8 reference-forward smoke prompt")
    return parser


def legacy_main(argv: Sequence[str] | None = None) -> int:
    args = build_legacy_parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tf_version
    except ImportError as exc:
        raise SystemExit("transformers is required; use the repository .venv") from exc

    spec = _format(args.format)
    projection_device = _device(args.device)
    load_dtype = _torch_dtype(args.load_dtype)
    scale_dtype = _torch_dtype(args.scale_dtype)
    if load_dtype != torch.float32 and load_dtype != scale_dtype:
        raise SystemExit(
            "a non-FP32 baked checkpoint must use matching --load-dtype and "
            "--scale-dtype so every external row scale remains exactly representable")
    revision = _resolved_revision(args.model, args.revision, args.local_files_only)
    print(f"[source] loading {args.model}@{revision} as {load_dtype} on CPU")
    load_kwargs = {
        "revision": revision,
        "dtype": load_dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": False,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, local_files_only=args.local_files_only,
        trust_remote_code=False)
    config_before = model.config.to_dict()

    target_regexes = tuple(args.target_regex or DEFAULT_TARGET_REGEXES)
    keep_regexes = tuple(args.keep_fp_regex or DEFAULT_KEEP_FP_REGEXES)
    linears, kept, unmatched = classify_linears(
        model, target_regexes, keep_regexes,
        allow_unmatched=args.allow_unmatched_linears)
    target_params = sum(module.weight.numel() for _, module in linears)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[inventory] targets={len(linears)} kept_fp={len(kept)} "
        f"unmatched={len(unmatched)} ternary_params={target_params:,} "
        f"({target_params / max(total_params, 1):.2%} of model)")

    if args.codebook:
        codebook = _load_codebook(args.codebook, spec)
    elif args.codebook_source == "iq1":
        if spec != FORMATS["v11"]:
            raise SystemExit("--codebook-source iq1 requires --format v11")
        print(f"[codebook] loading IQ1 grid from {args.iq1_reference_dir}")
        codebook = load_iq1_grid_codebook(args.iq1_reference_dir)
    elif args.codebook_source == "universal":
        print(f"[codebook] building universal {spec.n_shapes:,}-shape table")
        codebook = learn_sign_canonical_codebook(
            spec, None, frequency_fraction=0.0)
    else:
        print("[codebook] collecting scalar-ternary pattern frequencies")
        pattern_counts = collect_pattern_counts(
            linears, rows_per_linear=args.codebook_rows_per_linear)
        print(f"[codebook] learning {spec.n_shapes:,} sign-canonical shapes")
        codebook = learn_sign_canonical_codebook(
            spec, pattern_counts, frequency_fraction=args.frequency_fraction)
    coverage = codebook.construction.get("coverage")
    if coverage is None:
        coverage = codebook_coverage(codebook)
        codebook.construction["coverage"] = coverage
    print(
        f"[codebook] hash={codebook.hash()} "
        f"coverage={coverage}")

    importance = None
    if args.calibration_file:
        texts = _calibration_texts(args.calibration_file, tokenizer,
                                   args.calibration_samples)
        calibration_device = _device(args.calibration_device or args.device)
        print(
            f"[calibration] collecting E[x^2] from {len(texts)} samples "
            f"on {calibration_device}")
        importance = collect_activation_importance(
            model, tokenizer, linears, texts, device=calibration_device,
            seq_len=args.calibration_seq_len)
    elif args.metric == "activation":
        raise SystemExit("--metric activation requires --calibration-file")

    packed_tensors, tensor_reports = quantize_model(
        model,
        linears,
        codebook,
        device=projection_device,
        candidate_count=args.candidates,
        chunk_groups=args.chunk_groups,
        metric=args.metric,
        iterations=args.iterations,
        scale_dtype=scale_dtype,
        activation_importance=importance,
    )
    if model.config.to_dict() != config_before:
        raise AssertionError("model config drifted during quantization")

    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_model": args.model,
        "source_revision": revision,
        "format": spec.name,
        "strict_ternary_codewords": True,
        "quantization_method": "importance-aware alternating PTQ",
        "qat_performed": False,
        "runtime_compatibility": {
            "reference_hf_baked": True,
            "stock_llama_cpp": False,
            "reason": (
                "legacy schema-1 artifacts are research-only; the pinned "
                "llama.cpp integration consumes canonical schema-2 artifacts"
            ),
        },
        "group_size": GROUP_SIZE,
        "payload_groups": PAYLOAD_GROUPS,
        "index_bits": spec.index_bits,
        "raw_bpw": spec.raw_bpw,
        "row_scale_dtype": args.scale_dtype,
        "metric": args.metric,
        "iterations": args.iterations,
        "candidate_count": args.candidates,
        "codebook_hash": codebook.hash(),
        "codebook_encoding": codebook.encoding,
        "codebook_source": ("artifact" if args.codebook else args.codebook_source),
        "codebook": codebook.construction,
        "target_regexes": list(target_regexes),
        "keep_fp_regexes": list(keep_regexes),
        "target_modules": [name for name, _ in linears],
        "kept_fp_modules": kept,
        "unmatched_linears": unmatched,
        "model_parameters": total_params,
        "ternary_parameter_fraction": target_params / max(total_params, 1),
        "aggregate": _aggregate_report(tensor_reports),
        "tensors": tensor_reports,
        "provenance": {
            "quantizer_source_hash": _source_hash(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": tf_version,
            "platform": platform.platform(),
            "projection_device": str(projection_device),
            "iq1_reference": _iq1_reference_provenance(args.iq1_reference_dir),
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        },
    }
    _save_artifacts(output, model, tokenizer, packed_tensors, report,
                    max_shard_size=args.max_shard_size)
    print(f"[save] wrote baked HF checkpoint, packed artifact, and report to {output}")

    if args.verify_prompt:
        print("[verify] running reference W2A8 forward")
        enable_reference_w2a8(model, (name for name, _ in linears))
        model.to(projection_device).eval()
        encoded = tokenizer(args.verify_prompt, return_tensors="pt")
        encoded = {key: value.to(projection_device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits
        if not torch.isfinite(logits).all():
            raise AssertionError("reference W2A8 forward produced non-finite logits")
        print(f"[verify] finite logits {tuple(logits.shape)}")
    return 0


# Schema 2 is the canonical entry point.  The schema-1 implementation above is
# retained only as an import-compatible research oracle for its historical
# tests; it cannot be selected accidentally from the production CLI.
from bitnet_train.tq1.cli import build_parser, main  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
