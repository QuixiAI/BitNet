"""Explicit tied Q2/Q4/Q8 fallbacks for the QI-2 quality gate.

These are deliberately named alternatives, not silent mixed-precision escape
hatches.  They preserve one shared tensor and use one FP16 scale per 128
weights.  They are evaluation/training reference modules; promotion to a GGUF
physical type still requires a QI-1 row and the normal runtime-format gate.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SharedFallbackSpec:
    bits: int
    group_size: int = 128
    scale_dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.bits not in {2, 4, 8}:
            raise ValueError("shared fallback bits must be Q2, Q4, or Q8")
        if self.group_size != 128:
            raise ValueError("QI-2 shared fallbacks use group size 128")
        if self.scale_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("shared fallback scales must be FP16 or BF16")

    @property
    def qmax(self) -> int:
        return {2: 1, 4: 7, 8: 127}[self.bits]

    @property
    def physical_type(self) -> str:
        return f"SHARED_Q{self.bits}_G128_{str(self.scale_dtype).removeprefix('torch.').upper()}"


@dataclass(frozen=True)
class PackedSharedFallback:
    payload: torch.Tensor
    scales: torch.Tensor
    logical_shape: tuple[int, int]
    spec: SharedFallbackSpec

    @property
    def physical_bytes(self) -> int:
        return (self.payload.numel() * self.payload.element_size()
                + self.scales.numel() * self.scales.element_size())


def quantize_shared_fallback(weight: torch.Tensor, spec: SharedFallbackSpec) \
        -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.shape[1] % spec.group_size:
        raise ValueError("shared fallback weight must be [N,K], K divisible by 128")
    value = weight.detach().float().reshape(weight.shape[0], -1, spec.group_size)
    maximum = value.abs().amax(-1)
    scale = maximum / spec.qmax
    denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(value / denominator[..., None]).clamp(
        -spec.qmax, spec.qmax).to(torch.int8)
    codes[maximum == 0] = 0
    scale = scale.to(spec.scale_dtype)
    if not torch.isfinite(scale).all():
        raise ValueError("shared fallback scales are nonfinite")
    return codes.reshape_as(weight), scale


def dequantize_shared_fallback(codes: torch.Tensor, scales: torch.Tensor,
                               spec: SharedFallbackSpec) -> torch.Tensor:
    if codes.ndim != 2 or codes.dtype != torch.int8 \
            or codes.shape[1] % spec.group_size:
        raise ValueError("shared fallback codes have an invalid shape/dtype")
    expected = (codes.shape[0], codes.shape[1] // spec.group_size)
    if tuple(scales.shape) != expected or scales.dtype != spec.scale_dtype:
        raise ValueError("shared fallback scales have an invalid shape/dtype")
    if torch.any(codes < -spec.qmax) or torch.any(codes > spec.qmax):
        raise ValueError("shared fallback code is outside the declared range")
    return (codes.float().reshape(*expected, spec.group_size)
            * scales.float()[..., None]).reshape_as(codes).float()


def pack_shared_fallback(codes: torch.Tensor, scales: torch.Tensor,
                         spec: SharedFallbackSpec) -> PackedSharedFallback:
    dequantize_shared_fallback(codes, scales, spec)  # complete validation
    rows, width = codes.shape
    if spec.bits == 8:
        payload = codes.contiguous().view(torch.uint8).cpu()
    else:
        offset = spec.qmax
        unsigned = (codes.to(torch.int16) + offset).to(torch.uint8).reshape(rows, -1)
        if spec.bits == 4:
            payload = (unsigned[:, 0::2]
                       | torch.bitwise_left_shift(unsigned[:, 1::2], 4))
        else:
            payload = (unsigned[:, 0::4]
                       | torch.bitwise_left_shift(unsigned[:, 1::4], 2)
                       | torch.bitwise_left_shift(unsigned[:, 2::4], 4)
                       | torch.bitwise_left_shift(unsigned[:, 3::4], 6))
        payload = payload.contiguous().cpu()
    return PackedSharedFallback(payload, scales.detach().contiguous().cpu(),
                                (rows, width), spec)


def unpack_shared_fallback(packed: PackedSharedFallback) -> torch.Tensor:
    rows, width = packed.logical_shape
    spec = packed.spec
    if spec.bits == 8:
        expected = rows * width
        if packed.payload.numel() != expected:
            raise ValueError("Q8 shared fallback payload size mismatch")
        codes = packed.payload.contiguous().view(torch.int8).reshape(rows, width)
    else:
        per_byte = 8 // spec.bits
        if packed.payload.numel() != rows * width // per_byte:
            raise ValueError("shared fallback packed payload size mismatch")
        shifts = range(0, 8, spec.bits)
        mask = (1 << spec.bits) - 1
        lanes = [torch.bitwise_and(
            torch.bitwise_right_shift(packed.payload, shift), mask)
                 for shift in shifts]
        codes = torch.stack(lanes, -1).reshape(rows, width).to(torch.int16)
        # The all-ones spelling is reserved by the symmetric Q2/Q4 contract.
        if torch.any(codes == mask):
            raise ValueError("shared fallback payload contains a reserved code")
        codes = (codes - spec.qmax).to(torch.int8)
    dequantize_shared_fallback(codes, packed.scales, spec)
    return codes


class SharedFallbackEmbedding(nn.Module):
    """One latent Q2/Q4/Q8 fake-quant weight with lookup and head consumers."""

    def __init__(self, weight: torch.Tensor, spec: SharedFallbackSpec, *,
                 padding_idx: int | None = None):
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] % spec.group_size:
            raise ValueError("shared fallback latent shape is invalid")
        self.weight = nn.Parameter(weight.detach().float().clone())
        self.spec = spec
        self.padding_idx = padding_idx

    @property
    def num_embeddings(self) -> int:
        return self.weight.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.weight.shape[1]

    def projected_weight(self) -> torch.Tensor:
        codes, scales = quantize_shared_fallback(self.weight, self.spec)
        hard = dequantize_shared_fallback(codes, scales, self.spec).to(self.weight.device)
        return self.weight + (hard - self.weight).detach()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.projected_weight(), self.padding_idx)

    def linear(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.projected_weight().to(hidden.dtype))

    @torch.no_grad()
    def export(self) -> PackedSharedFallback:
        codes, scales = quantize_shared_fallback(self.weight, self.spec)
        return pack_shared_fallback(codes.cpu(), scales.cpu(), self.spec)


class SharedFallbackOutputHead(nn.Module):
    def __init__(self, shared: SharedFallbackEmbedding):
        super().__init__()
        object.__setattr__(self, "_shared_ref", weakref.ref(shared))
        self.in_features = shared.embedding_dim
        self.out_features = shared.num_embeddings
        self.bias = None

    @property
    def shared_weight(self) -> SharedFallbackEmbedding:
        value = self._shared_ref()
        if value is None:  # pragma: no cover
            raise RuntimeError("shared fallback embedding was destroyed")
        return value

    @property
    def weight(self) -> nn.Parameter:
        return self.shared_weight.weight

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "weight" and "_shared_ref" in self.__dict__:
            if value is not self.shared_weight.weight:
                raise ValueError("cannot untie a shared fallback output head")
            return
        super().__setattr__(name, value)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.shared_weight.linear(hidden)


class PackedSharedFallbackEmbedding(nn.Module):
    """Inference reference over one packed shared fallback payload."""

    def __init__(self, packed: PackedSharedFallback,
                 *, output_dtype: torch.dtype = torch.float32,
                 padding_idx: int | None = None):
        super().__init__()
        codes = unpack_shared_fallback(packed)
        self.register_buffer("codes", codes)
        self.register_buffer("scales", packed.scales.clone())
        self.spec = packed.spec
        self.output_dtype = output_dtype
        self.padding_idx = padding_idx
        self.logical_shape = list(packed.logical_shape)

    def weight(self) -> torch.Tensor:
        return dequantize_shared_fallback(self.codes, self.scales, self.spec)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        unique, inverse = torch.unique(ids.cpu().reshape(-1), sorted=True,
                                       return_inverse=True)
        rows = self.weight()[unique]
        return rows[inverse].reshape(*ids.shape, self.logical_shape[1]).to(
            device=ids.device, dtype=self.output_dtype)

    def linear(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight().to(device=hidden.device, dtype=hidden.dtype))


class PackedSharedFallbackOutputHead(nn.Module):
    """Parameter-free inference consumer of one packed fallback matrix."""

    def __init__(self, shared: PackedSharedFallbackEmbedding):
        super().__init__()
        if not isinstance(shared, PackedSharedFallbackEmbedding):
            raise TypeError("packed fallback head requires a packed shared embedding")
        object.__setattr__(self, "shared_weight", shared)
        self.in_features = shared.logical_shape[1]
        self.out_features = shared.logical_shape[0]
        self.bias = None

    @property
    def weight(self) -> PackedSharedFallbackEmbedding:
        return self.shared_weight

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "weight" and "shared_weight" in self.__dict__:
            if value is not self.shared_weight:
                raise ValueError("cannot untie a packed shared fallback head")
            return
        super().__setattr__(name, value)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.shared_weight.linear(hidden)
