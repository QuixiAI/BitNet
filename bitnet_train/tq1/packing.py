"""Bit-exact format-v1 TQ1 physical payload packing."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProfileLayout:
    name: str
    index_bits: int
    raw_index_bytes: int
    block_bytes: int
    scale_mode: str
    affine: bool = False


PROFILE_LAYOUTS = {
    "tq1_v11-j-r": ProfileLayout("tq1_v11-j-r", 11, 44, 44, "row"),
    "tq1_v11-i-r": ProfileLayout("tq1_v11-i-r", 11, 44, 44, "row"),
    "tq1_v11-p-r": ProfileLayout("tq1_v11-p-r", 11, 44, 44, "row"),
    "tq1_v12-j-r": ProfileLayout("tq1_v12-j-r", 12, 48, 48, "row"),
    "tq1_v12-p-r": ProfileLayout("tq1_v12-p-r", 12, 48, 48, "row"),
    "tq1_v11-j-b": ProfileLayout("tq1_v11-j-b", 11, 44, 46, "block256"),
    "tq1_v12-j-b": ProfileLayout("tq1_v12-j-b", 12, 48, 50, "block256"),
    "tq1_v11-j-a4-r": ProfileLayout("tq1_v11-j-a4-r", 11, 44, 48, "row", True),
}


def layout(profile: str | ProfileLayout) -> ProfileLayout:
    if isinstance(profile, ProfileLayout):
        return profile
    try:
        return PROFILE_LAYOUTS[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported TQ1 profile {profile!r}") from exc


def pack_indices(indices: torch.Tensor, profile: str | ProfileLayout) -> torch.Tensor:
    spec = layout(profile)
    values = indices.detach().to(torch.int64).cpu()
    if values.ndim < 1 or values.shape[-1] % 32:
        raise ValueError("index rows must contain a multiple of 32 groups")
    if values.numel() and (int(values.min()) < 0 or int(values.max()) >= (1 << spec.index_bits)):
        raise ValueError(f"index is outside the {spec.index_bits}-bit range")
    leading = values.shape[:-1]
    blocks = values.shape[-1] // 32
    flat = values.reshape(-1, 32)
    low = (flat & 0xff).to(torch.uint8)
    high = flat >> 8
    high_bits = spec.index_bits - 8
    qh = torch.zeros((flat.shape[0], high_bits * 4), dtype=torch.int64)
    for group in range(32):
        bit_position = group * high_bits
        byte, shift = divmod(bit_position, 8)
        qh[:, byte] |= (high[:, group] << shift) & 0xff
        if shift + high_bits > 8:
            qh[:, byte + 1] |= high[:, group] >> (8 - shift)
    packed = torch.cat((low, qh.to(torch.uint8)), dim=1)
    return packed.reshape(*leading, blocks, spec.raw_index_bytes)


def unpack_indices(payload: torch.Tensor, profile: str | ProfileLayout) -> torch.Tensor:
    spec = layout(profile)
    data = payload.detach().to(torch.uint8).cpu()
    if data.ndim < 2 or data.shape[-1] != spec.raw_index_bytes:
        raise ValueError(f"index payload must end in {spec.raw_index_bytes} bytes")
    leading = data.shape[:-2]
    blocks = data.shape[-2]
    flat = data.reshape(-1, spec.raw_index_bytes).to(torch.int64)
    low, qh = flat[:, :32], flat[:, 32:]
    high_bits = spec.index_bits - 8
    values = torch.empty((flat.shape[0], 32), dtype=torch.int64)
    for group in range(32):
        bit_position = group * high_bits
        byte, shift = divmod(bit_position, 8)
        high = qh[:, byte] >> shift
        if shift + high_bits > 8:
            high |= qh[:, byte + 1] << (8 - shift)
        high &= (1 << high_bits) - 1
        values[:, group] = low[:, group] | (high << 8)
    return values.reshape(*leading, blocks * 32)


def _fp16_bytes(values: torch.Tensor) -> torch.Tensor:
    fp16 = values.detach().to(device="cpu", dtype=torch.float16).contiguous()
    return fp16.view(torch.uint8).reshape(*fp16.shape, 2)


def _fp16_from_bytes(values: torch.Tensor) -> torch.Tensor:
    raw = values.detach().to(torch.uint8).cpu().contiguous()
    if raw.shape[-1] != 2:
        raise ValueError("FP16 byte payload must end in two bytes")
    return raw.reshape(*raw.shape[:-1], 2).view(torch.float16).squeeze(-1)


def pack_payload(indices: torch.Tensor, profile: str | ProfileLayout, *,
                 block_scales: torch.Tensor | None = None,
                 affine_nibbles: torch.Tensor | None = None) -> torch.Tensor:
    spec = layout(profile)
    packed = pack_indices(indices, spec)
    if spec.scale_mode == "block256":
        if block_scales is None or tuple(block_scales.shape) != tuple(packed.shape[:-1]):
            raise ValueError("block-scale payload requires one scale per payload block")
        if affine_nibbles is not None:
            raise ValueError("block-scale payload cannot contain affine metadata")
        return torch.cat((_fp16_bytes(block_scales), packed), dim=-1)
    if block_scales is not None:
        raise ValueError("row-scale payloads store scales in a companion tensor")
    if not spec.affine:
        if affine_nibbles is not None:
            raise ValueError("strict payload cannot contain affine metadata")
        return packed
    if affine_nibbles is None or tuple(affine_nibbles.shape) != (*packed.shape[:-1], 8):
        raise ValueError("A4 requires eight affine nibbles per payload block")
    nibble = affine_nibbles.detach().to(torch.int64).cpu()
    if nibble.numel() and (int(nibble.min()) < 0 or int(nibble.max()) > 11):
        raise ValueError("A4 nibble uses a reserved value outside the 12 legal rho/mu pairs")
    if torch.any(((nibble >> 2) & 3) == 3):
        raise ValueError("A4 reserved mu_id=3 is forbidden")
    affine = (nibble[..., 0::2] | (nibble[..., 1::2] << 4)).to(torch.uint8)
    return torch.cat((packed, affine), dim=-1)


def unpack_payload(payload: torch.Tensor, profile: str | ProfileLayout) \
        -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    spec = layout(profile)
    data = payload.detach().to(torch.uint8).cpu()
    if data.ndim < 2 or data.shape[-1] != spec.block_bytes:
        raise ValueError(f"{spec.name} payload must end in {spec.block_bytes} bytes")
    if spec.scale_mode == "block256":
        scales = _fp16_from_bytes(data[..., :2])
        indices = unpack_indices(data[..., 2:], spec)
        return indices, scales, None
    indices = unpack_indices(data[..., :spec.raw_index_bytes], spec)
    if not spec.affine:
        return indices, None, None
    packed = data[..., spec.raw_index_bytes:]
    nibbles = torch.empty((*packed.shape[:-1], 8), dtype=torch.uint8)
    nibbles[..., 0::2] = packed & 0x0f
    nibbles[..., 1::2] = packed >> 4
    if torch.any(((nibbles >> 2) & 3) == 3):
        raise ValueError("payload contains reserved A4 mu_id=3")
    return indices, None, nibbles
