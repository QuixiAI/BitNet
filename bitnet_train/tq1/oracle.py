"""Permanent pure-PyTorch TQ1 decode and W2A8 scalar oracle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .codebook import Codebook
from .packing import layout, unpack_payload


@dataclass(frozen=True)
class ActivationCodes:
    codes: torch.Tensor
    scales: torch.Tensor
    mode: str

    def dequantize(self) -> torch.Tensor:
        if self.mode == "a8_token":
            return self.codes.float() * self.scales[..., None]
        return (self.codes.float().reshape(*self.codes.shape[:-1], -1, 256)
                * self.scales[..., None]).reshape_as(self.codes)


def quantize_activation(x: torch.Tensor, mode: str = "a8_token") -> ActivationCodes:
    value = x.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError("activation contains NaN or infinity")
    if mode == "a8_token":
        maximum = value.abs().amax(dim=-1)
        scale = maximum / 127.0
        denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round(value / denominator[..., None]).clamp(-127, 127).to(torch.int8)
        codes = torch.where((scale > 0)[..., None], codes, torch.zeros_like(codes))
        return ActivationCodes(codes, scale, mode)
    if mode == "a8_block256":
        if value.shape[-1] % 256:
            raise ValueError("a8_block256 requires a width divisible by 256")
        blocks = value.reshape(*value.shape[:-1], -1, 256)
        maximum = blocks.abs().amax(dim=-1)
        scale = maximum / 127.0
        denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round(blocks / denominator[..., None]).clamp(-127, 127).to(torch.int8)
        codes = torch.where((scale > 0)[..., None], codes, torch.zeros_like(codes))
        return ActivationCodes(codes.reshape_as(value), scale, mode)
    raise ValueError(f"unsupported activation mode {mode!r}")


def _validate_profile_codebook(profile: str, codebook: Codebook) -> None:
    spec = layout(profile)
    if codebook.index_bits != spec.index_bits:
        raise ValueError("payload and codebook index widths disagree")
    expected = "sign_canonical"
    if "-i-" in profile:
        expected = "direct_joint"
    elif "-p-" in profile:
        expected = "product"
    if codebook.encoding != expected:
        raise ValueError("payload profile and codebook encoding disagree")


def decode_normalized(payload: torch.Tensor, profile: str, codebook: Codebook) \
        -> tuple[torch.Tensor, torch.Tensor | None]:
    """Decode to normalized weights [...,N,K] plus optional block scales."""
    _validate_profile_codebook(profile, codebook)
    indices, block_scales, affine = unpack_payload(payload, profile)
    codebook.validate_indices(indices)
    codewords = codebook.decode(indices)
    if affine is None:
        return codewords, block_scales
    # Eight affine subblocks per 256 weights; each controls four codewords.
    rho_num = torch.tensor([6, 7, 8, 9], dtype=torch.int64)
    mu_num = torch.tensor([0, 1, -1], dtype=torch.int64)
    affine_flat = affine.reshape(*affine.shape[:-2], -1).to(torch.int64)
    rho_id = (affine_flat & 3).repeat_interleave(4, dim=-1)
    mu_id = ((affine_flat >> 2) & 3).repeat_interleave(4, dim=-1)
    if torch.any(mu_id == 3):
        raise ValueError("payload contains the reserved A4 mu value")
    normalized = (rho_num[rho_id][..., None].float()
                  * (8.0 * codewords.float() + mu_num[mu_id][..., None].float()) / 64.0)
    return normalized, None


def dequantize_weight(payload: torch.Tensor, profile: str, codebook: Codebook, *,
                      row_scales: torch.Tensor | None = None) -> torch.Tensor:
    normalized, block_scales = decode_normalized(payload, profile, codebook)
    if layout(profile).scale_mode == "row":
        if row_scales is None or tuple(row_scales.shape) != tuple(normalized.shape[:-2]):
            raise ValueError("row profile requires one scale per logical output row")
        scale = row_scales.detach().float().cpu()[..., None, None]
        weight = normalized.float() * scale
    else:
        if row_scales is not None or block_scales is None:
            raise ValueError("block profile takes scales only from its payload")
        scale = block_scales.float().repeat_interleave(32, dim=-1)[..., None]
        weight = normalized.float() * scale
    return weight.reshape(*weight.shape[:-2], -1)


def _strict_group_dots(codes: torch.Tensor, codewords: torch.Tensor) -> torch.Tensor:
    # codes [M,G,8], codewords [N,G,8] -> exact signed int32 [M,N,G]
    products = (codes[:, None].to(torch.int32) * codewords[None].to(torch.int32))
    return products.sum(dim=-1, dtype=torch.int32)


def linear_w2a8(x: torch.Tensor, payload: torch.Tensor, profile: str,
                codebook: Codebook, *, row_scales: torch.Tensor | None = None,
                activation_mode: str = "a8_token",
                output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Execute a logical 2-D weight matrix through the scalar packed oracle."""
    if payload.ndim != 3:
        raise ValueError("linear oracle currently requires payload [N,K/256,bytes]")
    original_shape = x.shape[:-1]
    x2 = x.detach().float().cpu().reshape(-1, x.shape[-1])
    if 127 * x2.shape[1] > 2**31 - 1:
        raise ValueError("W2A8 int32 accumulator range is exceeded")
    activation = quantize_activation(x2, activation_mode)
    q = activation.codes.reshape(x2.shape[0], -1, 8)
    indices, block_scales, affine = unpack_payload(payload, profile)
    codebook.validate_indices(indices)
    codewords = codebook.decode(indices)
    if codewords.shape[1] * 8 != x2.shape[1]:
        raise ValueError("activation width and packed weight width disagree")
    if affine is not None:
        # Sum exact /64 rational numerators per 32-weight affine subblock.
        dots = _strict_group_dots(q, codewords).reshape(x2.shape[0], payload.shape[0], -1, 4)
        qsum = q.reshape(x2.shape[0], -1, 4, 8).sum((-1, -2), dtype=torch.int64)
        rho_num = torch.tensor([6, 7, 8, 9], dtype=torch.int64)
        mu_num = torch.tensor([0, 1, -1], dtype=torch.int64)
        affine_flat = affine.reshape(affine.shape[0], -1).to(torch.int64)
        rho_id = affine_flat & 3
        mu_id = (affine_flat >> 2) & 3
        if torch.any(mu_id == 3):
            raise ValueError("payload contains reserved A4 metadata")
        numerator = rho_num[rho_id][None] * (
            8 * dots.sum(-1, dtype=torch.int64)
            + mu_num[mu_id][None] * qsum[:, None]
        )
        if activation_mode == "a8_token":
            acc = numerator.sum(-1).float() / 64.0
        else:
            # Eight affine 32-weight subblocks share one activation-scale block.
            acc = numerator.reshape(x2.shape[0], payload.shape[0], -1, 8).sum(
                -1, dtype=torch.int64).float() / 64.0
    else:
        dots = _strict_group_dots(q, codewords)
        acc = dots
    spec = layout(profile)
    if spec.scale_mode == "row":
        if row_scales is None or tuple(row_scales.shape) != (payload.shape[0],):
            raise ValueError("row profile requires [N] companion scales")
        weight_scale = row_scales.detach().float().cpu()[None]
        if activation_mode == "a8_token":
            if acc.ndim == 3:
                acc = acc.sum(-1, dtype=torch.int32).float()
            out = acc * activation.scales[:, None] * weight_scale
        else:
            block_acc = (acc if affine is not None else
                         acc.reshape(x2.shape[0], payload.shape[0], -1, 32).sum(
                             -1, dtype=torch.int32).float())
            out = (block_acc * activation.scales[:, None]).sum(-1) * weight_scale
    else:
        if row_scales is not None or block_scales is None:
            raise ValueError("block-scale profile reads scales from payload")
        block_acc = acc.reshape(x2.shape[0], payload.shape[0], -1, 32).sum(
            -1, dtype=torch.int32).float()
        if activation_mode == "a8_token":
            out = (block_acc * block_scales[None].float()).sum(-1)
            out *= activation.scales[:, None]
        else:
            out = (block_acc * block_scales[None].float()
                   * activation.scales[:, None]).sum(-1)
    return out.to(output_dtype).reshape(*original_shape, payload.shape[0])


def linear_w_only(x: torch.Tensor, payload: torch.Tensor, profile: str,
                  codebook: Codebook, *, row_scales: torch.Tensor | None = None,
                  output_dtype: torch.dtype | None = None) -> torch.Tensor:
    weight = dequantize_weight(payload, profile, codebook, row_scales=row_scales)
    result = torch.nn.functional.linear(x.detach().float().cpu(), weight)
    return result.to(output_dtype or x.dtype)
