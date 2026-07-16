"""Permanent pure-PyTorch TQ1 decode and W2A8 scalar oracle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .codebook import Codebook
from .packing import layout, unpack_payload


def _validate_scales(scales: torch.Tensor, name: str, *, embedded: bool = False) \
        -> torch.Tensor:
    legal_dtypes = {torch.float16} if embedded else {torch.float16, torch.bfloat16}
    if scales.dtype not in legal_dtypes:
        expected = "float16" if embedded else "float16 or bfloat16"
        raise ValueError(f"{name} must use {expected} runtime storage")
    value = scales.detach().cpu()
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _cast_finite_output(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    result = value.to(dtype)
    if not torch.isfinite(result).all():
        raise ValueError("TQ1 oracle output overflows the requested dtype")
    return result


def validate_scale_index_invariants(
        indices: torch.Tensor, scales: torch.Tensor, codebook: Codebook, *,
        block_scales: bool, affine_nibbles: torch.Tensor | None = None) -> None:
    """Require every zero-scale unit to use its canonical zero representation."""
    legal = codebook.legal_index_mask()
    physical = torch.arange(codebook.index_count, dtype=torch.int64)
    zero = torch.nonzero(
        (codebook.decode(physical) == 0).all(-1) & legal).flatten()
    if zero.numel() != 1:
        raise ValueError("codebook does not have exactly one legal zero index")
    zero_index = int(zero[0])
    if block_scales:
        blocks = scales.shape[-1]
        grouped = indices.reshape(-1, blocks, 32)
        mask = scales.reshape(-1, blocks) == 0
        if torch.any(grouped[mask] != zero_index):
            raise ValueError("zero-scale block contains a nonzero codebook index")
        if affine_nibbles is not None:
            raise ValueError("format-v1 block scales cannot carry A4 metadata")
        return
    grouped = indices.reshape(scales.numel(), -1)
    mask = scales.reshape(-1) == 0
    if torch.any(grouped[mask] != zero_index):
        raise ValueError("zero-scale row contains a nonzero codebook index")
    if affine_nibbles is not None:
        affine = affine_nibbles.reshape(scales.numel(), -1)
        if torch.any(affine[mask] != 0):
            raise ValueError("zero-scale row contains nonzero A4 metadata")


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
    if block_scales is not None:
        block_scales = _validate_scales(
            block_scales, "embedded block scales", embedded=True)
        validate_scale_index_invariants(
            indices, block_scales, codebook, block_scales=True,
            affine_nibbles=affine)
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
        row_scales = _validate_scales(row_scales, "row scales")
        indices, _, affine = unpack_payload(payload, profile)
        validate_scale_index_invariants(
            indices, row_scales, codebook, block_scales=False,
            affine_nibbles=affine)
        scale = row_scales.float()[..., None, None]
        weight = normalized.float() * scale
    else:
        if row_scales is not None or block_scales is None:
            raise ValueError("block profile takes scales only from its payload")
        scale = block_scales.float().repeat_interleave(32, dim=-1)[..., None]
        weight = normalized.float() * scale
    result = weight.reshape(*weight.shape[:-2], -1)
    if not torch.isfinite(result).all():
        raise ValueError("decoded weight contains NaN or infinity")
    return result


def _strict_group_dots(codes: torch.Tensor, codewords: torch.Tensor) -> torch.Tensor:
    # codes [M,G,8], codewords [N,G,8] -> exact signed int32 [M,N,G]
    products = (codes[:, None].to(torch.int32) * codewords[None].to(torch.int32))
    return products.sum(dim=-1, dtype=torch.int32)


def linear_w2a8(x: torch.Tensor, payload: torch.Tensor, profile: str,
                codebook: Codebook, *, row_scales: torch.Tensor | None = None,
                activation_mode: str = "a8_token",
                output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Execute a logical 2-D weight matrix through the scalar packed oracle."""
    if output_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("TQ1 oracle output dtype must be float16, bfloat16, or float32")
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
    if block_scales is not None:
        block_scales = _validate_scales(
            block_scales, "embedded block scales", embedded=True)
        validate_scale_index_invariants(
            indices, block_scales, codebook, block_scales=True,
            affine_nibbles=affine)
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
        row_scales = _validate_scales(row_scales, "row scales")
        validate_scale_index_invariants(
            indices, row_scales, codebook, block_scales=False,
            affine_nibbles=affine)
        weight_scale = row_scales.float()[None]
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
    if not torch.isfinite(out).all():
        raise ValueError("TQ1 oracle output contains NaN or infinity")
    return _cast_finite_output(out, output_dtype).reshape(
        *original_shape, payload.shape[0])


def linear_w_only(x: torch.Tensor, payload: torch.Tensor, profile: str,
                  codebook: Codebook, *, row_scales: torch.Tensor | None = None,
                  output_dtype: torch.dtype | None = None) -> torch.Tensor:
    value = x.detach().float().cpu()
    if not torch.isfinite(value).all():
        raise ValueError("activation contains NaN or infinity")
    dtype = output_dtype or x.dtype
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("TQ1 oracle output dtype must be float16, bfloat16, or float32")
    weight = dequantize_weight(payload, profile, codebook, row_scales=row_scales)
    result = torch.nn.functional.linear(value, weight)
    if not torch.isfinite(result).all():
        raise ValueError("TQ1 oracle output contains NaN or infinity")
    return _cast_finite_output(result, dtype)
