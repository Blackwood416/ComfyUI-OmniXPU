"""ConvRot W4A4 implementation using omni native rotation and INT8 GEMM."""

from __future__ import annotations

import torch
from omni_xpu_kernel import int8, svdq

from comfy_kitchen.backends.eager import convrot_w4a4 as eager_convrot

_QUANT_GROUP_SIZE = 64


def _validate_group(quant_group_size: int) -> None:
    if quant_group_size != _QUANT_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {_QUANT_GROUP_SIZE}")


def prepare_int4_weight_for_int8_linear(weight: torch.Tensor) -> torch.Tensor:
    """Unpack persistent signed W4 storage only when an INT8 GEMM path needs it."""
    return svdq.unpack_int4(weight.view(torch.uint8), signed=True)


def quantize_and_rotate_rowwise(
    x: torch.Tensor,
    H: torch.Tensor,  # noqa: N803
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Honor the caller-provided rotation matrix, then use omni row quantization."""
    if x.shape[-1] % group_size:
        raise ValueError(f"features {x.shape[-1]} not divisible by group_size {group_size}")
    grouped = x.reshape(-1, x.shape[-1] // group_size, group_size)
    rotated = torch.matmul(grouped, H.to(device=x.device, dtype=x.dtype)).reshape_as(x)
    return int8.quantize_int8_rowwise(rotated, stochastic_rounding or 0)


def quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _QUANT_GROUP_SIZE,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_group(quant_group_size)
    eager_convrot.validate_w4a4_shape(weight, convrot_groupsize, quant_group_size)
    rotated = int8._get_native().rotate_convrot(weight, convrot_groupsize)
    return eager_convrot.quantize_signed_int4_rowwise(rotated, stochastic_rounding)


def dequantize_convrot_w4a4_weight(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _QUANT_GROUP_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    _validate_group(quant_group_size)
    unpacked = svdq.unpack_int4(qdata.view(torch.uint8), signed=True).float()
    rotated = unpacked * scales.float().reshape(-1, 1)
    return int8._get_native().rotate_convrot(rotated, convrot_groupsize).to(output_dtype)


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = _QUANT_GROUP_SIZE,
    linear_dtype: str = "int4",
) -> torch.Tensor:
    _validate_group(quant_group_size)
    if linear_dtype not in {"int4", "int8"}:
        raise ValueError(
            f"ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got {linear_dtype!r}"
        )
    if x.shape[-1] != qweight.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qweight K={qweight.shape[-1] * 2}")
    if x.shape[-1] % convrot_groupsize:
        raise ValueError(
            f"Input K={x.shape[-1]} not divisible by convrot_groupsize {convrot_groupsize}"
        )

    original_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    rotated = int8._get_native().rotate_convrot(x2d, convrot_groupsize)
    if linear_dtype == "int4":
        packed_act, xscale = eager_convrot.quantize_signed_int4_rowwise(rotated)
        qact = svdq.unpack_int4(packed_act.view(torch.uint8), signed=True)
    else:
        qact, xscale = int8.quantize_int8_rowwise(rotated)
        xscale = xscale.reshape(-1)
    qweight_int8 = svdq.unpack_int4(qweight.view(torch.uint8), signed=True)
    accum = int8.mm_int8(qact, qweight_int8.t().contiguous())
    output = accum.float() * xscale.float().reshape(-1, 1)
    output = output * wscales.float().reshape(1, -1)
    output = output.to(x.dtype)
    if bias is not None:
        output = output + bias.to(x.dtype).reshape(1, -1)
    return output.reshape(*original_shape[:-1], qweight.shape[0])


__all__ = [
    "convrot_w4a4_linear",
    "dequantize_convrot_w4a4_weight",
    "prepare_int4_weight_for_int8_linear",
    "quantize_and_rotate_rowwise",
    "quantize_convrot_w4a4_weight",
]
