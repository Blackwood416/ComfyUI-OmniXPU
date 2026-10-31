# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GGUF block layout formulas follow City96/ComfyUI-GGUF's Apache-2.0
# dequantization implementation.

"""Portable PyTorch reference implementation for GGUF block dequantization."""

from __future__ import annotations

import torch

from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE
from comfy_kitchen.gguf import (
    GGUF_BLOCK_BYTES,
    GGUF_BLOCK_ELEMENTS,
    GGUF_LAYOUT_CODE_TO_NAME,
    GGUF_QUANT_CODE_TO_TYPE,
    _record_gguf_route,
)
from comfy_kitchen.registry import registry

_QK_K = 256
_K_SCALE_SIZE = 12


def _validate_codes(
    quant_type_code: int,
    output_dtype_code: int,
    layout_code: int,
) -> tuple[torch.dtype, str]:
    if quant_type_code not in GGUF_QUANT_CODE_TO_TYPE:
        raise ValueError(f"unsupported GGUF quant_type_code {quant_type_code}")
    if output_dtype_code not in DTYPE_CODE_TO_DTYPE:
        raise ValueError(f"unsupported GGUF output_dtype_code {output_dtype_code}")
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported GGUF output dtype {output_dtype}")
    if layout_code not in GGUF_LAYOUT_CODE_TO_NAME:
        raise ValueError(f"unsupported GGUF layout_code {layout_code}")
    layout = GGUF_LAYOUT_CODE_TO_NAME[layout_code]
    if quant_type_code != 0 and layout != "comfyui":
        raise ValueError(
            f"layout={layout!r} is only defined for q4_0; "
            f"{GGUF_QUANT_CODE_TO_TYPE[quant_type_code]} requires layout='comfyui'"
        )
    return output_dtype, layout


def _blocks(data: torch.Tensor, quant_type_code: int) -> torch.Tensor:
    block_bytes = GGUF_BLOCK_BYTES[quant_type_code]
    if data.dtype != torch.uint8:
        raise TypeError(f"GGUF storage must be torch.uint8, got {data.dtype}")
    if data.numel() % block_bytes:
        raise ValueError(
            f"{GGUF_QUANT_CODE_TO_TYPE[quant_type_code]} data has {data.numel()} bytes; "
            f"expected a multiple of {block_bytes}"
        )
    block_count = data.numel() // block_bytes
    return data.reshape(-1).contiguous().view(block_count, block_bytes)


def _fp16_column(data: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return data.contiguous().view(torch.float16).to(output_dtype)


def _dequantize_q4_0(
    blocks: torch.Tensor,
    output_dtype: torch.dtype,
    layout: str,
) -> torch.Tensor:
    scale = _fp16_column(blocks[:, :2], output_dtype)
    packed = blocks[:, 2:]
    low = (packed & 0x0F).to(torch.int8) - 8
    high = (packed >> 4).to(torch.int8) - 8
    if layout == "comfyui":
        values = torch.cat((low, high), dim=1)
    else:
        values = torch.stack((low, high), dim=-1).reshape(blocks.shape[0], 32)
    return (scale * values.to(output_dtype)).reshape(-1)


def _dequantize_q4_1(
    blocks: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    scale = _fp16_column(blocks[:, :2], output_dtype)
    minimum = _fp16_column(blocks[:, 2:4], output_dtype)
    packed = blocks[:, 4:]
    low = packed & 0x0F
    high = packed >> 4
    values = torch.cat((low, high), dim=1).to(output_dtype)
    return (scale * values + minimum).reshape(-1)


def _dequantize_q8_0(
    blocks: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    scale = _fp16_column(blocks[:, :2], output_dtype)
    values = blocks[:, 2:].view(torch.int8).to(output_dtype)
    return (scale * values).reshape(-1)


def _get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_blocks = scales.shape[0]
    scales = scales.reshape(n_blocks, 3, 4)
    scale_low, min_low, mixed_high = scales.unbind(dim=1)
    scale = torch.cat(
        (
            scale_low & 0x3F,
            (mixed_high & 0x0F) | ((scale_low >> 2) & 0x30),
        ),
        dim=1,
    )
    minimum = torch.cat(
        (
            min_low & 0x3F,
            (mixed_high >> 4) | ((min_low >> 2) & 0x30),
        ),
        dim=1,
    )
    return scale, minimum


def _dequantize_q4_k(
    blocks: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    d = _fp16_column(blocks[:, :2], output_dtype)
    dmin = _fp16_column(blocks[:, 2:4], output_dtype)
    scales, minimum = _get_scale_min(blocks[:, 4 : 4 + _K_SCALE_SIZE])

    group_scale = (d * scales.to(output_dtype)).reshape(blocks.shape[0], 8, 1)
    group_min = (dmin * minimum.to(output_dtype)).reshape(blocks.shape[0], 8, 1)

    packed = blocks[:, 4 + _K_SCALE_SIZE :]
    low = packed.reshape(blocks.shape[0], 4, 32) & 0x0F
    high = packed.reshape(blocks.shape[0], 4, 32) >> 4
    values = torch.stack((low, high), dim=2).reshape(blocks.shape[0], 8, 32)
    return (group_scale * values.to(output_dtype) - group_min).reshape(-1)


def _dequantize_q6_k(
    blocks: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    ql = blocks[:, : _QK_K // 2]
    qh = blocks[:, _QK_K // 2 : 3 * _QK_K // 4]
    scales = blocks[:, 3 * _QK_K // 4 : 3 * _QK_K // 4 + _QK_K // 16]
    d = blocks[:, -2:]

    ql_chunks = ql.reshape(blocks.shape[0], 2, 64)
    ql_values = torch.stack((ql_chunks & 0x0F, ql_chunks >> 4), dim=2).reshape(
        blocks.shape[0], 8, 32
    )

    qh_chunks = qh.reshape(blocks.shape[0], 2, 32)
    qh_values = torch.stack(
        tuple((qh_chunks >> shift) & 0x03 for shift in (0, 2, 4, 6)),
        dim=2,
    ).reshape(blocks.shape[0], 8, 32)

    values = (ql_values | (qh_values << 4)).to(torch.int8) - 32
    values = values.reshape(blocks.shape[0], _QK_K // 16, 16)
    group_scale = (
        _fp16_column(d, output_dtype) * scales.view(torch.int8).to(output_dtype)
    ).reshape(blocks.shape[0], _QK_K // 16, 1)
    return (group_scale * values.to(output_dtype)).reshape(-1)


def _dequantize_gguf_impl(
    data: torch.Tensor,
    quant_type_code: int,
    output_dtype_code: int,
    layout_code: int,
) -> torch.Tensor:
    """Dequantize supported GGUF blocks with portable PyTorch operations."""
    output_dtype, layout = _validate_codes(
        quant_type_code, output_dtype_code, layout_code
    )
    blocks = _blocks(data, quant_type_code)
    if quant_type_code == 0:
        return _dequantize_q4_0(blocks, output_dtype, layout)
    if quant_type_code == 1:
        return _dequantize_q8_0(blocks, output_dtype)
    if quant_type_code == 2:
        return _dequantize_q4_k(blocks, output_dtype)
    if quant_type_code == 3:
        return _dequantize_q6_k(blocks, output_dtype)
    return _dequantize_q4_1(blocks, output_dtype)


def dequantize_gguf(
    data: torch.Tensor,
    quant_type_code: int,
    output_dtype_code: int,
    layout_code: int,
) -> torch.Tensor:
    output = _dequantize_gguf_impl(
        data, quant_type_code, output_dtype_code, layout_code
    )
    _record_gguf_route("eager")
    return output


@torch.library.custom_op("comfy_kitchen::dequantize_gguf", mutates_args=())
def _op_dequantize_gguf(
    data: torch.Tensor,
    quant_type_code: int,
    output_dtype_code: int,
    layout_code: int,
) -> torch.Tensor:
    kwargs = {
        "data": data,
        "quant_type_code": quant_type_code,
        "output_dtype_code": output_dtype_code,
        "layout_code": layout_code,
    }
    impl = registry.get_implementation("dequantize_gguf", kwargs=kwargs)
    return impl(**kwargs)


@_op_dequantize_gguf.register_fake
def _op_dequantize_gguf_fake(
    data: torch.Tensor,
    quant_type_code: int,
    output_dtype_code: int,
    layout_code: int,
) -> torch.Tensor:
    output_dtype, _ = _validate_codes(
        quant_type_code, output_dtype_code, layout_code
    )
    block_count = data.numel() // GGUF_BLOCK_BYTES[quant_type_code]
    output_elements = block_count * GGUF_BLOCK_ELEMENTS[quant_type_code]
    return torch.empty(output_elements, dtype=output_dtype, device=data.device)


__all__ = ["dequantize_gguf"]
