"""Portable eager reference for SVDQuant W4A16 linear."""

from __future__ import annotations

import torch

from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE
from comfy_kitchen.registry import registry
from comfy_kitchen.svdquant_w4a16 import _record_svdquant_w4a16_route


def _unpack_prepared_weight(packed_u4: torch.Tensor) -> torch.Tensor:
    signed = packed_u4.view(torch.uint8) ^ 0x88
    low = (signed & 0x0F).to(torch.int16)
    high = ((signed >> 4) & 0x0F).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack((low, high), dim=-1).reshape(
        signed.shape[0],
        signed.shape[1] * 2,
    )


def _svdquant_w4a16_linear_impl(
    x: torch.Tensor,
    packed_u4: torch.Tensor,
    scales_f16: torch.Tensor,
    rcp_smooth_f16: torch.Tensor | None,
    lora_down: torch.Tensor | None,
    lora_up: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype_code: int,
) -> torch.Tensor:
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    m, k = x.shape
    n = packed_u4.shape[0]
    groups = scales_f16.shape[0]
    group_size = k // groups

    if rcp_smooth_f16 is None:
        x_gemm = x.to(torch.float16)
    else:
        x_gemm = (
            x.float() * rcp_smooth_f16.float()
        ).to(torch.float16)
        x_gemm = torch.nan_to_num(
            x_gemm,
            nan=0.0,
            posinf=65504.0,
            neginf=-65504.0,
        )

    weight_int = _unpack_prepared_weight(packed_u4).float()
    weight = (
        weight_int.reshape(n, groups, group_size)
        * scales_f16.float().t().unsqueeze(-1)
    ).reshape(n, k)
    main = x_gemm.float() @ weight.t()

    has_lora = lora_down is not None and lora_up is not None
    if has_lora or bias is not None:
        dst = torch.zeros(m, n, dtype=torch.bfloat16, device=x.device)
        if has_lora:
            lora = (
                x.to(torch.bfloat16) @ lora_down.to(torch.bfloat16)
            ) @ lora_up.to(torch.bfloat16).t()
            dst.add_(lora)
        if bias is not None:
            dst.add_(bias.to(torch.bfloat16))
        result = (dst.float() + main).to(torch.bfloat16)
    else:
        result = main.to(torch.float16)
    return result.to(output_dtype)


def svdquant_w4a16_linear(
    x: torch.Tensor,
    packed_u4: torch.Tensor,
    scales_f16: torch.Tensor,
    rcp_smooth_f16: torch.Tensor | None,
    lora_down: torch.Tensor | None,
    lora_up: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype_code: int,
) -> torch.Tensor:
    output = _svdquant_w4a16_linear_impl(
        x,
        packed_u4,
        scales_f16,
        rcp_smooth_f16,
        lora_down,
        lora_up,
        bias,
        output_dtype_code,
    )
    _record_svdquant_w4a16_route("eager")
    return output


@torch.library.custom_op(
    "comfy_kitchen::svdquant_w4a16_linear",
    mutates_args=(),
)
def _op_svdquant_w4a16_linear(
    x: torch.Tensor,
    packed_u4: torch.Tensor,
    scales_f16: torch.Tensor,
    rcp_smooth_f16: torch.Tensor | None,
    lora_down: torch.Tensor | None,
    lora_up: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype_code: int,
) -> torch.Tensor:
    kwargs = {
        "x": x,
        "packed_u4": packed_u4,
        "scales_f16": scales_f16,
        "rcp_smooth_f16": rcp_smooth_f16,
        "lora_down": lora_down,
        "lora_up": lora_up,
        "bias": bias,
        "output_dtype_code": output_dtype_code,
    }
    implementation = registry.get_implementation(
        "svdquant_w4a16_linear",
        kwargs=kwargs,
    )
    return implementation(**kwargs)


@_op_svdquant_w4a16_linear.register_fake
def _op_svdquant_w4a16_linear_fake(
    x,
    packed_u4,
    scales_f16,
    rcp_smooth_f16,
    lora_down,
    lora_up,
    bias,
    output_dtype_code,
):
    del scales_f16, rcp_smooth_f16, lora_down, lora_up, bias
    return torch.empty(
        x.shape[0],
        packed_u4.shape[0],
        dtype=DTYPE_CODE_TO_DTYPE[output_dtype_code],
        device=x.device,
    )


__all__ = ["svdquant_w4a16_linear"]
