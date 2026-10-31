"""SVDQuant W4A4 adapters for Intel XPU via omni_xpu_kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from omni_xpu_kernel import svdq

from comfy_kitchen.backends.eager import svdquant as eager_svdquant

_GROUP_SIZE = 64


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def prepare_svdquant_weights(
    wgt: torch.Tensor,
    wscales: torch.Tensor,
    lora_up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Kitchen tile-packed checkpoint tensors to omni natural layout.

    Callers that own model-loading code may call this once and retain the
    returned tensors. Natural-layout inputs are returned without conversion.
    """
    natural_wgt = eager_svdquant._tile_packed_weight_to_row_major(wgt)
    natural_scales = eager_svdquant._tile_packed_scales_to_natural(wscales)
    natural_lora_up = eager_svdquant._tile_packed_lora_up_to_natural(lora_up)
    return (
        natural_wgt.contiguous(),
        natural_scales.contiguous(),
        natural_lora_up.contiguous(),
    )


def quantize_svdquant_w4a4(
    x: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor,
    pad_size: int = 256,
    act_unsigned: bool = False,
    lora_x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize signed or unsigned activations with omni ESIMD kernels."""
    if act_unsigned and not hasattr(svdq, "quantize_act_uint4"):
        return eager_svdquant.quantize_svdquant_w4a4(
            x, smooth, lora_down, pad_size, act_unsigned, lora_x
        )

    if x.dim() != 2:
        raise ValueError(f"expected 2D input, got shape {tuple(x.shape)}")
    m, k = x.shape
    if k % _GROUP_SIZE:
        raise ValueError(f"K={k} not divisible by group_size={_GROUP_SIZE}")

    lora_src = lora_x if lora_x is not None else x
    lora_act = lora_src.float() @ lora_down.float()
    quantize = svdq.quantize_act_uint4 if act_unsigned else svdq.quantize_act_int4
    q_x, ascales = quantize((x / smooth).contiguous(), _GROUP_SIZE)

    m_pad = _ceil_div(m, pad_size) * pad_size
    if m_pad > m:
        pad = m_pad - m
        q_x = F.pad(q_x, (0, 0, 0, pad))
        ascales = F.pad(ascales, (0, pad))
        lora_act = F.pad(lora_act, (0, 0, 0, pad))

    # Kitchen stores packed nibbles in int8; omni uses uint8 for the same bits.
    return q_x.view(torch.int8), ascales, lora_act


def scaled_mm_svdquant_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
) -> torch.Tensor:
    """Run Kitchen-equivalent SVDQuant using omni dequant and oneDNN GEMM."""
    if act_unsigned and not hasattr(svdq, "dequantize_u4"):
        return eager_svdquant.scaled_mm_svdquant_w4a4(
            act, wgt, ascales, wscales, lora_act_in, lora_up, bias, act_unsigned
        )

    wgt, wscales, lora_up = prepare_svdquant_weights(wgt, wscales, lora_up)
    compute_dtype = wscales.dtype

    dequantize = svdq.dequantize_u4 if act_unsigned else svdq.dequantize_w4
    act_fp = dequantize(act.view(torch.uint8), ascales, compute_dtype)
    out = svdq.onednn_int4_gemm(
        act_fp,
        wgt.view(torch.uint8),
        wscales,
    )
    lora = lora_act_in.float() @ lora_up.float().t()
    out = out + lora.to(out.dtype)
    if bias is not None:
        out = out + bias
    return out


def scaled_mm_svdquant_w4a4_preconverted(
    act: torch.Tensor,
    packed_u4: torch.Tensor,
    ascales: torch.Tensor,
    scales_f16: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run SVDQuant with destructively prepared, single-copy XPU weights."""
    act_fp = svdq.dequantize_w4(act.view(torch.uint8), ascales, compute_dtype)
    out = svdq.onednn_int4_gemm_preconverted(
        act_fp,
        packed_u4.view(torch.uint8),
        scales_f16,
    )
    lora = lora_act_in.float() @ lora_up.float().t()
    out = out + lora.to(out.dtype)
    if bias is not None:
        out = out + bias
    return out


__all__ = [
    "prepare_svdquant_weights",
    "quantize_svdquant_w4a4",
    "scaled_mm_svdquant_w4a4",
    "scaled_mm_svdquant_w4a4_preconverted",
]
