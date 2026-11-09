"""Decode Kitchen's grouped 4/6-bit storage for the native XPU INT8 linear."""

from __future__ import annotations

import math

import torch

from comfy_kitchen.allocation import allocation_context
from comfy_kitchen.backends._activations import apply_input_act, apply_residual
from comfy_kitchen.backends.eager.quantization import quantize_and_rotate_rowwise
from comfy_kitchen.backends.eager.w4a8_int8 import (
    _unpack_codes,
    _w4a8_geometry,
    validate_w4a8_operands,
)
from comfy_kitchen.constraints import ValidationResult
from comfy_kitchen.tensor.int8_utils import _build_hadamard


def _dequant_int4_grouped_to_int8_xpu(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    codebook: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Decode grouped INT4/INT6 storage to the INT8 grid consumed by the GEMM safely on XPU."""
    n, k, bits = _w4a8_geometry(qdata, s_rel, group_size)
    groups = k // group_size
    quantized = _unpack_codes(qdata, k, bits)
    if codebook is not None:
        # torch 2.15 on Intel DG2 triggers UR_RESULT_ERROR_OUT_OF_RESOURCES / DEVICE_LOST
        # when indexing a 1D tensor with a large 2D index tensor (codebook[quantized]).
        # torch.nn.functional.embedding is fully supported and reliable on XPU.
        cb = codebook.to(device=qdata.device, dtype=torch.float32)
        values = torch.nn.functional.embedding(quantized.long(), cb.unsqueeze(1)).squeeze(-1)
    else:
        values = quantized.float() - float(1 << (bits - 1))
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    return values.view(n, k).round().clamp_(-127, 127).to(torch.int8)


def native_call_rule(kwargs):
    if kwargs.get("correction") is not None:
        return ValidationResult.fail("correction", "asymmetric W4A8 uses the eager route")
    return ValidationResult.ok()


def w4a8_int8_linear(
    x, qdata, s_rel, s_channel, codebook=None, correction=None, bias=None,
    group_size=16, convrot_groupsize=256, out_dtype=torch.bfloat16,
    input_act=None, input_act_weight=None, input_act_eps=0.0,
    residual=None, residual_scale=None,
):
    from . import _GATED_QUANTIZE_AVAILABLE, _int8

    _, k, bits = validate_w4a8_operands(
        qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize,
    )
    # Activations preserve the leading dimensions, including SiTU/SwiGLU's
    # last-dimension split. Validate decode before any activation preparation.
    rows = math.prod(x.shape[:-1])
    packed_decode = bits == 4 and 0 < rows <= 8
    if packed_decode:
        if not qdata.is_contiguous():
            raise ValueError("XPU packed W4A8 decode requires contiguous qdata")
        if not s_rel.is_contiguous():
            raise ValueError("XPU packed W4A8 decode requires contiguous s_rel")
        if s_rel.dtype not in (torch.float32, torch.float8_e4m3fn):
            raise TypeError("XPU packed W4A8 decode requires FP32 or FP8 E4M3 s_rel")
        packed_available = getattr(_int8, "supports_w4a8_linear_prequantized", None)
        if (not callable(packed_available) or not packed_available()
                or not callable(getattr(_int8, "w4a8_linear_prequantized", None))):
            raise RuntimeError(
                "XPU W4A8 decode with 1-8 rows requires a matching omni_xpu_kernel "
                "companion exporting int8.w4a8_linear_prequantized"
            )
    prepared = None
    if (input_act == "situ" and _GATED_QUANTIZE_AVAILABLE and x.device.type == "xpu"
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.shape[-1] == 2 * k and k <= 8192
            and convrot_groupsize in (64, 256) and k % convrot_groupsize == 0):
        from omni_xpu_kernel import kitchen

        prepared = kitchen.gated_quantize_int8(x, input_act, convrot_groupsize)
    if prepared is None:
        x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
        if x.shape[-1] != k:
            raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={k}")
    if correction is not None:
        raise ValueError("asymmetric W4A8 correction requires the eager route")
    if not packed_decode:
        weight = _dequant_int4_grouped_to_int8_xpu(qdata, s_rel, codebook, group_size)
    if prepared is None:
        with allocation_context():
            h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        quantized, scale = quantize_and_rotate_rowwise(x, h, convrot_groupsize)
    else:
        quantized, scale = prepared
    if packed_decode:
        out = _int8.w4a8_linear_prequantized(
            quantized.contiguous(), scale, qdata, s_rel, s_channel.float(),
            codebook=codebook.float().contiguous() if codebook is not None else None,
            bias=bias, group_size=group_size, out_dtype=out_dtype,
        )
    else:
        out = _int8.int8_linear_prequantized(
            quantized, scale, weight, s_channel.float(), bias=bias, out_dtype=out_dtype,
        )
    return apply_residual(out, residual, residual_scale)
