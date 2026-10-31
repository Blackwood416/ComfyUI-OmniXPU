"""SVDQuant W4A16 adapter for Omni XPU native kernels."""

from __future__ import annotations

import threading
import warnings

import torch
from omni_xpu_kernel import svdq

from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE
from comfy_kitchen.backends.eager.svdquant_w4a16 import (
    _svdquant_w4a16_linear_impl as _eager_svdquant_w4a16_linear,
)
from comfy_kitchen.svdquant_w4a16 import _record_svdquant_w4a16_route

_QUARANTINE_LOCK = threading.Lock()
_QUARANTINED_ROUTES: dict[tuple, str] = {}


def _route_key(
    x: torch.Tensor,
    packed_u4: torch.Tensor,
    rcp_smooth_f16: torch.Tensor | None,
    lora_down: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype_code: int,
) -> tuple:
    return (
        x.device.type,
        x.device.index,
        tuple(x.shape),
        tuple(packed_u4.shape),
        x.dtype,
        rcp_smooth_f16 is not None,
        lora_down is not None,
        bias is not None,
        output_dtype_code,
    )


def _call_native(
    x: torch.Tensor,
    packed_u4: torch.Tensor,
    scales_f16: torch.Tensor,
    rcp_smooth_f16: torch.Tensor | None,
    lora_down: torch.Tensor | None,
    lora_up: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype_code: int,
) -> torch.Tensor:
    x = x.contiguous()
    if rcp_smooth_f16 is None:
        x_gemm = x.to(torch.float16)
    else:
        x_gemm = svdq.fused_smooth_mul_convert(
            x,
            rcp_smooth_f16,
        )
        x_gemm.nan_to_num_(
            nan=0.0,
            posinf=65504.0,
            neginf=-65504.0,
        )

    has_lora = lora_down is not None and lora_up is not None
    if has_lora or bias is not None:
        dst = torch.zeros(
            x.shape[0],
            packed_u4.shape[0],
            dtype=torch.bfloat16,
            device=x.device,
        )
        if has_lora:
            lora = (
                x.to(torch.bfloat16) @ lora_down.to(torch.bfloat16)
            ) @ lora_up.to(torch.bfloat16).t()
            dst.add_(lora)
        if bias is not None:
            dst.add_(bias.to(torch.bfloat16))
        svdq.onednn_int4_gemm_add_to_output(
            x_gemm,
            packed_u4,
            scales_f16,
            dst,
        )
        output = dst
    else:
        output = svdq.onednn_int4_gemm_preconverted(
            x_gemm,
            packed_u4,
            scales_f16,
        )
    return output.to(DTYPE_CODE_TO_DTYPE[output_dtype_code])


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
    """Run native W4A16; quarantine a failed retry-safe route."""
    route_key = _route_key(
        x,
        packed_u4,
        rcp_smooth_f16,
        lora_down,
        bias,
        output_dtype_code,
    )
    with _QUARANTINE_LOCK:
        quarantined_reason = _QUARANTINED_ROUTES.get(route_key)
    if quarantined_reason is not None:
        _record_svdquant_w4a16_route(
            "eager",
            fallback_reason=f"quarantined after {quarantined_reason}",
        )
        return _eager_svdquant_w4a16_linear(
            x,
            packed_u4,
            scales_f16,
            rcp_smooth_f16,
            lora_down,
            lora_up,
            bias,
            output_dtype_code,
        )

    try:
        output = _call_native(
            x,
            packed_u4,
            scales_f16,
            rcp_smooth_f16,
            lora_down,
            lora_up,
            bias,
            output_dtype_code,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        with _QUARANTINE_LOCK:
            first_failure = route_key not in _QUARANTINED_ROUTES
            _QUARANTINED_ROUTES[route_key] = reason
        if first_failure:
            warnings.warn(
                "Comfy Kitchen SVDQuant W4A16 XPU route failed and was "
                f"quarantined; retrying with eager: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
        _record_svdquant_w4a16_route("eager", fallback_reason=reason)
        return _eager_svdquant_w4a16_linear(
            x,
            packed_u4,
            scales_f16,
            rcp_smooth_f16,
            lora_down,
            lora_up,
            bias,
            output_dtype_code,
        )

    _record_svdquant_w4a16_route("xpu")
    return output


def _reset_svdquant_w4a16_quarantine_for_tests() -> None:
    with _QUARANTINE_LOCK:
        _QUARANTINED_ROUTES.clear()


__all__ = ["svdquant_w4a16_linear"]
