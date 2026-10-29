"""GGUF adapter for the optional Omni XPU native backend."""

from __future__ import annotations

import threading
import warnings

import torch
from omni_xpu_kernel import gguf as _omni_gguf

from comfy_kitchen.backends.eager.gguf import (
    _dequantize_gguf_impl as _eager_dequantize_gguf,
)
from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE
from comfy_kitchen.gguf import (
    GGUF_LAYOUT_CODE_TO_NAME,
    GGUF_QUANT_CODE_TO_TYPE,
    _record_gguf_route,
)

_QUARANTINE_LOCK = threading.Lock()
_QUARANTINED_ROUTES: dict[tuple, str] = {}


def _route_key(
    data: torch.Tensor,
    quant_type_code: int,
    output_dtype_code: int,
    layout_code: int,
) -> tuple:
    return (
        data.device.type,
        data.device.index,
        data.numel(),
        quant_type_code,
        output_dtype_code,
        layout_code,
    )


def _call_native(
    data: torch.Tensor,
    quant_type_code: int,
    output_dtype_code: int,
    layout_code: int,
) -> torch.Tensor:
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    quant_type = GGUF_QUANT_CODE_TO_TYPE[quant_type_code]
    layout = GGUF_LAYOUT_CODE_TO_NAME[layout_code]
    if quant_type == "q4_0":
        if layout == "comfyui":
            return _omni_gguf.dequantize_q4_0_comfyui(data, output_dtype)
        return _omni_gguf.dequantize_q4_0(data, output_dtype)
    if layout != "comfyui":
        raise ValueError(
            f"layout={layout!r} is only defined for q4_0; "
            f"{quant_type} requires layout='comfyui'"
        )
    return getattr(_omni_gguf, f"dequantize_{quant_type}")(data, output_dtype)


def dequantize_gguf(
    data: torch.Tensor,
    quant_type_code: int,
    output_dtype_code: int,
    layout_code: int,
) -> torch.Tensor:
    """Run Omni when healthy; retry safely through eager after a native failure."""
    route_key = _route_key(
        data, quant_type_code, output_dtype_code, layout_code
    )
    with _QUARANTINE_LOCK:
        quarantined_reason = _QUARANTINED_ROUTES.get(route_key)
    if quarantined_reason is not None:
        _record_gguf_route(
            "eager",
            fallback_reason=f"quarantined after {quarantined_reason}",
        )
        return _eager_dequantize_gguf(
            data, quant_type_code, output_dtype_code, layout_code
        )

    try:
        output = _call_native(
            data, quant_type_code, output_dtype_code, layout_code
        )
    except (ImportError, OSError, RuntimeError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        with _QUARANTINE_LOCK:
            first_failure = route_key not in _QUARANTINED_ROUTES
            _QUARANTINED_ROUTES[route_key] = reason
        if first_failure:
            warnings.warn(
                "Comfy Kitchen GGUF XPU route failed and was quarantined for "
                f"this process; retrying with eager: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
        _record_gguf_route("eager", fallback_reason=reason)
        return _eager_dequantize_gguf(
            data, quant_type_code, output_dtype_code, layout_code
        )

    _record_gguf_route("xpu")
    return output


def _reset_gguf_quarantine_for_tests() -> None:
    with _QUARANTINE_LOCK:
        _QUARANTINED_ROUTES.clear()


__all__ = ["dequantize_gguf"]
