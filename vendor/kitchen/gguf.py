"""Backend-neutral GGUF dequantization contract."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Final

import torch

GGUF_QUANT_TYPE_TO_CODE: Final = {
    "q4_0": 0,
    "q8_0": 1,
    "q4_k": 2,
    "q6_k": 3,
    "q4_1": 4,
}
GGUF_QUANT_CODE_TO_TYPE: Final = {
    code: quant_type for quant_type, code in GGUF_QUANT_TYPE_TO_CODE.items()
}

GGUF_LAYOUT_TO_CODE: Final = {
    "comfyui": 0,
    "interleaved": 1,
}
GGUF_LAYOUT_CODE_TO_NAME: Final = {
    code: layout for layout, code in GGUF_LAYOUT_TO_CODE.items()
}

GGUF_BLOCK_BYTES: Final = {
    0: 18,
    1: 34,
    2: 144,
    3: 210,
    4: 20,
}
GGUF_BLOCK_ELEMENTS: Final = {
    0: 32,
    1: 32,
    2: 256,
    3: 256,
    4: 32,
}
GGUF_OUTPUT_DTYPES: Final = frozenset({torch.float16, torch.bfloat16})
GGUF_OUTPUT_DTYPE_TO_CODE: Final = {
    torch.float16: 1,
    torch.bfloat16: 2,
}

_DIAGNOSTICS_LOCK = threading.Lock()
_ROUTE_COUNTS: Counter[str] = Counter()
_FALLBACK_COUNTS: Counter[str] = Counter()


def _record_gguf_route(route: str, fallback_reason: str | None = None) -> None:
    with _DIAGNOSTICS_LOCK:
        _ROUTE_COUNTS[route] += 1
        if fallback_reason is not None:
            _FALLBACK_COUNTS[fallback_reason] += 1


def get_gguf_route_diagnostics(*, reset: bool = False) -> dict:
    """Return routes that actually completed GGUF dequantization calls.

    ``routes`` counts completed implementations, not startup capability guesses.
    When an XPU attempt fails or is quarantined, the completed route is eager and
    ``fallbacks`` records the native failure reason.
    """
    with _DIAGNOSTICS_LOCK:
        snapshot = {
            "routes": dict(sorted(_ROUTE_COUNTS.items())),
            "fallbacks": dict(sorted(_FALLBACK_COUNTS.items())),
        }
        if reset:
            _ROUTE_COUNTS.clear()
            _FALLBACK_COUNTS.clear()
    return snapshot


def _normalize_name(value: str, *, kind: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{kind} must be a string, got {type(value).__name__}")
    return value.strip().lower()


def _validate_dequantize_gguf(
    data: torch.Tensor,
    quant_type: str,
    output_dtype: torch.dtype,
    layout: str,
) -> tuple[int, int, int]:
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"data must be a torch.Tensor, got {type(data).__name__}")
    if data.dtype != torch.uint8:
        raise TypeError(f"data must have dtype torch.uint8, got {data.dtype}")

    quant_type = _normalize_name(quant_type, kind="quant_type")
    if quant_type not in GGUF_QUANT_TYPE_TO_CODE:
        supported = ", ".join(sorted(GGUF_QUANT_TYPE_TO_CODE))
        raise ValueError(f"unsupported GGUF quant_type {quant_type!r}; expected one of {supported}")
    quant_type_code = GGUF_QUANT_TYPE_TO_CODE[quant_type]

    if output_dtype not in GGUF_OUTPUT_DTYPES:
        supported = ", ".join(str(dtype) for dtype in sorted(GGUF_OUTPUT_DTYPES, key=str))
        raise ValueError(
            f"unsupported GGUF output_dtype {output_dtype}; expected one of {supported}"
        )
    output_dtype_code = GGUF_OUTPUT_DTYPE_TO_CODE[output_dtype]

    layout = _normalize_name(layout, kind="layout")
    if layout not in GGUF_LAYOUT_TO_CODE:
        supported = ", ".join(sorted(GGUF_LAYOUT_TO_CODE))
        raise ValueError(f"unsupported GGUF layout {layout!r}; expected one of {supported}")
    layout_code = GGUF_LAYOUT_TO_CODE[layout]
    if quant_type != "q4_0" and layout != "comfyui":
        raise ValueError(
            f"layout={layout!r} is only defined for quant_type='q4_0'; "
            f"{quant_type!r} requires layout='comfyui'"
        )

    block_bytes = GGUF_BLOCK_BYTES[quant_type_code]
    if data.numel() % block_bytes:
        raise ValueError(
            f"{quant_type} data has {data.numel()} bytes; expected a multiple of {block_bytes}"
        )
    return quant_type_code, output_dtype_code, layout_code


def dequantize_gguf(
    data: torch.Tensor,
    quant_type: str,
    *,
    output_dtype: torch.dtype = torch.float16,
    layout: str = "comfyui",
) -> torch.Tensor:
    """Dequantize one packed GGUF tensor through the selected Kitchen backend.

    The result is flat. Callers such as ComfyUI-GGUF retain ownership of the
    logical tensor shape and reshape the result after dequantization.

    Args:
        data: Packed GGUF storage. Any shape is accepted; storage is flattened
            in logical order and made contiguous before backend dispatch.
        quant_type: One of ``q4_0``, ``q4_1``, ``q8_0``, ``q4_k`` or
            ``q6_k``.
        output_dtype: ``torch.float16`` or ``torch.bfloat16``.
        layout: ``comfyui`` for the plugin's canonical layout. Q4_0 also accepts
            ``interleaved`` for Omni's packed-byte low/high ordering.
    """
    quant_type_code, output_dtype_code, layout_code = _validate_dequantize_gguf(
        data, quant_type, output_dtype, layout
    )
    return torch.ops.comfy_kitchen.dequantize_gguf(
        data.reshape(-1).contiguous(),
        quant_type_code,
        output_dtype_code,
        layout_code,
    )


__all__ = [
    "GGUF_BLOCK_BYTES",
    "GGUF_BLOCK_ELEMENTS",
    "GGUF_LAYOUT_CODE_TO_NAME",
    "GGUF_LAYOUT_TO_CODE",
    "GGUF_OUTPUT_DTYPE_TO_CODE",
    "GGUF_OUTPUT_DTYPES",
    "GGUF_QUANT_CODE_TO_TYPE",
    "GGUF_QUANT_TYPE_TO_CODE",
    "dequantize_gguf",
    "get_gguf_route_diagnostics",
]
