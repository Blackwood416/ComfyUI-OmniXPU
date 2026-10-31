"""Backend-neutral SVDQuant W4A16 preparation and linear contract."""

from __future__ import annotations

import os
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .backends.eager.quantization import DTYPE_TO_CODE
from .exceptions import BackendError

_GROUP_SIZE = 64
_OUTPUT_DTYPES = frozenset({torch.float16, torch.bfloat16})
_DIAGNOSTICS_LOCK = threading.Lock()
_ROUTE_COUNTS: Counter[str] = Counter()
_FALLBACK_COUNTS: Counter[str] = Counter()
_PROFILE_BOUNDARY_ENV = "COMFY_KITCHEN_PROFILE_BOUNDARIES"


def _profile_operator_boundary_enabled() -> bool:
    """Return whether trace runs require the public torch.library boundary."""
    return os.environ.get(_PROFILE_BOUNDARY_ENV, "").strip() == "1"


@dataclass(frozen=True)
class PreparedSVDQuantW4A16:
    """OneDNN-ready natural-layout SVDQuant weight storage.

    ``packed_u4`` aliases ``source_qweight`` when ``destructive=True``. The
    full packed weight therefore has one storage allocation; only the much
    smaller FP16 scale tensor is converted and retained separately.
    """

    packed_u4: torch.Tensor
    scales_f16: torch.Tensor
    rcp_smooth_f16: torch.Tensor | None
    source_qweight: torch.Tensor
    source_wscales: torch.Tensor
    source_smooth: torch.Tensor | None
    destructive: bool
    group_size: int
    in_features: int
    out_features: int
    xpu_linear_impl: Callable[..., torch.Tensor] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


def _record_svdquant_w4a16_route(
    route: str,
    fallback_reason: str | None = None,
) -> None:
    with _DIAGNOSTICS_LOCK:
        _ROUTE_COUNTS[route] += 1
        if fallback_reason is not None:
            _FALLBACK_COUNTS[fallback_reason] += 1


def get_svdquant_w4a16_route_diagnostics(*, reset: bool = False) -> dict:
    """Return implementations that completed W4A16 calls and fallback reasons."""
    with _DIAGNOSTICS_LOCK:
        snapshot = {
            "routes": dict(sorted(_ROUTE_COUNTS.items())),
            "fallbacks": dict(sorted(_FALLBACK_COUNTS.items())),
        }
        if reset:
            _ROUTE_COUNTS.clear()
            _FALLBACK_COUNTS.clear()
    return snapshot


def _validate_source_storage(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    smooth: torch.Tensor | None,
) -> tuple[int, int]:
    if not isinstance(qweight, torch.Tensor):
        raise TypeError(
            f"qweight must be a torch.Tensor, got {type(qweight).__name__}"
        )
    if not isinstance(wscales, torch.Tensor):
        raise TypeError(
            f"wscales must be a torch.Tensor, got {type(wscales).__name__}"
        )
    if qweight.dim() != 2 or qweight.dtype not in (torch.int8, torch.uint8):
        raise ValueError(
            "qweight must be natural-layout int8/uint8 [N, K//2], "
            f"got shape={tuple(qweight.shape)} dtype={qweight.dtype}"
        )
    if wscales.dim() != 2 or wscales.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError(
            "wscales must be FP16/BF16 [K//64, N], "
            f"got shape={tuple(wscales.shape)} dtype={wscales.dtype}"
        )
    if qweight.device != wscales.device:
        raise ValueError(
            f"qweight and wscales must share a device, got "
            f"{qweight.device} and {wscales.device}"
        )
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * 2
    expected_scales = (in_features // _GROUP_SIZE, out_features)
    if in_features % _GROUP_SIZE or tuple(wscales.shape) != expected_scales:
        raise ValueError(
            f"wscales shape must be {expected_scales} for qweight "
            f"shape {tuple(qweight.shape)}, got {tuple(wscales.shape)}"
        )
    if smooth is not None:
        if (
            not isinstance(smooth, torch.Tensor)
            or smooth.dim() != 1
            or smooth.numel() != in_features
            or smooth.dtype not in (torch.float16, torch.bfloat16)
            or smooth.device != qweight.device
        ):
            raise ValueError(
                "smooth must be FP16/BF16 [K] on the weight device; "
                f"expected K={in_features}"
            )
        if not torch.isfinite(smooth).all().item() or (smooth == 0).any().item():
            raise ValueError("smooth must contain only finite non-zero values")
    return in_features, out_features


def prepare_svdquant_w4a16_for_xpu(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    smooth: torch.Tensor | None = None,
    *,
    destructive: bool = True,
) -> PreparedSVDQuantW4A16:
    """Prepare natural-layout signed INT4 weights for oneDNN W4A16.

    Args:
        qweight: Nunchaku natural-layout signed INT4 bytes, shape ``[N, K//2]``.
        wscales: Per-group scales, shape ``[K//64, N]``.
        smooth: Optional input smooth factor ``[K]``. Its FP16 reciprocal is
            prepared once to preserve the current Nunchaku XPU arithmetic
            boundary.
        destructive: XOR-convert ``qweight`` to unsigned oneDNN storage in
            place. This is the memory-efficient model-load path.

    The destructive form is intentionally one-shot. Call
    :func:`restore_svdquant_w4a16_source_` before serializing or preparing the
    same source tensor again.
    """
    in_features, out_features = _validate_source_storage(
        qweight,
        wscales,
        smooth,
    )
    if destructive and not qweight.is_contiguous():
        raise ValueError("destructive preparation requires contiguous qweight")
    if getattr(qweight, "_comfy_kitchen_w4a16_prepared", False):
        raise RuntimeError(
            "qweight is already destructively prepared; restore it before "
            "preparing again"
        )

    # Allocate all derived small tensors before destructively changing the
    # full weight. If preparation runs out of memory, the model parameter
    # remains in its checkpoint representation and is safe to retry.
    scales_f16 = wscales.to(torch.float16).contiguous()
    rcp_smooth_f16 = (
        (1.0 / smooth.float()).to(torch.float16).contiguous()
        if smooth is not None
        else None
    )

    xpu_linear_impl = None
    if qweight.device.type == "xpu":
        from .registry import registry

        try:
            xpu_linear_impl = registry.get_implementation(
                "svdquant_w4a16_linear",
                backend="xpu",
            )
        except BackendError:
            # Keep preparation portable when the companion wheel is absent or
            # too old. The public call will use normal registry dispatch.
            pass

    if destructive:
        packed_u4 = qweight.view(torch.uint8)
        packed_u4.bitwise_xor_(0x88)
        qweight._comfy_kitchen_w4a16_prepared = True
    else:
        packed_u4 = (qweight.view(torch.uint8) ^ 0x88).contiguous()
    return PreparedSVDQuantW4A16(
        packed_u4=packed_u4,
        scales_f16=scales_f16,
        rcp_smooth_f16=rcp_smooth_f16,
        source_qweight=qweight,
        source_wscales=wscales,
        source_smooth=smooth,
        destructive=destructive,
        group_size=_GROUP_SIZE,
        in_features=in_features,
        out_features=out_features,
        xpu_linear_impl=xpu_linear_impl,
    )


def restore_svdquant_w4a16_source_(
    prepared: PreparedSVDQuantW4A16,
) -> torch.Tensor:
    """Restore a destructively prepared source qweight to signed INT4 bytes."""
    if not isinstance(prepared, PreparedSVDQuantW4A16):
        raise TypeError("prepared must be PreparedSVDQuantW4A16")
    if not prepared.destructive:
        return prepared.source_qweight
    source = prepared.source_qweight
    if not getattr(source, "_comfy_kitchen_w4a16_prepared", False):
        raise RuntimeError("prepared source was already restored")
    source.view(torch.uint8).bitwise_xor_(0x88)
    source._comfy_kitchen_w4a16_prepared = False
    return source


def _validate_linear(
    x: torch.Tensor,
    prepared: PreparedSVDQuantW4A16,
    lora_down: torch.Tensor | None,
    lora_up: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> None:
    if not isinstance(prepared, PreparedSVDQuantW4A16):
        raise TypeError("prepared must be PreparedSVDQuantW4A16")
    if prepared.destructive and not getattr(
        prepared.source_qweight,
        "_comfy_kitchen_w4a16_prepared",
        False,
    ):
        raise RuntimeError(
            "destructively prepared qweight was restored; prepare it again "
            "before running W4A16"
        )
    if x.dim() != 2 or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"x must be FP16/BF16 [M, K], got shape={tuple(x.shape)} "
            f"dtype={x.dtype}"
        )
    if x.shape[1] != prepared.in_features:
        raise ValueError(
            f"x K={x.shape[1]} does not match prepared K={prepared.in_features}"
        )
    tensors = [prepared.packed_u4, prepared.scales_f16]
    if prepared.rcp_smooth_f16 is not None:
        tensors.append(prepared.rcp_smooth_f16)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("x and prepared tensors must share a device")
    if (lora_down is None) != (lora_up is None):
        raise ValueError("lora_down and lora_up must be both present or both absent")
    if lora_down is not None and lora_up is not None:
        rank = lora_down.shape[1] if lora_down.dim() == 2 else -1
        if tuple(lora_down.shape) != (prepared.in_features, rank) or (
            tuple(lora_up.shape) != (prepared.out_features, rank)
        ):
            raise ValueError(
                "LoRA shapes must be [K, R] and [N, R], got "
                f"{tuple(lora_down.shape)} and {tuple(lora_up.shape)}"
            )
        if (
            lora_down.dtype not in _OUTPUT_DTYPES
            or lora_up.dtype not in _OUTPUT_DTYPES
            or lora_down.device != x.device
            or lora_up.device != x.device
        ):
            raise ValueError("LoRA tensors must be FP16/BF16 on the input device")
    if bias is not None and (
        tuple(bias.shape) != (prepared.out_features,)
        or bias.dtype not in _OUTPUT_DTYPES
        or bias.device != x.device
    ):
        raise ValueError("bias must be FP16/BF16 [N] on the input device")
    if output_dtype not in _OUTPUT_DTYPES:
        raise ValueError(f"output_dtype must be FP16/BF16, got {output_dtype}")


def svdquant_w4a16_linear(
    x: torch.Tensor,
    prepared: PreparedSVDQuantW4A16,
    *,
    lora_down: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Run W4A16 main path plus LoRA and bias on raw, unsmoothed input.

    LoRA always consumes ``x`` before smoothing. The main path consumes an FP16
    ``x * reciprocal(smooth)`` intermediate when a smooth factor was prepared.
    This fixes the operator boundary used by the current Nunchaku XPU route.

    Set ``validate=False`` only for a model-owned prepared path whose tensor
    shapes, dtypes, and devices were already fixed and tested. The default
    public contract validates every call.
    """
    selected_dtype = x.dtype if output_dtype is None else output_dtype
    if validate:
        _validate_linear(
            x,
            prepared,
            lora_down,
            lora_up,
            bias,
            selected_dtype,
        )
    op_args = (
        x,
        prepared.packed_u4,
        prepared.scales_f16,
        prepared.rcp_smooth_f16,
        lora_down,
        lora_up,
        bias,
        DTYPE_TO_CODE[selected_dtype],
    )

    # Prepared XPU weights already fixed the backend and validated its native
    # capability once at model load. Bypass torch.library and registry lookup
    # on the per-layer hot path, while preserving explicit use_backend()
    # overrides and runtime backend disablement for testing/recovery.
    if (
        prepared.xpu_linear_impl is not None
        and not _profile_operator_boundary_enabled()
    ):
        from .registry import registry

        if (
            registry.get_backend_override() is None
            and registry.is_available("xpu")
            and x.device.type == "xpu"
            and x.dtype == torch.bfloat16
        ):
            return prepared.xpu_linear_impl(*op_args)

    return torch.ops.comfy_kitchen.svdquant_w4a16_linear(*op_args)


__all__ = [
    "PreparedSVDQuantW4A16",
    "get_svdquant_w4a16_route_diagnostics",
    "prepare_svdquant_w4a16_for_xpu",
    "restore_svdquant_w4a16_source_",
    "svdquant_w4a16_linear",
]
