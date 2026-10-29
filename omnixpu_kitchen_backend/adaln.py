"""Adaptive LayerNorm adapter using omni's native XPU LayerNorm."""

import torch
from omni_xpu_kernel import norm

from comfy_kitchen.backends._modulation import adaln_prep_modulation
from comfy_kitchen.backends.eager.adaln import adaln as eager_adaln
from comfy_kitchen.backends.eager.adaln import rms_adaln as eager_rms_adaln


def adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize with ESIMD when supported, preserving Kitchen broadcasting."""
    hidden = x.shape[-1]
    if hidden % 32 or hidden > 8192 or hidden == 0:
        return eager_adaln(x, scale, shift, eps)

    shape = x.shape
    x_2d = x.reshape(-1, hidden).contiguous()
    mapping = _modulation_mapping(x, scale, shift)
    native = norm._get_native()
    if mapping is not None and hasattr(native, "fused_adaln"):
        scale_2d, shift_2d, row_repeat = mapping
        return norm.fused_adaln(x_2d, scale_2d, shift_2d, row_repeat, eps).reshape(shape)
    normalized = norm.layer_norm(x_2d, None, None, eps).reshape(shape)
    return (normalized * (1 + scale) + shift).contiguous()


def rms_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm plus AdaLN modulation with native common-layout support."""
    hidden = x.shape[-1]
    if hidden % 32 or hidden > 8192 or hidden == 0:
        return eager_rms_adaln(x, scale, shift, eps)

    shape = x.shape
    x_2d = x.reshape(-1, hidden).contiguous()
    mapping = _modulation_mapping(x, scale, shift)
    native = norm._get_native()
    if mapping is not None and hasattr(native, "fused_rms_adaln"):
        scale_2d, shift_2d, row_repeat = mapping
        return norm.fused_rms_adaln(
            x_2d, scale_2d, shift_2d, row_repeat, eps
        ).reshape(shape)
    return eager_rms_adaln(x, scale, shift, eps)


def _modulation_mapping(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Apply Kitchen's shared broadcast contract to the native row mapping."""
    if scale.dtype != x.dtype or shift.dtype != x.dtype:
        return None
    if scale.device != x.device or shift.device != x.device:
        return None
    if scale.shape != shift.shape or scale.dim() > x.dim() or scale.dim() == 0:
        return None
    rows, hidden = x.numel() // x.shape[-1], x.shape[-1]
    try:
        scale_2d, scale_repeat = adaln_prep_modulation(
            scale, x, rows, hidden
        )
        shift_2d, shift_repeat = adaln_prep_modulation(
            shift, x, rows, hidden
        )
    except RuntimeError:
        return None
    if scale_repeat != shift_repeat:
        return None
    return scale_2d, shift_2d, scale_repeat


__all__ = ["adaln", "rms_adaln"]
