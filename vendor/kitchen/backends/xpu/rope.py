"""Kitchen-compatible arbitrary-matrix RoPE adapters for Intel XPU."""

import torch
from omni_xpu_kernel import rotary

from comfy_kitchen._rope_utils import check_rope_inplace


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return rotary.apply_kitchen_rope1(x, freqs_cis)


def apply_rope1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    return rotary.apply_kitchen_rope1_(x, freqs_cis)


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return rotary.apply_kitchen_rope(xq, xk, freqs_cis)


def apply_rope_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    return rotary.apply_kitchen_rope_(xq, xk, freqs_cis)


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return rotary.apply_kitchen_rope_split_half1(x, freqs_cis)


def apply_rope_split_half1_(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    return rotary.apply_kitchen_rope_split_half1_(x, freqs_cis)


def apply_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return rotary.apply_kitchen_rope_split_half(xq, xk, freqs_cis)


def apply_rope_split_half_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    return rotary.apply_kitchen_rope_split_half_(xq, xk, freqs_cis)


def rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return rotary.rms_kitchen_rope1(x, freqs_cis, scale, epsilon)


def rms_rope1_(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    return rotary.rms_kitchen_rope1_(x, freqs_cis, scale, epsilon)


def rms_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return rotary.rms_kitchen_rope(
        q, k, freqs_cis, q_scale, k_scale, epsilon
    )


def rms_rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    return rotary.rms_kitchen_rope_(
        q, k, freqs_cis, q_scale, k_scale, epsilon
    )


def rms_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return rotary.rms_kitchen_rope_split_half1(
        x, freqs_cis, scale, epsilon
    )


def rms_rope_split_half1_(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    return rotary.rms_kitchen_rope_split_half1_(
        x, freqs_cis, scale, epsilon
    )


def rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return rotary.rms_kitchen_rope_split_half(
        q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim
    )


def rms_rope_split_half_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    return rotary.rms_kitchen_rope_split_half_(
        q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim
    )


__all__ = [
    "apply_rope",
    "apply_rope_",
    "apply_rope1",
    "apply_rope1_",
    "apply_rope_split_half",
    "apply_rope_split_half_",
    "apply_rope_split_half1",
    "apply_rope_split_half1_",
    "rms_rope",
    "rms_rope_",
    "rms_rope1",
    "rms_rope1_",
    "rms_rope_split_half",
    "rms_rope_split_half_",
    "rms_rope_split_half1",
    "rms_rope_split_half1_",
]
