"""FP8 adapters for Intel XPU."""

import torch
from omni_xpu_kernel import fp8, linear


def quantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Quantize with Kitchen's per-tensor scale convention."""
    return fp8.quantize_per_tensor(x, scale, output_type)


def dequantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 values to a Kitchen standard floating dtype."""
    return fp8.dequantize_per_tensor(x, scale, output_type)


def stochastic_rounding_fp8(
    x: torch.Tensor,
    rng: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Stochastically round using Kitchen's caller-provided uint8 randomness."""
    return fp8.stochastic_rounding(x, rng, output_type)


def fp8_weight_only_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run W8A16 oneDNN linear with Kitchen's scalar weight scale."""
    out_features = weight.shape[0]
    scales = scale.reshape(-1)
    if scales.numel() == 1:
        scales = scales.expand(out_features).contiguous()
    return linear.onednn_w8a16_fp8(
        x.reshape(-1, x.shape[-1]).contiguous(),
        weight.contiguous(),
        scales,
        bias,
    ).reshape(*x.shape[:-1], out_features)


__all__ = [
    "dequantize_per_tensor_fp8",
    "fp8_weight_only_linear",
    "quantize_per_tensor_fp8",
    "stochastic_rounding_fp8",
]
