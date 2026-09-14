"""Narrow A770 bridge when the optional Kitchen XPU backend is absent."""

import logging
import os

import torch

from ..patches.debug import trace_patch
from .errors import is_fatal_accelerator_error

log = logging.getLogger("ComfyUI-OmniXPU")


_MEASURED_CONVROT_WEIGHTS = frozenset({
    (21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336),
    (1536, 6144), (6144, 6144), (16384, 6144), (6144, 16384),
})


def _convrot_dequant_contract(q, scale, group_size, output_dtype_code):
    """Measured H3/Krea2 LoRA casts, with no training or broadcast scales."""
    return (
        isinstance(q, torch.Tensor) and isinstance(scale, torch.Tensor)
        and q.device.type == "xpu" and scale.device == q.device
        and q.dtype == torch.int8 and scale.dtype == torch.float32
        and tuple(q.shape) in _MEASURED_CONVROT_WEIGHTS
        and tuple(scale.shape) in ((q.shape[0],), (q.shape[0], 1))
        and group_size == 256 and output_dtype_code == 2
        and q.is_contiguous() and scale.is_contiguous()
        and q.storage_offset() % 16 == 0
        and not torch.is_grad_enabled() and not scale.requires_grad
    )


def _register_convrot_dequant(omni_int8):
    operator = "comfy_kitchen::dequantize_int8_convrot_weight_dtype"
    if os.environ.get("OMNIXPU_CONVROT_DEQUANT", "1") == "0":
        return False
    if not hasattr(omni_int8, "dequantize_int8_convrot_weight_dtype"):
        return False
    if not hasattr(torch.ops.comfy_kitchen, "dequantize_int8_convrot_weight_dtype"):
        return False
    if torch._C._dispatch_has_kernel_for_dispatch_key(operator, "XPU"):
        return False

    from comfy_kitchen.registry import registry

    @torch.library.impl(operator, "XPU")
    def _xpu_convrot(q, scale, group_size, output_dtype_code):
        if _convrot_dequant_contract(q, scale, group_size, output_dtype_code):
            try:
                return omni_int8.dequantize_int8_convrot_weight_dtype(
                    q, scale, group_size, torch.bfloat16)
            except Exception as error:
                if is_fatal_accelerator_error(error):
                    raise
                log.warning("[OmniXPU] ConvRot dequantization fallback: %s", error)
        kwargs = dict(q=q, scale=scale, group_size=group_size,
                      output_dtype_code=output_dtype_code)
        return registry.get_implementation(
            "dequantize_int8_convrot_weight_dtype", kwargs=kwargs)(**kwargs)

    log.info("[OmniXPU] A770: registered H3/Krea2 BF16 ConvRot weight dequantization")
    return True


def apply():
    try:
        import omni_xpu_kernel as omni_package
        from omni_xpu_kernel import int8 as omni_int8
    except ImportError:
        return False, "omni_xpu_kernel.int8 not available"

    if getattr(omni_package, "__xpu_target__", None) != "dg2":
        return False, "A770/DG2-only compatibility route"

    try:
        from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE
    except ImportError:
        return False, "comfy_kitchen not available"

    convrot_registered = _register_convrot_dequant(omni_int8)
    operator = "comfy_kitchen::int8_linear"
    if not hasattr(torch.ops, "comfy_kitchen") or not hasattr(
        torch.ops.comfy_kitchen, "int8_linear"
    ):
        return convrot_registered, f"{operator} not registered"
    if torch._C._dispatch_has_kernel_for_dispatch_key(operator, "XPU"):
        return convrot_registered, "Kitchen XPU backend already owns int8_linear"

    @torch.library.impl(operator, "XPU")
    @trace_patch(
        "int8_linear",
        (
            "x",
            "weight",
            "weight_scale",
            "bias",
            "output_dtype_code",
            "convrot",
            "convrot_groupsize",
            "input_act",
        ),
        details={"backend": "omni_dg2_compat"},
    )
    def _xpu_impl(
        x,
        weight,
        weight_scale,
        bias,
        output_dtype_code,
        convrot=False,
        convrot_groupsize=256,
        input_act=None,
    ):
        return omni_int8.int8_linear(
            x,
            weight,
            weight_scale,
            bias,
            DTYPE_CODE_TO_DTYPE[output_dtype_code],
            convrot,
            convrot_groupsize,
            input_act,
        )

    log.info("[OmniXPU] A770: registered missing %s XPU implementation", operator)
    return True, ""
