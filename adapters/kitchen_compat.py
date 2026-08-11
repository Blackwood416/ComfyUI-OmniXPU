"""Narrow A770 bridge when the optional Kitchen XPU backend is absent."""

import logging

import torch

from ..patches.debug import trace_patch

log = logging.getLogger("ComfyUI-OmniXPU")


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

    operator = "comfy_kitchen::int8_linear"
    if not hasattr(torch.ops, "comfy_kitchen") or not hasattr(
        torch.ops.comfy_kitchen, "int8_linear"
    ):
        return False, f"{operator} not registered"
    if torch._C._dispatch_has_kernel_for_dispatch_key(operator, "XPU"):
        return False, "Kitchen XPU backend already owns int8_linear"

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
