"""A770: keep offloaded TensorWise INT8 weights on the quantized path.

ComfyUI's vbar/AIMDO cast path dequantizes a TensorWise INT8 weight whenever
the storage dtype differs from the compute dtype and no lowvram requant hook
is present.  That sends long-sequence H3 MLP/attention projections through
``dequantize + bf16 F.linear`` and hides the real GEMM cost behind page-in and
cast work.  This adapter intercepts ``comfy.ops.cast_bias_weight`` for
offloaded TensorWise INT8 modules and returns a device-resident
``QuantizedTensor`` backed by the raw qdata/scale, so the normal
``comfy_kitchen::int8_linear`` path can run without first materializing a
bf16 weight.

The route is A770/DG2-specific and opt-in until it is validated end to end:
set ``OMNIXPU_INT8_DIRECT_CAST=1`` to enable.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import torch

from ..patches.debug import log_debug_event

log = logging.getLogger("ComfyUI-OmniXPU")

_MARKER = "__omnixpu_int8_direct_cast__"


def _module_eligible(s: Any, input: torch.Tensor) -> tuple[bool, str]:
    if not isinstance(input, torch.Tensor) or not input.is_xpu:
        return False, "input_not_xpu"
    if comfy_model_management.in_training:
        return False, "training"

    weight = getattr(s, "weight", None)
    if weight is None:
        return False, "missing_weight"
    if not isinstance(weight, torch.Tensor) or getattr(
        weight, "_layout_cls", None
    ) != "TensorWiseINT8Layout":
        return False, "not_tensorwise_int8"
    if getattr(weight._params, "transposed", False):
        return False, "transposed_weight"
    if weight.device == input.device:
        return False, "already_resident"
    if not hasattr(s, "_v"):
        return False, "not_vbar"

    if (
        getattr(s, "weight_function", None)
        or getattr(s, "bias_function", None)
        or getattr(s, "weight_lowvram_function", None)
        or getattr(s, "bias_lowvram_function", None)
    ):
        return False, "patched_weight"

    qdata = getattr(weight, "_qdata", None)
    scale = getattr(getattr(weight, "_params", None), "scale", None)
    if (
        not isinstance(qdata, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or qdata.dtype != torch.int8
        or qdata.ndim != 2
    ):
        return False, "weight_storage"
    if qdata.device.type in ("meta",) or scale.device.type in ("meta",):
        return False, "meta_weight"
    if (
        input.shape[-1] != qdata.shape[1]
        and input.shape[-1] != qdata.shape[1] * 2
    ):
        return False, "shape_mismatch"
    return True, ""


def apply():
    try:
        import omni_xpu_kernel as omni_package
    except ImportError:
        return False, "omni_xpu_kernel.int8 not available"
    if getattr(omni_package, "__xpu_target__", None) != "dg2":
        return False, "A770/DG2-only route"

    try:
        import comfy.model_management as comfy_model_management
        import comfy.ops as comfy_ops
        from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout
    except ImportError as exc:
        return False, f"ComfyUI imports unavailable ({exc})"

    global comfy_model_management
    comfy_model_management = comfy_model_management

    original_cast = comfy_ops.cast_bias_weight
    if getattr(original_cast, _MARKER, False):
        return True, ""

    def _patched_cast_bias_weight(s, input=None, dtype=None, device=None,
                                  bias_dtype=None, offloadable=False,
                                  compute_dtype=None, want_requant=False):
        eligible, reason = _module_eligible(s, input)
        if not eligible:
            log_debug_event(
                "dispatch",
                "int8_direct_cast",
                {"input": input} if isinstance(input, torch.Tensor) else {},
                details={"hit": False, "reason": reason},
                verbose_only=True,
            )
            return original_cast(
                s,
                input=input,
                dtype=dtype,
                device=device,
                bias_dtype=bias_dtype,
                offloadable=offloadable,
                compute_dtype=compute_dtype,
                want_requant=want_requant,
            )

        weight = s.weight
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        target = input.device
        qdata = qdata.to(device=target, non_blocking=True).contiguous()
        scale = scale.to(device=target, non_blocking=True)
        params = dataclasses.replace(weight._params, scale=scale)
        qt = QuantizedTensor(qdata, weight._layout_cls, params)

        bias = None
        if s.bias is not None:
            bias = s.bias.to(
                device=target,
                dtype=bias_dtype if bias_dtype is not None else dtype,
                non_blocking=True,
            )

        log_debug_event(
            "dispatch",
            "int8_direct_cast",
            {"input": input, "weight": qdata, "weight_scale": scale, "bias": bias},
            details={
                "hit": True,
                "reason": reason,
                "qdata_bytes": qdata.numel() * qdata.element_size(),
            },
            verbose_only=True,
        )
        # offload_stream is intentionally None: we did not touch the vbar
        # pin, so uncast_bias_weight has nothing to wait for or unpin.
        return qt, bias, (None, None, None)

    setattr(_patched_cast_bias_weight, _MARKER, True)
    comfy_ops.cast_bias_weight = _patched_cast_bias_weight
    log.info("[OmniXPU] A770: INT8 direct cast enabled for offloaded TensorWise weights")
    return True, ""
