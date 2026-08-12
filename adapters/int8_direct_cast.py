"""A770: keep offloaded TensorWise INT8 weights on the quantized path.

ComfyUI's offload cast path (vbar/AIMDO or regular lowvram) dequantizes a
TensorWise INT8 weight whenever
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
import os
from typing import Any

import torch

from ..patches.debug import log_debug_event

log = logging.getLogger("ComfyUI-OmniXPU")

_MARKER = "__omnixpu_int8_direct_cast__"
_QuantizedTensor = None
_TensorWiseINT8Layout = None
_comfy_ops = None
_original_cast = None


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

    if getattr(s, "weight_function", None):
        return False, "weight_function"
    if getattr(s, "bias_function", None):
        return False, "bias_function"
    if getattr(s, "weight_lowvram_function", None):
        return False, "weight_lowvram"
    if getattr(s, "bias_lowvram_function", None):
        return False, "bias_lowvram"

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
    global _QuantizedTensor, _TensorWiseINT8Layout, _comfy_ops, _original_cast

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

    _QuantizedTensor = QuantizedTensor
    _TensorWiseINT8Layout = TensorWiseINT8Layout
    _comfy_ops = comfy_ops

    original_cast = comfy_ops.cast_bias_weight
    _original_cast = original_cast
    if getattr(original_cast, _MARKER, False):
        return True, ""

    setattr(_patched_cast_bias_weight, _MARKER, True)
    comfy_ops.cast_bias_weight = _patched_cast_bias_weight
    log.info("[OmniXPU] A770: INT8 direct cast enabled for offloaded TensorWise weights")
    return True, ""


def _direct_cast_bias_weight(s, input, dtype, bias_dtype):
    weight = s.weight
    qdata, scale = _TensorWiseINT8Layout.get_plain_tensors(weight)
    target = input.device
    qdata = qdata.to(device=target, non_blocking=True).contiguous()
    scale = scale.to(device=target, non_blocking=True)
    params = dataclasses.replace(weight._params, scale=scale)
    qt = _QuantizedTensor(qdata, weight._layout_cls, params)

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
            "reason": "raw_qdata",
            "qdata_bytes": qdata.numel() * qdata.element_size(),
        },
        verbose_only=True,
    )
    # offload_stream is intentionally None: we did not touch the vbar
    # pin, so uncast_bias_weight has nothing to wait for or unpin.
    return qt, bias, (None, None, None)


def _cached_patched_weight(s, input, dtype, bias_dtype):
    """Return a cached patched bf16 weight (no per-step LoRA recompute)."""
    cache = getattr(s, "_omnixpu_patched_cache", None)
    if cache is not None:
        weight_cpu, bias_cpu = cache
        weight = weight_cpu.to(
            input.device,
            dtype=dtype,
            non_blocking=True,
        )
        bias = None
        if bias_cpu is not None:
            bias = bias_cpu.to(
                input.device,
                dtype=bias_dtype if bias_dtype is not None else dtype,
                non_blocking=True,
            )
        return weight, bias, (None, None, None)

    weight, bias, offload_stream = _original_cast(
        s,
        input=input,
        dtype=dtype,
        device=input.device,
        bias_dtype=bias_dtype,
        offloadable=True,
        compute_dtype=dtype,
        want_requant=True,
    )
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        if offload_stream is not None:
            _comfy_ops.uncast_bias_weight(s, weight, bias, offload_stream)
        return None, "unexpected_patched_weight"

    weight_cpu = weight.detach().cpu()
    bias_cpu = bias.detach().cpu() if bias is not None else None
    s._omnixpu_patched_cache = (weight_cpu, bias_cpu)
    if offload_stream is not None:
        _comfy_ops.uncast_bias_weight(s, weight, bias, offload_stream)

    return weight, bias, (None, None, None)


def _patched_cast_bias_weight(s, input=None, dtype=None, device=None,
                              bias_dtype=None, offloadable=False,
                              compute_dtype=None, want_requant=False):
    eligible, reason = _module_eligible(s, input)
    if eligible:
        return _direct_cast_bias_weight(s, input, dtype, bias_dtype)
    if (
        reason == "weight_function"
        and not getattr(s, "bias_function", None)
        and not getattr(s, "weight_lowvram_function", None)
        and not getattr(s, "bias_lowvram_function", None)
        and os.environ.get("OMNIXPU_INT8_PATCH_CACHE", "0").strip().lower()
        not in ("", "0", "false", "no", "off")
    ):
        result = _cached_patched_weight(s, input, dtype, bias_dtype)
        if result is not None and result[0] is not None:
            log_debug_event(
                "dispatch",
                "int8_direct_cast",
                {"input": input, "weight": result[0]},
                details={
                    "hit": True,
                    "reason": "patched_weight_cached",
                    "weight_bytes": result[0].numel()
                    * result[0].element_size(),
                },
                verbose_only=True,
            )
            return result
    return _original_cast(
        s,
        input=input,
        dtype=dtype,
        device=device,
        bias_dtype=bias_dtype,
        offloadable=offloadable,
        compute_dtype=compute_dtype,
        want_requant=want_requant,
    )
