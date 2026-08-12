"""Patch comfy.ops fp8_linear and mixed_precision_ops to use
omni_xpu_kernel's oneDNN W8A16 FP8 / INT8 GEMMs when running on XPU.

Exactly mirrors the logic from comfyui_for_multi_arc.patch.
"""

import logging
import os

import torch
import comfy.model_management

from ..patches.debug import log_debug_event
from .errors import is_fatal_accelerator_error

log = logging.getLogger("ComfyUI-OmniXPU")

_omni_fp8_linear = None
_omni_int8 = None
_logged_first_use = False
_INT8_FAST_FORWARD = (
    os.environ.get("OMNIXPU_INT8_FAST_FORWARD", "1").strip().lower()
    not in ("", "0", "false", "no", "off")
)
_INT8_FAST_FORWARD_COPY = (
    os.environ.get("OMNIXPU_INT8_FAST_FORWARD_COPY", "1").strip().lower()
    not in ("", "0", "false", "no", "off")
)


def _log_first(msg):
    global _logged_first_use
    if not _logged_first_use:
        _logged_first_use = True
        log.info("[OmniXPU] fp8_gemm first use: %s", msg)


def _dispatch_details(module):
    weight = getattr(module, "weight", None)
    layout = getattr(module, "layout_type", None)
    if layout is None:
        layout = getattr(weight, "_layout_cls", None)
    return {
        "quant_format": getattr(module, "quant_format", None),
        "layout": layout,
    }


def _int8_skip_reason(self, input, QuantizedTensor, TensorWiseINT8Layout):
    """Return why the INT8 fast forward did not take this call (or None)."""
    if not _INT8_FAST_FORWARD:
        return "env_disabled"
    if _omni_int8 is None:
        return "omni_int8_unavailable"
    if not input.is_xpu:
        return "input_not_xpu"
    if len(self.weight_function):
        return f"weight_function={len(self.weight_function)}"
    if len(self.bias_function):
        return f"bias_function={len(self.bias_function)}"
    qf = getattr(self, "quant_format", None)
    if qf != "int8_tensorwise":
        return f"quant_format={qf!r}"
    if getattr(self, "_full_precision_mm", False):
        return "full_precision_mm"
    if getattr(self, "comfy_force_cast_weights", False):
        return "comfy_force_cast_weights"
    if getattr(self, "pre_quant_scale", None) is not None:
        return "pre_quant_scale_set"
    if getattr(self, "input_scale", None) is not None:
        return "input_scale_set"
    w = self.weight
    if not isinstance(w, QuantizedTensor):
        return f"weight_type={type(w).__name__}"
    if getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout":
        return f"layout={getattr(w, '_layout_cls', None)!r}"
    if (
        not _INT8_FAST_FORWARD_COPY
        and getattr(w, "device", None) != input.device
    ):
        return (
            f"weight_device={getattr(w, 'device', None)} "
            f"input_device={input.device}"
        )
    try:
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(w)
    except Exception as e:
        return f"get_plain_tensors_error={e}"
    if not qdata.is_contiguous():
        return "qdata_not_contiguous"
    if qdata.device != input.device:
        return f"qdata_device={qdata.device}"
    if scale.device != input.device:
        return f"scale_device={scale.device}"
    return None


_int8_skip_logged = set()


def _log_int8_skip(reason):
    if reason is None or reason in _int8_skip_logged:
        return
    if len(_int8_skip_logged) >= 10:
        return
    _int8_skip_logged.add(reason)
    log.info("[OmniXPU] int8 fast path skipped: %s", reason)


def _prepare_scale(scale, weight, input):
    scale = torch.as_tensor(scale, device=input.device, dtype=torch.float32).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(weight.shape[0]).contiguous()
    return scale


def apply():
    global _omni_fp8_linear, _omni_int8
    import sys
    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe.linear_fp8 is None:
        return False, "omni_xpu_kernel linear_fp8 not available"
    _omni_fp8_linear = probe.linear_fp8
    _omni_int8 = probe.int8
    log.info(
        "[OmniXPU] int8 fast forward: %s (copy=%s)",
        "enabled" if _INT8_FAST_FORWARD else "disabled (OMNIXPU_INT8_FAST_FORWARD=0)",
        "on" if _INT8_FAST_FORWARD_COPY else "off",
    )

    import comfy.ops as comfy_ops

    # --- Patch fp8_linear module-level function ---
    if hasattr(comfy_ops, "fp8_linear"):
        _orig_fp8_linear = comfy_ops.fp8_linear

        def _patched_fp8_linear(self, input):
            log_debug_event(
                "dispatch",
                "fp8_linear",
                {"input": input},
                details=_dispatch_details(self),
                verbose_only=True,
            )
            dtype = self.weight.dtype
            if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                return None

            input_dtype = input.dtype
            input_shape = input.shape
            tensor_3d = input.ndim == 3
            if tensor_3d:
                input = input.reshape(-1, input_shape[2])
            if input.ndim != 2:
                return None

            lora_compute_dtype = comfy.model_management.lora_compute_dtype(input.device)
            w, bias, offload_stream = comfy_ops.cast_bias_weight(
                self, input, dtype=dtype, bias_dtype=input_dtype,
                offloadable=True, compute_dtype=lora_compute_dtype, want_requant=True,
            )

            # --- omni_xpu_kernel oneDNN FP8 GEMM (fast path for XPU) ---
            if _omni_fp8_linear is not None and input.is_xpu:
                _log_first(f"input={list(input.shape)} weight={list(w.shape)} dtype={dtype}")
                scale_weight = self.scale_weight if hasattr(self, 'scale_weight') and self.scale_weight is not None else torch.ones((), device=input.device, dtype=torch.float32)
                try:
                    scale_weight = _prepare_scale(scale_weight, w, input)
                    o = _omni_fp8_linear(input, w, scale_weight, bias)
                    if o is not None:
                        log_debug_event(
                            "kernel",
                            "fp8_linear",
                            {"input": input, "weight": w, "weight_scale": scale_weight, "bias": bias},
                            details={"backend": "omni_xpu", "format": dtype},
                        )
                        comfy_ops.uncast_bias_weight(self, w, bias, offload_stream)
                        if tensor_3d:
                            o = o.reshape((input_shape[0], input_shape[1], w.shape[0]))
                        return o
                except Exception as e:
                    if is_fatal_accelerator_error(e):
                        raise
                    _log_first(f"failed, falling back: {e}")

            # --- Original path: QuantizedTensor dispatch ---
            comfy_ops.uncast_bias_weight(self, w, bias, offload_stream)
            return _orig_fp8_linear(self, input)

        comfy_ops.fp8_linear = _patched_fp8_linear

    # --- Patch mixed_precision_ops Linear ---
    if hasattr(comfy_ops, "mixed_precision_ops"):
        _orig_mixed = comfy_ops.mixed_precision_ops
        QuantizedTensor = getattr(comfy_ops, "QuantizedTensor", None)
        TensorWiseINT8Layout = getattr(comfy_ops, "TensorWiseINT8Layout", None)

        def _patched_mixed(*args, **kwargs):
            klass = _orig_mixed(*args, **kwargs)

            _orig_fwd = klass.Linear.forward
            _orig_inner_fwd = klass.Linear._forward

            # -- Intercept 1: _forward(input, weight, bias) --
            # Called from forward_comfy_cast_weights after cast_bias_weight.
            def _mp_inner_forward(self, input, weight, bias):
                if (_omni_fp8_linear is not None and input.is_xpu and input.ndim == 2 and
                        hasattr(weight, 'dtype') and weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)):
                    _log_first(f"input={list(input.shape)} weight={list(weight.shape)} dtype={weight.dtype}")
                    try:
                        scale_w = getattr(self, 'scale_weight', None)
                        if scale_w is None:
                            p = getattr(self.weight, 'params', None) or getattr(self.weight, '_layout_params', None)
                            scale_w = getattr(p, 'scale', None) if p else None
                        if scale_w is None:
                            scale_w = torch.ones((), device=input.device, dtype=torch.float32)
                        scale_w = _prepare_scale(scale_w, weight, input)
                        output = _omni_fp8_linear(input, weight, scale_w, bias)
                        if output is not None:
                            log_debug_event(
                                "kernel",
                                "fp8_linear",
                                {"input": input, "weight": weight, "weight_scale": scale_w, "bias": bias},
                                details={"backend": "omni_xpu", "format": weight.dtype},
                            )
                            return output
                    except Exception as e:
                        if is_fatal_accelerator_error(e):
                            raise
                        _log_first(f"_forward failed, falling back: {e}")
                return _orig_inner_fwd(self, input, weight, bias)

            # -- Intercept 2: forward() --
            # Intercepts before comfy_kitchen QuantizedTensor dispatch.
            def _mp_forward(self, input, *fwd_args, **fwd_kwargs):
                log_debug_event(
                    "dispatch",
                    "mixed_precision.Linear",
                    {"input": input},
                    details=_dispatch_details(self),
                    verbose_only=True,
                )
                # INT8 fast path (A770/DG2): call the omni oneDNN s8 GEMM
                # directly instead of going through
                # QuantizedTensor.from_float -> cast_bias_weight -> torch
                # dispatch. The original path quantizes the activation, hands
                # it to a kernel that dequantizes it back to the compute dtype,
                # and our registered kernel then re-quantizes rowwise; that
                # round trip is pure overhead for TensorWise INT8. Mirror the
                # exact conditions linear_input_act uses for its fused path:
                # quantized TensorWise INT8 weight, no LoRA functions, no AWQ
                # smoothing / static input scale, weight resident on XPU.
                if (
                    _INT8_FAST_FORWARD
                    and
                    _omni_int8 is not None
                    and input.is_xpu
                    and len(self.weight_function) == 0
                    and len(self.bias_function) == 0
                    and getattr(self, "quant_format", None) == "int8_tensorwise"
                    and not getattr(self, "_full_precision_mm", False)
                    and not getattr(self, "comfy_force_cast_weights", False)
                    and getattr(self, "pre_quant_scale", None) is None
                    and getattr(self, "input_scale", None) is None
                ):
                    try:
                        comfy_ops.run_every_op()
                        input_shape = input.shape
                        input_2d = (
                            input.reshape(-1, input_shape[-1])
                            if input.ndim >= 3
                            else input
                        )
                        w = self.weight
                        if (
                            input_2d.ndim == 2
                            and QuantizedTensor is not None
                            and isinstance(w, QuantizedTensor)
                            and getattr(w, "_layout_cls", None)
                            == "TensorWiseINT8Layout"
                            and TensorWiseINT8Layout is not None
                        ):
                            qdata, scale = TensorWiseINT8Layout.get_plain_tensors(w)
                            device_ok = (
                                qdata.is_contiguous()
                                and qdata.device == input.device
                                and scale.device == input.device
                            )
                            if device_ok or _INT8_FAST_FORWARD_COPY:
                                if qdata.device != input.device:
                                    qdata = qdata.to(device=input.device)
                                if scale.device != input.device:
                                    scale = scale.to(device=input.device)
                                if not qdata.is_contiguous():
                                    qdata = qdata.contiguous()
                                params = getattr(w, "_params", None) or getattr(
                                    w, "params", None
                                )
                                bias = None
                                if self.bias is not None:
                                    bias = comfy.model_management.cast_to_device(
                                        self.bias, input.device, input.dtype
                                    )
                                out = _omni_int8.int8_linear(
                                    input_2d,
                                    qdata,
                                    scale,
                                    bias,
                                    out_dtype=input.dtype,
                                    convrot=bool(getattr(params, "convrot", False)),
                                    convrot_groupsize=int(
                                        getattr(params, "convrot_groupsize", 256)
                                    ),
                                    input_act=None,
                                )
                                if out is not None:
                                    log_debug_event(
                                        "kernel",
                                        "int8_linear",
                                        {
                                            "x": input_2d,
                                            "weight": qdata,
                                            "weight_scale": scale,
                                            "bias": bias,
                                        },
                                        details={
                                            "backend": "omni_dg2_compat_fast"
                                        },
                                    )
                                    if input.ndim >= 3:
                                        out = out.reshape(
                                            *input_shape[:-1], -1
                                        )
                                    return out
                            else:
                                _log_int8_skip(
                                    "qdata/scale not contiguous or not on "
                                    f"input device and copy disabled "
                                    f"(qdata_device="
                                    f"{getattr(qdata, 'device', None)}, "
                                    f"scale_device={getattr(scale, 'device', None)})"
                                )
                        else:
                            _log_int8_skip(
                                f"shape={tuple(input.shape)} "
                                + (
                                    _int8_skip_reason(
                                        self, input, QuantizedTensor,
                                        TensorWiseINT8Layout,
                                    )
                                    or "unknown"
                                )
                            )
                    except Exception as e:
                        if is_fatal_accelerator_error(e):
                            raise
                        _log_first(f"int8 fast forward failed, falling back: {e}")
                else:
                    _log_int8_skip(
                        f"shape={tuple(input.shape)} "
                        + (
                            _int8_skip_reason(
                                self, input, QuantizedTensor,
                                TensorWiseINT8Layout,
                            )
                            or "unknown"
                        )
                    )

                if (_omni_fp8_linear is not None and input.is_xpu and
                        getattr(self, 'quant_format', None) in ('float8_e4m3fn', 'float8_e5m2') and
                        len(self.weight_function) == 0 and len(self.bias_function) == 0):
                    input_shape = input.shape
                    input_2d = input.reshape(-1, input_shape[-1]) if input.ndim == 3 else input
                    if input_2d.ndim == 2:
                        try:
                            w = self.weight
                            fp8_dtype = torch.float8_e4m3fn if self.quant_format == 'float8_e4m3fn' else torch.float8_e5m2
                            if QuantizedTensor is not None and isinstance(w, QuantizedTensor):
                                w_fp8 = w._qdata
                                scale_w = getattr(w.params, 'scale', None)
                            else:
                                w_fp8 = w if w.dtype == fp8_dtype else w.view(fp8_dtype)
                                scale_w = getattr(self, 'scale_weight', None)
                            if scale_w is None:
                                scale_w = torch.ones((), device=input.device, dtype=torch.float32)
                            scale_w = comfy.model_management.cast_to_device(scale_w, input.device, torch.float32)
                            w_fp8 = comfy.model_management.cast_to_device(w_fp8, input.device, None)
                            scale_w = _prepare_scale(scale_w, w_fp8, input_2d)
                            bias = (comfy.model_management.cast_to_device(self.bias, input.device, input.dtype)
                                    if self.bias is not None else None)

                            _log_first(f"input={list(input_2d.shape)} weight={list(w_fp8.shape)} "
                                       f"dtype={w_fp8.dtype} format={self.quant_format}")

                            o = _omni_fp8_linear(input_2d, w_fp8, scale_w, bias)
                            if o is not None:
                                log_debug_event(
                                    "kernel",
                                    "fp8_linear",
                                    {"input": input_2d, "weight": w_fp8, "weight_scale": scale_w, "bias": bias},
                                    details={"backend": "omni_xpu", "format": self.quant_format},
                                )
                                if input.ndim == 3:
                                    o = o.reshape(input_shape[0], input_shape[1], -1)
                                return o
                        except Exception as e:
                            if is_fatal_accelerator_error(e):
                                raise
                            _log_first(f"forward failed, falling back: {e}")

                return _orig_fwd(self, input, *fwd_args, **fwd_kwargs)

            klass.Linear.forward = _mp_forward
            klass.Linear._forward = _mp_inner_forward
            return klass

        comfy_ops.mixed_precision_ops = _patched_mixed

    return True, None
