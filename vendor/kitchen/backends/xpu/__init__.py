"""Intel XPU backend powered by the optional omni_xpu_kernel package."""

from __future__ import annotations

import sys

import torch

from comfy_kitchen.allocation import allocation_context
from comfy_kitchen.backends._activations import apply_input_act as _apply_input_act
from comfy_kitchen.constraints import (
    ExactDims, FunctionConstraints, ParamConstraint, ValidationResult, sol_attn_common_call_rule,
    with_out_param,
)
from comfy_kitchen.registry import registry

from .w4a8_int8 import native_call_rule as _w4a8_native_call_rule
from .w4a8_int8 import w4a8_int8_linear

__all__ = [
    "sol_attn",
    "sol_attn_chunked",
    "adaln",
    "rms_adaln",
    "apply_rope",
    "apply_rope_",
    "apply_rope1",
    "apply_rope1_",
    "apply_rope_split_half",
    "apply_rope_split_half_",
    "apply_rope_split_half1",
    "apply_rope_split_half1_",
    "convrot_w4a4_linear",
    "dequantize_convrot_w4a4_weight",
    "dequantize_gguf",
    "dequantize_per_tensor_fp8",
    "dequantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight_dtype",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "int8_linear",
    "int8_linear_modulated",
    "int8_linear_rms_modulated",
    "mm_int8",
    "prepare_int4_weight_for_int8_linear",
    "quantize_and_rotate_rowwise",
    "quantize_int8_convrot_weight",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
    "quantize_convrot_w4a4_weight",
    "quantize_per_tensor_fp8",
    "rms_rope",
    "rms_rope_",
    "rms_rope1",
    "rms_rope1_",
    "rms_rope_split_half",
    "rms_rope_split_half_",
    "rms_rope_split_half1",
    "rms_rope_split_half1_",
    "quantize_svdquant_w4a4",
    "scaled_mm_svdquant_w4a4",
    "svdquant_w4a16_linear",
    "stochastic_rounding_fp8",
    "gated_delta_decode_is_available",
    "gated_delta_decode_fused",
    "deltanet_conv_step",
    "group_norm_stats3d",
    "group_norm_silu_pad3d",
    "group_norm_silu_pad3d_out",
    "fp16_linear",
    "fp16_conv3d",
    "fp16_conv3d_out",
    "gemv_awq_w4a16",
    "w4a8_int8_linear",
]

_AVAILABLE = False
_ERROR = None
_NATIVE_CAPABILITIES = frozenset()
_INT8_AVAILABLE = False
_INT8_ERROR = None
_SVDQ_AVAILABLE = False
_SVDQ_W4A16_AVAILABLE = False
_NORM_AVAILABLE = False
_FP8_AVAILABLE = False
_FP8_QDQ_AVAILABLE = False
_ROPE_AVAILABLE = False
_RMS_ROPE_AVAILABLE = False
_CONVROT_NATIVE_AVAILABLE = False
_GGUF_AVAILABLE = False
_SOL_AVAILABLE = False
_SOL_ERROR = None
_GROUP_NORM_STATS3D_AVAILABLE = False
_GROUP_NORM_SILU_PAD3D_AVAILABLE = False
_GROUP_NORM_SILU_PAD3D_OUT_AVAILABLE = False
_FP16_LINEAR_AVAILABLE = False
_FP16_CONV3D_AVAILABLE = False
_FP16_CONV3D_OUT_AVAILABLE = False
_AWQ_W4A16_AVAILABLE = False
_RMS_NORM_FOR_INT8_AVAILABLE = False
_RMS_NORM_QUANTIZE_AVAILABLE = False
_RMS_NORM_CONVROT_QUANT_AVAILABLE = False
_GATED_QUANTIZE_AVAILABLE = False
_SCALED_RESIDUAL_AVAILABLE = False
_FUSED_RESIDUAL_AVAILABLE = False


def gated_delta_decode_is_available(
    key_head_dim: int = 128, value_head_dim: int = 128,
) -> bool:
    if not _AVAILABLE or key_head_dim != 128 or value_head_dim % 32 != 0 or not 0 < value_head_dim <= 512:
        return False
    try:
        from omni_xpu_kernel import kitchen
    except ImportError:
        return False
    return (
        kitchen.supports_deltanet_conv_step()
        and kitchen.supports_gated_delta_decode_fused()
    )


def deltanet_conv_step(proj, conv_state, conv_w, conv_b=None, snapshots=None):
    from omni_xpu_kernel import kitchen

    return kitchen.deltanet_conv_step(
        proj, conv_state, conv_w, conv_b, snapshots,
    )


def gated_delta_decode_fused(
    mixed_qkv, x, w_a, w_b, dt_bias, g_decay, state,
    key_dim, num_key_heads, scale, z, norm_weight, eps, snapshots=None,
):
    from omni_xpu_kernel import kitchen

    return kitchen.gated_delta_decode_fused(
        mixed_qkv, x, w_a, w_b, dt_bias, g_decay, state,
        key_dim, num_key_heads, scale, z, norm_weight, eps, snapshots,
    )


def group_norm_stats3d(x, num_groups=32, eps=1e-6):
    from omni_xpu_kernel import kitchen
    return kitchen.group_norm_stats3d(x, num_groups, eps)


def group_norm_silu_pad3d(x, weight, bias, num_groups, eps, pad, silu,
                          zero_pad=False, stats=None):
    from omni_xpu_kernel import kitchen

    if min(pad) < 0:
        raise ValueError("group_norm_silu_pad3d: padding must be non-negative")
    if (zero_pad and not _GROUP_NORM_SILU_PAD3D_OUT_AVAILABLE) or (stats is not None and not _GROUP_NORM_STATS3D_AVAILABLE):
        from comfy_kitchen.backends.eager import group_norm_silu_pad3d as eager_group_norm
        return eager_group_norm(
            x, weight, bias, num_groups, eps, pad, silu, zero_pad, stats,
        )
    return kitchen.group_norm_silu_pad3d(
        x,
        None if weight is None else weight.to(x.dtype),
        None if bias is None else bias.to(x.dtype),
        num_groups, eps, tuple(pad), silu, zero_pad, stats,
    )


def group_norm_silu_pad3d_out(x, weight, bias, num_groups, eps, pad, silu,
                              zero_pad, out, stats=None):
    from omni_xpu_kernel import kitchen

    if min(pad) < 0:
        raise ValueError("group_norm_silu_pad3d: padding must be non-negative")
    if stats is not None and not _GROUP_NORM_STATS3D_AVAILABLE:
        from comfy_kitchen.backends.eager import group_norm_silu_pad3d_out as eager_out
        return eager_out(x, weight, bias, num_groups, eps, pad, silu, zero_pad, out, stats)
    kitchen.group_norm_silu_pad3d_out(
        x,
        None if weight is None else weight.to(x.dtype),
        None if bias is None else bias.to(x.dtype),
        num_groups, eps, tuple(pad), silu, zero_pad, out, stats,
    )


def fp16_linear(x, weight, bias=None, residual=None, residual_scale=None):
    from omni_xpu_kernel import kitchen

    return kitchen.fp16_linear(
        x, weight,
        None if bias is None else bias.to(x.dtype),
        None if residual is None else residual.to(x.dtype),
        None if residual_scale is None else residual_scale.to(x.dtype),
    )


def fp16_conv3d(x, weight, bias=None, residual=None, stride=None):
    from omni_xpu_kernel import kitchen

    return kitchen.fp16_conv3d(
        x, weight,
        None if bias is None else bias.to(x.dtype),
        None if residual is None else residual.to(x.dtype),
        tuple((1, 1, 1) if stride is None else stride),
    )


def fp16_conv3d_out(x, weight, bias, residual, stride, out):
    from omni_xpu_kernel import kitchen

    kitchen.fp16_conv3d_out(
        x, weight,
        None if bias is None else bias.to(x.dtype),
        None if residual is None else residual.to(x.dtype),
        stride, out,
    )


def gemv_awq_w4a16(x, qweight, wscales, wzeros, bias=None, group_size=64):
    from omni_xpu_kernel import kitchen

    compute_dtype = wscales.dtype
    return kitchen.gemv_awq_w4a16(
        x.to(compute_dtype), qweight,
        wscales, wzeros.to(compute_dtype),
        None if bias is None else bias.to(compute_dtype),
        group_size,
    )

_REQUIRED_NATIVE_INT8_OPS = frozenset(
    {
        "dequantize_int8_simple",
        "dequantize_int8_simple_dtype",
        "int8_linear",
        "mm_int8",
        "quantize_int8_rowwise",
        "quantize_int8_tensorwise",
    }
)

try:
    import omni_xpu_kernel
    from omni_xpu_kernel import int8 as _int8

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        _ERROR = "PyTorch XPU is not available on this system"
    elif not omni_xpu_kernel.is_available():
        _ERROR = "omni_xpu_kernel native extension is not available"
    else:
        _extension = omni_xpu_kernel._load_extension()
        _AVAILABLE = True
        _native_kitchen = getattr(_extension, "kitchen", None)
        _GROUP_NORM_STATS3D_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "group_norm_stats3d")
        )
        _GROUP_NORM_SILU_PAD3D_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "group_norm_silu_pad3d")
        )
        _GROUP_NORM_SILU_PAD3D_OUT_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "group_norm_silu_pad3d_out")
        )
        _FP16_LINEAR_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "fp16_linear")
        )
        _FP16_CONV3D_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "fp16_conv3d")
        )
        _FP16_CONV3D_OUT_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "fp16_conv3d_out")
        )
        _AWQ_W4A16_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "gemv_awq_w4a16")
        )
        _RMS_NORM_FOR_INT8_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "rms_norm_for_int8")
        )
        _RMS_NORM_QUANTIZE_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "rms_norm_quantize_int8")
        )
        _RMS_NORM_CONVROT_QUANT_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "rms_norm_convrot_quantize_int8")
        )
        _GATED_QUANTIZE_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "gated_quantize_int8")
        )
        _SCALED_RESIDUAL_AVAILABLE = (
            _native_kitchen is not None
            and hasattr(_native_kitchen, "scaled_residual")
        )
        _native_int8 = getattr(_extension, "int8", None)
        _FUSED_RESIDUAL_AVAILABLE = (
            _native_int8 is not None
            and hasattr(_native_int8, "int8_linear_prequantized_residual")
        )
        _NATIVE_CAPABILITIES = frozenset(
            name
            for name in _REQUIRED_NATIVE_INT8_OPS
            if _native_int8 is not None and hasattr(_native_int8, name)
        )
        missing = _REQUIRED_NATIVE_INT8_OPS - _NATIVE_CAPABILITIES
        if missing:
            _INT8_ERROR = "omni_xpu_kernel INT8 extension is missing: " + ", ".join(
                sorted(missing)
            )
        else:
            _INT8_AVAILABLE = True

        _native_svdq = getattr(_extension, "svdq", None)
        _SVDQ_AVAILABLE = _native_svdq is not None and all(
            hasattr(_native_svdq, name)
            for name in (
                "dequantize_svdq_w4",
                "quantize_svdq_act_int4",
                "onednn_int4_gemm",
            )
        )
        _SVDQ_W4A16_AVAILABLE = _native_svdq is not None and all(
            hasattr(_native_svdq, name)
            for name in (
                "fused_smooth_mul_convert",
                "onednn_int4_gemm_add_to_output",
                "onednn_int4_gemm_preconverted",
            )
        )
        _native_norm = getattr(_extension, "norm", None)
        _NORM_AVAILABLE = _native_norm is not None and hasattr(
            _native_norm, "layer_norm"
        )
        _native_linear = getattr(_extension, "linear", None)
        _FP8_AVAILABLE = _native_linear is not None and hasattr(
            _native_linear, "onednn_w8a16_fp8"
        )
        _native_fp8 = getattr(_extension, "fp8", None)
        _FP8_QDQ_AVAILABLE = _native_fp8 is not None and all(
            hasattr(_native_fp8, name)
            for name in (
                "dequantize_per_tensor",
                "quantize_per_tensor",
                "stochastic_rounding",
            )
        )
        _native_rotary = getattr(_extension, "rotary", None)
        _ROPE_AVAILABLE = _native_rotary is not None and all(
            hasattr(_native_rotary, name)
            for name in (
                "apply_kitchen_rope",
                "apply_kitchen_rope1",
                "apply_kitchen_rope_split_half",
                "apply_kitchen_rope_split_half1",
            )
        )
        _RMS_ROPE_AVAILABLE = _native_rotary is not None and all(
            hasattr(_native_rotary, name)
            for name in ("rms_kitchen_rope", "rms_kitchen_rope1")
        )
        _CONVROT_NATIVE_AVAILABLE = _native_int8 is not None and all(
            hasattr(_native_int8, name)
            for name in (
                "dequantize_int8_convrot_weight",
                "quantize_int8_convrot_weight",
                "rotate_convrot",
            )
        )
        _native_gguf = getattr(_extension, "gguf", None)
        _GGUF_AVAILABLE = _native_gguf is not None and all(
            hasattr(_native_gguf, name)
            for name in (
                "dequantize_q4_0",
                "dequantize_q4_1",
                "dequantize_q8_0",
                "dequantize_q4_k",
                "dequantize_q6_k",
            )
        )
except (ImportError, OSError, RuntimeError) as exc:
    _ERROR = f"{type(exc).__name__}: {exc}"


# The optional CUTE sidecar is independent of the core extension. An older
# sidecar must not disable the other Kitchen XPU capabilities.
if _AVAILABLE:
    try:
        from omni_xpu_kernel.cute import sol_attn_v2 as _sol
        if hasattr(_sol, "set_allocation_context_factory"):
            _sol.set_allocation_context_factory(allocation_context)
        _SOL_AVAILABLE = _sol.is_available()
        if not _SOL_AVAILABLE:
            _SOL_ERROR = "the Sol sidecar does not expose the complete native API"
    except (ImportError, OSError, RuntimeError) as exc:
        _SOL_ERROR = f"{type(exc).__name__}: {exc}"


if _AVAILABLE:
    if _SOL_AVAILABLE:
        sol_attn = _sol.sol_attn
        sol_attn_chunked = _sol.sol_attn_chunked
    if _INT8_AVAILABLE:
        quantize_int8_tensorwise = _int8.quantize_int8_tensorwise
        quantize_int8_rowwise = _int8.quantize_int8_rowwise
        dequantize_int8_simple = _int8.dequantize_int8_simple
        mm_int8 = _int8.mm_int8
        quantize_int8_convrot_weight = _int8.quantize_int8_convrot_weight
        dequantize_int8_convrot_weight = _int8.dequantize_int8_convrot_weight

        _INT8_LINEAR_DTYPES = (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        )
        _INT8_LINEAR_INPUT_ACTS = (None, "none", "gelu_tanh", "swiglu", "situ")

        @torch.library.custom_op(
            "comfy_kitchen_xpu::int8_linear",
            mutates_args=(),
        )
        def _compiled_int8_linear(
            x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.Tensor | None,
            output_dtype_code: int,
            convrot: bool,
            convrot_groupsize: int,
            input_act_code: int,
        ) -> torch.Tensor:
            return _int8.int8_linear(
                x,
                weight,
                weight_scale,
                bias,
                _INT8_LINEAR_DTYPES[output_dtype_code],
                convrot,
                convrot_groupsize,
                _INT8_LINEAR_INPUT_ACTS[input_act_code],
            )

        @_compiled_int8_linear.register_fake
        def _compiled_int8_linear_fake(
            x,
            weight,
            weight_scale,
            bias,
            output_dtype_code,
            convrot,
            convrot_groupsize,
            input_act_code,
        ):
            del weight_scale, bias, convrot, convrot_groupsize, input_act_code
            return x.new_empty(
                (*x.shape[:-1], weight.shape[0]),
                dtype=_INT8_LINEAR_DTYPES[output_dtype_code],
            )

        @torch.library.custom_op(
            "comfy_kitchen_xpu::int8_linear_residual",
            mutates_args=(),
        )
        def _compiled_int8_linear_residual(
            x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.Tensor | None,
            output_dtype_code: int,
            residual: torch.Tensor,
            residual_scale: torch.Tensor,
        ) -> torch.Tensor:
            x_int8, x_scale = _int8.quantize_int8_rowwise(x)
            return _native_int8.int8_linear_prequantized_residual(
                x_int8, x_scale, weight, weight_scale, bias,
                output_dtype_code, residual, residual_scale,
            )

        @_compiled_int8_linear_residual.register_fake
        def _compiled_int8_linear_residual_fake(
            x, weight, weight_scale, bias, output_dtype_code,
            residual, residual_scale,
        ):
            del weight_scale, bias, residual, residual_scale
            return x.new_empty(
                (*x.shape[:-1], weight.shape[0]),
                dtype=_INT8_LINEAR_DTYPES[output_dtype_code],
            )

        @torch.library.custom_op(
            "comfy_kitchen_xpu::int8_linear_prequantized_residual",
            mutates_args=(),
        )
        def _compiled_int8_linear_prequantized_residual(
            x_int8: torch.Tensor,
            x_scale: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.Tensor | None,
            output_dtype_code: int,
            residual: torch.Tensor,
            residual_scale: torch.Tensor,
        ) -> torch.Tensor:
            return _native_int8.int8_linear_prequantized_residual(
                x_int8, x_scale, weight, weight_scale, bias,
                output_dtype_code, residual, residual_scale,
            )

        @_compiled_int8_linear_prequantized_residual.register_fake
        def _compiled_int8_linear_prequantized_residual_fake(
            x_int8, x_scale, weight, weight_scale, bias,
            output_dtype_code, residual, residual_scale,
        ):
            del x_scale, weight_scale, bias, residual, residual_scale
            return x_int8.new_empty(
                (*x_int8.shape[:-1], weight.shape[0]),
                dtype=_INT8_LINEAR_DTYPES[output_dtype_code],
            )

        def int8_linear(
            x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.Tensor | None = None,
            out_dtype: torch.dtype | None = None,
            convrot: bool = False,
            convrot_groupsize: int = 256,
            input_act: str | None = None,
            input_act_weight: torch.Tensor | None = None,
            input_act_eps: float = 0.0,
            residual: torch.Tensor | None = None,
            residual_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            prepared = None
            if input_act == "situ" and not _GATED_QUANTIZE_AVAILABLE:
                # An older companion still supplies the INT8 GEMM. Prepare the
                # new activation here rather than passing it an unknown code.
                x = _apply_input_act(x, input_act)
                input_act = None
            if input_act == "rms_norm":
                if input_act_weight is None:
                    raise ValueError("input_act 'rms_norm' requires input_act_weight")
                if not _RMS_NORM_FOR_INT8_AVAILABLE:
                    raise RuntimeError("Omni XPU RMSNorm for INT8 is unavailable")
                norm_weight = input_act_weight.to(x.dtype).contiguous()
                from omni_xpu_kernel import kitchen

                if (
                    convrot
                    and _RMS_NORM_CONVROT_QUANT_AVAILABLE
                    and convrot_groupsize in (64, 256)
                    and x.dtype in (torch.float16, torch.bfloat16)
                    and x.shape[-1] % convrot_groupsize == 0
                ):
                    prepared = kitchen.rms_norm_convrot_quantize_int8(
                        x, norm_weight, input_act_eps, convrot_groupsize,
                    )
                    convrot = False
                elif _RMS_NORM_QUANTIZE_AVAILABLE and not convrot:
                    prepared = kitchen.rms_norm_quantize_int8(
                        x, norm_weight, input_act_eps,
                    )
                else:
                    x = kitchen.rms_norm_for_int8(
                        x, norm_weight, input_act_eps,
                    )
                input_act = None
            if prepared is not None:
                actual_dtype = x.dtype if out_dtype is None else out_dtype
                output_dtype_code = {
                    torch.float32: 0,
                    torch.float16: 1,
                    torch.bfloat16: 2,
                }.get(actual_dtype, 2)
                if (
                    residual is not None
                    and residual_scale is not None
                    and _FUSED_RESIDUAL_AVAILABLE
                ):
                    add = residual.to(actual_dtype).contiguous()
                    add_scale = residual_scale.to(actual_dtype).contiguous()
                    if torch.compiler.is_compiling():
                        return _compiled_int8_linear_prequantized_residual(
                            *prepared, weight, weight_scale, bias,
                            output_dtype_code, add, add_scale,
                        )
                    return _native_int8.int8_linear_prequantized_residual(
                        *prepared, weight, weight_scale, bias,
                        output_dtype_code, add, add_scale,
                    )
                out = _int8.int8_linear_prequantized(
                    *prepared, weight, weight_scale, bias, actual_dtype,
                )
                if residual is None:
                    return out
                if residual_scale is None:
                    raise ValueError("residual requires residual_scale")
                if not _SCALED_RESIDUAL_AVAILABLE:
                    raise RuntimeError("Omni XPU scaled residual is unavailable")
                add = residual.to(actual_dtype).contiguous()
                add_scale = residual_scale.to(actual_dtype).contiguous()
                return kitchen.scaled_residual(out, add, add_scale)
            if (
                residual is not None
                and residual_scale is not None
                and _FUSED_RESIDUAL_AVAILABLE
                and not convrot
                and input_act in (None, "none")
            ):
                actual_dtype = x.dtype if out_dtype is None else out_dtype
                output_dtype_code = {
                    torch.float32: 0,
                    torch.float16: 1,
                    torch.bfloat16: 2,
                }.get(actual_dtype, 2)
                add = residual.to(actual_dtype).contiguous()
                add_scale = residual_scale.to(actual_dtype).contiguous()
                if torch.compiler.is_compiling():
                    return _compiled_int8_linear_residual(
                        x, weight, weight_scale, bias, output_dtype_code,
                        add, add_scale,
                    )
                x_int8, x_scale = _int8.quantize_int8_rowwise(x)
                return _native_int8.int8_linear_prequantized_residual(
                    x_int8, x_scale, weight, weight_scale, bias,
                    output_dtype_code, add, add_scale,
                )
            if not torch.compiler.is_compiling():
                out = _int8.int8_linear(
                    x,
                    weight,
                    weight_scale,
                    bias,
                    out_dtype,
                    convrot,
                    convrot_groupsize,
                    input_act,
                )
            else:
                actual_dtype = x.dtype if out_dtype is None else out_dtype
                output_dtype_code = {
                    torch.float32: 0,
                    torch.float16: 1,
                    torch.bfloat16: 2,
                }.get(actual_dtype, 2)
                try:
                    input_act_code = _INT8_LINEAR_INPUT_ACTS.index(input_act)
                except ValueError as exc:
                    raise ValueError(f"Unsupported input_act: {input_act!r}") from exc
                out = _compiled_int8_linear(
                    x,
                    weight,
                    weight_scale,
                    bias,
                    output_dtype_code,
                    convrot,
                    convrot_groupsize,
                    input_act_code,
                )
            if residual is None:
                return out
            if residual_scale is None:
                raise ValueError("residual requires residual_scale")
            if not _SCALED_RESIDUAL_AVAILABLE:
                raise RuntimeError("Omni XPU scaled residual is unavailable")
            add = residual.to(out.dtype).contiguous()
            add_scale = residual_scale.to(out.dtype).contiguous()
            from omni_xpu_kernel import kitchen

            return kitchen.scaled_residual(
                out, add, add_scale,
            )

        def _int8_linear_modulated_impl(
            x: torch.Tensor, modulation_scale: torch.Tensor,
            weight: torch.Tensor, weight_scale: torch.Tensor,
            bias: torch.Tensor | None = None,
            out_dtype: torch.dtype = torch.bfloat16,
            convrot: bool = False, convrot_groupsize: int = 256,
        ) -> torch.Tensor:
            # Materialize dtype-rounded modulation and portable quantization
            # before native GEMM. Legacy fused quantization divides in FP32;
            # the new portable contract rounds that division in the input dtype.
            # Bias also rounds after the projection, as in the reference.
            from comfy_kitchen.backends.eager.quantization import (
                _modulate_int8_input, quantize_and_rotate_rowwise,
                quantize_int8_rowwise as reference_quantize,
            )
            x = _modulate_int8_input(x, modulation_scale)
            if convrot:
                from comfy_kitchen.tensor.int8_utils import _build_hadamard
                h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
                quantized, scale = quantize_and_rotate_rowwise(x, h, convrot_groupsize)
            else:
                quantized, scale = reference_quantize(x)
            result = _int8.int8_linear_prequantized(
                quantized, scale, weight, weight_scale, None, out_dtype,
            )
            return result if bias is None else result + bias.to(result.dtype)

        def _int8_linear_rms_modulated_impl(
            x: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float,
            modulation_scale: torch.Tensor,
            weight: torch.Tensor, weight_scale: torch.Tensor,
            bias: torch.Tensor | None = None,
            out_dtype: torch.dtype = torch.bfloat16,
            convrot: bool = False, convrot_groupsize: int = 256,
        ) -> torch.Tensor:
            # RMSNorm rounds before modulation in the upstream reference.
            normalized = torch.nn.functional.rms_norm(
                x, (x.shape[-1],), norm_weight, norm_eps,
            )
            return _int8_linear_modulated_impl(
                normalized, modulation_scale, weight, weight_scale, bias,
                out_dtype, convrot, convrot_groupsize,
            )

        # Keep the input-dtype rounding boundaries opaque to Inductor fusion.
        @torch.library.custom_op("comfy_kitchen_xpu::int8_linear_modulated", mutates_args=())
        def _compiled_modulated(
            x: torch.Tensor, modulation_scale: torch.Tensor,
            weight: torch.Tensor, weight_scale: torch.Tensor,
            bias: torch.Tensor | None, output_dtype_code: int,
            convrot: bool, convrot_groupsize: int,
        ) -> torch.Tensor:
            return _int8_linear_modulated_impl(
                x, modulation_scale, weight, weight_scale, bias,
                _INT8_LINEAR_DTYPES[output_dtype_code], convrot, convrot_groupsize,
            )

        @_compiled_modulated.register_fake
        def _compiled_modulated_fake(x, modulation_scale, weight, weight_scale,
                                      bias, output_dtype_code, convrot, convrot_groupsize):
            return x.new_empty((*x.shape[:-1], weight.shape[0]),
                               dtype=_INT8_LINEAR_DTYPES[output_dtype_code])

        @torch.library.custom_op("comfy_kitchen_xpu::int8_linear_rms_modulated", mutates_args=())
        def _compiled_rms_modulated(
            x: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float,
            modulation_scale: torch.Tensor, weight: torch.Tensor,
            weight_scale: torch.Tensor, bias: torch.Tensor | None,
            output_dtype_code: int, convrot: bool, convrot_groupsize: int,
        ) -> torch.Tensor:
            return _int8_linear_rms_modulated_impl(
                x, norm_weight, norm_eps, modulation_scale, weight, weight_scale,
                bias, _INT8_LINEAR_DTYPES[output_dtype_code], convrot, convrot_groupsize,
            )

        @_compiled_rms_modulated.register_fake
        def _compiled_rms_modulated_fake(x, norm_weight, norm_eps, modulation_scale,
                                         weight, weight_scale, bias, output_dtype_code,
                                         convrot, convrot_groupsize):
            return x.new_empty((*x.shape[:-1], weight.shape[0]),
                               dtype=_INT8_LINEAR_DTYPES[output_dtype_code])

        def int8_linear_modulated(
            x: torch.Tensor, modulation_scale: torch.Tensor,
            weight: torch.Tensor, weight_scale: torch.Tensor,
            bias: torch.Tensor | None = None, out_dtype: torch.dtype = torch.bfloat16,
            convrot: bool = False, convrot_groupsize: int = 256,
        ) -> torch.Tensor:
            if torch.compiler.is_compiling():
                return _compiled_modulated(x, modulation_scale, weight, weight_scale,
                    bias, _INT8_LINEAR_DTYPES.index(out_dtype), convrot, convrot_groupsize)
            return _int8_linear_modulated_impl(x, modulation_scale, weight, weight_scale,
                                               bias, out_dtype, convrot, convrot_groupsize)

        def int8_linear_rms_modulated(
            x: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float,
            modulation_scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
            bias: torch.Tensor | None = None, out_dtype: torch.dtype = torch.bfloat16,
            convrot: bool = False, convrot_groupsize: int = 256,
        ) -> torch.Tensor:
            if torch.compiler.is_compiling():
                return _compiled_rms_modulated(x, norm_weight, norm_eps, modulation_scale,
                    weight, weight_scale, bias, _INT8_LINEAR_DTYPES.index(out_dtype),
                    convrot, convrot_groupsize)
            return _int8_linear_rms_modulated_impl(x, norm_weight, norm_eps, modulation_scale,
                weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize)

    if _NORM_AVAILABLE:
        from .adaln import adaln, rms_adaln

    if _SVDQ_AVAILABLE:
        from .svdquant import quantize_svdquant_w4a4, scaled_mm_svdquant_w4a4

    if _SVDQ_W4A16_AVAILABLE:
        from .svdquant_w4a16 import svdquant_w4a16_linear

    if _FP8_QDQ_AVAILABLE:
        from .fp8 import (
            dequantize_per_tensor_fp8,
            quantize_per_tensor_fp8,
            stochastic_rounding_fp8,
        )

    if _ROPE_AVAILABLE:
        from .rope import (
            apply_rope,
            apply_rope_,
            apply_rope1,
            apply_rope1_,
            apply_rope_split_half,
            apply_rope_split_half_,
            apply_rope_split_half1,
            apply_rope_split_half1_,
        )

    if _RMS_ROPE_AVAILABLE:
        from .rope import (
            rms_rope,
            rms_rope_,
            rms_rope1,
            rms_rope1_,
            rms_rope_split_half,
            rms_rope_split_half_,
            rms_rope_split_half1,
            rms_rope_split_half1_,
        )

    if _CONVROT_NATIVE_AVAILABLE and _SVDQ_AVAILABLE:
        from .convrot_w4a4 import (
            convrot_w4a4_linear,
            dequantize_convrot_w4a4_weight,
            prepare_int4_weight_for_int8_linear,
            quantize_and_rotate_rowwise,
            quantize_convrot_w4a4_weight,
        )

    if _GGUF_AVAILABLE:
        from .gguf import dequantize_gguf

    if _INT8_AVAILABLE:

        def dequantize_int8_simple_dtype(
            q: torch.Tensor,
            scale: torch.Tensor,
            output_dtype_code: int,
        ) -> torch.Tensor:
            """Adapt Kitchen's dtype-code ABI to omni's torch.dtype API."""
            out_dtype = _CODE_TO_DTYPE[output_dtype_code]
            return _int8.dequantize_int8_simple_dtype(q, scale, out_dtype)

        def dequantize_int8_convrot_weight_dtype(
            q: torch.Tensor,
            scale: torch.Tensor,
            group_size: int,
            output_dtype_code: int,
        ) -> torch.Tensor:
            """Dequantize ConvRot weights and convert to the requested Kitchen dtype."""
            return _int8.dequantize_int8_convrot_weight(
                q, scale, group_size
            ).to(_CODE_TO_DTYPE[output_dtype_code])


_CODE_TO_DTYPE = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
}


def _modulated_int8_call_rule(kwargs):
    x = kwargs.get("x")
    for name in ("modulation_scale", "norm_weight"):
        value = kwargs.get(name)
        if value is not None and value.dtype != x.dtype:
            return ValidationResult.fail(name, "mixed-dtype modulation uses the eager route")
    return ValidationResult.ok()


def _build_constraints() -> dict[str, FunctionConstraints]:
    xpu = frozenset({"xpu"})
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    int8_2d = ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),))

    capabilities = {
        "quantize_int8_tensorwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "quantize_int8_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "dequantize_int8_simple": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=frozenset({torch.int8})),
                "scale": ParamConstraint(dtypes=floats),
            },
            default_devices=xpu,
        ),
        "dequantize_int8_simple_dtype": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=frozenset({torch.int8})),
                "scale": ParamConstraint(dtypes=floats),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
                "weight": int8_2d,
                "weight_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "bias": ParamConstraint(dtypes=floats),
                "out_dtype": ParamConstraint(dtypes=floats),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
                "input_act_weight": ParamConstraint(dtypes=floats),
                "input_act_eps": ParamConstraint(dtypes=frozenset({float})),
                "residual": ParamConstraint(dtypes=floats),
                "residual_scale": ParamConstraint(dtypes=floats),
            },
            default_devices=xpu,
        ),
        "mm_int8": FunctionConstraints(
            params={"a": int8_2d, "b": int8_2d},
            default_devices=xpu,
        ),
        "quantize_int8_convrot_weight": FunctionConstraints(
            params={
                "weight": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "dequantize_int8_convrot_weight": FunctionConstraints(
            params={
                "q": int8_2d,
                "scale": ParamConstraint(dtypes=floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "dequantize_int8_convrot_weight_dtype": FunctionConstraints(
            params={
                "q": int8_2d,
                "scale": ParamConstraint(dtypes=floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
    }
    if _INT8_AVAILABLE and hasattr(_native_int8, "int8_linear_prequantized"):
        for name in ("int8_linear_modulated", "int8_linear_rms_modulated"):
            params = {key: value for key, value in capabilities["int8_linear"].params.items()
                      if key not in ("input_act", "input_act_weight", "input_act_eps",
                                     "residual", "residual_scale")}
            params["modulation_scale"] = ParamConstraint(dtypes=floats)
            if name == "int8_linear_rms_modulated":
                params["norm_weight"] = ParamConstraint(dtypes=floats)
                params["norm_eps"] = ParamConstraint(dtypes=frozenset({float}))
            capabilities[name] = FunctionConstraints(
                params=params, default_devices=xpu, call_rules=(_modulated_int8_call_rule,),
            )
    if not _INT8_AVAILABLE:
        capabilities.clear()
    elif hasattr(_native_int8, "int8_linear_prequantized"):
        capabilities["w4a8_int8_linear"] = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
                "qdata": int8_2d,
                "s_rel": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float8_e4m3fn}),
                    shape_rules=(ExactDims(2),),
                ),
                "s_channel": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                "codebook": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                "correction": ParamConstraint(dtypes=floats),
                "bias": ParamConstraint(dtypes=floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "out_dtype": ParamConstraint(dtypes=floats),
                "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
                "input_act_weight": ParamConstraint(dtypes=floats),
                "input_act_eps": ParamConstraint(dtypes=frozenset({float, int})),
                "residual": ParamConstraint(dtypes=floats),
                "residual_scale": ParamConstraint(dtypes=floats),
            },
            default_devices=xpu,
            call_rules=(_w4a8_native_call_rule,),
        )
    if _SVDQ_AVAILABLE:
        capabilities.update(
            {
                "quantize_svdquant_w4a4": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                        "smooth": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                        "lora_down": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                        "pad_size": ParamConstraint(dtypes=frozenset({int})),
                        "act_unsigned": ParamConstraint(dtypes=frozenset({bool})),
                        "lora_x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                    },
                    default_devices=xpu,
                ),
                "scaled_mm_svdquant_w4a4": FunctionConstraints(
                    params={
                        "act": ParamConstraint(
                            dtypes=frozenset({torch.int8, torch.uint8}), shape_rules=(ExactDims(2),)
                        ),
                        "wgt": ParamConstraint(dtypes=frozenset({torch.int8, torch.uint8})),
                        "ascales": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                        "wscales": ParamConstraint(dtypes=floats),
                        "lora_act_in": ParamConstraint(
                            dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(2),)
                        ),
                        "lora_up": ParamConstraint(dtypes=floats),
                        "bias": ParamConstraint(dtypes=floats),
                        "act_unsigned": ParamConstraint(dtypes=frozenset({bool})),
                    },
                    default_devices=xpu,
                ),
            }
        )
    if _SVDQ_W4A16_AVAILABLE:
        capabilities["svdquant_w4a16_linear"] = FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "packed_u4": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2),),
                ),
                "scales_f16": ParamConstraint(
                    dtypes=frozenset({torch.float16}),
                    shape_rules=(ExactDims(2),),
                ),
                "rcp_smooth_f16": ParamConstraint(
                    dtypes=frozenset({torch.float16}),
                    shape_rules=(ExactDims(1),),
                ),
                "lora_down": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "lora_up": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "bias": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(1),),
                ),
                "output_dtype_code": ParamConstraint(
                    dtypes=frozenset({int}),
                ),
            },
            default_devices=xpu,
        )
    if _NORM_AVAILABLE:
        capabilities["adaln"] = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "scale": ParamConstraint(dtypes=floats),
                "shift": ParamConstraint(dtypes=floats),
            },
            default_devices=xpu,
        )
        if _native_norm is not None and hasattr(_native_norm, "fused_rms_adaln"):
            capabilities["rms_adaln"] = FunctionConstraints(
                params={
                    "x": ParamConstraint(dtypes=floats),
                    "scale": ParamConstraint(dtypes=floats),
                    "shift": ParamConstraint(dtypes=floats),
                },
                default_devices=xpu,
            )
    if _GROUP_NORM_STATS3D_AVAILABLE:
        capabilities["group_norm_stats3d"] = FunctionConstraints(
            params={"x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(5),))},
            default_devices=xpu,
        )
    if _GROUP_NORM_SILU_PAD3D_AVAILABLE:
        capabilities["group_norm_silu_pad3d"] = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(5),)),
                "weight": ParamConstraint(dtypes=floats),
                "bias": ParamConstraint(dtypes=floats),
                "stats": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=xpu,
        )
        if _GROUP_NORM_SILU_PAD3D_OUT_AVAILABLE:
            capabilities["group_norm_silu_pad3d_out"] = with_out_param(
                capabilities["group_norm_silu_pad3d"]
            )
    if _FP16_LINEAR_AVAILABLE:
        capabilities["fp16_linear"] = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=frozenset({torch.float16})),
                "weight": ParamConstraint(dtypes=frozenset({torch.float16})),
                "bias": ParamConstraint(dtypes=floats),
                "residual": ParamConstraint(dtypes=floats),
                "residual_scale": ParamConstraint(dtypes=floats),
            },
            default_devices=xpu,
        )
    if _FP16_CONV3D_AVAILABLE:
        capabilities["fp16_conv3d"] = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=frozenset({torch.float16}), shape_rules=(ExactDims(5),)),
                "weight": ParamConstraint(dtypes=frozenset({torch.float16}), shape_rules=(ExactDims(5),)),
                "bias": ParamConstraint(dtypes=floats),
                "residual": ParamConstraint(dtypes=floats),
            },
            default_devices=xpu,
        )
        if _FP16_CONV3D_OUT_AVAILABLE:
            capabilities["fp16_conv3d_out"] = with_out_param(
                capabilities["fp16_conv3d"]
            )
    if _AWQ_W4A16_AVAILABLE:
        low_precision = frozenset({torch.float16, torch.bfloat16})
        capabilities["gemv_awq_w4a16"] = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=low_precision),
                "qweight": int8_2d,
                "wscales": ParamConstraint(
                    dtypes=low_precision, shape_rules=(ExactDims(2),),
                ),
                "wzeros": ParamConstraint(
                    dtypes=low_precision, shape_rules=(ExactDims(2),),
                ),
                "bias": ParamConstraint(dtypes=low_precision),
            },
            default_devices=xpu,
        )
    if _FP8_QDQ_AVAILABLE:
        fp8_dtypes = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
        capabilities.update(
            {
                "quantize_per_tensor_fp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats),
                        "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                        "output_type": ParamConstraint(dtypes=fp8_dtypes),
                    },
                    default_devices=xpu,
                ),
                "dequantize_per_tensor_fp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=fp8_dtypes),
                        "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                        "output_type": ParamConstraint(dtypes=floats),
                    },
                    default_devices=xpu,
                ),
                "stochastic_rounding_fp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats),
                        "rng": ParamConstraint(dtypes=frozenset({torch.uint8})),
                        "output_type": ParamConstraint(dtypes=fp8_dtypes),
                    },
                    default_devices=xpu,
                ),
            }
        )
    if _GGUF_AVAILABLE:
        capabilities["dequantize_gguf"] = FunctionConstraints(
            params={
                "data": ParamConstraint(dtypes=frozenset({torch.uint8})),
                "quant_type_code": ParamConstraint(dtypes=frozenset({int})),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
                "layout_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        )
    if _ROPE_AVAILABLE:
        rope_input = ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),))
        rope_freqs = ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),))
        capabilities.update(
            {
                "apply_rope1": FunctionConstraints(
                    params={"x": rope_input, "freqs_cis": rope_freqs},
                    default_devices=xpu,
                ),
                "apply_rope": FunctionConstraints(
                    params={"xq": rope_input, "xk": rope_input, "freqs_cis": rope_freqs},
                    default_devices=xpu,
                ),
                "apply_rope_split_half1": FunctionConstraints(
                    params={"x": rope_input, "freqs_cis": rope_freqs},
                    default_devices=xpu,
                ),
                "apply_rope_split_half": FunctionConstraints(
                    params={"xq": rope_input, "xk": rope_input, "freqs_cis": rope_freqs},
                    default_devices=xpu,
                ),
            }
        )
        for inplace_name, functional_name in {
            "apply_rope_": "apply_rope",
            "apply_rope1_": "apply_rope1",
            "apply_rope_split_half_": "apply_rope_split_half",
            "apply_rope_split_half1_": "apply_rope_split_half1",
        }.items():
            capabilities[inplace_name] = capabilities[functional_name]
    if _RMS_ROPE_AVAILABLE:
        rope_input = ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),))
        rope_freqs = ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),))
        rope_scale = ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),))
        capabilities.update(
            {
                "rms_rope1": FunctionConstraints(
                    params={
                        "x": rope_input,
                        "freqs_cis": rope_freqs,
                        "scale": rope_scale,
                    },
                    default_devices=xpu,
                ),
                "rms_rope": FunctionConstraints(
                    params={
                        "q": rope_input,
                        "k": rope_input,
                        "freqs_cis": rope_freqs,
                        "q_scale": rope_scale,
                        "k_scale": rope_scale,
                    },
                    default_devices=xpu,
                ),
                "rms_rope_split_half1": FunctionConstraints(
                    params={
                        "x": rope_input,
                        "freqs_cis": rope_freqs,
                        "scale": rope_scale,
                    },
                    default_devices=xpu,
                ),
                "rms_rope_split_half": FunctionConstraints(
                    params={
                        "q": rope_input,
                        "k": rope_input,
                        "freqs_cis": rope_freqs,
                        "q_scale": rope_scale,
                        "k_scale": rope_scale,
                    },
                    default_devices=xpu,
                ),
            }
        )
        for inplace_name, functional_name in {
            "rms_rope_": "rms_rope",
            "rms_rope1_": "rms_rope1",
            "rms_rope_split_half_": "rms_rope_split_half",
            "rms_rope_split_half1_": "rms_rope_split_half1",
        }.items():
            capabilities[inplace_name] = capabilities[functional_name]
    if _CONVROT_NATIVE_AVAILABLE and _SVDQ_AVAILABLE:
        capabilities.update(
            {
                "quantize_convrot_w4a4_weight": FunctionConstraints(
                    params={
                        "weight": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                        "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                        "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=xpu,
                ),
                "dequantize_convrot_w4a4_weight": FunctionConstraints(
                    params={
                        "qdata": ParamConstraint(
                            dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                        ),
                        "scales": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                        "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                        "output_dtype": ParamConstraint(dtypes=floats),
                    },
                    default_devices=xpu,
                ),
                "convrot_w4a4_linear": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats),
                        "qweight": ParamConstraint(
                            dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                        ),
                        "wscales": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                        "bias": ParamConstraint(dtypes=floats),
                        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                        "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                        "linear_dtype": ParamConstraint(dtypes=frozenset({str})),
                    },
                    default_devices=xpu,
                ),
                "prepare_int4_weight_for_int8_linear": FunctionConstraints(
                    params={
                        "weight": ParamConstraint(
                            dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                        )
                    },
                    default_devices=xpu,
                ),
                "quantize_and_rotate_rowwise": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats),
                        "H": ParamConstraint(dtypes=floats),
                        "group_size": ParamConstraint(dtypes=frozenset({int})),
                        "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=xpu,
                ),
            }
        )
    if _SOL_AVAILABLE:
        capabilities["sol_attn"] = FunctionConstraints(
            params={name: ParamConstraint(
                dtypes=frozenset({torch.bfloat16, torch.float16}),
                shape_rules=(ExactDims(4),),
            ) for name in ("q", "k", "v")},
            default_devices=xpu,
            call_rules=(sol_attn_common_call_rule,),
        )
    return capabilities


def _register() -> None:
    if not _AVAILABLE:
        registry.mark_unavailable("xpu", _ERROR or "omni_xpu_kernel is not available")
        return
    capabilities = _build_constraints()
    if not capabilities:
        registry.mark_unavailable(
            "xpu",
            _INT8_ERROR or "omni_xpu_kernel exposes no supported Kitchen capabilities",
        )
        return
    registry.register(
        name="xpu",
        module=sys.modules[__name__],
        capabilities=capabilities,
    )


_register()
