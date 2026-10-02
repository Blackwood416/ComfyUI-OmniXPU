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


# kernel 已有、comfy_kitchen 也有同名算子 → 注册 XPU 实现。
# 每项：comfy_kitchen 的参数顺序，以及 kernel 调用怎么取这些参数。
_KERNEL_KITCHEN_OPS = {
    "gemv_awq_w4a16": (
        ("x", "qweight", "wscales", "wzeros", "bias", "group_size"),
        lambda k, c: k.gemv_awq_w4a16(
            c["x"], c["qweight"], c["wscales"], c["wzeros"],
            c.get("bias"), c.get("group_size", 64)),
    ),
    "fp16_conv3d": (
        ("x", "weight", "bias", "residual", "stride"),
        lambda k, c: k.fp16_conv3d(
            c["x"], c["weight"], c.get("bias"), c.get("residual"),
            c.get("stride", [1, 1, 1])),
    ),
    "fp16_conv3d_out": (
        ("x", "weight", "bias", "residual", "stride", "out"),
        lambda k, c: k.fp16_conv3d_out(
            c["x"], c["weight"], c.get("bias"), c.get("residual"),
            c.get("stride", [1, 1, 1]), c["out"]),
    ),
    "group_norm_silu_pad3d": (
        ("x", "weight", "bias", "num_groups", "eps", "pad", "silu", "zero_pad"),
        lambda k, c: k.group_norm_silu_pad3d(
            c["x"], c.get("weight"), c.get("bias"), c["num_groups"], c["eps"],
            c["pad"], c["silu"], c.get("zero_pad", False)),
    ),
    "group_norm_silu_pad3d_out": (
        ("x", "weight", "bias", "num_groups", "eps", "pad", "silu", "zero_pad", "out"),
        lambda k, c: k.group_norm_silu_pad3d_out(
            c["x"], c.get("weight"), c.get("bias"), c["num_groups"], c["eps"],
            c["pad"], c["silu"], c.get("zero_pad", False), c["out"]),
    ),
}


# output_dtype_code → torch.dtype，apply() 里填（comfy_kitchen 与 kernel 各用一套命名）
_DTYPE_CODE_TO_DTYPE = {}


# kernel.int8 已有、comfy_kitchen 也有同名算子的量化/反量化
_KERNEL_INT8_OPS = {
    "quantize_int8_rowwise": (
        ("x",),
        lambda i8, c: i8.quantize_int8_rowwise(c["x"]),
    ),
    "quantize_int8_tensorwise": (
        ("x",),
        lambda i8, c: i8.quantize_int8_tensorwise(c["x"], None, 0),
    ),
    "quantize_int8_convrot_weight": (
        ("weight", "group_size"),
        lambda i8, c: i8.quantize_int8_convrot_weight(
            c["weight"], c.get("group_size", 256)),
    ),
    "dequantize_int8_convrot_weight": (
        ("q", "scale", "group_size"),
        lambda i8, c: i8.dequantize_int8_convrot_weight(
            c["q"], c["scale"], c.get("group_size", 256)),
    ),
    "dequantize_int8_simple": (
        ("q", "scale"),
        lambda i8, c: i8.dequantize_int8_simple(c["q"], c["scale"]),
    ),
    "dequantize_int8_simple_dtype": (
        ("q", "scale", "output_dtype_code"),
        lambda i8, c: i8.dequantize_int8_simple_dtype(
            c["q"], c["scale"],
            _DTYPE_CODE_TO_DTYPE.get(c["output_dtype_code"], torch.bfloat16)),
    ),
}


def _register_ops(table, module):
    """把 kernel 已有的算子接到 comfy_kitchen 的 XPU 实现上。"""
    from comfy_kitchen.registry import registry

    registered = []
    for op_name, (schema, call) in table.items():
        operator = "comfy_kitchen::" + op_name
        if os.environ.get("OMNIXPU_KITCHEN_" + op_name.upper(), "1") == "0":
            continue
        if not hasattr(torch.ops, "comfy_kitchen") or not hasattr(
            torch.ops.comfy_kitchen, op_name
        ):
            continue
        if torch._C._dispatch_has_kernel_for_dispatch_key(operator, "XPU"):
            continue

        def _make_impl(op_name=op_name, schema=schema, call=call):
            @torch.library.impl("comfy_kitchen::" + op_name, "XPU")
            def _impl(*args, **kwargs):
                inputs = dict(zip(schema, args))
                inputs.update(kwargs)
                try:
                    return call(module, inputs)
                except Exception as error:
                    if is_fatal_accelerator_error(error):
                        raise
                    log.warning(
                        "[OmniXPU] %s native route failed, falling back: %s",
                        op_name, error,
                    )
                return registry.get_implementation(op_name, kwargs=inputs)(**inputs)

            return _impl

        _make_impl()
        registered.append(op_name)
    return registered


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

    from omni_xpu_kernel import kitchen as omni_kitchen

    _DTYPE_CODE_TO_DTYPE.update(DTYPE_CODE_TO_DTYPE)
    convrot_registered = _register_convrot_dequant(omni_int8)
    kitchen_ops = _register_ops(_KERNEL_KITCHEN_OPS, omni_kitchen)
    kitchen_ops += _register_ops(_KERNEL_INT8_OPS, omni_int8)
    if kitchen_ops:
        log.info("[OmniXPU] A770: registered Kitchen XPU ops: %s",
                 ", ".join(kitchen_ops))
    operator = "comfy_kitchen::int8_linear"
    if not hasattr(torch.ops, "comfy_kitchen") or not hasattr(
        torch.ops.comfy_kitchen, "int8_linear"
    ):
        return convrot_registered, f"{operator} not registered"
    if torch._C._dispatch_has_kernel_for_dispatch_key(operator, "XPU"):
        return convrot_registered or bool(kitchen_ops), (
            "Kitchen XPU backend already owns int8_linear"
        )

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
            "input_act_weight",
            "input_act_eps",
            "residual",
            "residual_scale",
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
        input_act_weight=None,
        input_act_eps=0.0,
        residual=None,
        residual_scale=None,
    ):
        out_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
        try:
            from omni_xpu_kernel import _load_extension
            native_int8 = _load_extension().int8
        except Exception:
            native_int8 = None

        # 融合残差：kernel 先做 rowwise 量化（必要时先旋转），残差在 GEMM 尾部累加。
        if native_int8 is not None and residual is not None and residual_scale is not None:
            try:
                x2 = x.reshape(-1, x.shape[-1])
                if convrot:
                    x2 = omni_int8.rotate_convrot(x2, convrot_groupsize)
                x8, x_scale = omni_int8.quantize_int8_rowwise(x2)
                result = native_int8.int8_linear_prequantized_residual(
                    x8, x_scale, weight, weight_scale, bias, output_dtype_code,
                    residual.reshape(-1, residual.shape[-1]), residual_scale,
                )
                return result.reshape(*x.shape[:-1], result.shape[-1]).to(out_dtype)
            except Exception as error:
                if is_fatal_accelerator_error(error):
                    raise
                log.warning(
                    "[OmniXPU] int8 residual native route failed, falling back: %s",
                    error,
                )

        # rms_norm 输入激活：kernel 一条 kernel 完成 norm(+ConvRot)+量化。
        if (
            native_int8 is not None
            and input_act == "rms_norm"
            and input_act_weight is not None
            and input_act_eps not in (0.0, None)
        ):
            try:
                if convrot:
                    x8, x_scale = omni_kitchen.rms_norm_convrot_quantize_int8(
                        x, input_act_weight, input_act_eps, convrot_groupsize,
                    )
                else:
                    x8, x_scale = omni_kitchen.rms_norm_quantize_int8(
                        x, input_act_weight, input_act_eps,
                    )
                result = native_int8.int8_linear_prequantized(
                    x8, x_scale, weight, weight_scale, bias, output_dtype_code,
                )
                return result.reshape(*x.shape[:-1], result.shape[-1]).to(out_dtype)
            except Exception as error:
                if is_fatal_accelerator_error(error):
                    raise
                log.warning(
                    "[OmniXPU] int8 rms_norm native route failed, falling back: %s",
                    error,
                )

        # 其余进阶参数不猜语义，交回 Kitchen 自己的实现。
        if (
            input_act_weight is not None
            or input_act_eps not in (0.0, None)
            or residual is not None
            or residual_scale is not None
        ):
            from comfy_kitchen.registry import registry

            kwargs = dict(
                x=x, weight=weight, weight_scale=weight_scale, bias=bias,
                output_dtype_code=output_dtype_code, convrot=convrot,
                convrot_groupsize=convrot_groupsize, input_act=input_act,
                input_act_weight=input_act_weight, input_act_eps=input_act_eps,
                residual=residual, residual_scale=residual_scale,
            )
            return registry.get_implementation("int8_linear", kwargs=kwargs)(**kwargs)
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
