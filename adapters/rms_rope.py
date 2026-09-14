"""A770 bridges for measured MiniMax H3, Z-Image and Krea2 RMS-RoPE contracts.

The H3 DiT blocks call ``comfy_kitchen.rms_rope_split_half_`` for every
attention head norm+rope. On A770 the Kitchen eager path materializes the
normalized Q/K and applies RoPE with several elementwise passes (measured
4-11x slower than the omni generic fused kernel on the exact H3 contract).
This adapter keeps the generic Kitchen registry and torch custom ops intact,
and only replaces the public Python wrapper for the narrow H3 contract.
"""

from __future__ import annotations

import logging
import os

import torch

from ..patches.debug import log_debug_event
from .errors import is_fatal_accelerator_error

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_rms_rope_original__"
_routed_calls = 0
_fallback_calls = 0
_failed_calls = 0
_zimage_routed_calls = 0
_krea2_routed_calls = 0
_logged_route_shapes = set()


def get_stats() -> dict:
    return {
        "routed": _routed_calls,
        "fallback": _fallback_calls,
        "failed": _failed_calls,
        "zimage_routed": _zimage_routed_calls,
        "krea2_routed": _krea2_routed_calls,
    }


def _h3_contract(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None,
    rot_dim: int,
) -> bool:
    """Match the exact MiniMax H3 q/k RMSNorm+partial split-half RoPE call."""
    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
        return False
    if q.device.type != "xpu" or k.device != q.device:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype:
        return False
    if q.dim() != 4 or k.dim() != 4 or q.shape != k.shape:
        return False
    if q.shape[0] != 1 or q.shape[2] != 56 or q.shape[3] != 128:
        return False
    if rot_dim != 96:
        return False
    # q/k are views into a packed [S, 3*7168] qkv buffer.
    packed_stride = 3 * 7168
    if (
        q.stride(1) != packed_stride
        or q.stride(2) != 128
        or q.stride(3) != 1
        or k.stride(1) != packed_stride
        or k.stride(2) != 128
        or k.stride(3) != 1
    ):
        return False
    if (
        not isinstance(freqs_cis, torch.Tensor)
        or freqs_cis.device != q.device
        or freqs_cis.dtype != q.dtype
        or freqs_cis.dim() != 6
        or freqs_cis.shape[0] != 1
        or freqs_cis.shape[1] != q.shape[1]
        or freqs_cis.shape[2] != 1
        or freqs_cis.shape[3] != 48
        or tuple(freqs_cis.shape[4:]) != (2, 2)
        or not freqs_cis.is_contiguous()
    ):
        return False
    for scale in (q_scale, k_scale):
        if not isinstance(scale, torch.Tensor):
            return False
        if (
            scale.device != q.device
            or scale.dtype != q.dtype
            or scale.dim() != 1
            or scale.numel() != 128
            or not scale.is_contiguous()
        ):
            return False
    return True


def _make_router(omni_rotary, functional: bool):
    """Return a patched Kitchen wrapper that routes the H3 contract."""
    import comfy_kitchen as ck

    original_name = "rms_rope_split_half" if functional else "rms_rope_split_half_"
    original = getattr(ck, original_name)

    def routed(
        q,
        k,
        freqs_cis,
        q_scale,
        k_scale=None,
        epsilon=1e-6,
        rot_dim=0,
    ):
        global _routed_calls, _fallback_calls, _failed_calls
        if _h3_contract(q, k, freqs_cis, q_scale, k_scale, rot_dim):
            route_shape = (q.shape[1], q.stride(1), k.stride(1))
            if route_shape not in _logged_route_shapes:
                log.info(
                    "[OmniXPU] rms_rope: A770 H3 fused RMSNorm+RoPE route "
                    "q=%s freqs=%s rot_dim=%s",
                    tuple(q.shape),
                    tuple(freqs_cis.shape),
                    rot_dim,
                )
                _logged_route_shapes.add(route_shape)
            log_debug_event(
                "kernel",
                "rms_rope_split_half" if functional else "rms_rope_split_half_",
                {"q": q, "k": k, "freqs_cis": freqs_cis},
                details={"backend": "omni_dg2", "rot_dim": rot_dim},
            )
            try:
                if functional:
                    result = omni_rotary.rms_kitchen_rope_split_half(
                        q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim
                    )
                else:
                    result = omni_rotary.rms_kitchen_rope_split_half_(
                        q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim
                    )
                _routed_calls += 1
                return result
            except Exception as exc:  # keep the original route as the safety net
                _failed_calls += 1
                if is_fatal_accelerator_error(exc):
                    raise
                log.warning(
                    "[OmniXPU] rms_rope native route failed, falling back: %s",
                    exc,
                )
        _fallback_calls += 1
        return original(q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim)

    setattr(routed, _PATCH_MARKER, original)
    return routed


def _zimage_contract(q, k, freqs_cis, q_scale, k_scale):
    """The measured 1024-square Z-Image BF16 packed-QKV contract only."""
    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
        return False
    if q.device.type != "xpu" or k.device != q.device:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype:
        return False
    if tuple(q.shape) != (1, 4256, 30, 128) or k.shape != q.shape:
        return False
    packed_stride = (4256 * 11520, 11520, 128, 1)
    if tuple(q.stride()) != packed_stride or tuple(k.stride()) != packed_stride:
        return False
    if q.requires_grad or k.requires_grad:
        return False
    if not isinstance(freqs_cis, torch.Tensor) or (
        freqs_cis.device != q.device
        or freqs_cis.dtype != torch.float32
        or tuple(freqs_cis.shape) != (1, 4256, 1, 64, 2, 2)
        or not freqs_cis.is_contiguous()
        or freqs_cis.requires_grad
    ):
        return False
    for scale in (q_scale, k_scale):
        if not isinstance(scale, torch.Tensor) or (
            scale.device != q.device
            or scale.dtype != torch.bfloat16
            or tuple(scale.shape) != (128,)
            or not scale.is_contiguous()
            or scale.requires_grad
        ):
            return False
    return True


def _make_zimage_router(omni_rotary):
    import comfy_kitchen as ck

    original = ck.rms_rope

    def routed(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
        global _zimage_routed_calls, _fallback_calls, _failed_calls
        if _zimage_contract(q, k, freqs_cis, q_scale, k_scale):
            try:
                out = omni_rotary.rms_kitchen_rope(
                    q, k, freqs_cis, q_scale, k_scale, epsilon
                )
                if not _zimage_routed_calls:
                    log.info("[OmniXPU] A770 Z-Image fused RMS-RoPE: q=%s", tuple(q.shape))
                _zimage_routed_calls += 1
                return out
            except Exception as exc:
                _failed_calls += 1
                if is_fatal_accelerator_error(exc):
                    raise
                if _failed_calls == 1:
                    log.warning("[OmniXPU] Z-Image RMS-RoPE fallback: %s", exc)
        _fallback_calls += 1
        return original(q, k, freqs_cis, q_scale, k_scale, epsilon)

    setattr(routed, _PATCH_MARKER, original)
    return routed


def _krea2_input_contract(attention, x, freqs, transformer_options):
    """Measured Krea2 1024-square GQA, without intermediate attention hooks."""
    if not isinstance(x, torch.Tensor) or x.device.type != "xpu":
        return False
    if x.dtype != torch.bfloat16 or tuple(x.shape) != (1, 4192, 6144):
        return False
    if x.requires_grad or torch.is_grad_enabled():
        return False
    if (attention.heads, attention.kvheads, attention.headdim) != (48, 12, 128):
        return False
    if transformer_options.get("patches"):
        return False
    module_hooks = torch.nn.modules.module
    if module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks:
        return False
    if not isinstance(freqs, torch.Tensor) or (
        freqs.device != x.device or freqs.dtype != torch.float32
        or tuple(freqs.shape) != (1, 1, 4192, 64, 2, 2)
        or not freqs.is_contiguous()
        or freqs.requires_grad
    ):
        return False
    for module in (attention.qknorm, attention.qknorm.qnorm, attention.qknorm.knorm):
        if module._forward_hooks or module._forward_pre_hooks:
            return False
    for module in (attention.qknorm.qnorm, attention.qknorm.knorm):
        if (not isinstance(module.scale, torch.Tensor)
                or tuple(module.scale.shape) != (128,)
                or module.scale.dtype not in (torch.bfloat16, torch.float32)
                or module.eps != 1e-5):
            return False
    return True


def _make_krea2_router(omni_rotary, krea2_model):
    original = krea2_model.Attention.forward

    def routed(self, x, freqs=None, mask=None, transformer_options={}):
        global _krea2_routed_calls, _failed_calls
        if not _krea2_input_contract(self, x, freqs, transformer_options):
            return original(self, x, freqs, mask, transformer_options)

        # Keep the original projections, GQA repetition, gated attention and
        # output projection. Only the QK norm -> BF16 -> FP32 RoPE chain is
        # fused. Each projection still goes through Comfy's cast/LoRA owner.
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = q.view(1, 4192, 48, 128).transpose(1, 2)
        k = k.view(1, 4192, 12, 128).transpose(1, 2)
        v = v.view(1, 4192, 12, 128).transpose(1, 2)
        expected = ((25755648, 128, 6144, 1), (6438912, 128, 1536, 1))
        use_native = tuple(q.stride()) == expected[0] and tuple(k.stride()) == expected[1]
        if use_native:
            qscale = krea2_model.comfy.model_management.cast_to(
                self.qknorm.qnorm.scale, dtype=torch.float32, device=x.device) + 1.0
            kscale = krea2_model.comfy.model_management.cast_to(
                self.qknorm.knorm.scale, dtype=torch.float32, device=x.device) + 1.0
            try:
                q, k = omni_rotary.rms_kitchen_rope(q, k, freqs, qscale, kscale, 1e-5)
                if not _krea2_routed_calls:
                    log.info("[OmniXPU] A770 Krea2 fused RMS-RoPE: q=%s k=%s", tuple(q.shape), tuple(k.shape))
                _krea2_routed_calls += 1
            except Exception as exc:
                _failed_calls += 1
                if is_fatal_accelerator_error(exc):
                    raise
                log.warning("[OmniXPU] Krea2 RMS-RoPE fallback: %s", exc)
                use_native = False
        if not use_native:
            q, k = self.qknorm(q, k)
            q, k = krea2_model.apply_rope(q, k, freqs)
        k = k.repeat_interleave(4, dim=1)
        v = v.repeat_interleave(4, dim=1)
        out = krea2_model.optimized_attention_masked(
            q, k, v, self.heads, mask=mask, skip_reshape=True,
            transformer_options=transformer_options)
        return self.wo(out * torch.nn.functional.sigmoid(gate))

    setattr(routed, _PATCH_MARKER, original)
    return routed


def apply():
    """Install the A770/DG2 RMS-RoPE bridge on comfy_kitchen wrappers."""
    global _routed_calls, _fallback_calls, _failed_calls, _zimage_routed_calls, _krea2_routed_calls

    try:
        import omni_xpu_kernel as omni_package
        from omni_xpu_kernel import rotary as omni_rotary
    except ImportError:
        return False, "omni_xpu_kernel.rotary not available"
    if getattr(omni_package, "__xpu_target__", None) != "dg2":
        return False, "A770/DG2-only compatibility route"

    try:
        import comfy_kitchen as ck
    except ImportError:
        return False, "comfy_kitchen not available"

    if not callable(getattr(omni_rotary, "rms_kitchen_rope_split_half_", None)):
        return False, "omni rotary in-place split-half kernel not available"

    installed = []
    for functional in (False, True):
        name = "rms_rope_split_half" if functional else "rms_rope_split_half_"
        original = getattr(ck, name, None)
        if original is None or not callable(original):
            return False, f"comfy_kitchen.{name} not available"
        if getattr(original, _PATCH_MARKER, None) is not None:
            installed.append(name)
            continue
        patched = _make_router(omni_rotary, functional)
        setattr(ck, name, patched)
        installed.append(name)

    if os.environ.get("OMNIXPU_ZIMAGE_RMS_ROPE", "1") == "1":
        original = getattr(ck, "rms_rope", None)
        if callable(original) and callable(getattr(omni_rotary, "rms_kitchen_rope", None)):
            if getattr(original, _PATCH_MARKER, None) is None:
                ck.rms_rope = _make_zimage_router(omni_rotary)
            installed.append("rms_rope (Z-Image)")

    if (os.environ.get("OMNIXPU_KREA2_RMS_ROPE", "1") == "1"
            and callable(getattr(omni_rotary, "rms_kitchen_rope", None))):
        try:
            import comfy.ldm.krea2.model as krea2_model
        except ImportError:
            pass
        else:
            if getattr(krea2_model.Attention.forward, _PATCH_MARKER, None) is None:
                krea2_model.Attention.forward = _make_krea2_router(omni_rotary, krea2_model)
            installed.append("RMS-RoPE (Krea2)")

    _routed_calls = 0
    _fallback_calls = 0
    _failed_calls = 0
    _zimage_routed_calls = 0
    _krea2_routed_calls = 0
    log.info("[OmniXPU] A770: installed rms_rope bridge (%s)", ", ".join(installed))
    return True, ""


__all__ = ["apply", "get_stats"]
