"""A770 bridge: route MiniMax H3 fused RMSNorm+RoPE to omni_xpu_kernel.

The H3 DiT blocks call ``comfy_kitchen.rms_rope_split_half_`` for every
attention head norm+rope. On A770 the Kitchen eager path materializes the
normalized Q/K and applies RoPE with several elementwise passes (measured
4-11x slower than the omni generic fused kernel on the exact H3 contract).
This adapter keeps the generic Kitchen registry and torch custom ops intact,
and only replaces the public Python wrapper for the narrow H3 contract.
"""

from __future__ import annotations

import logging

import torch

from ..patches.debug import log_debug_event

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_rms_rope_original__"
_routed_calls = 0
_fallback_calls = 0
_failed_calls = 0
_logged_route_shapes = set()


def get_stats() -> dict:
    return {
        "routed": _routed_calls,
        "fallback": _fallback_calls,
        "failed": _failed_calls,
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
                log.warning(
                    "[OmniXPU] rms_rope native route failed, falling back: %s",
                    exc,
                )
        _fallback_calls += 1
        return original(q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim)

    setattr(routed, _PATCH_MARKER, original)
    return routed


def apply():
    """Install the A770/DG2 RMS-RoPE bridge on comfy_kitchen wrappers."""
    global _routed_calls, _fallback_calls, _failed_calls

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

    _routed_calls = 0
    _fallback_calls = 0
    _failed_calls = 0
    log.info("[OmniXPU] A770: installed rms_rope bridge (%s)", ", ".join(installed))
    return True, ""


__all__ = ["apply", "get_stats"]
