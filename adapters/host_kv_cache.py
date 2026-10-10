"""Refuse host-side KV/prefix caches on builds that cannot unmap host memory.

ComfyUI keeps the step-independent Qwen-Image 2.1 prefix K/V in a
``PoseBranchCache`` that may live on the host: ``QwenImage21Cache`` with
``device="cpu"``, or ``device="auto"`` falling back to host under VRAM
pressure.

Releasing that host side goes through ``comfy.model_management.unpin_memory``,
which calls ``torch.cuda.cudart().cudaHostUnregister``.  ``MAX_PINNED_MEMORY``
is only assigned for CUDA/ROCm, so on Intel XPU (and with
``--disable-pinned-memory`` anywhere) ``pin_memory``/``unpin_memory`` are
no-ops and the driver keeps the host pages mapped in the process' GPU address
space: WDDM "Shared GPU memory" retains the full cache size until ComfyUI is
restarted.  Measured on Arc A770 16GB, Qwen-Image 2.1 edit, 1024x1024:
~1.0-1.8 GB retained, unchanged by ``/free {unload_models, free_memory}`` or
``torch.xpu.empty_cache``, and it reproduces with comfy-aimdo disabled and
with every ComfyUI-OmniXPU patch disabled (upstream Comfy-Org/ComfyUI#16463).

The recompute path is the documented fallback when the cache cannot be
allocated, and it retains nothing, so this adapter makes such builds take it:

  - Qwen-Image 2.1: the host store is refused; a device-side store (``gpu``,
    or ``auto`` while VRAM is spare) is still used.  With ``device="cpu"`` the
    cache effectively behaves like ``device="off"``.

``WanAnimate2Cache`` (``device="cpu"`` by default) shares the same
``PoseBranchCache`` and the same class of leak, but it is not guarded here:
this machine has no Wan-Animate2 checkpoint to verify against.

Anything unexpected inside the guard falls through to the upstream call, so
the adapter cannot fail a sampling run.

Env ``OMNIXPU_HOST_KV_CACHE``: ``1`` (default) guard on, ``0`` upstream
behaviour (host cache, retained mapping).
"""

import logging
import os

log = logging.getLogger("ComfyUI-OmniXPU")


def host_cache_releasable() -> bool:
    """True when host pages used for a KV cache can be unregistered later."""

    try:
        import comfy.model_management as mm
    except Exception:  # pragma: no cover - depends on host ComfyUI
        return True
    return getattr(mm, "MAX_PINNED_MEMORY", -1) > 0


def _patch_qwen21() -> str:
    """Refuse the host prefix K/V store; keep the device-side one."""

    try:
        import comfy.ldm.qwen_image21.model as qwen_model
    except Exception as exc:  # pragma: no cover - depends on host ComfyUI
        return f"skipped (qwen_image21 unavailable: {exc})"

    cls = getattr(qwen_model, "QwenImage21Transformer2DModel", None)
    if cls is None or not hasattr(cls, "select_prefix_cache"):
        return "skipped (select_prefix_cache not found)"
    if getattr(cls.select_prefix_cache, "_omnixpu_host_guard", False):
        return "qwen21 guarded"

    original = cls.select_prefix_cache

    def select_prefix_cache(self, key, cache_bytes, device, options):
        try:
            if options.get("device") == "cpu" and not host_cache_releasable():
                return None, False
        except Exception:  # pragma: no cover - never break the forward pass
            return original(self, key, cache_bytes, device, options)
        cache, filled = original(self, key, cache_bytes, device, options)
        try:
            if (cache is not None and not host_cache_releasable()
                    and getattr(cache, "store_device", None) is not None
                    and cache.store_device.type == "cpu"):
                # `auto` fell back to host: drop it before any K/V is stored.
                cache.free()
                if self.prefix_cache is cache:
                    self.prefix_cache = None
                return None, False
        except Exception:  # pragma: no cover - keep upstream behaviour
            return cache, filled
        return cache, filled

    select_prefix_cache._omnixpu_host_guard = True
    cls.select_prefix_cache = select_prefix_cache
    return "qwen21 guarded"


def apply():
    if os.environ.get("OMNIXPU_HOST_KV_CACHE", "1") == "0":
        return False, "disabled by env"

    try:
        import comfy.model_management as mm
    except Exception as exc:  # pragma: no cover - depends on host ComfyUI
        return False, f"comfy.model_management unavailable: {exc}"
    # The retained mapping is an XPU-only observation.  Gate on the device so a
    # CUDA/ROCm run never changes behaviour, not even with
    # --disable-pinned-memory (which also leaves MAX_PINNED_MEMORY at -1).
    if getattr(mm.get_torch_device(), "type", None) != "xpu":
        return False, "requires XPU"
    if host_cache_releasable():
        return False, "host memory is registerable; upstream host cache kept"

    results = [_patch_qwen21()]
    guarded = [item for item in results if item.endswith("guarded")]
    if not guarded:
        return False, "; ".join(results)
    return (
        True,
        "host KV cache refused where host memory cannot be unregistered "
        "(Comfy-Org/ComfyUI#16463) - " + "; ".join(results),
    )
