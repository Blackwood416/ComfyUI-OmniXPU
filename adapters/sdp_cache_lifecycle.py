"""Auto-release the SDP sidecar's packed Q/K/V USM cache.

The DG2 SDP sidecar allocates packed Q/K/V device buffers with
``sycl::aligned_alloc_device`` and caches the most recent shape (~0.2 GB at
seq=8192/H32, up to ~0.7-0.9 GB for H3-style seq=20685).  These allocations
are invisible to torch's caching allocator, so ComfyUI's model unload /
``torch.xpu.empty_cache`` cannot free them; without intervention they persist
until process exit or a manual ``OmniXPU Clear SDP Cache`` node runs.

This adapter hooks ComfyUI's model-memory lifecycle so the sidecar cache is
released when the model is truly unloaded:

  - ``model_management.unload_all_models`` (explicit unload / checkpoint swap)
    -> clears the sidecar cache by default.

``soft_empty_cache`` is deliberately NOT hooked by default: ComfyUI calls it
on every model load / GC, which used to clear the packed Q/K/V cache before
each new sampling run and cost a 2s+ repack on the first attention step.
Keep the cache across runs while the model stays resident.

Env ``OMNIXPU_SDP_CACHE_AUTOCLEAR``:
  - ``1`` (default): clear on ``unload_all_models``; keep across soft clears.
  - ``keep`` (or ``0``/``false``/``off``): keep the sidecar cache even across
    model unloads (retains ~0.2-0.9 GB invisible USM; opt-in).
  - ``aggressive`` (or ``soft``/``old``): clear on ``soft_empty_cache`` too.
The manual ``OmniXPU Clear SDP Cache`` node remains available for on-demand
control.
"""

import logging
import os

log = logging.getLogger("ComfyUI-OmniXPU")


def apply():
    enabled = os.environ.get("OMNIXPU_SDP_CACHE_AUTOCLEAR", "1").strip().lower()
    aggressive = enabled in ("aggressive", "soft", "old")
    keep = enabled in ("keep", "0", "false", "no", "off")
    if keep:
        return False, "OMNIXPU_SDP_CACHE_AUTOCLEAR=keep (hooks disabled, cache retained)"

    try:
        import comfy.model_management as mm
        from omni_xpu_kernel import sdp as omni_sdp
    except Exception as exc:  # pragma: no cover - depends on host ComfyUI
        return False, f"imports unavailable: {exc}"

    if not hasattr(omni_sdp, "clear_cache"):
        return False, "omni_xpu_kernel.sdp.clear_cache unavailable"

    orig_soft = mm.soft_empty_cache
    orig_unload = mm.unload_all_models
    _diag_count = [0]
    _unload_count = [0]

    def _clear_sidecar():
        try:
            omni_sdp.clear_cache()
        except Exception:
            # Device may be lost or the sidecar unloaded; never break the
            # host memory-management path.
            log.debug("sdp.clear_cache failed", exc_info=True)

    def soft_empty_cache(force=False):
        # Default: do NOT clear the SDP sidecar cache here. ComfyUI calls
        # soft_empty_cache on every model load/GC; clearing there forces a
        # full Q/K/V repack on the first attention of the next run (2s+).
        # Only the aggressive opt-in restores the old behavior.
        _diag_count[0] += 1
        if _diag_count[0] <= 5:
            log.info(
                "[OmniXPU] sdp_cache soft_empty_cache #%d (clear_sidecar=%s)",
                _diag_count[0], aggressive,
            )
        if aggressive:
            _clear_sidecar()
        return orig_soft(force)

    def unload_all_models():
        _unload_count[0] += 1
        log.info(
            "[OmniXPU] sdp_cache unload_all_models #%d (clear_sidecar=%s)",
            _unload_count[0], aggressive,
        )
        orig_unload()
        # 默认在真实卸载时清理（不驻留不可见 USM）；OMNIXPU_SDP_CACHE_AUTOCLEAR
        # =keep 才跨卸载保留（下一次 Queue 首步省一次 2s+ repack，opt-in）。
        if not keep:
            _clear_sidecar()

    mm.soft_empty_cache = soft_empty_cache
    mm.unload_all_models = unload_all_models
    return (
        True,
        "hooked model lifecycle -> sidecar cache kept across soft clears, "
        "cleared on unload by default (OMNIXPU_SDP_CACHE_AUTOCLEAR=keep to "
        "retain across unloads; =aggressive to clear on soft_empty_cache too)",
    )
