"""Release AIMDO cast-buffer aliases before the boundary that unmaps them.

ComfyUI destroys the AIMDO cast buffer at every node boundary. execution.py
runs cleanup_prefetch_queues() and then reset_cast_buffers(), and the latter
clears STREAM_AIMDO_CAST_BUFFERS; that drops the last VRAMBuffer reference,
whose __del__ calls vrambuf_destroy() and unmaps the VA (src/vrambuf.c).

The ComfyUI cleanup only visits modules still reachable from PREFETCH_QUEUES.
A module whose _prefetch survived that list keeps a bare-pointer tensor
(ops.py -> aimdo_to_tensor -> xfer_dest) pointing into the range just
unmapped, and LowVramPatch.prepared_patches holds views derived the same way.
The next round hands one of those views to host_buffer.read_file_to_device and
the driver faults while submitting the H2D copy: ze_intel_gpu64.dll c0000005
at +0x308b4e. MiniMax H3 round 2 is the standing reproducer.

This adapter sweeps every module of every loaded model at that boundary and
releases the residual state with the same call ComfyUI itself uses, before the
unmaps run. It allocates nothing and issues no GPU work, so it is safe to
leave enabled. Set OMNIXPU_CAST_LIFETIME_TRACE=1 to log the counts.
"""

from __future__ import annotations

import functools
import logging
import os


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "_omnixpu_cast_buffer_lifetime"
_TRACE_ENV = "OMNIXPU_CAST_LIFETIME_TRACE"
_DISABLED = ("", "0", "false", "no", "off")

_stats = {"boundaries": 0, "prefetch": 0, "prepared": 0, "block_faulted": 0}


def _trace_enabled() -> bool:
    return os.environ.get(_TRACE_ENV, "").strip().lower() not in _DISABLED


def _loaded_models(model_management):
    """Inner nn.Modules of every live loaded model."""
    out = []
    for loaded in list(model_management.current_loaded_models):
        patcher = getattr(loaded, "model", None)
        if patcher is None:
            continue
        try:
            dead = patcher.is_dead()
        except Exception as exc:
            # A patcher that cannot report liveness is still worth sweeping:
            # skipping it would leave exactly the aliases this guard removes.
            log.debug("[OmniXPU] cast buffer sweep: liveness check failed: %r", exc)
            dead = False
        if dead:
            continue
        inner = getattr(patcher, "model", None)
        if inner is not None:
            out.append(inner)
    return out


def _has_prepared(module):
    for key in ("weight", "bias"):
        fn = getattr(module, key + "_lowvram_function", None)
        if fn is not None and getattr(fn, "prepared_patches", None) is not None:
            return True
    return False


def _stale(module):
    """True when the module still aliases AIMDO memory ComfyUI is about to unmap."""
    return (
        getattr(module, "_prefetch", None) is not None
        or _has_prepared(module)
        or getattr(module, "_v_block_faulted", False)
    )


def _release_module(model_prefetch, module):
    """Release one module's residual aliases, if any.

    Uses ComfyUI's own cleanup_prefetched_modules wherever it applies, so the
    unpin semantics stay exactly ComfyUI's, and fills its two gaps:

      * its loop skips a module once _prefetch is gone, so a surviving
        prepared_patches set is never cleared; clear_prepared covers that.
      * _v_block_faulted is read off its first argument, which is the block
        root rather than the _v-bearing children; the root is visited in its
        own right by the caller, so the block branch still runs.

    Every branch is guarded the same way ComfyUI guards it, so a second visit
    after a successful release is a no-op and nothing is unpinned twice.
    """
    had_prefetch = getattr(module, "_prefetch", None) is not None
    had_prepared = _has_prepared(module)
    had_block = bool(getattr(module, "_v_block_faulted", False))
    if not (had_prefetch or had_prepared or had_block):
        return 0, 0, 0

    try:
        if had_prefetch:
            # Official path: unpins on a signature, drops _prefetch.
            model_prefetch.cleanup_prefetched_modules(module, [module])
        else:
            # Gap 1: prepared patches with no _prefetch left.
            for key in ("weight", "bias"):
                fn = getattr(module, key + "_lowvram_function", None)
                if fn is not None:
                    fn.clear_prepared()
            # Gap 2: the block branch, reached with an empty module list.
            if had_block:
                model_prefetch.cleanup_prefetched_modules(module, [])
    except Exception as exc:
        log.warning("[OmniXPU] cast buffer release failed for %r: %r", module, exc)
        return 0, 0, 0

    released_prefetch = int(had_prefetch and getattr(module, "_prefetch", None) is None)
    released_prepared = int(had_prepared and not _has_prepared(module))
    released_block = int(had_block and not getattr(module, "_v_block_faulted", False))

    _stats["prefetch"] += released_prefetch
    _stats["prepared"] += released_prepared
    _stats["block_faulted"] += released_block
    return released_prefetch, released_prepared, released_block


def install(model_management, model_prefetch) -> bool:
    original = model_management.reset_cast_buffers
    if getattr(original, _PATCH_MARKER, False):
        return False

    @functools.wraps(original)
    def reset_cast_buffers():
        # Order is the whole point: release the aliases while the VA is still
        # mapped, then let the original run its own reset, which unmaps it.
        released = [0, 0, 0]
        try:
            for inner in _loaded_models(model_management):
                for module in inner.modules():
                    pf, prepared, blocked = _release_module(model_prefetch, module)
                    released[0] += pf
                    released[1] += prepared
                    released[2] += blocked
        except Exception as exc:  # never block the boundary
            log.warning("[OmniXPU] cast buffer sweep failed: %r", exc)
        if any(released):
            _stats["boundaries"] += 1
            if _trace_enabled():
                log.info(
                    "[OmniXPU] cast buffer sweep: released %d residual _prefetch, "
                    "%d prepared patch set(s), %d block-fault flag(s) before the unmaps",
                    released[0],
                    released[1],
                    released[2],
                )
        return original()

    setattr(reset_cast_buffers, _PATCH_MARKER, True)
    model_management.reset_cast_buffers = reset_cast_buffers
    return True


def apply():
    try:
        import comfy.memory_management as memory_management
        import comfy.model_management as model_management
        import comfy.model_prefetch as model_prefetch
    except Exception as exc:
        return False, f"comfy modules unavailable: {exc}"

    # aimdo_enabled lives on comfy.memory_management; main.py sets it once
    # control.init_devices() succeeds. Without AIMDO there are no cast buffers
    # to guard, and reset_cast_buffers is a no-op.
    if not getattr(memory_management, "aimdo_enabled", False):
        return False, "AIMDO not enabled"

    if not install(model_management, model_prefetch):
        return False, "already installed"

    log.info("[OmniXPU] cast buffer lifetime guard applied")
    return True, None
