"""Stage large SeedVR2 VAE decode results on CPU when a single XPU
allocation would exceed the platform limit.

Measured on A770 + torch 2.13: a single ``urUSMDeviceAlloc`` fails above
4 GiB even with 14+ GiB free.  SeedVR2's ``tiled_vae`` allocates the full
FP32 decode result in one tensor, so high-resolution/long videos OOM at
``torch.zeros(...)`` regardless of free memory.  This fix keeps the result,
blend weights, and counter on CPU when the output exceeds a threshold, and
returns the CPU tensor (the video save path accepts it) instead of moving it
back to XPU.

The rewrite is source guarded: it only changes the exact known
``storage_device`` and return lines and declines unknown implementations.
"""

from __future__ import annotations

import functools
import inspect
import textwrap

_DEFAULT_THRESHOLD_BYTES = int(3.5 * 1024**3)
_PATCH_MARKER = "_omnixpu_seedvr_vae_decode_patched"

_STORAGE_LINE = "    storage_device = vae_model.device"
_STORAGE_REPLACEMENT = """    storage_device = vae_model.device
    _omni_cpu_staged = False
    if not encode and x.device.type == "xpu":
        _omni_output_elements = (
            int(x.shape[0]) * 3 * int(target_d) * int(target_h) * int(target_w)
        )
        if _omni_output_elements * 4 > _OMNIXPU_SEEDVR_VAE_CPU_STAGE_BYTES:
            storage_device = torch.device("cpu")
            _omni_cpu_staged = True
"""

_TAIL_CONTRACT = """    result.div_(count.clamp(min=1e-6))

    if result.device != x.device or result.dtype != x.dtype:
        result = result.to(device=x.device, dtype=x.dtype)
"""
_TAIL_REPLACEMENT = """    result.div_(count.clamp(min=1e-6))

    if _omni_cpu_staged:
        if result.dtype != x.dtype:
            result = result.to(dtype=x.dtype)
    elif result.device != x.device or result.dtype != x.dtype:
        result = result.to(device=x.device, dtype=x.dtype)
"""


def _rewrite_tiled_vae(tiled_vae, threshold_bytes):
    if tiled_vae.__code__.co_freevars:
        return None, "tiled_vae has unsupported closure state"

    try:
        source = textwrap.dedent(inspect.getsource(tiled_vae))
    except (OSError, TypeError) as exc:
        return None, f"tiled_vae source unavailable: {exc}"

    if source.count(_STORAGE_LINE) != 1:
        return None, "unsupported tiled_vae storage_device contract"
    if source.count(_TAIL_CONTRACT) != 1:
        return None, "unsupported tiled_vae return contract"

    source = source.replace(_STORAGE_LINE, _STORAGE_REPLACEMENT)
    source = source.replace(_TAIL_CONTRACT, _TAIL_REPLACEMENT)

    namespace = {}
    filename = inspect.getsourcefile(tiled_vae) or "<seedvr_vae_decode.py>"
    tiled_vae.__globals__["_OMNIXPU_SEEDVR_VAE_CPU_STAGE_BYTES"] = (
        threshold_bytes
    )
    exec(  # noqa: S102 - exact, source-guarded upstream function rewrite
        compile(
            source,
            f"{filename}:OmniXPU-SeedVR-VAE-decode",
            "exec",
        ),
        tiled_vae.__globals__,
        namespace,
    )
    patched = namespace.get(tiled_vae.__name__)
    if patched is None:
        return None, "rewritten tiled_vae was not defined"
    functools.update_wrapper(patched, tiled_vae)
    return patched, None


def apply():
    try:
        from comfy.ldm.seedvr import vae as seedvr_vae
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy.ldm.seedvr", "comfy.ldm.seedvr.vae"}:
            return False, "ComfyUI SeedVR2 VAE is not available"
        raise

    tiled_vae = getattr(seedvr_vae, "tiled_vae", None)
    if tiled_vae is None or not callable(tiled_vae):
        return False, "ComfyUI SeedVR2 tiled_vae is not available"
    if getattr(tiled_vae, _PATCH_MARKER, False):
        return False, "SeedVR2 VAE decode fix is already applied"

    import os

    raw = os.environ.get("OMNIXPU_SEEDVR_VAE_CPU_STAGE_BYTES", "")
    try:
        threshold = int(raw) if raw else _DEFAULT_THRESHOLD_BYTES
    except ValueError:
        return False, "OMNIXPU_SEEDVR_VAE_CPU_STAGE_BYTES must be an int"
    if threshold <= 0:
        return False, "OMNIXPU_SEEDVR_VAE_CPU_STAGE_BYTES must be positive"

    patched, reason = _rewrite_tiled_vae(tiled_vae, threshold)
    if patched is None:
        return False, reason

    seedvr_vae.tiled_vae = patched
    setattr(patched, _PATCH_MARKER, True)
    return True, None


__all__ = ["apply"]
