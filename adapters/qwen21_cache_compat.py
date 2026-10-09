"""Qwen-Image 2.1 prefix cache: slot bookkeeping + hook safety on XPU.

Subset of upstream ``omni/ComfyUI-OmniXPU/adapters/qwen_image21_cache.py``
(introduced by llm-scaler #732) that does not depend on anything the A-series
install lacks.  Two independent pieces are ported, both XPU-generic:

1. ``PoseBranchCache.select`` slot bookkeeping - upstream replaces the
   by-value ``self.slots.remove(s)`` with ``enumerate`` + ``pop(index)``.
   Removing by value compares the slot dicts (which hold tensors), so with
   more than one slot it can pop the wrong one or raise.
2. ``_guard_prefix_cache`` - a cached step runs target rows only, so any hook
   attached to the graph would see a different sequence than the first step.
   Upstream wraps the Qwen2.1 diffusion model and disables the prefix cache
   for a forward whose ``transformer_options`` carry ``post_input`` /
   ``attn1_patch`` / ``single_block`` patches or ``patches_replace["dit"]``.

Not ported from that file, deliberately:

* ``_pinned_put`` - stores the host K/V in pinned memory.  Measured here it is
  a no-op in ``device="gpu"`` mode and a regression in ``device="cpu"`` mode
  (2066 MB retained vs 1073 MB, neither released).
* ``_copy_prefix_factory`` - needs CUTE -> the A-series wheel has no CUTE
  extension (``omni_xpu_kernel.cute.is_available()`` is False) -> not done.

Env ``OMNIXPU_QWEN21_CACHE_COMPAT``: ``1`` (default) on, ``0`` off.
"""

import functools
import inspect
import logging
import os
import textwrap

log = logging.getLogger("ComfyUI-OmniXPU")

_MARKER = "__omnixpu_qwen21_cache_compat_original__"
_WRAPPER_KEY = "omnixpu_qwen21_prefix_cache"
_PREFIX_PATCHES = ("post_input", "attn1_patch", "single_block")


def _rewrite(function, replacements):
    """Apply exact source replacements, keeping the rest of upstream's code."""

    if function.__closure__:
        raise ValueError("cache method unexpectedly captures closure state")
    source = textwrap.dedent(inspect.getsource(function))
    if not source.startswith(f"def {function.__name__}("):
        raise ValueError("unsupported decorated cache method")
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"unsupported {function.__name__} cache contract")
        source = source.replace(before, after)
    namespace = dict(function.__globals__)
    exec(compile(source, inspect.getfile(function), "exec"), namespace)
    return functools.update_wrapper(namespace[function.__name__], function)


def _guard_prefix_cache(executor, x, timesteps, context, ref_latents=None,
                        image_slots=None, transformer_options=None, **kwargs):
    options = {} if transformer_options is None else transformer_options
    patches = options.get("patches", {})
    unsafe = any(patches.get(name) for name in _PREFIX_PATCHES)
    unsafe = unsafe or bool(options.get("patches_replace", {}).get("dit", {}))
    model = executor.class_obj
    if x.device.type != "xpu" or not unsafe:
        return executor(x, timesteps, context, ref_latents, image_slots, options, **kwargs)

    enabled = model.prefix_cache_enabled
    model.reset_prefix_cache(False)
    try:
        return executor(x, timesteps, context, ref_latents, image_slots, options, **kwargs)
    finally:
        # Never restore a stale slot, including after an interrupted forward.
        model.reset_prefix_cache(enabled)


def apply():
    if os.environ.get("OMNIXPU_QWEN21_CACHE_COMPAT", "1") == "0":
        return False, "disabled by env"

    try:
        import comfy.ldm.qwen_image21.model as qwen
        import comfy.model_base as model_base
        import comfy.model_management as mm
        import comfy.model_patcher as model_patcher
        from comfy.patcher_extension import WrappersMP
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy", "comfy.ldm.qwen_image21",
                        "comfy.ldm.qwen_image21.model"}:
            return False, "Qwen Image 2.1 is not available in this ComfyUI"
        raise

    if mm.get_torch_device().type != "xpu":
        return False, "requires XPU"

    try:
        select = qwen.PoseBranchCache.select
        patcher = model_patcher.ModelPatcher
    except AttributeError as exc:
        return False, f"Qwen Image 2.1 cache API is unavailable: {exc}"
    if hasattr(select, _MARKER) and hasattr(patcher.__init__, _MARKER):
        return False, "already patched"
    if hasattr(select, _MARKER) != hasattr(patcher.__init__, _MARKER):
        return False, "partially applied"

    try:
        if tuple(inspect.signature(select).parameters) != ("self", "k", "create"):
            raise ValueError("unsupported PoseBranchCache.select signature")
        if inspect.signature(select).parameters["create"].default is not True:
            raise ValueError("unsupported PoseBranchCache.select default")
        forward = qwen.QwenImage21Transformer2DModel._forward
        if tuple(inspect.signature(forward).parameters) != (
            "self", "x", "timesteps", "context", "ref_latents", "image_slots",
            "transformer_options", "kwargs",
        ):
            raise ValueError("unsupported Qwen Image 2.1 forward signature")
        entry = inspect.getsource(qwen.QwenImage21Transformer2DModel.forward)
        body = inspect.getsource(forward)
        if ("WrappersMP.DIFFUSION_MODEL" not in entry
                or "self._forward" not in entry
                or "self.prefix_cache_enabled and prefix_len > 0 and not hooked" not in body
                or any(f'patches.get("{name}")' not in body for name in _PREFIX_PATCHES)):
            raise ValueError("unsupported Qwen Image 2.1 wrapper/cache contract")
        qwen_type = model_base.QwenImage21
        if not callable(getattr(patcher, "add_wrapper_with_key", None)):
            raise ValueError("ModelPatcher diffusion wrappers are unavailable")
        reset = inspect.getsource(qwen.QwenImage21Transformer2DModel.reset_prefix_cache)
        if ("self.prefix_cache.free()" not in reset
                or "self.prefix_cache_enabled = enabled" not in reset):
            raise ValueError("unsupported Qwen Image 2.1 cache reset contract")
        select_xpu = _rewrite(select, (
            ("for s in self.slots:", "for index, s in enumerate(self.slots):"),
            ("self.slots.remove(s)", "self.slots.pop(index)"),
        ))
    except (AttributeError, OSError, TypeError, ValueError, SyntaxError) as exc:
        return False, str(exc)

    original_select = select
    original_init = patcher.__init__

    @functools.wraps(original_select)
    def select(self, k, create=True):
        if k.device.type != "xpu":
            return original_select(self, k, create=create)
        return select_xpu(self, k, create=create)

    @functools.wraps(original_init)
    def init(self, model, load_device, offload_device, *args, **kwargs):
        original_init(self, model, load_device, offload_device, *args, **kwargs)
        if isinstance(model, qwen_type) and load_device.type == "xpu":
            self.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY,
                                      _guard_prefix_cache)

    setattr(select, _MARKER, original_select)
    setattr(init, _MARKER, original_init)
    qwen.PoseBranchCache.select = select
    patcher.__init__ = init
    return True, "PoseBranchCache.select slot bookkeeping + hook-safety guard"
