"""Scope ComfyUI's async weight prefetch to MiniMax H3 diffusion sampling.

A770 measurements on the 0.4 MP / 124-frame H3 workflow: leaving ComfyUI's
prefetch streams at their default produced a 37.9 s median sampling step,
while enabling two non-blocking offload streams for the duration of DiT
sampling produced 25.8 s. The gain comes from overlapping weight transfer with
compute inside the sampling loop.

Enabling it for the whole process is not safe here: model loading, the ClipProj
encoder release and VAE decoding all run outside the sampling loop and rely on
the default stream policy. This adapter therefore flips the two existing
ComfyUI switches only after the diffusion model is loaded and restores them at
the sampling boundary, waiting for the extra streams on the owner thread rather
than inside any allocation hook.
"""

from __future__ import annotations

import functools
import logging

import torch


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "_omnixpu_sampling_prefetch_patched"
_STREAMS = 2
_logged = False


def _is_minimax_h3(model_patcher):
    model = getattr(model_patcher, "model", None)
    return type(model).__name__ == "MiniMaxH3"


def _install_scope(comfy_samplers, model_management, model_prefetch):
    original_sample = comfy_samplers.CFGGuider.sample
    original_make_queue = model_prefetch.make_prefetch_queue
    state = {"active": False, "enabled": False}

    def make_prefetch_queue(queue, device, transformer_options):
        global _logged
        if state["active"] and not state["enabled"] and device.type == "xpu":
            # Entering the first sampling step: the diffusion model is fully
            # loaded, so the extra streams can no longer disturb model loading.
            torch.xpu.current_stream(device).synchronize()
            model_management.NUM_STREAMS = _STREAMS
            model_management.args.force_non_blocking = True
            state["enabled"] = True
            if not _logged:
                log.info(
                    "[OmniXPU] H3 sampling prefetch: %d non-blocking offload "
                    "streams", _STREAMS,
                )
                _logged = True
        return original_make_queue(queue, device, transformer_options)

    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        if not _is_minimax_h3(getattr(self, "model_patcher", None)):
            return original_sample(self, *args, **kwargs)
        previous_streams = model_management.NUM_STREAMS
        previous_non_blocking = model_management.args.force_non_blocking
        state["active"] = True
        state["enabled"] = False
        try:
            return original_sample(self, *args, **kwargs)
        finally:
            # Owner boundary: drain the extra streams before restoring the
            # default policy that model loading and VAE decoding depend on.
            if state["enabled"]:
                device = self.model_patcher.load_device
                torch.xpu.current_stream(device).synchronize()
                for stream in model_management.STREAMS.get(device, ()):
                    stream.synchronize()
            model_management.NUM_STREAMS = previous_streams
            model_management.args.force_non_blocking = previous_non_blocking
            state["active"] = False
            state["enabled"] = False

    setattr(make_prefetch_queue, _PATCH_MARKER, True)
    setattr(sample, _PATCH_MARKER, True)
    model_prefetch.make_prefetch_queue = make_prefetch_queue
    comfy_samplers.CFGGuider.sample = sample


def apply():
    import comfy.model_management as model_management
    import comfy.model_prefetch as model_prefetch
    import comfy.samplers as comfy_samplers

    if getattr(comfy_samplers.CFGGuider.sample, _PATCH_MARKER, False):
        return True, None
    _install_scope(comfy_samplers, model_management, model_prefetch)
    return True, None
