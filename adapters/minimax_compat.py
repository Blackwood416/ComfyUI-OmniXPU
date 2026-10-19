"""Compatibility bridges for MiniMax H3 ecosystem nodes."""

import functools
import logging

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "_omnixpu_minimax_final_layer_compat"
_UNITS_MARKER = "_omnixpu_minimax_dynamic_units_compat"
_TE_PATCH_MARKER = "_omnixpu_te_cpu_fallback"


def apply():
    # 1. FinalLayer.forward compatibility for legacy 4-arg callers (e.g., TE-Speed-MiniMaxH3)
    try:
        import comfy.ldm.minimax.model as mmm
        orig = getattr(mmm.FinalLayer, "forward", None)
        if orig is not None and not getattr(orig, _PATCH_MARKER, False):
            @functools.wraps(orig)
            def forward(self, x, t_emb, video_seg, audio_seg, sigma=None, sample_sigmas=None, shifts=None, *args, **kwargs):
                # Precompiled acceleration nodes (e.g., TE-Speed-MiniMaxH3) invoke FinalLayer.forward
                # with the legacy 4 positional arguments (x, t_emb, video_seg, audio_seg).
                # For standard MiniMax H3 models (n == 1), sigma, sample_sigmas and shifts are unused.
                return orig(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts, *args, **kwargs)

            setattr(forward, _PATCH_MARKER, True)
            mmm.FinalLayer.forward = forward
            log.info("[OmniXPU] MiniMax H3 FinalLayer 4-arg legacy compatibility applied")
    except Exception as exc:
        log.warning("[OmniXPU] MiniMax H3 FinalLayer compat skipped: %s", exc)

    # 2. DynamicVRAM unit segmentation for MiniMaxH3
    try:
        import comfy.model_base as mb
        if hasattr(mb, "MiniMaxH3") and not getattr(mb.MiniMaxH3, _UNITS_MARKER, False):
            def get_dynamic_vram__units(self):
                diffusion_model = getattr(self, "diffusion_model", None)
                blocks = getattr(diffusion_model, "blocks", None)
                if blocks is not None:
                    return list(blocks), []
                return [], []

            setattr(mb.MiniMaxH3, _UNITS_MARKER, True)
            mb.MiniMaxH3.get_dynamic_vram__units = get_dynamic_vram__units
            log.info("[OmniXPU] MiniMax H3 DynamicVRAM units registered")
    except Exception as exc:
        log.warning("[OmniXPU] MiniMax H3 DynamicVRAM units skipped: %s", exc)

    # 3. Safe CPU caching for TE-Speed-MiniMaxH3 on 16GB cards
    try:
        import sys
        for mod in list(sys.modules.values()):
            te_cls = getattr(mod, "TESpeedMiniMaxH3", None)
            if te_cls is not None:
                orig_patch = getattr(te_cls, "patch", None)
                if orig_patch is not None and not getattr(orig_patch, _TE_PATCH_MARKER, False):
                    @functools.wraps(orig_patch)
                    def te_patch(self, model, processing_control_value, processing_percent_1, processing_percent_2, mcs, device, mode):
                        if device == "gpu":
                            log.info("[OmniXPU] Routing TE-Speed cache device to CPU to avoid VRAM overcommit on A770 16GB")
                            device = "cpu"
                        return orig_patch(self, model, processing_control_value, processing_percent_1, processing_percent_2, mcs, device, mode)

                    setattr(te_patch, _TE_PATCH_MARKER, True)
                    te_cls.patch = te_patch
                    log.info("[OmniXPU] TE-Speed-MiniMaxH3 safe CPU caching adapter applied")
                    break
    except Exception as exc:
        log.warning("[OmniXPU] TE-Speed safe CPU caching adapter skipped: %s", exc)

    return True, None
