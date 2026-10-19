"""Compatibility and DynamicVRAM support for MOSS-Image."""

import logging

log = logging.getLogger("ComfyUI-OmniXPU")

_UNITS_MARKER = "_omnixpu_moss_image_dynamic_units_compat"


def apply():
    try:
        import comfy.model_base as mb
        if hasattr(mb, "MOSSImage") and not getattr(mb.MOSSImage, _UNITS_MARKER, False):
            def get_dynamic_vram__units(self):
                diffusion_model = getattr(self, "diffusion_model", None)
                layers = getattr(diffusion_model, "layers", None)
                if layers is not None:
                    return list(layers), []
                return [], []

            setattr(mb.MOSSImage, _UNITS_MARKER, True)
            mb.MOSSImage.get_dynamic_vram__units = get_dynamic_vram__units
            log.info("[OmniXPU] MOSS-Image DynamicVRAM units registered")
            return True, None
    except Exception as exc:
        log.warning("[OmniXPU] MOSS-Image DynamicVRAM units skipped: %s", exc)
        return False, str(exc)

    return True, None
