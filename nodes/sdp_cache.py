"""Standalone node to release the OmniXPU SDP sidecar's packed Q/K/V cache.

ComfyUI's ``unload_all_models`` / ``torch.xpu.empty_cache`` cannot see these
USM buffers. Put this node at the end of a workflow (optionally after a model
pass-through) to free the sidecar cache between runs without touching any
other model-management behavior.
"""


class OmniXPUClearSDPCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clear_sdp_cache": ("BOOLEAN", {"default": True}),
                "empty_cache": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "any_input": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "OmniXPU"
    OUTPUT_NODE = True

    def run(self, clear_sdp_cache, empty_cache, any_input=None):
        import torch

        from omni_xpu_kernel import sdp

        lines = []
        if clear_sdp_cache:
            sdp.clear_cache()
            lines.append("sdp cache cleared")
        if empty_cache:
            torch.xpu.empty_cache()
            lines.append("torch.xpu.empty_cache()")
        if not lines:
            lines.append("no-op")
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "OmniXPUClearSDPCache": OmniXPUClearSDPCache,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniXPUClearSDPCache": "OmniXPU Clear SDP Cache",
}
