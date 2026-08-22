import os


class Config:
    """Feature flags controlled via environment variables.

    Generic XPU operators are selected by comfy_kitchen and therefore do not
    have custom-node flags here.  Legacy global workarounds are opt-in.
    """

    def __init__(self):
        master = os.environ.get("OMNIXPU_ENABLE", "1") != "0"
        self.attention = master and os.environ.get("OMNIXPU_ATTENTION", "1") != "0"
        self.rotary = master and os.environ.get("OMNIXPU_ROTARY", "1") != "0"
        self.rms_rope = master and os.environ.get("OMNIXPU_RMS_ROPE", "1") != "0"
        self.norm = master and os.environ.get("OMNIXPU_NORM", "1") != "0"
        self.fp8_gemm = master and os.environ.get("OMNIXPU_FP8_GEMM", "1") != "0"
        self.int8_ffn = master and os.environ.get("OMNIXPU_INT8_FFN", "1") != "0"
        self.int4_gemm = master and os.environ.get("OMNIXPU_INT4_GEMM", "1") != "0"
        self.lora_memory = (
            master and os.environ.get("OMNIXPU_LORA_MEMORY", "1") != "0"
        )
        self.seedvr_ada_reshape = (
            master
            and os.environ.get("OMNIXPU_SEEDVR_ADA_RESHAPE", "1") != "0"
        )
        self.seedvr_capacity = (
            master
            and os.environ.get("OMNIXPU_SEEDVR_CAPACITY", "1") != "0"
        )
        self.seedvr_cat_pad = (
            master
            and os.environ.get("OMNIXPU_SEEDVR_CAT_PAD", "1") != "0"
        )
        self.seedvr_vae_cpu_stage = (
            master
            and os.environ.get("OMNIXPU_SEEDVR_VAE_CPU_STAGE", "1") != "0"
        )
        self.large_video_preprocess = (
            master
            and os.environ.get("OMNIXPU_LARGE_VIDEO_PREPROCESS", "1") != "0"
        )
        self.dynamic_vram_boundary_trim = (
            master
            and os.environ.get("OMNIXPU_DYNAMIC_VRAM_BOUNDARY_TRIM", "1") != "0"
        )
        self.int8_direct_cast = (
            master and os.environ.get("OMNIXPU_INT8_DIRECT_CAST", "0") != "0"
        )
        self.int8_patch_cache = (
            master and os.environ.get("OMNIXPU_INT8_PATCH_CACHE", "0") != "0"
        )
        self.kitchen_compat = (
            master and os.environ.get("OMNIXPU_KITCHEN_COMPAT", "1") != "0"
        )
        self.interpolate_fix = (
            master and os.environ.get("OMNIXPU_INTERPOLATE_FIX", "0") != "0"
        )
        self.median_fix = (
            master and os.environ.get("OMNIXPU_MEDIAN_FIX", "0") != "0"
        )
        self.sdp_cache_lifecycle = (
            master
            and os.environ.get("OMNIXPU_SDP_CACHE_AUTOCLEAR", "1")
            .strip().lower()
            not in ("keep", "0", "false", "no", "off")
        )


config = Config()
