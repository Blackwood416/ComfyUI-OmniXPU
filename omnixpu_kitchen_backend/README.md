# Kitchen XPU backend (plugin port)

Ported from `xiangyuT/comfy-kitchen`, branch
`dev/kitchen-0240-xpu-20261028` (commit `3a5728c`), which extends
comfy-kitchen 0.2.40 with the Intel XPU backend.

The plugin imports this package before `patches.apply_all_patches()`, so the
`xpu` backend is registered in `comfy_kitchen.registry` before the OmniXPU
adapters probe it. Set `OMNIXPU_KITCHEN_BACKEND=0` to disable.

## Requirements

- `comfy-kitchen` 0.2.40 installed.
- Four Python modules from the branch above are not part of the 0.2.40
  wheel and must be present in `comfy_kitchen` (copy them verbatim):
  - `comfy_kitchen/gguf.py`
  - `comfy_kitchen/svdquant_w4a16.py`
  - `comfy_kitchen/backends/eager/gguf.py`
  - `comfy_kitchen/backends/eager/svdquant_w4a16.py`

Without them the backend raises `ModuleNotFoundError` and the plugin logs
`Kitchen XPU backend unavailable`, leaving the previous per-op registration
path in effect.

## Coverage

Capabilities are gated by what `omni_xpu_kernel` actually exposes on the
target GPU. On DG2/A770 this registers 48 capabilities (adaln, rope, fp8,
convrot_w4a4, svdquant, w4a8_int8, gguf, int8, conv3d, ...). Kernel features
that only exist on BMG (CUTE `sol_attn`, `na3d`, mxfp8/nvfp4 paths,
`group_norm_stats3d`, ...) are simply absent from the capability list.

`decode.py` is included for completeness (LLM decode adapters). It only
exports `is_available` / `gqa` / `step_merge` / `tree_merge` and is not
registered in the capability table; image workflows do not use it.
