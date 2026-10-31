# Kitchen XPU backend (vendored)

`comfy-kitchen` 0.2.40 — required by ComfyUI 0.43.x — ships `ascend`, `cuda`,
`eager`, `hip` and `triton` backends only. The XPU backend lives in Intel's
`xiangyuT/comfy-kitchen-xpu` fork and, in Intel's own `llm-scaler-omni` image,
reaches the runtime as a separate `comfy-kitchen-xpu-runtime` provider wheel.

That provider is a third component users must install and keep in sync. This
package vendors the same backend so the historical two-component install
(ComfyUI-OmniXPU + omni_xpu_kernel) keeps working.

## What is vendored

| Path | Source |
|---|---|
| `backends/xpu/*.py` | `xiangyuT/comfy-kitchen-xpu@3a5728c` `comfy_kitchen/backends/xpu/` |
| `gguf.py` | same revision, `comfy_kitchen/gguf.py` |
| `svdquant_w4a16.py` | same revision |
| `backends/eager/gguf.py` | same revision |
| `backends/eager/svdquant_w4a16.py` | same revision |

These five groups are the only files the published 0.2.40 wheel lacks. No
existing `comfy_kitchen` file is replaced.

## How it installs

`bootstrap.install()` runs from `__init__.py` before any adapter probes
`comfy_kitchen.registry`:

1. Skip unless `comfy-kitchen >= 0.2.40` is importable.
2. Skip when an XPU backend is already registered (a provider is installed, or
   the backend was grafted by a previous import).
3. Append `kitchen_xpu/` and its `backends/` subdirectories to
   `comfy_kitchen.__path__` so `comfy_kitchen.backends.xpu` resolves here.
4. Import the vendored modules under their official dotted names, dependencies
   first, so `comfy_kitchen.backends.xpu` sees the gguf/svdquant modules.
   Importing the backend registers itself via `registry.register("xpu", ...)`.
5. Prepend `xpu` to the registry priority list.

Step 5 matters: the stock priority is `["ascend", "cuda", "triton", "eager"]`,
so a registered-but-unprioritized XPU backend still loses every dispatch to
Triton or Eager. Measured cost of omitting it on an Arc A770: 1.128 s/it
instead of 1.010 s/it (Z-Image 1024², 24 steps, warm).

## Switches

`OMNIXPU_KITCHEN_GRAFT=0` disables the graft, restoring stock `comfy_kitchen`
behaviour (the adapters then report the XPU backend as unavailable).

## Notes

- If `comfy-kitchen-xpu-runtime` is installed, step 2 sees its backend and does
  nothing; the two mechanisms do not conflict.
- The backend is pure Python. It forwards to `omni_xpu_kernel`'s compiled
  kernels, so no rebuild is needed to pick up a new `comfy-kitchen` version.
- Operators the installed `omni_xpu_kernel` does not expose are excluded from
  the capability table automatically (`group_norm_stats3d`,
  `flash_attention_decode_tree_merge` on a DG2 build that predates them); the
  registry then falls back to Eager for those calls.
