# ComfyUI-OmniXPU

Thin Intel XPU integration for upstream ComfyUI.

The runtime is deliberately split into three layers:

1. `omni_xpu_kernel` supplies native XPU kernels.
2. `comfy_kitchen` owns generic operator APIs, capability checks, dispatch,
   and safe eager fallback.
3. `ComfyUI-OmniXPU` only adapts ComfyUI call sites that do not yet expose a
   Kitchen entry point, plus a small set of opt-in legacy correctness fixes.

No workflow or model-pipeline replacement is required.

This repository is the standalone home of the custom node, extracted from
[intel/llm-scaler](https://github.com/intel/llm-scaler). The node is also
bundled with the `llm-scaler-omni` ComfyUI image.

## Ownership

| Layer | Current responsibility |
|---|---|
| Kitchen XPU backend | INT8/QTensor operations, FP8 QDQ and stochastic rounding, SVDQuant, AdaLN, four RoPE APIs, and ConvRot |
| ComfyUI adapter | Attention routing, LayerNorm/RMSNorm class integration, the remaining FP8 model/factory bridge, and fused Lumina/Z-Image INT8 FFN wiring |
| Legacy fix | Global `F.interpolate` and `torch.median`/`torch.nanmedian` workarounds; disabled by default |

RoPE, generic INT8 linear dispatch, and the old FP8 negative-zero wrapper are
normally not registered by this custom node. The A770/DG2 compatibility
profile is the narrow exception: when Kitchen has no XPU implementation for
`comfy_kitchen::int8_linear`, the adapter registers the Omni implementation.
It skips registration when a Kitchen XPU backend is already present.

## Install

Install as a ComfyUI custom node (no tag needed; update with `git pull`):

```bash
git clone https://github.com/Blackwood416/ComfyUI-OmniXPU ComfyUI/custom_nodes/ComfyUI-OmniXPU
git -C ComfyUI/custom_nodes/ComfyUI-OmniXPU pull   # later updates
```

The node requires:

- an `omni_xpu_kernel` wheel built for the active XPU target and Torch minor
  (prebuilt Windows wheels are published at
  <https://github.com/Blackwood416/omni-xpu-kernel/releases>; for A770 /
  PyTorch 2.13 use `omni_xpu_kernel-0.2.0b1+torch213.dg2-cp313-cp313-win_amd64.whl`);
- the pinned `comfy_kitchen` XPU integration;
- upstream ComfyUI.

If an Intel XPU is unavailable, initialization is skipped.

The A770 bridge requires the normal `comfy_kitchen` package that ships with
ComfyUI, but does not require a separate `comfy-kitchen-xpu` installation.

## Components and switches

Adapters are enabled by default and always retain the original ComfyUI route
for unsupported inputs:

```bash
OMNIXPU_ENABLE=0            # Disable every custom-node component
OMNIXPU_ATTENTION=0         # Disable the attention adapter
OMNIXPU_NORM=0              # Disable the norm adapter
OMNIXPU_FP8_GEMM=0          # Disable the temporary FP8 model/factory adapter
OMNIXPU_INT8_FFN=0          # Disable fused Lumina/Z-Image INT8 FFN wiring
OMNIXPU_KITCHEN_COMPAT=0    # Disable the A770 missing-backend bridge
OMNIXPU_INT8_DIRECT_CAST=1  # A770 opt-in: keep offloaded TensorWise INT8 on the quantized path
OMNIXPU_INT8_PATCH_CACHE=1  # A770 opt-in: cache patched bf16 weights across sampling steps
```

Validated sub-routes can be disabled independently:

```bash
OMNI_ATTN_BACKEND=auto      # auto, cute, esimd, or torch; Windows defaults to torch
OMNIXPU_NONCONTIG_RMSNORM=0
OMNIXPU_H120_RMSNORM=0
OMNIXPU_KREA2_RMSNORM=0
```

For diagnostics, the per-call CUTE output scan can be enabled explicitly. It
is disabled by default because validated CUTE routes accumulate in FP32 and a
full output scan adds a shape-proportional temporary allocation. Explicit
ESIMD FP16 routing retains its overflow scan regardless of this setting.

```bash
OMNIXPU_VALIDATE_ATTENTION_OUTPUT=1
```

The two global workarounds are opt-in:

```bash
OMNIXPU_INTERPOLATE_FIX=1
OMNIXPU_MEDIAN_FIX=1
OMNIXPU_MEDIAN_STRICT_INDICES=1
```

`OMNIXPU_MEDIAN_STRICT_INDICES=1` reproduces the exact tie-break indices. The
median workaround was only verified on BMG with Torch 2.10 and must not be
enabled by default on PTL-H or another Torch version.

`OMNIXPU_INT8_DIRECT_CAST=1` is an A770-only experiment for partially
offloaded INT8 models. ComfyUI's offload cast path can dequantize a
TensorWise INT8 weight to bf16 before a linear, which hides the real GEMM
cost behind page-in/cast work. The adapter moves the raw qdata/scale to XPU
and returns a device-resident `QuantizedTensor`, so `comfy_kitchen`'s INT8
linear path runs without the bf16 materialization. It only applies when the
module has no LoRA/lowvram/weight functions and is off the current device.
For modules with a LoRA `weight_function`, the first cast still computes the
patched bf16 weight, but the result is cached on CPU; later sampling steps
reuse the cached patched bf16 weight instead of recomputing the LoRA every
step. This is separate (`OMNIXPU_INT8_PATCH_CACHE=1`) because early
measurements showed it can regress under VRAM pressure; leave it off unless
you are A/B testing that specific cache.

## Adapter behavior

Attention uses explicit capability guards. `auto` selects CUTE routes for
matching platform, Torch-version, dtype, layout, and operator contracts, and
uses the original PyTorch attention path for every remaining contract. It
never selects ESIMD. `cute` and `esimd` are explicit diagnostic policies;
unsupported contracts still fall back safely.

On Windows, an unset `OMNI_ATTN_BACKEND` defaults to `torch`, leaving ComfyUI's
PyTorch SDPA route unpatched. ESIMD remains available as an explicit diagnostic
or performance opt-in with `OMNI_ATTN_BACKEND=esimd`; it is never selected
automatically.

On BMG with Torch 2.11, the experimental LTX-style BF16 D128 route accepts
dense B2/H32 self-attention and B1/B2/H32 KV1024 cross-attention inputs as
`[B,L,H*D]` tensors or dense BHLD views. The adapter makes the BHLD view
without a layout copy. B2 self-attention uses CUTE from sequence length 768,
and B1/B2 cross-attention uses it from query length 1024 when KV length is
1024. There is no generation-size-derived upper limit: larger lengths are
selected from the public kernel capability instead of an exact traced shape.

The first use logs a warning with the global rollback setting. If the native
operation raises for a contract, that call falls back to PyTorch and the
contract is quarantined for the rest of the process. Set
`OMNI_ATTN_BACKEND=torch` before ComfyUI startup to disable the experimental
route globally.

The norm adapter preserves ComfyUI cast/offload hooks and uses native kernels
only for eligible tensors. PTL-H H120 and non-contiguous split-QKV routes also
require native feature markers, preventing a stale wheel from taking them.

The FP8 adapter is temporary ComfyUI integration around model/factory paths
that are not completely expressed as Kitchen operations. Generic FP8 tensor
quantization and dequantization remain Kitchen-owned.

The fused INT8 FFN adapter wires eligible Lumina/Z-Image `FeedForward` blocks
to Kitchen/native primitives. It does not register `comfy_kitchen::int8_linear`
and does not replace a model pipeline. LoRA, offloaded weights, bias, training,
unsupported layouts, and unsupported shapes retain ComfyUI's original route.

## Debugging and diagnostics

Kernel-only tracing:

```bash
OMNIXPU_DEBUG=1 python main.py
```

Dispatch decisions and fallback reasons:

```bash
OMNIXPU_DEBUG_VERBOSE=1 python main.py
```

Set tracing variables before startup. The **OmniXPU Status** node reports:

- GPU and `omni_xpu_kernel` capabilities;
- each component's kind (`adapter` or `legacy_fix`) and apply status;
- attention and fused INT8 FFN routing counters.

Kitchen backend ownership can be inspected independently:

```bash
python -c 'import comfy_kitchen as ck; print(ck.list_backends()["xpu"])'
```

## Contribution boundary

New device-generic math, layouts, quantization, or fallback logic belongs in
`comfy_kitchen`. A custom-node adapter is appropriate only when a ComfyUI class
or call site cannot yet use the Kitchen API. Global correctness workarounds
must be opt-in and should carry a concrete upstream removal plan.

Model-pipeline or model-`forward` changes are outside this layer and require a
separate review.
