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
[intel/llm-scaler](https://github.com/intel/llm-scaler)
(`omni/ComfyUI-OmniXPU`), with A770/DG2-specific additions. The node is also
bundled with the `llm-scaler-omni` ComfyUI image.

The official upstream README is preserved in
[`UPSTREAM_README.md`](UPSTREAM_README.md); this file documents the fork.

## Documentation layout

| File | Content |
|---|---|
| `README.md` | This fork: install, switches, adapter behavior, A770 additions |
| `UPSTREAM_README.md` | Official upstream README, preserved verbatim |

## Fork changes vs upstream

- Standalone custom-node install (clone into `custom_nodes/`, update with
  `git pull`).
- A770/DG2 compatibility profile: RMS-RoPE bridge, DG2 attention routing
  (`esimd` opt-in plus measured torch fallbacks such as
  `dg2_torch_d64_fp16`), ConvRot fused-path wiring, and INT8 fast paths with
  cached qdata/scale copies.
- Memory adapters: cached whole-LoRA model budgets (`lora_memory.py`) and
  Windows DynamicVRAM boundary trim (`dynamic_vram.py`).
- Windows attention policy defaults to `torch`; A770 users enable ESIMD with
  `OMNI_ATTN_BACKEND=esimd`.

## Ownership

| Layer | Current responsibility |
|---|---|
| Kitchen XPU backend | INT8/QTensor operations, FP8 QDQ and stochastic rounding, SVDQuant, AdaLN, four RoPE APIs, and ConvRot |
| ComfyUI adapter | Attention routing, LayerNorm/RMSNorm class integration, the remaining FP8 model/factory bridge, and fused Lumina/Z-Image INT8 FFN wiring |
| Memory adapter | Cached whole-LoRA model budgets plus optional DynamicVRAM per-layer XPU staging measurements |
| SeedVR2 capacity | Guarded Ada broadcast plus byte-bounded RMSNorm, SwiGLU, and window-attention materialization |
| SeedVR2 native adapters | Validated BMG FP16 GroupNorm and causal-prefix cat-pad routing |
| Large-video preprocessing | Source-guarded, bounded CPU materialization for PIL Lanczos resize, SeedVR input padding, and XPU VAE input staging |
| Legacy fix | Global `F.interpolate` and `torch.median`/`torch.nanmedian` workarounds; disabled by default |

RoPE, generic INT8 linear dispatch, and the old FP8 negative-zero wrapper are
normally not registered by this custom node. The A770/DG2 compatibility
profile is the narrow exception: when Kitchen has no XPU implementation for
`comfy_kitchen::int8_linear`, the adapter registers the Omni implementation.
It skips registration when a Kitchen XPU backend is already present.

ComfyUI's quantized-format eligibility recognizes `int8_tensorwise` when the
active Kitchen XPU backend provides its native INT8 and ConvRot operations.
Model requests for full-precision matrix multiplication still apply. Other
quantized formats retain ComfyUI's own eligibility decisions.

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
  PyTorch 2.13 use the newest `0.2.0b1` wheel, currently build 1:
  `omni_xpu_kernel-0.2.0b1+torch213.dg2.1-cp313-cp313-win_amd64.whl`);
- the pinned `comfy_kitchen` XPU integration;
- upstream ComfyUI.

If an Intel XPU is unavailable, initialization is skipped.

The A770 bridge requires the normal `comfy_kitchen` package that ships with
ComfyUI, but does not require a separate `comfy-kitchen-xpu` installation.
## Official packages and XPU providers

The provider distributions use private top-level package names and do not own
any `comfy_kitchen/*` or `comfy_aimdo/*` file. The official packages can
therefore be reinstalled or upgraded without overwriting the XPU runtime.
ComfyUI's launcher and Python entry point are unchanged.

During the normal custom-node prestartup phase, OmniXPU discovers lightweight
provider metadata before importing PyTorch. It verifies the official package
version, exact Torch XPU build, platform and image target, source revision,
source-wheel hash, and every vendored runtime file hash. Kitchen is then routed
only when its canonical package is first imported.

AIMDO takeover additionally requires explicit DynamicVRAM enablement, an
unimported PyTorch runtime on Linux, and an official `comfy_aimdo.control`
module with no device context or allocator. ComfyUI calls official AIMDO init
before custom-node prestartup; on an XPU Torch build that can leave only its
pre-device CUDA DSO state live. OmniXPU permits exactly that reversible state,
calls the official public `deinit()`, and verifies that it returned to a
pristine module before takeover. Any device or allocator state is rejected.
OmniXPU then executes the provider control implementation in that same module
object so the reference imported by ComfyUI remains valid. A reversible
provider failure restores the deinitialized official module. A failure after
provider allocator or native state becomes live stops startup because
allocator ownership cannot be rolled back safely.

Provider routing defaults to `auto` and can be controlled without changing the
launcher:

```bash
OMNIXPU_PROVIDER_BOOTSTRAP=off       # Keep every official runtime
OMNIXPU_PROVIDER_BOOTSTRAP=auto      # Use each compatible XPU provider
OMNIXPU_PROVIDER_BOOTSTRAP=required  # Fail unless both providers activate
```

After an official package upgrade, an incompatible provider is skipped in
`auto` mode instead of being forced into a new API contract. Upgrade the
corresponding provider wheel to restore XPU routing.

The image's Linux provider defaults to `native_hook`, keeping Torch's native
XPU allocator while AIMDO manages DynamicVRAM weights. The standard
`start_comfyui.sh` entrypoint validates the provider, resolves its default and
preloads its exact native library before starting Python. Set
`AIMDO_XPU_ALLOCATOR_MODE=global` to select the Linux pluggable allocator.
Older providers retain their own advertised default.

Native mode requires DynamicVRAM and fails startup if activation fails after
selection; it cannot silently fall back to another memory policy. Direct Python
launchers enabling DynamicVRAM must also prepare the native preload before
startup. Allocator modes do not change the model graph or enable XPU memory
compilation.

AIMDO 0.5.3 memory compilation (recording and replaying allocation graphs) is
not yet supported on XPU. Its basic APIs and DynamicVRAM model-weight
offloading remain available. This limitation does not disable OmniXPU's
`torch.compile` support.

## Components and switches

Adapters are enabled by default and always retain the original ComfyUI route
for unsupported inputs:

```bash
OMNIXPU_ENABLE=0            # Disable every custom-node component
OMNIXPU_ATTENTION=0         # Disable the attention adapter
OMNIXPU_SPARSE_ATTENTION=0  # Disable XPU eligibility for Model Sparse Attention
OMNIXPU_NORM=0              # Disable the norm adapter
OMNIXPU_RMS_ROPE=0          # Disable the A770 RMS-RoPE bridge
OMNIXPU_FP8_GEMM=0          # Disable the temporary FP8 model/factory adapter
OMNIXPU_QUANTIZED_MATMUL=0  # Disable native INT8 model-format eligibility
OMNIXPU_INT8_FFN=0          # Disable fused Lumina/Z-Image INT8 FFN wiring
OMNIXPU_DYNAMIC_VRAM_BOUNDARY_TRIM=0  # Disable Windows XPU model-boundary trim
OMNIXPU_LORA_MEMORY=0       # Disable cached whole-LoRA budgets and staging logs
OMNIXPU_KITCHEN_COMPAT=0    # Disable the A770 missing-backend bridge
OMNIXPU_INT8_DIRECT_CAST=1  # A770 opt-in: keep offloaded TensorWise INT8 on the quantized path
OMNIXPU_INT8_PATCH_CACHE=1  # A770 opt-in: cache patched bf16 weights across sampling steps
OMNIXPU_SEEDVR_ADA_RESHAPE=0  # Disable the guarded SeedVR2 Ada reshape patch
OMNIXPU_SEEDVR_CAPACITY=0     # Disable bounded SeedVR2 activation scheduling
OMNIXPU_SEEDVR_CAT_PAD=0      # Disable validated BMG causal-prefix cat-pad routing
OMNIXPU_LARGE_VIDEO_PREPROCESS=0  # Disable bounded large-video CPU preprocessing
```

On Windows XPU, the boundary trim turns an unmet DynamicVRAM minimum-memory
budget into an explicit partial VBAR reclaim before model loading. It preserves
loaded models and is enabled by default; the environment variable above is the
A/B-test escape hatch.

Validated sub-routes can be disabled independently:

```bash
OMNI_ATTN_BACKEND=auto      # auto, cute, esimd, or torch; Windows defaults to torch
OMNIXPU_NONCONTIG_RMSNORM=0
OMNIXPU_H120_RMSNORM=0
OMNIXPU_KREA2_RMSNORM=0
OMNIXPU_SEEDVR_GROUPNORM=0
```

On Windows, CUTE is never selected implicitly. A wheel built explicitly with
`OMNI_XPU_REQUIRE_CUTE=1` still uses PyTorch SDPA by default; set
`OMNI_ATTN_BACKEND=cute` before launching ComfyUI to enable the CUTE routes.

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

## torch.compile on A770

The kernel wheel ships Dynamo dispatch boundaries (`omni_xpu_kernel._compile_ops`,
`torch.ops.omni_xpu.*`) so the native kernels are opaque custom operators inside
a compiled graph. Without them Dynamo cannot trace a quantized checkpoint.

Enable it in a workflow with ComfyUI's own `TorchCompileModel` node
(`backend=inductor`) between the model loader and the sampler. Two requirements:

- Inductor needs a C++ compiler on `PATH`; start ComfyUI from a shell that has
  run `...\VC\Auxiliary\Build\vcvars64.bat`, otherwise compilation fails with
  `InvalidCxxCompiler: Compiler: cl is not found`.
- Point `TORCHINDUCTOR_CACHE_DIR` at a stable directory. The default on Windows
  is `%TEMP%\torchinductor_<user>`, which system cleanup can remove.

While tracing, the adapter defers to ComfyUI's own quantized dispatch and skips
the eager fp8 / int8 fast paths: those paths specialise on every weight shape
(one graph per layer until Dynamo hits its per-code recompile limit and falls
back to eager), and keeping them out of the graph with a graph break trips
Dynamo's `transformer_options` resume path (`KeyError: 'total_blocks'` on
Krea2). Eager execution is unchanged.

Measured on Arc A770, Krea2 turbo int8 convrot with the fp8 text encoder
(768x1280, 8 steps):

| scenario | eager | torch.compile (inductor) |
|---|---|---|
| first run, empty cache | 30.8 s | 195 s (compile) |
| restart, disk cache present | 30.8 s | 43.7 s |
| in-process second run | 21.7 s | 17.9 s |

The compile cost is paid once per graph and shape; it pays off when a session
runs many images at the same resolution, not for one-off generations.

### Compiled models bypass AIMDO's dynamic VRAM

ComfyUI's `TorchCompileModel` clones the patcher with `disable_dynamic=True`
because a compiled graph captures weight addresses once; weights cannot be
paged or re-cast per call afterwards. The compiled model is therefore loaded
statically (`Model <name> prepared for dynamic VRAM loading` never appears for
it in the log) while everything else in the process keeps using AIMDO's
dynamic VRAM path.

That is a trade-off, not a bug: compile only helps models that fit in VRAM as
a resident copy. On a 16 GB A770, a 12.8 GB INT8 DiT can be compiled (the text
encoder stays dynamic), while a 32 GB H3 checkpoint cannot and must keep using
AIMDO's dynamic VM. Upstream's runtime bootstrap states the same limitation in
one line: `AIMDO memory compiler is not yet supported on XPU; DynamicVRAM
model-weight offloading is available. This does not disable torch.compile.`

## Model Sparse Attention

Use the upstream **Model Sparse Attention** node (`BlockSparseAttention`)
with a matching complete native sparse API for SOL, SLA and VSA on XPU. Its
generic Sol/SLA path and MiniMax-H3 chunked producer call Kitchen's public APIs. The upstream node owns block selection,
4096-token projection chunks, previous-step statistics, VSA tiling and cleanup.
The adapter only extends its device eligibility checks. An unavailable native
API or an unsupported upstream eligibility contract leaves the original node
behavior in place.

See [native sparse attention usage](../docs/SPARSE_ATTENTION.md) for model
connections, trained SLA/VSA recipes, fallback diagnostics and migration from
the deprecated **Patch Sol-Attn** custom node. The old experimental environment
gate is not needed; `OMNIXPU_SPARSE_ATTENTION` controls this adapter.

## Native compiled inference

Upstream `TorchCompileModel` clones the diffusion model with
`disable_dynamic=True`. That clone does not disable service-wide DynamicVRAM
for text encoding or VAE execution. Match the service memory mode and model
cloning when comparing eager and compiled performance; use ComfyUI's separate
`--disable-dynamic-vram` option when explicitly selecting a service without
DynamicVRAM.

Compiled FP16/BF16 pointwise operations can round differently from eager even
when each native operator matches. See the kernel package's
[compiled-inference guidance](../omni_xpu_kernel/README.md#compiled-inference).
On an installed Torch build that supports the option,
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` before ComfyUI startup selects eager
precision emulation for that process. This is an explicit numerical policy to
validate for the model, not a default enabled by OmniXPU.

## Debugging and diagnostics

Kernel-only tracing:

```bash
OMNIXPU_DEBUG=1 python main.py
```

Dispatch decisions and fallback reasons:

```bash
OMNIXPU_DEBUG_VERBOSE=1 python main.py
```

LoRA weights are measured once when the LoRA node executes. Unique tensor sizes
are cached in a `ModelPatcher` attachment, inherited by clones, accumulated for
stacked LoRAs, and added to both `memory_required` and an explicitly supplied
`minimum_memory_required`. The base model's `model_size()` semantics stay
unchanged. Model loads read the cached attachment instead of rescanning patches.
DynamicVRAM layer scanning is disabled by default. To diagnose every LoRA
staging operation, including its XPU state and any failure, enable:

```bash
OMNIXPU_LORA_MEMORY_TRACE=1 python main.py
```

LoRA memory logs report `xpu_memory=available`, `partial`, or `unavailable`
for the diagnostic snapshot, not GPU availability. Allocator counters and
device free/total memory are queried independently; successful values remain
visible if another query fails. Missing values are listed rather than reported
as zero, and query errors include the interface, exception type and message.
Statistics failures do not change LoRA budgets or interrupt model loading.

Set tracing variables before startup. The **OmniXPU Status** node reports:

- GPU and `omni_xpu_kernel` capabilities;
- runtime-provider activation, skip, and rejection reasons;
- each component's kind (`adapter`, `compatibility_patch`, or `legacy_fix`)
  and apply status;
- attention and fused INT8 FFN routing counters.

Attention, INT8 FFN and H3 RMS modulation counters record eager calls. During `torch.compile`,
these diagnostic counters and logs are excluded from tracing so changing a
counter cannot cause recompilation. Use the Torch profiler's operator events
to inspect compiled native calls.

Kitchen backend ownership can be inspected independently:

```bash
python -c 'import comfy_kitchen as ck; print(ck.list_backends()["xpu"])'
```

The INT8 fast forward uses ComfyUI's `cast_bias_weight` / `uncast_bias_weight`
pair, including Dynamic VRAM residency and low-VRAM LoRA patches. It avoids
the redundant activation quantize/dequantize round trip, without keeping a
second GPU copy of CPU model weights. `OMNIXPU_INT8_FAST_FORWARD=0` disables
this shortcut for comparison. The former `OMNIXPU_INT8_FAST_FORWARD_COPY*`
and `OMNIXPU_INT8_QDATA_CACHE*` controls no longer apply: AIMDO owns weight
residency, and its cast views must not be cached after unpinning.

On A770, the RMS-RoPE adapter also fuses the measured 1024-square Z-Image
(`B=1, S=4256, H=30, D=128`) and Krea2 (`B=1, S=4192, Hq=48, Hkv=12,
D=128`) BF16 contracts. Other shapes use their existing routes. Krea2 keeps
its FP32 `scale + 1` convention and BF16 rounding before RoPE, and preserves
the ordinary projection casts, LoRA, grouped attention and output gate.
Training, attention patches and intermediate forward hooks use the original
Krea2 forward. `OMNIXPU_ZIMAGE_RMS_ROPE=0` and `OMNIXPU_KREA2_RMS_ROPE=0`
disable the respective routes for comparison.

## SeedVR2 on A770

The SeedVR2 adapters (from upstream #622 plus A770 validation) route:

- `seedvr_ada_reshape`: bounded Ada modulation, removing a >4 GiB
  `repeat_interleave` that OOMs K sampling on A770;
- `seedvr_capacity` / `large_video_preprocess`: bounded attention, SwiGLU,
  Lanczos and VAE staging;
- `seedvr_cat_pad` / `norm` SeedVR GroupNorm: enabled on DG2 with the
  validated kernel route (shapes match the BMG-validated set);
- `seedvr_vae_decode`: CPU-stages the tiled decode result when a single XPU
  allocation would exceed the ~4 GiB per-allocation limit; auto-disables
  when `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`;
- DG2 D128 attention: torch SDPA for `q_len` in `[1024, 2048)` and
  `(2048, 4096)` (measured 1.1-1.6x slower with ESIMD on A770), ESIMD for
  `q_len == 2048` and `q_len >= 4096`.

Measured end-to-end on A770 (`seedvr2_3b_int8_upscale_video.json`):
681 s -> 607 s. Negatives recorded in `WHL_BUILD_INSTALL.md`: spatial tile
1024 OOMs, temporal chunk 125 gives no gain, and lowering
the former `OMNIXPU_INT8_FAST_FORWARD_COPY_MIN_ELEMS` below 16 Mi did not
speed up sampling. Those copy controls have since been removed.

## Contribution boundary

New device-generic math, layouts, quantization, or fallback logic belongs in
`comfy_kitchen`. A custom-node adapter is appropriate only when a ComfyUI class
or call site cannot yet use the Kitchen API. Global correctness workarounds
must be opt-in and should carry a concrete upstream removal plan.

Model-pipeline or model-`forward` changes are outside this layer and require a
separate review.
