"""Opt-in real Comfy cast/LoRA/VBAR oracle, run in an isolated XPU process.

Usage: python tests/run_int8_cast_vbar_integration.py --comfy-root PATH
No model files or ComfyUI source are modified. Peak test storage is small.
"""
import argparse
import faulthandler
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import types

parser = argparse.ArgumentParser()
parser.add_argument("--comfy-root", required=True)
parser.add_argument("--async-offload", type=int, default=0)
args = parser.parse_args()
faulthandler.enable()
watchdog = threading.Timer(120, lambda: os._exit(124))
watchdog.daemon = True
watchdog.start()
sys.argv = [sys.argv[0]]
sys.path.insert(0, args.comfy_root)
os.environ["OMNIXPU_PROVIDER_BOOTSTRAP"] = "auto"
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
root = Path(__file__).resolve().parents[1]
package = "ComfyUI-OmniXPU"
for name, directory in ((package, root), (package + ".adapters", root / "adapters"),
                        (package + ".patches", root / "patches")):
    module = types.ModuleType(name)
    module.__path__ = [str(directory)]
    sys.modules[name] = module

def load(relative):
    name = package + "." + relative.replace("/", ".").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

from comfy_aimdo import control
state = load("runtime_bootstrap.py").bootstrap(dynamic_vram_override=True)
assert state["providers"]["comfy_aimdo.xpu"]["status"] == "active", state

import torch
import comfy_kitchen as ck
ck.disable_backend("triton")
from comfy.cli_args import args as comfy_args
comfy_args.enable_dynamic_vram = True
comfy_args.async_offload = args.async_offload
comfy_args.force_non_blocking = args.async_offload > 0
import comfy.ops
import comfy.memory_management
from comfy.model_patcher import LowVramPatch
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout
from comfy_aimdo.model_vbar import ModelVBAR, vbar_signature_compare
from comfy_aimdo.host_buffer import HostBuffer
from omni_xpu_kernel import int8

adapter = load("adapters/fp8_gemm.py")
adapter._omni_int8 = int8
device = torch.device("xpu", 0)
assert "A770" in torch.xpu.get_device_name(device)
assert control.init_devices([0])
assert comfy.model_management.NUM_STREAMS == args.async_offload
assert comfy.model_management.device_supports_non_blocking(device) == (args.async_offload > 0)
control.set_simple_vram_headroom(64 << 20)
torch.set_num_threads(8)
torch.manual_seed(906301)

class CountedLoRA(LowVramPatch):
    calls = 0
    def __call__(self, weight):
        self.calls += 1
        return super().__call__(weight)

reports = []
with torch.inference_mode():
    for convrot in (False, True):
        layer = torch.nn.Module()
        dense = torch.randn(128, 256, dtype=torch.bfloat16) * 0.125
        base = QuantizedTensor.from_float(dense, "TensorWiseINT8Layout",
                    per_channel=True, convrot=convrot, convrot_groupsize=64)
        layer.weight = base
        layer.bias = torch.randn(128, dtype=torch.bfloat16) * 0.02
        layer.weight_function = []
        layer.bias_function = []
        layer.seed_key = "integration.linear.weight"
        layer.quant_format = "int8_tensorwise"
        layer._pin_state = {
            name: (HostBuffer(0, 8 << 20, 64 << 20), [], [-1], [0], [0], {})
            for name in ("weights", "weights-loaded", "patches", "patches-loaded")
        }
        layer._v_signature = None
        up = torch.randn(128, 8, dtype=torch.bfloat16) * 0.15
        down = torch.randn(8, 256, dtype=torch.bfloat16) * 0.15
        lora = LoRAAdapter(set(), (up, down, 8.0, None, None, None))
        patches = {layer.seed_key: [(0.8, lora, 1.0, None, None)]}
        layer.weight_lowvram_function = CountedLoRA(layer.seed_key, patches)
        layer.weight_lowvram_function._pin_state = layer._pin_state
        vbar = ModelVBAR(32 << 20, 0)
        layer._v = vbar.alloc(comfy.memory_management.vram_aligned_size([base, layer.bias]))
        original_qdata = base._qdata.clone()
        x_cpu = torch.randn(32, 256, dtype=torch.bfloat16) * 0.25
        x = x_cpu.to(device)
        base_dense = base.dequantize().float()
        unpatched = torch.nn.functional.linear(x_cpu.float(), base_dense, layer.bias.float())

        def check(strength, stage):
            out = adapter._int8_forward_cast(comfy.ops, layer, x,
                                             QuantizedTensor, TensorWiseINT8Layout)
            actual = out.cpu().float()
            # Independent CPU FP32 matrix expression, including BF16 patch rounding.
            delta = (up.float() @ down.float()).to(torch.bfloat16)
            patched = (base_dense.to(torch.bfloat16) + strength * delta).to(torch.bfloat16)
            expected = torch.nn.functional.linear(x_cpu.float(), patched.float(), layer.bias.float())
            torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.06)
            assert (actual - unpatched).abs().max().item() > 0.1
            assert torch.equal(base._qdata, original_qdata)
            return {"stage": stage, "max_abs": (actual-expected).abs().max().item(),
                    "patch_calls": layer.weight_lowvram_function.calls,
                    "map_calls": control.get_xpu_vmm_stats()["map_calls"]}

        stages = [check(0.8, "first_fault")]
        first_signature = layer._v_signature
        stages.append(check(0.8, "resident_hit"))
        assert layer.weight_lowvram_function.calls == 1
        assert stages[0]["map_calls"] == stages[1]["map_calls"]
        assert vbar_signature_compare(first_signature, layer._v_signature)
        torch.xpu.current_stream().synchronize()
        torch.xpu.synchronize()
        assert vbar.free_memory(32 << 20) == 32 << 20
        vbar.prioritize()
        stages.append(check(0.8, "evict_refault"))
        assert layer.weight_lowvram_function.calls == 2
        assert not vbar_signature_compare(first_signature, layer._v_signature)
        torch.xpu.current_stream().synchronize()
        torch.xpu.synchronize()
        assert vbar.free_memory(32 << 20) == 32 << 20
        patches[layer.seed_key] = [(-0.4, lora, 1.0, None, None)]
        layer.weight_lowvram_function = CountedLoRA(layer.seed_key, patches)
        layer.weight_lowvram_function._pin_state = layer._pin_state
        vbar.prioritize()
        stages.append(check(-0.4, "updated_lora_refault"))
        assert layer.weight_lowvram_function.calls == 1
        torch.xpu.current_stream().synchronize()
        torch.xpu.synchronize()
        reports.append({"convrot": convrot, "stages": stages})
        print("CASE_PASS", json.dumps(reports[-1]), flush=True)
        del layer, x, vbar
        gc.collect()

torch.xpu.current_stream().synchronize()
torch.xpu.synchronize()
print("REAL_CAST_LORA_VBAR_PASS", json.dumps(reports), flush=True)
print("ASYNC_STREAMS", args.async_offload, flush=True)
watchdog.cancel()
