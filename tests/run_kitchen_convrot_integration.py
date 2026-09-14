"""One-shot check that real Kitchen XPU dispatch reaches the ConvRot bridge."""
import faulthandler
import importlib.util
import os
from pathlib import Path
import sys
import threading
import types

import torch
import comfy_kitchen
from comfy_kitchen.backends.eager import quantization as eager
from omni_xpu_kernel import int8

faulthandler.enable()
watchdog = threading.Timer(120, lambda: os._exit(124))
watchdog.daemon = True
watchdog.start()
root = Path(__file__).resolve().parents[1]
package = "_omnixpu_kitchen_integration"
for name, directory in ((package, root), (package + ".adapters", root / "adapters"),
                        (package + ".patches", root / "patches")):
    module = types.ModuleType(name)
    module.__path__ = [str(directory)]
    sys.modules[name] = module
debug = types.ModuleType(package + ".patches.debug")
debug.trace_patch = lambda *args, **kwargs: lambda function: function
sys.modules[debug.__name__] = debug
spec = importlib.util.spec_from_file_location(package + ".adapters.kitchen_compat", root / "adapters/kitchen_compat.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
assert bridge.apply()[0]
native = int8.dequantize_int8_convrot_weight_dtype
calls = []
def recorded(*args, **kwargs):
    calls.append(tuple(args[0].shape))
    return native(*args, **kwargs)
int8.dequantize_int8_convrot_weight_dtype = recorded
torch.manual_seed(90675001)
torch.set_num_threads(8)
assert "A770" in torch.xpu.get_device_name()
with torch.inference_mode():
    for shape in ((21504, 5376), (1536, 6144)):
        q = torch.randint(-128, 128, shape, device="xpu", dtype=torch.int8)
        scale = torch.rand(shape[0], 1, device="xpu") * 0.002
        actual = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(q, scale, 256, 2)
        torch.xpu.current_stream().synchronize()
        assert calls[-1] == shape
        rows = [0, 1, 29, shape[0] // 2, shape[0] - 1]
        reference = eager.dequantize_int8_convrot_weight_dtype(q[rows].cpu(), scale[rows].cpu(), 256, 2)
        torch.testing.assert_close(actual[rows].cpu(), reference, rtol=0.008, atol=2e-5)
        del actual
    small_q, small_scale = q[:3].clone(), scale[:3].clone()
    fallback = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(small_q, small_scale, 256, 0)
    torch.xpu.current_stream().synchronize()
    torch.testing.assert_close(fallback.cpu(), eager.dequantize_int8_convrot_weight_dtype(
        small_q.cpu(), small_scale.cpu(), 256, 0), rtol=2e-5, atol=2e-6)
    assert calls == [(21504, 5376), (1536, 6144)]
print("PASS: real Kitchen XPU dispatch, BF16 native output, unmeasured FP32 fallback", flush=True)
watchdog.cancel()
