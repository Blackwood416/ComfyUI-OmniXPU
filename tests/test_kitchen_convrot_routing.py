"""Exercise Kitchen registration and fallback without allocating GPU memory."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode


@pytest.fixture
def bridge(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    package = "_omnixpu_kitchen_test"
    for name, directory in ((package, root), (package + ".adapters", root / "adapters"),
                            (package + ".patches", root / "patches")):
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, name, module)
    debug = types.ModuleType(package + ".patches.debug")
    debug.trace_patch = lambda *args, **kwargs: lambda function: function
    monkeypatch.setitem(sys.modules, debug.__name__, debug)
    registry_module = types.ModuleType("comfy_kitchen.registry")
    events = []

    def get_implementation(name, kwargs):
        assert name == "dequantize_int8_convrot_weight_dtype"
        def original(**actual):
            assert actual == kwargs
            events.append(("fallback", actual))
            return "original result"
        return original

    registry_module.registry = types.SimpleNamespace(get_implementation=get_implementation)
    monkeypatch.setitem(sys.modules, registry_module.__name__, registry_module)
    monkeypatch.setattr(torch.ops, "comfy_kitchen", types.SimpleNamespace(
        dequantize_int8_convrot_weight_dtype=object()))
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", lambda *args: False)
    callbacks = {}
    def register(operator, key):
        def decorator(function):
            assert key == "XPU"
            callbacks[operator] = function
            return function
        return decorator
    monkeypatch.setattr(torch.library, "impl", register)
    name = package + ".adapters.kitchen_compat"
    spec = importlib.util.spec_from_file_location(name, root / "adapters/kitchen_compat.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, callbacks, events


def inputs(shape=(21504, 5376), *, scale_shape=None, device="xpu", dtype=torch.int8,
           scale_dtype=torch.float32, offset=0):
    mode = FakeTensorMode()
    q_meta = torch.empty(shape[0] * shape[1] + offset, dtype=dtype, device="meta")
    q_meta = q_meta[offset:].view(shape)
    scale_meta = torch.empty(scale_shape or (shape[0], 1), dtype=scale_dtype, device="meta")
    return (FakeTensor(mode, q_meta, torch.device(device)),
            FakeTensor(mode, scale_meta, torch.device(device)))


def install(bridge, kernel=None):
    module, callbacks, events = bridge
    if kernel is None:
        def kernel(*args):
            events.append(("native", args))
            return "native result"
    native = types.SimpleNamespace(dequantize_int8_convrot_weight_dtype=kernel)
    assert module._register_convrot_dequant(native)
    return callbacks["comfy_kitchen::dequantize_int8_convrot_weight_dtype"]


@pytest.mark.parametrize("shape", [(21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336),
                                  (1536, 6144), (6144, 6144), (16384, 6144), (6144, 16384)])
def test_measured_shapes_reach_dtype_aware_native(bridge, shape):
    function = install(bridge)
    q, scale = inputs(shape, offset=16)
    with torch.no_grad():
        assert function(q, scale, 256, 2) == "native result"
    assert bridge[2] == [("native", (q, scale, 256, torch.bfloat16))]


@pytest.mark.parametrize("change", ["cpu", "shape", "scale", "offset", "dtype", "group", "output", "grad"])
def test_unmeasured_contract_uses_original_registry(bridge, change):
    function = install(bridge)
    kwargs = {"device": "cpu"} if change == "cpu" else {}
    if change == "shape":
        kwargs["shape"] = (32, 256)
    if change == "scale":
        kwargs["scale_shape"] = (1,)
    if change == "offset":
        kwargs["offset"] = 1
    if change == "dtype":
        kwargs["dtype"] = torch.float16
    q, scale = inputs(**kwargs)
    group, output = (64 if change == "group" else 256), (0 if change == "output" else 2)
    with torch.set_grad_enabled(change == "grad"):
        assert function(q, scale, group, output) == "original result"
    assert bridge[2] == [("fallback", dict(q=q, scale=scale, group_size=group, output_dtype_code=output))]


@pytest.mark.parametrize("fatal", [False, True])
def test_native_failure_only_falls_back_when_safe(bridge, fatal):
    def fail(*args):
        raise RuntimeError("UR_RESULT_ERROR_DEVICE_LOST" if fatal else "unsupported native operation")
    function = install(bridge, fail)
    q, scale = inputs()
    with torch.no_grad():
        if fatal:
            with pytest.raises(RuntimeError, match="DEVICE_LOST"):
                function(q, scale, 256, 2)
            assert not bridge[2]
        else:
            assert function(q, scale, 256, 2) == "original result"
            assert len(bridge[2]) == 1


@pytest.mark.parametrize("reason", ["disabled", "existing_owner", "old_extension", "old_kitchen"])
def test_registration_respects_owner_and_availability(bridge, monkeypatch, reason):
    module, callbacks, _ = bridge
    native = types.SimpleNamespace(dequantize_int8_convrot_weight_dtype=object())
    if reason == "disabled":
        monkeypatch.setenv("OMNIXPU_CONVROT_DEQUANT", "0")
    elif reason == "existing_owner":
        monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", lambda *args: True)
    elif reason == "old_extension":
        native = types.SimpleNamespace()
    else:
        monkeypatch.setattr(torch.ops, "comfy_kitchen", types.SimpleNamespace())
    assert not module._register_convrot_dequant(native)
    assert not callbacks
