"""Krea2 routing/ownership tests; no accelerator is initialized here."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    package = "_omnixpu_krea_rope_test"
    for name, directory in ((package, root),
                            (package + ".adapters", root / "adapters"),
                            (package + ".patches", root / "patches")):
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, name, module)
    debug = types.ModuleType(package + ".patches.debug")
    debug.log_debug_event = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, debug.__name__, debug)
    name = package + ".adapters.rms_rope"
    spec = importlib.util.spec_from_file_location(name, root / "adapters/rms_rope.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class TensorSpec:
    def __init__(self, shape, dtype=torch.bfloat16, device="xpu:0"):
        self.shape, self.dtype, self.device = shape, dtype, torch.device(device)
        self.requires_grad, self.contiguous = False, True

    def is_contiguous(self):
        return self.contiguous


@pytest.fixture
def contract(adapter, monkeypatch):
    spec_torch = types.SimpleNamespace(Tensor=TensorSpec, bfloat16=torch.bfloat16,
        float32=torch.float32, nn=torch.nn, is_grad_enabled=lambda: False)
    monkeypatch.setattr(adapter, "torch", spec_torch)
    def norm():
        return types.SimpleNamespace(scale=TensorSpec((128,)), eps=1e-5,
                                     _forward_hooks={}, _forward_pre_hooks={})
    qknorm = types.SimpleNamespace(qnorm=norm(), knorm=norm(),
                                  _forward_hooks={}, _forward_pre_hooks={})
    attention = types.SimpleNamespace(heads=48, kvheads=12, headdim=128, qknorm=qknorm)
    x = TensorSpec((1, 4192, 6144))
    freqs = TensorSpec((1, 1, 4192, 64, 2, 2), dtype=torch.float32)
    return attention, x, freqs, spec_torch


def test_exact_contract_and_transformer_patches(adapter, contract):
    attention, x, freqs, _ = contract
    assert adapter._krea2_input_contract(attention, x, freqs, {})
    assert not adapter._krea2_input_contract(attention, x, freqs,
                                           {"patches": {"attn1_patch": [object()]}})
    assert not adapter._krea2_input_contract(attention, x, None, {})


@pytest.mark.parametrize("target,field,value", [
    ("x", "shape", (1, 4191, 6144)),
    ("x", "dtype", torch.float16),
    ("x", "device", torch.device("cpu")),
    ("x", "requires_grad", True),
    ("freqs", "shape", (1, 1, 4191, 64, 2, 2)),
    ("freqs", "dtype", torch.bfloat16),
    ("freqs", "device", torch.device("xpu:1")),
    ("freqs", "contiguous", False),
    ("freqs", "requires_grad", True),
    ("attention", "kvheads", 48),
])
def test_outside_measured_contract_falls_back(adapter, contract, target, field, value):
    attention, x, freqs, _ = contract
    setattr({"attention": attention, "x": x, "freqs": freqs}[target], field, value)
    assert not adapter._krea2_input_contract(attention, x, freqs, {})


@pytest.mark.parametrize("which", ["qknorm", "qnorm", "knorm"])
@pytest.mark.parametrize("hook", ["_forward_hooks", "_forward_pre_hooks"])
def test_intermediate_module_hooks_preserve_original_route(adapter, contract, which, hook):
    attention, x, freqs, _ = contract
    module = attention.qknorm if which == "qknorm" else getattr(attention.qknorm, which)
    setattr(module, hook, {1: object()})
    assert not adapter._krea2_input_contract(attention, x, freqs, {})


@pytest.mark.parametrize("hook", ["_global_forward_hooks", "_global_forward_pre_hooks"])
def test_global_hooks_preserve_original_route(adapter, contract, monkeypatch, hook):
    attention, x, freqs, _ = contract
    monkeypatch.setattr(torch.nn.modules.module, hook, {1: object()})
    assert not adapter._krea2_input_contract(attention, x, freqs, {})


@pytest.mark.parametrize("change", ["grad", "scale_shape", "scale_dtype", "epsilon"])
def test_normalization_semantics_are_guarded(adapter, contract, change):
    attention, x, freqs, spec_torch = contract
    if change == "grad":
        spec_torch.is_grad_enabled = lambda: True
    elif change == "scale_shape":
        attention.qknorm.qnorm.scale.shape = (127,)
    elif change == "scale_dtype":
        attention.qknorm.knorm.scale.dtype = torch.int8
    else:
        attention.qknorm.knorm.eps = 1e-6
    assert not adapter._krea2_input_contract(attention, x, freqs, {})


@pytest.mark.parametrize("mode", ["native", "kernel_error", "layout", "fatal", "contract"])
def test_router_preserves_projection_ownership_and_fallback(adapter, monkeypatch, mode):
    events = []
    freqs, mask, options = object(), object(), {"block_index": 4}
    x = types.SimpleNamespace(device=torch.device("cpu"))
    scales = [torch.linspace(-0.25, 0.75, 128, dtype=torch.bfloat16),
              torch.linspace(0.5, -0.5, 128, dtype=torch.bfloat16)]

    def projection(name, width):
        def project(value):
            assert value is x
            events.append(name)
            shape = (1, 1 if mode == "layout" and name == "wq" else 4192, width)
            return torch.empty(shape, device="meta", dtype=torch.bfloat16).expand(1, 4192, width)
        return project

    class Norm:
        qnorm = types.SimpleNamespace(scale=scales[0])
        knorm = types.SimpleNamespace(scale=scales[1])

        def __call__(self, q, k):
            events.append("qknorm")
            return q, k

    def cast(scale, dtype, device):
        events.append("cast")
        assert device == x.device and dtype == torch.float32
        return scale.to(dtype)

    def native(q, k, pe, qs, ks, eps):
        events.append("native")
        assert pe is freqs and eps == 1e-5
        assert q.shape == (1, 48, 4192, 128)
        assert k.shape == (1, 12, 4192, 128)
        torch.testing.assert_close(qs, scales[0].float() + 1.0, rtol=0, atol=0)
        torch.testing.assert_close(ks, scales[1].float() + 1.0, rtol=0, atol=0)
        if mode in ("kernel_error", "fatal"):
            raise RuntimeError("UR_RESULT_ERROR_DEVICE_LOST" if mode == "fatal" else "unsupported kernel")
        return q, k

    def rope(q, k, pe):
        events.append("rope")
        assert pe is freqs
        return q, k

    def attention(q, k, v, heads, **kwargs):
        events.append("attention")
        assert heads == 48 and q.shape == k.shape == v.shape == (1, 48, 4192, 128)
        assert kwargs == {"mask": mask, "skip_reshape": True, "transformer_options": options}
        return torch.empty((1, 4192, 6144), device="meta", dtype=torch.bfloat16)

    def output(value):
        events.append("wo")
        assert value.shape == (1, 4192, 6144)
        return value

    sentinel = object()
    def original(self, value, pe, attn_mask, opts):
        events.append("original")
        assert (value, pe, attn_mask, opts) == (x, freqs, mask, options)
        return sentinel

    model = types.SimpleNamespace(Attention=types.SimpleNamespace(forward=original),
        comfy=types.SimpleNamespace(model_management=types.SimpleNamespace(cast_to=cast)),
        apply_rope=rope, optimized_attention_masked=attention)
    instance = types.SimpleNamespace(wq=projection("wq", 6144), wk=projection("wk", 1536),
        wv=projection("wv", 1536), gate=projection("gate", 6144), qknorm=Norm(), heads=48, wo=output)
    monkeypatch.setattr(adapter, "_krea2_input_contract", lambda *args: mode != "contract")
    router = adapter._make_krea2_router(types.SimpleNamespace(rms_kitchen_rope=native), model)
    assert getattr(router, adapter._PATCH_MARKER) is original
    if mode == "fatal":
        with pytest.raises(RuntimeError, match="DEVICE_LOST"):
            router(instance, x, freqs, mask, options)
        assert events == ["wq", "wk", "wv", "gate", "cast", "cast", "native"]
        return
    result = router(instance, x, freqs, mask, options)
    if mode == "contract":
        assert result is sentinel and events == ["original"]
        return
    expected = ["wq", "wk", "wv", "gate"]
    if mode != "layout":
        expected += ["cast", "cast", "native"]
    if mode != "native":
        expected += ["qknorm", "rope"]
    assert events == expected + ["attention", "wo"]
    assert adapter.get_stats()["krea2_routed"] == (mode == "native")
