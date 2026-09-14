"""Weight ownership and fallback contracts, without initializing an accelerator."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


class QuantizedWeight:
    _layout_cls = "TensorWiseINT8Layout"

    def __init__(self, value, scale=0.25, convrot=False):
        self._qdata = torch.full((3, 4), value, dtype=torch.int8)
        self._params = types.SimpleNamespace(
            scale=torch.tensor(scale), convrot=convrot,
            convrot_groupsize=64, transposed=False,
        )

    def dequantize(self):
        return self._qdata.float() * self._params.scale


class Layout:
    @staticmethod
    def get_plain_tensors(weight):
        return weight._qdata, weight._params.scale


@pytest.fixture
def adapter(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    package = "_omnixpu_int8_cast_test"
    for name, directory in (
        (package, root), (package + ".adapters", root / "adapters"),
        (package + ".patches", root / "patches"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, name, module)
    debug = types.ModuleType(package + ".patches.debug")
    debug.log_debug_event = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, debug.__name__, debug)
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.model_management.in_training = False
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", comfy.model_management)
    name = package + ".adapters.fp8_gemm"
    spec = importlib.util.spec_from_file_location(name, root / "adapters/fp8_gemm.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def cast_ops(weights, events):
    remaining = iter(weights)
    lease = object()

    def cast(module, x, **kwargs):
        assert kwargs == dict(offloadable=True, compute_dtype=x.dtype, want_requant=True)
        weight, bias = next(remaining)
        events.append(("cast", weight, bias))
        return weight, bias, lease

    def uncast(module, weight, bias, state):
        assert state is lease
        events.append(("uncast", weight, bias))

    return types.SimpleNamespace(
        cast_bias_weight=cast, uncast_bias_weight=uncast,
        INPUT_ACT_EAGER={"swiglu": lambda x: torch.nn.functional.silu(x[..., :4]) * x[..., 4:]},
    )


def test_uses_fresh_cast_weight_and_bias_after_lora_or_refault(adapter):
    events = []
    weights = [(QuantizedWeight(2, convrot=True), torch.ones(3)),
               (QuantizedWeight(7, convrot=True), torch.full((3,), 2.0))]
    ops = cast_ops(weights, events)
    module = types.SimpleNamespace(weight=QuantizedWeight(-30), bias=torch.zeros(3))
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    def kernel(value, qdata, scale, bias, **kwargs):
        assert events[-1][0] == "cast"
        assert qdata is events[-1][1]._qdata
        assert kwargs == dict(out_dtype=x.dtype, convrot=True,
                              convrot_groupsize=64, input_act=None)
        events.append(("kernel", qdata))
        return torch.nn.functional.linear(value, qdata.float() * scale, bias)

    adapter._omni_int8 = types.SimpleNamespace(int8_linear=kernel)
    for weight, bias in weights:
        actual = adapter._int8_forward_cast(ops, module, x, QuantizedWeight, Layout)
        torch.testing.assert_close(actual, torch.nn.functional.linear(x, weight.dequantize(), bias))
        assert events[-1] == ("uncast", weight, bias)
    assert [event[0] for event in events] == ["cast", "kernel", "uncast"] * 2


@pytest.mark.parametrize("failure", [None, RuntimeError("unsupported kernel")])
@pytest.mark.parametrize("activation", [None, "swiglu"])
def test_fallback_reuses_cast_without_reapplying_patch(adapter, failure, activation):
    events = []
    weight, bias = QuantizedWeight(5), torch.ones(3)
    ops = cast_ops([(weight, bias)], events)
    module = types.SimpleNamespace(weight=QuantizedWeight(-10))
    x = torch.arange(8 if activation else 4, dtype=torch.float32).reshape(1, -1)

    def kernel(*args, **kwargs):
        events.append(("kernel",))
        if failure is not None:
            raise failure
        return None

    adapter._omni_int8 = types.SimpleNamespace(int8_linear=kernel)
    actual = adapter._int8_forward_cast(ops, module, x, QuantizedWeight, Layout, activation)
    activated = ops.INPUT_ACT_EAGER[activation](x) if activation else x
    torch.testing.assert_close(actual, torch.nn.functional.linear(activated, weight.dequantize(), bias))
    assert [event[0] for event in events] == ["cast", "kernel", "uncast"]


def test_dense_cast_result_is_used_directly(adapter):
    events = []
    weight = torch.full((3, 4), 1.5)
    ops = cast_ops([(weight, None)], events)
    x = torch.ones(2, 4)
    adapter._omni_int8 = types.SimpleNamespace(int8_linear=lambda *args, **kwargs: pytest.fail("dense cast entered INT8 kernel"))
    actual = adapter._int8_forward_cast(ops, object(), x, QuantizedWeight, Layout)
    torch.testing.assert_close(actual, torch.full((2, 3), 6.0))
    assert [event[0] for event in events] == ["cast", "uncast"]


def test_fatal_device_error_releases_lease_without_retry(adapter):
    events = []
    weight = QuantizedWeight(3)
    ops = cast_ops([(weight, None)], events)

    def kernel(*args, **kwargs):
        events.append(("kernel",))
        raise RuntimeError("UR_RESULT_ERROR_DEVICE_LOST")

    adapter._omni_int8 = types.SimpleNamespace(int8_linear=kernel)
    with pytest.raises(RuntimeError, match="DEVICE_LOST"):
        adapter._int8_forward_cast(ops, object(), torch.ones(2, 4), QuantizedWeight, Layout)
    assert [event[0] for event in events] == ["cast", "kernel", "uncast"]


def test_dense_fallback_failure_releases_lease(adapter):
    events = []
    ops = cast_ops([(torch.ones(3, 5), None)], events)
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        adapter._int8_forward_cast(ops, object(), torch.ones(2, 4), QuantizedWeight, Layout)
    assert [event[0] for event in events] == ["cast", "uncast"]


@pytest.mark.parametrize("reason", ["training", "transposed_weight", "pre_quant_scale_set", "input_scale_set"])
def test_semantic_guards_before_cast(adapter, reason):
    adapter._omni_int8 = object()
    module = types.SimpleNamespace(
        weight=QuantizedWeight(2), quant_format="int8_tensorwise",
        weight_function=[], bias_function=[],
    )
    x = types.SimpleNamespace(is_xpu=True, ndim=2, requires_grad=False)
    if reason == "training":
        x.requires_grad = True
    elif reason == "transposed_weight":
        module.weight._params.transposed = True
    else:
        setattr(module, reason.removesuffix("_set"), torch.tensor(1.0))
    assert adapter._int8_skip_reason(module, x, QuantizedWeight, Layout) == reason
