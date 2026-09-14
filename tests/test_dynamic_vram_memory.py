"""AIMDO foreign residency must be visible to Comfy's existing memory policy."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch):
    aimdo = ModuleType("comfy_aimdo")
    aimdo.model_vbar = None
    monkeypatch.setitem(sys.modules, "comfy_aimdo", aimdo)
    path = Path(__file__).resolve().parents[1] / "adapters/dynamic_vram.py"
    spec = importlib.util.spec_from_file_location("_dynamic_vram_memory_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_memory(adapter, physical=700, active=100, reserved=1100):
    xpu = SimpleNamespace(type="xpu", index=0)
    calls = []

    def original(device=None, torch_free_too=False):
        calls.append((device, torch_free_too))
        return (15900, 1000) if torch_free_too else 15900

    mm = SimpleNamespace(get_free_memory=original, get_torch_device=lambda: xpu)
    torch = SimpleNamespace(xpu=SimpleNamespace(
        mem_get_info=lambda dev: (physical, 16000),
        memory_stats=lambda dev: {"active_bytes.all.current": active,
                                  "reserved_bytes.all.current": reserved}))
    adapter._patch_free_memory(mm, torch)
    return mm, torch, xpu, calls


def test_foreign_residency_limits_available_memory(adapter):
    mm, _, xpu, calls = setup_memory(adapter)
    assert mm.get_free_memory(xpu) == 1700
    assert mm.get_free_memory(None, torch_free_too=True) == (1700, 1000)
    assert calls == []


def test_other_devices_keep_original_contract(adapter):
    mm, _, _, calls = setup_memory(adapter)
    cpu = SimpleNamespace(type="cpu")
    assert mm.get_free_memory(cpu, True) == (15900, 1000)
    assert calls == [(cpu, True)]


def test_repeated_install_is_idempotent(adapter):
    mm, torch, _, _ = setup_memory(adapter)
    wrapped = mm.get_free_memory
    adapter._patch_free_memory(mm, torch)
    assert mm.get_free_memory is wrapped


def test_device_query_failure_is_not_reported_as_free_capacity(adapter):
    mm, torch, _, _ = setup_memory(adapter)
    def failed(device):
        raise RuntimeError("device lost")
    torch.xpu.mem_get_info = failed
    with pytest.raises(RuntimeError, match="device lost"):
        mm.get_free_memory()


def test_correct_budget_activates_existing_inactive_model_trim(adapter):
    mm, torch, xpu, _ = setup_memory(adapter, physical=700, active=100, reserved=100)
    requested = SimpleNamespace(is_dynamic=lambda: True, load_device=xpu)
    physical = [700]
    freed = []

    def unload(device, size):
        physical[0] += size
        freed.append(size)
        return size

    inactive = SimpleNamespace(is_dynamic=lambda: True, loaded_size=lambda: 3000,
                               partially_unload=unload, offload_device="cpu")
    torch.xpu.mem_get_info = lambda dev: (physical[0], 16000)
    mm.current_loaded_models = [SimpleNamespace(model=inactive, device=xpu,
                                                is_dead=lambda: False)]
    # Use a hashable device, just as torch.device is hashable in production.
    class Device:
        type = "xpu"
        index = 0
    device = Device()
    requested.load_device = device
    mm.current_loaded_models[0].device = device
    adapter._trim_dynamic_boundary(mm, [requested], 2000)
    assert freed == [1300]
    assert mm.get_free_memory(device) == 2000
