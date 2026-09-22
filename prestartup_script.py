"""Apply the Windows XPU allocator policy before ComfyUI imports torch."""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import NoReturn


_ENV_NAME = "OMNIXPU_XPU_MEMORY_FRACTION"
_MASTER_ENV_NAME = "OMNIXPU_ENABLE"
_DEFAULT_WINDOWS_FRACTION = 0.99
_DISABLED_VALUES = frozenset(("0", "disable", "disabled", "false", "no", "off"))
_LOG = logging.getLogger("ComfyUI-OmniXPU")


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"[OmniXPU] {_ENV_NAME}: {message}")


def _parse_fraction(raw_value: str) -> float:
    try:
        fraction = float(raw_value)
    except ValueError:
        _fail(f"expected a number in (0, 1], got {raw_value!r}")

    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        _fail(f"expected a finite number in (0, 1], got {raw_value!r}")
    return fraction


def _requested_fraction() -> tuple[float, str] | None:
    raw_value = os.environ.get(_ENV_NAME)
    if raw_value is not None:
        value = raw_value.strip()
        if not value or value.lower() in _DISABLED_VALUES:
            return None
        return _parse_fraction(value), "environment"

    if sys.platform == "win32":
        return _DEFAULT_WINDOWS_FRACTION, "Windows default"
    return None


def apply_xpu_memory_fraction() -> float | None:
    """Apply the caching-allocator fraction selected for this platform."""

    if os.environ.get(_MASTER_ENV_NAME, "1") == "0":
        _LOG.info(
            "[OmniXPU] XPU allocator memory fraction skipped because %s=0",
            _MASTER_ENV_NAME,
        )
        return None

    requested = _requested_fraction()
    if requested is None:
        return None
    fraction, source = requested

    try:
        import torch
    except Exception as exc:
        _fail(f"could not import torch: {exc}")

    xpu = getattr(torch, "xpu", None)
    try:
        available = xpu is not None and xpu.is_available()
    except Exception as exc:
        _fail(f"could not query torch.xpu availability: {exc}")
    if not available:
        _fail("torch.xpu is unavailable")

    setter = getattr(xpu, "set_per_process_memory_fraction", None)
    if not callable(setter):
        _fail("is unsupported by this PyTorch build")

    try:
        setter(fraction)
        getter = getattr(xpu, "get_per_process_memory_fraction", None)
        actual = float(getter()) if callable(getter) else fraction
    except Exception as exc:
        _fail(f"could not apply {fraction}: {exc}")

    _LOG.info(
        "[OmniXPU] XPU allocator memory fraction applied during prestartup: "
        "requested=%.12g actual=%.12g source=%s",
        fraction,
        actual,
        source,
    )
    return actual


apply_xpu_memory_fraction()


# ── XPU host-copy ("pinned memory") support ────────────────────────────────
# comfy gates its host-copy path to NVIDIA/AMD and registers that memory with
# torch.cuda.cudart().cudaHostRegister(), which the XPU torch build does not
# provide.  The AIMDO XPU provider already answers that driver call with a
# success no-op (src-xpu/dispatch.cpp: xpu_host_register -> CUDA_SUCCESS), so
# the only thing missing on XPU is walking the same comfy code with the
# equivalent no-op at the Python layer.
#
# Design constraints (keep the blast radius at zero on CUDA/ROCm):
#   * device gated   - the wrappers only divert when comfy.model_management
#                      reports an active Intel XPU device; every other device
#                      runs the untouched original function.
#   * name scoped    - the temporary cudart stand-in intercepts exactly the two
#                      host-register names and forwards every other attribute to
#                      the real object (on XPU those still raise exactly as they
#                      did before).
#   * call scoped    - the stand-in exists only while a wrapped call runs and is
#                      restored immediately afterwards; no global state is left.
#   * opt-in         - nothing happens unless OMNIXPU_ALLOW_HOST_PIN is set.
_HOST_PIN_ENV = "OMNIXPU_ALLOW_HOST_PIN"
_HOST_PIN_BUDGET_ENV = "OMNIXPU_PIN_BUDGET_MB"
_HOST_PIN_TRACE_ENV = "OMNIXPU_HOST_PIN_TRACE"


def _host_pin_budget() -> tuple[int, str]:
    """Host-copy budget: comfy's own rule for CUDA/ROCm (ram * 0.40 on Windows).

    OMNIXPU_PIN_BUDGET_MB overrides it when set.  Zero means "comfy default",
    i.e. the value comfy would have chosen for a supported device.
    """
    override = os.environ.get(_HOST_PIN_BUDGET_ENV, "").strip()
    if override:
        return max(0, int(override)) * 1024 * 1024, "env"

    import torch
    import comfy.model_management as _mm

    ram = _mm.get_total_memory(torch.device("cpu"))
    if getattr(_mm, "WINDOWS", os.name == "nt"):
        return int(ram * 0.40), "ram*0.40 (comfy rule for CUDA/ROCm on Windows)"

    swap = 0
    try:
        import comfy.system_memory as _sys_mem

        if _sys_mem.cgroup_memory_limit() is None:
            swap = _mm.get_disk_swap_total()
    except Exception:  # noqa: BLE001 - budget only, never fatal
        pass
    budget = max(ram * 0.40,
                 min(ram * 0.90, ram - 4 * 1024 ** 3, ram + swap - 16 * 1024 ** 3))
    return int(budget), "comfy rule for CUDA/ROCm on Linux"


def _xpu_cudart_window():
    """Context manager: make the two host-register calls succeed, forward the rest."""
    import contextlib

    import torch

    real_cudart = torch.cuda.cudart

    class _HostRegisterStandIn:
        def cudaHostRegister(self, ptr, size, flags=0):
            return 0

        def cudaHostUnregister(self, ptr):
            return 0

        def __getattr__(self, name):
            return getattr(real_cudart(), name)

    @contextlib.contextmanager
    def window():
        torch.cuda.cudart = lambda: _HostRegisterStandIn()
        try:
            yield
        finally:
            torch.cuda.cudart = real_cudart

    return window()


def _device_is_xpu() -> bool:
    try:
        import comfy.model_management as _mm

        return bool(_mm.is_intel_xpu())
    except Exception:  # noqa: BLE001
        return False


def _wrap_xpu_only(owner, name: str) -> bool:
    original = getattr(owner, name, None)
    if original is None or getattr(original, "_omnixpu_host_pin", False):
        return False

    def wrapper(*args, **kwargs):
        if not _device_is_xpu():
            return original(*args, **kwargs)
        with _xpu_cudart_window():
            return original(*args, **kwargs)

    wrapper._omnixpu_host_pin = True
    wrapper.__wrapped__ = original
    setattr(owner, name, wrapper)
    return True


def _install_host_pin_trace():
    """Count which file-read entry the loader uses (verification aid)."""
    import atexit

    try:
        import comfy_aimdo.host_buffer as _hb
    except Exception as _trace_import_exc:  # noqa: BLE001
        _LOG.warning("[OmniXPU] host pin trace unavailable: %r", _trace_import_exc)
        return None

    counters = {"reader_calls": 0, "reader_bytes": 0,
                "slice_calls": 0, "slice_bytes": 0}
    state = {"reported_at": 0}

    def report(tag: str) -> None:
        try:
            import comfy.model_management as _mm

            pinned_mb = _mm.TOTAL_PINNED_MEMORY / (1024 ** 2)
            regions = len(_mm.PINNED_MEMORY)
        except Exception:  # noqa: BLE001
            pinned_mb, regions = -1.0, -1
        _LOG.info(
            "[OmniXPU] host pin trace%s: reader=%d (%.2f GB) slice=%d (%.2f GB) "
            "pinned_total=%.1f MB legacy_regions=%d",
            tag,
            counters["reader_calls"], counters["reader_bytes"] / (1024 ** 3),
            counters["slice_calls"], counters["slice_bytes"] / (1024 ** 3),
            pinned_mb, regions,
        )

    def maybe_report() -> None:
        total = counters["reader_calls"] + counters["slice_calls"]
        if total <= 4 or total - state["reported_at"] >= 500:
            state["reported_at"] = total
            report("" if total > 4 else " (first calls)")

    reader = getattr(_hb, "read_file_to_device", None)
    patched_reader = False
    if reader is not None and not getattr(reader, "_omnixpu_trace", False):
        def traced_reader(file_obj, file_offset, size, *args, **kwargs):
            counters["reader_calls"] += 1
            counters["reader_bytes"] += int(size)
            maybe_report()
            return reader(file_obj, file_offset, size, *args, **kwargs)

        traced_reader._omnixpu_trace = True
        _hb.read_file_to_device = traced_reader
        patched_reader = True

    slice_reader = getattr(_hb.HostBuffer, "read_file_slice", None)
    patched_slice = False
    if slice_reader is not None and not getattr(slice_reader, "_omnixpu_trace", False):
        def traced_slice(self, file_obj, file_offset, size, *args, **kwargs):
            counters["slice_calls"] += 1
            counters["slice_bytes"] += int(size)
            maybe_report()
            return slice_reader(self, file_obj, file_offset, size, *args, **kwargs)

        traced_slice._omnixpu_trace = True
        _hb.HostBuffer.read_file_slice = traced_slice
        patched_slice = True

    _LOG.info(
        "[OmniXPU] host pin trace installed: module=%s reader=%s slice=%s",
        getattr(_hb, "__file__", "?"), patched_reader, patched_slice,
    )

    atexit.register(report, " (final)")
    return counters


def apply_xpu_host_pin():
    """Enable comfy's host-copy path on XPU (opt-in, device gated)."""
    # The trace is independent of the gate: it also has to show the baseline
    # (disk-backed) behaviour while the host-copy path stays disabled.
    tracing = os.environ.get(_HOST_PIN_TRACE_ENV, "0").strip().lower() not in _DISABLED_VALUES
    if tracing:
        _install_host_pin_trace()

    raw = os.environ.get(_HOST_PIN_ENV)
    if raw is None or raw.strip().lower() in _DISABLED_VALUES:
        return {"status": "disabled", "reason": f"{_HOST_PIN_ENV} not set",
                "trace": tracing}

    import comfy.model_management as _mm
    import comfy.model_patcher as _mp
    import comfy.pinned_memory as _pm

    budget, source = _host_pin_budget()
    _mm.MAX_PINNED_MEMORY = budget

    wrapped = []
    for owner, name in (
        (_pm, "get_pin"),
        (_pm, "pin_memory"),
        (_mm, "pin_memory"),
        (_mm, "unpin_memory"),
        (_mp.ModelPatcherDynamic, "unregister_inactive_pins"),
    ):
        if _wrap_xpu_only(owner, name):
            wrapped.append(f"{getattr(owner, '__name__', type(owner).__name__)}.{name}")

    return {
        "status": "enabled",
        "budget_mb": budget // (1024 ** 2),
        "budget_source": source,
        "wrapped": wrapped,
        "trace": tracing,
    }


def apply_runtime_providers():
    """Load the local bootstrap without importing the custom-node package.

    Must run after ComfyUI's import of ``comfy_aimdo.control`` (the official
    attempt) and before ``control.init_devices()``.
    """
    import importlib.util
    from pathlib import Path

    runtime_path = Path(__file__).with_name("runtime_bootstrap.py")
    if not runtime_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "_comfyui_omnixpu_runtime_bootstrap", runtime_path
    )
    module = importlib.util.module_from_spec(spec)
    # Register before execution: dataclasses resolves field types through
    # sys.modules[cls.__module__], and an unregistered synthetic module makes
    # @dataclass raise AttributeError('NoneType' object has no attribute ...).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.bootstrap()


try:
    _provider_state = apply_runtime_providers()
    if _provider_state is not None:
        _LOG.info("[OmniXPU] runtime providers: %s", _provider_state)
except Exception as _provider_exc:  # noqa: BLE001 - startup policy decides
    _mode = os.environ.get(_MASTER_ENV_NAME, "auto").strip().lower()
    if _mode == "required":
        raise
    import traceback as _traceback

    _traceback.print_exc()
    _LOG.warning("[OmniXPU] runtime provider activation skipped: %s", _provider_exc)


try:
    _host_pin_state = apply_xpu_host_pin()
    if _host_pin_state is not None:
        _LOG.info("[OmniXPU] host pin: %s", _host_pin_state)
except Exception as _host_pin_exc:  # noqa: BLE001 - startup policy decides
    import traceback as _host_pin_traceback

    _host_pin_traceback.print_exc()
    _LOG.warning("[OmniXPU] host pin setup skipped: %s", _host_pin_exc)
