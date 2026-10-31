"""Install the vendored Kitchen XPU backend into ``comfy_kitchen``.

ComfyUI 0.43.x requires ``comfy-kitchen>=0.2.40``.  That release changed what an
XPU backend has to provide, and the published wheel still ships only ``ascend``,
``cuda``, ``eager``, ``hip`` and ``triton`` — there is no XPU backend in the
distribution at all, and its ``__init__.py`` never imports one.

Intel carries the XPU backend out of band in ``xiangyuT/comfy-kitchen-xpu``.  The
``llm-scaler-omni`` image installs it as a separate ``comfy-kitchen-xpu-runtime``
provider wheel that re-homes the whole ``comfy_kitchen`` tree behind
``comfy_kitchen_xpu_runtime._vendor`` and needs a prestartup meta-path hook — a
third component to install and keep in step with ``comfy-kitchen``.

This module grafts the same backend from ``vendor/kitchen/`` at plugin import, so
the historical two-component install (ComfyUI-OmniXPU + omni_xpu_kernel) keeps an
XPU backend on 0.2.40 and later.  Only modules the official wheel lacks are
added; no installed file is replaced.

Set ``OMNIXPU_KITCHEN_GRAFT=0`` to skip the graft and keep stock behaviour.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("ComfyUI-OmniXPU")

GRAFT_ENV = "OMNIXPU_KITCHEN_GRAFT"
REQUIRED_KITCHEN = (0, 2, 40)

# Mirror of the upstream package layout, so the vendored tree reads exactly like
# comfy_kitchen/ and the diff against a future upstream release stays obvious.
_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "kitchen"
_VENDOR_PREFIX = "comfy_kitchen."

# backends/xpu imports the top-level and eager gguf / svdquant_w4a16 modules, so
# those have to be in sys.modules first.
_ORDER = (
    "gguf.py",
    "svdquant_w4a16.py",
    "backends/eager/gguf.py",
    "backends/eager/svdquant_w4a16.py",
    "backends/xpu/__init__.py",
)

_STATE = {"status": "not-run", "detail": ""}


def get_state() -> dict:
    return dict(_STATE)


def _set(status: str, detail: str = "") -> None:
    _STATE.update({"status": status, "detail": detail})


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(raw).split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _kitchen_version(kitchen) -> str:
    try:
        from importlib.metadata import version

        return version("comfy-kitchen")
    except Exception:
        return str(getattr(kitchen, "__version__", ""))


def _vendor_modules():
    """Yield (dotted official name, source file) in dependency order."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    for relative in _ORDER:
        path = _VENDOR_ROOT / relative
        if path.is_file():
            ordered.append(path)
            seen.add(path)
    for path in sorted(_VENDOR_ROOT.rglob("*.py")):
        if path not in seen:
            ordered.append(path)
    for path in ordered:
        relative = path.relative_to(_VENDOR_ROOT)
        if relative.as_posix() == "__init__.py":
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        yield _VENDOR_PREFIX + ".".join(parts), path


def _package_dirs():
    """Directories the backend expects on ``comfy_kitchen.__path__``."""
    return [
        _VENDOR_ROOT,
        _VENDOR_ROOT / "backends",
        _VENDOR_ROOT / "backends" / "eager",
        _VENDOR_ROOT / "backends" / "xpu",
    ]


def _ensure_search_path(kitchen) -> None:
    """Let ``comfy_kitchen.backends.xpu`` resolve to the vendored directory."""
    package_path = list(getattr(kitchen, "__path__", []))
    added = False
    for directory in _package_dirs():
        if directory.is_dir() and str(directory) not in package_path:
            package_path.append(str(directory))
            added = True
    if added:
        kitchen.__path__ = package_path


def _has_backend(kitchen) -> bool:
    try:
        return "xpu" in kitchen.list_backends()
    except Exception:
        return False


def _prefer_xpu() -> None:
    """Put ``xpu`` first in the registry dispatch order.

    Registering the backend is not enough: the registry only dispatches to
    backends in its priority list, and the stock order is
    ``["ascend", "cuda", "triton", "eager"]``.  Without this the Triton and Eager
    backends keep answering every call the XPU backend advertises.  Measured on
    an Arc A770 that costs the entire gain (1.128 s/it against 1.010 s/it) while
    every log line still reports the backend as available.
    """
    try:
        from comfy_kitchen.registry import registry

        priority = list(getattr(registry, "_priority", ()))
        if "xpu" not in priority:
            priority.insert(0, "xpu")
            registry.set_priority(priority)
            log.info("[OmniXPU] Kitchen backend priority: %s", priority)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("[OmniXPU] Kitchen XPU priority setup failed: %s", exc)


def install() -> bool:
    """Graft the XPU backend onto the installed ``comfy_kitchen``.

    Returns True when an XPU backend is registered afterwards.  Safe to call
    repeatedly; every failure is non-fatal so a stock install keeps working.
    """
    if os.environ.get(GRAFT_ENV, "1").strip().lower() in ("0", "false", "no", "off"):
        _set("skipped", f"{GRAFT_ENV}=0")
        return False

    try:
        import comfy_kitchen as kitchen
    except ImportError as exc:
        _set("skipped", f"comfy_kitchen not installed: {exc}")
        return False

    if _has_backend(kitchen):
        _set("present", "a Kitchen XPU backend is already registered")
        return True

    version = _kitchen_version(kitchen)
    if _version_tuple(version)[:3] < REQUIRED_KITCHEN:
        _set(
            "skipped",
            f"comfy-kitchen {version or 'unknown'} predates the 0.2.40 backend API",
        )
        log.info(
            "[OmniXPU] Kitchen XPU graft skipped: comfy-kitchen %s predates 0.2.40",
            version or "unknown",
        )
        return False

    try:
        _ensure_search_path(kitchen)
    except Exception as exc:  # pragma: no cover - defensive
        _set("failed", f"cannot extend comfy_kitchen.__path__: {exc}")
        log.warning("[OmniXPU] Kitchen XPU graft failed: %s", exc)
        return False

    grafted: list[str] = []
    for name, path in _vendor_modules():
        if name in sys.modules:
            continue
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            grafted.append(name)
        except Exception as exc:
            if name.endswith("backends.xpu"):
                sys.modules.pop(name, None)
                _set("failed", f"{name}: {exc}")
                log.warning("[OmniXPU] Kitchen XPU graft failed: %s: %s", name, exc)
                return False
            # Optional modules (gguf / svdquant) may want a newer comfy_kitchen.
            sys.modules.pop(name, None)
            log.debug("[OmniXPU] Kitchen XPU graft skipped module %s: %s", name, exc)

    if not _has_backend(kitchen):
        _set("failed", "backend import completed without registering 'xpu'")
        log.warning("[OmniXPU] Kitchen XPU graft did not register an 'xpu' backend")
        return False

    _prefer_xpu()

    capabilities = kitchen.list_backends().get("xpu", {}).get("capabilities") or []
    _set("grafted", f"{len(grafted)} modules, {len(capabilities)} capabilities")
    log.info(
        "[OmniXPU] Kitchen XPU backend grafted onto comfy-kitchen %s: "
        "%d modules, %d capabilities",
        version,
        len(grafted),
        len(capabilities),
    )
    return True


__all__ = ["install", "get_state", "GRAFT_ENV", "REQUIRED_KITCHEN"]
