"""A770 RMS-RoPE bridge: routing and fallback regression test.

Runs standalone (``python tests/test_rms_rope_bridge.py``) or under pytest.
Requires an A770/DG2 omni_xpu_kernel wheel and ComfyUI's comfy_kitchen.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent


def _load_package_modules():
    """Load only the adapter module with empty parent namespaces.

    Executing the real package ``__init__`` would auto-apply every adapter
    through ``patches.apply_all_patches`` and create a second module instance.
    """
    package = "ComfyUI-OmniXPU"
    for name, path in (
        (package, _ROOT),
        (f"{package}.patches", _ROOT / "patches"),
        (f"{package}.adapters", _ROOT / "adapters"),
    ):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod

    name = f"{package}.adapters.rms_rope"
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "adapters" / "rms_rope.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_h3_contract(seq: int = 256, seed: int = 7, rot_dim: int = 96):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    qkv = torch.randn((seq, 3 * 7168), dtype=torch.bfloat16, generator=gen).to("xpu")
    q = qkv[:, :7168].view(1, seq, 56, 128)
    k = qkv[:, 7168 : 2 * 7168].view(1, seq, 56, 128)
    angles = torch.randn((seq, rot_dim), dtype=torch.float32, generator=gen).to("xpu") * 0.5
    half = angles[:, : rot_dim // 2]
    c, s = torch.cos(half), torch.sin(half)
    freqs = (
        torch.stack([c, -s, s, c], dim=-1)
        .reshape(1, seq, 1, rot_dim // 2, 2, 2)
        .to(torch.bfloat16)
    )
    qw = torch.randn(128, dtype=torch.bfloat16, generator=gen).to("xpu")
    kw = torch.randn(128, dtype=torch.bfloat16, generator=gen).to("xpu")
    return qkv, q, k, freqs, qw, kw


def _run_standalone():
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is required")
    module = _load_package_modules()
    import comfy_kitchen as ck

    ok, reason = module.apply()
    assert ok, reason

    patched = ck.rms_rope_split_half_
    original = getattr(patched, module._PATCH_MARKER)

    # Native path (patched wrapper)
    qkv_a, q_a, k_a, freqs, qw, kw = _make_h3_contract()
    module.get_stats()  # ensure counters exist
    if not module._h3_contract(q_a, k_a, freqs, qw, kw, 96):
        print("contract debug:", q_a.shape, q_a.stride(), k_a.stride(),
              freqs.shape, freqs.dtype, qw.shape, qw.dtype)
    patched(q_a, k_a, freqs, qw, kw, epsilon=1e-5, rot_dim=96)

    # Eager path (saved original)
    qkv_b, q_b, k_b, freqs2, qw2, kw2 = _make_h3_contract()
    original(q_b, k_b, freqs2, qw2, kw2, epsilon=1e-5, rot_dim=96)

    qa = qkv_a[:, :7168].view(1, 256, 56, 128)
    qb = qkv_b[:, :7168].view(1, 256, 56, 128)
    dq = (qa.float() - qb.float()).abs().max().item()
    assert dq <= 0.07, f"native/eager mismatch too large: {dq}"

    stats = module.get_stats()
    assert stats["routed"] >= 1, stats

    # Non-H3 contract must fall back through the original eager path.
    qkv_c, q_c, k_c, freqs3, qw3, kw3 = _make_h3_contract(seq=64, rot_dim=64)
    before = module.get_stats()["fallback"]
    patched(q_c, k_c, freqs3, qw3, kw3, epsilon=1e-5, rot_dim=64)
    assert module.get_stats()["fallback"] > before

    print("rms_rope bridge OK: routed", stats["routed"], "max_abs", dq)


def test_rms_rope_bridge():
    _run_standalone()


def test_zimage_routing_and_exact_shape_fallback(monkeypatch):
    import comfy_kitchen as ck
    ck.disable_backend("triton")
    monkeypatch.setenv("OMNIXPU_ZIMAGE_RMS_ROPE", "1")
    module = _load_package_modules()
    ok, reason = module.apply()
    assert ok, reason
    patched = ck.rms_rope
    original = getattr(patched, module._PATCH_MARKER)
    gen = torch.Generator().manual_seed(119)
    packed = torch.randn((1, 4256, 3, 30, 128), dtype=torch.bfloat16, generator=gen).to("xpu")
    q, k = packed[:, :, 0], packed[:, :, 1]
    angle = torch.randn((1, 4256, 1, 64), generator=gen).to("xpu")
    freqs = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), -1).reshape(1, 4256, 1, 64, 2, 2)
    scales = [torch.rand(128, dtype=torch.bfloat16, generator=gen).to("xpu") + 0.5 for _ in range(2)]
    assert module._zimage_contract(q, k, freqs, *scales)
    assert not module._zimage_contract(q.contiguous(), k, freqs, *scales)
    assert not module._zimage_contract(q, k, freqs, scales[0], None)
    with torch.inference_mode():
        actual = patched(q, k, freqs, *scales, epsilon=1e-5)
        expected = original(q, k, freqs, *scales, epsilon=1e-5)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0.02, atol=0.02)
        assert module.get_stats()["zimage_routed"] == 1
        before = module.get_stats()["fallback"]
        patched(q[:, :64], k[:, :64], freqs[:, :64], *scales)
        assert module.get_stats()["fallback"] == before + 1
    torch.xpu.synchronize()


if __name__ == "__main__":
    _run_standalone()
