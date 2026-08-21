"""INT4 GEMM calling wrapper over omni_xpu_kernel svdq ops.

两条路线各自走各自的，互不互通：
- wa4（对称，无 zp）：``onednn_int4_gemm_preconverted(act, packed_u4, scales_f16)``
- tint4/torchao（非对称，per-block zp）：
  ``onednn_int4_gemm_torchao(act, packed_u4, zp_u8, scales_f16)``
  （raw qdata 字节视图未 XOR，w = (q - zp) * scale 在 oneDNN 内完成）

对应算子缺失时不处理（返回 None），模型走自身原有 python/torchao 路径。
"""

import logging

import torch

log = logging.getLogger("ComfyUI-OmniXPU")

_svdq = None


def _get_svdq():
    global _svdq
    if _svdq is None:
        from omni_xpu_kernel import svdq
        _svdq = svdq
    return _svdq


def int4_gemm(act, packed_u4, scales_f16, zp_u8=None):
    """INT4 GEMM：无 zp 走 wa4/preconverted，有 zp 走 tint4/torchao。

    Args:
        act: [M, K] bf16/f16/f32 activations.
        packed_u4: [N, K/2] uint8 — wa4 为 ^0x88 的 preconverted 约定；
            tint4 为 raw qdata 字节视图（未 XOR）。
        scales_f16: [G, N] f16 — per-group weight scales.
        zp_u8: [G, N] uint8, optional — tint4 per-block zero points。

    Returns:
        [M, N] same dtype as act；对应算子缺失时返回 None（不处理）。
    """
    svdq = _get_svdq()
    if zp_u8 is not None:
        if not hasattr(svdq, "onednn_int4_gemm_torchao"):
            return None
        return svdq.onednn_int4_gemm_torchao(act, packed_u4, zp_u8, scales_f16)
    if not hasattr(svdq, "onednn_int4_gemm_preconverted"):
        return None
    return svdq.onednn_int4_gemm_preconverted(act, packed_u4, scales_f16)


def apply():
    """报告两条 kernel 路线的可用性；都没有则组件不生效。"""
    try:
        svdq = _get_svdq()
        status = {
            "preconverted": hasattr(svdq, "onednn_int4_gemm_preconverted"),
            "torchao": hasattr(svdq, "onednn_int4_gemm_torchao"),
        }
        if not any(status.values()):
            return False, "kernel missing both INT4 GEMM ops"
        log.info("[OmniXPU] int4_gemm adapter: %s", status)
        return True, ""
    except Exception as exc:
        return False, str(exc)
