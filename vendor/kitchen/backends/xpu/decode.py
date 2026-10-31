"""Kitchen decode adapters for Omni's native SYCL core."""

import torch

from comfy_kitchen.registry import registry


@torch.compiler.assume_constant_result
def _device_count():
    # XPU topology is fixed after runtime initialization. Keep only this
    # non-Tensor hardware query constant; backend overrides remain observable.
    return torch.xpu.device_count()


def _device_available(device):
    if device is None or isinstance(device, int):
        return False
    device = torch.device(device)
    if device.type != "xpu":
        return False
    if not registry.is_available("xpu") or not torch.xpu.is_available():
        return False
    if registry.get_backend_override() not in (None, "xpu"):
        return False
    if device.index is not None and not 0 <= device.index < _device_count():
        return False
    return True


def is_available(device, dtype=torch.bfloat16):
    if dtype != torch.bfloat16 or not _device_available(device):
        return False
    try:
        from omni_xpu_kernel import kitchen
        return kitchen.supports_decode_attention()
    except (ImportError, AttributeError):
        return False


def int8_is_available(device):
    if not _device_available(device):
        return False
    try:
        from omni_xpu_kernel import kitchen
        return kitchen.supports_int8_decode()
    except (ImportError, AttributeError):
        return False


def gqa(q, k, v, lengths, return_lse=False, causal=True):
    from omni_xpu_kernel import kitchen

    _validate(q, k, v, lengths)
    output, lse = kitchen.decode_attention(q, k, v, lengths, causal)
    return (output, lse) if return_lse else output


def step_merge(out, lse, q, k, v, merged):
    from omni_xpu_kernel import kitchen
    _validate(q, k, v)
    kitchen.decode_step_merge(out, lse, q, k, v, merged)
    return merged


def tree_merge(out, lse, q, k, v, mask, merged):
    from omni_xpu_kernel import kitchen
    _validate(q, k, v)
    kitchen.decode_tree_merge(out, lse, q, k, v, mask, merged)
    return merged


def _validate(q, k, v, lengths=None):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("decode requires q [B,H,S,D] and matching k/v [B,Hk,L,D]")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("decode batch and head dimensions must match")
    if q.shape[-1] not in (64, 128, 256) or k.shape[1] == 0 or q.shape[1] % k.shape[1] or q.shape[2] == 0 or k.shape[2] == 0:
        raise ValueError("XPU decode requires D=64/128/256, nonempty cache/query and H divisible by Hk")
    operands = (k, v) if lengths is None else (k, v, lengths)
    if any(t.device != q.device for t in operands):
        raise ValueError("decode inputs must share one device")
    if k.dtype != q.dtype or v.dtype != q.dtype or q.dtype != torch.bfloat16:
        raise ValueError("XPU decode requires BF16 q/k/v")
    if lengths is not None and (lengths.dtype not in (torch.int32, torch.int64) or lengths.shape != (q.shape[0],) or not lengths.is_contiguous()):
        raise ValueError("decode lengths must be a contiguous int32/int64 [B] device tensor")
