"""Compatibility patches for Qwen3-VL / Qwen-Image 2.1 text encoder on Intel XPU.

PyTorch 2.15+xpu has driver assertion/access violation bugs in two indexing paths:
1. Qwen3VLVisionModel.rot_pos_emb: freq_table[pos_ids] triggers UR_RESULT_ERROR_DEVICE_LOST (20)
   when pos_ids is a 2D tensor on XPU. Replaced with torch.nn.functional.embedding(pos_ids, freq_table).
2. Llama2_.forward DeepStack injection: x[visual_pos_masks] = x[visual_pos_masks] + ds
   triggers AssertHandler::printMessage (process crash) on large 3D tensors with boolean masks on XPU.
   Replaced with contiguous span slicing along seq dim using embeds_info metadata.
"""

import inspect
import logging
import textwrap
import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-OmniXPU")

_VISION_MARKER = "_omnixpu_qwen3vl_rot_pos_emb_patched"
_LLAMA_MARKER = "_omnixpu_llama_deepstack_patched"


def _patch_rot_pos_emb():
    patched_any = False
    targets = []
    try:
        import comfy.text_encoders.qwen3vl as qwen3vl
        if hasattr(qwen3vl, "Qwen3VLVisionModel"):
            targets.append(qwen3vl.Qwen3VLVisionModel)
    except Exception:
        pass

    try:
        import comfy.text_encoders.qwen35 as qwen35
        if hasattr(qwen35, "Qwen35VisionTransformer"):
            targets.append(qwen35.Qwen35VisionTransformer)
    except Exception:
        pass

    for cls in targets:
        if getattr(cls, _VISION_MARKER, False):
            continue

        def safe_rot_pos_emb(self, grid_thw):
            merge_size = self.spatial_merge_size
            grid_thw_list = grid_thw.tolist()
            max_hw = max(max(h, w) for _, h, w in grid_thw_list)
            freq_table = self.rotary_pos_emb(max_hw)
            device = freq_table.device
            total_tokens = sum(int(t * h * w) for t, h, w in grid_thw_list)
            pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
            offset = 0
            for num_frames, height, width in grid_thw_list:
                num_frames, height, width = int(num_frames), int(height), int(width)
                merged_h, merged_w = height // merge_size, width // merge_size
                block_rows = torch.arange(merged_h, device=device)
                block_cols = torch.arange(merged_w, device=device)
                intra_row = torch.arange(merge_size, device=device)
                intra_col = torch.arange(merge_size, device=device)
                row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
                col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
                row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
                col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
                coords = torch.stack((row_idx, col_idx), dim=-1)
                if num_frames > 1:
                    coords = coords.repeat(num_frames, 1)
                num_tokens = coords.shape[0]
                pos_ids[offset:offset + num_tokens] = coords
                offset += num_tokens
            embeddings = F.embedding(pos_ids, freq_table)
            embeddings = embeddings.flatten(1)
            return embeddings

        cls.rot_pos_emb = safe_rot_pos_emb
        setattr(cls, _VISION_MARKER, True)
        patched_any = True

    return patched_any


def _patch_llama_deepstack():
    try:
        import comfy.text_encoders.llama as llama
        if not hasattr(llama, "Llama2_"):
            return False
        if getattr(llama.Llama2_, _LLAMA_MARKER, False):
            return True

        orig_forward = llama.Llama2_.forward
        src = textwrap.dedent(inspect.getsource(orig_forward))
        needle = "x[visual_pos_masks] = x[visual_pos_masks] + deepstack_embeds[i].to(x)"
        if needle not in src:
            return False

        replacement = """if x.device.type == "xpu" and embeds_info:
                images = sorted([e for e in embeds_info if e.get("type") == "image"], key=lambda e: e["index"])
                ds_layer = deepstack_embeds[i].to(x)
                ds_offset = 0
                for e in images:
                    st = e["index"]
                    sz = e["size"]
                    x[:, st:st + sz] += ds_layer[ds_offset:ds_offset + sz]
                    ds_offset += sz
            else:
                x[visual_pos_masks] = x[visual_pos_masks] + deepstack_embeds[i].to(x)"""

        src_mod = src.replace(needle, replacement)
        g = llama.__dict__.copy()
        l = {}
        exec(src_mod, g, l)
        llama.Llama2_.forward = l["forward"]
        setattr(llama.Llama2_, _LLAMA_MARKER, True)
        return True
    except Exception as exc:
        log.warning("[OmniXPU] Failed to patch Llama2_ DeepStack: %s", exc)
        return False


def apply():
    success_rot = _patch_rot_pos_emb()
    success_llama = _patch_llama_deepstack()
    if success_rot or success_llama:
        log.info("[OmniXPU] Qwen3-VL XPU index compatibility applied (embedding + span slicing)")
        return True, None
    log.warning("[OmniXPU] Qwen3-VL XPU compatibility skipped: targets not found")
    return False, "targets not found"
