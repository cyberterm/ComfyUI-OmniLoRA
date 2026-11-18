"""
Modular LoRA Training Pipeline (`LoRATrainingPipeline`).

Follows ComfyUI's node-graph & `model_management` philosophy:
1. HardwareBackend resolves `xpu` / `cuda` / `rocm` / `npu` / `hpu` / `dml` / `mps` / `cpu`
2. ComfyModelAdapter attaches or constructs the frozen backbone in the resolved dtype (`bf16` or `fp16`)
3. LoRAInjector attaches low-rank A/B matrices to discovered target layers
4. Trainer runs `torch.amp.autocast` (plus `FP32GradScaler` when `fp16` is selected
   on devices without native `bf16`) with pure-FP32 gradient accumulation, FP32 gradient
   clipping (zero `fp64` / `.double()` ops), and zero per-substep GPU-CPU sync stalls.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import math
import time
import traceback
import types
import weakref
from dataclasses import dataclass, field, is_dataclass, replace as dc_replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backends.base import (
    FP32GradScaler,
    HardwareBackend,
    is_custom_quantized_tensor,
    is_inference_tensor,
    safe_to_device,
    sanitize_module_no_fp64,
)
from ..lora.layers import LoRAConfig, LoRAInjector, LoRALinear
from ..models.base import BaseModelAdapter


def _clone_tensors_for_grad(obj: Any) -> Any:
    """
    Recursively clone tensors while preserving autograd connectivity (`CloneBackward0`),
    custom `dict` subclasses (`OmniTransformerOptions`), `namedtuple`s, and `@dataclass`
    containers (such as `ModulationOut(shift, scale, gate)` inside Flux 3 `spans`).
    Ensures in-place operations (`CopySlices` `x[:, :n] = ...`, `addcmul_`, `+=`, etc.) inside a block
    always operate on a non-leaf tensor (`is_leaf == False`) and never mutate the input or output
    tensors held by the caller or checkpoint frame.
    """
    if isinstance(obj, torch.Tensor):
        if is_custom_quantized_tensor(obj):
            return obj
        restore_dev = getattr(obj, "_omni_restore_device", None)
        # CPU-offloaded snapshots are detached, so a 5D one must still be moved back below.
        if not obj.requires_grad and obj.ndim >= 5 and restore_dev is None:
            return obj
        restore_req_grad = bool(getattr(obj, "_omni_requires_grad", False)) or bool(obj.requires_grad)
        if isinstance(restore_dev, torch.device) and obj.device != restore_dev:
            with torch.no_grad():
                restored = safe_to_device(obj.detach(), device=restore_dev)
            if restore_req_grad:
                restored.requires_grad_(True)
            out = restored.clone()
        else:
            out = obj.clone()
        if out.requires_grad and out.is_leaf and torch.is_grad_enabled():
            out = out * 1.0
        return out
    if isinstance(obj, tuple):
        if hasattr(obj, "_fields"):
            return type(obj)(*(_clone_tensors_for_grad(v) for v in obj))
        return tuple(_clone_tensors_for_grad(v) for v in obj)
    if isinstance(obj, list):
        return [_clone_tensors_for_grad(v) for v in obj]
    if isinstance(obj, dict):
        if obj.__class__.__name__ == "OmniTransformerOptions":
            return obj.copy()
        cloned_dict = {
            k: (v.copy() if (k == "transformer_options" and isinstance(v, dict)) else _clone_tensors_for_grad(v))
            for k, v in obj.items()
        }
        if type(obj) is not dict:
            try:
                return type(obj)(cloned_dict)
            except Exception:
                return cloned_dict
        return cloned_dict
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            updates = {
                f_name: _clone_tensors_for_grad(getattr(obj, f_name))
                for f_name in obj.__dataclass_fields__
            }
            return dc_replace(obj, **updates)
        except Exception:
            return obj
    return obj


def _snapshot_tensors_for_recompute(
    obj: Any,
    memo: Optional[Dict[int, Tuple[Any, int, int, torch.Tensor]]] = None,
    offload_to_cpu: bool = False,
) -> Any:
    """
    Create detached, isolated tensor snapshots (`_version == 0`) with `requires_grad`
    recorded so activation checkpoint recomputation can clone them under `torch.enable_grad()`
    into fresh non-leaf tensors with the exact pre-mutation input values:
    - Deduplicates unchanged shared conditioning tensors (`mod`, `pe`, `ref`, `mask`, `freqs_cis`)
      across all blocks in a step via weakref-verified `memo` (`id(obj) -> (weakref(obj), data_ptr, _version, snap)`),
      preventing 36-60x duplicate GPU clones of shared conditioning tensors while guaranteeing freed
      temporary tensors can never collide.
    - When `offload_to_cpu=True`, offloads >= 64 KB GPU input snapshots to CPU RAM (`_omni_restore_device`)
      so 40-60 checkpointed blocks consume zero GPU VRAM for `saved_inputs`.
    """
    if isinstance(obj, torch.Tensor):
        if is_custom_quantized_tensor(obj):
            return obj
        if not obj.requires_grad and obj.ndim >= 5:
            return obj
        obj_id = id(obj)
        ptr = 0
        ver = 0
        wref: Any = None
        if memo is not None and getattr(obj, "_base", None) is None:
            try:
                ver = int(obj._version) if not is_inference_tensor(obj) else 0
                ptr = int(obj.data_ptr()) if obj.numel() > 0 else 0
                entry = memo.get(obj_id)
                if entry is not None and entry[0]() is obj and entry[1] == ptr and entry[2] == ver:
                    cached_snap: torch.Tensor = entry[3]
                    # If this tensor is shared across >= 2 blocks (memo hit!) and was offloaded to CPU
                    # on its first block, promote the single shared snapshot back to device once so
                    # backward recomputation across 40+ blocks never copies it from CPU 40+ times.
                    if getattr(cached_snap, "_omni_restore_device", None) is not None and obj.device.type != "cpu":
                        with torch.no_grad():
                            promoted = obj.detach().clone()
                            if obj.requires_grad:
                                promoted.requires_grad_(True)
                        memo[obj_id] = (entry[0], ptr, ver, promoted)
                        return promoted
                    return cached_snap
                wref = weakref.ref(obj)
            except Exception:
                wref = None
        with torch.no_grad():
            if offload_to_cpu and obj.device.type != "cpu" and (obj.numel() * obj.element_size() >= 65536):
                snap = obj.detach().to(device="cpu", copy=True)
                setattr(snap, "_omni_restore_device", obj.device)
                setattr(snap, "_omni_requires_grad", bool(obj.requires_grad))
            else:
                snap = obj.detach().clone()
                if obj.requires_grad:
                    snap.requires_grad_(True)
        if memo is not None and wref is not None:
            memo[obj_id] = (wref, ptr, ver, snap)
        return snap
    if isinstance(obj, tuple):
        if hasattr(obj, "_fields"):
            return type(obj)(*(
                _snapshot_tensors_for_recompute(v, memo=memo, offload_to_cpu=offload_to_cpu) for v in obj
            ))
        return tuple(_snapshot_tensors_for_recompute(v, memo=memo, offload_to_cpu=offload_to_cpu) for v in obj)
    if isinstance(obj, list):
        return [_snapshot_tensors_for_recompute(v, memo=memo, offload_to_cpu=offload_to_cpu) for v in obj]
    if isinstance(obj, dict):
        if obj.__class__.__name__ == "OmniTransformerOptions":
            return obj.copy()
        snap_dict = {
            k: (
                v.copy()
                if (k == "transformer_options" and isinstance(v, dict))
                else _snapshot_tensors_for_recompute(v, memo=memo, offload_to_cpu=offload_to_cpu)
            )
            for k, v in obj.items()
        }
        if type(obj) is not dict:
            try:
                return type(obj)(snap_dict)
            except Exception:
                return snap_dict
        return snap_dict
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            updates = {
                f_name: _snapshot_tensors_for_recompute(
                    getattr(obj, f_name), memo=memo, offload_to_cpu=offload_to_cpu
                )
                for f_name in obj.__dataclass_fields__
            }
            return dc_replace(obj, **updates)
        except Exception:
            return obj
    return obj


def _clear_checkpoint_recompute_state(backbone: Optional[nn.Module]) -> None:
    """Release all pending `recomputed` activation caches and `saved_inputs` snapshots on `backbone`."""
    if backbone is None:
        return
    ckpt_cache = getattr(backbone, "_omni_ckpt_recompute_cache", None)
    if isinstance(ckpt_cache, list) and ckpt_cache and ckpt_cache[0] is not None:
        ckpt_cache[0].clear()
        ckpt_cache[0] = None
    ckpt_state = getattr(backbone, "_omni_ckpt_state", None)
    if isinstance(ckpt_state, dict):
        saved_list = ckpt_state.get("saved_inputs_list")
        if isinstance(saved_list, list):
            for item in saved_list:
                if isinstance(item, dict):
                    item.clear()
            saved_list.clear()
        memo = ckpt_state.get("snapshot_memo")
        if isinstance(memo, dict):
            memo.clear()


def _find_first_tensor_device(obj: Any) -> Optional[torch.device]:
    """Return the device of the first accelerator `torch.Tensor` found inside `obj` (falling back to CPU)."""
    fallback_cpu: Optional[torch.device] = None
    if isinstance(obj, torch.Tensor):
        return obj.device
    if isinstance(obj, (tuple, list)):
        for item in obj:
            dev = _find_first_tensor_device(item)
            if dev is not None:
                if dev.type != "cpu":
                    return dev
                if fallback_cpu is None:
                    fallback_cpu = dev
    elif isinstance(obj, dict):
        for item in obj.values():
            dev = _find_first_tensor_device(item)
            if dev is not None:
                if dev.type != "cpu":
                    return dev
                if fallback_cpu is None:
                    fallback_cpu = dev
    return fallback_cpu


def _any_tensor_requires_grad(obj: Any) -> bool:
    """Return True if any `torch.Tensor` inside `obj` has `requires_grad=True`."""
    if isinstance(obj, torch.Tensor):
        return bool(obj.requires_grad)
    if isinstance(obj, (tuple, list)):
        return any(_any_tensor_requires_grad(v) for v in obj)
    if isinstance(obj, dict):
        return any(_any_tensor_requires_grad(v) for v in obj.values())
    return False


_OUT_KWARG_TORCH_OPS: Tuple[str, ...] = (
    "cat",
    "stack",
    "hstack",
    "vstack",
    "dstack",
    "column_stack",
    "row_stack",
    "add",
    "sub",
    "mul",
    "div",
    "bmm",
    "matmul",
    "mm",
    "addmm",
    "baddbmm",
    "lerp",
    "addcmul",
    "addcdiv",
    "sin",
    "cos",
    "exp",
    "sqrt",
    "rsqrt",
    "abs",
    "neg",
)


def _copy_diff_result_into_out(out_target: Any, res: Any) -> Any:
    """
    Copy a differentiable result `res` into a pre-allocated `out=` target tensor in-place
    while preserving full autograd graph connectivity (`<CopyBackwards>` / `<CopySlices>` -> `res.grad_fn`).
    """
    if isinstance(out_target, torch.Tensor) and isinstance(res, torch.Tensor):
        base = getattr(out_target, "_base", None)
        if base is not None and base.is_leaf and base.requires_grad:
            base.requires_grad_(False)
        elif out_target.is_leaf and out_target.requires_grad:
            out_target.requires_grad_(False)
        if base is None and out_target.shape != res.shape:
            out_target.resize_(res.shape)
        out_target.copy_(res)
        return out_target

    if isinstance(out_target, (tuple, list)) and isinstance(res, (tuple, list)):
        for ot, rt in zip(out_target, res):
            _copy_diff_result_into_out(ot, rt)
        return out_target

    return res


def _slice_sdpa_attn_mask(
    attn_mask: Optional[torch.Tensor],
    q_len: int,
    start: int,
    end: int,
) -> Optional[torch.Tensor]:
    """Slice `attn_mask` along its query sequence dimension (`-2`) when its length matches `q_len`."""
    if attn_mask is None or not isinstance(attn_mask, torch.Tensor):
        return attn_mask
    if attn_mask.ndim >= 2 and attn_mask.shape[-2] == q_len:
        return attn_mask[..., start:end, :]
    return attn_mask


class _FlashChunkedSDPAFn(torch.autograd.Function):
    """
    Memory-bounded, exact Scaled Dot-Product Attention autograd function for large token sequences
    or non-multiple-of-32 head dimensions (such as SD 1.5 `UNetModel` outer `128x128` attention with
    `L = 16,384..16,640` and `head_dim = 40`, which otherwise falls back to `SDPBackend.MATH` and
    attempts a single `4.13 GiB` contiguous allocation that exceeds Intel Arc Level Zero's 4.00 GiB
    `maxMemAllocSize` limit).
    - Forward: computes query chunks under `torch.no_grad()` and saves only `(q, k, v, attn_mask)`
      (~32 MB instead of 4,430 MB).
    - Backward: recomputes one query chunk (`<= 64 MB`) at a time and accumulates `grad_k`, `grad_v`
      in FP32 before casting back to `k.dtype`, `v.dtype`.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        scale: Optional[float],
        enable_gqa: bool,
        chunk_q: int,
        orig_sdpa: Any,
    ) -> torch.Tensor:
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa
        ctx.chunk_q = int(chunk_q)
        ctx.orig_sdpa = orig_sdpa
        ctx.has_mask = attn_mask is not None
        if attn_mask is not None:
            ctx.save_for_backward(q, k, v, attn_mask)
        else:
            ctx.save_for_backward(q, k, v)

        q_len = q.shape[-2]
        sdpa_kwargs: Dict[str, Any] = {}
        if scale is not None:
            sdpa_kwargs["scale"] = scale
        if enable_gqa:
            sdpa_kwargs["enable_gqa"] = True

        out_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, q_len, chunk_q):
                end = min(q_len, start + chunk_q)
                q_c = q[:, :, start:end, :]
                m_c = _slice_sdpa_attn_mask(attn_mask, q_len, start, end)
                out_c = orig_sdpa(
                    q_c,
                    k,
                    v,
                    attn_mask=m_c,
                    dropout_p=0.0,
                    is_causal=False,
                    **sdpa_kwargs,
                )
                out_chunks.append(out_c)
        return torch.cat(out_chunks, dim=-2)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        saved = ctx.saved_tensors
        if ctx.has_mask:
            q, k, v, attn_mask = saved
        else:
            q, k, v = saved
            attn_mask = None

        scale = ctx.scale
        enable_gqa = ctx.enable_gqa
        chunk_q = ctx.chunk_q
        orig_sdpa = ctx.orig_sdpa

        need_q = bool(ctx.needs_input_grad[0])
        need_k = bool(ctx.needs_input_grad[1])
        need_v = bool(ctx.needs_input_grad[2])

        if not (need_q or need_k or need_v):
            return None, None, None, None, None, None, None, None

        q_len = q.shape[-2]
        grad_q_chunks: List[torch.Tensor] = []
        grad_k_fp32 = torch.zeros(k.shape, dtype=torch.float32, device=k.device) if need_k else None
        grad_v_fp32 = torch.zeros(v.shape, dtype=torch.float32, device=v.device) if need_v else None

        sdpa_kwargs: Dict[str, Any] = {}
        if scale is not None:
            sdpa_kwargs["scale"] = scale
        if enable_gqa:
            sdpa_kwargs["enable_gqa"] = True

        for start in range(0, q_len, chunk_q):
            end = min(q_len, start + chunk_q)
            m_c = _slice_sdpa_attn_mask(attn_mask, q_len, start, end)
            go_c = grad_out[:, :, start:end, :].contiguous()

            with torch.enable_grad():
                q_c = q[:, :, start:end, :].detach().clone().requires_grad_(True)
                k_c = k.detach().requires_grad_(True)
                v_c = v.detach().requires_grad_(True)
                out_c = orig_sdpa(
                    q_c,
                    k_c,
                    v_c,
                    attn_mask=m_c,
                    dropout_p=0.0,
                    is_causal=False,
                    **sdpa_kwargs,
                )
                grad_inputs: List[torch.Tensor] = []
                if need_q:
                    grad_inputs.append(q_c)
                if need_k:
                    grad_inputs.append(k_c)
                if need_v:
                    grad_inputs.append(v_c)
                grads = torch.autograd.grad(
                    out_c,
                    grad_inputs,
                    go_c,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

            g_idx = 0
            if need_q:
                gq = grads[g_idx]
                g_idx += 1
                if gq is not None:
                    grad_q_chunks.append(gq.to(dtype=q.dtype))
                else:
                    grad_q_chunks.append(torch.zeros_like(q[:, :, start:end, :]))
            if need_k:
                gk = grads[g_idx]
                g_idx += 1
                if grad_k_fp32 is not None and gk is not None:
                    grad_k_fp32.add_(gk.to(dtype=torch.float32))
            if need_v:
                gv = grads[g_idx]
                g_idx += 1
                if grad_v_fp32 is not None and gv is not None:
                    grad_v_fp32.add_(gv.to(dtype=torch.float32))
            del out_c, grads, q_c, k_c, v_c, go_c

        grad_q = torch.cat(grad_q_chunks, dim=-2) if need_q and grad_q_chunks else None
        grad_k = grad_k_fp32.to(dtype=k.dtype) if grad_k_fp32 is not None else None
        grad_v = grad_v_fp32.to(dtype=v.dtype) if grad_v_fp32 is not None else None
        return grad_q, grad_k, grad_v, None, None, None, None, None


def _run_chunked_sdpa(
    orig_sdpa: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
    chunk_q: int,
) -> torch.Tensor:
    """Execute query-chunked SDPA either via `_FlashChunkedSDPAFn` (autograd) or no_grad loop."""
    if k.dtype != q.dtype:
        k = k.to(dtype=q.dtype)
    if v.dtype != q.dtype:
        v = v.to(dtype=q.dtype)

    mask_needs_grad = isinstance(attn_mask, torch.Tensor) and bool(attn_mask.requires_grad)
    if (
        torch.is_grad_enabled()
        and (q.requires_grad or k.requires_grad or v.requires_grad)
        and not mask_needs_grad
    ):
        return _FlashChunkedSDPAFn.apply(
            q,
            k,
            v,
            attn_mask,
            scale,
            enable_gqa,
            chunk_q,
            orig_sdpa,
        )

    q_len = q.shape[-2]
    sdpa_kwargs: Dict[str, Any] = {}
    if scale is not None:
        sdpa_kwargs["scale"] = scale
    if enable_gqa:
        sdpa_kwargs["enable_gqa"] = True

    out_chunks: List[torch.Tensor] = []
    for start in range(0, q_len, chunk_q):
        end = min(q_len, start + chunk_q)
        q_c = q[:, :, start:end, :]
        m_c = _slice_sdpa_attn_mask(attn_mask, q_len, start, end)
        out_chunks.append(
            orig_sdpa(
                q_c,
                k,
                v,
                attn_mask=m_c,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **sdpa_kwargs,
            )
        )
    return torch.cat(out_chunks, dim=-2)


def _install_chunked_sdpa_wrapper() -> None:
    """
    Wrap `torch.nn.functional.scaled_dot_product_attention` so that:
    1. Non-Flash head dimensions (`head_dim % 32 != 0`, e.g. SD 1.5 `head_dim = 40` or `80`) or
       high-resolution outer UNet blocks (`L_q >= 8192`) automatically chunk along the query sequence
       dimension (`<= 64 MB` per chunk) instead of materializing a multi-gigabyte `[B, H, L_q, L_k]`
       matrix in `SDPBackend.MATH` (which hits Intel Arc Level Zero's 4.00 GiB single-allocation limit).
    2. Any unexpected `OutOfMemoryError` inside native SDPA automatically falls back to query-chunked
       SDPA without failing the training step.
    """
    import torch.nn.functional as F

    cur_fn = getattr(F, "scaled_dot_product_attention", None)
    if cur_fn is None or getattr(cur_fn, "_omni_chunked_sdpa_safe", False):
        return

    orig_sdpa = getattr(F, "_omni_orig_sdpa", cur_fn)
    setattr(F, "_omni_orig_sdpa", orig_sdpa)

    target_chunk_bytes = 64 * 1024 * 1024  # 64 MB per query chunk

    def _pure_fp32_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        scale_f = float(scale) if scale is not None else (1.0 / math.sqrt(max(1.0, float(q.shape[-1]))))
        dml_ctx = (
            torch.inference_mode(False)
            if (q.device.type in ("privateuseone", "dml") and torch.is_inference_mode_enabled())
            else contextlib.nullcontext()
        )
        with dml_ctx:
            if q.device.type in ("privateuseone", "dml"):
                if is_inference_tensor(q) or not q.is_contiguous() or q.storage_offset() != 0:
                    q = q.clone()
                if is_inference_tensor(k) or not k.is_contiguous() or k.storage_offset() != 0:
                    k = k.clone()
                if is_inference_tensor(v) or not v.is_contiguous() or v.storage_offset() != 0:
                    v = v.clone()
            if not (q.dtype == k.dtype == v.dtype):
                target_dt = (
                    v.dtype
                    if v.dtype in (torch.float16, torch.bfloat16)
                    else (k.dtype if k.dtype in (torch.float16, torch.bfloat16) else q.dtype)
                )
                if q.dtype != target_dt:
                    q = q.to(dtype=target_dt)
                if k.dtype != target_dt:
                    k = k.to(dtype=target_dt)
                if v.dtype != target_dt:
                    v = v.to(dtype=target_dt)
            if enable_gqa and k.ndim >= 4 and q.ndim >= 4 and k.shape[-3] != q.shape[-3]:
                rep = max(1, int(q.shape[-3]) // max(1, int(k.shape[-3])))
                k = k.repeat_interleave(rep, dim=-3)
                v = v.repeat_interleave(rep, dim=-3)
            kt = k.transpose(-2, -1)
            if q.device.type in ("privateuseone", "dml"):
                kt = kt.contiguous()
            scores = torch.matmul(q, kt) * scale_f
            if is_causal:
                q_len, k_len = int(q.shape[-2]), int(k.shape[-2])
                causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device="cpu").tril()
                scores = scores.masked_fill(
                    (~causal_mask).to(device=scores.device),
                    -65504.0 if scores.dtype == torch.float16 else -1e9,
                )
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool or not attn_mask.is_floating_point():
                    mask_bool = attn_mask if attn_mask.dtype == torch.bool else attn_mask.to(dtype=torch.bool)
                    scores = scores.masked_fill(
                        ~mask_bool.to(device=scores.device),
                        -65504.0 if scores.dtype == torch.float16 else -1e9,
                    )
                else:
                    scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)
            probs = torch.softmax(scores.to(dtype=torch.float32), dim=-1).to(dtype=v.dtype)
            return torch.matmul(probs, v)

    def _wrapped_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if isinstance(attn_mask, torch.Tensor) and attn_mask.dtype != torch.bool and not attn_mask.is_floating_point():
            attn_mask = attn_mask.to(dtype=torch.bool)
        q_dev_type = getattr(query.device, "type", "cpu") if isinstance(query, torch.Tensor) else "cpu"
        active_sdpa = _pure_fp32_sdpa if q_dev_type in ("privateuseone", "dml") else orig_sdpa

        # When RoPE (`freqs_cis` in FP32) promotes 16-bit `query`/`key` to `float32` before SDPA,
        # PyTorch's `F.scaled_dot_product_attention` does not automatically downcast `float32` inputs
        # under `torch.amp.autocast`, forcing slow FP32 math attention (8.1 TFLOPS on T4 vs 65 TFLOPS FP16)
        # and 2x activation memory (or dtype mismatch if `value` remained 16-bit from `_StreamedLinearFn`).
        # Unify `query`, `key`, `value` to the active 16-bit compute dtype across all backends.
        if (
            isinstance(query, torch.Tensor)
            and isinstance(key, torch.Tensor)
            and isinstance(value, torch.Tensor)
            and (
                query.dtype == torch.float32
                or key.dtype == torch.float32
                or value.dtype == torch.float32
                or not (query.dtype == key.dtype == value.dtype)
            )
        ):
            ac_dt: Optional[torch.dtype] = None
            if q_dev_type not in ("privateuseone", "dml"):
                try:
                    if torch.is_autocast_enabled(q_dev_type):
                        cand_dt = torch.get_autocast_dtype(q_dev_type)
                        if cand_dt in (torch.float16, torch.bfloat16):
                            ac_dt = cand_dt
                except TypeError:
                    if q_dev_type != "cpu":
                        try:
                            if torch.is_autocast_enabled():
                                cand_dt = torch.get_autocast_gpu_dtype()
                                if cand_dt in (torch.float16, torch.bfloat16):
                                    ac_dt = cand_dt
                        except Exception:
                            pass
                except Exception:
                    pass
            if ac_dt is None and not (query.dtype == key.dtype == value.dtype):
                for t_cand in (value, key, query):
                    if t_cand.dtype in (torch.float16, torch.bfloat16):
                        ac_dt = t_cand.dtype
                        break
            if ac_dt in (torch.float16, torch.bfloat16):
                if q_dev_type in ("privateuseone", "dml") and ac_dt == torch.bfloat16:
                    ac_dt = torch.float16
                if query.dtype != ac_dt:
                    query = query.to(dtype=ac_dt)
                if key.dtype != ac_dt:
                    key = key.to(dtype=ac_dt)
                if value.dtype != ac_dt:
                    value = value.to(dtype=ac_dt)
                if isinstance(attn_mask, torch.Tensor) and attn_mask.is_floating_point() and attn_mask.dtype != ac_dt:
                    attn_mask = attn_mask.to(dtype=ac_dt)

        if (
            not is_causal
            and dropout_p == 0.0
            and isinstance(query, torch.Tensor)
            and isinstance(key, torch.Tensor)
            and isinstance(value, torch.Tensor)
            and query.ndim == 4
            and key.ndim == 4
            and value.ndim == 4
            and query.shape[2] > 256
        ):
            heads = max(query.shape[1], key.shape[1])
            elem_bytes = max(2, query.element_size())
            bytes_per_q_token = max(1, query.shape[0] * heads * key.shape[2] * elem_bytes)
            full_attn_bytes = query.shape[2] * bytes_per_q_token
            non_flash_head_dim = (
                (query.shape[-1] % 32 != 0)
                or (key.shape[-1] % 32 != 0)
                or (value.shape[-1] % 32 != 0)
            )
            # On CUDA with 16-bit inputs and standard head dims (`% 32 == 0`), native FlashAttention /
            # cutlass Memory-Efficient SDPA uses O(N) tiled SRAM and never allocates `full_attn_bytes`;
            # only chunk proactively on CUDA when running in FP32, with non-Flash head dims, or huge masks.
            is_cuda_fused_16bit = (
                q_dev_type == "cuda"
                and query.dtype in (torch.float16, torch.bfloat16)
                and not non_flash_head_dim
                and attn_mask is None
            )
            needs_proactive_chunking = (
                (non_flash_head_dim and full_attn_bytes > target_chunk_bytes)
                or (q_dev_type in ("xpu", "dml", "privateuseone", "mps", "cpu") and full_attn_bytes > target_chunk_bytes)
                or (not is_cuda_fused_16bit and full_attn_bytes > 256 * 1024 * 1024)
            )
            if needs_proactive_chunking:
                chunk_q = max(128, min(1024, target_chunk_bytes // bytes_per_q_token))
                if query.shape[2] > chunk_q:
                    return _run_chunked_sdpa(
                        active_sdpa,
                        query,
                        key,
                        value,
                        attn_mask,
                        dropout_p,
                        is_causal,
                        scale,
                        enable_gqa,
                        chunk_q,
                    )

        if q_dev_type in ("privateuseone", "dml"):
            return _pure_fp32_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        sdpa_kwargs: Dict[str, Any] = {}
        if scale is not None:
            sdpa_kwargs["scale"] = scale
        if enable_gqa:
            sdpa_kwargs["enable_gqa"] = True

        try:
            return orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **sdpa_kwargs,
            )
        except (RuntimeError, NotImplementedError) as exc:
            err_l = str(exc).lower()
            if (
                not is_causal
                and dropout_p == 0.0
                and isinstance(query, torch.Tensor)
                and query.ndim == 4
                and query.shape[2] > 128
                and (
                    "out of memory" in err_l
                    or "ur_result_error_out_of" in err_l
                    or "bad_alloc" in err_l
                    or "could not allocate" in err_l
                )
            ):
                dev_mod = getattr(torch, query.device.type, None)
                if dev_mod is not None and hasattr(dev_mod, "empty_cache"):
                    try:
                        dev_mod.empty_cache()
                    except Exception:
                        pass
                heads = max(query.shape[1], key.shape[1])
                elem_bytes = max(2, query.element_size())
                bytes_per_q_token = max(1, query.shape[0] * heads * key.shape[2] * elem_bytes)
                chunk_q = max(64, min(512, target_chunk_bytes // bytes_per_q_token))
                if query.shape[2] > chunk_q:
                    return _run_chunked_sdpa(
                        orig_sdpa,
                        query,
                        key,
                        value,
                        attn_mask,
                        dropout_p,
                        is_causal,
                        scale,
                        enable_gqa,
                        chunk_q,
                    )
            if "double" in err_l or "float64" in err_l or "notimplemented" in err_l or "privateuse" in err_l:
                return _pure_fp32_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
            raise

    setattr(_wrapped_sdpa, "_omni_chunked_sdpa_safe", True)
    setattr(_wrapped_sdpa, "_omni_orig_fn", orig_sdpa)
    F.scaled_dot_product_attention = _wrapped_sdpa

    # Install cross-backend & DirectML (`privateuseone` / `dml`) normalization (`layer_norm`, `rms_norm`, `group_norm`),
    # zero-offset 16-bit-preserving `linear`/`conv1d`/`conv2d`/`conv3d`, and 5D-max `pixel_unshuffle`/`pixel_shuffle`
    # guards so:
    # 1. All dtype alignments are pure, deterministic functions of `input.dtype` and `weight.dtype` (guaranteeing
    #    100% identical op graphs during forward and activation-checkpoint backward recomputation).
    # 2. `F.rms_norm` unifies `weight` to `input.dtype` when `weight` is a frozen parameter of different dtype,
    #    preventing PyTorch's `Mismatch dtype between input and weight... Cannot dispatch to fused implementation` fallback.
    # 3. DX12 shaders on DirectML never hang or fail with inference version_counter / 6D descriptor errors.
    def _adapt_norm_last_dim_inline(input_t: torch.Tensor, normalized_shape: Any) -> torch.Tensor:
        if (
            isinstance(input_t, torch.Tensor)
            and input_t.ndim >= 2
            and isinstance(normalized_shape, (tuple, list, torch.Size))
            and len(normalized_shape) == 1
        ):
            expected_dim = int(normalized_shape[0])
            actual_dim = int(input_t.shape[-1])
            if actual_dim != expected_dim and actual_dim > 0 and expected_dim > 0:
                if expected_dim > actual_dim and expected_dim % actual_dim == 0:
                    return input_t.repeat(*([1] * (input_t.ndim - 1)), expected_dim // actual_dim)
                if actual_dim > expected_dim and actual_dim % expected_dim == 0:
                    return input_t[..., -expected_dim:]
        return input_t

    orig_layer_norm = getattr(F, "_omni_orig_layer_norm", getattr(F, "layer_norm", None))
    if orig_layer_norm is not None and not getattr(F.layer_norm, "_omni_dml_safe", False):
        setattr(F, "_omni_orig_layer_norm", orig_layer_norm)

        def _dml_safe_layer_norm(
            input: torch.Tensor,
            normalized_shape: List[int],
            weight: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            eps: float = 1e-5,
        ) -> torch.Tensor:
            if isinstance(input, torch.Tensor):
                input = _adapt_norm_last_dim_inline(input, normalized_shape)
                if input.device.type in ("privateuseone", "dml"):
                    ctx = (
                        torch.inference_mode(False)
                        if torch.is_inference_mode_enabled()
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        if is_inference_tensor(input) or not input.is_contiguous() or input.storage_offset() != 0:
                            input = input.clone()
                        orig_dt = (
                            torch.float16
                            if (isinstance(weight, torch.Tensor) and weight.dtype in (torch.float16, torch.bfloat16))
                            else (torch.float16 if input.dtype == torch.bfloat16 else input.dtype)
                        )
                        ndim_norm = len(normalized_shape)
                        norm_dims = tuple(range(input.ndim - ndim_norm, input.ndim))
                        x_fp32 = input.to(dtype=torch.float32)
                        mean = x_fp32.mean(dim=norm_dims, keepdim=True)
                        diff = x_fp32 - mean
                        var = (diff * diff).mean(dim=norm_dims, keepdim=True)
                        out_fp32 = diff * torch.rsqrt(var + float(eps))
                        if isinstance(weight, torch.Tensor):
                            w_32 = safe_to_device(weight, device=input.device, dtype=torch.float32)
                            out_fp32 = out_fp32 * w_32
                        if isinstance(bias, torch.Tensor):
                            b_32 = safe_to_device(bias, device=input.device, dtype=torch.float32)
                            out_fp32 = out_fp32 + b_32
                        return out_fp32.to(dtype=orig_dt)
                if (
                    isinstance(weight, torch.Tensor)
                    and not weight.requires_grad
                    and (weight.device != input.device or weight.dtype != input.dtype)
                ):
                    weight = safe_to_device(weight, device=input.device, dtype=input.dtype)
                if (
                    isinstance(bias, torch.Tensor)
                    and not bias.requires_grad
                    and (bias.device != input.device or bias.dtype != input.dtype)
                ):
                    bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
            return orig_layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

        setattr(_dml_safe_layer_norm, "_omni_dml_safe", True)
        F.layer_norm = _dml_safe_layer_norm

    orig_rms_norm = getattr(F, "_omni_orig_rms_norm", getattr(F, "rms_norm", None))
    if orig_rms_norm is not None and not getattr(F.rms_norm, "_omni_dml_safe", False):
        setattr(F, "_omni_orig_rms_norm", orig_rms_norm)

        def _dml_safe_rms_norm(
            input: torch.Tensor,
            normalized_shape: List[int],
            weight: Optional[torch.Tensor] = None,
            eps: Optional[float] = None,
        ) -> torch.Tensor:
            if isinstance(input, torch.Tensor):
                input = _adapt_norm_last_dim_inline(input, normalized_shape)
                if input.device.type in ("privateuseone", "dml"):
                    ctx = (
                        torch.inference_mode(False)
                        if torch.is_inference_mode_enabled()
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        if is_inference_tensor(input) or not input.is_contiguous() or input.storage_offset() != 0:
                            input = input.clone()
                        orig_dt = torch.float16 if input.dtype == torch.bfloat16 else input.dtype
                        eps_val = float(eps) if eps is not None else 1e-6
                        ndim_norm = len(normalized_shape)
                        norm_dims = tuple(range(input.ndim - ndim_norm, input.ndim))
                        x_fp32 = input.to(dtype=torch.float32)
                        ms = (x_fp32 * x_fp32).mean(dim=norm_dims, keepdim=True)
                        out_fp32 = x_fp32 * torch.rsqrt(ms + eps_val)
                        if isinstance(weight, torch.Tensor):
                            w_32 = safe_to_device(weight, device=input.device, dtype=torch.float32)
                            out_fp32 = out_fp32 * w_32
                        return out_fp32.to(dtype=orig_dt)
                if (
                    isinstance(weight, torch.Tensor)
                    and not weight.requires_grad
                    and (weight.device != input.device or weight.dtype != input.dtype)
                ):
                    weight = safe_to_device(weight, device=input.device, dtype=input.dtype)
            return orig_rms_norm(input, normalized_shape, weight=weight, eps=eps)

        setattr(_dml_safe_rms_norm, "_omni_dml_safe", True)
        F.rms_norm = _dml_safe_rms_norm

    orig_group_norm = getattr(F, "_omni_orig_group_norm", getattr(F, "group_norm", None))
    if orig_group_norm is not None and not getattr(F.group_norm, "_omni_dml_safe", False):
        setattr(F, "_omni_orig_group_norm", orig_group_norm)

        def _dml_safe_group_norm(
            input: torch.Tensor,
            num_groups: int,
            weight: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            eps: float = 1e-5,
        ) -> torch.Tensor:
            if isinstance(input, torch.Tensor):
                if input.device.type in ("privateuseone", "dml"):
                    ctx = (
                        torch.inference_mode(False)
                        if torch.is_inference_mode_enabled()
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        if is_inference_tensor(input) or not input.is_contiguous() or input.storage_offset() != 0:
                            input = input.clone()
                        orig_dt = (
                            torch.float16
                            if (isinstance(weight, torch.Tensor) and weight.dtype in (torch.float16, torch.bfloat16))
                            else (torch.float16 if input.dtype == torch.bfloat16 else input.dtype)
                        )
                        shape = input.shape
                        b, c = int(shape[0]), int(shape[1])
                        g = int(num_groups)
                        x_fp32 = input.to(dtype=torch.float32).reshape(b, g, -1)
                        mean = x_fp32.mean(dim=-1, keepdim=True)
                        diff = x_fp32 - mean
                        var = (diff * diff).mean(dim=-1, keepdim=True)
                        out_fp32 = (diff * torch.rsqrt(var + float(eps))).reshape(shape)
                        bcast = [1, c] + [1] * (input.ndim - 2)
                        if isinstance(weight, torch.Tensor):
                            w_32 = safe_to_device(weight, device=input.device, dtype=torch.float32).reshape(bcast)
                            out_fp32 = out_fp32 * w_32
                        if isinstance(bias, torch.Tensor):
                            b_32 = safe_to_device(bias, device=input.device, dtype=torch.float32).reshape(bcast)
                            out_fp32 = out_fp32 + b_32
                        return out_fp32.to(dtype=orig_dt)
                if (
                    isinstance(weight, torch.Tensor)
                    and not weight.requires_grad
                    and (weight.device != input.device or weight.dtype != input.dtype)
                ):
                    weight = safe_to_device(weight, device=input.device, dtype=input.dtype)
                if (
                    isinstance(bias, torch.Tensor)
                    and not bias.requires_grad
                    and (bias.device != input.device or bias.dtype != input.dtype)
                ):
                    bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
            return orig_group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)

        setattr(_dml_safe_group_norm, "_omni_dml_safe", True)
        F.group_norm = _dml_safe_group_norm

    orig_linear = getattr(F, "_omni_orig_linear", getattr(F, "linear", None))
    if orig_linear is not None and not getattr(F.linear, "_omni_dml_safe", False):
        setattr(F, "_omni_orig_linear", orig_linear)

        def _dml_safe_linear(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if isinstance(input, torch.Tensor):
                if input.device.type in ("privateuseone", "dml"):
                    ctx = (
                        torch.inference_mode(False)
                        if torch.is_inference_mode_enabled()
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        if (
                            isinstance(weight, torch.Tensor)
                            and weight.dtype in (torch.float16, torch.bfloat16)
                            and input.dtype == torch.float32
                        ):
                            target_dt = torch.float16 if input.device.type in ("privateuseone", "dml") else weight.dtype
                            input = input.to(dtype=target_dt)
                        if is_inference_tensor(input) or not input.is_contiguous() or input.storage_offset() != 0:
                            input = input.clone()
                        if isinstance(weight, torch.Tensor):
                            if weight.device != input.device or weight.dtype != input.dtype:
                                weight = safe_to_device(weight, device=input.device, dtype=input.dtype)
                            if is_inference_tensor(weight) or not weight.is_contiguous() or weight.storage_offset() != 0:
                                weight = weight.clone()
                        if isinstance(bias, torch.Tensor):
                            if bias.device != input.device or bias.dtype != input.dtype:
                                bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
                            if is_inference_tensor(bias) or not bias.is_contiguous() or bias.storage_offset() != 0:
                                bias = bias.clone()
                        return orig_linear(input, weight, bias)
                if (
                    isinstance(weight, torch.Tensor)
                    and not weight.requires_grad
                    and (weight.device != input.device or weight.dtype != input.dtype)
                ):
                    target_dt = input.dtype
                    if input.dtype == torch.float32 and weight.dtype in (torch.float16, torch.bfloat16):
                        try:
                            dev_t = input.device.type
                            if torch.is_autocast_enabled(dev_t):  # type: ignore[arg-type]
                                ac_dt = torch.get_autocast_dtype(dev_t)  # type: ignore[arg-type]
                                if ac_dt in (torch.float16, torch.bfloat16):
                                    target_dt = ac_dt
                                else:
                                    target_dt = weight.dtype
                            else:
                                target_dt = weight.dtype
                        except Exception:
                            target_dt = weight.dtype
                        input = input.to(dtype=target_dt)
                    weight = safe_to_device(weight, device=input.device, dtype=target_dt)
                    if isinstance(bias, torch.Tensor) and (bias.device != input.device or bias.dtype != target_dt):
                        bias = safe_to_device(bias, device=input.device, dtype=target_dt)
                elif isinstance(bias, torch.Tensor) and (bias.device != input.device or bias.dtype != input.dtype):
                    bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
            return orig_linear(input, weight, bias)

        setattr(_dml_safe_linear, "_omni_dml_safe", True)
        F.linear = _dml_safe_linear

    orig_conv1d = getattr(F, "_omni_orig_conv1d", getattr(F, "conv1d", None))
    if orig_conv1d is not None and not getattr(F.conv1d, "_omni_dml_safe", False):
        setattr(F, "_omni_orig_conv1d", orig_conv1d)

        def _dml_safe_conv1d(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            stride: Any = 1,
            padding: Any = 0,
            dilation: Any = 1,
            groups: int = 1,
        ) -> torch.Tensor:
            if isinstance(input, torch.Tensor) and isinstance(weight, torch.Tensor):
                if (
                    not weight.requires_grad
                    and (weight.device != input.device or weight.dtype != input.dtype)
                ):
                    if input.device.type in ("privateuseone", "dml"):
                        target_dt = torch.float16 if input.dtype in (torch.float16, torch.bfloat16, torch.float32) else input.dtype
                        if input.dtype != target_dt:
                            input = input.to(dtype=target_dt)
                    else:
                        target_dt = input.dtype
                        if input.dtype == torch.float32 and weight.dtype in (torch.float16, torch.bfloat16):
                            try:
                                dev_t = input.device.type
                                if torch.is_autocast_enabled(dev_t):  # type: ignore[arg-type]
                                    ac_dt = torch.get_autocast_dtype(dev_t)  # type: ignore[arg-type]
                                    target_dt = ac_dt if ac_dt in (torch.float16, torch.bfloat16) else weight.dtype
                                else:
                                    target_dt = weight.dtype
                            except Exception:
                                target_dt = weight.dtype
                            input = input.to(dtype=target_dt)
                    weight = safe_to_device(weight, device=input.device, dtype=target_dt)
                    if isinstance(bias, torch.Tensor) and (bias.device != input.device or bias.dtype != target_dt):
                        bias = safe_to_device(bias, device=input.device, dtype=target_dt)
                elif isinstance(bias, torch.Tensor) and (bias.device != input.device or bias.dtype != input.dtype):
                    bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
            return orig_conv1d(input, weight, bias, stride, padding, dilation, groups)

        setattr(_dml_safe_conv1d, "_omni_dml_safe", True)
        F.conv1d = _dml_safe_conv1d

    orig_conv2d = getattr(F, "_omni_orig_conv2d", getattr(F, "conv2d", None))
    if orig_conv2d is not None and not getattr(F.conv2d, "_omni_dml_safe", False):
        setattr(F, "_omni_orig_conv2d", orig_conv2d)

        def _dml_safe_conv2d(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            stride: Any = 1,
            padding: Any = 0,
            dilation: Any = 1,
            groups: int = 1,
        ) -> torch.Tensor:
            if isinstance(input, torch.Tensor):
                if input.device.type in ("privateuseone", "dml"):
                    ctx = (
                        torch.inference_mode(False)
                        if torch.is_inference_mode_enabled()
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        if (
                            isinstance(weight, torch.Tensor)
                            and weight.dtype in (torch.float16, torch.bfloat16)
                            and input.dtype == torch.float32
                        ):
                            target_dt = torch.float16 if input.device.type in ("privateuseone", "dml") else weight.dtype
                            input = input.to(dtype=target_dt)
                        if is_inference_tensor(input) or not input.is_contiguous() or input.storage_offset() != 0:
                            input = input.clone()
                        if isinstance(weight, torch.Tensor):
                            if weight.device != input.device or weight.dtype != input.dtype:
                                weight = safe_to_device(weight, device=input.device, dtype=input.dtype)
                            if is_inference_tensor(weight) or not weight.is_contiguous() or weight.storage_offset() != 0:
                                weight = weight.clone()
                        if isinstance(bias, torch.Tensor):
                            if bias.device != input.device or bias.dtype != input.dtype:
                                bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
                            if is_inference_tensor(bias) or not bias.is_contiguous() or bias.storage_offset() != 0:
                                bias = bias.clone()
                        return orig_conv2d(input, weight, bias, stride, padding, dilation, groups)
                if (
                    isinstance(weight, torch.Tensor)
                    and not weight.requires_grad
                    and (weight.device != input.device or weight.dtype != input.dtype)
                ):
                    target_dt = input.dtype
                    if input.dtype == torch.float32 and weight.dtype in (torch.float16, torch.bfloat16):
                        try:
                            dev_t = input.device.type
                            if torch.is_autocast_enabled(dev_t):  # type: ignore[arg-type]
                                ac_dt = torch.get_autocast_dtype(dev_t)  # type: ignore[arg-type]
                                target_dt = ac_dt if ac_dt in (torch.float16, torch.bfloat16) else weight.dtype
                            else:
                                target_dt = weight.dtype
                        except Exception:
                            target_dt = weight.dtype
                        input = input.to(dtype=target_dt)
                    weight = safe_to_device(weight, device=input.device, dtype=target_dt)
                    if isinstance(bias, torch.Tensor) and (bias.device != input.device or bias.dtype != target_dt):
                        bias = safe_to_device(bias, device=input.device, dtype=target_dt)
                elif isinstance(bias, torch.Tensor) and (bias.device != input.device or bias.dtype != input.dtype):
                    bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
                # Ensure 64-byte alignment for Intel XPU / CPU oneDNN conv2d kernels when input is a sliced/permuted view
                if not input.is_contiguous():
                    input = input.contiguous()
                if (input.storage_offset() * input.element_size()) % 64 != 0:
                    input = input.clone()
                if isinstance(weight, torch.Tensor):
                    if not weight.is_contiguous():
                        weight = weight.contiguous()
                    if (weight.storage_offset() * weight.element_size()) % 64 != 0:
                        weight = weight.clone()
            return orig_conv2d(input, weight, bias, stride, padding, dilation, groups)

        setattr(_dml_safe_conv2d, "_omni_dml_safe", True)
        F.conv2d = _dml_safe_conv2d

    orig_conv3d = getattr(F, "_omni_orig_conv3d", getattr(F, "conv3d", None))
    if orig_conv3d is not None and not getattr(F.conv3d, "_omni_dml_safe", False):
        setattr(F, "_omni_orig_conv3d", orig_conv3d)

        def _dml_safe_conv3d(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            stride: Any = 1,
            padding: Any = 0,
            dilation: Any = 1,
            groups: int = 1,
        ) -> torch.Tensor:
            if isinstance(input, torch.Tensor) and isinstance(weight, torch.Tensor):
                if (
                    not weight.requires_grad
                    and (weight.device != input.device or weight.dtype != input.dtype)
                ):
                    if input.device.type in ("privateuseone", "dml"):
                        target_dt = torch.float16 if input.dtype in (torch.float16, torch.bfloat16, torch.float32) else input.dtype
                        if input.dtype != target_dt:
                            input = input.to(dtype=target_dt)
                    else:
                        target_dt = input.dtype
                        if input.dtype == torch.float32 and weight.dtype in (torch.float16, torch.bfloat16):
                            try:
                                dev_t = input.device.type
                                if torch.is_autocast_enabled(dev_t):  # type: ignore[arg-type]
                                    ac_dt = torch.get_autocast_dtype(dev_t)  # type: ignore[arg-type]
                                    target_dt = ac_dt if ac_dt in (torch.float16, torch.bfloat16) else weight.dtype
                                else:
                                    target_dt = weight.dtype
                            except Exception:
                                target_dt = weight.dtype
                            input = input.to(dtype=target_dt)
                    weight = safe_to_device(weight, device=input.device, dtype=target_dt)
                    if isinstance(bias, torch.Tensor) and (bias.device != input.device or bias.dtype != target_dt):
                        bias = safe_to_device(bias, device=input.device, dtype=target_dt)
                elif isinstance(bias, torch.Tensor) and (bias.device != input.device or bias.dtype != input.dtype):
                    bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
            if (
                isinstance(input, torch.Tensor)
                and input.device.type in ("privateuseone", "dml")
                and input.ndim == 5
                and isinstance(weight, torch.Tensor)
                and weight.ndim == 5
                and isinstance(padding, (int, tuple, list))
            ):
                ctx = (
                    torch.inference_mode(False)
                    if torch.is_inference_mode_enabled()
                    else contextlib.nullcontext()
                )
                with ctx:
                    if is_inference_tensor(input) or not input.is_contiguous() or input.storage_offset() != 0:
                        input = input.clone()
                    if weight.device != input.device or weight.dtype != input.dtype:
                        weight = safe_to_device(weight, device=input.device, dtype=input.dtype)
                    if is_inference_tensor(weight) or not weight.is_contiguous() or weight.storage_offset() != 0:
                        weight = weight.clone()
                    if isinstance(bias, torch.Tensor):
                        if bias.device != input.device or bias.dtype != input.dtype:
                            bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
                        if is_inference_tensor(bias) or not bias.is_contiguous() or bias.storage_offset() != 0:
                            bias = bias.clone()
                    s_3 = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
                    p_3 = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
                    d_3 = (dilation, dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
                    s_t, s_h, s_w = int(s_3[0]), int(s_3[1]), int(s_3[2])
                    p_t, p_h, p_w = int(p_3[0]), int(p_3[1]), int(p_3[2])
                    d_t, d_h, d_w = int(d_3[0]), int(d_3[1]), int(d_3[2])
                    k_t = int(weight.shape[2])
                    b, c, t_len, h, w = input.shape
                    if k_t == 1 and s_t == 1 and p_t == 0:
                        x_4d = input.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t_len, c, h, w)
                        w_4d = weight.squeeze(2).clone()
                        out_4d = F.conv2d(
                            x_4d,
                            w_4d,
                            bias,
                            stride=(s_h, s_w),
                            padding=(p_h, p_w),
                            dilation=(d_h, d_w),
                            groups=groups,
                        )
                        _, c_out, h_out, w_out = out_4d.shape
                        return out_4d.reshape(b, t_len, c_out, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()
                    if p_t > 0:
                        input = F.pad(input, (0, 0, 0, 0, p_t, p_t))
                    t_padded = int(input.shape[2])
                    eff_k_t = (k_t - 1) * d_t + 1
                    t_out_len = max(1, (t_padded - eff_k_t) // s_t + 1)
                    frames: List[torch.Tensor] = []
                    for t_idx in range(t_out_len):
                        t_start = t_idx * s_t
                        acc: Optional[torch.Tensor] = None
                        for kt_idx in range(k_t):
                            x_slice = input[:, :, t_start + kt_idx * d_t, :, :].clone()
                            w_slice = weight[:, :, kt_idx, :, :].clone()
                            conv_step = F.conv2d(
                                x_slice,
                                w_slice,
                                bias if kt_idx == 0 else None,
                                stride=(s_h, s_w),
                                padding=(p_h, p_w),
                                dilation=(d_h, d_w),
                                groups=groups,
                            )
                            acc = conv_step if acc is None else (acc + conv_step)
                        assert acc is not None
                        frames.append(acc.unsqueeze(2))
                    return torch.cat(frames, dim=2)
            return orig_conv3d(input, weight, bias, stride, padding, dilation, groups)

        setattr(_dml_safe_conv3d, "_omni_dml_safe", True)
        F.conv3d = _dml_safe_conv3d

    orig_pixel_unshuffle = getattr(F, "_omni_orig_pixel_unshuffle", getattr(F, "pixel_unshuffle", None))
    if orig_pixel_unshuffle is not None and not getattr(F.pixel_unshuffle, "_omni_dml_safe", False):
        setattr(F, "_omni_orig_pixel_unshuffle", orig_pixel_unshuffle)

        def _dml_safe_pixel_unshuffle(input: torch.Tensor, downscale_factor: int) -> torch.Tensor:
            if (
                isinstance(input, torch.Tensor)
                and input.device.type in ("privateuseone", "dml")
                and input.ndim == 4
            ):
                ctx = (
                    torch.inference_mode(False)
                    if torch.is_inference_mode_enabled()
                    else contextlib.nullcontext()
                )
                with ctx:
                    r = int(downscale_factor)
                    b, c, h, w = input.shape
                    if is_inference_tensor(input) or not input.is_contiguous() or input.storage_offset() != 0:
                        input = input.clone()
                    # Merge (B * C) so intermediate tensor is 5D (within DirectML's 5D tensor descriptor limit), never 6D
                    x_5d = input.reshape(b * c, h // r, r, w // r, r).permute(0, 2, 4, 1, 3).contiguous()
                    return x_5d.reshape(b, c * r * r, h // r, w // r)
            return orig_pixel_unshuffle(input, downscale_factor)

        setattr(_dml_safe_pixel_unshuffle, "_omni_dml_safe", True)
        F.pixel_unshuffle = _dml_safe_pixel_unshuffle

    orig_pixel_shuffle = getattr(F, "_omni_orig_pixel_shuffle", getattr(F, "pixel_shuffle", None))
    if orig_pixel_shuffle is not None and not getattr(F.pixel_shuffle, "_omni_dml_safe", False):
        setattr(F, "_omni_orig_pixel_shuffle", orig_pixel_shuffle)

        def _dml_safe_pixel_shuffle(input: torch.Tensor, upscale_factor: int) -> torch.Tensor:
            if (
                isinstance(input, torch.Tensor)
                and input.device.type in ("privateuseone", "dml")
                and input.ndim == 4
            ):
                ctx = (
                    torch.inference_mode(False)
                    if torch.is_inference_mode_enabled()
                    else contextlib.nullcontext()
                )
                with ctx:
                    r = int(upscale_factor)
                    b, c_in, h, w = input.shape
                    c_out = c_in // (r * r)
                    if is_inference_tensor(input) or not input.is_contiguous() or input.storage_offset() != 0:
                        input = input.clone()
                    # Merge (B * C_out) so intermediate tensor is 5D (within DirectML's 5D tensor descriptor limit), never 6D
                    x_5d = input.reshape(b * c_out, r, r, h, w).permute(0, 3, 1, 4, 2).contiguous()
                    return x_5d.reshape(b, c_out, h * r, w * r)
            return orig_pixel_shuffle(input, upscale_factor)

        setattr(_dml_safe_pixel_shuffle, "_omni_dml_safe", True)
        F.pixel_shuffle = _dml_safe_pixel_shuffle


_OUT_KWARG_WRAPPERS_INSTALLED: bool = False
_SEQ_PROMOTE_OPS: Tuple[str, ...] = ("cat", "stack", "hstack", "vstack", "dstack", "column_stack", "row_stack")
_MATMUL_PROMOTE_OPS: Tuple[str, ...] = ("bmm", "matmul", "mm", "addmm", "baddbmm")


def _unify_16bit_seq_tensors(tensors: Any) -> Any:
    """
    When `torch.cat` or `torch.stack` on DirectML (`privateuseone` / `dml`) receives a mix of
    `float32` and `float16` tensors, unify to `float16` so DirectML shaders do not fail on mixed-dtype
    sequence concatenation.
    """
    if not isinstance(tensors, (tuple, list)) or len(tensors) < 2:
        return tensors
    has_fp32 = False
    target_16bit: Optional[torch.dtype] = None
    first_dev_type = "cpu"
    for t in tensors:
        if isinstance(t, torch.Tensor):
            first_dev_type = t.device.type
            if t.dtype == torch.float32:
                has_fp32 = True
            elif t.dtype in (torch.float16, torch.bfloat16) and target_16bit is None:
                target_16bit = t.dtype
    if not has_fp32 or target_16bit is None or first_dev_type not in ("privateuseone", "dml"):
        return tensors
    target_16bit = torch.float16
    converted = [
        (t.to(dtype=target_16bit) if isinstance(t, torch.Tensor) and t.dtype == torch.float32 else t)
        for t in tensors
    ]
    return tuple(converted) if isinstance(tensors, tuple) else converted


def _unify_16bit_matmul_args(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """
    Prevent `RuntimeError: expected scalar type Half/BFloat16 but found Float` when `torch.matmul`,
    `torch.bmm`, `torch.mm`, `torch.addmm`, or `torch.baddbmm` is called with mixed `(float32, 16-bit)`
    inputs (e.g. `probs @ v` after `F.softmax` in `float32` and `v` in `16-bit`).
    """
    if len(args) < 2:
        return args
    has_fp32 = False
    target_16bit: Optional[torch.dtype] = None
    for a in args:
        if isinstance(a, torch.Tensor):
            if a.dtype == torch.float32:
                has_fp32 = True
            elif a.dtype in (torch.float16, torch.bfloat16) and target_16bit is None:
                target_16bit = torch.float16 if a.device.type in ("privateuseone", "dml") else a.dtype
    if not has_fp32 or target_16bit is None:
        return args
    return tuple(
        (a.to(dtype=target_16bit) if isinstance(a, torch.Tensor) and a.dtype == torch.float32 else a)
        for a in args
    )


def _install_out_kwarg_autograd_wrappers() -> None:
    """
    Wrap `torch.cat`, `torch.stack`, `torch.add`, `torch.matmul`, etc. so that:
    1. When an inference-authored ComfyUI model (such as `comfy/ldm/flux3/model.py` calling
       `torch.cat([img] + [...], dim=1, out=x)`) passes `out=x` while `torch.is_grad_enabled()`
       is True and inputs require grad, the operation executes without `out=` to build the
       autograd graph and copies the differentiable result into `x` via `x.copy_(res)`.
    2. Mixed `(float32, 16-bit)` inputs to `matmul`/`bmm`/`mm`/`addmm` are unified to 16-bit.
    3. Installs `_install_chunked_sdpa_wrapper()` for memory-bounded high-res / non-Flash SDPA.
    """
    global _OUT_KWARG_WRAPPERS_INSTALLED
    if (
        _OUT_KWARG_WRAPPERS_INSTALLED
        and getattr(torch.cat, "_omni_out_kwarg_safe", False)
        and getattr(F.scaled_dot_product_attention, "_omni_chunked_sdpa_safe", False)
    ):
        return
    _install_chunked_sdpa_wrapper()
    for op_name in _OUT_KWARG_TORCH_OPS:
        fn = getattr(torch, op_name, None)
        if fn is None or getattr(fn, "_omni_out_kwarg_safe", False):
            continue
        orig_fn = getattr(torch, f"_omni_orig_{op_name}", fn)
        setattr(torch, f"_omni_orig_{op_name}", orig_fn)

        def _make_wrapper(target_fn: Any, name: str) -> Any:
            is_seq_op = name in _SEQ_PROMOTE_OPS
            is_mm_op = name in _MATMUL_PROMOTE_OPS

            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                if is_seq_op and len(args) >= 1:
                    unified_seq = _unify_16bit_seq_tensors(args[0])
                    if unified_seq is not args[0]:
                        args = (unified_seq,) + args[1:]
                elif is_mm_op and len(args) >= 2:
                    args = _unify_16bit_matmul_args(args)
                if not kwargs or "out" not in kwargs or kwargs["out"] is None or not torch.is_grad_enabled():
                    return target_fn(*args, **kwargs)
                if not _any_tensor_requires_grad(args) and not _any_tensor_requires_grad(kwargs):
                    return target_fn(*args, **kwargs)
                out_target = kwargs.pop("out")
                res = target_fn(*args, **kwargs)
                return _copy_diff_result_into_out(out_target, res)

            setattr(_wrapped, "_omni_out_kwarg_safe", True)
            setattr(_wrapped, "_omni_orig_fn", target_fn)
            return _wrapped

        try:
            setattr(torch, op_name, _make_wrapper(orig_fn, op_name))
        except Exception:
            pass
    _OUT_KWARG_WRAPPERS_INSTALLED = True


@contextlib.contextmanager
def _inplace_safe_outer_autograd_context(gradient_checkpointing: bool = True) -> Iterator[None]:
    """
    Outer `saved_tensors_hooks` context active for layers outside checkpointed transformer blocks
    (or for the full model when `gradient_checkpointing=False`).
    1. Installs `_install_out_kwarg_autograd_wrappers()` so `torch.cat(..., out=x)` and related
       `out=` operations support autograd during training.
    2. Snapshots non-parameter activation tensors at the moment they are saved for backward so
       subsequent in-place slice assignments (`CopySlices`, `addcmul_`, `out=x`) cannot overwrite
       saved activation values.
    3. Keeps `hooks_` non-null in PyTorch C++ `SavedVariable::unpack()`, preventing spurious
       `modified by an inplace operation` version-counter exceptions on inference-authored ComfyUI models.
    """
    _install_out_kwarg_autograd_wrappers()

    max_outer_clone_bytes = 8 * 1024 * 1024  # 8 MB cap so large 4D/3D activations are never duplicated

    def _outer_pack(t: torch.Tensor) -> torch.Tensor:
        if (
            not gradient_checkpointing
            or isinstance(t, nn.Parameter)
            or is_custom_quantized_tensor(t)
            or t.device.type in ("privateuseone", "dml")
            or (not t.requires_grad and t.ndim >= 2 and t.numel() >= 16384)
            or (t.numel() * t.element_size() > max_outer_clone_bytes)
        ):
            return t
        with torch.no_grad():
            return t.detach().clone()

    def _outer_unpack(t: torch.Tensor) -> torch.Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(_outer_pack, _outer_unpack):
        yield


def _ensure_nonleaf_grad_outputs(obj: Any) -> Any:
    """
    Ensure any tensor output with `requires_grad=True` that is still a leaf (`is_leaf == True`)
    is cloned into a non-leaf tensor, while leaving already non-leaf tensors untouched so the
    packed autograd graph inside `saved_tensors_hooks` is not wrapped in redundant `CloneBackward0` nodes.
    """
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad and obj.is_leaf and torch.is_grad_enabled():
            out = obj.clone()
            if out.is_leaf:
                out = out * 1.0
            return out
        return obj
    if isinstance(obj, tuple):
        if hasattr(obj, "_fields"):
            return type(obj)(*(_ensure_nonleaf_grad_outputs(v) for v in obj))
        return tuple(_ensure_nonleaf_grad_outputs(v) for v in obj)
    if isinstance(obj, list):
        return [_ensure_nonleaf_grad_outputs(v) for v in obj]
    if isinstance(obj, dict):
        if obj.__class__.__name__ == "OmniTransformerOptions":
            return obj
        return {k: _ensure_nonleaf_grad_outputs(v) for k, v in obj.items()}
    return obj


def _checkpoint_block_inplace_safe(
    orig_fn: Any,
    active_recompute_cache: List[Optional[Dict[int, torch.Tensor]]],
    *args: Any,
    preserve_rng: bool = False,
    active_ckpt_state: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    In-Place-Safe Non-Reentrant Activation Checkpointing (`saved_tensors_hooks`).
    Drop-in replacement for `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` that
    prevents `RuntimeError: one of the variables needed for gradient computation has been modified
    by an inplace operation (CopySlices)` and `RuntimeError: a view of a leaf Variable that requires
    grad is being used in an in-place operation` on ComfyUI models (such as Qwen-Image 2.1, Lumina,
    Flux, Wan) that perform in-place slice updates (`x[:, prefix_len:].addcmul_`, `x[:, :n] = ...`):
      1. Snapshots isolated copies of block inputs (`args`, `kwargs`) before `orig_fn` runs and
         clones `fwd_args`/`fwd_kwargs` under `grad_enabled` so inputs inside `orig_fn` are always
         non-leaf tensors (`is_leaf == False`) and never corrupt caller tensors.
      2. Discards intermediate activation tensors during the forward pass (recording only integer
         slot IDs `0..N-1` and expected tensor shapes), achieving the full ~65% VRAM reduction of
         activation checkpointing.
      3. During `loss.backward()`, clones `saved_inputs` inside `with torch.enable_grad():` so
         `rec_args`/`rec_kwargs` are non-leaf tensors (`is_leaf == False`, `CloneBackward0`), and
         snapshots `t.detach().clone()` inside `_rec_pack_hook` at the exact instant each operation
         inside `orig_fn` saves `t` for backward.
      4. Automatically clears the previous block's `recomputed` dictionary as soon as the next block
         begins unpacking so at most one block's recomputed activations exist in VRAM at any time.
    """
    if not torch.is_grad_enabled():
        return orig_fn(*args, **kwargs)

    dev = _find_first_tensor_device(args) or _find_first_tensor_device(kwargs)
    dev_type = dev.type if (dev is not None and dev.type != "cpu") else None
    if dev_type in ("privateuseone", "dml"):
        # DirectML does not use a stream-ordered caching allocator; keep activation `ID3D12Resource`
        # buffers alive in standard `SavedVariable`s until backward finishes so no buffer is released
        # on the CPU while a D3D12 command list referencing it is still open.
        fwd_args = _clone_tensors_for_grad(args)
        fwd_kwargs = _clone_tensors_for_grad(kwargs)
        dml_out = _clone_tensors_for_grad(orig_fn(*fwd_args, **fwd_kwargs))
        return _ensure_nonleaf_grad_outputs(dml_out)
    dev_idx = dev.index if dev is not None else None
    dev_mod = getattr(torch, dev_type, None) if (preserve_rng and dev_type) else None

    cpu_rng_state: Optional[torch.Tensor] = torch.get_rng_state() if preserve_rng else None
    dev_rng_state: Optional[torch.Tensor] = None
    if preserve_rng and dev_mod is not None and hasattr(dev_mod, "get_rng_state") and hasattr(dev_mod, "set_rng_state"):
        try:
            dev_rng_state = dev_mod.get_rng_state(dev_idx) if dev_idx is not None else dev_mod.get_rng_state()
        except TypeError:
            try:
                dev_rng_state = dev_mod.get_rng_state()
            except Exception:
                dev_rng_state = None
        except Exception:
            dev_rng_state = None

    ac_dev_type = dev_type or "cuda"
    ac_enabled = False
    ac_dtype = torch.bfloat16
    try:
        ac_enabled = bool(torch.is_autocast_enabled(ac_dev_type))
        ac_dtype = torch.get_autocast_dtype(ac_dev_type)
    except TypeError:
        try:
            ac_enabled = bool(torch.is_autocast_enabled())
            ac_dtype = torch.get_autocast_gpu_dtype()
        except Exception:
            pass
    except Exception:
        pass

    memo = active_ckpt_state.get("snapshot_memo") if isinstance(active_ckpt_state, dict) else None
    offload_to_cpu = (
        bool(active_ckpt_state.get("offload_inputs_to_cpu", False))
        if isinstance(active_ckpt_state, dict)
        else False
    )
    saved_inputs: Dict[str, Any] = {
        "args": _snapshot_tensors_for_recompute(args, memo=memo, offload_to_cpu=offload_to_cpu),
        "kwargs": _snapshot_tensors_for_recompute(kwargs, memo=memo, offload_to_cpu=offload_to_cpu),
    }
    if isinstance(active_ckpt_state, dict):
        saved_list = active_ckpt_state.get("saved_inputs_list")
        if isinstance(saved_list, list):
            saved_list.append(saved_inputs)
    fwd_args = _clone_tensors_for_grad(args)
    fwd_kwargs = _clone_tensors_for_grad(kwargs)

    fwd_counter = [0]
    recomputed: Dict[int, torch.Tensor] = {}
    recomputed_done = [False]

    def _fwd_pack_hook(t: torch.Tensor) -> Any:
        if isinstance(t, nn.Parameter) or is_custom_quantized_tensor(t):
            return ("param", t)
        slot = fwd_counter[0]
        fwd_counter[0] += 1
        return ("act", (slot, tuple(t.shape)))

    def _fwd_unpack_hook(packed: Any) -> torch.Tensor:
        kind, payload = packed
        if kind == "param":
            return payload

        slot, expected_shape = payload if isinstance(payload, tuple) else (int(payload), None)
        if not recomputed_done[0]:
            recomputed_done[0] = True
            if active_recompute_cache and active_recompute_cache[0] is not None and active_recompute_cache[0] is not recomputed:
                active_recompute_cache[0].clear()
            if active_recompute_cache:
                active_recompute_cache[0] = recomputed

            raw_saved_args = saved_inputs.get("args", ())
            raw_saved_kwargs = saved_inputs.get("kwargs", {})
            saved_inputs.clear()

            rec_counter = [0]

            def _rec_pack_hook(t: torch.Tensor) -> Any:
                if isinstance(t, nn.Parameter) or is_custom_quantized_tensor(t):
                    return ("param", t)
                r_slot = rec_counter[0]
                rec_counter[0] += 1
                if not t.requires_grad and t.ndim >= 2 and t.numel() >= 16384:
                    recomputed[r_slot] = t.detach()
                else:
                    with torch.no_grad():
                        recomputed[r_slot] = t.detach().clone()
                return ("act", r_slot)

            def _rec_unpack_hook(r_packed: Any) -> torch.Tensor:
                r_kind, r_payload = r_packed
                if r_kind == "param":
                    return r_payload
                return recomputed[r_payload]

            prev_cpu_rng: Optional[torch.Tensor] = torch.get_rng_state() if preserve_rng else None
            prev_dev_rng: Optional[torch.Tensor] = None
            if preserve_rng and dev_mod is not None and dev_rng_state is not None:
                try:
                    prev_dev_rng = dev_mod.get_rng_state(dev_idx) if dev_idx is not None else dev_mod.get_rng_state()
                except TypeError:
                    try:
                        prev_dev_rng = dev_mod.get_rng_state()
                    except Exception:
                        prev_dev_rng = None
                except Exception:
                    prev_dev_rng = None

            try:
                if preserve_rng and cpu_rng_state is not None:
                    torch.set_rng_state(cpu_rng_state)
                if preserve_rng and dev_mod is not None and dev_rng_state is not None:
                    try:
                        if dev_idx is not None:
                            dev_mod.set_rng_state(dev_rng_state, dev_idx)
                        else:
                            dev_mod.set_rng_state(dev_rng_state)
                    except TypeError:
                        try:
                            dev_mod.set_rng_state(dev_rng_state)
                        except Exception:
                            pass
                    except Exception:
                        pass

                ac_ctx = (
                    torch.amp.autocast(device_type=ac_dev_type, dtype=ac_dtype, enabled=True)
                    if ac_enabled
                    else contextlib.nullcontext()
                )
                with torch.enable_grad(), ac_ctx:
                    # Clone inside `torch.enable_grad()` BEFORE entering `saved_tensors_hooks` so every
                    # `requires_grad=True` input tensor is a non-leaf (`is_leaf == False`, `CloneBackward0`).
                    # This prevents `RuntimeError: a view of a leaf Variable that requires grad is being
                    # used in an in-place operation` when `orig_fn` runs `x[:, prefix_len:].addcmul_(...)`.
                    rec_args = _clone_tensors_for_grad(raw_saved_args)
                    rec_kwargs = _clone_tensors_for_grad(raw_saved_kwargs)
                    del raw_saved_args, raw_saved_kwargs
                    with torch.autograd.graph.saved_tensors_hooks(_rec_pack_hook, _rec_unpack_hook):
                        _ = orig_fn(*rec_args, **rec_kwargs)
            finally:
                if preserve_rng and prev_cpu_rng is not None:
                    torch.set_rng_state(prev_cpu_rng)
                if preserve_rng and dev_mod is not None and prev_dev_rng is not None:
                    try:
                        if dev_idx is not None:
                            dev_mod.set_rng_state(prev_dev_rng, dev_idx)
                        else:
                            dev_mod.set_rng_state(prev_dev_rng)
                    except TypeError:
                        try:
                            dev_mod.set_rng_state(prev_dev_rng)
                        except Exception:
                            pass
                    except Exception:
                        pass

        cand = recomputed.get(slot)
        if cand is not None and (expected_shape is None or tuple(cand.shape) == expected_shape):
            return cand
        if expected_shape is not None:
            for offset in range(1, max(1, len(recomputed)) + 1):
                for adj_slot in (slot - offset, slot + offset):
                    adj = recomputed.get(adj_slot)
                    if adj is not None and tuple(adj.shape) == expected_shape:
                        return adj
        if cand is not None:
            return cand
        raise KeyError(f"[OmniLoRA] Missing recomputed activation slot {slot} (expected shape {expected_shape})")

    with torch.autograd.graph.saved_tensors_hooks(_fwd_pack_hook, _fwd_unpack_hook):
        out = orig_fn(*fwd_args, **fwd_kwargs)

    return _ensure_nonleaf_grad_outputs(out)


@dataclass
class TrainingConfig:
    """End-to-end training hyperparameters."""

    output_dir: str = "./outputs/omni_lora"
    conda_env: str = "xpu"
    backend: str = "auto"
    device_index: int = 0
    dtype: str = "auto"
    vram_mode: str = "auto"
    reserve_vram_gb: float = 0.0
    model_id: str = "auto"
    resolution: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "radamw"
    learning_rate: float = 1e-4
    lora_plus_lr_ratio: float = 1.0
    lr_scheduler: str = "cosine"
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    weight_decay: float = 1e-3
    max_train_steps: int = 1000
    save_every_n_steps: int = 250
    max_grad_norm: float = 1.0
    seed: int = 42
    gradient_checkpointing: bool = True
    compile_mode: str = "eager"
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class StepMetrics:
    """Telemetry captured at each optimization step."""

    step: int
    loss: float
    lr: float
    step_time_ms: float
    allocated_vram_mb: float
    reserved_vram_mb: float
    grad_scaler_active: bool = False
    timestep: float = 0.5


class LoRATrainingPipeline:
    """
    Hardware-agnostic PyTorch LoRA training pipeline.
    Executes identically across all 8 backends (`xpu`, `cuda`, `rocm`, `npu`, `hpu`, `dml`, `mps`, `cpu`).
    Strictly avoids `torch.float64` (`.double()`) so GPUs without hardware FP64 (eg Intel Arc,
    Apple MPS, DirectML) never raise `Required aspect fp64 is not supported on the device`.
    """

    def __init__(
        self,
        backend: HardwareBackend,
        model_adapter: BaseModelAdapter,
        config: TrainingConfig,
    ) -> None:
        self.backend = backend
        self.model_adapter = model_adapter
        self.config = config

        self.backbone: Optional[nn.Module] = None
        self.injected_modules: List[str] = []
        self._trainable_params: List[nn.Parameter] = []
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.grad_scaler: Optional[FP32GradScaler] = None

    @staticmethod
    def clip_grad_norm_fp32_(parameters: List[nn.Parameter], max_norm: float) -> float:
        """
        Pure-FP32 L2 gradient norm clipping:
        - Never uses `torch.float64` (`.double()`).
        - Avoids `aten::linalg_vector_norm` on `float16` tensors (which is unsupported in
          `torch-directml` `privateuseone` and overflows when squared norms exceed `65504.0`).
        - Accumulates squared norms in `torch.float32` on the target device via a single batched
          stack reduction and a single host read.
        """
        if not parameters or max_norm <= 0.0:
            return 0.0
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return 0.0
        with torch.no_grad():
            dev = grads[0].device
            sq_norms = [
                (g.detach().to(device=dev, dtype=torch.float32) ** 2).sum()
                for g in grads
            ]
            total_sq = torch.stack(sq_norms).sum()
            total_norm = float(torch.sqrt(total_sq).item())
            if math.isfinite(total_norm) and total_norm > max_norm:
                clip_coef = float(max_norm / (total_norm + 1e-6))
                for g in grads:
                    g.mul_(clip_coef)
            return total_norm

    def _resolve_lora_torch_dtype(self) -> torch.dtype:
        """
        Resolve the `torch.dtype` for trainable LoRA adapter weights (`lora_A` and `lora_B`).
        - `"bf16"` (default) or `"auto"`: uses `torch.bfloat16` when supported by the backend
          (`supports_bf16=True`), falling back automatically to `torch.float16` (or
          `self.backend.torch_dtype`) if hardware does not support `bf16`.
        - `"fp32"` / `"float32"`: keeps master LoRA weights in 32-bit float even when base model
          compute runs in `bf16` or `fp16`.
        - `"fp16"` / `"float16"`: stores LoRA weights in 16-bit float.
        """
        raw = (getattr(self.config.lora, "lora_dtype", None) or "bf16").strip().lower()
        caps = self.backend.get_capabilities()
        if raw in ("fp32", "float32"):
            return torch.float32
        if raw in ("fp16", "float16", "half"):
            return torch.float16
        if raw in ("bf16", "bfloat16", "auto"):
            if caps.supports_bf16:
                return torch.bfloat16
            if caps.supports_fp16:
                return torch.float16
            return self.backend.torch_dtype
        return self.backend.torch_dtype

    def _build_fp32_safe_optimizer(self, trainable_params: List[nn.Parameter]) -> torch.optim.Optimizer:
        """
        Construct the requested optimizer (`radamw`, `adamw`, `adafactor`, `sgd`) with `foreach=False`
        so multi-tensor kernels never allocate internal `float64` step counters on XPU/MPS/DML.
        - `radamw` (and legacy alias `radam`) enables `decoupled_weight_decay=True` for AdamW weight decay + automatic variance warmup.
        - `adafactor` implements Shazeer & Stern (2018) factored 2nd moments (`exp_avg_sq_row` + `exp_avg_sq_col`,
          zero 1st-moment `exp_avg`, zero persistent `master_param`), on-device RMS update clipping (`d=1.0`),
          and unbiased 16-bit stochastic rounding, cutting optimizer VRAM by ~75-85%.
        - Supports `LoRA+` differential learning rates (`lora_plus_lr_ratio > 1.0`), assigning
          `lr` to `lora_A` (`lora_down`) and `lr * lora_plus_lr_ratio` to `lora_B` (`lora_up`).
        - When any trainable LoRA parameter is in `torch.float16` or on DirectML (`privateuseone`) with
          `radamw`/`adamw`/`sgd`, maintains 32-bit `master_param`, `exp_avg`, and `exp_avg_sq` optimizer
          state tensors and executes all moment updates in pure `float32` elementwise ops (`*`, `+`, `/`, `sqrt`).
        """
        opt_name = (self.config.optimizer_type or "radamw").strip().lower()
        lr = float(self.config.learning_rate)
        wd = float(self.config.weight_decay)
        lora_plus_ratio = max(1.0, float(getattr(self.config, "lora_plus_lr_ratio", 2.0)))

        param_groups: Any = trainable_params
        if self.backbone is not None and abs(lora_plus_ratio - 1.0) > 1e-6:
            lora_b_ids = {
                id(m.lora_B.weight)
                for m in self.backbone.modules()
                if isinstance(m, LoRALinear) and hasattr(m, "lora_B")
            }
            group_a = [p for p in trainable_params if id(p) not in lora_b_ids]
            group_b = [p for p in trainable_params if id(p) in lora_b_ids]
            if group_a and group_b:
                param_groups = [
                    {"params": group_a, "lr": lr},
                    {"params": group_b, "lr": lr * lora_plus_ratio},
                ]

        if opt_name == "adafactor":
            class PureFP32Adafactor(torch.optim.Optimizer):
                def __init__(
                    self,
                    params: Any,
                    lr: float = 1e-4,
                    weight_decay: float = 1e-2,
                    clip_threshold: float = 1.0,
                    decay_rate: float = -0.8,
                    beta2_cap: float = 0.999,
                    eps1: float = 1e-30,
                ) -> None:
                    defaults = dict(
                        lr=float(lr),
                        weight_decay=float(weight_decay),
                        clip_threshold=float(clip_threshold),
                        decay_rate=float(decay_rate),
                        beta2_cap=float(beta2_cap),
                        eps1=float(eps1),
                    )
                    super().__init__(params, defaults)

                @torch.no_grad()
                def step(self, closure: Optional[Any] = None) -> Optional[Any]:
                    loss = None
                    if closure is not None:
                        with torch.enable_grad():
                            loss = closure()
                    for group in self.param_groups:
                        g_lr = float(group["lr"])
                        g_wd = float(group.get("weight_decay", 0.0))
                        clip_d = max(1e-3, float(group.get("clip_threshold", 1.0)))
                        decay_r = float(group.get("decay_rate", -0.8))
                        beta2_cap = float(group.get("beta2_cap", 0.999))
                        eps1 = max(1e-30, float(group.get("eps1", 1e-30)))

                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            grad = p.grad.detach().to(dtype=torch.float32)
                            state = self.state[p]
                            factored = p.ndim >= 2
                            if len(state) == 0:
                                state["step"] = 0
                                if factored:
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        p.shape[:-1], dtype=torch.float32, device=p.device
                                    )
                                    state["exp_avg_sq_col"] = torch.zeros(
                                        p.shape[:-2] + p.shape[-1:], dtype=torch.float32, device=p.device
                                    )
                                else:
                                    state["exp_avg_sq"] = torch.zeros(
                                        p.shape, dtype=torch.float32, device=p.device
                                    )

                            state["step"] = int(state["step"]) + 1
                            t = state["step"]
                            rho_t = min(beta2_cap, 1.0 - (float(t) ** decay_r))
                            one_minus_rho = 1.0 - rho_t
                            grad_sq = grad * grad + eps1

                            if factored:
                                r_sq: torch.Tensor = state["exp_avg_sq_row"]
                                c_sq: torch.Tensor = state["exp_avg_sq_col"]
                                r_sq.mul_(rho_t).add_(grad_sq.mean(dim=-1) * one_minus_rho)
                                c_sq.mul_(rho_t).add_(grad_sq.mean(dim=-2) * one_minus_rho)
                                r_mean = r_sq.mean(dim=-1, keepdim=True).clamp(min=eps1)
                                r_factor = torch.rsqrt((r_sq / r_mean).clamp(min=eps1)).unsqueeze(-1)
                                c_factor = torch.rsqrt(c_sq.clamp(min=eps1)).unsqueeze(-2)
                                update = grad * (r_factor * c_factor)
                            else:
                                v_sq: torch.Tensor = state["exp_avg_sq"]
                                v_sq.mul_(rho_t).add_(grad_sq * one_minus_rho)
                                update = grad * torch.rsqrt(v_sq.clamp(min=eps1))

                            # On-device RMS update clipping (d = clip_threshold = 1.0) with zero CPU-GPU syncs
                            rms_div = (torch.sqrt((update * update).mean()) / clip_d).clamp(min=1.0)
                            update = update / rms_div

                            w_f32 = p.detach().to(dtype=torch.float32)
                            if g_wd != 0.0:
                                w_f32 = w_f32 * (1.0 - g_lr * g_wd)
                            w_f32 = w_f32 - update * g_lr

                            if p.dtype in (torch.float16, torch.bfloat16) and p.device.type not in ("privateuseone", "dml"):
                                # Unbiased stochastic rounding for 16-bit parameter storage without FP32 master weights
                                mantissa_eps = 0.0009765625 if p.dtype == torch.float16 else 0.0078125
                                min_ulp = 6.103515625e-5 if p.dtype == torch.float16 else 1.17549435e-38
                                try:
                                    ulp = (w_f32.abs() * mantissa_eps).clamp(min=min_ulp)
                                    dither = (torch.rand_like(w_f32) - 0.5) * ulp
                                    w_f32 = w_f32 + dither
                                except Exception:
                                    pass

                            p.copy_(w_f32.to(dtype=p.dtype))
                    return loss

            return PureFP32Adafactor(param_groups, lr=lr, weight_decay=wd)

        has_lr_warmup = int(getattr(self.config, "warmup_steps", 0) or 0) > 0

        if opt_name == "sgd":
            opt: torch.optim.Optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=0.9,
                weight_decay=wd,
                nesterov=True,
                foreach=False,
            )
        elif opt_name == "adam":
            opt = torch.optim.Adam(
                param_groups,
                lr=lr,
                betas=(0.9, 0.99),
                weight_decay=wd,
                eps=1e-8,
                foreach=False,
            )
        elif opt_name in ("radamw", "radam"):
            # When an explicit LR warmup schedule (`warmup_steps > 0`) is already active,
            # stacking RAdam's analytic variance rectification (`rect`) on top of linear warmup
            # suppresses the effective LR by ~6.6x across a 500-step LoRA run. Use `betas=(0.9, 0.99)`
            # (`rho_inf = 199`) and only apply analytic `rect` when `warmup_steps == 0`.
            if has_lr_warmup:
                opt = torch.optim.AdamW(
                    param_groups,
                    lr=lr,
                    betas=(0.9, 0.99),
                    weight_decay=wd,
                    eps=1e-8,
                    foreach=False,
                )
            else:
                try:
                    opt = torch.optim.RAdam(
                        param_groups,
                        lr=lr,
                        betas=(0.9, 0.99),
                        weight_decay=wd,
                        eps=1e-8,
                        decoupled_weight_decay=True,
                        foreach=False,
                    )
                except TypeError:
                    opt = torch.optim.RAdam(
                        param_groups,
                        lr=lr,
                        betas=(0.9, 0.99),
                        weight_decay=wd,
                        eps=1e-8,
                        foreach=False,
                    )
        else:
            opt = torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=(0.9, 0.99),
                weight_decay=wd,
                eps=1e-8,
                foreach=False,
            )

        needs_fp32_master_step = any(
            p.dtype in (torch.float16, torch.bfloat16) or p.device.type in ("privateuseone", "dml")
            for p in trainable_params
        )
        if needs_fp32_master_step:
            def _fp32_master_step(
                self_opt: torch.optim.Optimizer,
                closure: Optional[Any] = None,
            ) -> Optional[Any]:
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
                with torch.no_grad():
                    for group in self_opt.param_groups:
                        g_lr = float(group["lr"])
                        g_wd = float(group.get("weight_decay", 0.0))
                        betas = group.get("betas", (0.9, 0.99))
                        beta1 = float(betas[0]) if isinstance(betas, (tuple, list)) else 0.9
                        beta2 = float(betas[1]) if isinstance(betas, (tuple, list)) else 0.99
                        eps = max(float(group.get("eps", 1e-8)), 1e-8)
                        momentum = float(group.get("momentum", 0.9))
                        nesterov = bool(group.get("nesterov", True))

                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            grad = p.grad.detach().to(dtype=torch.float32)
                            state = self_opt.state[p]
                            if "master_param" not in state:
                                state["step"] = 0
                                state["master_param"] = p.detach().to(dtype=torch.float32).clone()
                                if opt_name == "sgd":
                                    state["momentum_buffer"] = torch.zeros_like(state["master_param"])
                                else:
                                    state["exp_avg"] = torch.zeros_like(state["master_param"])
                                    state["exp_avg_sq"] = torch.zeros_like(state["master_param"])

                            state["step"] = int(state["step"]) + 1
                            t = state["step"]
                            w_f32: torch.Tensor = state["master_param"]

                            if opt_name == "sgd":
                                d_p = grad + w_f32 * g_wd if g_wd != 0.0 else grad
                                buf: torch.Tensor = state["momentum_buffer"]
                                buf.mul_(momentum).add_(d_p)
                                step_vec = (d_p + buf * momentum) if nesterov else buf
                                w_f32.add_(step_vec * (-g_lr))
                            else:
                                exp_avg: torch.Tensor = state["exp_avg"]
                                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                                if g_wd != 0.0:
                                    if opt_name == "adam":
                                        grad = grad + w_f32 * g_wd
                                    else:
                                        w_f32.mul_(1.0 - g_lr * g_wd)

                                exp_avg.mul_(beta1).add_(grad * (1.0 - beta1))
                                exp_avg_sq.mul_(beta2).add_((grad * grad) * (1.0 - beta2))

                                bias_corr1 = 1.0 - (beta1 ** t)
                                bias_corr2 = 1.0 - (beta2 ** t)

                                if opt_name in ("radamw", "radam") and not has_lr_warmup:
                                    rho_inf = 2.0 / (1.0 - beta2) - 1.0
                                    beta2_t = beta2 ** t
                                    rho_t = rho_inf - 2.0 * float(t) * beta2_t / max(1e-12, 1.0 - beta2_t)
                                    if rho_t > 5.0:
                                        rect = math.sqrt(
                                            ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                                            / max(1e-12, (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                                        )
                                        step_size = g_lr * rect / bias_corr1
                                        denom = torch.sqrt(exp_avg_sq / bias_corr2) + eps
                                        w_f32.add_((exp_avg / denom) * (-step_size))
                                    else:
                                        step_size = g_lr / bias_corr1
                                        w_f32.add_(exp_avg * (-step_size))
                                else:
                                    step_size = g_lr / bias_corr1
                                    denom = torch.sqrt(exp_avg_sq) / math.sqrt(bias_corr2) + eps
                                    w_f32.add_((exp_avg / denom) * (-step_size))

                            p.copy_(w_f32.to(dtype=p.dtype))
                return loss

            opt.step = types.MethodType(_fp32_master_step, opt)  # type: ignore[method-assign]

        return opt

    def _build_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[torch.optim.lr_scheduler.LRScheduler, str, int]:
        """
        Build an FP64-free learning rate scheduler supporting linear warmup followed by:
        - `"cosine"` (default): Cosine annealing decay to `min_lr_ratio * base_lr`.
        - `"rex"`: Reflected Exponential (REX) schedule `(1 - p) / (1 - 0.5 * p)` to `min_lr_ratio * base_lr`.
        - `"constant_with_warmup"`: Linear warmup then constant `1.0 * base_lr`.
        - `"constant"`: Constant `1.0 * base_lr` (or warmup + constant if `warmup_steps > 0`).
        - `"linear"` / `"polynomial"`: Linear decay to `min_lr_ratio * base_lr`.
        - `"cosine_with_restarts"`: 2-cycle cosine annealing with warm restarts to `min_lr_ratio * base_lr`.

        Uses `LambdaLR` so multiple parameter groups (such as `LoRA+` `lora_A` and `lora_B`)
        scale proportionally to their respective base learning rates.
        """
        sched_name = (getattr(self.config, "lr_scheduler", "cosine") or "cosine").strip().lower()
        total_steps = max(1, int(self.config.max_train_steps))
        raw_warmup = max(0, int(getattr(self.config, "warmup_steps", 0) or 0))
        min_ratio = min(1.0, max(0.0, float(getattr(self.config, "min_lr_ratio", 0.1))))

        if total_steps <= 1:
            effective_warmup = 0
        elif sched_name == "constant":
            effective_warmup = min(raw_warmup, total_steps // 2)
        elif sched_name == "constant_with_warmup":
            default_w = raw_warmup if raw_warmup > 0 else max(1, total_steps // 10)
            effective_warmup = min(default_w, max(1, total_steps - 1))
        else:
            effective_warmup = min(raw_warmup, total_steps // 2)

        def _lr_lambda(epoch: int) -> float:
            step_num = int(epoch) + 1  # 1-indexed step (epoch=0 corresponds to Step 1)
            if effective_warmup > 0 and step_num <= effective_warmup:
                return float(step_num) / float(effective_warmup)
            if sched_name in ("constant", "constant_with_warmup"):
                return 1.0
            if effective_warmup > 0:
                decay_steps = max(1, total_steps - effective_warmup)
                progress = min(1.0, max(0.0, float(step_num - effective_warmup) / float(decay_steps)))
            else:
                decay_steps = max(1, total_steps - 1)
                progress = min(1.0, max(0.0, float(step_num - 1) / float(decay_steps)))
            if sched_name in ("linear", "polynomial"):
                return float(1.0 - (1.0 - min_ratio) * progress)
            if sched_name == "rex":
                denom = max(1e-6, 1.0 - 0.5 * progress)
                rex_val = (1.0 - progress) / denom
                return float(min_ratio + (1.0 - min_ratio) * rex_val)
            if sched_name == "cosine_with_restarts":
                cycle_prog = (progress * 2.0) % 1.0 if progress < 1.0 else 1.0
                cos_val = 0.5 * (1.0 + math.cos(math.pi * cycle_prog))
                return float(min_ratio + (1.0 - min_ratio) * cos_val)
            cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_ratio + (1.0 - min_ratio) * cos_val)

        orig_step_func = getattr(optimizer.step, "__func__", None)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        if orig_step_func is not None and not hasattr(optimizer.step, "__func__"):
            try:
                setattr(optimizer.step, "__func__", orig_step_func)
            except Exception:
                pass
        return scheduler, sched_name, effective_warmup

    def setup(
        self,
        external_backbone: Optional[nn.Module] = None,
        initial_lora_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Attach ComfyUI backbone, inject LoRA adapters, optionally load starting LoRA weights, initialize optimizer, and configure optional FP32GradScaler."""
        torch.set_default_dtype(torch.float32)
        try:
            import comfy.model_management as mm  # type: ignore

            mm.in_training = True
        except Exception:
            pass
        self.backend.configure_environment()
        _install_out_kwarg_autograd_wrappers()
        self.backend.manual_seed_all(self.config.seed)

        device = self.backend.get_device()
        dtype = self.backend.torch_dtype
        lora_torch_dtype = self._resolve_lora_torch_dtype()

        if external_backbone is not None:
            self.backend.validate_quantized_model_compatibility(external_backbone)
            backbone = sanitize_module_no_fp64(external_backbone)
        else:
            backbone = self.model_adapter.build_backbone(dtype=dtype, device=torch.device("cpu"))
            self.backend.validate_quantized_model_compatibility(backbone)

        # Assign `self.backbone` BEFORE `LoRAInjector.inject()` so that if any subsequent step in
        # `setup()` fails or is interrupted, caller `finally:` blocks (`LoRAInjector.uninject(pipeline.backbone)`)
        # always restore the live ComfyUI model cleanly.
        self.backbone = backbone
        self.injected_modules = LoRAInjector.inject(
            model=backbone,
            config=self.config.lora,
            lora_dtype=lora_torch_dtype,
            device=device,
        )
        if not self.injected_modules and external_backbone is not None:
            backbone = self.model_adapter.build_backbone(dtype=dtype, device=torch.device("cpu"))
            self.backbone = backbone
            self.injected_modules = LoRAInjector.inject(
                model=backbone,
                config=self.config.lora,
                lora_dtype=lora_torch_dtype,
                device=device,
            )

        resumed_lora_layers = 0
        if initial_lora_state:
            load_stats = LoRAInjector.load_lora_state_dict(
                backbone,
                initial_lora_state,
                comfy_model=getattr(self.model_adapter, "comfy_model", None),
            )
            resumed_lora_layers = int(load_stats.get("loaded_layers", 0))

        # Release any cached allocator blocks from unloaded text encoders / VAE BEFORE
        # querying free VRAM in `enforce_vram_budget` so the full GPU capacity is available.
        try:
            self.backend.synchronize()
            self.backend.empty_cache()
        except Exception:
            pass

        # Enforce deterministic VRAM ledger budget: keeps LoRA adapters + norms on GPU,
        # places frozen base weights on GPU up to `max_resident_model_mb`, and leaves overflow
        # frozen base weights on CPU to stream per-layer via `_StreamedLinearFn`.
        vram_budget = self.backend.enforce_vram_budget(
            backbone,
            vram_mode=self.config.vram_mode,
            resolution=self.config.resolution,
            batch_size=self.config.batch_size,
            reserve_vram_gb=self.config.reserve_vram_gb,
            optimizer_type=self.config.optimizer_type,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )
        self.backbone = self.backend.optimize_module(
            backbone,
            mode=self.config.compile_mode,
            move_to_device=False,
        )
        self.backbone.train()
        self._patch_inplace_model_helpers(self.backbone)

        total_params = sum(p.numel() for p in self.backbone.parameters())
        offloaded_mb = float(vram_budget.get("offloaded_model_mb", 0.0))
        resident_mb = float(vram_budget.get("resident_model_mb", 0.0))
        usable_mb = float(vram_budget.get("usable_vram_mb", 0.0))
        if not self.config.gradient_checkpointing and (
            offloaded_mb > 0.0
            or total_params >= 50_000_000
            or (usable_mb > 0.0 and resident_mb > usable_mb * 0.40)
        ):
            self.config.gradient_checkpointing = True
            logging.info(
                "[OmniLoRA] Auto-enabled gradient_checkpointing=True to keep large model activations within VRAM."
            )

        if self.backend.backend_id != "cpu" and (
            bool(vram_budget.get("offload_ckpt_inputs", 0.0))
            or offloaded_mb > 0.0
            or (usable_mb > 0.0 and (usable_mb - resident_mb) < 4200.0)
        ):
            setattr(self.backbone, "_omni_offload_ckpt_inputs", True)

        checkpointed_blocks = 0
        if self.config.gradient_checkpointing:
            checkpointed_blocks = self._enable_gradient_checkpointing(self.backbone)

        trainable_params = LoRAInjector.get_trainable_parameters(self.backbone)
        if not trainable_params:
            raise RuntimeError("No trainable LoRA parameters were injected. Check `lora.target_modules`.")
        self._trainable_params = trainable_params

        self.optimizer = self._build_fp32_safe_optimizer(trainable_params)
        self.lr_scheduler, resolved_sched_name, effective_warmup = self._build_lr_scheduler(self.optimizer)

        # Activate pure-FP32 GradScaler when either compute dtype or LoRA parameter dtype is FP16
        use_scaler = (dtype == torch.float16) or (lora_torch_dtype == torch.float16)
        self.grad_scaler = self.backend.create_grad_scaler(enabled=use_scaler)

        trainable_count = sum(p.numel() for p in trainable_params)
        target_summary = LoRAInjector.summarize_targets(self.backbone, self.injected_modules)

        return {
            "backend": self.backend.backend_id,
            "device": str(device),
            "dtype": str(dtype),
            "lora_dtype": str(lora_torch_dtype),
            "optimizer": (self.config.optimizer_type or "radamw").lower(),
            "lr_scheduler": resolved_sched_name,
            "warmup_steps": effective_warmup,
            "min_lr_ratio": float(getattr(self.config, "min_lr_ratio", 0.1)),
            "lora_plus_lr_ratio": float(getattr(self.config, "lora_plus_lr_ratio", 2.0)),
            "weight_decay": float(getattr(self.config, "weight_decay", 1e-2)),
            "grad_scaler_enabled": self.grad_scaler is not None,
            "gradient_checkpointing": bool(self.config.gradient_checkpointing),
            "checkpointed_blocks": checkpointed_blocks,
            "fp64_free_verified": True,
            "injected_layer_count": len(self.injected_modules),
            "target_summary": target_summary,
            "resumed_lora_layers": resumed_lora_layers,
            "total_parameters": total_params,
            "trainable_parameters": trainable_count,
            "trainable_ratio_pct": round((trainable_count / max(1, total_params)) * 100.0, 4),
            "vram_budget": vram_budget,
        }

    def train_step(
        self,
        step_idx: int,
        batch: Optional[Dict[str, Any]] = None,
        micro_batches: Optional[List[Dict[str, Any]]] = None,
    ) -> StepMetrics:
        """Run a single gradient-accumulated training step with `torch.amp.autocast` and synchronized step timing."""
        if self.backbone is None or self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError("Pipeline is not initialized. Call `pipeline.setup()` first.")

        try:
            import comfy.model_management as mm  # type: ignore

            mm.in_training = True
        except Exception:
            pass

        device = self.backend.get_device()
        dtype = self.backend.torch_dtype
        accum_steps = max(1, self.config.gradient_accumulation_steps)

        self.backend.synchronize()
        t0 = time.perf_counter()

        max_oom_retries = 3
        oom_attempt = 0
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = torch.zeros((), dtype=torch.float32, device=device)
            mean_loss = 0.0
            try:
                for sub_idx in range(accum_steps):
                    try:
                        import comfy.model_management as mm  # type: ignore

                        if hasattr(mm, "throw_exception_if_processing_interrupted"):
                            mm.throw_exception_if_processing_interrupted()
                    except ImportError:
                        pass
                    if micro_batches and len(micro_batches) > 0:
                        step_batch = micro_batches[sub_idx % len(micro_batches)]
                    elif batch is not None:
                        step_batch = batch
                    else:
                        step_batch = self.model_adapter.create_synthetic_batch(
                            batch_size=self.config.batch_size,
                            resolution=self.config.resolution,
                            device=device,
                            dtype=dtype,
                        )

                    with (
                        self.backend.get_sdpa_context(),
                        self.backend.autocast_context(enabled=True),
                        _inplace_safe_outer_autograd_context(
                            gradient_checkpointing=self.config.gradient_checkpointing
                        ),
                    ):
                        raw_loss = self.model_adapter.compute_training_loss(
                            backbone=self.backbone,
                            batch=step_batch,
                            device=device,
                            dtype=dtype,
                        )
                        # Ensure loss is strictly float32 (never float64)
                        loss = raw_loss.to(dtype=torch.float32) / float(accum_steps)

                    try:
                        if self.grad_scaler is not None:
                            self.grad_scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    finally:
                        _clear_checkpoint_recompute_state(self.backbone)

                    # Accumulate detached on-device scalar and flush the command queue per sub-step
                    # so multi-substep gradient accumulation never exceeds the Windows WDDM 2s TDR timeout
                    # (`UR_RESULT_ERROR_DEVICE_LOST`) or piles up unfreed sub-step activation buffers.
                    accumulated_loss = accumulated_loss + raw_loss.detach().to(device=device, dtype=torch.float32)
                    del raw_loss, loss
                    if accum_steps > 1 or self.backend.backend_id == "tpu":
                        self.backend.synchronize()

                # Single host-device scalar read inside the OOM-guarded try block
                mean_loss = float((accumulated_loss / float(accum_steps)).item())
                del accumulated_loss
                break
            except RuntimeError as exc:
                err_msg = str(exc).lower()
                oom_cls = getattr(torch, "OutOfMemoryError", None)
                is_oom = (oom_cls is not None and isinstance(exc, oom_cls)) or any(
                    k in err_msg
                    for k in (
                        "out of memory",
                        "bad_alloc",
                        "ur_result_error_out_of",
                        "insufficient memory",
                        "failed to allocate",
                        "could not allocate",
                    )
                )
                if not is_oom or oom_attempt >= max_oom_retries:
                    _clear_checkpoint_recompute_state(self.backbone)
                    raise
                oom_attempt += 1
                raw_loss = None
                loss = None
                accumulated_loss = None
                self.optimizer.zero_grad(set_to_none=True)
                _clear_checkpoint_recompute_state(self.backbone)
                try:
                    if exc.__traceback__ is not None:
                        traceback.clear_frames(exc.__traceback__)
                except Exception:
                    pass
                gc.collect()
                ckpt_newly_enabled = False
                if not self.config.gradient_checkpointing:
                    self.config.gradient_checkpointing = True
                    self._enable_gradient_checkpointing(self.backbone)
                    ckpt_newly_enabled = True
                if not getattr(self.backbone, "_omni_offload_ckpt_inputs", False):
                    setattr(self.backbone, "_omni_offload_ckpt_inputs", True)
                    ckpt_newly_enabled = True
                ckpt_state = getattr(self.backbone, "_omni_ckpt_state", None)
                if isinstance(ckpt_state, dict):
                    ckpt_state["offload_inputs_to_cpu"] = True
                demoted_mb = self.backend.demote_resident_layers(self.backbone, target_demote_mb=1536.0)
                gc.collect()
                try:
                    self.backend.empty_cache()
                except Exception:
                    pass
                if demoted_mb <= 0 and not ckpt_newly_enabled:
                    raise
                logging.warning(
                    f"[OmniLoRA] Memory pressure detected at step {step_idx}; "
                    f"auto-offloaded {demoted_mb:.0f} MB of resident base weights to CPU "
                    f"(gradient_checkpointing={self.config.gradient_checkpointing}, retry {oom_attempt}/{max_oom_retries})."
                )
        if not math.isfinite(mean_loss):
            self.optimizer.zero_grad(set_to_none=True)
            logging.error(f"[OmniLoRA] Non-finite loss ({mean_loss}) at step {step_idx}. Aborting step.")
            raise RuntimeError(
                f"[OmniLoRA] Training loss became non-finite ({mean_loss}) at step {step_idx}. "
                "Training aborted to prevent corrupting LoRA weights."
            )

        trainable_params = self._trainable_params or LoRAInjector.get_trainable_parameters(self.backbone)
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            if self.config.max_grad_norm > 0:
                self.clip_grad_norm_fp32_(trainable_params, self.config.max_grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            if self.config.max_grad_norm > 0:
                self.clip_grad_norm_fp32_(trainable_params, self.config.max_grad_norm)
            self.optimizer.step()

        applied_lr = float(self.optimizer.param_groups[0]["lr"])
        self.lr_scheduler.step()
        # Free gradient buffers immediately after optimizer.step() so .grad never coexists
        # with next step's forward activations.
        self.optimizer.zero_grad(set_to_none=True)

        # Synchronize accelerator stream BEFORE measuring step duration so optimizer.step() is included.
        # Note: Do NOT call `self.backend.empty_cache()` inside `train_step()` on XPU unless layers were
        # explicitly demoted — unmapping active PyTorch caching-allocator pools mid-training forces Level Zero
        # `zeMemFree`/`zeMemAllocDevice` churn and can trigger `UR_RESULT_ERROR_UNKNOWN` / `UR_RESULT_ERROR_DEVICE_LOST`.
        self.backend.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mem_snap = self.backend.get_memory_snapshot()

        # Proactive WDDM / Level Zero PCIe spill guard & user reserve_vram_gb ceiling across all steps:
        # If `reserved_mb` (allocator pool) exceeds the user's requested reserve ceiling or 95% of
        # physical VRAM on XPU/DML (e.g. when a larger resolution image or emulated QuantizedTensor op
        # expands the pool above physical VRAM into shared system RAM on Step 4+), first release unused
        # cached allocator blocks (`empty_cache`) and only demote resident `base_layer` weights if needed.
        if self.backend.backend_id not in ("cpu", "mps"):
            reserve_user_mb = max(0.0, float(self.config.reserve_vram_gb or 0.0) * 1024.0)
            ceiling_mb = (
                max(1024.0, mem_snap.total_vram_mb - reserve_user_mb)
                if reserve_user_mb > 0.0
                else (mem_snap.total_vram_mb * 0.95 if self.backend.backend_id in ("xpu", "dml") else 0.0)
            )
            if (
                mem_snap.total_vram_mb > 0
                and ceiling_mb > 0.0
                and mem_snap.reserved_mb > ceiling_mb
            ):
                if self.backend.backend_id != "xpu" and (mem_snap.reserved_mb - mem_snap.allocated_mb) > 256.0:
                    try:
                        self.backend.empty_cache()
                        mem_snap = self.backend.get_memory_snapshot()
                    except Exception:
                        pass
                if mem_snap.reserved_mb > ceiling_mb:
                    trim_count = int(getattr(self.backbone, "_omni_pool_trim_count", 0))
                    if trim_count < 3:
                        # On XPU, try trimming unused cached allocator blocks once before permanently
                        # demoting resident weights to CPU when free pool slack exceeds 512 MB.
                        if self.backend.backend_id == "xpu" and (mem_snap.reserved_mb - mem_snap.allocated_mb) > 512.0:
                            try:
                                self.backend.empty_cache()
                                mem_snap = self.backend.get_memory_snapshot()
                            except Exception:
                                pass
                        if mem_snap.reserved_mb > ceiling_mb:
                            setattr(self.backbone, "_omni_pool_trim_count", trim_count + 1)
                            target_after_mb = (
                                max(1024.0, mem_snap.total_vram_mb - reserve_user_mb - 256.0)
                                if reserve_user_mb > 0.0
                                else (mem_snap.total_vram_mb * 0.88)
                            )
                            trim_mb = max(512.0, mem_snap.reserved_mb - target_after_mb)
                            demoted_mb = self.backend.demote_resident_layers(self.backbone, target_demote_mb=trim_mb)
                            if demoted_mb > 0:
                                try:
                                    self.backend.empty_cache()
                                except Exception:
                                    pass
                                mem_snap = self.backend.get_memory_snapshot()
                                logging.info(
                                    f"[OmniLoRA] Proactively offloaded {demoted_mb:.0f} MB of resident base weights to CPU "
                                    f"at Step {step_idx} and trimmed allocator pool (pool now {mem_snap.reserved_mb / 1024.0:.2f} GB "
                                    f"/ {mem_snap.total_vram_mb / 1024.0:.2f} GB) to prevent PCIe shared-memory paging."
                                )

        cur_timestep = float(getattr(self.model_adapter, "last_timestep", 0.5))
        return StepMetrics(
            step=step_idx,
            loss=round(mean_loss, 6),
            lr=applied_lr,
            step_time_ms=round(elapsed_ms, 2),
            allocated_vram_mb=mem_snap.allocated_mb,
            reserved_vram_mb=mem_snap.reserved_mb,
            grad_scaler_active=self.grad_scaler is not None,
            timestep=round(cur_timestep, 4),
        )

    @staticmethod
    def _iter_checkpointable_block_lists(backbone: nn.Module) -> List[nn.ModuleList]:
        """
        Discover repeatable transformer/UNet `nn.ModuleList` block containers in any ComfyUI backbone
        (including top-level `nn.ModuleList`s like `layers`, `blocks`, `double_blocks`, `single_blocks`,
        `txt_mode_blocks` AND `nn.ModuleList`s inside `nn.ModuleDict` containers like Flux 3's
        `content_mode_blocks = nn.ModuleDict({"image": nn.ModuleList(...), ...})`) without nesting
        checkpointed containers inside other checkpointed containers.
        """
        block_lists: List[nn.ModuleList] = []
        seen_ids: set[int] = set()
        covered_submodules: set[int] = set()

        def _register_module_list(ml: nn.ModuleList) -> None:
            if len(ml) > 0 and id(ml) not in seen_ids and id(ml) not in covered_submodules:
                seen_ids.add(id(ml))
                block_lists.append(ml)
                for sub in ml.modules():
                    covered_submodules.add(id(sub))

        # 1. Check standard named block lists and any top-level ModuleList / ModuleDict-of-ModuleLists
        # (excluding UNet `input_blocks` / `output_blocks` which wrap heterogenous ResNet+SpatialTransformer tuples)
        unet_skip_names = {"input_blocks", "output_blocks", "down_blocks", "up_blocks"}
        for name in (
            "layers",
            "blocks",
            "double_blocks",
            "single_blocks",
            "transformer_blocks",
            "joint_blocks",
            "v_blocks",
            "noise_refiner",
            "context_refiner",
            "txt_mode_blocks",
        ):
            cand = getattr(backbone, name, None)
            if isinstance(cand, nn.ModuleList):
                _register_module_list(cand)
            elif isinstance(cand, nn.ModuleDict):
                for sub_val in cand.values():
                    if isinstance(sub_val, nn.ModuleList):
                        _register_module_list(sub_val)

        for child_name, child_mod in backbone.named_children():
            if child_name in unet_skip_names:
                continue
            if isinstance(child_mod, nn.ModuleList):
                _register_module_list(child_mod)
            elif isinstance(child_mod, nn.ModuleDict):
                for sub_val in child_mod.values():
                    if isinstance(sub_val, nn.ModuleList):
                        _register_module_list(sub_val)

        has_unet_outer_blocks = any(
            hasattr(backbone, unet_name) for unet_name in unet_skip_names
        )
        if not has_unet_outer_blocks:
            # For DiT / Transformer architectures (including nested `self.model.layers` or `self.transformer.blocks`),
            # register any additional non-overlapping ModuleList of >= 2 blocks anywhere in the hierarchy.
            for mod_name, mod in backbone.named_modules():
                if not mod_name or any(part in unet_skip_names for part in mod_name.split(".")):
                    continue
                if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
                    _register_module_list(mod)

        if not block_lists:
            # 2. Prefer leaf transformer_blocks first (e.g. inside UNetModel SpatialTransformers)
            for mod_name, mod in backbone.named_modules():
                if (
                    isinstance(mod, nn.ModuleList)
                    and len(mod) >= 1
                    and mod_name.endswith("transformer_blocks")
                ):
                    _register_module_list(mod)

        if not block_lists:
            # 3. Fallback to top-most ModuleLists, skipping any ModuleList nested inside an already selected one
            for _mod_name, mod in backbone.named_modules():
                if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
                    _register_module_list(mod)
        return block_lists

    @classmethod
    def _patch_inplace_model_helpers(cls, backbone: nn.Module) -> None:
        """
        Wrap module-level helper functions (such as `_gated_residual` and `_modulated_norm` in
        `comfy.ldm.qwen_image21.model`) that perform inference-optimized in-place operations
        (`addcmul_`, `addcdiv_`, `add_`, `mul_`, `sub_`, `div_`, `copy_`) on slice views so that
        during training (`torch.is_grad_enabled()`), input tensors are cloned as non-leaf tensors
        before in-place mutation.
        """
        patched_records: List[Any] = getattr(backbone, "_omni_patched_globals", [])
        seen_globals: set[int] = set()
        target_helper_names = {"_gated_residual", "_modulated_norm", "gated_residual", "modulated_norm"}
        inplace_op_names = {"addcmul_", "addcdiv_", "add_", "mul_", "sub_", "div_", "copy_"}

        def make_safe_helper(orig_helper: Any) -> Any:
            def safe_helper(*h_args: Any, **h_kwargs: Any) -> Any:
                if not torch.is_grad_enabled():
                    return orig_helper(*h_args, **h_kwargs)
                safe_args = _clone_tensors_for_grad(h_args)
                safe_kwargs = _clone_tensors_for_grad(h_kwargs)
                return _clone_tensors_for_grad(orig_helper(*safe_args, **safe_kwargs))

            setattr(safe_helper, "_omni_inplace_safe_wrapped", True)
            setattr(safe_helper, "_omni_orig_helper", orig_helper)
            return safe_helper

        for mod in backbone.modules():
            fwd = getattr(mod, "forward", None)
            g = getattr(fwd, "__globals__", None)
            if not isinstance(g, dict) or id(g) in seen_globals:
                continue
            seen_globals.add(id(g))
            mod_name = getattr(mod.__class__, "__module__", None)

            for fn_name, fn_obj in list(g.items()):
                if not callable(fn_obj) or isinstance(fn_obj, type):
                    continue
                if getattr(fn_obj, "_omni_inplace_safe_wrapped", False):
                    continue
                should_wrap = fn_name in target_helper_names
                if not should_wrap and mod_name and getattr(fn_obj, "__module__", None) == mod_name:
                    code_obj = getattr(fn_obj, "__code__", None)
                    co_names = set(getattr(code_obj, "co_names", ()) or ())
                    if co_names & inplace_op_names:
                        should_wrap = True
                if should_wrap:
                    g[fn_name] = make_safe_helper(fn_obj)
                    patched_records.append((g, fn_name, fn_obj))

        backbone._omni_patched_globals = patched_records  # type: ignore[attr-defined]

    @classmethod
    def _enable_gradient_checkpointing(cls, backbone: nn.Module) -> int:
        """
        Wrap block forward methods with `_checkpoint_block_inplace_safe` to drop peak VRAM by ~65%
        while remaining 100% immune to in-place slice mutations (`CopySlices` `x[:, :n] = ...`)
        used inside Qwen-Image, Lumina, Flux, MOSS-Image, and other ComfyUI transformer blocks.
        """
        active_recompute_cache: List[Optional[Dict[int, torch.Tensor]]] = [None]
        active_ckpt_state: Dict[str, Any] = {
            "saved_inputs_list": [],
            "snapshot_memo": {},
            "offload_inputs_to_cpu": bool(getattr(backbone, "_omni_offload_ckpt_inputs", False)),
        }
        backbone._omni_ckpt_recompute_cache = active_recompute_cache  # type: ignore[attr-defined]
        backbone._omni_ckpt_state = active_ckpt_state  # type: ignore[attr-defined]
        wrapped_count = 0

        for block_list in cls._iter_checkpointable_block_lists(backbone):
            for block in block_list:
                cur_fwd = getattr(block, "forward", None)
                if cur_fwd is not None and not getattr(cur_fwd, "_omni_ckpt_wrapped", False):
                    orig_fwd = getattr(block, "_orig_forward", cur_fwd)
                    block._orig_forward = orig_fwd
                    has_active_dropout = any(
                        isinstance(sub, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout))
                        and float(getattr(sub, "p", 0.0) or 0.0) > 0.0
                        for sub in block.modules()
                    )

                    def make_ckpt_forward(orig_fn: Any, preserve_rng_flag: bool) -> Any:
                        def ckpt_forward(*args: Any, **kwargs: Any) -> Any:
                            return _checkpoint_block_inplace_safe(
                                orig_fn,
                                active_recompute_cache,
                                *args,
                                preserve_rng=preserve_rng_flag,
                                active_ckpt_state=active_ckpt_state,
                                **kwargs,
                            )

                        setattr(ckpt_forward, "_omni_ckpt_wrapped", True)
                        return ckpt_forward

                    block.forward = make_ckpt_forward(orig_fwd, has_active_dropout)
                    block._gradient_checkpointing_enabled = True
                    wrapped_count += 1
                elif getattr(cur_fwd, "_omni_ckpt_wrapped", False):
                    wrapped_count += 1
        return wrapped_count

    def disable_gradient_checkpointing(self) -> None:
        """Restore original block forward methods and module helpers so ComfyUI inference is unencumbered."""
        if self.backbone is None:
            return
        _clear_checkpoint_recompute_state(self.backbone)
        if hasattr(self.backbone, "_omni_ckpt_recompute_cache"):
            delattr(self.backbone, "_omni_ckpt_recompute_cache")
        if hasattr(self.backbone, "_omni_ckpt_state"):
            delattr(self.backbone, "_omni_ckpt_state")
        if hasattr(self.backbone, "_omni_offload_ckpt_inputs"):
            delattr(self.backbone, "_omni_offload_ckpt_inputs")
        patched_records = getattr(self.backbone, "_omni_patched_globals", None)
        if isinstance(patched_records, list):
            for g, fn_name, orig_fn in reversed(patched_records):
                if isinstance(g, dict) and fn_name in g:
                    g[fn_name] = orig_fn
            patched_records.clear()
        if hasattr(self.backbone, "_omni_patched_globals"):
            delattr(self.backbone, "_omni_patched_globals")
        for mod in self.backbone.modules():
            if getattr(mod, "_gradient_checkpointing_enabled", False):
                if hasattr(mod, "_orig_forward"):
                    mod.forward = mod._orig_forward
                    delattr(mod, "_orig_forward")
                mod._gradient_checkpointing_enabled = False

    def save_checkpoint(self, step: int) -> Path:
        """Export only the lightweight LoRA weights to disk in ComfyUI-universal LoRA format."""
        if self.backbone is None:
            raise RuntimeError("Cannot save checkpoint before setup().")
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"{self.config.model_id}_lora_step_{step:05d}.pt"
        lora_state = LoRAInjector.extract_lora_state_dict(self.backbone)
        torch.save(
            {
                "step": step,
                "model_id": self.config.model_id,
                "backend": self.backend.backend_id,
                "dtype": self.backend.dtype_name,
                "rank": self.config.lora.rank,
                "alpha": self.config.lora.alpha,
                "state_dict": lora_state,
            },
            ckpt_path,
        )
        return ckpt_path
