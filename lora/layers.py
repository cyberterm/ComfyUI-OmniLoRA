"""
Pure-PyTorch Low-Rank Adaptation (LoRA) Layers & Dynamic Injector.

Designed with zero vendor-specific custom C++/CUDA kernels so all LoRA forward
and backward passes execute cleanly through native PyTorch ATen / OneDNN XMX BF16
primitives on XPU, CUDA, ROCm, Ascend NPU, Gaudi HPU, MPS, and CPU.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diagnostics import report_exception

_NUMERIC_SEGMENT_RE = re.compile(r"(?:^|\.)\d+(?=\.|$)")


def _strip_numeric_segments(module_path: str) -> str:
    """Convert numbered block paths like `layers.0.blocks.12.qkv` into `layers.blocks.qkv`."""
    parts = [p for p in module_path.split(".") if not p.isdigit()]
    return ".".join(parts)


def _collapse_numeric_segments(module_path: str) -> str:
    """Convert numbered block paths like `input_blocks.4.1.proj_in` into `input_blocks.*.*.proj_in`."""
    parts = ["*" if p.isdigit() else p for p in module_path.split(".") if p]
    return ".".join(parts)


@dataclass
class LoRAConfig:
    """Configuration for LoRA injection."""

    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ()
    lora_dtype: str = "bf16"
    use_rslora: bool = False
    include_conv: bool = False


from ..backends.base import (
    assign_module_parameter,
    is_bitsandbytes_module,
    is_bitsandbytes_tensor,
    is_custom_quantized_tensor,
    is_fp8_dtype,
    is_inference_tensor,
    move_custom_quantized_tensor,
    safe_to_device,
)


def _infer_packed_weight_shape(weight: Any) -> Optional[Tuple[int, int]]:
    """
    Extract logical `(out_features, in_features)` from custom quantized or packed tensors
    such as ComfyUI-GGUF (`weight.tensor_shape`) or bitsandbytes 4-bit (`weight.quant_state.shape`).
    """
    if weight is None:
        return None
    t_shape = getattr(weight, "tensor_shape", None)
    if isinstance(t_shape, (tuple, list, torch.Size)) and len(t_shape) >= 2:
        return int(t_shape[0]), int(t_shape[1])
    q_state = getattr(weight, "quant_state", None)
    if q_state is not None:
        qs_shape = getattr(q_state, "shape", None)
        if isinstance(qs_shape, (tuple, list, torch.Size)) and len(qs_shape) >= 2:
            return int(qs_shape[0]), int(qs_shape[1])
    orig_shape = getattr(weight, "orig_shape", None)
    if isinstance(orig_shape, (tuple, list, torch.Size)) and len(orig_shape) >= 2:
        return int(orig_shape[0]), int(orig_shape[1])
    return None


def _is_linear_like(module: nn.Module) -> bool:
    """Return True for `nn.Linear` and custom ComfyUI (`comfy.ops` / FP8 / GGUF / BitsAndBytes) linear layers."""
    if isinstance(module, LoRALinear):
        return False
    if isinstance(module, nn.Linear):
        return True
    if isinstance(
        module,
        (
            nn.Embedding,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
        ),
    ):
        return False
    in_f = getattr(module, "in_features", None)
    out_f = getattr(module, "out_features", None)
    if isinstance(in_f, int) and isinstance(out_f, int) and in_f > 0 and out_f > 0:
        return True
    weight = getattr(module, "weight", None)
    packed_dims = _infer_packed_weight_shape(weight)
    if packed_dims is not None and packed_dims[0] > 0 and packed_dims[1] > 0:
        return True
    if isinstance(weight, torch.Tensor) and weight.ndim == 2 and weight.shape[0] > 0 and weight.shape[1] > 0:
        return True
    return False


def _is_conv_like(module: nn.Module) -> bool:
    """Return True for `nn.Conv2d`/`Conv1d`/`Conv3d` and ComfyUI `comfy.ops` convolution layers."""
    if isinstance(module, LoRALinear):
        return False
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return True
    in_c = getattr(module, "in_channels", None)
    out_c = getattr(module, "out_channels", None)
    weight = getattr(module, "weight", None)
    if (
        isinstance(in_c, int)
        and isinstance(out_c, int)
        and isinstance(weight, torch.Tensor)
        and weight.ndim in (3, 4, 5)
    ):
        return True
    return False


def _is_pointwise_conv(module: nn.Module) -> bool:
    """
    Return True for pointwise 1x1 convolutions (`kernel_size` all 1s), which act as
    channel-wise linear projections (e.g. `SpatialTransformer.proj_in` / `proj_out` when `use_linear=False`).
    """
    if not _is_conv_like(module):
        return False
    ks = getattr(module, "kernel_size", None)
    if isinstance(ks, int):
        return ks == 1
    if isinstance(ks, (tuple, list)):
        return len(ks) > 0 and all(int(k) == 1 for k in ks)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim in (3, 4, 5):
        return all(int(d) == 1 for d in weight.shape[2:])
    return False


def _extract_layer_dims(module: nn.Module) -> Tuple[int, int, bool]:
    """Return `(in_features, out_features, is_conv)` for any linear-like or conv-like module."""
    if _is_conv_like(module):
        weight = getattr(module, "weight", None)
        in_c = getattr(module, "in_channels", None)
        out_c = getattr(module, "out_channels", None)
        if not isinstance(in_c, int) and isinstance(weight, torch.Tensor) and weight.ndim >= 2:
            in_c = int(weight.shape[1])
        if not isinstance(out_c, int) and isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            out_c = int(weight.shape[0])
        return int(in_c or 1), int(out_c or 1), True

    weight = getattr(module, "weight", None)
    in_f = getattr(module, "in_features", None)
    out_f = getattr(module, "out_features", None)
    packed_dims = _infer_packed_weight_shape(weight)
    if packed_dims is not None:
        if not isinstance(out_f, int):
            out_f = packed_dims[0]
        if not isinstance(in_f, int):
            in_f = packed_dims[1]
    if not isinstance(in_f, int) and isinstance(weight, torch.Tensor) and weight.ndim >= 2:
        in_f = int(weight.shape[1])
    if not isinstance(out_f, int) and isinstance(weight, torch.Tensor) and weight.ndim >= 1:
        out_f = int(weight.shape[0])
    return int(in_f or 1), int(out_f or 1), False


_STREAM_WEIGHT_BUFFERS: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}


def _sync_stream_device(device: torch.device) -> None:
    """Flush pending kernels on `device` before freeing or resizing a shared GPU staging buffer."""
    dev_type = getattr(device, "type", "cpu")
    if dev_type == "cpu":
        return
    dev_mod = getattr(torch, dev_type, None)
    if dev_mod is not None and hasattr(dev_mod, "synchronize"):
        try:
            dev_mod.synchronize(device)
        except TypeError:
            try:
                dev_mod.synchronize()
            except (RuntimeError, AttributeError) as exc:
                report_exception(
                    "lora.sync_stream_device_fallback",
                    "Device synchronize() failed for streamed weight buffer",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="lora",
                )
        except (RuntimeError, AttributeError) as exc:
            report_exception(
                "lora.sync_stream_device",
                "Device synchronize(device) failed for streamed weight buffer",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="lora",
            )


def prepare_streamed_weight_buffer(
    target_device: torch.device,
    compute_dtype: torch.dtype,
    min_numel: int,
) -> None:
    """
    Pre-allocate the reusable GPU staging buffer to `min_numel` elements before training starts
    so `_materialize_streamed_weight` never reallocates or drops an in-flight buffer mid-step.
    """
    if target_device.type == "cpu" or min_numel <= 0:
        return
    if compute_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        compute_dtype = torch.bfloat16
    key = (str(target_device), compute_dtype)
    buf = _STREAM_WEIGHT_BUFFERS.get(key)
    if buf is None or buf.device != target_device or buf.numel() < min_numel or is_inference_tensor(buf):
        if buf is not None:
            _sync_stream_device(target_device)
        with torch.inference_mode(False):
            _STREAM_WEIGHT_BUFFERS[key] = torch.empty(
                (int(min_numel),),
                device=target_device,
                dtype=compute_dtype,
            )


def release_streamed_weight_buffers() -> None:
    """Release any reusable GPU staging buffers allocated for CPU-streamed or quantized frozen weights."""
    for (_, _), buf in list(_STREAM_WEIGHT_BUFFERS.items()):
        if isinstance(buf, torch.Tensor) and buf.device.type != "cpu":
            _sync_stream_device(buf.device)
            break
    _STREAM_WEIGHT_BUFFERS.clear()


def _resolve_layer_weight_scale(layer: nn.Module) -> Optional[torch.Tensor]:
    """
    Resolve an optional per-tensor or per-channel FP8/INT8 dequantization scale attached to `layer`
    (supports OmniLoRA's `_omni_weight_scale` as well as ComfyUI's `scale_weight` and `weight_scale`).
    """
    for attr in ("_omni_weight_scale", "scale_weight", "weight_scale"):
        val = getattr(layer, attr) if hasattr(layer, attr) else None
        if isinstance(val, torch.Tensor):
            return val
    return None


def _materialize_streamed_weight(
    weight: torch.Tensor,
    target_device: torch.device,
    compute_dtype: torch.dtype,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Copy an offloaded CPU, FP8 (`float8_e4m3fn`/`float8_e5m2`), or INT8 (`int8_qlora`) frozen
    weight tensor into a reusable per-(device, dtype) GPU staging buffer and apply optional
    per-tensor or per-channel `weight_scale` dequantization in-place.
    Guarantees zero dynamic GPU weight allocations across forward/checkpoint/backward passes.
    """
    if (
        weight.device == target_device
        and weight.dtype == compute_dtype
        and weight_scale is None
        and not is_inference_tensor(weight)
    ):
        return weight
    numel = weight.numel()
    key = (str(target_device), compute_dtype)
    buf = _STREAM_WEIGHT_BUFFERS.get(key)
    if buf is None or buf.device != target_device or buf.numel() < numel or is_inference_tensor(buf):
        if buf is not None:
            _sync_stream_device(target_device)
        with torch.inference_mode(False):
            buf = torch.empty((numel,), device=target_device, dtype=compute_dtype)
        _STREAM_WEIGHT_BUFFERS[key] = buf
    view = buf[:numel].view(weight.shape)
    w_src = weight.detach()
    use_async_dma = (
        target_device.type in ("cuda", "npu")
        and w_src.device.type == "cpu"
        and w_src.is_pinned()
    )
    if w_src.device.type == "cpu" and target_device.type != "cpu" and w_src.dtype != compute_dtype:
        # Never upcast on the CPU before copying over PCIe: CPU `.to(dtype=compute_dtype)` runs
        # single-threaded on host RAM, strips `is_pinned()`, and doubles PCIe traffic for 1-byte
        # FP8/INT8 weights. Instead, DMA the compact source tensor to a matching-dtype GPU staging
        # buffer first (`q_view.dtype == w_src.dtype` so `cudaMemcpyAsync` runs at hardware speed)
        # and cast to `compute_dtype` on the GPU.
        q_key = (str(target_device), w_src.dtype)
        q_buf = _STREAM_WEIGHT_BUFFERS.get(q_key)
        try:
            if q_buf is None or q_buf.device != target_device or q_buf.numel() < numel or is_inference_tensor(q_buf):
                with torch.inference_mode(False):
                    q_buf = torch.empty((numel,), device=target_device, dtype=w_src.dtype)
                _STREAM_WEIGHT_BUFFERS[q_key] = q_buf
            q_view = q_buf[:numel].view(weight.shape)
            q_view.copy_(w_src, non_blocking=use_async_dma)
            view.copy_(q_view, non_blocking=use_async_dma)
        except RuntimeError:
            # Fallback if the target backend cannot allocate `w_src.dtype` (e.g. FP8 on older drivers)
            view.copy_(w_src.to(dtype=compute_dtype), non_blocking=False)
    else:
        view.copy_(w_src, non_blocking=use_async_dma)
    if isinstance(weight_scale, torch.Tensor):
        s_dev = (
            weight_scale.detach()
            if (weight_scale.device == target_device and weight_scale.dtype == compute_dtype)
            else safe_to_device(weight_scale.detach(), device=target_device, dtype=compute_dtype)
        )
        if s_dev.ndim == 1 and view.ndim >= 2 and s_dev.numel() == view.shape[0]:
            s_dev = s_dev.view(-1, *([1] * (view.ndim - 1)))
        view.mul_(s_dev)
    return view


def _resolve_linear_compute_dtype(
    x: torch.Tensor,
    weight: Any,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.dtype:
    """
    Resolve the execution dtype for frozen linear layers (`_StreamedLinearFn` / `_QuantizedCallbackLinearFn`):
    - Prioritizes active `torch.amp.autocast` 16-bit dtype (`bf16`/`fp16`) or `x.dtype` when `x` is
      already 16-bit (`bf16`/`fp16`), so an `fp16` checkpoint (`weights=fp16`) trained under `compute=bf16`
      (`GradScaler=False`) executes forward and backward in `bf16` via the reusable staging buffer instead
      of downcasting `bf16` activations, `lora_delta`, and unscaled backward gradients to `fp16`.
    - When `x` is `float32` (e.g. FP32 AdaLN modulation `x = norm(x) * (1 + scale) + shift` in `moss_image`,
      `flux`, `wan`, `qwen_image`, `sd3_5`), downcasts to active 16-bit autocast or 16-bit `weight` dtype
      so 32-bit modulation does not upcast subsequent transformer blocks.
    - Preserves `float32` when both `x` and `weight` are `float32` and autocast is disabled.
    """
    dev_type = x.device.type
    default_dt = torch.bfloat16
    w_dtype = getattr(weight, "dtype", None) if isinstance(weight, torch.Tensor) else None
    orig_dtype = getattr(weight, "_omni_orig_dtype", None)

    compute_dtype: Optional[torch.dtype] = None
    ac_dev = dev_type if dev_type != "cpu" else "cpu"
    try:
        if torch.is_autocast_enabled(ac_dev):
            ac_dt = torch.get_autocast_dtype(ac_dev)
            if ac_dt in (torch.bfloat16, torch.float16):
                compute_dtype = ac_dt
    except TypeError:
        try:
            if torch.is_autocast_enabled():
                ac_dt = torch.get_autocast_gpu_dtype()
                if ac_dt in (torch.bfloat16, torch.float16):
                    compute_dtype = ac_dt
        except (RuntimeError, AttributeError) as exc:
            report_exception(
                "lora.linear_autocast_legacy_probe",
                "Legacy autocast dtype probe failed in _resolve_linear_compute_dtype",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="lora",
            )
    except (RuntimeError, AttributeError) as exc:
        report_exception(
            "lora.linear_autocast_probe",
            "Autocast dtype probe failed in _resolve_linear_compute_dtype",
            exc,
            level=logging.DEBUG,
            once=True,
            logger_name="lora",
        )

    if compute_dtype is None:
        if x.dtype in (torch.bfloat16, torch.float16):
            compute_dtype = x.dtype
        elif w_dtype in (torch.bfloat16, torch.float16) and weight_scale is None:
            compute_dtype = w_dtype
        elif orig_dtype in (torch.bfloat16, torch.float16):
            compute_dtype = orig_dtype
        elif w_dtype in (torch.bfloat16, torch.float16, torch.float32) and weight_scale is None:
            compute_dtype = w_dtype
        elif x.dtype in (torch.bfloat16, torch.float16, torch.float32):
            compute_dtype = x.dtype
        else:
            compute_dtype = default_dt

    return compute_dtype


class _StreamedLinearFn(torch.autograd.Function):
    """
    Memory-efficient Linear autograd function for frozen base weights (`requires_grad=False`).
    - Never saves a duplicate GPU copy of `weight` in C++ `SavedVariable` (preventing both
      `Inference tensors cannot be saved for backward` and VRAM accumulation across layers).
    - If `weight` is offloaded to CPU, stored in FP8 (`float8_e4m3fn`/`float8_e5m2`), or quantized
      to INT8 (`int8_qlora`), streams and dequantizes `weight` in-place inside a reusable per-device
      staging buffer so zero dynamic weight allocations fragment the caching allocator across steps.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        weight_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx.weight_ref = weight
        ctx.weight_scale_ref = weight_scale
        ctx.x_dtype = x.dtype
        compute_dtype = _resolve_linear_compute_dtype(x, weight, weight_scale=weight_scale)
        ctx.compute_dtype = compute_dtype
        x_in = x if x.dtype == compute_dtype else x.to(dtype=compute_dtype)
        w_dev = _materialize_streamed_weight(weight, x.device, compute_dtype, weight_scale=weight_scale)
        b_dev: Optional[torch.Tensor] = None
        if isinstance(bias, torch.Tensor):
            b_dev = (
                bias
                if (bias.device == x.device and bias.dtype == compute_dtype and not is_inference_tensor(bias))
                else safe_to_device(bias.detach(), device=x.device, dtype=compute_dtype).clone()
            )
        out = F.linear(x_in, w_dev, b_dev)
        return out if out.dtype == compute_dtype else out.to(dtype=compute_dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        weight: torch.Tensor = ctx.weight_ref
        weight_scale: Optional[torch.Tensor] = ctx.weight_scale_ref
        compute_dtype: torch.dtype = getattr(ctx, "compute_dtype", None) or _resolve_linear_compute_dtype(
            grad_output, weight, weight_scale=weight_scale
        )
        g_out = grad_output if grad_output.dtype == compute_dtype else grad_output.to(dtype=compute_dtype)
        w_dev = _materialize_streamed_weight(weight, g_out.device, compute_dtype, weight_scale=weight_scale)
        grad_x = torch.matmul(g_out, w_dev)
        target_grad_dtype = getattr(ctx, "x_dtype", grad_output.dtype)
        return (grad_x if grad_x.dtype == target_grad_dtype else grad_x.to(dtype=target_grad_dtype)), None, None, None


def _dequantize_callback_tensor(
    base_layer: nn.Module,
    tensor: Any,
    compute_dtype: torch.dtype,
    target_device: torch.device,
) -> torch.Tensor:
    """
    Dequantize a block-quantized tensor (such as ComfyUI-GGUF `GGMLTensor`) on `target_device`:
    1. If `tensor` is on CPU (when a layer is offloaded), moves the compact quantized `tensor`
       to `target_device` FIRST so dequantization runs on the GPU instead of the CPU.
    2. Calls `base_layer.get_weight(w_src, compute_dtype)` directly (avoiding per-layer
       `TypeError` exceptions on `ComfyUI-GGUF` which takes `(tensor, dtype)` without `device=`).
    """
    w_src = (
        move_custom_quantized_tensor(tensor, target_device)
        if (isinstance(tensor, torch.Tensor) and tensor.device != target_device)
        else tensor
    )
    if callable(getattr(base_layer, "get_weight", None)):
        try:
            w_deq = base_layer.get_weight(w_src, compute_dtype)  # type: ignore[attr-defined]
        except TypeError:
            w_deq = base_layer.get_weight(w_src, compute_dtype, device=target_device)  # type: ignore[attr-defined]
    elif callable(getattr(w_src, "dequantize", None)):
        try:
            w_deq = w_src.dequantize(dtype=compute_dtype, device=target_device)
        except TypeError:
            w_deq = w_src.dequantize()
    elif hasattr(w_src, "_qdata") and isinstance(getattr(w_src, "_qdata", None), torch.Tensor):
        qdata = getattr(w_src, "_qdata").to(device=target_device, dtype=compute_dtype)
        qparams = getattr(w_src, "_params", None)
        qscale = getattr(qparams, "scale", None) if qparams is not None else getattr(w_src, "scale", None)
        w_deq = qdata * qscale.to(device=target_device, dtype=compute_dtype) if isinstance(qscale, torch.Tensor) else qdata
    else:
        w_deq = w_src.to(device=target_device, dtype=compute_dtype)
    if isinstance(w_deq, torch.Tensor) and (w_deq.device != target_device or w_deq.dtype != compute_dtype):
        w_deq = w_deq.to(device=target_device, dtype=compute_dtype)
    return w_deq


class _QuantizedCallbackLinearFn(torch.autograd.Function):
    """
    Autograd function for external block-quantized layers (such as ComfyUI-GGUF `GGMLOps.Linear`)
    that expose a `base_layer.get_weight(weight, dtype)` dequantization method.
    Dequantizes on-the-fly on the active accelerator during forward and backward without storing
    the dequantized weight in `SavedVariable`, enabling fast QLoRA-style training on GGUF
    (Q4_K_M / Q5_K / Q8_0) models.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        base_layer: nn.Module,
        weight: Any,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx.base_layer_ref = base_layer
        ctx.weight_ref = weight
        ctx.x_dtype = x.dtype
        compute_dtype = _resolve_linear_compute_dtype(x, weight)
        ctx.compute_dtype = compute_dtype
        with torch.no_grad():
            w_deq = _dequantize_callback_tensor(base_layer, weight, compute_dtype, x.device)
            b_dev: Optional[torch.Tensor] = None
            if isinstance(bias, torch.Tensor):
                if is_custom_quantized_tensor(bias) and callable(getattr(base_layer, "get_weight", None)):
                    b_dev = _dequantize_callback_tensor(base_layer, bias, compute_dtype, x.device)
                else:
                    b_dev = (
                        bias
                        if (bias.device == x.device and bias.dtype == compute_dtype)
                        else bias.detach().to(device=x.device, dtype=compute_dtype)
                    )
        x_in = x if x.dtype == compute_dtype else x.to(dtype=compute_dtype)
        out = F.linear(x_in, w_deq, b_dev)
        return out if out.dtype == compute_dtype else out.to(dtype=compute_dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        base_layer: nn.Module = ctx.base_layer_ref
        weight: Any = ctx.weight_ref
        compute_dtype: torch.dtype = getattr(ctx, "compute_dtype", None) or _resolve_linear_compute_dtype(
            grad_output, weight
        )
        with torch.no_grad():
            w_deq = _dequantize_callback_tensor(base_layer, weight, compute_dtype, grad_output.device)
        g_out = grad_output if grad_output.dtype == compute_dtype else grad_output.to(dtype=compute_dtype)
        grad_x = torch.matmul(g_out, w_deq)
        target_grad_dtype = getattr(ctx, "x_dtype", grad_output.dtype)
        return (grad_x if grad_x.dtype == target_grad_dtype else grad_x.to(dtype=target_grad_dtype)), None, None, None


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper around an existing frozen `nn.Linear`, ComfyUI `comfy.ops`
    Linear (including FP8, INT8 QLoRA, GGUF, and BitsAndBytes), or `nn.Conv2d`/`Conv1d`/`Conv3d` module.
    Computes: y = W_0(x) + (alpha / r_scale) * B(A(dropout(x)))
    """

    comfy_force_forward: bool = True

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        lora_dtype: torch.dtype = torch.bfloat16,
        use_rslora: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        # Unwrap if base_layer was already wrapped in a previous ComfyUI workflow execution
        while isinstance(base_layer, LoRALinear):
            base_layer = base_layer.base_layer

        self.base_layer = base_layer
        self.in_features, self.out_features, self.is_conv = _extract_layer_dims(base_layer)
        self.rank = rank
        self.alpha = float(alpha)
        self.use_rslora = use_rslora
        denom = math.sqrt(rank) if use_rslora else float(rank)
        self.scaling = self.alpha / denom

        # Freeze all base weights and biases, and temporarily stash/clear any residual GGUF `.patches`
        for param in self.base_layer.parameters():
            param.requires_grad_(False)
            p_patches = getattr(param, "patches", None)
            if isinstance(p_patches, list) and p_patches:
                if not hasattr(param, "_omni_saved_patches"):
                    setattr(param, "_omni_saved_patches", list(p_patches))
                param.patches = []

        resolved_device = device
        if resolved_device is None:
            weight = getattr(base_layer, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
                resolved_device = weight.device
            else:
                resolved_device = torch.device("cpu")
        elif resolved_device.type == "meta":
            resolved_device = torch.device("cpu")

        if lora_dtype == torch.float64 or is_fp8_dtype(lora_dtype):
            lora_dtype = torch.bfloat16

        if self.is_conv:
            weight = getattr(base_layer, "weight", None)
            w_ndim = weight.ndim if isinstance(weight, torch.Tensor) else 4
            if isinstance(base_layer, nn.Conv1d) or w_ndim == 3:
                conv_cls: Any = nn.Conv1d
                default_k: Tuple[int, ...] = (1,)
            elif isinstance(base_layer, nn.Conv3d) or w_ndim == 5:
                conv_cls = nn.Conv3d
                default_k = (1, 1, 1)
            else:
                conv_cls = nn.Conv2d
                default_k = (1, 1)

            kernel_size = getattr(base_layer, "kernel_size", None)
            if kernel_size is None:
                if isinstance(weight, torch.Tensor) and weight.ndim >= 3:
                    kernel_size = tuple(int(d) for d in weight.shape[2:])
                else:
                    kernel_size = default_k
            stride = getattr(base_layer, "stride", 1)
            padding = getattr(base_layer, "padding", 0)
            dilation = getattr(base_layer, "dilation", 1)
            raw_groups = int(getattr(base_layer, "groups", 1) or 1)
            eff_groups = (
                raw_groups
                if raw_groups > 1 and self.in_features % raw_groups == 0 and rank % raw_groups == 0
                else 1
            )
            self.lora_A = conv_cls(
                self.in_features,
                rank,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=eff_groups,
                bias=False,
                device=resolved_device,
                dtype=lora_dtype,
            )
            self.lora_B = conv_cls(
                rank,
                self.out_features,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                device=resolved_device,
                dtype=lora_dtype,
            )
        else:
            self.lora_A = nn.Linear(
                self.in_features,
                rank,
                bias=False,
                device=resolved_device,
                dtype=lora_dtype,
            )
            self.lora_B = nn.Linear(
                rank,
                self.out_features,
                bias=False,
                device=resolved_device,
                dtype=lora_dtype,
            )
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.reset_lora_parameters()
        self.lora_A.weight.requires_grad_(True)
        self.lora_B.weight.requires_grad_(True)

    def reset_lora_parameters(self) -> None:
        """Initialize A with Kaiming uniform and B with zeros so initial delta W = 0."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def _run_base_layer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute frozen `base_layer(x)` with zero weight cloning:
        - For ComfyUI-GGUF layers exposing `get_weight(weight, dtype, device)`, uses
          `_QuantizedCallbackLinearFn` so block-quantized weights dequantize on the fly
          without holding dequantized copies in `SavedVariable`.
        - For bitsandbytes layers (`Linear4bit`, `Linear8bitLt`, `Params4bit`, `Int8Params`),
          explicitly rejects non-CUDA execution (`xpu`, `mps`, `npu`, `hpu`, `cpu`) and
          wraps CUDA execution so any missing/broken `libbitsandbytes` binary raises a clear,
          actionable error instead of crashing.
        - For standard/FP8/INT8 Linear layers with a 2D `weight` tensor, uses `_StreamedLinearFn`
          with optional `weight_scale` dequantization inside the reusable GPU staging buffer.
        """
        weight = getattr(self.base_layer, "weight", None)
        bias = getattr(self.base_layer, "bias", None)

        if is_bitsandbytes_module(self.base_layer) or is_bitsandbytes_tensor(weight):
            if x.device.type != "cuda":
                raise ValueError(
                    f"[OmniLoRA] Unsupported bitsandbytes quantized layer ({type(self.base_layer).__name__}) "
                    f"on device '{x.device}'. bitsandbytes (4-bit NF4 / FP4 / LLM.int8) requires CUDA C++ "
                    f"kernels and is not supported on {x.device.type.upper()}. Load a standard BF16/FP16, FP8, "
                    f"or GGUF model and use OmniLoRA's pure-PyTorch 'precision=fp8_e4m3fn' or 'precision=int8_qlora'."
                )
            try:
                return self.base_layer(x)
            except Exception as exc:
                raise ValueError(
                    f"[OmniLoRA] bitsandbytes layer ({type(self.base_layer).__name__}) failed on device '{x.device}': "
                    f"{exc}. Load a standard BF16/FP16, FP8, or GGUF model and use OmniLoRA's pure-PyTorch "
                    f"'precision=fp8_e4m3fn' or 'precision=int8_qlora'."
                ) from exc

        if not self.is_conv and isinstance(weight, torch.Tensor):
            if (
                callable(getattr(self.base_layer, "get_weight", None))
                or callable(getattr(weight, "dequantize", None))
                or hasattr(weight, "_qdata")
            ) and (is_custom_quantized_tensor(weight) or weight.ndim != 2):
                return _QuantizedCallbackLinearFn.apply(
                    x,
                    self.base_layer,
                    weight,
                    bias if isinstance(bias, torch.Tensor) else None,
                )
            if is_custom_quantized_tensor(weight) or weight.dtype == torch.uint8:
                return self.base_layer(x)

        weight_scale = _resolve_layer_weight_scale(self.base_layer)
        if (
            not self.is_conv
            and isinstance(weight, torch.Tensor)
            and weight.ndim == 2
            and not getattr(self.base_layer, "_forward_pre_hooks", None)
        ):
            return _StreamedLinearFn.apply(
                x,
                weight,
                bias if isinstance(bias, torch.Tensor) else None,
                weight_scale,
            )

        try:
            return self.base_layer(x)
        except AttributeError:
            if isinstance(self.base_layer, nn.Linear):
                return nn.Linear.forward(self.base_layer, x)
            raise
        except RuntimeError:
            if isinstance(weight, torch.Tensor):
                if not self.is_conv and weight.ndim == 2:
                    return _StreamedLinearFn.apply(
                        x,
                        weight,
                        bias if isinstance(bias, torch.Tensor) else None,
                        weight_scale,
                    )
                compute_dtype = (
                    weight.dtype
                    if (weight.device == x.device and weight.dtype in (torch.bfloat16, torch.float16, torch.float32))
                    else (x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16)
                )
                x_in = x if x.dtype == compute_dtype else x.to(dtype=compute_dtype)
                w_dev = _materialize_streamed_weight(weight, x.device, compute_dtype, weight_scale=weight_scale)
                b_dev = (
                    bias.detach().to(device=x.device, dtype=compute_dtype, non_blocking=False)
                    if isinstance(bias, torch.Tensor)
                    else None
                )
                if self.is_conv and w_dev.ndim in (3, 4, 5):
                    stride = getattr(self.base_layer, "stride", 1)
                    padding = getattr(self.base_layer, "padding", 0)
                    dilation = getattr(self.base_layer, "dilation", 1)
                    groups = getattr(self.base_layer, "groups", 1)
                    if w_dev.ndim == 3:
                        out = F.conv1d(x_in, w_dev, b_dev, stride=stride, padding=padding, dilation=dilation, groups=groups)
                    elif w_dev.ndim == 4:
                        out = F.conv2d(x_in, w_dev, b_dev, stride=stride, padding=padding, dilation=dilation, groups=groups)
                    else:
                        out = F.conv3d(x_in, w_dev, b_dev, stride=stride, padding=padding, dilation=dilation, groups=groups)
                    return out if out.dtype == x.dtype else out.to(dtype=x.dtype)
            raise

    def _project_lora(self, lora_in: torch.Tensor) -> torch.Tensor:
        """
        Execute `lora_B(lora_A(lora_in)) * scaling` in the exact dtype of `lora_A.weight`
        (`bf16`, `fp32`, or `fp16`). When `lora_A.weight.dtype == torch.float32`, disables
        outer AMP autocast for the low-rank projection so `fp32` master LoRA weights are not
        downcast back to `bf16`/`fp16` during forward/backward.
        """
        target_dt = self.lora_A.weight.dtype
        if lora_in.dtype != target_dt:
            lora_in = lora_in.to(dtype=target_dt)
        dev_type = lora_in.device.type
        if torch.is_grad_enabled() and getattr(lora_in, "_base", None) is not None:
            lora_in = lora_in.clone()
        if target_dt == torch.float32 and dev_type in ("cuda", "xpu", "cpu", "mps", "npu", "hpu"):
            try:
                with torch.amp.autocast(device_type=dev_type, enabled=False):
                    return self.lora_B(self.lora_A(lora_in)) * self.scaling
            except (RuntimeError, TypeError, AttributeError) as exc:
                report_exception(
                    "lora.fp32_branch_autocast_disable",
                    "Could not disable autocast for FP32 LoRA branch; running directly",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="lora",
                )
        return self.lora_B(self.lora_A(lora_in)) * self.scaling

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) and hasattr(x, "feats") and callable(getattr(x, "replace", None)):
            return x.replace(self.forward(x.feats, *args, **kwargs))  # type: ignore[return-value]

        if args or kwargs:
            try:
                base_out = self.base_layer(x, *args, **kwargs)
            except AttributeError:
                base_out = self._run_base_layer(x)
        else:
            base_out = self._run_base_layer(x)

        if self.lora_A.weight.device != x.device or self.lora_B.weight.device != x.device:
            raise RuntimeError(
                f"[OmniLoRA] LoRALinear weights on device '{self.lora_A.weight.device}' / "
                f"'{self.lora_B.weight.device}' do not match input tensor device '{x.device}'. "
                "Refusing to re-allocate `lora_A`/`lora_B` inside forward() because doing so "
                "severs optimizer parameter references."
            )

        if self.is_conv and x.ndim >= 3:
            if isinstance(self.lora_A, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and x.ndim == self.lora_A.weight.ndim:
                lora_in = self.lora_dropout(x)
                lora_delta = self._project_lora(lora_in)
            else:
                x_perm = x.movedim(1, -1)
                lora_in = self.lora_dropout(x_perm)
                if isinstance(self.lora_A, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    target_dt = self.lora_A.weight.dtype
                    li = lora_in if lora_in.dtype == target_dt else lora_in.to(dtype=target_dt)
                    wa = self.lora_A.weight.reshape(self.rank, -1)
                    wb = self.lora_B.weight.reshape(self.out_features, self.rank)
                    lora_delta = (F.linear(F.linear(li, wa), wb) * self.scaling).movedim(-1, 1)
                else:
                    lora_delta = self._project_lora(lora_in).movedim(-1, 1)
            if lora_delta.shape[2:] != base_out.shape[2:] and base_out.ndim in (4, 5):
                try:
                    lora_delta = F.interpolate(lora_delta, size=base_out.shape[2:], mode="nearest")
                except RuntimeError:
                    lora_delta = F.interpolate(
                        lora_delta.to(dtype=torch.float32),
                        size=base_out.shape[2:],
                        mode="nearest",
                    ).to(dtype=base_out.dtype)
            return base_out + (lora_delta if lora_delta.dtype == base_out.dtype else lora_delta.to(dtype=base_out.dtype))

        lora_in = self.lora_dropout(x)
        lora_delta = self._project_lora(lora_in)
        return base_out + (lora_delta if lora_delta.dtype == base_out.dtype else lora_delta.to(dtype=base_out.dtype))

    @property
    def trainable_parameter_count(self) -> int:
        return self.lora_A.weight.numel() + self.lora_B.weight.numel()

    @property
    def weight(self) -> torch.Tensor:
        return getattr(self.base_layer, "weight", None)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.base_layer, "bias", None)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "base_layer" in self.__dict__:
                return getattr(self.base_layer, name)
            raise


_UNET_BLOCK_CONTAINER_NAMES = frozenset(
    {
        "input_blocks",
        "middle_block",
        "output_blocks",
        "down_blocks",
        "mid_block",
        "up_blocks",
    }
)


def _unwrap_lora_base(module: nn.Module) -> nn.Module:
    base = module
    while isinstance(base, LoRALinear):
        base = base.base_layer
    return base


def _iter_named_modules_skipping_lora(
    root: nn.Module,
    prefix: str = "",
    *,
    include_self: bool = False,
    _memo: Optional[set[int]] = None,
) -> Iterable[Tuple[str, nn.Module]]:
    """
    Walk `root` like `nn.Module.named_modules()` (deduplicating shared aliases),
    stopping recursion at `LoRALinear` wrappers so `.base_layer`, `.lora_A`, and `.lora_B`
    never pollute structural role classification or overwrite `module_to_path`.
    """
    if _memo is None:
        _memo = set()
    if id(root) in _memo:
        return
    _memo.add(id(root))
    if include_self:
        yield (prefix, root)
    if isinstance(root, LoRALinear):
        return
    for attr_name, child in root.named_children():
        if child is None:
            continue
        full_name = f"{prefix}.{attr_name}" if prefix else attr_name
        if id(child) in _memo:
            continue
        if isinstance(child, LoRALinear):
            _memo.add(id(child))
            yield (full_name, child)
        else:
            yield from _iter_named_modules_skipping_lora(
                child,
                full_name,
                include_self=True,
                _memo=_memo,
            )


def _collect_linear_like_leaves(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    leaves: List[Tuple[str, nn.Module]] = []
    for sub_name, sub in _iter_named_modules_skipping_lora(module, include_self=True):
        if isinstance(sub, LoRALinear):
            base = _unwrap_lora_base(sub)
            if _is_linear_like(base):
                leaves.append((sub_name, base))
        elif _is_linear_like(sub):
            leaves.append((sub_name, sub))
    return leaves


def _count_linear_like_descendants(module: nn.Module) -> int:
    return len(_collect_linear_like_leaves(module))


def _is_plain_sequential_mlp(module: nn.Module) -> bool:
    """
    Return True when `module` is an `nn.Sequential` whose children are exclusively leaf
    Linearprojections and parameterless activations (e.g., ACE-Step SSL `projectors`),
    distinguishing MLP head lists from UNet `TimestepEmbedSequential` or Transformer blocks.
    """
    if not isinstance(module, nn.Sequential):
        return False
    children = list(module.children())
    if not children:
        return True
    for child in children:
        base_c = _unwrap_lora_base(child)
        if _is_linear_like(base_c):
            continue
        if _count_linear_like_descendants(base_c) > 0:
            return False
    return True


def _has_norm_or_activation_child(module: nn.Module) -> bool:
    for _, sub in _iter_named_modules_skipping_lora(module, include_self=True):
        base_s = _unwrap_lora_base(sub)
        if _is_linear_like(base_s):
            continue
        cls_l = type(base_s).__name__.lower()
        if (
            hasattr(base_s, "eps")
            or hasattr(base_s, "normalized_shape")
            or any(tok in cls_l for tok in ("silu", "gelu", "relu", "swish", "mish", "norm"))
        ):
            return True
    return False


def _has_attention_attributes(module: nn.Module) -> bool:
    for _, sub in _iter_named_modules_skipping_lora(module, include_self=True):
        base_s = _unwrap_lora_base(sub)
        if any(
            hasattr(base_s, attr)
            for attr in ("num_heads", "heads", "head_dim", "n_heads", "num_key_value_heads", "comfy_attention")
        ):
            return True
    return False


def _is_composite_transformer_block(module: nn.Module) -> bool:
    """
    Return True if `module` is a composite Transformer / MMDiT / SingleStreamDiT / Refiner /
    Perceiver block (containing attention head attributes or >= 4 linear-like projections, and
    not a plain `nn.Sequential` MLP, 2-linear timestep `MLPEmbedder`, or rotary frequency generator).
    """
    base_m = _unwrap_lora_base(module)
    if _is_linear_like(base_m) or _is_plain_sequential_mlp(module):
        return False
    leaves = _collect_linear_like_leaves(module)
    n_linears = len(leaves)
    if n_linears < 2:
        return False
    has_multi_linear_child = any(
        not _is_linear_like(_unwrap_lora_base(c)) and _count_linear_like_descendants(c) >= 2
        for _, c in module.named_children()
    )
    if not has_multi_linear_child:
        dims = [_extract_layer_dims(lin) for _, lin in leaves]
        out_set = {d[1] for d in dims if d[1] > 0}
        if dims and dims[0][0] > 0 and dims[0][0] not in out_set:
            return False
    return _has_attention_attributes(module) or n_linears >= 4


def _collect_repeating_block_module_ids(model: nn.Module) -> set[int]:
    """
    Structurally identify all submodules that reside inside repeating Transformer / MMDiT /
    SingleStreamDiT / Refiner / Perceiver `nn.ModuleList` block containers or UNet block sections
    (`input_blocks`, `middle_block`, `output_blocks`, `down_blocks`, `mid_block`, `up_blocks`).

    Any `nn.ModuleList` at backbone depth <= 2 whose elements are composite transformer blocks
    (`_is_composite_transformer_block`) is recognized as a repeating block container,
    structurally excluding top-level patchifiers, 2-linear timestep/guidance `MLPEmbedder` stacks
    (e.g. Chroma `Approximator`), plain `nn.Sequential` MLP heads (e.g. ACE-Step `projectors`),
    deeply nested stem token refiners (depth >= 3 when top-level blocks exist), and output heads.
    """
    has_generative_stream = any(
        hasattr(model, attr)
        for attr in ("double_stream_blocks", "single_stream_blocks", "double_blocks", "transformer_blocks")
    )
    has_decoder_backbone = hasattr(model, "decoder")
    has_top_level_block_list = any(
        isinstance(child, nn.ModuleList)
        and len(child) >= 1
        and any(_is_composite_transformer_block(el) for el in child)
        for _, child in model.named_children()
    )
    valid_ids: set[int] = set()
    for name, sub in _iter_named_modules_skipping_lora(model):
        if not name:
            continue
        stripped_lower = _strip_numeric_segments(name.replace("__", ".")).lower()
        parts_list = [p for p in stripped_lower.split(".") if p]
        parts = set(parts_list)
        if has_generative_stream and (stripped_lower == "visual" or stripped_lower.startswith("visual.")):
            continue
        if has_decoder_backbone and (
            stripped_lower in ("tokenizer", "detokenizer")
            or stripped_lower.startswith(("tokenizer.", "detokenizer."))
        ):
            continue
        is_block_container = False
        if isinstance(sub, nn.ModuleList) and len(sub) >= 1:
            if has_top_level_block_list and len(parts_list) >= 3:
                continue
            if any(
                _is_composite_transformer_block(child)
                or (isinstance(child, nn.ModuleList) and any(_is_composite_transformer_block(gc) for gc in child))
                for child in sub
            ):
                is_block_container = True
        elif (parts_list and parts_list[-1] in _UNET_BLOCK_CONTAINER_NAMES) and _count_linear_like_descendants(sub) >= 1:
            is_block_container = True

        if is_block_container:
            for _, desc in _iter_named_modules_skipping_lora(sub, include_self=True):
                valid_ids.add(id(desc))
                valid_ids.add(id(_unwrap_lora_base(desc)))
    return valid_ids


def _iter_effective_transformer_blocks(block: nn.Module) -> Iterable[nn.Module]:
    """
    Unwrap composite stream/sequential wrappers inside a repeating block list (such as SD3's
    `JointBlock` wrapping `context_block` + `x_block`, or UNet's `TimestepEmbedSequential` /
    StableCascade's nested `nn.ModuleList` wrapping `ResBlock` + `TimestepBlock` + `AttnBlock`)
    so structural sub-branch analysis always runs at the granularity of an individual stream/transformer block.
    """
    children = [c for _, c in block.named_children() if _count_linear_like_descendants(c) > 0]
    if isinstance(block, (nn.Sequential, nn.ModuleList)):
        for child in children:
            yield from _iter_effective_transformer_blocks(child)
        return
    # Check if every linear-bearing child is itself a multi-branch sub-block (e.g. SD3 JointBlock)
    if children and all(
        not (_is_linear_like(_unwrap_lora_base(c)))
        and sum(1 for _, gc in c.named_children() if _count_linear_like_descendants(gc) > 0) >= 2
        and not any(isinstance(gc, nn.ModuleList) for _, gc in c.named_children())
        and not _has_attention_attributes(c)
        for c in children
    ):
        for child in children:
            yield child
        return
    yield block


def classify_block_structural_roles(model: nn.Module) -> Dict[str, str]:
    """
    Classify every linear-like module in `model` by its structural role in the residual stream
    topology without relying on fragile string-name blacklists:
      - `"stem_or_head"`: Input patchifiers, timestep/caption/vector stems, SSL projectors, and
        final output heads outside repeating transformer/UNet block containers.
      - `"modulation"`: Conditioning / AdaLN shift-scale-gate projections inside repeating blocks,
        identified structurally by:
          (a) 1-linear composite sub-branches on a block (`modC`, `modX`, `img_mod`, `txt_mod`,
              `modulation`, `adaLN_modulation`, `img_norm1..3`, `norm1`, `emb_layers`) that lack
              attention head attributes and either expand `D -> k*D` (`k >= 2`) or attach to
              an activation/normalization module,
          (b) 2-linear bottleneck sub-branches (`adaln_modulation_*`, 2-layer `modulation`) whose
              final linear expands to `>= 2 * D` instead of contracting back to stream width `D`,
          (c) direct leaf `Linear` attributes on a block (`mod_x`, `mod_y`, `skip_linear`, `TimestepBlock.mapper*`)
              that either expand `D -> k*D` (`k >= 2`) with no paired contraction or act as a lone
              U-ViT skip-concatenation adapter when the block already delegates Attention/FFN to composite child branches.
      - `"block_projection"`: Input/output boundary projections on containers wrapping nested repeating
        transformer blocks (such as UNet SpatialTransformer `proj_in` and `proj_out`, whether Linear
        or 1x1 Conv2d) that adapt between outer feature maps and inner sequence tokens.
      - `"moe_router"`: MoE router gates (`ff_i.gate`, `moe.gate`) that are leaf `Linear` siblings
        of an `nn.ModuleList` of composite expert blocks (`experts`) projecting `D -> num_experts`.
      - `"unused_auxiliary"`: Dead auxiliary projections inside a block that are never invoked by
        dense forward passes (e.g. MiniMax H3 `Attention.to_gate_compress`).
      - `"unused_secondary_modality"`: Secondary audio branches inside dual-branch audio-video
        transformer blocks (e.g. LTXAV `audio_attn1`, `audio_attn2`, `audio_to_video_attn`,
        `video_to_audio_attn`, `audio_ff`) and paired top-level `audio_*` repeating connectors
        (e.g. `audio_embeddings_connector` paired with `video_embeddings_connector`) that receive
        zero gradients during image/video-only training. Audio-only models (no paired visual branch)
        and shared single-stream AV models (e.g. MiniMax H3) are never affected.
      - `"attention"`: Multi-linear self/cross/joint attention projections (`q/k/v/qkv/out`).
      - `"ffn"`: Multi-linear feed-forward / SwiGLU / GeGLU / MoE expert projections (`up/gate/down`).
    """
    def _is_secondary_audio_attr(attr_name: str) -> bool:
        low = attr_name.lower()
        return (
            low.startswith("audio_")
            or low.endswith("_audio")
            or "audio_to_video" in low
            or "video_to_audio" in low
            or "_audio_attn" in low
            or "_audio_ff" in low
            or "_audio_mlp" in low
        )

    module_to_path: Dict[int, str] = {}
    all_linear_paths: Dict[str, nn.Module] = {}
    for full_name, sub in _iter_named_modules_skipping_lora(model):
        if not full_name:
            continue
        base_sub = _unwrap_lora_base(sub)
        module_to_path[id(sub)] = full_name
        module_to_path[id(base_sub)] = full_name
        if _is_linear_like(base_sub):
            all_linear_paths[full_name] = base_sub

    block_module_ids = _collect_repeating_block_module_ids(model)
    roles: Dict[str, str] = {}
    for path, lin in all_linear_paths.items():
        if block_module_ids and id(lin) not in block_module_ids:
            roles[path] = "stem_or_head"
        else:
            roles[path] = "attention_or_ffn"

    # 1. Detect MoE router gates: any leaf Linear sibling of an `nn.ModuleList` of expert blocks
    for _, sub in _iter_named_modules_skipping_lora(model, include_self=True):
        expert_lists = [
            child
            for _, child in sub.named_children()
            if isinstance(child, nn.ModuleList)
            and len(child) >= 2
            and all(_count_linear_like_descendants(el) >= 2 for el in child)
        ]
        if not expert_lists:
            continue
        num_experts = max(len(el) for el in expert_lists)
        for _, child in sub.named_children():
            base_c = _unwrap_lora_base(child)
            if _is_linear_like(base_c):
                in_f, out_f, _ = _extract_layer_dims(base_c)
                if out_f == num_experts or (out_f <= 64 and in_f >= 2 * out_f):
                    path = module_to_path.get(id(base_c))
                    if path:
                        roles[path] = "moe_router"

    # 2. Inspect repeating block structures for AdaLN modulation vs Attention vs FFN
    visited_blocks: set[int] = set()
    has_dual_branch_av_blocks = False
    for name, sub in _iter_named_modules_skipping_lora(model):
        if not name:
            continue
        stripped_lower = _strip_numeric_segments(name.replace("__", ".")).lower()
        parts_list = [p for p in stripped_lower.split(".") if p]
        block_candidates: List[nn.Module] = []
        if isinstance(sub, nn.ModuleList) and len(sub) >= 1:
            if any(
                _is_composite_transformer_block(c)
                or (isinstance(c, nn.ModuleList) and any(_is_composite_transformer_block(gc) for gc in c))
                for c in sub
            ):
                block_candidates.extend(list(sub))
        elif (parts_list and parts_list[-1] in _UNET_BLOCK_CONTAINER_NAMES) and _count_linear_like_descendants(sub) >= 1:
            if isinstance(sub, (nn.ModuleList, nn.Sequential)):
                block_candidates.extend(list(sub))
            else:
                block_candidates.append(sub)

        for raw_block in block_candidates:
            for block in _iter_effective_transformer_blocks(raw_block):
                if id(block) in visited_blocks:
                    continue
                visited_blocks.add(id(block))

                block_leaves = _collect_linear_like_leaves(block)
                if len(block_leaves) == 1:
                    # A block with only 1 linear total (e.g. UNet ResBlock with `emb_layers.1`)
                    only_lin = block_leaves[0][1]
                    p = module_to_path.get(id(only_lin))
                    if p:
                        roles[p] = "modulation"
                    continue

                direct_leaf_linears: List[nn.Module] = []
                direct_leaf_linear_attrs: Dict[int, str] = {}
                direct_leaf_pointwise_convs: List[nn.Module] = []
                composite_branches: List[Tuple[str, nn.Module, List[Tuple[str, nn.Module]]]] = []
                has_nested_block_list = False
                for child_attr, child_mod in block.named_children():
                    base_child = _unwrap_lora_base(child_mod)
                    if _is_linear_like(base_child):
                        direct_leaf_linears.append(base_child)
                        direct_leaf_linear_attrs[id(base_child)] = child_attr
                    elif _is_pointwise_conv(base_child):
                        direct_leaf_pointwise_convs.append(base_child)
                    elif isinstance(child_mod, nn.ModuleList) and any(
                        _is_composite_transformer_block(el)
                        or (isinstance(el, nn.ModuleList) and any(_is_composite_transformer_block(gc) for gc in el))
                        for el in child_mod
                    ):
                        # Nested repeating transformer block list (e.g. UNet SpatialTransformer.transformer_blocks):
                        # defer sub-branch classification to the individual transformer blocks inside `child_mod`.
                        has_nested_block_list = True
                    else:
                        c_leaves = _collect_linear_like_leaves(child_mod)
                        if c_leaves:
                            composite_branches.append((child_attr, child_mod, c_leaves))

                if has_nested_block_list and (direct_leaf_linears or direct_leaf_pointwise_convs):
                    for proj_mod in direct_leaf_linears + direct_leaf_pointwise_convs:
                        p = module_to_path.get(id(proj_mod))
                        if p:
                            roles[p] = "block_projection"
                    direct_leaf_linears = []

                has_multi_linear_composite = has_nested_block_list or any(
                    len(c_leaves) >= 2 for _, _, c_leaves in composite_branches
                )

                # Pure modulation block with only expanding leaf Linears (e.g. StableCascade `TimestepBlock`)
                if not composite_branches and direct_leaf_linears and not _has_attention_attributes(block):
                    direct_dims_only = {id(lin): _extract_layer_dims(lin) for lin in direct_leaf_linears}
                    if all(out_f >= 2 * in_f for in_f, out_f, _ in direct_dims_only.values()):
                        for lin in direct_leaf_linears:
                            p = module_to_path.get(id(lin))
                            if p:
                                roles[p] = "modulation"
                        continue

                # Direct leaf Linears on a block that ALREADY has composite Attention/FFN branches:
                # only classify expanding un-paired projections (`out_f >= 2 * in_f`, e.g. `mod_x`, `mod_y`)
                # as `"modulation"`, so stream/skip projections like HunyuanDiT `skip_linear` (`2*D -> D`) remain targeted.
                if has_multi_linear_composite and direct_leaf_linears:
                    direct_dims = {id(lin): _extract_layer_dims(lin) for lin in direct_leaf_linears}
                    for lin in direct_leaf_linears:
                        in_f, out_f, _ = direct_dims[id(lin)]
                        if out_f >= 2 * in_f:
                            has_paired_contraction = any(
                                other is not lin
                                and (
                                    direct_dims[id(other)][0] in (out_f, out_f // 2)
                                    and direct_dims[id(other)][1] == in_f
                                )
                                for other in direct_leaf_linears
                            )
                            if not has_paired_contraction:
                                p = module_to_path.get(id(lin))
                                if p:
                                    roles[p] = "modulation"

                # Detect dual-branch audio-video blocks (e.g. LTXAV `BasicAVTransformerBlock`) that
                # contain BOTH primary visual attention/FFN branches AND dedicated secondary audio /
                # cross-modal-audio branches (`audio_attn1`, `audio_attn2`, `audio_to_video_attn`,
                # `video_to_audio_attn`, `audio_ff`). Audio-only backbones (no paired visual branch)
                # and shared single-stream AV backbones (e.g. MiniMax H3) never satisfy this predicate.
                audio_branch_attrs = {
                    child_attr
                    for child_attr, _, _ in composite_branches
                    if _is_secondary_audio_attr(child_attr)
                }
                has_primary_non_audio_compute = any(
                    (not _is_secondary_audio_attr(child_attr))
                    and (len(c_leaves) >= 2 or _has_attention_attributes(child_mod))
                    for child_attr, child_mod, c_leaves in composite_branches
                )
                is_dual_branch_av_block = bool(audio_branch_attrs) and has_primary_non_audio_compute
                if is_dual_branch_av_block:
                    has_dual_branch_av_blocks = True
                    for lin in direct_leaf_linears:
                        d_attr = direct_leaf_linear_attrs.get(id(lin), "")
                        if d_attr and _is_secondary_audio_attr(d_attr):
                            p = module_to_path.get(id(lin))
                            if p:
                                roles[p] = "unused_secondary_modality"

                # Analyze composite child sub-branches on the block
                for child_attr, child_mod, c_leaves in composite_branches:
                    if is_dual_branch_av_block and child_attr in audio_branch_attrs:
                        for _, lin in c_leaves:
                            p = module_to_path.get(id(lin))
                            if p:
                                roles[p] = "unused_secondary_modality"
                        continue
                    is_mod_branch = False
                    if len(c_leaves) == 1 and not _has_attention_attributes(child_mod):
                        only_lin = c_leaves[0][1]
                        in_f, out_f, _ = _extract_layer_dims(only_lin)
                        if out_f >= 2 * in_f or _has_norm_or_activation_child(child_mod):
                            p = module_to_path.get(id(only_lin))
                            if p:
                                roles[p] = "modulation"
                            is_mod_branch = True
                    elif len(c_leaves) == 2 and not _has_attention_attributes(child_mod):
                        l1, l2 = c_leaves[0][1], c_leaves[1][1]
                        in1, out1, _ = _extract_layer_dims(l1)
                        in2, out2, _ = _extract_layer_dims(l2)
                        # In a 2-layer AdaLN bottleneck (e.g. Cosmos/Anima `adaln_modulation_*` or
                        # Kandinsky5 `modulation`), L1 bottlenecks/projects `D -> B` (with `out1 <= in1`)
                        # and L2 expands to `k*D` (`out2 >= 2 * in1` and `out2 >= 2 * in2`).
                        if out1 <= in1 and out2 >= 2 * in1 and out2 >= 2 * in2:
                            for lin in (l1, l2):
                                p = module_to_path.get(id(lin))
                                if p:
                                    roles[p] = "modulation"
                            is_mod_branch = True
                    if not is_mod_branch:
                        is_attn = _has_attention_attributes(child_mod) or any(
                            tok in child_attr.lower() for tok in ("attn", "attention")
                        )
                        branch_role = "attention" if is_attn else "ffn"
                        is_minimax_attn = is_attn and getattr(child_mod.__class__, "__module__", "").startswith(
                            "comfy.ldm.minimax."
                        )
                        for leaf_attr, lin in c_leaves:
                            p = module_to_path.get(id(lin))
                            if not p:
                                continue
                            # MiniMax H3 / H3-Image `Attention.to_gate_compress` is a VSA sparse-attention
                            # gate never called by dense `Attention.forward`; exclude it so training never
                            # allocates dead zero-gradient LoRA adapters.
                            if is_minimax_attn and (
                                leaf_attr == "to_gate_compress" or p.endswith(".to_gate_compress")
                            ):
                                roles[p] = "unused_auxiliary"
                            elif roles.get(p) == "attention_or_ffn":
                                roles[p] = branch_role

    # 3. Exclude paired top-level secondary `audio_*` repeating towers/connectors (e.g. LTXAV
    # `audio_embeddings_connector` paired with `video_embeddings_connector`) while leaving
    # audio-only models (which have no paired `video_*`/`image_*` counterpart) completely untouched.
    top_child_names = {child_name for child_name, _ in model.named_children()}
    paired_audio_prefixes = {
        attr
        for attr in top_child_names
        if attr.lower().startswith("audio_")
        and (
            ("video_" + attr[6:]) in top_child_names
            or ("image_" + attr[6:]) in top_child_names
            or (has_dual_branch_av_blocks and "connector" in attr.lower())
        )
    }
    if paired_audio_prefixes:
        for p in list(roles.keys()):
            top_seg = p.split(".", 1)[0]
            if top_seg in paired_audio_prefixes and roles.get(p) != "stem_or_head":
                roles[p] = "unused_secondary_modality"

    for k in list(roles.keys()):
        if ".modCX." in k:
            roles[k.replace(".modCX.", ".modX.")] = roles[k]

    # Harmonize structural roles across repeating numeric block containers:
    # If any block's module path has been classified as "modulation", "stem_or_head", or "moe_router",
    # ensure corresponding modules across all repeating blocks of the same container inherit that role.
    canonical_non_target_roles: Dict[str, str] = {}
    for p, r in roles.items():
        if r in (
            "modulation",
            "stem_or_head",
            "moe_router",
            "block_projection",
            "unused_auxiliary",
            "unused_secondary_modality",
        ):
            canonical_non_target_roles[_strip_numeric_segments(p)] = r

    for p in list(roles.keys()):
        stripped = _strip_numeric_segments(p)
        if stripped in canonical_non_target_roles:
            roles[p] = canonical_non_target_roles[stripped]

    return roles


def trace_single_block_dataflow(
    model: nn.Module,
    sample_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Combine static residual-stream structural topology (`classify_block_structural_roles`) with
    a 1-block forward hook probe on the first block (`block[0]`) of each repeating `nn.ModuleList`.
    Records input/output tensor ranks (`3D` sequence tokens `(B, N, D)` vs `2D` global conditioning
    `(B, D_cond)`) when `sample_inputs` are provided or when leaf sub-branches can be probed in
    isolation, caching the verified structural role map on `model._omni_traced_block_roles`.
    """
    roles = classify_block_structural_roles(model)
    name_to_module: Dict[str, nn.Module] = dict(_iter_named_modules_skipping_lora(model))

    if sample_inputs:
        hooks = []
        observed_ranks: Dict[str, Tuple[int, int]] = {}

        def _make_hook(path: str):
            def _hook(_mod: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
                in_t = next((x for x in inputs if isinstance(x, torch.Tensor)), None)
                out_t = output[0] if isinstance(output, (tuple, list)) and output else output
                if isinstance(in_t, torch.Tensor) and isinstance(out_t, torch.Tensor):
                    observed_ranks[path] = (in_t.ndim, out_t.ndim)

            return _hook

        for path, role in list(roles.items()):
            if role != "stem_or_head":
                sub = name_to_module.get(path)
                if sub is not None:
                    hooks.append(sub.register_forward_hook(_make_hook(path)))
        try:
            with torch.no_grad():
                model(**sample_inputs)
        except Exception as exc:
            report_exception(
                "lora.role_trace_forward",
                "Structural role trace forward pass exited early; using partial/static roles",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="lora",
            )
        finally:
            for h in hooks:
                h.remove()

        # Any linear inside a block that only ever receives 2D conditioning `(B, D_cond)` while
        # siblings process 3D+ sequence tokens `(B, N, D)` is definitively a conditioning modulation
        has_sequence_tokens = any(in_ndim >= 3 for in_ndim, _ in observed_ranks.values())
        if has_sequence_tokens:
            for path, (in_ndim, out_ndim) in observed_ranks.items():
                if in_ndim == 2 and out_ndim == 2 and roles.get(path) not in (
                    "stem_or_head",
                    "moe_router",
                    "block_projection",
                    "unused_auxiliary",
                    "unused_secondary_modality",
                ):
                    roles[path] = "modulation"

    try:
        setattr(model, "_omni_traced_block_roles", dict(roles))
    except (AttributeError, TypeError) as exc:
        report_exception(
            "lora.role_trace_cache_attr",
            "Could not cache traced block roles on model",
            exc,
            level=logging.DEBUG,
            once=True,
            logger_name="lora",
        )
    return roles


class LoRAInjector:
    """
    Traverses any ComfyUI `diffusion_model` (`nn.Module`) and replaces matching Linear / 1x1 Conv
    layers with trainable `LoRALinear` adapters in-place.
    Safe to run repeatedly on cached ComfyUI `ModelPatcher` objects without double-wrapping.
    """

    @staticmethod
    def _matches_target(module_name: str, patterns: Iterable[str]) -> bool:
        normalized = module_name.replace("__", ".")
        stripped_name = _strip_numeric_segments(normalized)
        for pattern in patterns:
            if not pattern:
                continue
            norm_pat = pattern.replace("__", ".")
            stripped_pat = _strip_numeric_segments(norm_pat)
            if (
                normalized == norm_pat
                or normalized.endswith(f".{norm_pat}")
                or stripped_name == stripped_pat
                or stripped_name.endswith(f".{stripped_pat}")
            ):
                return True
            if any(ch in pattern for ch in ("*", "^", "$", "[", "]", "|", "+", "?")):
                try:
                    if re.search(pattern, normalized):
                        return True
                except re.error:
                    pass
        return False

    @classmethod
    def _iter_injectable_modules(
        cls,
        root: nn.Module,
        prefix: str = "",
    ) -> Iterable[Tuple[nn.Module, str, str, nn.Module]]:
        """
        Walk `root` without recursing inside existing `LoRALinear` wrappers.
        If a child is already a `LoRALinear` (from a prior ComfyUI queue run), yield its
        underlying `base_layer` at the original module path so re-injection is idempotent.
        """
        for attr_name, child in root.named_children():
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(child, LoRALinear):
                base = child.base_layer
                while isinstance(base, LoRALinear):
                    base = base.base_layer
                yield (root, attr_name, full_name, base)
            else:
                yield (root, attr_name, full_name, child)
                yield from cls._iter_injectable_modules(child, prefix=full_name)

    @classmethod
    def inject(
        cls,
        model: nn.Module,
        config: LoRAConfig,
        lora_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> List[str]:
        """
        Freeze the entire base model and inject trainable `LoRALinear` adapters using
        progressive discovery:
          Tier 1: Configured `config.target_modules` (if non-empty)
          Tier 2: Universal attention/MLP Linear-like layers via structural residual-stream topology
                  (`classify_block_structural_roles`) + stem/modulation exclusion
          Tier 3: All Linear-like layers in the model
          Tier 4: Conv2d/Conv1d/Conv3d layers (preferring 1x1 projections first)
        Returns the list of injected module paths.
        """
        if lora_dtype == torch.float64:
            lora_dtype = torch.float32

        model.requires_grad_(False)
        injected_paths: List[str] = []

        all_nodes = list(cls._iter_injectable_modules(model))
        candidates: List[Tuple[nn.Module, str, str, nn.Module]] = []

        # Tier 1: Match configured target_modules against linear-like layers (and pointwise 1x1 block
        # projections) first; only match general conv layers for patterns that did not match a linear
        # layer (or when `include_conv=True`).
        if config.target_modules:
            linear_candidates: List[Tuple[nn.Module, str, str, nn.Module]] = []
            matched_patterns: set[str] = set()
            for parent, attr_name, full_name, sub in all_nodes:
                if _is_linear_like(sub):
                    for pat in config.target_modules:
                        if pat and cls._matches_target(full_name, (pat,)):
                            linear_candidates.append((parent, attr_name, full_name, sub))
                            matched_patterns.add(pat)
                            break
            excluded_conv_containers = (
                (_UNET_BLOCK_CONTAINER_NAMES | {"out", "conv_in", "conv_out"})
                if config.include_conv
                else (_UNET_BLOCK_CONTAINER_NAMES | {"out", "conv_in", "conv_out", "in_layers", "out_layers"})
            )
            unmatched_patterns = [
                pat
                for pat in config.target_modules
                if pat
                and (config.include_conv or pat not in matched_patterns)
                and _strip_numeric_segments(pat.replace("__", ".")).lower() not in excluded_conv_containers
            ]
            conv_candidates: List[Tuple[nn.Module, str, str, nn.Module]] = []
            if unmatched_patterns or not linear_candidates:
                active_conv_patterns = unmatched_patterns if linear_candidates else config.target_modules
                conv_candidates = [
                    (parent, attr_name, full_name, sub)
                    for (parent, attr_name, full_name, sub) in all_nodes
                    if _is_conv_like(sub)
                    and _strip_numeric_segments(full_name.replace("__", ".")).lower()
                    not in (_UNET_BLOCK_CONTAINER_NAMES | {"out", "conv_in", "conv_out"})
                    and (config.include_conv or _is_pointwise_conv(sub))
                    and cls._matches_target(full_name, active_conv_patterns)
                ]
            candidates = linear_candidates + conv_candidates

        # Tier 2: Universal attention/MLP/block-projection discovery inside repeating transformer/UNet
        # blocks via structural residual-stream role classification (zero string blacklists).
        if not candidates:
            block_module_ids = _collect_repeating_block_module_ids(model)
            structural_roles = classify_block_structural_roles(model)
            allowed_roles = ("attention", "ffn", "attention_or_ffn", "block_projection")
            candidates = [
                (parent, attr_name, full_name, sub)
                for (parent, attr_name, full_name, sub) in all_nodes
                if (
                    _is_linear_like(sub)
                    or (_is_pointwise_conv(sub) and structural_roles.get(full_name) == "block_projection")
                )
                and (not block_module_ids or id(sub) in block_module_ids)
                and structural_roles.get(full_name) in allowed_roles
            ]
            if config.include_conv:
                seen_paths = {p for _, _, p, _ in candidates}
                candidates.extend(
                    (parent, attr_name, full_name, sub)
                    for (parent, attr_name, full_name, sub) in all_nodes
                    if _is_conv_like(sub)
                    and full_name not in seen_paths
                    and (not block_module_ids or id(sub) in block_module_ids)
                    and _strip_numeric_segments(full_name.replace("__", ".")).lower()
                    not in (_UNET_BLOCK_CONTAINER_NAMES | {"out", "conv_in", "conv_out"})
                )

        # Tier 3: Any linear-like module in the backbone
        if not candidates:
            candidates = [
                (parent, attr_name, full_name, sub)
                for (parent, attr_name, full_name, sub) in all_nodes
                if _is_linear_like(sub)
            ]

        # Tier 4: Conv2d / Conv1d / Conv3d fallback (prefer 1x1 projections first, then all convs)
        if not candidates:
            conv_1x1 = [
                (parent, attr_name, full_name, sub)
                for (parent, attr_name, full_name, sub) in all_nodes
                if _is_conv_like(sub)
                and tuple(getattr(sub, "kernel_size", (1, 1))) in ((1,), (1, 1), (1, 1, 1))
            ]
            candidates = conv_1x1 or [
                (parent, attr_name, full_name, sub)
                for (parent, attr_name, full_name, sub) in all_nodes
                if _is_conv_like(sub)
            ]

        for parent, attr_name, full_name, target_layer in candidates:
            lora_layer = LoRALinear(
                base_layer=target_layer,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                lora_dtype=lora_dtype,
                use_rslora=config.use_rslora,
                device=device,
            )
            if isinstance(parent, nn.ModuleDict):
                parent[attr_name] = lora_layer
            elif isinstance(parent, (nn.Sequential, nn.ModuleList)) and attr_name.isdigit():
                parent[int(attr_name)] = lora_layer
            else:
                setattr(parent, attr_name, lora_layer)
            injected_paths.append(full_name)

        # Explicitly guarantee requires_grad=True on all injected LoRA A/B weights
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.weight.requires_grad_(True)
                module.lora_B.weight.requires_grad_(True)

        return injected_paths

    @classmethod
    def summarize_targets(
        cls,
        model: nn.Module,
        injected_paths: Iterable[str],
        roles: Optional[Dict[str, str]] = None,
        max_groups_per_role: int = 16,
    ) -> Dict[str, Any]:
        """
        Produce a concise structural summary of which layers were adapted (and which
        structural roles were excluded) so training logs make layer targeting immediately
        verifiable at startup across any architecture.
        """
        injected_list = list(injected_paths)
        injected_set = set(injected_list)
        name_map: Dict[str, nn.Module] = dict(_iter_named_modules_skipping_lora(model))
        if roles is None:
            roles = classify_block_structural_roles(model)

        linear_count = 0
        pointwise_conv_count = 0
        spatial_conv_count = 0

        # role -> (collapsed_pattern, kind_tag) -> count
        role_pattern_counts: Dict[str, Dict[Tuple[str, str], int]] = {}
        role_totals: Dict[str, int] = {}

        for path in injected_list:
            mod = name_map.get(path)
            base = _unwrap_lora_base(mod) if mod is not None else None
            if base is not None and _is_pointwise_conv(base):
                pointwise_conv_count += 1
                kind_tag = "1x1 Conv"
            elif base is not None and _is_conv_like(base):
                spatial_conv_count += 1
                kind_tag = "Conv"
            else:
                linear_count += 1
                kind_tag = ""

            role = roles.get(path)
            if not role:
                role = "conv" if kind_tag in ("Conv", "1x1 Conv") else "other"

            collapsed = _collapse_numeric_segments(path.replace("__", ".")) or path
            role_totals[role] = role_totals.get(role, 0) + 1
            bucket = role_pattern_counts.setdefault(role, {})
            bucket[(collapsed, kind_tag)] = bucket.get((collapsed, kind_tag), 0) + 1

        def _format_role_groups(bucket: Dict[Tuple[str, str], int]) -> List[str]:
            # Group sibling leaf attributes sharing (parent, count, kind_tag) -> parent.{a, b} (xN)
            raw_entries: List[Tuple[str, str, int, str]] = []
            direct_parents: set[Tuple[str, int, str]] = set()
            for (pat, kind_tag), cnt in bucket.items():
                if "." in pat:
                    parent, leaf = pat.rsplit(".", 1)
                    if leaf != "*":
                        direct_parents.add((parent, cnt, kind_tag))
                else:
                    parent, leaf = "", pat
                raw_entries.append((parent, leaf, cnt, kind_tag))

            grouped: Dict[Tuple[str, int, str], List[str]] = {}
            for parent, leaf, cnt, kind_tag in raw_entries:
                if leaf == "*" and "." in parent:
                    grandparent, sub = parent.rsplit(".", 1)
                    if (grandparent, cnt, kind_tag) in direct_parents:
                        parent, leaf = grandparent, f"{sub}.*"
                grouped.setdefault((parent, cnt, kind_tag), []).append(leaf)

            rendered: List[Tuple[int, str]] = []
            for (parent, cnt, kind_tag), leaves in grouped.items():
                leaves_sorted = sorted(set(leaves))
                if parent and len(leaves_sorted) > 1:
                    expr = f"{parent}.{{{', '.join(leaves_sorted)}}}"
                elif parent:
                    expr = f"{parent}.{leaves_sorted[0]}"
                else:
                    expr = leaves_sorted[0]
                meta_parts = [f"x{cnt}"]
                if kind_tag:
                    meta_parts.append(kind_tag)
                rendered.append((cnt, f"{expr} ({', '.join(meta_parts)})"))

            rendered.sort(key=lambda item: (-item[0], item[1]))
            texts = [text for _, text in rendered[:max_groups_per_role]]
            if len(rendered) > max_groups_per_role:
                texts.append(f"(+{len(rendered) - max_groups_per_role} more)")
            return texts

        role_order = ("attention", "ffn", "attention_or_ffn", "block_projection", "conv", "other")
        by_role: Dict[str, List[str]] = {}
        for r in role_order:
            if r in role_pattern_counts:
                by_role[r] = _format_role_groups(role_pattern_counts[r])
        for r in sorted(role_pattern_counts):
            if r not in by_role:
                by_role[r] = _format_role_groups(role_pattern_counts[r])

        # Summarize excluded structural roles (only counting real modules present in name_map)
        excluded_counts: Dict[str, Dict[str, int]] = {}
        excluded_totals: Dict[str, int] = {}
        for mod_path, role in roles.items():
            if mod_path in injected_set or mod_path not in name_map:
                continue
            if role in (
                "stem_or_head",
                "modulation",
                "moe_router",
                "unused_auxiliary",
                "unused_secondary_modality",
            ):
                collapsed = _collapse_numeric_segments(mod_path.replace("__", ".")) or mod_path
                excluded_totals[role] = excluded_totals.get(role, 0) + 1
                ebucket = excluded_counts.setdefault(role, {})
                ebucket[collapsed] = ebucket.get(collapsed, 0) + 1

        type_parts: List[str] = []
        if linear_count:
            type_parts.append(f"{linear_count} Linear")
        if pointwise_conv_count:
            type_parts.append(f"{pointwise_conv_count} 1x1 Conv")
        if spatial_conv_count:
            type_parts.append(f"{spatial_conv_count} Conv")
        type_str = ", ".join(type_parts) if type_parts else "0 modules"

        summary_lines: List[str] = [f"Target modules ({len(injected_list)} adapted: {type_str}):"]
        for r, groups in by_role.items():
            summary_lines.append(f"  {r} ({role_totals[r]}): {', '.join(groups)}")

        if excluded_totals:
            excl_parts: List[str] = []
            for r in (
                "stem_or_head",
                "modulation",
                "moe_router",
                "unused_auxiliary",
                "unused_secondary_modality",
            ):
                if r not in excluded_totals:
                    continue
                top_pats = sorted(excluded_counts[r].items(), key=lambda kv: (-kv[1], kv[0]))
                shown = [p for p, _ in top_pats[:3]]
                if len(top_pats) > 3:
                    shown.append(f"+{len(top_pats) - 3} more")
                excl_parts.append(f"{r}={excluded_totals[r]} ({', '.join(shown)})")
            if excl_parts:
                summary_lines.append(
                    f"  excluded ({sum(excluded_totals.values())}): {'; '.join(excl_parts)}"
                )

        return {
            "total": len(injected_list),
            "linear": linear_count,
            "pointwise_conv": pointwise_conv_count,
            "spatial_conv": spatial_conv_count,
            "role_totals": role_totals,
            "by_role": by_role,
            "excluded_totals": excluded_totals,
            "summary_lines": summary_lines,
        }

    @classmethod
    def uninject(
        cls,
        model: nn.Module,
        offload_device: Optional[torch.device] = None,
    ) -> int:
        """
        Restore all `LoRALinear` wrapped modules in `model` back to their original
        `base_layer` in-place so ComfyUI's cached `diffusion_model` retains its native
        `.weight` parameter paths, dtypes (including FP8 `float8_e4m3fn`), `fast_disk=True`
        safetensors `mmap` backing, and uniform device placement for `LoraLoader` and `KSampler`.
        Also resets any model-level prefix KV caches (such as `QwenImage21Transformer2DModel.reset_prefix_cache`).
        """
        # Wait for any in-flight GPU kernels on XPU/CUDA/NPU/HPU before replacing or releasing GPU parameters
        for dev_attr in ("xpu", "cuda", "npu", "hpu"):
            dev_mod = getattr(torch, dev_attr, None)
            if dev_mod is not None and hasattr(dev_mod, "is_available") and hasattr(dev_mod, "synchronize"):
                try:
                    if dev_mod.is_available():
                        dev_mod.synchronize()
                except (RuntimeError, AttributeError) as exc:
                    report_exception(
                        f"lora.uninject_sync.{dev_attr}",
                        f"Device {dev_attr}.synchronize() failed during LoRA uninject",
                        exc,
                        level=logging.DEBUG,
                        once=True,
                        logger_name="lora",
                    )

        restored = 0
        target_dev: Optional[torch.device] = (
            offload_device
            if isinstance(offload_device, torch.device)
            else getattr(model, "_omni_initial_device", None)
        )
        with torch.inference_mode(False), torch.no_grad():
            for parent in list(model.modules()):
                for attr_name, child in list(parent.named_children()):
                    if isinstance(child, LoRALinear):
                        try:
                            child.lora_A.weight.grad = None
                            child.lora_B.weight.grad = None
                        except (AttributeError, RuntimeError) as exc:
                            report_exception(
                                "lora.uninject_clear_grad",
                                "Could not clear LoRA adapter gradients during uninject",
                                exc,
                                level=logging.DEBUG,
                                once=True,
                                logger_name="lora",
                            )
                        base = child.base_layer
                        while isinstance(base, LoRALinear):
                            base = base.base_layer
                        for b_param in base.parameters():
                            saved_p = getattr(b_param, "_omni_saved_patches", None)
                            if isinstance(saved_p, list):
                                b_param.patches = list(saved_p)
                                delattr(b_param, "_omni_saved_patches")
                        if isinstance(parent, nn.ModuleDict):
                            parent[attr_name] = base
                        elif isinstance(parent, (nn.Sequential, nn.ModuleList)) and attr_name.isdigit():
                            parent[int(attr_name)] = base
                        else:
                            setattr(parent, attr_name, base)
                        restored += 1

            # Restore all modified parameters, buffers, and scales from preserved CPU (`fast_disk=True` mmap)
            # references without running GPU-to-CPU DMA transfers or allocating new pagefile heap RAM.
            for sub in list(model.modules()):
                if hasattr(sub, "reset_prefix_cache") and callable(getattr(sub, "reset_prefix_cache")):
                    try:
                        sub.reset_prefix_cache()  # type: ignore[operator]
                    except Exception as exc:
                        report_exception(
                            "lora.uninject_reset_prefix_cache",
                            f"reset_prefix_cache() failed on {type(sub).__name__} during uninject",
                            exc,
                            level=logging.DEBUG,
                            once=True,
                            logger_name="lora",
                        )

                orig_map: Optional[Dict[str, Tuple[torch.Tensor, bool]]] = getattr(sub, "_omni_orig_params", None)
                restored_param_names: set[str] = set()
                if isinstance(orig_map, dict):
                    for pname, (orig_t, orig_req_grad) in list(orig_map.items()):
                        restored_param_names.add(pname)
                        cur_p = getattr(sub, pname, None)
                        if isinstance(cur_p, torch.Tensor) and cur_p.grad is not None:
                            cur_p.grad = None
                        dest_d = (
                            target_dev
                            if isinstance(target_dev, torch.device)
                            else (cur_p.device if isinstance(cur_p, torch.Tensor) else orig_t.device)
                        )
                        if orig_t.device == dest_d:
                            if is_custom_quantized_tensor(orig_t):
                                assign_module_parameter(sub, pname, orig_t)
                            elif isinstance(orig_t, nn.Parameter) and bool(orig_t.requires_grad) == bool(orig_req_grad):
                                setattr(sub, pname, orig_t)
                            else:
                                setattr(sub, pname, nn.Parameter(orig_t, requires_grad=bool(orig_req_grad)))
                        else:
                            try:
                                if is_custom_quantized_tensor(orig_t):
                                    assign_module_parameter(
                                        sub,
                                        pname,
                                        move_custom_quantized_tensor(orig_t, dest_d),
                                    )
                                else:
                                    restored_t = orig_t.detach().to(device=dest_d)
                                    setattr(sub, pname, nn.Parameter(restored_t, requires_grad=bool(orig_req_grad)))
                            except Exception as exc:
                                report_exception(
                                    "lora.uninject_restore_param",
                                    f"Could not move restored parameter '{pname}' to {dest_d}; keeping original device",
                                    exc,
                                    level=logging.WARNING,
                                    once=True,
                                    logger_name="lora",
                                )
                                if is_custom_quantized_tensor(orig_t):
                                    assign_module_parameter(sub, pname, orig_t)
                                elif isinstance(orig_t, nn.Parameter):
                                    setattr(sub, pname, orig_t)
                                else:
                                    setattr(sub, pname, nn.Parameter(orig_t.detach().to(device="cpu"), requires_grad=bool(orig_req_grad)))
                    orig_map.clear()
                    delattr(sub, "_omni_orig_params")

                orig_buf_map: Optional[Dict[str, torch.Tensor]] = getattr(sub, "_omni_orig_buffers", None)
                if isinstance(orig_buf_map, dict):
                    for bname, orig_b in list(orig_buf_map.items()):
                        if isinstance(orig_b, torch.Tensor):
                            dest_d = target_dev if isinstance(target_dev, torch.device) else orig_b.device
                            if orig_b.device == dest_d:
                                if is_custom_quantized_tensor(orig_b):
                                    assign_module_parameter(sub, bname, orig_b)
                                else:
                                    sub._buffers[bname] = orig_b
                            else:
                                try:
                                    if is_custom_quantized_tensor(orig_b):
                                        assign_module_parameter(sub, bname, move_custom_quantized_tensor(orig_b, dest_d))
                                    else:
                                        sub._buffers[bname] = orig_b.detach().to(device=dest_d)
                                except Exception as exc:
                                    report_exception(
                                        "lora.uninject_restore_buffer",
                                        f"Could not move restored buffer '{bname}' to {dest_d}; keeping original device",
                                        exc,
                                        level=logging.WARNING,
                                        once=True,
                                        logger_name="lora",
                                    )
                                    sub._buffers[bname] = orig_b
                    orig_buf_map.clear()
                    delattr(sub, "_omni_orig_buffers")

                orig_scale_map: Optional[Dict[str, torch.Tensor]] = getattr(sub, "_omni_orig_scales", None)
                if isinstance(orig_scale_map, dict):
                    for s_attr, orig_s in list(orig_scale_map.items()):
                        if isinstance(orig_s, torch.Tensor):
                            dest_d = target_dev if isinstance(target_dev, torch.device) else orig_s.device
                            if orig_s.device == dest_d:
                                setattr(sub, s_attr, orig_s)
                            else:
                                try:
                                    setattr(sub, s_attr, orig_s.detach().to(device=dest_d))
                                except Exception as exc:
                                    report_exception(
                                        "lora.uninject_restore_scale",
                                        f"Could not move restored scale '{s_attr}' to {dest_d}; keeping original device",
                                        exc,
                                        level=logging.WARNING,
                                        once=True,
                                        logger_name="lora",
                                    )
                                    setattr(sub, s_attr, orig_s)
                    orig_scale_map.clear()
                    delattr(sub, "_omni_orig_scales")

                scale_t = getattr(sub, "_omni_weight_scale", None)
                orig_dt = getattr(sub, "_omni_orig_dtype", None)
                w_param = getattr(sub, "weight", None)
                if (
                    "weight" not in restored_param_names
                    and isinstance(scale_t, torch.Tensor)
                    and isinstance(orig_dt, torch.dtype)
                    and isinstance(w_param, torch.Tensor)
                    and (is_fp8_dtype(w_param.dtype) or w_param.dtype == torch.int8)
                ):
                    w_f32 = w_param.detach().to(dtype=torch.float32)
                    s_f32 = scale_t.detach().to(device=w_f32.device, dtype=torch.float32)
                    if s_f32.ndim == 1 and w_f32.ndim >= 2 and s_f32.numel() == w_f32.shape[0]:
                        s_f32 = s_f32.view(-1, *([1] * (w_f32.ndim - 1)))
                    w_restored = (w_f32 * s_f32).to(dtype=orig_dt)
                    sub.weight = nn.Parameter(w_restored, requires_grad=False)
                if hasattr(sub, "_omni_weight_scale"):
                    if "_omni_weight_scale" in sub._buffers:
                        del sub._buffers["_omni_weight_scale"]
                    else:
                        delattr(sub, "_omni_weight_scale")
                if hasattr(sub, "_omni_orig_dtype"):
                    delattr(sub, "_omni_orig_dtype")

                if hasattr(sub, "comfy_patched_weights"):
                    try:
                        delattr(sub, "comfy_patched_weights")
                    except (AttributeError, TypeError) as exc:
                        report_exception(
                            "lora.uninject_del_comfy_patched",
                            "Could not delete comfy_patched_weights attribute; setting False",
                            exc,
                            level=logging.DEBUG,
                            once=True,
                            logger_name="lora",
                        )
                        setattr(sub, "comfy_patched_weights", False)
                if hasattr(sub, "_v_signature"):
                    try:
                        setattr(sub, "_v_signature", None)
                    except (AttributeError, TypeError) as exc:
                        report_exception(
                            "lora.uninject_clear_v_signature",
                            "Could not clear _v_signature cache on module",
                            exc,
                            level=logging.DEBUG,
                            once=True,
                            logger_name="lora",
                        )
                comb = getattr(sub, "_combined_linear", None)
                if comb is not None and hasattr(comb, "_v_weight"):
                    try:
                        setattr(comb, "_v_weight", None)
                    except (AttributeError, TypeError) as exc:
                        report_exception(
                            "lora.uninject_clear_v_weight",
                            "Could not clear _combined_linear._v_weight cache on module",
                            exc,
                            level=logging.DEBUG,
                            once=True,
                            logger_name="lora",
                        )

                for p in sub.parameters(recurse=False):
                    if p.grad is not None:
                        p.grad = None

            # Ensure all parameters, buffers, and scale_weight tensors sit on a single consistent device
            # (never split half-on-GPU and half-on-CPU after VRAM-budgeted training)
            if isinstance(target_dev, torch.device):
                dev_types = {
                    p.device.type
                    for p in model.parameters()
                    if isinstance(p, torch.Tensor) and p.device.type != "meta"
                }
                if len(dev_types) > 1 or (len(dev_types) == 1 and next(iter(dev_types)) != target_dev.type):
                    unify_dev = (
                        offload_device
                        if isinstance(offload_device, torch.device)
                        else (target_dev if "cpu" not in dev_types else torch.device("cpu"))
                    )
                    for sub in model.modules():
                        for pname, param in list(sub.named_parameters(recurse=False)):
                            if param.device != unify_dev:
                                try:
                                    if is_custom_quantized_tensor(param):
                                        assign_module_parameter(
                                            sub,
                                            pname,
                                            move_custom_quantized_tensor(param, unify_dev),
                                        )
                                    else:
                                        setattr(
                                            sub,
                                            pname,
                                            nn.Parameter(
                                                param.detach().to(device=unify_dev),
                                                requires_grad=param.requires_grad,
                                            ),
                                        )
                                except Exception as exc:
                                    report_exception(
                                        "lora.uninject_unify_param",
                                        f"Could not unify parameter '{pname}' onto {unify_dev} after uninject",
                                        exc,
                                        level=logging.WARNING,
                                        once=True,
                                        logger_name="lora",
                                    )
                        for bname, buf in list(sub.named_buffers(recurse=False)):
                            if isinstance(buf, torch.Tensor) and buf.device != unify_dev:
                                try:
                                    if is_custom_quantized_tensor(buf):
                                        assign_module_parameter(
                                            sub,
                                            bname,
                                            move_custom_quantized_tensor(buf, unify_dev),
                                        )
                                    else:
                                        setattr(sub, bname, buf.detach().to(device=unify_dev))
                                except Exception as exc:
                                    report_exception(
                                        "lora.uninject_unify_buffer",
                                        f"Could not unify buffer '{bname}' onto {unify_dev} after uninject",
                                        exc,
                                        level=logging.WARNING,
                                        once=True,
                                        logger_name="lora",
                                    )
                        for s_attr in ("scale_weight", "weight_scale"):
                            s_val = getattr(sub, s_attr, None)
                            if isinstance(s_val, torch.Tensor) and s_val.device != unify_dev:
                                try:
                                    setattr(sub, s_attr, s_val.detach().to(device=unify_dev))
                                except Exception as exc:
                                    report_exception(
                                        "lora.uninject_unify_scale",
                                        f"Could not unify scale '{s_attr}' onto {unify_dev} after uninject",
                                        exc,
                                        level=logging.WARNING,
                                        once=True,
                                        logger_name="lora",
                                    )
                if hasattr(model, "_omni_initial_device"):
                    delattr(model, "_omni_initial_device")

        return restored

    @staticmethod
    def capture_comfy_lora_key_map(
        backbone: nn.Module,
        comfy_model: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Query ComfyUI's native `comfy.lora.model_lora_keys_unet` for `comfy_model` (while
        uninjected) and cache the resulting `key_map` on `backbone._omni_native_lora_key_map`.
        If called after `LoRAInjector.inject(...)` has already wrapped layers with `.base_layer`,
        automatically normalizes `.base_layer` out of key names and weight targets.
        """
        cached = getattr(backbone, "_omni_native_lora_key_map", None)
        if isinstance(cached, dict) and cached:
            return cached

        target_obj = getattr(comfy_model, "model", comfy_model) if comfy_model is not None else None
        if target_obj is None:
            return None

        has_injected = any(isinstance(m, LoRALinear) for m in backbone.modules())

        try:
            import comfy.lora  # type: ignore

            orig_fn = getattr(
                comfy.lora,
                "_omni_orig_model_lora_keys_unet",
                getattr(comfy.lora, "model_lora_keys_unet", None),
            )
            if callable(orig_fn):
                raw_map = dict(orig_fn(target_obj, {}))
                if raw_map:
                    if has_injected:
                        clean_map: Dict[str, Any] = {}
                        for k, v in raw_map.items():
                            if ".lora_A" in k or ".lora_B" in k or "_lora_A" in k or "_lora_B" in k:
                                continue
                            ck = k.replace(".base_layer.", ".")
                            if ck.endswith(".base_layer"):
                                ck = ck[: -len(".base_layer")]
                            if isinstance(v, str):
                                if ".lora_A." in v or ".lora_B." in v:
                                    continue
                                cv: Any = v.replace(".base_layer.weight", ".weight").replace(".base_layer.", ".")
                            elif isinstance(v, tuple) and len(v) >= 2 and isinstance(v[0], str):
                                if ".lora_A." in v[0] or ".lora_B." in v[0]:
                                    continue
                                cv = (v[0].replace(".base_layer.weight", ".weight").replace(".base_layer.", "."),) + v[1:]
                            else:
                                cv = v
                            clean_map[ck] = cv
                        raw_map = clean_map
                    if raw_map:
                        setattr(backbone, "_omni_native_lora_key_map", raw_map)
                        return raw_map
        except Exception as exc:
            report_exception(
                "lora.capture_comfy_key_map",
                "Could not capture ComfyUI native LoRA key map; using offline fallback",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="lora",
            )
        return None

    @classmethod
    def extract_lora_state_dict(
        cls,
        model: nn.Module,
        prefix: str = "",
        comfy_model: Optional[Any] = None,
        export_comfy_qwen21_unfused_mlp: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract only the trainable LoRA A/B weights in ComfyUI `LoraLoader` compatible
        state_dict format by querying `comfy.lora.model_lora_keys_unet` directly:
        1. If `target_weight = "diffusion_model.<module>.weight"` has a direct 1-to-1 key in
           ComfyUI's `key_map` (such as `<module>` or `diffusion_model.<module>`), exports under
           that exact key without artificial splitting.
        2. If `target_weight` is only mapped via tuple slices `(target_weight, (dim, offset, length))`
           in `key_map`, slices `lora_up` along `dim` and exports under the exact slice keys in `key_map`.
        3. When `key_map` is unavailable (e.g., offline unit tests), emits `<module>` (or
           `<prefix>.<module>` if `prefix` is explicitly provided).
        """
        state_dict: Dict[str, torch.Tensor] = {}
        native_map = cls.capture_comfy_lora_key_map(model, comfy_model=comfy_model)

        # Build reverse lookup from "diffusion_model.<module>.weight" -> direct keys & tuple slice keys
        direct_keys_by_weight: Dict[str, List[str]] = {}
        slice_keys_by_weight: Dict[str, List[Tuple[str, Tuple[int, int, int]]]] = {}
        if isinstance(native_map, dict) and native_map:
            for k, v in native_map.items():
                if not isinstance(k, str):
                    continue
                if isinstance(v, str) and v.endswith(".weight"):
                    norm_v = v.replace(".base_layer.weight", ".weight")
                    direct_keys_by_weight.setdefault(norm_v, []).append(k)
                elif (
                    isinstance(v, tuple)
                    and len(v) >= 2
                    and isinstance(v[0], str)
                    and isinstance(v[1], tuple)
                    and len(v[1]) == 3
                ):
                    norm_v = v[0].replace(".base_layer.weight", ".weight")
                    dim_i, off_i, len_i = int(v[1][0]), int(v[1][1]), int(v[1][2])
                    slice_keys_by_weight.setdefault(norm_v, []).append((k, (dim_i, off_i, len_i)))

        for module_name, module in model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            norm_name = module_name.replace("__", ".")
            down_w = module.lora_A.weight.detach().cpu()
            up_w = module.lora_B.weight.detach().cpu()
            effective_alpha = (
                float(module.scaling) * float(module.rank)
                if getattr(module, "use_rslora", False)
                else float(module.alpha)
            )
            alpha_t = torch.tensor(effective_alpha, dtype=torch.float32)

            if export_comfy_qwen21_unfused_mlp and (
                norm_name.endswith(".img_mlp.gate_up") or norm_name == "img_mlp.gate_up"
            ):
                base_mlp = (
                    norm_name[: -len(".gate_up")]
                    if norm_name.endswith(".img_mlp.gate_up")
                    else norm_name[: -len("gate_up")]
                )
                half = up_w.shape[0] // 2
                gate_key = f"{base_mlp}.gate_layer" if base_mlp else "gate_layer"
                proj_key = f"{base_mlp}.proj" if base_mlp else "proj"
                if prefix:
                    gate_key = f"{prefix}.{gate_key}"
                    proj_key = f"{prefix}.{proj_key}"

                state_dict[f"{gate_key}.lora_down.weight"] = down_w.clone()
                state_dict[f"{gate_key}.lora_up.weight"] = up_w[:half].contiguous()
                state_dict[f"{gate_key}.alpha"] = alpha_t.clone()

                state_dict[f"{proj_key}.lora_down.weight"] = down_w.clone()
                state_dict[f"{proj_key}.lora_up.weight"] = up_w[half:].contiguous()
                state_dict[f"{proj_key}.alpha"] = alpha_t.clone()
                continue

            target_w = f"diffusion_model.{norm_name}.weight"
            direct_candidates = direct_keys_by_weight.get(target_w) or direct_keys_by_weight.get(f"{norm_name}.weight")
            slice_candidates = slice_keys_by_weight.get(target_w) or slice_keys_by_weight.get(f"{norm_name}.weight")

            chosen_direct_key: Optional[str] = None
            if direct_candidates:
                # Honor an explicit `prefix` first if requested; otherwise prefer `diffusion_model.<module_name>`
                # (which stock `ComfyUI/comfy/lora.py:205` registers for 100% of ComfyUI models), then `<module_name>`.
                if prefix == "lora_unet" and f"lora_unet_{norm_name.replace('.', '_')}" in direct_candidates:
                    chosen_direct_key = f"lora_unet_{norm_name.replace('.', '_')}"
                elif prefix and f"{prefix}.{norm_name}" in direct_candidates:
                    chosen_direct_key = f"{prefix}.{norm_name}"
                elif f"diffusion_model.{norm_name}" in direct_candidates:
                    chosen_direct_key = f"diffusion_model.{norm_name}"
                elif norm_name in direct_candidates:
                    chosen_direct_key = norm_name
                else:
                    non_unet = [c for c in direct_candidates if not c.startswith("lora_unet_")]
                    chosen_direct_key = non_unet[0] if non_unet else direct_candidates[0]

            if chosen_direct_key is not None:
                state_dict[f"{chosen_direct_key}.lora_down.weight"] = down_w
                state_dict[f"{chosen_direct_key}.lora_up.weight"] = up_w
                state_dict[f"{chosen_direct_key}.alpha"] = alpha_t
                continue

            if slice_candidates:
                # Group by unique (dim, offset, length) slice and pick the cleanest key name per slice
                by_slice: Dict[Tuple[int, int, int], List[str]] = {}
                for k_name, s_spec in slice_candidates:
                    by_slice.setdefault(s_spec, []).append(k_name)
                emitted_any = False
                for (s_dim, s_off, s_len), k_list in by_slice.items():
                    if s_dim < up_w.ndim and s_off >= 0 and s_len > 0 and s_off + s_len <= up_w.shape[s_dim]:
                        non_unet = [c for c in k_list if not c.startswith("lora_unet_") and not c.startswith("diffusion_model.")]
                        if not non_unet:
                            non_unet = [c for c in k_list if not c.startswith("lora_unet_")]
                        s_key = non_unet[0] if non_unet else k_list[0]
                        state_dict[f"{s_key}.lora_down.weight"] = down_w.clone()
                        state_dict[f"{s_key}.lora_up.weight"] = up_w.narrow(s_dim, s_off, s_len).contiguous()
                        state_dict[f"{s_key}.alpha"] = alpha_t.clone()
                        emitted_any = True
                if emitted_any:
                    continue

            # Fallback when `comfy.lora.model_lora_keys_unet` is not loaded in the current Python environment:
            # Stock ComfyUI's `model_lora_keys_unet` unconditionally maps `norm_name` (without `diffusion_model.`)
            # for every model in `comfy/lora.py`.
            if prefix == "lora_unet":
                clean_key = f"lora_unet_{norm_name.replace('.', '_')}"
            elif prefix:
                clean_key = f"{prefix}.{norm_name}"
            else:
                clean_key = norm_name
            state_dict[f"{clean_key}.lora_down.weight"] = down_w
            state_dict[f"{clean_key}.lora_up.weight"] = up_w
            state_dict[f"{clean_key}.alpha"] = alpha_t
        return state_dict

    @staticmethod
    def _strip_lora_key_prefix_and_suffix(raw_key: str) -> Tuple[str, str]:
        """
        Parse a LoRA state_dict key into `(clean_module_path, component)` where component is
        one of `'down'`, `'up'`, `'alpha'`, or `''`.
        """
        key = raw_key.strip()
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.", "unet.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.startswith("lora_unet_"):
            key = key[len("lora_unet_") :]

        down_suffixes = (".lora_down.weight", ".lora_A.weight", ".lora_linear_layer.down.weight", ".lora.down.weight")
        up_suffixes = (".lora_up.weight", ".lora_B.weight", ".lora_linear_layer.up.weight", ".lora.up.weight")

        for sfx in down_suffixes:
            if key.endswith(sfx):
                return key[: -len(sfx)], "down"
        for sfx in up_suffixes:
            if key.endswith(sfx):
                return key[: -len(sfx)], "up"
        if key.endswith(".alpha"):
            return key[: -len(".alpha")], "alpha"
        return key, ""

    @classmethod
    def inspect_lora_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Inspect an existing LoRA state_dict and optional safetensors metadata header to
        automatically infer `rank`, `alpha`, `steps_trained`, `model_architecture`, and
        discovered target module paths.
        """
        meta = metadata or {}
        inferred_rank: Optional[int] = None
        inferred_alpha: Optional[float] = None
        module_paths: List[str] = []

        for k, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            mod_path, comp = cls._strip_lora_key_prefix_and_suffix(k)
            if comp == "down":
                if mod_path and mod_path not in module_paths:
                    module_paths.append(mod_path)
                if inferred_rank is None and tensor.ndim >= 2:
                    inferred_rank = int(tensor.shape[0])
            elif comp == "alpha" and inferred_alpha is None:
                try:
                    inferred_alpha = float(tensor.detach().cpu().to(dtype=torch.float32).item())
                except (ValueError, TypeError, RuntimeError) as exc:
                    report_exception(
                        "lora.inspect_alpha_tensor",
                        "Could not read scalar alpha tensor from LoRA state dict",
                        exc,
                        level=logging.DEBUG,
                        once=True,
                        logger_name="lora",
                    )

        if inferred_rank is None:
            for rk_key in ("omni_lora.rank", "ss_network_dim", "lora_rank"):
                if rk_key in meta:
                    try:
                        inferred_rank = max(1, int(float(meta[rk_key])))
                        break
                    except (ValueError, TypeError) as exc:
                        report_exception(
                            "lora.inspect_meta_rank",
                            f"Could not parse LoRA rank metadata key '{rk_key}'",
                            exc,
                            level=logging.DEBUG,
                            once=True,
                            logger_name="lora",
                        )
        resolved_rank = inferred_rank if inferred_rank is not None and inferred_rank > 0 else 16

        meta_rslora = str(meta.get("omni_lora.use_rslora", "")).strip().lower() in ("true", "1", "yes")
        meta_alpha: Optional[float] = None
        for al_key in ("omni_lora.alpha", "ss_network_alpha", "lora_alpha"):
            if al_key in meta:
                try:
                    meta_alpha = float(meta[al_key])
                    break
                except (ValueError, TypeError) as exc:
                    report_exception(
                        "lora.inspect_meta_alpha",
                        f"Could not parse LoRA alpha metadata key '{al_key}'",
                        exc,
                        level=logging.DEBUG,
                        once=True,
                        logger_name="lora",
                    )

        if meta_alpha is not None and meta_alpha > 0:
            # If a checkpoint recorded the effective rsLoRA alpha (`nominal * sqrt(rank)`, e.g. 181.019 for rank=32)
            # in both `.alpha` tensor and `omni_lora.alpha`, recover the clean nominal alpha (32.0).
            if (
                meta_rslora
                and resolved_rank > 1
                and inferred_alpha is not None
                and abs(meta_alpha - inferred_alpha) < 1e-2
            ):
                unscaled = meta_alpha / math.sqrt(float(resolved_rank))
                if abs(unscaled - round(unscaled)) < 1e-2:
                    resolved_alpha = float(round(unscaled))
                else:
                    resolved_alpha = meta_alpha
            else:
                resolved_alpha = meta_alpha
        elif inferred_alpha is not None and inferred_alpha > 0:
            # If the checkpoint was saved with `use_rslora=True`, the `.alpha` tensor stores
            # `effective_alpha = nominal_alpha * sqrt(rank)` for ComfyUI LoraLoader compatibility.
            if meta_rslora and resolved_rank > 1:
                resolved_alpha = round(inferred_alpha / math.sqrt(float(resolved_rank)), 4)
            else:
                resolved_alpha = inferred_alpha
        else:
            resolved_alpha = float(resolved_rank)

        steps_trained = 0
        for st_key in ("omni_lora.steps", "ss_steps", "steps"):
            if st_key in meta:
                try:
                    steps_trained = max(0, int(float(meta[st_key])))
                    break
                except (ValueError, TypeError) as exc:
                    report_exception(
                        "lora.inspect_meta_steps",
                        f"Could not parse LoRA steps metadata key '{st_key}'",
                        exc,
                        level=logging.DEBUG,
                        once=True,
                        logger_name="lora",
                    )

        arch = meta.get("modelspec.architecture") or meta.get("omni_lora.architecture") or "auto"
        trigger_word = meta.get("omni_lora.trigger_word") or meta.get("ss_trigger_word") or ""

        return {
            "rank": resolved_rank,
            "alpha": resolved_alpha,
            "use_rslora": meta_rslora,
            "steps_trained": steps_trained,
            "model_architecture": arch,
            "trigger_word": trigger_word,
            "num_lora_modules": len(module_paths),
            "target_modules": module_paths,
        }

    @classmethod
    def load_lora_state_dict(
        cls,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        comfy_model: Optional[Any] = None,
    ) -> Dict[str, int]:
        """
        Load existing LoRA weights (`lora_down`/`lora_A`, `lora_up`/`lora_B`, `alpha`) into
        already-injected `LoRALinear` layers in `model` so training can seamlessly resume.
        Queries ComfyUI's `comfy.lora.model_lora_keys_unet` (`capture_comfy_lora_key_map`) to
        automatically recombine or slice any tuple-sliced weights `(target_weight, (dim, offset, length))`
        (such as `MOSSImage` `q/k/v -> qkv`, `Flux3` `q_proj/k_proj/v_proj -> qkv`, or `QwenImage21`
        `gate_layer/proj -> gate_up`), with an offline fallback when no `comfy_model` is attached.
        """
        parsed_entries: Dict[str, Dict[str, torch.Tensor]] = {}
        underscored_lookup: Dict[str, str] = {}

        for raw_key, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            mod_path, comp = cls._strip_lora_key_prefix_and_suffix(raw_key)
            if not mod_path or not comp:
                continue
            norm_path = mod_path.replace("__", ".")
            entry = parsed_entries.setdefault(norm_path, {})
            entry[comp] = tensor
            underscored_lookup[norm_path.replace(".", "_")] = norm_path

        native_map = cls.capture_comfy_lora_key_map(model, comfy_model=comfy_model)
        slice_keys_by_weight: Dict[str, List[Tuple[str, Tuple[int, int, int]]]] = {}
        slice_spec_by_key: Dict[str, Tuple[str, Tuple[int, int, int]]] = {}
        if isinstance(native_map, dict) and native_map:
            for k, v in native_map.items():
                if (
                    isinstance(k, str)
                    and isinstance(v, tuple)
                    and len(v) >= 2
                    and isinstance(v[0], str)
                    and isinstance(v[1], tuple)
                    and len(v[1]) == 3
                ):
                    norm_v = v[0].replace(".base_layer.weight", ".weight").removeprefix("diffusion_model.")
                    short_k = k.removeprefix("diffusion_model.")
                    s_spec = (int(v[1][0]), int(v[1][1]), int(v[1][2]))
                    slice_keys_by_weight.setdefault(norm_v, []).append((short_k, s_spec))
                    slice_spec_by_key[short_k] = (norm_v[: -len(".weight")] if norm_v.endswith(".weight") else norm_v, s_spec)

        loaded_layers = 0
        total_lora_layers = 0

        with torch.no_grad():
            for module_name, module in model.named_modules():
                if not isinstance(module, LoRALinear):
                    continue
                total_lora_layers += 1
                norm_name = module_name.replace("__", ".")
                entry = parsed_entries.get(norm_name)
                if entry is None:
                    alt_key = underscored_lookup.get(norm_name.replace(".", "_"))
                    if alt_key is not None:
                        entry = parsed_entries.get(alt_key)

                # 1. Native ComfyUI tuple-slice recombination (`slice_keys -> fused module`) or slicing (`fused -> slice`)
                if entry is None and slice_keys_by_weight:
                    slice_cands = slice_keys_by_weight.get(f"{norm_name}.weight", [])
                    if slice_cands:
                        matched_slices: List[Tuple[int, int, int, Dict[str, torch.Tensor]]] = []
                        seen_offsets: set[Tuple[int, int, int]] = set()
                        for s_key, (s_dim, s_off, s_len) in slice_cands:
                            if (s_dim, s_off, s_len) in seen_offsets:
                                continue
                            s_entry = parsed_entries.get(s_key)
                            if (
                                s_entry is not None
                                and isinstance(s_entry.get("down"), torch.Tensor)
                                and isinstance(s_entry.get("up"), torch.Tensor)
                            ):
                                seen_offsets.add((s_dim, s_off, s_len))
                                matched_slices.append((s_dim, s_off, s_len, s_entry))
                        if matched_slices:
                            matched_slices.sort(key=lambda item: item[1])
                            s_dim = matched_slices[0][0]
                            first_entry = matched_slices[0][3]
                            entry = {
                                "down": first_entry["down"],
                                "up": torch.cat([m[3]["up"] for m in matched_slices], dim=s_dim),
                            }
                            if "alpha" in first_entry:
                                entry["alpha"] = first_entry["alpha"]

                if entry is None and norm_name in slice_spec_by_key:
                    fused_mod_name, (s_dim, s_off, s_len) = slice_spec_by_key[norm_name]
                    fused_entry = parsed_entries.get(fused_mod_name)
                    if (
                        fused_entry is not None
                        and isinstance(fused_entry.get("down"), torch.Tensor)
                        and isinstance(fused_entry.get("up"), torch.Tensor)
                        and s_dim < fused_entry["up"].ndim
                        and s_off + s_len <= fused_entry["up"].shape[s_dim]
                    ):
                        entry = {
                            "down": fused_entry["down"],
                            "up": fused_entry["up"].narrow(s_dim, s_off, s_len).contiguous(),
                        }
                        if "alpha" in fused_entry:
                            entry["alpha"] = fused_entry["alpha"]

                # 2. Offline fallback when `comfy_model` is not attached
                if entry is None and norm_name.endswith(".img_mlp.gate_up"):
                    base_mlp = norm_name[: -len(".gate_up")]
                    gate_entry = parsed_entries.get(f"{base_mlp}.gate_layer")
                    proj_entry = parsed_entries.get(f"{base_mlp}.proj")
                    if (
                        gate_entry is not None
                        and proj_entry is not None
                        and isinstance(gate_entry.get("down"), torch.Tensor)
                        and isinstance(gate_entry.get("up"), torch.Tensor)
                        and isinstance(proj_entry.get("up"), torch.Tensor)
                    ):
                        entry = {
                            "down": gate_entry["down"],
                            "up": torch.cat([gate_entry["up"], proj_entry["up"]], dim=0),
                        }
                        if "alpha" in gate_entry:
                            entry["alpha"] = gate_entry["alpha"]
                elif entry is None and (
                    norm_name.endswith(".img_mlp.gate_layer") or norm_name.endswith(".img_mlp.proj")
                ):
                    is_gate_half = norm_name.endswith(".img_mlp.gate_layer")
                    sfx = ".gate_layer" if is_gate_half else ".proj"
                    base_mlp = norm_name[: -len(sfx)]
                    fused_entry = parsed_entries.get(f"{base_mlp}.gate_up")
                    if (
                        fused_entry is not None
                        and isinstance(fused_entry.get("down"), torch.Tensor)
                        and isinstance(fused_entry.get("up"), torch.Tensor)
                        and fused_entry["up"].ndim >= 2
                        and fused_entry["up"].shape[0] % 2 == 0
                    ):
                        half = fused_entry["up"].shape[0] // 2
                        sliced_up = fused_entry["up"][:half] if is_gate_half else fused_entry["up"][half:]
                        entry = {
                            "down": fused_entry["down"],
                            "up": sliced_up.contiguous(),
                        }
                        if "alpha" in fused_entry:
                            entry["alpha"] = fused_entry["alpha"]
                if entry is None:
                    continue

                down_t = entry.get("down")
                up_t = entry.get("up")
                if down_t is None or up_t is None:
                    continue

                target_a = module.lora_A.weight
                target_b = module.lora_B.weight

                if target_a.ndim == 2:
                    # Flatten 1x1 Conv2d/Conv3d LoRA weights ([r, in, 1, 1] -> [r, in]) when loading into a Linear LoRA
                    if down_t.ndim > 2:
                        down_t = down_t.reshape(down_t.shape[0], -1)
                    if up_t.ndim > 2:
                        up_t = up_t.reshape(up_t.shape[0], -1)
                else:
                    # Conv1d/Conv2d/Conv3d LoRA: preserve 3D/4D/5D kernel geometry ([r, in, *k] and [out, r, 1, ...])
                    if tuple(down_t.shape) != tuple(target_a.shape):
                        if down_t.numel() == target_a.numel() and down_t.shape[0] == target_a.shape[0]:
                            down_t = down_t.reshape(target_a.shape)
                        elif down_t.ndim == 2 and tuple(down_t.shape) == tuple(target_a.shape[:2]):
                            # Upgrade legacy 2D [r, in_c] conv LoRA by placing 1x1 weights at the center spatial tap
                            padded_down = torch.zeros(
                                target_a.shape,
                                dtype=down_t.dtype,
                                device=down_t.device,
                            )
                            center_idx = (slice(None), slice(None)) + tuple(k // 2 for k in target_a.shape[2:])
                            padded_down[center_idx] = down_t
                            down_t = padded_down
                    if tuple(up_t.shape) != tuple(target_b.shape):
                        if up_t.numel() == target_b.numel() and up_t.shape[:2] == target_b.shape[:2]:
                            up_t = up_t.reshape(target_b.shape)

                if tuple(down_t.shape) != tuple(target_a.shape) or tuple(up_t.shape) != tuple(target_b.shape):
                    continue

                target_a.copy_(safe_to_device(down_t.detach(), device=target_a.device, dtype=target_a.dtype))
                target_b.copy_(safe_to_device(up_t.detach(), device=target_b.device, dtype=target_b.dtype))

                alpha_t = entry.get("alpha")
                if isinstance(alpha_t, torch.Tensor):
                    try:
                        # In `.safetensors` state_dicts, `.alpha` is always stored as `effective_alpha = scaling * rank`
                        # (for stock ComfyUI `LoraLoader` compatibility). Divide by `rank` to recover exact `scaling`
                        # without compounding `sqrt(rank)` when `use_rslora=True`.
                        alpha_val = float(alpha_t.detach().cpu().to(dtype=torch.float32).item())
                        if alpha_val > 0:
                            module.scaling = alpha_val / float(max(1, module.rank))
                            module.alpha = (
                                module.scaling * math.sqrt(float(max(1, module.rank)))
                                if getattr(module, "use_rslora", False)
                                else alpha_val
                            )
                    except (ValueError, TypeError, RuntimeError) as exc:
                        report_exception(
                            "lora.load_alpha_tensor",
                            f"Could not apply alpha tensor for layer '{module_name}'",
                            exc,
                            level=logging.DEBUG,
                            once=True,
                            logger_name="lora",
                        )

                target_a.requires_grad_(True)
                target_b.requires_grad_(True)
                loaded_layers += 1

        return {
            "loaded_layers": loaded_layers,
            "total_lora_layers": total_lora_layers,
        }

    @staticmethod
    def get_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            # Re-enable requires_grad on LoRALinear weights if an external hook froze the module
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.weight.requires_grad_(True)
                    module.lora_B.weight.requires_grad_(True)
                    params.append(module.lora_A.weight)
                    params.append(module.lora_B.weight)
        return params

