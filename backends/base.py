"""
Unified Hardware Backend Abstraction Layer (ComfyUI-style model_management).

Decouples training pipelines and model adapters from vendor-specific APIs
(e.g., torch.cuda, xformers, nccl) so workloads run natively on Intel Arc XPU,
NVIDIA CUDA, AMD ROCm, Google Cloud TPU, Huawei Ascend NPU, Intel Gaudi HPU,
Apple Silicon MPS, and CPU.

Includes `FP32GradScaler` and `sanitize_module_no_fp64()` to guarantee zero
`torch.float64` (`.double()`) operations on GPUs that lack hardware FP64
(such as Intel Arc A770 `xpu` and Apple `mps`).
"""

from __future__ import annotations

import abc
import contextlib
import gc
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, ContextManager, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from ..diagnostics import report_exception


def query_comfy_device_memory_mb(device: torch.device) -> Optional[Tuple[float, float]]:
    """
    Query `(total_vram_mb, free_vram_mb)` via ComfyUI's `comfy.model_management` when running
    inside ComfyUI (supports CUDA, ROCm, XPU, MPS, and CPU).
    """
    try:
        import comfy.model_management as mm  # type: ignore

        total_bytes = float(mm.get_total_memory(device))
        free_bytes = float(mm.get_free_memory(device))
        if total_bytes > 0:
            total_mb = total_bytes / (1024**2)
            free_mb = max(0.0, min(total_mb, free_bytes / (1024**2)))
            return (round(total_mb, 1), round(free_mb, 1))
    except ImportError:
        return None
    except Exception as exc:
        report_exception(
            "backends.query_comfy_mem",
            "comfy.model_management memory query failed",
            exc,
            level=logging.DEBUG,
            once=True,
            logger_name="backends",
        )
    return None


def query_host_ram_mb() -> Tuple[float, float]:
    """
    Cross-platform query for `(total_ram_mb, available_ram_mb)` across Windows, Linux, and macOS.
    """
    # 1. ComfyUI model_management / psutil (bundled in standard ComfyUI environments)
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total_mb = float(vm.total) / (1024**2)
        avail_mb = float(vm.available) / (1024**2)
        if total_mb > 0:
            return (round(total_mb, 1), round(avail_mb, 1))
    except ImportError:
        pass
    except Exception as exc:
        report_exception(
            "backends.query_host_ram_psutil",
            "psutil.virtual_memory() failed",
            exc,
            level=logging.DEBUG,
            once=True,
            logger_name="backends",
        )

    comfy_cpu_mem = query_comfy_device_memory_mb(torch.device("cpu"))
    if comfy_cpu_mem is not None and comfy_cpu_mem[0] > 0:
        return comfy_cpu_mem

    # 2. Linux via /proc/meminfo
    if os.path.exists("/proc/meminfo"):
        try:
            mem_total_kb = 0.0
            mem_avail_kb = 0.0
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total_kb = float(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_avail_kb = float(line.split()[1])
            if mem_total_kb > 0:
                total_mb = mem_total_kb / 1024.0
                avail_mb = (mem_avail_kb / 1024.0) if mem_avail_kb > 0 else (total_mb * 0.75)
                return (round(total_mb, 1), round(avail_mb, 1))
        except (OSError, ValueError) as exc:
            report_exception(
                "backends.query_host_ram_proc",
                "Could not parse /proc/meminfo",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="backends",
            )

    # 3. POSIX sysconf (macOS / BSD / Linux fallback)
    try:
        if hasattr(os, "sysconf"):
            page_size = float(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = float(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and phys_pages > 0:
                total_mb = (page_size * phys_pages) / (1024**2)
                return (round(total_mb, 1), round(total_mb * 0.75, 1))
    except (OSError, ValueError) as exc:
        report_exception(
            "backends.query_host_ram_sysconf",
            "os.sysconf host RAM query failed",
            exc,
            level=logging.DEBUG,
            once=True,
            logger_name="backends",
        )

    return (32768.0, 24576.0)


FP8_DTYPES: Tuple[torch.dtype, ...] = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, name)
)


def is_fp8_dtype(dtype: Any) -> bool:
    """Return True if `dtype` is one of PyTorch's 8-bit floating-point dtypes."""
    return isinstance(dtype, torch.dtype) and dtype in FP8_DTYPES


def is_bitsandbytes_tensor(tensor: Any) -> bool:
    """
    Return True if `tensor` is a `bitsandbytes` quantized parameter (`Params4bit`, `Int8Params`)
    or carries `bitsandbytes` quantization state (`quant_state`, `CB`, `SCB`).
    """
    if not isinstance(tensor, torch.Tensor):
        return False
    cls_name = type(tensor).__name__
    mod_name = (getattr(type(tensor), "__module__", "") or "").lower()
    if cls_name in ("Params4bit", "Int8Params") or "bitsandbytes" in mod_name:
        return True
    if hasattr(tensor, "quant_state") or hasattr(tensor, "CB") or hasattr(tensor, "SCB"):
        return True
    return False


def is_bitsandbytes_module(module: nn.Module) -> bool:
    """
    Return True if `module` is a `bitsandbytes` quantized layer (`Linear4bit`, `Linear8bitLt`,
    `LinearFP4`, `LinearNF4`) or wraps a `bitsandbytes` quantized weight tensor.
    """
    cls_name = type(module).__name__
    mod_name = (getattr(type(module), "__module__", "") or "").lower()
    if "bitsandbytes" in mod_name or cls_name in (
        "Linear4bit",
        "Linear8bitLt",
        "LinearFP4",
        "LinearNF4",
    ):
        return True
    weight = getattr(module, "weight", None)
    return is_bitsandbytes_tensor(weight)


def probe_bitsandbytes_support(device: torch.device, backend_id: str) -> bool:
    """
    Return True ONLY if `bitsandbytes` is genuinely supported on the active accelerator.
    `bitsandbytes` relies on CUDA C++ kernels (`libbitsandbytes_cuda`) and is unsupported on
    Intel XPU (`xpu`), Apple MPS (`mps`), Ascend NPU (`npu`), Gaudi HPU (`hpu`), and CPU.
    """
    if backend_id != "cuda" or device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes as bnb  # type: ignore

        cext = getattr(bnb, "cextension", None)
        if cext is not None and not getattr(cext, "COMPILED_WITH_CUDA", False):
            return False
        return True
    except Exception:
        return False


def is_custom_quantized_tensor(tensor: Any) -> bool:
    """
    Return True if `tensor` is a quantized tensor or custom tensor subclass
    (e.g., ComfyUI `QuantizedTensor`, ComfyUI-GGUF `GGMLTensor`, bitsandbytes `Params4bit`/`Int8Params`, TorchAO).
    Such tensors must not be blindly cloned into plain `nn.Parameter(t.detach().clone())`
    as doing so strips quantization state (`_qdata`, `_params`, `tensor_type`, `tensor_shape`, `quant_state`).
    """
    if not isinstance(tensor, torch.Tensor):
        return False
    if getattr(tensor, "is_quantized", False) or is_bitsandbytes_tensor(tensor):
        return True
    if hasattr(tensor, "_qdata") or hasattr(tensor, "tensor_type") or hasattr(tensor, "tensor_shape"):
        return True
    if type(tensor) not in (torch.Tensor, nn.Parameter):
        return True
    return False


def get_tensor_storage_bytes(tensor: Any, elem_size_override: Optional[int] = None) -> int:
    """
    Return the actual physical storage footprint in bytes of `tensor`.
    For wrapper tensor subclasses such as ComfyUI's `QuantizedTensor` (`comfy/quant_ops.py` /
    `comfy_kitchen`), `tensor.numel() * tensor.element_size()` returns the uncompressed logical
    `bfloat16` size (2 bytes/elem) while the actual compressed payload lives in `tensor._qdata`
    (1-byte INT8 / FP8 or 0.5-byte packed W4A4) and `tensor._params`.
    """
    if not isinstance(tensor, torch.Tensor):
        return 0
    if elem_size_override is not None:
        return int(tensor.numel()) * int(elem_size_override)
    qdata = getattr(tensor, "_qdata", None)
    if isinstance(qdata, torch.Tensor):
        total_b = int(qdata.numel() * qdata.element_size())
        params_obj = getattr(tensor, "_params", None)
        if params_obj is not None:
            if isinstance(params_obj, dict):
                for pv in params_obj.values():
                    if isinstance(pv, torch.Tensor):
                        total_b += int(pv.numel() * pv.element_size())
            else:
                for pv in getattr(params_obj, "__dict__", {}).values():
                    if isinstance(pv, torch.Tensor):
                        total_b += int(pv.numel() * pv.element_size())
        return max(1, total_b)
    return int(tensor.numel() * tensor.element_size())


def move_custom_quantized_tensor(tensor: torch.Tensor, dest_device: torch.device) -> torch.Tensor:
    """
    Move a custom quantized tensor (such as ComfyUI `QuantizedTensor` or ComfyUI-GGUF `GGMLTensor`)
    to `dest_device` while preserving all custom metadata attributes (`_qdata`, `_params`,
    `tensor_type`, `tensor_shape`, `patches`, `quant_state`, `orig_shape`).
    """
    qdata = getattr(tensor, "_qdata", None)
    cur_dev = qdata.device if isinstance(qdata, torch.Tensor) else tensor.device
    if cur_dev == dest_device and tensor.device == dest_device:
        return tensor
    moved = tensor.to(device=dest_device)
    for attr_k, attr_v in getattr(tensor, "__dict__", {}).items():
        if not hasattr(moved, attr_k):
            try:
                if isinstance(attr_v, torch.Tensor) and attr_v.device != dest_device:
                    setattr(moved, attr_k, attr_v.to(device=dest_device))
                else:
                    setattr(moved, attr_k, attr_v)
            except (AttributeError, TypeError, RuntimeError) as exc:
                report_exception(
                    "backends.move_custom_quant_attr",
                    f"Could not preserve custom quantized tensor attribute '{attr_k}' on move",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends",
                )
    return moved


def assign_module_parameter(module: nn.Module, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """
    Assign `tensor` to `module` under `name`, supporting both `nn.Parameter` and custom
    `torch.Tensor` subclasses (such as ComfyUI-GGUF `GGMLTensor` and ComfyUI `QuantizedTensor`)
    that do not subclass `nn.Parameter`. Calling `setattr(module, 'weight', custom_tensor)` directly
    raises `TypeError: cannot assign '...' as parameter 'weight' (torch.nn.Parameter or None expected)`
    in `nn.Module.__setattr__`; writing to `module._parameters[name]` (matching PyTorch's
    own `nn.Module._apply`) preserves the custom tensor subclass on the target device.
    """
    params = getattr(module, "_parameters", None)
    if isinstance(params, dict) and name in params:
        if isinstance(tensor, nn.Parameter) or is_custom_quantized_tensor(tensor):
            params[name] = tensor  # type: ignore[assignment]
            return tensor
        wrapped = nn.Parameter(tensor, requires_grad=False)
        params[name] = wrapped
        return wrapped
    buffers = getattr(module, "_buffers", None)
    if isinstance(buffers, dict) and name in buffers:
        buffers[name] = tensor
        return tensor
    setattr(module, name, tensor)
    return getattr(module, name)


def detect_model_weight_format(
    module: Optional[nn.Module],
    active_quant_override: Optional[str] = None,
) -> str:
    """
    Inspect `module` parameters to report the actual base weight storage format
    (e.g., `gguf (q8_0)`, `quantized (int8/w4a4)`, `fp8_e4m3fn`, `int8`, `bf16`, `fp16`, `fp32`).
    """
    if active_quant_override:
        return active_quant_override
    if module is None:
        return "bf16"

    fallback_dt: Optional[str] = None
    gguf_fallback: Optional[str] = None
    quant_layouts: List[str] = []
    for p in module.parameters():
        if p.requires_grad:
            continue
        t_type = getattr(p, "tensor_type", None)
        if t_type is not None:
            t_name = getattr(t_type, "name", str(t_type))
            if t_name and not t_name.startswith("torch.") and not t_name.isdigit():
                label = f"gguf ({t_name.lower()})"
                if t_name.upper() not in ("F32", "F16", "BF16"):
                    return label
                if gguf_fallback is None:
                    gguf_fallback = label
            elif gguf_fallback is None:
                gguf_fallback = "gguf"
            continue
        cls_lower = type(p).__name__.lower()
        if "ggml" in cls_lower or "gguf" in cls_lower:
            if gguf_fallback is None:
                gguf_fallback = "gguf"
            continue
        qdata = getattr(p, "_qdata", None)
        if isinstance(qdata, torch.Tensor) or "quantized" in cls_lower:
            layout_obj = getattr(p, "_layout", getattr(p, "layout", getattr(p, "layout_type", None)))
            l_name = ""
            if layout_obj is not None:
                l_cls = layout_obj.__name__ if isinstance(layout_obj, type) else type(layout_obj).__name__
                l_str = str(layout_obj).lower()
                if "w4a4" in l_str or "w4a4" in l_cls.lower():
                    l_name = "w4a4"
                elif "int8" in l_str or "int8" in l_cls.lower():
                    l_name = "int8"
                elif "fp4" in l_str or "fp4" in l_cls.lower():
                    l_name = "fp4"
                elif "fp8" in l_str or "fp8" in l_cls.lower():
                    l_name = "fp8"
            if not l_name and isinstance(qdata, torch.Tensor):
                if qdata.dtype == torch.int8:
                    l_name = "int8"
                elif is_fp8_dtype(qdata.dtype):
                    l_name = "fp8"
                elif qdata.dtype == torch.uint8:
                    l_name = "int4/uint8"
            l_name = l_name or "mixed"
            if l_name not in quant_layouts:
                quant_layouts.append(l_name)
            continue
        if is_bitsandbytes_tensor(p):
            return "bitsandbytes"
        if is_fp8_dtype(p.dtype):
            return "fp8_e5m2" if "e5m2" in str(p.dtype) else "fp8_e4m3fn"
        if p.dtype == torch.int8:
            return "int8"
        if fallback_dt is None and p.ndim >= 2:
            if p.dtype == torch.bfloat16:
                fallback_dt = "bf16"
            elif p.dtype == torch.float16:
                fallback_dt = "fp16"
            elif p.dtype == torch.float32:
                fallback_dt = "fp32"
    if quant_layouts:
        return f"quantized ({'/'.join(quant_layouts[:2])})"
    return gguf_fallback or fallback_dt or "bf16"


def probe_device_fp8_support(device: torch.device) -> bool:
    """
    Probe whether `device` supports allocating `torch.float8_e4m3fn` tensors and casting
    them to 16-bit compute precision (`bfloat16`/`float16`) for dequantization.
    """
    fp8_dt = getattr(torch, "float8_e4m3fn", None)
    if fp8_dt is None:
        return False
    try:
        t = torch.ones((2, 2), device=device, dtype=torch.float32).to(dtype=fp8_dt)
        _ = t.to(dtype=torch.float16)
        return True
    except Exception:
        return False


def quantize_weight_fp8_or_int8(
    weight: torch.Tensor,
    mode: str,
    dest_device: torch.device,
    supports_fp8: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically quantize a frozen floating-point weight tensor (`bf16`/`fp16`/`fp32`) into
    1-byte per-channel scaled FP8 (`float8_e4m3fn` / `float8_e5m2`) or 1-byte per-channel
    symmetric INT8 (`int8_qlora`) using strictly FP32 math (zero FP64 operations).
    Returns `(quantized_weight_on_dest_device, scale_fp32_on_dest_device)`.
    """
    req = (mode or "").lower()
    with torch.no_grad():
        w_fp32 = weight.detach().to(dtype=torch.float32)
        if w_fp32.ndim >= 2:
            reduce_dims = tuple(range(1, w_fp32.ndim))
            amax = w_fp32.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
        else:
            amax = w_fp32.abs().amax().clamp(min=1e-8)

        want_fp8 = req in ("fp8", "fp8_e4m3fn", "fp8_e5m2") and supports_fp8 and len(FP8_DTYPES) > 0
        if want_fp8:
            fp8_dt = (
                getattr(torch, "float8_e5m2", FP8_DTYPES[0])
                if "e5m2" in req
                else getattr(torch, "float8_e4m3fn", FP8_DTYPES[0])
            )
            fp8_max = 57344.0 if "e5m2" in req else 448.0
            scale = (amax / fp8_max).to(dtype=torch.float32)
            scaled = (w_fp32 / scale).clamp(-fp8_max, fp8_max)
            try:
                q_weight = scaled.to(device=dest_device, dtype=fp8_dt)
                q_scale = scale.to(device=dest_device, dtype=torch.float32)
                return q_weight, q_scale
            except RuntimeError as exc:
                # Automatic fallback to universal 1-byte INT8 QLoRA if device driver rejects FP8 tensor
                report_exception(
                    "backends.quantize_fp8_fallback_int8",
                    f"FP8 weight quantization rejected on {dest_device}; falling back to INT8 QLoRA",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends",
                )

        scale = (amax / 127.0).to(dtype=torch.float32)
        q_int8 = torch.clamp(torch.round(w_fp32 / scale), -127.0, 127.0).to(dtype=torch.int8)
        return q_int8.to(device=dest_device), scale.to(device=dest_device, dtype=torch.float32)


def is_inference_tensor(tensor: Any) -> bool:
    """Return True if `tensor` was created inside `torch.inference_mode()` and has no autograd version counter."""
    if not isinstance(tensor, torch.Tensor):
        return False
    try:
        if tensor.is_inference():
            return True
    except (RuntimeError, AttributeError) as exc:
        report_exception(
            "backends.is_inference_probe",
            "tensor.is_inference() check failed; falling back to _version probe",
            exc,
            level=logging.DEBUG,
            once=True,
            logger_name="backends",
        )
    try:
        _ = tensor._version
        return False
    except RuntimeError:
        return True


def _same_device(d1: torch.device, d2: torch.device) -> bool:
    """Compare two `torch.device` objects, treating `index=None` and `index=0` as equivalent."""
    if d1.type != d2.type:
        return False
    if d1.type == "cpu":
        return True
    return (d1.index or 0) == (d2.index or 0)


def _safe_tensor_storage_id(tensor: torch.Tensor) -> int:
    """
    Return a unique integer identifier for `tensor`'s underlying storage without crashing
    on tensors that do not expose a linear host/device storage pointer (such as `meta`
    or storage-less custom tensor subclasses).
    """
    dev_type = getattr(tensor.device, "type", "cpu")
    if dev_type == "meta":
        return id(tensor)
    try:
        return int(tensor.data_ptr())
    except Exception:
        return id(tensor)


def safe_to_device(
    tensor: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    """
    Vendor-safe replacement for `tensor.to(device=..., dtype=...)`:
    1. Prevents accidental `torch.float64` transfers onto accelerators without hardware FP64.
    2. When moving from CPU to an accelerator with a dtype conversion (`tensor.dtype != target_dt`),
       casts on CPU FIRST (`tensor.to(dtype=target_dt)`) before copying to `target_dev`.
    3. When moving from an accelerator to CPU with a dtype conversion, moves to CPU FIRST
       (`tensor.to(device="cpu")`) before casting on CPU.
    """
    target_dev = (
        device
        if isinstance(device, torch.device)
        else (torch.device(device) if device is not None else tensor.device)
    )
    target_dt = dtype if isinstance(dtype, torch.dtype) else tensor.dtype

    if target_dt == torch.float64:
        target_dt = torch.float32

    if _same_device(tensor.device, target_dev) and tensor.dtype == target_dt:
        return tensor

    if tensor.device.type == "cpu" and target_dev.type != "cpu" and tensor.dtype != target_dt:
        return tensor.to(dtype=target_dt).to(device=target_dev, non_blocking=non_blocking)

    if tensor.device.type != "cpu" and target_dev.type == "cpu" and tensor.dtype != target_dt:
        return tensor.to(device=target_dev, non_blocking=non_blocking).to(dtype=target_dt)

    return tensor.to(device=target_dev, dtype=target_dt, non_blocking=non_blocking)


def sanitize_module_no_fp64(module: nn.Module) -> nn.Module:
    """
    Ensure every parameter and buffer in `module`:
    1. Is converted from `torch.float64` to `torch.float32` (preventing fp64 kernel crashes on XPU/MPS).
    2. Is NOT an inference tensor (`param.is_inference() == True`) for small/non-linear layers that run
       standard PyTorch autograd `SavedVariable` paths (such as 1D LayerNorm/RMSNorm weights, biases,
       or Conv/Embedding layers), while NEVER cloning large 2D `weight` matrices on CPU or GPU so
       ComfyUI's `fast_disk=True` `safetensors` memory-mapped (`mmap`) weights are never copied into
       anonymous Windows pagefile heap RAM.
    3. Records any replaced CPU parameter or buffer in `_omni_orig_params` / `_omni_orig_buffers` so
       `LoRAInjector.uninject()` restores the exact original tensor references after training.
    """
    with torch.inference_mode(False):
        for sub in module.modules():
            sub_cls_name = sub.__class__.__name__
            for name, param in list(sub.named_parameters(recurse=False)):
                if is_custom_quantized_tensor(param):
                    continue
                is_inf = is_inference_tensor(param)

                if param.dtype == torch.float64:
                    if param.device.type == "cpu":
                        orig_map: Dict[str, Tuple[torch.Tensor, bool]] = getattr(sub, "_omni_orig_params", {})
                        if name not in orig_map:
                            orig_map[name] = (param, bool(param.requires_grad))
                            sub._omni_orig_params = orig_map  # type: ignore[attr-defined]
                    new_p = nn.Parameter(
                        param.detach().to(dtype=torch.float32).clone(),
                        requires_grad=param.requires_grad,
                    )
                    setattr(sub, name, new_p)
                elif is_inf:
                    # Never clone large frozen 2D Linear weights (on CPU mmap or GPU VRAM or FP8/INT8):
                    # - LoRA-wrapped layers execute via `_StreamedLinearFn` (which never saves `weight` in `SavedVariable`
                    #   and streams CPU `mmap` weights directly into a single reusable staging buffer).
                    # - Keeping CPU 2D weights untouched preserves ComfyUI's `fast_disk=True` OS file-backed `mmap`
                    #   instead of allocating gigabytes of anonymous pagefile RAM.
                    is_large_frozen_2d = (
                        not param.requires_grad
                        and param.ndim == 2
                        and name == "weight"
                        and (param.numel() >= 65536 or is_fp8_dtype(param.dtype) or param.dtype == torch.int8)
                        and sub_cls_name != "Embedding"
                    )
                    if not is_large_frozen_2d:
                        if param.device.type == "cpu":
                            orig_map = getattr(sub, "_omni_orig_params", {})
                            if name not in orig_map:
                                orig_map[name] = (param, bool(param.requires_grad))
                                sub._omni_orig_params = orig_map  # type: ignore[attr-defined]
                        new_p = nn.Parameter(
                            param.detach().clone(),
                            requires_grad=param.requires_grad,
                        )
                        setattr(sub, name, new_p)

            for name, buf in list(sub.named_buffers(recurse=False)):
                if buf is not None and not is_custom_quantized_tensor(buf):
                    is_inf = is_inference_tensor(buf)
                    if is_inf or buf.dtype == torch.float64:
                        if buf.device.type == "cpu":
                            orig_bufs: Dict[str, torch.Tensor] = getattr(sub, "_omni_orig_buffers", {})
                            if name not in orig_bufs:
                                orig_bufs[name] = buf
                                sub._omni_orig_buffers = orig_bufs  # type: ignore[attr-defined]
                        new_dt = torch.float32 if buf.dtype == torch.float64 else buf.dtype
                        setattr(sub, name, buf.detach().to(dtype=new_dt).clone())
    return module


def trim_process_working_set() -> None:
    """
    Reclaim cyclic Python references and release unreferenced tensor storage without
    evicting active process working-set pages.
    """
    try:
        gc.collect()
    except Exception as exc:
        report_exception(
            "backends.trim_working_set_gc",
            "gc.collect() failed in trim_process_working_set",
            exc,
            level=logging.DEBUG,
            once=True,
            logger_name="backends",
        )



class FP32GradScaler:
    """
    Pure-FP32 Dynamic Loss Scaler for FP16 mixed-precision training.

    Why this exists:
    PyTorch's built-in `torch.amp.GradScaler.unscale_()` executes:
      `self._scale.double().reciprocal().float()`
    on the target device (`grad_scaler.py:352`), which allocates a `torch.float64`
    tensor on the GPU and crashes on Intel Arc XPU (`RuntimeError: Required aspect
    fp64 is not supported on the device`) and Apple MPS.

    `FP32GradScaler` implements the exact same dynamic loss-scaling state machine
    using strictly Python floats and `torch.float32` operations—never `fp64`.
    """

    def __init__(
        self,
        init_scale: float = 1024.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self._scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self._growth_tracker = 0
        self._found_inf = False
        self._unscaled_optimizers: set[int] = set()

    def is_enabled(self) -> bool:
        return self.enabled

    def get_scale(self) -> float:
        return self._scale if self.enabled else 1.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return loss
        # Multiply by an explicit float32 scalar created on CPU first—never float64
        scale_t = torch.tensor(float(self._scale), dtype=torch.float32, device="cpu")
        if loss.device.type != "cpu":
            scale_t = scale_t.to(device=loss.device)
        return loss.to(dtype=torch.float32) * scale_t

    @staticmethod
    def _iter_opt_params(optimizer: torch.optim.Optimizer) -> Iterable[nn.Parameter]:
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    yield param

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.enabled:
            return
        opt_id = id(optimizer)
        if opt_id in self._unscaled_optimizers:
            return
        self._unscaled_optimizers.add(opt_id)

        inv_scale = float(1.0 / max(self._scale, 1e-12))
        found_inf = False

        with torch.no_grad():
            for param in self._iter_opt_params(optimizer):
                grad = param.grad
                if grad is None:
                    continue
                if grad.dtype == torch.float64:
                    grad = grad.to(dtype=torch.float32)
                # Unscale in float32 without ever touching float64 (.double())
                grad_fp32 = grad.to(dtype=torch.float32) * inv_scale
                if not bool(torch.isfinite(grad_fp32).all().item()):
                    found_inf = True
                param.grad.copy_(grad_fp32.to(dtype=grad.dtype))

        self._found_inf = self._found_inf or found_inf

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.enabled:
            optimizer.step()
            return

        if id(optimizer) not in self._unscaled_optimizers:
            self.unscale_(optimizer)

        if not self._found_inf:
            optimizer.step()

    def update(self, new_scale: Optional[float] = None) -> None:
        if not self.enabled:
            return

        if new_scale is not None:
            self._scale = float(new_scale)
        elif self._found_inf:
            self._scale = max(1.0, self._scale * self.backoff_factor)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self._scale = min(32768.0, self._scale * self.growth_factor)
                self._growth_tracker = 0

        self._found_inf = False
        self._unscaled_optimizers.clear()


@dataclass
class DeviceCapabilities:
    """Hardware capabilities queried from the active PyTorch backend."""

    backend_name: str
    device_type: str
    device_name: str
    device_index: int
    total_vram_mb: float
    supports_bf16: bool
    supports_fp16: bool
    supports_tf32: bool
    supports_fp64: bool = False
    supports_fp8: bool = False
    sdpa_backends: List[str] = field(default_factory=list)
    distributed_backend: str = "gloo"
    driver_info: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Current and peak memory telemetry in megabytes (MB)."""

    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    total_vram_mb: float
    free_vram_mb: float = 0.0

    @property
    def utilization_pct(self) -> float:
        if self.total_vram_mb <= 0:
            return 0.0
        return round((self.reserved_mb / self.total_vram_mb) * 100.0, 2)


class HardwareBackend(abc.ABC):
    """Abstract base class for all PyTorch hardware accelerator backends."""

    def __init__(self, device_index: int = 0, dtype: str = "bf16") -> None:
        torch.set_default_dtype(torch.float32)
        self.device_index = device_index
        req = (dtype or "bf16").lower()
        self.base_weight_quantization: Optional[str] = (
            req if req in ("fp8", "fp8_e4m3fn", "fp8_e5m2", "int8_qlora") else None
        )
        self.dtype_name = "bf16" if self.base_weight_quantization is not None else req
        self.torch_dtype = self.resolve_dtype(self.dtype_name)
        if self.torch_dtype == torch.bfloat16:
            try:
                if self.is_available() and not self.get_capabilities().supports_bf16:
                    self.dtype_name = "fp16"
                    self.torch_dtype = torch.float16
            except Exception as exc:
                report_exception(
                    "backends.init_bf16_probe",
                    "BF16 capability check failed during backend init; keeping requested dtype",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends",
                )

    @property
    @abc.abstractmethod
    def backend_id(self) -> str:
        """Short identifier, e.g., 'xpu', 'cuda', 'rocm', 'mps', 'cpu'."""

    @property
    @abc.abstractmethod
    def device_type(self) -> str:
        """PyTorch device type string passed to torch.device and torch.amp.autocast."""

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Return True if this hardware backend is available in the current PyTorch build."""

    @abc.abstractmethod
    def get_device(self) -> torch.device:
        """Return the active torch.device instance."""

    @abc.abstractmethod
    def get_capabilities(self) -> DeviceCapabilities:
        """Probe hardware capabilities, VRAM capacity, and supported dtypes."""

    @abc.abstractmethod
    def synchronize(self) -> None:
        """Wait for all kernels on the active stream/queue to complete."""

    @abc.abstractmethod
    def empty_cache(self) -> None:
        """Release cached allocator blocks back to the device driver."""

    @abc.abstractmethod
    def get_memory_snapshot(self) -> MemorySnapshot:
        """Return current allocated, reserved, and peak memory in MB."""

    @abc.abstractmethod
    def manual_seed_all(self, seed: int) -> None:
        """Seed both CPU and accelerator random number generators deterministically."""

    def configure_environment(self) -> Dict[str, str]:
        """Set optimal runtime environment flags and enforce FP32 default dtype (no FP64)."""
        torch.set_default_dtype(torch.float32)
        return {}

    @contextlib.contextmanager
    def autocast_context(self, enabled: bool = True) -> Any:
        """
        Return a vendor-agnostic `torch.amp.autocast` context manager.
        Safely resolves the active device type (`cpu` when a backend runs in offline fallback mode)
        and falls back to a no-op context if the active PyTorch build does not support autocast
        for `(device_type, dtype)`.
        """
        active_dev_type = self.get_device().type
        if (
            not enabled
            or self.torch_dtype == torch.float32
            or (active_dev_type == "cpu" and self.torch_dtype != torch.bfloat16)
        ):
            yield
            return

        cm = None
        entered = False
        try:
            cm = torch.amp.autocast(
                device_type=active_dev_type,
                dtype=self.torch_dtype,
                enabled=True,
            )
            cm.__enter__()
            entered = True
        except Exception:
            entered = False

        try:
            yield
        finally:
            if entered and cm is not None:
                cm.__exit__(*sys.exc_info())

    def create_grad_scaler(self, enabled: bool = False) -> Optional[FP32GradScaler]:
        """
        Return an `FP32GradScaler` only when training in `fp16`.
        Note:
          1. `bf16` has the same 8-bit exponent range as `fp32` and never needs a GradScaler.
          2. When `fp16` IS used, `FP32GradScaler` avoids PyTorch `GradScaler.unscale_()`'s
             internal `self._scale.double()` call, which crashes on Intel Arc XPU (`fp64` unsupported).
        """
        if not enabled or self.torch_dtype != torch.float16:
            return None
        return FP32GradScaler(enabled=True)

    def get_free_vram_mb(self) -> float:
        """
        Query live free VRAM in MB across any vendor:
        1. Prefers ComfyUI's `comfy.model_management.get_free_memory(device)` when running inside ComfyUI.
        2. Falls back to backend `get_memory_snapshot().free_vram_mb`.
        3. Estimates from `total_vram_mb - max(allocated_mb, reserved_mb)`.
        """
        if self.is_available():
            comfy_mem = query_comfy_device_memory_mb(self.get_device())
            if comfy_mem is not None and comfy_mem[1] > 0:
                return comfy_mem[1]
        snap = self.get_memory_snapshot()
        if snap.free_vram_mb > 0:
            return snap.free_vram_mb
        total = snap.total_vram_mb if snap.total_vram_mb > 0 else self.get_capabilities().total_vram_mb
        used = max(snap.allocated_mb, snap.reserved_mb)
        return round(max(0.0, total - used), 1)

    @staticmethod
    def _inspect_module_memory_profile(
        module: Optional[nn.Module],
        device: torch.device,
        target_dtype: torch.dtype,
        quantize_base_weights: bool = False,
    ) -> Dict[str, float]:
        """
        Inspect `module` to measure exact model & block graph memory characteristics:
        - `non_base_mb`: non-LoRA backbone norms/embeddings + trainable `lora_A`/`lora_B` parameters
        - `trainable_mb`: trainable parameter footprint in storage dtype
        - `trainable_numel`: exact number of trainable scalar elements
        - `base_layers_total_mb`: total size of all frozen `LoRALinear.base_layer` parameters
        - `max_single_layer_mb`: 2-byte compute-dtype size of the single largest `base_layer` weight
        - `max_hidden_dim`: largest linear/conv feature width in the backbone
        - `model_dim`: primary hidden state width `D_model` of the repeatable transformer blocks
        - `block_count`: number of repeatable transformer/UNet blocks
        - `peak_block_elems_per_token`: exact per-token saved activation + backward matmul elements for 1 block
        - `peak_single_proj_dim`: largest feature dimension inside a repeatable block (e.g. 16384 for FFN up/gate)
        - `already_on_device_mb`: bytes of `module` already allocated on `device`
        """
        if module is None:
            return {
                "non_base_mb": 256.0,
                "trainable_mb": 64.0,
                "trainable_numel": float(32 * 1024 * 1024),
                "base_layers_total_mb": 0.0,
                "max_single_layer_mb": 96.0,
                "max_hidden_dim": 3072.0,
                "model_dim": 3072.0,
                "block_count": 36.0,
                "peak_block_elems_per_token": 0.0,
                "peak_single_proj_dim": 3072.0,
                "already_on_device_mb": 0.0,
            }

        base_param_ids: set[int] = set()
        base_layers_bytes = 0
        max_single_layer_bytes = 0
        max_hidden_dim = 0
        already_on_device_bytes = 0
        seen_ptrs: set[int] = set()
        in_dim_counts: Dict[int, int] = {}

        for sub in module.modules():
            if sub.__class__.__name__ == "LoRALinear" and hasattr(sub, "base_layer"):
                base = sub.base_layer
                layer_bytes = 0
                for p in base.parameters():
                    base_param_ids.add(id(p))
                    if (
                        quantize_base_weights
                        and not is_custom_quantized_tensor(p)
                        and p.ndim >= 2
                        and p.dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16)
                    ):
                        pb = get_tensor_storage_bytes(p, elem_size_override=1)
                    elif (
                        not is_custom_quantized_tensor(p)
                        and p.dtype in (torch.float32, torch.float64)
                        and target_dtype in (torch.bfloat16, torch.float16)
                    ):
                        pb = get_tensor_storage_bytes(p, elem_size_override=2)
                    else:
                        pb = get_tensor_storage_bytes(p)
                    layer_bytes += pb
                    t_shape = getattr(p, "tensor_shape", None)
                    if isinstance(t_shape, (tuple, list)) and len(t_shape) >= 2:
                        d_out, d_in = int(t_shape[0]), int(t_shape[1])
                        max_hidden_dim = max(max_hidden_dim, d_out, d_in)
                        if d_in >= 256:
                            in_dim_counts[d_in] = in_dim_counts.get(d_in, 0) + 1
                    elif p.ndim >= 2:
                        d_out, d_in = int(p.shape[0]), int(p.shape[1])
                        max_hidden_dim = max(max_hidden_dim, d_out, d_in)
                        if d_in >= 256:
                            in_dim_counts[d_in] = in_dim_counts.get(d_in, 0) + 1
                for b in base.buffers():
                    if b is not None:
                        base_param_ids.add(id(b))
                base_layers_bytes += layer_bytes
                # Staging buffer always materializes in 2-byte compute dtype (bf16/fp16)
                deq_layer_bytes = max(layer_bytes, sum(p.numel() * 2 for p in base.parameters()))
                max_single_layer_bytes = max(max_single_layer_bytes, deq_layer_bytes)

        non_base_bytes = 0
        trainable_bytes = 0
        trainable_numel = 0
        for p in module.parameters():
            qdata = getattr(p, "_qdata", None)
            ptr_src = qdata if isinstance(qdata, torch.Tensor) else p
            ptr = _safe_tensor_storage_id(ptr_src)
            if ptr not in seen_ptrs and _same_device(ptr_src.device, device):
                seen_ptrs.add(ptr)
                already_on_device_bytes += get_tensor_storage_bytes(p)
            if p.ndim >= 2:
                max_hidden_dim = max(max_hidden_dim, int(p.shape[0]), int(p.shape[1]))
            if id(p) in base_param_ids:
                continue
            if (
                not p.requires_grad
                and not is_custom_quantized_tensor(p)
                and (p.dtype == torch.float32 or is_fp8_dtype(p.dtype))
                and target_dtype in (torch.bfloat16, torch.float16)
            ):
                pb = get_tensor_storage_bytes(p, elem_size_override=2)
            else:
                pb = get_tensor_storage_bytes(p)
            non_base_bytes += pb
            if p.requires_grad:
                trainable_bytes += pb
                trainable_numel += int(p.numel())

        for b in module.buffers():
            if b is None:
                continue
            qdata = getattr(b, "_qdata", None)
            ptr_src = qdata if isinstance(qdata, torch.Tensor) else b
            ptr = _safe_tensor_storage_id(ptr_src)
            if ptr not in seen_ptrs and _same_device(ptr_src.device, device):
                seen_ptrs.add(ptr)
                already_on_device_bytes += get_tensor_storage_bytes(b)
            if id(b) not in base_param_ids:
                non_base_bytes += get_tensor_storage_bytes(b)

        # Primary hidden dimension `D_model` is the most frequent input projection width across blocks
        model_dim = (
            max(in_dim_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if in_dim_counts
            else min(3072, max_hidden_dim or 3072)
        )

        # Inspect repeatable block lists to compute exact single-block backward recomputation footprint
        block_count = 0
        peak_block_elems_per_token = 0.0
        peak_single_proj_dim = float(model_dim)
        covered_blocks: set[int] = set()
        for _mod_name, sub_mod in module.named_modules():
            if isinstance(sub_mod, nn.ModuleList) and len(sub_mod) >= 2:
                if any(id(b) in covered_blocks for b in sub_mod):
                    continue
                for b in sub_mod:
                    covered_blocks.add(id(b))
                block_count += len(sub_mod)
                # Inspect up to the first 4 distinct block types in this ModuleList (handles both homogeneous DiTs and heterogeneous UNets)
                seen_block_sigs: set[tuple[str, int]] = set()
                for sample_block in sub_mod:
                    sig = (sample_block.__class__.__name__, len(list(sample_block.children())))
                    if sig in seen_block_sigs and len(seen_block_sigs) >= 4:
                        continue
                    seen_block_sigs.add(sig)
                    b_in_sum = 0
                    b_out_sum = 0
                    b_rank_sum = 0
                    b_max_pair = 0
                    for leaf in sample_block.modules():
                        if leaf.__class__.__name__ == "LoRALinear":
                            d_in = int(getattr(leaf, "in_features", 0) or 0)
                            d_out = int(getattr(leaf, "out_features", 0) or 0)
                            r_lora = int(getattr(leaf, "rank", 0) or 0)
                            if d_in > 0 and d_out > 0:
                                b_in_sum += d_in
                                b_out_sum += d_out
                                b_rank_sum += r_lora
                                b_max_pair = max(b_max_pair, d_in + d_out)
                                peak_single_proj_dim = max(peak_single_proj_dim, float(d_in), float(d_out))
                    if b_in_sum > 0:
                        b_elems = float(b_in_sum + b_rank_sum) + 0.5 * float(b_out_sum) + float(b_max_pair)
                        peak_block_elems_per_token = max(peak_block_elems_per_token, b_elems)

        return {
            "non_base_mb": non_base_bytes / (1024**2),
            "trainable_mb": trainable_bytes / (1024**2),
            "trainable_numel": float(trainable_numel),
            "base_layers_total_mb": base_layers_bytes / (1024**2),
            "max_single_layer_mb": max(32.0, max_single_layer_bytes / (1024**2)),
            "max_hidden_dim": float(max_hidden_dim or 3072),
            "model_dim": float(model_dim or 3072),
            "block_count": float(block_count or 36),
            "peak_block_elems_per_token": float(peak_block_elems_per_token),
            "peak_single_proj_dim": float(max(peak_single_proj_dim, float(max_hidden_dim or 3072))),
            "already_on_device_mb": already_on_device_bytes / (1024**2),
        }

    def compute_vram_budget(
        self,
        vram_mode: str = "auto",
        resolution: int = 1024,
        batch_size: int = 1,
        module: Optional[nn.Module] = None,
        reserve_vram_gb: float = 0.0,
        optimizer_type: str = "radamw",
        gradient_checkpointing: bool = True,
        has_control_latents: bool = False,
    ) -> Dict[str, float]:
        """
        Compute a deterministic, graph-inspected VRAM budget across any GPU size (6 GB to 80 GB+)
        and any vendor (`cuda`, `rocm`, `xpu`, `tpu`, `mps`, `npu`, `hpu`, `cpu`).
        Replaces arbitrary percentages with a bottom-up memory ledger:
        1. `usable_vram_mb`: Hardware capacity minus OS/WDDM/CUDA driver context reserve and `reserve_vram_gb`.
        2. `optimizer_allowance_mb`: Exact LoRA gradients (`p.grad`) + FP32 optimizer state buffers
           (`exp_avg`/`exp_avg_sq` for AdamW/RAdamW, single buffer for Lion/SGD, factored for Adafactor, 4 buffers for Prodigy).
        3. `activation_allowance_mb`: Exact single-block backward recomputation saved activations + peak
           backward matmul (`grad_out` + `grad_in`) + non-checkpointed outer stem/loss tensors for token count `S`.
        4. `ckpt_inputs_mb`: Checkpointed block input snapshots (`2` in-flight blocks when CPU-offloaded,
           or `block_count` blocks when kept on GPU on 24+ GB cards).
        5. `allocator_workspace_mb`: cuBLAS/SDPA/oneDNN kernel workspace + caching allocator pool headroom
           scaled by the block's single largest projection tensor (`peak_single_act_mb`).
        """
        caps = self.get_capabilities()
        device = self.get_device()
        comfy_mem = query_comfy_device_memory_mb(device) if self.is_available() else None
        total_mb = (
            float(comfy_mem[0])
            if (comfy_mem is not None and comfy_mem[0] > 0)
            else (float(caps.total_vram_mb) if caps.total_vram_mb > 0 else 16384.0)
        )
        mode = (vram_mode or "auto").lower()
        reserve_user_mb = max(0.0, float(reserve_vram_gb or 0.0) * 1024.0)

        if (
            self.backend_id == "cpu"
            or mode in ("offload", "offload_vram", "novram", "cpu")
            or (total_mb > 0 and reserve_user_mb >= total_mb * 0.85)
        ):
            return {
                "total_vram_mb": round(total_mb, 1),
                "free_vram_mb": round(total_mb, 1),
                "usable_vram_mb": round(max(0.0, total_mb * 0.85 - reserve_user_mb), 1),
                "reserved_user_vram_mb": round(reserve_user_mb, 1),
                "activation_allowance_mb": 0.0,
                "optimizer_allowance_mb": 0.0,
                "ckpt_inputs_mb": 0.0,
                "allocator_workspace_mb": 0.0,
                "offload_ckpt_inputs": 1.0,
                "headroom_mb": round(total_mb, 1),
                "max_resident_model_mb": 0.0,
            }

        profile = self._inspect_module_memory_profile(
            module,
            device,
            self.torch_dtype,
            quantize_base_weights=(self.base_weight_quantization is not None),
        )
        free_mb = self.get_free_vram_mb() + profile["already_on_device_mb"]
        free_mb = min(total_mb, max(0.0, free_mb))

        # 1. Hardware Available Ceiling (OS / WDDM paging reserve / CUDA context reserve)
        if self.backend_id in ("xpu", "mps"):
            os_driver_reserve_mb = max(768.0, total_mb * 0.125)
        else:
            os_driver_reserve_mb = max(512.0, total_mb * 0.085)
        spec_usable_mb = max(512.0, total_mb - os_driver_reserve_mb)
        base_usable_mb = min(spec_usable_mb, max(512.0, free_mb - 384.0)) if free_mb > 512.0 else spec_usable_mb
        usable_mb = max(512.0, base_usable_mb - reserve_user_mb)

        # 2. Exact Optimizer + Gradient Footprint (from trainable_numel & optimizer state buffers in FP32)
        trainable_mb = float(profile.get("trainable_mb", 64.0))
        trainable_numel = float(profile.get("trainable_numel", 0.0))
        if trainable_numel <= 0.0 and trainable_mb > 0.0:
            trainable_numel = (trainable_mb * (1024.0**2)) / 2.0
        opt_name = (optimizer_type or "radamw").lower()
        needs_fp32_master = self.torch_dtype in (torch.float16, torch.bfloat16)
        if opt_name == "adafactor":
            # Adafactor (beta1=0, factored row/col FP32 2nd moments + stochastic rounding): ~0.5 B/elem
            state_bytes_per_elem = 0.5
        elif opt_name == "sgd":
            # SGD with momentum: 1 FP32 momentum buffer (4 B/elem) + FP32 master param & unscale copy if fp16 (+6 B/elem)
            state_bytes_per_elem = 10.0 if needs_fp32_master else 4.0
        else:
            # RAdamW / AdamW: 2 FP32 moment buffers (`exp_avg` + `exp_avg_sq` = 8 B/elem)
            # + FP32 `master_param` (4 B/elem) + FP32 grad unscale/step buffer (2 B/elem) when running in fp16
            state_bytes_per_elem = 14.0 if needs_fp32_master else 8.0
        opt_state_mb = (trainable_numel * state_bytes_per_elem) / (1024.0**2)
        optimizer_allowance_mb = max(32.0, trainable_mb + opt_state_mb + 16.0)

        # 3. Exact Single-Block Peak Recomputed Activation + Outer Stem Footprint for Sequence Length `S`
        eff_batch = max(1, int(batch_size))
        img_tokens = max(256, int(math.ceil(float(resolution) / 16.0)) ** 2)
        # Paired image-editing models (`reference_latents`) concatenate reference tokens alongside
        # target image tokens, roughly doubling visual sequence length through joint attention blocks.
        ctrl_tok_mult = 2.0 if bool(has_control_latents) else 1.0
        seq_tokens = float(eff_batch * (img_tokens * ctrl_tok_mult + 256))
        compute_elem_bytes = 4.0 if self.torch_dtype == torch.float32 else 2.0
        max_hidden_dim = float(profile.get("max_hidden_dim", 3072.0) or 3072.0)
        model_dim = float(profile.get("model_dim", 0.0) or min(3072.0, max_hidden_dim))
        peak_block_elems = float(profile.get("peak_block_elems_per_token", 0.0) or 0.0)
        peak_single_proj_dim = float(profile.get("peak_single_proj_dim", 0.0) or max_hidden_dim)

        if peak_block_elems > 0.0:
            single_block_recompute_mb = (seq_tokens * peak_block_elems * compute_elem_bytes) / (1024.0**2)
        else:
            est_elems_per_token = model_dim * 12.0 + max_hidden_dim * 4.0
            single_block_recompute_mb = (seq_tokens * est_elems_per_token * compute_elem_bytes) / (1024.0**2)
        outer_stem_mb = (seq_tokens * model_dim * 6.0 * compute_elem_bytes) / (1024.0**2) + 32.0
        activation_allowance_mb = single_block_recompute_mb + outer_stem_mb

        # 4. Checkpointed Block Input Snapshots (`ckpt_inputs_mb`)
        block_count = max(1.0, float(profile.get("block_count", 36.0) or 36.0))
        per_block_input_mb = (seq_tokens * model_dim * compute_elem_bytes) / (1024.0**2)
        ckpt_inputs_gpu_mb = block_count * per_block_input_mb
        ckpt_inputs_cpu_mb = 2.0 * per_block_input_mb

        # 5. Allocator Pool & Kernel Workspace (`allocator_workspace_mb`)
        peak_single_act_mb = (seq_tokens * max(model_dim, peak_single_proj_dim) * compute_elem_bytes) / (1024.0**2)
        staging_buffer_mb = float(profile.get("max_single_layer_mb", 96.0))
        total_model_mb = float(profile.get("non_base_mb", 0.0)) + float(profile.get("base_layers_total_mb", 0.0))

        if mode in ("low", "low_vram", "lowvram") or (mode == "auto" and total_mb < 12000.0):
            offload_ckpt_inputs = True
            ckpt_inputs_mb = ckpt_inputs_cpu_mb
            allocator_workspace_mb = max(
                600.0 if total_mb <= 6144.0 else 900.0,
                450.0 + 4.0 * peak_single_act_mb,
            )
            base_headroom_mb = allocator_workspace_mb + activation_allowance_mb * 0.85 + optimizer_allowance_mb + ckpt_inputs_mb
        elif mode in ("high", "high_vram", "highvram") or (mode == "auto" and total_mb >= 24000.0):
            allocator_workspace_mb = max(1600.0, 1024.0 + 6.0 * peak_single_act_mb)
            headroom_gpu_ckpt = allocator_workspace_mb + activation_allowance_mb + optimizer_allowance_mb + ckpt_inputs_gpu_mb
            if total_model_mb > 0.0 and (total_model_mb + headroom_gpu_ckpt) <= usable_mb:
                offload_ckpt_inputs = False
                ckpt_inputs_mb = ckpt_inputs_gpu_mb
            else:
                offload_ckpt_inputs = True
                ckpt_inputs_mb = ckpt_inputs_cpu_mb
            base_headroom_mb = allocator_workspace_mb + activation_allowance_mb + optimizer_allowance_mb + ckpt_inputs_mb
        else:
            # NORMAL_VRAM (12 GB - 23.9 GB, including 14.56 GiB T4 and 16 GB GPUs):
            # Offload checkpointed block input snapshots to CPU RAM (`ckpt_inputs_cpu_mb`) so 40-60 blocks
            # never consume 1.1+ GB of GPU VRAM, and scale allocator pool workspace by `peak_single_act_mb`.
            # On CUDA / ROCm with < 15.5 GB VRAM (e.g. 14.56 GiB Tesla T4 running in fp16 + GradScaler),
            # `cudaMallocAsync` requires ~1.4 GB of additional unfragmented pool headroom so the driver
            # never stalls trimming the pool mid-backward.
            offload_ckpt_inputs = True
            ckpt_inputs_mb = ckpt_inputs_cpu_mb
            if self.backend_id in ("xpu", "mps"):
                allocator_workspace_mb = max(2450.0, 1750.0 + 6.5 * peak_single_act_mb)
            elif total_mb < 15600.0 or needs_fp32_master:
                allocator_workspace_mb = max(3600.0, 2650.0 + 10.0 * peak_single_act_mb)
            else:
                allocator_workspace_mb = max(2500.0, 1850.0 + 8.0 * peak_single_act_mb)
            base_headroom_mb = allocator_workspace_mb + activation_allowance_mb + optimizer_allowance_mb + ckpt_inputs_mb

        needs_streaming_buffer = (
            module is None
            or mode in ("low", "low_vram", "lowvram")
            or total_model_mb <= 0.0
            or total_model_mb > (usable_mb - base_headroom_mb)
        )
        headroom_mb = base_headroom_mb + (staging_buffer_mb if needs_streaming_buffer else 0.0)

        floor_resident_mb = 0.0 if (reserve_user_mb > 0.0 or total_mb <= 8192.0) else 512.0
        max_resident_mb = max(floor_resident_mb, usable_mb - headroom_mb)
        return {
            "total_vram_mb": round(total_mb, 1),
            "free_vram_mb": round(free_mb, 1),
            "usable_vram_mb": round(usable_mb, 1),
            "reserved_user_vram_mb": round(reserve_user_mb, 1),
            "activation_allowance_mb": round(activation_allowance_mb, 1),
            "optimizer_allowance_mb": round(optimizer_allowance_mb, 1),
            "ckpt_inputs_mb": round(ckpt_inputs_mb, 1),
            "allocator_workspace_mb": round(allocator_workspace_mb, 1),
            "offload_ckpt_inputs": 1.0 if offload_ckpt_inputs else 0.0,
            "headroom_mb": round(headroom_mb + reserve_user_mb, 1),
            "max_resident_model_mb": round(max_resident_mb, 1),
        }

    def validate_quantized_model_compatibility(self, module: nn.Module) -> None:
        """
        Inspect `module` for external `bitsandbytes` layers (`Linear4bit`, `Linear8bitLt`, `Params4bit`,
        `Int8Params`) and fail fast with a clear, actionable `ValueError` on unsupported backends
        (such as Intel Arc XPU `xpu`, Apple `mps`, Ascend `npu`, Gaudi `hpu`, or `cpu`,
        or on `cuda` when `bitsandbytes` CUDA C++ binaries are unavailable/broken).
        """
        bnb_layers: List[Tuple[str, str]] = []
        for mod_name, sub in module.named_modules():
            if sub.__class__.__name__ == "LoRALinear":
                continue
            if is_bitsandbytes_module(sub):
                w = getattr(sub, "weight", None)
                cls_label = f"{type(sub).__name__}({type(w).__name__})" if w is not None else type(sub).__name__
                bnb_layers.append((mod_name or "<root>", cls_label))

        if not bnb_layers:
            return

        device = self.get_device()
        if not probe_bitsandbytes_support(device, self.backend_id):
            first_path, first_cls = bnb_layers[0]
            raise ValueError(
                f"[OmniLoRA] Unsupported bitsandbytes quantized model detected ({len(bnb_layers)} layers, "
                f"e.g. '{first_path}': {first_cls}) on backend '{self.backend_id}' (device '{device}'). "
                f"bitsandbytes (4-bit NF4 / FP4 / LLM.int8) requires CUDA C++ kernels and is not supported on "
                f"{self.backend_id.upper()} (including Intel Arc XPU, Apple MPS, Ascend NPU, Gaudi HPU, and CPU). "
                f"To train in 8-bit on {self.backend_id.upper()}, load a standard BF16/FP16, FP8, or GGUF model and set "
                f"OmniLoRA's 'precision' to 'fp8_e4m3fn' or 'int8_qlora' (which uses 100% pure-PyTorch quantization "
                f"with zero bitsandbytes dependency)."
            )

    def enforce_vram_budget(
        self,
        module: nn.Module,
        vram_mode: str = "auto",
        resolution: int = 1024,
        batch_size: int = 1,
        reserve_vram_gb: float = 0.0,
        optimizer_type: str = "radamw",
        gradient_checkpointing: bool = True,
        has_control_latents: bool = False,
    ) -> Dict[str, float]:
        """
        Enforce a safe VRAM resident weight ceiling across `module`:
        1. Validates that any external `bitsandbytes` layers in `module` are supported by the active backend.
        2. Keeps all trainable LoRA adapters (`lora_A`, `lora_B`) and non-LoRA modules
           (norms, embeddings, input/output projections) on the active accelerator in `self.torch_dtype`.
           Any non-`base_layer` FP8 weights (`img_in`, `txt_in`, `final_layer`) are dequantized to
           `self.torch_dtype` so standard PyTorch layers never fail on `Float8` matmuls.
        3. Supports both pre-loaded FP8 / GGUF checkpoints AND on-the-fly base weight quantization when
           `self.base_weight_quantization` is `"fp8_e4m3fn"`, `"fp8_e5m2"`, or `"int8_qlora"`, cutting
           16-bit frozen weights by 50% in pure PyTorch (zero `bitsandbytes` dependency).
        4. Keeps frozen `LoRALinear.base_layer` weights on the accelerator up to `max_resident_model_mb`,
           and moves any excess frozen base weights to CPU so they stream via `_StreamedLinearFn` without
           exhausting VRAM or fragmenting the allocator.
        """
        self.validate_quantized_model_compatibility(module)
        module = sanitize_module_no_fp64(module)
        device = self.get_device()
        target_dtype = self.torch_dtype
        caps = self.get_capabilities()
        budget = self.compute_vram_budget(
            vram_mode=vram_mode,
            resolution=resolution,
            batch_size=batch_size,
            module=module,
            reserve_vram_gb=reserve_vram_gb,
            optimizer_type=optimizer_type,
            gradient_checkpointing=gradient_checkpointing,
            has_control_latents=has_control_latents,
        )
        max_resident_bytes = int(budget["max_resident_model_mb"] * (1024**2))

        use_pinned_cpu = self.backend_id in ("cuda", "rocm", "npu")
        resident_bytes = 0
        offloaded_bytes = 0
        quantized_layer_count = 0
        base_layer_param_ids: set[int] = set()

        # Identify all LoRALinear wrappers and their frozen base_layer parameters
        lora_wrappers: List[nn.Module] = []
        initial_dev: Optional[torch.device] = None
        for sub in module.modules():
            if sub.__class__.__name__ == "LoRALinear" and hasattr(sub, "base_layer"):
                lora_wrappers.append(sub)
                for p in sub.base_layer.parameters():
                    base_layer_param_ids.add(id(p))
                    if initial_dev is None and p.device.type != "meta":
                        initial_dev = p.device
                for b in sub.base_layer.buffers():
                    base_layer_param_ids.add(id(b))
        if initial_dev is None:
            for p in module.parameters():
                if p.device.type != "meta":
                    initial_dev = p.device
                    break
        if not hasattr(module, "_omni_initial_device"):
            module._omni_initial_device = initial_dev or torch.device("cpu")  # type: ignore[attr-defined]

        # MiniMax H3 / H3-Image checkpoints intentionally keep output heads (especially PDD offset
        # banks), patchifiers, time_embedder, and curve-form AdaLN basis projections in float32.
        # Keep those small islands in float32 so training matches inference numerically without
        # affecting VRAM placement for any other architecture.
        fp32_preserve_sub_ids: set[int] = set()
        if getattr(module.__class__, "__module__", "").startswith("comfy.ldm.minimax."):
            for attr_name in ("video_patch_proj", "audio_patch_proj"):
                sub_m = getattr(module, attr_name, None)
                if isinstance(sub_m, nn.Module):
                    fp32_preserve_sub_ids.add(id(sub_m))
            t_emb_mod = getattr(module, "time_embedder", None)
            if isinstance(t_emb_mod, nn.Module):
                for desc in t_emb_mod.modules():
                    fp32_preserve_sub_ids.add(id(desc))
            fl_mod = getattr(module, "final_layer", None)
            if isinstance(fl_mod, nn.Module):
                for head_attr in ("video_out", "audio_out"):
                    head_m = getattr(fl_mod, head_attr, None)
                    if isinstance(head_m, nn.Module):
                        fp32_preserve_sub_ids.add(id(head_m))
            if bool(getattr(module, "use_adaln_curves", False)):
                for desc in module.modules():
                    adaln_m = getattr(desc, "adaln_proj", None)
                    if isinstance(adaln_m, nn.Module):
                        lin_m = getattr(adaln_m, "linear", None)
                        if isinstance(lin_m, nn.Module):
                            fp32_preserve_sub_ids.add(id(lin_m))

        # 1. Move all non-base_layer parameters & buffers (including lora_A, lora_B, norms, embeddings) to device
        with torch.inference_mode(False):
            for sub in module.modules():
                if sub.__class__.__name__ == "LoRALinear":
                    for name, param in list(sub.named_parameters(recurse=False)):
                        lp_dt = (
                            target_dtype
                            if (param.dtype == torch.bfloat16 and not caps.supports_bf16)
                            else param.dtype
                        )
                        if not _same_device(param.device, device) or param.dtype != lp_dt or is_inference_tensor(param):
                            moved_lp = safe_to_device(param.detach(), device=device, dtype=lp_dt)
                            if is_inference_tensor(moved_lp):
                                moved_lp = moved_lp.clone()
                            setattr(sub, name, nn.Parameter(moved_lp, requires_grad=param.requires_grad))
                            param = getattr(sub, name)
                        if device.type != "cpu":
                            resident_bytes += param.numel() * param.element_size()
                    continue

                for name, param in list(sub.named_parameters(recurse=False)):
                    if id(param) in base_layer_param_ids:
                        continue
                    if is_custom_quantized_tensor(param):
                        try:
                            if not _same_device(param.device, device):
                                orig_map: Dict[str, Tuple[torch.Tensor, bool]] = getattr(sub, "_omni_orig_params", {})
                                if name not in orig_map and param.device.type == "cpu":
                                    orig_map[name] = (param, bool(param.requires_grad))
                                    sub._omni_orig_params = orig_map  # type: ignore[attr-defined]
                                moved_p = move_custom_quantized_tensor(param, device)
                                param = assign_module_parameter(sub, name, moved_p)
                        except Exception as exc:
                            report_exception(
                                "backends.budget_move_nonbase_quant_param",
                                f"Could not move custom quantized parameter '{name}' to {device}",
                                exc,
                                level=logging.WARNING,
                                once=True,
                                logger_name="backends",
                            )
                        p_bytes = get_tensor_storage_bytes(param)
                        if param.device.type == device.type and device.type != "cpu":
                            resident_bytes += p_bytes
                        elif device.type != "cpu":
                            offloaded_bytes += p_bytes
                        continue

                    if is_fp8_dtype(param.dtype):
                        # Save original FP8 parameter on CPU so `LoRAInjector.uninject` restores the exact
                        # FP8 weight (and any `fast_disk=True` mmap backing) during post-training inference.
                        orig_map = getattr(sub, "_omni_orig_params", {})
                        if name not in orig_map:
                            orig_ref = param if param.device.type == "cpu" else param.detach().to(device="cpu")
                            orig_map[name] = (orig_ref, bool(param.requires_grad))
                            sub._omni_orig_params = orig_map  # type: ignore[attr-defined]
                        # Non-LoRA layers (e.g. img_in, txt_in, final_layer) run standard PyTorch forward
                        # and must be dequantized from FP8 to `target_dtype` (bf16/fp16) to avoid matmul dtype errors.
                        w_deq = safe_to_device(param.detach(), device=device, dtype=torch.float32)
                        scale_t = getattr(sub, "scale_weight", getattr(sub, "weight_scale", None))
                        if isinstance(scale_t, torch.Tensor) and name == "weight":
                            s_dev = safe_to_device(scale_t.detach(), device=device, dtype=torch.float32)
                            if s_dev.ndim == 1 and w_deq.ndim >= 2 and s_dev.numel() == w_deq.shape[0]:
                                s_dev = s_dev.view(-1, *([1] * (w_deq.ndim - 1)))
                            w_deq = w_deq * s_dev
                        w_deq_cast = safe_to_device(w_deq, device=device, dtype=target_dtype)
                        if is_inference_tensor(w_deq_cast):
                            w_deq_cast = w_deq_cast.clone()
                        new_p = nn.Parameter(w_deq_cast, requires_grad=param.requires_grad)
                        setattr(sub, name, new_p)
                        param = new_p
                    else:
                        if param.dtype == torch.bfloat16 and not caps.supports_bf16:
                            dt = target_dtype
                        elif param.dtype == torch.float32 and id(sub) in fp32_preserve_sub_ids:
                            dt = torch.float32
                        elif (
                            not param.requires_grad
                            and param.dtype in (torch.float32, torch.float16, torch.bfloat16)
                            and param.dtype != target_dtype
                        ):
                            dt = target_dtype
                        else:
                            dt = param.dtype
                        is_inf_p = is_inference_tensor(param)
                        if not _same_device(param.device, device) or param.dtype != dt or is_inf_p:
                            orig_map = getattr(sub, "_omni_orig_params", {})
                            if name not in orig_map:
                                orig_ref = param if param.device.type == "cpu" else param.detach().to(device="cpu")
                                orig_map[name] = (orig_ref, bool(param.requires_grad))
                                sub._omni_orig_params = orig_map  # type: ignore[attr-defined]
                            moved_t = safe_to_device(param.detach(), device=device, dtype=dt)
                            if is_inference_tensor(moved_t):
                                moved_t = moved_t.clone()
                            new_p = nn.Parameter(
                                moved_t,
                                requires_grad=param.requires_grad,
                            )
                            setattr(sub, name, new_p)
                            param = new_p
                    if device.type != "cpu":
                        resident_bytes += get_tensor_storage_bytes(param)

                for name, buf in list(sub.named_buffers(recurse=False)):
                    if buf is None or id(buf) in base_layer_param_ids:
                        continue
                    if is_custom_quantized_tensor(buf):
                        try:
                            if not _same_device(buf.device, device):
                                orig_bufs: Dict[str, torch.Tensor] = getattr(sub, "_omni_orig_buffers", {})
                                if name not in orig_bufs and buf.device.type == "cpu":
                                    orig_bufs[name] = buf
                                    sub._omni_orig_buffers = orig_bufs  # type: ignore[attr-defined]
                                buf = assign_module_parameter(sub, name, move_custom_quantized_tensor(buf, device))
                        except Exception as exc:
                            report_exception(
                                "backends.budget_move_nonbase_quant_buf",
                                f"Could not move custom quantized buffer '{name}' to {device}",
                                exc,
                                level=logging.WARNING,
                                once=True,
                                logger_name="backends",
                            )
                        if buf.device.type == device.type and device.type != "cpu":
                            resident_bytes += get_tensor_storage_bytes(buf)
                        continue
                    if buf.dtype == torch.float64:
                        dt = torch.float32
                    elif buf.dtype == torch.bfloat16 and not caps.supports_bf16:
                        dt = target_dtype
                    else:
                        dt = buf.dtype
                    is_inf_b = is_inference_tensor(buf)
                    if not _same_device(buf.device, device) or buf.dtype != dt or is_inf_b:
                        orig_bufs = getattr(sub, "_omni_orig_buffers", {})
                        if name not in orig_bufs and buf.device.type == "cpu":
                            orig_bufs[name] = buf
                            sub._omni_orig_buffers = orig_bufs  # type: ignore[attr-defined]
                        new_b = safe_to_device(buf.detach(), device=device, dtype=dt)
                        if is_inference_tensor(new_b):
                            new_b = new_b.clone()
                        setattr(sub, name, new_b)
                        buf = new_b
                    if device.type != "cpu":
                        resident_bytes += get_tensor_storage_bytes(buf)

        # 2. Allocate frozen LoRALinear.base_layer weights on device up to max_resident_bytes; offload overflow to CPU
        max_streamed_numel = 0
        bytes_since_sync = 0
        sync_chunk_bytes = 256 * 1024 * 1024  # Flush Level Zero copy queue every 256 MB so WDDM host DMA staging never bloats RAM
        with torch.inference_mode(False):
            for wrapper in lora_wrappers:
                base = wrapper.base_layer
                layer_bytes = 0
                for p in base.parameters():
                    if (
                        self.base_weight_quantization is not None
                        and not is_custom_quantized_tensor(p)
                        and p.ndim >= 2
                        and p.dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16)
                    ):
                        layer_bytes += get_tensor_storage_bytes(p, elem_size_override=1)
                    elif (
                        not is_custom_quantized_tensor(p)
                        and p.dtype in (torch.float32, torch.float64)
                        and target_dtype in (torch.bfloat16, torch.float16)
                    ):
                        layer_bytes += get_tensor_storage_bytes(p, elem_size_override=2)
                    else:
                        layer_bytes += get_tensor_storage_bytes(p)

                place_on_gpu = (device.type != "cpu") and ((resident_bytes + layer_bytes) <= max_resident_bytes)
                dest_device = device if place_on_gpu else torch.device("cpu")

                for b_sub in base.modules():
                    for pname, param in list(b_sub.named_parameters(recurse=False)):
                        if is_custom_quantized_tensor(param):
                            try:
                                if param.device != dest_device:
                                    orig_map = getattr(b_sub, "_omni_orig_params", {})
                                    if pname not in orig_map and param.device.type == "cpu":
                                        orig_map[pname] = (param, bool(param.requires_grad))
                                        b_sub._omni_orig_params = orig_map  # type: ignore[attr-defined]
                                    moved_p = move_custom_quantized_tensor(param, dest_device)
                                    param = assign_module_parameter(b_sub, pname, moved_p)
                            except Exception as exc:
                                report_exception(
                                    "backends.budget_move_base_quant_param",
                                    f"Could not move base custom quantized parameter '{pname}' to {dest_device}",
                                    exc,
                                    level=logging.WARNING,
                                    once=True,
                                    logger_name="backends",
                                )
                            p_bytes = get_tensor_storage_bytes(param)
                            if param.device.type == device.type and device.type != "cpu":
                                resident_bytes += p_bytes
                            elif device.type != "cpu":
                                offloaded_bytes += p_bytes
                            continue

                        # On-the-fly FP8 / INT8 QLoRA quantization of 2D floating-point frozen base weights
                        if (
                            self.base_weight_quantization is not None
                            and pname == "weight"
                            and param.ndim >= 2
                            and param.dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16)
                        ):
                            orig_map = getattr(b_sub, "_omni_orig_params", {})
                            if pname not in orig_map:
                                orig_ref = param if param.device.type == "cpu" else param.detach().to(device="cpu")
                                orig_map[pname] = (orig_ref, bool(param.requires_grad))
                                b_sub._omni_orig_params = orig_map  # type: ignore[attr-defined]
                            orig_dt = target_dtype if param.dtype in (torch.float32, torch.float64) else param.dtype
                            q_weight, q_scale = quantize_weight_fp8_or_int8(
                                param,
                                mode=self.base_weight_quantization,
                                dest_device=dest_device,
                                supports_fp8=caps.supports_fp8 or dest_device.type == "cpu",
                            )
                            if not place_on_gpu and use_pinned_cpu and not q_weight.is_pinned():
                                try:
                                    q_weight = q_weight.pin_memory()
                                except RuntimeError as exc:
                                    report_exception(
                                        "backends.budget_pin_quant_weight",
                                        "pin_memory() failed for quantized CPU-streamed weight; using pageable RAM",
                                        exc,
                                        level=logging.DEBUG,
                                        once=True,
                                        logger_name="backends",
                                    )
                            new_p = nn.Parameter(q_weight, requires_grad=False)
                            setattr(b_sub, pname, new_p)
                            if hasattr(b_sub, "_omni_weight_scale"):
                                b_sub._omni_weight_scale = q_scale
                            else:
                                b_sub.register_buffer("_omni_weight_scale", q_scale, persistent=False)
                            b_sub._omni_orig_dtype = orig_dt
                            param = new_p
                            quantized_layer_count += 1
                            max_streamed_numel = max(max_streamed_numel, int(param.numel()))
                        else:
                            if param.dtype == torch.bfloat16 and not caps.supports_bf16:
                                dest_dt = target_dtype
                            elif param.dtype in (torch.float32, torch.float64) and target_dtype in (torch.bfloat16, torch.float16):
                                dest_dt = target_dtype
                            elif param.dtype == torch.float64:
                                dest_dt = torch.float32
                            else:
                                dest_dt = param.dtype

                            if (
                                param.device != dest_device
                                or param.dtype != dest_dt
                                or is_inference_tensor(param)
                            ):
                                # Preserve original CPU parameter reference (e.g. ComfyUI `fast_disk=True` safetensors
                                # mmap view) before moving to GPU or materializing CPU-streamed weights so
                                # `demote_resident_layers` and `uninject` restore it directly without allocating
                                # new heap RAM or running GPU-to-CPU DMA transfers.
                                orig_map = getattr(b_sub, "_omni_orig_params", {})
                                if pname not in orig_map:
                                    orig_ref = param if param.device.type == "cpu" else param.detach().to(device="cpu")
                                    orig_map[pname] = (orig_ref, bool(param.requires_grad))
                                    b_sub._omni_orig_params = orig_map  # type: ignore[attr-defined]
                                try:
                                    t_moved = safe_to_device(param.detach(), device=dest_device, dtype=dest_dt)
                                except RuntimeError:
                                    # If a pre-loaded FP8 checkpoint is moved to a device driver that does not
                                    # support holding Float8 tensors on GPU, convert seamlessly to 1-byte INT8 QLoRA
                                    # and save the original FP8 tensor on CPU so `uninject` restores it for inference.
                                    if is_fp8_dtype(param.dtype):
                                        w_f32 = param.detach().to(device="cpu", dtype=torch.float32)
                                        s_orig = getattr(b_sub, "scale_weight", getattr(b_sub, "weight_scale", None))
                                        if isinstance(s_orig, torch.Tensor):
                                            s_cpu = s_orig.detach().to(device="cpu", dtype=torch.float32)
                                            if s_cpu.ndim == 1 and w_f32.ndim >= 2 and s_cpu.numel() == w_f32.shape[0]:
                                                s_cpu = s_cpu.view(-1, *([1] * (w_f32.ndim - 1)))
                                            w_f32 = w_f32 * s_cpu
                                        t_moved, q_scale = quantize_weight_fp8_or_int8(
                                            w_f32,
                                            mode="int8_qlora",
                                            dest_device=dest_device,
                                            supports_fp8=False,
                                        )
                                        if hasattr(b_sub, "_omni_weight_scale"):
                                            b_sub._omni_weight_scale = q_scale
                                        else:
                                            b_sub.register_buffer("_omni_weight_scale", q_scale, persistent=False)
                                        b_sub._omni_orig_dtype = target_dtype
                                    else:
                                        raise
                                if is_inference_tensor(t_moved):
                                    t_moved = t_moved.clone()
                                if not place_on_gpu and use_pinned_cpu and not t_moved.is_pinned():
                                    try:
                                        t_moved = t_moved.pin_memory()
                                    except RuntimeError as exc:
                                        report_exception(
                                            "backends.budget_pin_moved_weight",
                                            "pin_memory() failed for CPU-streamed weight; using pageable RAM",
                                            exc,
                                            level=logging.DEBUG,
                                            once=True,
                                            logger_name="backends",
                                        )
                                new_p = nn.Parameter(t_moved, requires_grad=False)
                                setattr(b_sub, pname, new_p)
                                param = new_p
                            elif not place_on_gpu and use_pinned_cpu and not param.is_pinned():
                                try:
                                    orig_map = getattr(b_sub, "_omni_orig_params", {})
                                    if pname not in orig_map and param.device.type == "cpu":
                                        orig_map[pname] = (param, bool(param.requires_grad))
                                        b_sub._omni_orig_params = orig_map  # type: ignore[attr-defined]
                                    new_p = nn.Parameter(param.detach().pin_memory(), requires_grad=False)
                                    setattr(b_sub, pname, new_p)
                                    param = new_p
                                except RuntimeError as exc:
                                    report_exception(
                                        "backends.budget_pin_existing_weight",
                                        "pin_memory() failed for existing CPU weight; using pageable RAM",
                                        exc,
                                        level=logging.DEBUG,
                                        once=True,
                                        logger_name="backends",
                                    )
                            if is_fp8_dtype(param.dtype) or param.dtype == torch.int8:
                                quantized_layer_count += 1
                                max_streamed_numel = max(max_streamed_numel, int(param.numel()))
                            elif (not place_on_gpu or param.dtype != target_dtype) and pname == "weight" and param.ndim >= 2:
                                max_streamed_numel = max(max_streamed_numel, int(param.numel()))

                        p_bytes = get_tensor_storage_bytes(param)
                        if param.device.type == device.type and device.type != "cpu":
                            resident_bytes += p_bytes
                            bytes_since_sync += p_bytes
                            if self.backend_id in ("xpu", "mps") and bytes_since_sync >= sync_chunk_bytes:
                                self.synchronize()
                                bytes_since_sync = 0
                        elif device.type != "cpu":
                            offloaded_bytes += p_bytes

                    for bname, buf in list(b_sub.named_buffers(recurse=False)):
                        if buf is None:
                            continue
                        if is_custom_quantized_tensor(buf):
                            try:
                                if buf.device != dest_device:
                                    orig_bufs = getattr(b_sub, "_omni_orig_buffers", {})
                                    if bname not in orig_bufs and buf.device.type == "cpu":
                                        orig_bufs[bname] = buf
                                        b_sub._omni_orig_buffers = orig_bufs  # type: ignore[attr-defined]
                                    assign_module_parameter(b_sub, bname, move_custom_quantized_tensor(buf, dest_device))
                            except Exception as exc:
                                report_exception(
                                    "backends.budget_move_base_quant_buf",
                                    f"Could not move base custom quantized buffer '{bname}' to {dest_device}",
                                    exc,
                                    level=logging.WARNING,
                                    once=True,
                                    logger_name="backends",
                                )
                            continue
                        if buf.dtype == torch.float64:
                            dest_dt = torch.float32
                        elif buf.dtype == torch.bfloat16 and not caps.supports_bf16:
                            dest_dt = target_dtype
                        else:
                            dest_dt = buf.dtype
                        if buf.device != dest_device or buf.dtype != dest_dt:
                            orig_bufs = getattr(b_sub, "_omni_orig_buffers", {})
                            if bname not in orig_bufs and buf.device.type == "cpu":
                                orig_bufs[bname] = buf
                                b_sub._omni_orig_buffers = orig_bufs  # type: ignore[attr-defined]
                            new_b = safe_to_device(buf.detach(), device=dest_device, dtype=dest_dt)
                            setattr(b_sub, bname, new_b)

                    # Also ensure ComfyUI's `scale_weight` / `weight_scale` tensor attributes (if stored as plain attributes)
                    # are moved alongside the layer and kept in FP32
                    for s_attr in ("scale_weight", "weight_scale"):
                        s_val = getattr(b_sub, s_attr, None)
                        if isinstance(s_val, torch.Tensor) and s_attr not in b_sub._buffers:
                            s_dt = torch.float32 if s_val.dtype == torch.float64 else s_val.dtype
                            if s_val.device != dest_device or s_val.dtype != s_dt:
                                orig_scales: Dict[str, torch.Tensor] = getattr(b_sub, "_omni_orig_scales", {})
                                if s_attr not in orig_scales and s_val.device.type == "cpu":
                                    orig_scales[s_attr] = s_val
                                    b_sub._omni_orig_scales = orig_scales  # type: ignore[attr-defined]
                                setattr(b_sub, s_attr, safe_to_device(s_val.detach(), device=dest_device, dtype=s_dt))

        if max_streamed_numel > 0 and device.type != "cpu":
            try:
                from ..lora.layers import prepare_streamed_weight_buffer

                prepare_streamed_weight_buffer(device, target_dtype, max_streamed_numel)
            except Exception as exc:
                report_exception(
                    "backends.budget_prepare_stream_buf",
                    "Could not pre-allocate GPU staging buffer during VRAM budget enforcement",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends",
                )

        self.synchronize()
        budget["resident_model_mb"] = round(resident_bytes / (1024**2), 1)
        budget["offloaded_model_mb"] = round(offloaded_bytes / (1024**2), 1)
        budget["quantized_base_layers"] = float(quantized_layer_count)
        budget["weight_format"] = detect_model_weight_format(module, self.base_weight_quantization)
        return budget

    def demote_resident_layers(
        self,
        module: nn.Module,
        target_demote_mb: float = 1200.0,
    ) -> float:
        """
        Dynamically move up to `target_demote_mb` of currently resident `LoRALinear.base_layer`
        weights from the accelerator to CPU at runtime (starting from the tail blocks).
        Used by `LoRATrainingPipeline.train_step()` to self-heal if an unknown model architecture
        hits an OOM spike or WDDM PCIe spill.
        """
        device = self.get_device()
        if device.type in ("cpu", "mps") or self.backend_id in ("cpu", "mps") or target_demote_mb <= 0:
            return 0.0

        self.synchronize()
        use_pinned_cpu = self.backend_id in ("cuda", "rocm", "npu")
        target_bytes = int(target_demote_mb * (1024**2))
        demoted_bytes = 0
        max_demoted_numel = 0

        lora_wrappers: List[nn.Module] = [
            sub
            for sub in module.modules()
            if sub.__class__.__name__ == "LoRALinear" and hasattr(sub, "base_layer")
        ]

        with torch.inference_mode(False):
            for wrapper in reversed(lora_wrappers):
                if demoted_bytes >= target_bytes:
                    break
                base = wrapper.base_layer
                has_gpu_param = any(_same_device(p.device, device) for p in base.parameters())
                if not has_gpu_param:
                    continue

                for b_sub in base.modules():
                    orig_map: Dict[str, Tuple[torch.Tensor, bool]] = getattr(b_sub, "_omni_orig_params", {})
                    orig_bufs: Dict[str, torch.Tensor] = getattr(b_sub, "_omni_orig_buffers", {})
                    orig_scales: Dict[str, torch.Tensor] = getattr(b_sub, "_omni_orig_scales", {})
                    for pname, param in list(b_sub.named_parameters(recurse=False)):
                        if _same_device(param.device, device):
                            # Never offload bitsandbytes Params4bit/Int8Params to CPU as bitsandbytes
                            # cannot execute or dequantize 4-bit weights from host memory.
                            if is_bitsandbytes_tensor(param):
                                continue
                            p_bytes = get_tensor_storage_bytes(param)
                            if (
                                self.base_weight_quantization is None
                                and pname in orig_map
                                and isinstance(orig_map[pname][0], torch.Tensor)
                                and orig_map[pname][0].device.type == "cpu"
                                and orig_map[pname][0].dtype == param.dtype
                            ):
                                orig_t, orig_req = orig_map[pname]
                                if is_inference_tensor(orig_t) or (use_pinned_cpu and not orig_t.is_pinned()):
                                    t_cpu = orig_t.detach().clone() if is_inference_tensor(orig_t) else orig_t.detach()
                                    if use_pinned_cpu and not t_cpu.is_pinned():
                                        try:
                                            t_cpu = t_cpu.pin_memory()
                                        except RuntimeError as exc:
                                            report_exception(
                                                "backends.demote_pin_orig",
                                                "pin_memory() failed during layer demotion; using pageable RAM",
                                                exc,
                                                level=logging.DEBUG,
                                                once=True,
                                                logger_name="backends",
                                            )
                                    restored_p = nn.Parameter(t_cpu, requires_grad=False)
                                else:
                                    restored_p = (
                                        orig_t
                                        if isinstance(orig_t, nn.Parameter) and orig_t.requires_grad == bool(orig_req)
                                        else nn.Parameter(orig_t, requires_grad=False)
                                    )
                                assign_module_parameter(b_sub, pname, restored_p)
                                demoted_bytes += p_bytes
                                if pname == "weight" and restored_p.ndim >= 2:
                                    max_demoted_numel = max(max_demoted_numel, int(restored_p.numel()))
                                continue
                            if is_custom_quantized_tensor(param):
                                try:
                                    moved_cpu = move_custom_quantized_tensor(param, torch.device("cpu"))
                                    assign_module_parameter(b_sub, pname, moved_cpu)
                                    demoted_bytes += p_bytes
                                except Exception as exc:
                                    report_exception(
                                        "backends.demote_quant_param",
                                        f"Could not demote custom quantized parameter '{pname}' to CPU",
                                        exc,
                                        level=logging.WARNING,
                                        once=True,
                                        logger_name="backends",
                                    )
                                continue
                            t_cpu = param.detach().to(device="cpu")
                            if is_inference_tensor(t_cpu):
                                t_cpu = t_cpu.clone()
                            if use_pinned_cpu and not t_cpu.is_pinned():
                                try:
                                    t_cpu = t_cpu.pin_memory()
                                except RuntimeError as exc:
                                    report_exception(
                                        "backends.demote_pin_param",
                                        "pin_memory() failed during layer demotion; using pageable RAM",
                                        exc,
                                        level=logging.DEBUG,
                                        once=True,
                                        logger_name="backends",
                                    )
                            setattr(b_sub, pname, nn.Parameter(t_cpu, requires_grad=False))
                            demoted_bytes += p_bytes
                            if pname == "weight" and t_cpu.ndim >= 2:
                                max_demoted_numel = max(max_demoted_numel, int(t_cpu.numel()))

                    for bname, buf in list(b_sub.named_buffers(recurse=False)):
                        if buf is not None and _same_device(buf.device, device):
                            if bname in orig_bufs and orig_bufs[bname].device.type == "cpu":
                                assign_module_parameter(b_sub, bname, orig_bufs[bname])
                            elif is_custom_quantized_tensor(buf):
                                try:
                                    assign_module_parameter(b_sub, bname, move_custom_quantized_tensor(buf, torch.device("cpu")))
                                except Exception as exc:
                                    report_exception(
                                        "backends.demote_quant_buf",
                                        f"Could not demote custom quantized buffer '{bname}' to CPU",
                                        exc,
                                        level=logging.WARNING,
                                        once=True,
                                        logger_name="backends",
                                    )
                            else:
                                setattr(b_sub, bname, buf.detach().to(device="cpu"))

                    for s_attr in ("scale_weight", "weight_scale"):
                        s_val = getattr(b_sub, s_attr, None)
                        if isinstance(s_val, torch.Tensor) and s_attr not in b_sub._buffers and _same_device(s_val.device, device):
                            if s_attr in orig_scales and orig_scales[s_attr].device.type == "cpu":
                                setattr(b_sub, s_attr, orig_scales[s_attr])
                            else:
                                setattr(b_sub, s_attr, s_val.detach().to(device="cpu"))

        if max_demoted_numel > 0 and device.type != "cpu":
            try:
                from ..lora.layers import prepare_streamed_weight_buffer

                prepare_streamed_weight_buffer(device, self.torch_dtype, max_demoted_numel)
            except Exception as exc:
                report_exception(
                    "backends.demote_prepare_stream_buf",
                    "Could not resize GPU staging buffer after layer demotion",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends",
                )
        if demoted_bytes > 0:
            self.synchronize()
        return round(demoted_bytes / (1024**2), 1)

    def optimize_module(self, module: nn.Module, mode: str = "eager", move_to_device: bool = True) -> nn.Module:
        """
        Strip any accidental `float64` tensors, optionally move module to target device,
        and apply backend-specific graph compilation or IPEX optimization.
        """
        module = sanitize_module_no_fp64(module)
        if move_to_device:
            module = module.to(device=self.get_device())
        if mode == "compile" and hasattr(torch, "compile"):
            return torch.compile(module)
        return module

    def get_sdpa_context(self) -> ContextManager[Any]:
        """
        Configure PyTorch Scaled Dot-Product Attention (F.scaled_dot_product_attention).
        Never calls xformers; relies on native PyTorch 2.x SDPA kernels.
        """
        try:
            attn_mod = getattr(torch.nn, "attention", None)
            if attn_mod is not None and hasattr(attn_mod, "sdpa_kernel"):
                from torch.nn.attention import SDPBackend, sdpa_kernel

                backends = [
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.MATH,
                ]
                cudnn_sdp = getattr(SDPBackend, "CUDNN_ATTENTION", None)
                if cudnn_sdp is not None and self.backend_id == "cuda":
                    backends.insert(0, cudnn_sdp)
                return sdpa_kernel(backends)
        except Exception as exc:
            report_exception(
                "backends.sdpa_kernel_context",
                "Could not configure torch.nn.attention.sdpa_kernel; using default SDPA",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="backends",
            )
        return contextlib.nullcontext()

    @staticmethod
    def resolve_dtype(dtype_str: str) -> torch.dtype:
        mapping: Dict[str, torch.dtype] = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            # When FP8 or INT8 QLoRA precision is selected, activations & trainable LoRA weights
            # execute in BF16 (or FP16 via select_optimal_dtype) while frozen base weights are 8-bit.
            "fp8": torch.bfloat16,
            "fp8_e4m3fn": torch.bfloat16,
            "fp8_e5m2": torch.bfloat16,
            "int8_qlora": torch.bfloat16,
        }
        key = dtype_str.lower()
        if key in ("fp64", "float64", "double"):
            raise ValueError(
                "FP64 (double) is explicitly disallowed: consumer GPUs such as Intel Arc (XPU) "
                "do not support hardware aspect fp64. Use 'bf16', 'fp16', 'fp8_e4m3fn', 'fp8_e5m2', 'int8_qlora', or 'fp32'."
            )
        if key not in mapping:
            raise ValueError(f"Unsupported precision '{dtype_str}'. Choose from: {list(mapping.keys())}")
        return mapping[key]
