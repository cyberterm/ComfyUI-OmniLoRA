"""
ComfyUI-Style Automatic Device Detection, VRAM Tiering, and Precision Cascade (`model_management`).

Automatically probes installed PyTorch backends in priority order:
  1. Intel XPU (`torch.xpu` — Arc / Battlemage / Max)
  2. AMD ROCm HIP (`torch.cuda` + `torch.version.hip`)
  3. NVIDIA CUDA (`torch.cuda`)
  4. Huawei Ascend NPU (`torch_npu` / `torch.npu`)
  5. Intel Gaudi HPU (`habana_frameworks.torch` / `torch.hpu`)
  6. Windows DirectML (`torch_directml`)
  7. Apple Metal MPS (`torch.backends.mps`)
  8. Host CPU (`cpu`)

Then classifies available VRAM into 4 ComfyUI-inspired tiers (`HIGH_VRAM`, `NORMAL_VRAM`,
`LOW_VRAM`, `OFFLOAD_VRAM`) and selects the optimal training precision (`bf16` -> `fp16` + GradScaler -> `fp32`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Type

from .base import DeviceCapabilities, HardwareBackend
from .cpu import CPUBackend
from .cuda import CUDABackend
from .dml import DirectMLBackend
from .hpu import HPUBackend
from .mps import MPSBackend
from .npu import NPUBackend
from .rocm import ROCmBackend
from .tpu import TPUBackend
from .xpu import XPUBackend


class VRAMState(str, Enum):
    """ComfyUI-style automatic VRAM management states."""

    HIGH_VRAM = "HIGH_VRAM"        # >= 24 GB: Keep VAE & Text Encoder resident on device
    NORMAL_VRAM = "NORMAL_VRAM"    # 12 - 23.9 GB: 2-Stage Pre-Cache & Offload + Grad Checkpointing
    LOW_VRAM = "LOW_VRAM"          # 8 - 11.9 GB: Pre-Cache & Offload + Grad Checkpointing + 768px cap
    OFFLOAD_VRAM = "OFFLOAD_VRAM"  # < 8 GB or CPU: Host CPU streaming + CPU optimizer offload


BACKEND_REGISTRY: Dict[str, Type[HardwareBackend]] = {
    "xpu": XPUBackend,
    "rocm": ROCmBackend,
    "cuda": CUDABackend,
    "tpu": TPUBackend,
    "npu": NPUBackend,
    "hpu": HPUBackend,
    "dml": DirectMLBackend,
    "mps": MPSBackend,
    "cpu": CPUBackend,
}

AUTO_PROBE_ORDER: Tuple[str, ...] = (
    "xpu",
    "rocm",
    "cuda",
    "tpu",
    "npu",
    "hpu",
    "dml",
    "mps",
    "cpu",
)


@dataclass
class AutoHardwarePlan:
    """Complete auto-configured hardware & memory execution plan."""

    backend: HardwareBackend
    capabilities: DeviceCapabilities
    vram_state: VRAMState
    resolved_dtype: str
    requires_grad_scaler: bool
    recommended_conditioning_mode: str
    recommended_gradient_checkpointing: bool
    recommended_resolution: int
    recommended_batch_size: int
    recommended_grad_accum: int
    base_weight_quantization: Optional[str] = None


def classify_vram_state(total_vram_mb: float, backend_id: str, override_mode: str = "auto") -> VRAMState:
    """Classify device VRAM capacity into ComfyUI-style `VRAMState` tiers."""
    mode = override_mode.lower()
    if mode in ("high", "high_vram", "highvram"):
        return VRAMState.HIGH_VRAM
    if mode in ("normal", "normal_vram", "normalvram"):
        return VRAMState.NORMAL_VRAM
    if mode in ("low", "low_vram", "lowvram"):
        return VRAMState.LOW_VRAM
    if mode in ("offload", "offload_vram", "novram", "cpu"):
        return VRAMState.OFFLOAD_VRAM

    if backend_id == "cpu":
        return VRAMState.OFFLOAD_VRAM
    if total_vram_mb >= 24000.0:
        return VRAMState.HIGH_VRAM
    if total_vram_mb >= 12000.0:
        return VRAMState.NORMAL_VRAM
    if total_vram_mb >= 8000.0:
        return VRAMState.LOW_VRAM
    return VRAMState.OFFLOAD_VRAM


def select_optimal_dtype(capabilities: DeviceCapabilities, requested_dtype: str = "auto") -> Tuple[str, bool]:
    """
    Automatic Precision Cascade:
      1. If `requested_dtype` is an explicit compute dtype (`bf16`, `fp16`, `fp32`), honor it.
      2. If `requested_dtype` is `"auto"` or an 8-bit base-weight quantization mode
         (`"fp8_e4m3fn"`, `"fp8_e5m2"`, `"int8_qlora"`), pick `bf16` for activations & LoRA
         adapters if natively supported by hardware matrix engines, else `fp16` + `FP32GradScaler`.
    Returns `(compute_dtype_str, requires_grad_scaler)`.
    """
    req = requested_dtype.lower()
    if req in ("bf16", "bfloat16"):
        return "bf16", False
    if req in ("fp16", "float16"):
        return "fp16", True
    if req in ("fp32", "float32"):
        return "fp32", False

    if capabilities.supports_bf16:
        return "bf16", False
    if capabilities.supports_fp16:
        return "fp16", True
    return "fp32", False


def resolve_backend(
    preferred: str = "auto",
    device_index: int = 0,
    dtype: str = "auto",
    allow_fallback: bool = True,
) -> HardwareBackend:
    """
    Instantiate the requested HardwareBackend or auto-detect the best available
    PyTorch accelerator in `AUTO_PROBE_ORDER`.
    """
    pref = preferred.lower()
    if pref == "xla":
        pref = "tpu"
    req_dt = dtype.lower()
    quant_mode: Optional[str] = (
        "fp8_e4m3fn"
        if req_dt == "fp8"
        else (req_dt if req_dt in ("fp8_e4m3fn", "fp8_e5m2", "int8_qlora") else None)
    )
    initial_dtype = "bf16" if (req_dt == "auto" or quant_mode is not None) else req_dt

    if pref in BACKEND_REGISTRY and pref != "auto":
        backend_cls = BACKEND_REGISTRY[pref]
        if backend_cls.is_available() or not allow_fallback:
            backend = backend_cls(device_index=device_index, dtype=initial_dtype)
            caps = backend.get_capabilities()
            resolved_dt, _ = select_optimal_dtype(caps, dtype)
            backend.dtype_name = resolved_dt
            backend.torch_dtype = backend.resolve_dtype(resolved_dt)
            backend.base_weight_quantization = quant_mode
            backend.configure_environment()
            return backend

    for candidate in AUTO_PROBE_ORDER:
        cls = BACKEND_REGISTRY[candidate]
        if cls.is_available():
            backend = cls(device_index=device_index, dtype=initial_dtype)
            caps = backend.get_capabilities()
            resolved_dt, _ = select_optimal_dtype(caps, dtype)
            backend.dtype_name = resolved_dt
            backend.torch_dtype = backend.resolve_dtype(resolved_dt)
            backend.base_weight_quantization = quant_mode
            backend.configure_environment()
            return backend

    fallback = CPUBackend(device_index=0, dtype=initial_dtype)
    caps = fallback.get_capabilities()
    resolved_dt, _ = select_optimal_dtype(caps, dtype)
    fallback.dtype_name = resolved_dt
    fallback.torch_dtype = fallback.resolve_dtype(resolved_dt)
    fallback.base_weight_quantization = quant_mode
    fallback.configure_environment()
    return fallback



def detect_and_configure(
    preferred_backend: str = "auto",
    device_index: int = 0,
    requested_dtype: str = "auto",
    vram_mode: str = "auto",
    override_vram_mb: Optional[float] = None,
    allow_fallback: bool = True,
) -> AutoHardwarePlan:
    """
    ComfyUI-style one-call hardware probe & hyperparameter auto-tuner.
    Detects the active PyTorch device, measures VRAM, classifies `VRAMState`,
    picks the optimal precision (`bf16` -> `fp16`+GradScaler -> `fp32`), configures
    optional 8-bit base weight quantization (`fp8_e4m3fn`, `fp8_e5m2`, `int8_qlora`),
    and returns tailored training recommendations.
    """
    backend = resolve_backend(
        preferred=preferred_backend,
        device_index=device_index,
        dtype=requested_dtype,
        allow_fallback=allow_fallback,
    )
    caps = backend.get_capabilities()
    if override_vram_mb is not None and override_vram_mb > 0:
        caps.total_vram_mb = float(override_vram_mb)

    req_dt = requested_dtype.lower()
    quant_mode: Optional[str] = (
        "fp8_e4m3fn"
        if req_dt == "fp8"
        else (req_dt if req_dt in ("fp8_e4m3fn", "fp8_e5m2", "int8_qlora") else None)
    )
    resolved_dtype, requires_grad_scaler = select_optimal_dtype(caps, requested_dtype)
    backend.dtype_name = resolved_dtype
    backend.torch_dtype = backend.resolve_dtype(resolved_dtype)
    backend.base_weight_quantization = quant_mode

    vram_state = classify_vram_state(caps.total_vram_mb, backend.backend_id, override_mode=vram_mode)

    if vram_state == VRAMState.HIGH_VRAM:
        cond_mode = "on_the_fly_xpu"
        grad_ckpt = True
        resolution = 1024
        batch_size = 2
        grad_accum = 1
    elif vram_state == VRAMState.NORMAL_VRAM:
        cond_mode = "precache_offload"
        grad_ckpt = True
        resolution = 1024
        batch_size = 1
        grad_accum = 1
    elif vram_state == VRAMState.LOW_VRAM:
        cond_mode = "precache_offload"
        grad_ckpt = True
        resolution = 768
        batch_size = 1
        grad_accum = 1
    else:
        cond_mode = "cpu_offload"
        grad_ckpt = True
        resolution = 512
        batch_size = 1
        grad_accum = 1

    return AutoHardwarePlan(
        backend=backend,
        capabilities=caps,
        vram_state=vram_state,
        resolved_dtype=resolved_dtype,
        requires_grad_scaler=requires_grad_scaler,
        recommended_conditioning_mode=cond_mode,
        recommended_gradient_checkpointing=grad_ckpt,
        recommended_resolution=resolution,
        recommended_batch_size=batch_size,
        recommended_grad_accum=grad_accum,
        base_weight_quantization=quant_mode,
    )
