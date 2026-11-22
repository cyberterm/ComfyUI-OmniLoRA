from .base import (
    FP8_DTYPES,
    DeviceCapabilities,
    FP32GradScaler,
    HardwareBackend,
    MemorySnapshot,
    get_tensor_storage_bytes,
    is_bitsandbytes_module,
    is_bitsandbytes_tensor,
    is_custom_quantized_tensor,
    is_fp8_dtype,
    is_inference_tensor,
    probe_bitsandbytes_support,
    probe_device_fp8_support,
    quantize_weight_fp8_or_int8,
    safe_to_device,
    sanitize_module_no_fp64,
    trim_process_working_set,
)
from .cpu import CPUBackend
from .cuda import CUDABackend
from .hpu import HPUBackend
from .model_management import (
    AUTO_PROBE_ORDER,
    BACKEND_REGISTRY,
    AutoHardwarePlan,
    VRAMState,
    classify_vram_state,
    detect_and_configure,
    resolve_backend,
    select_optimal_dtype,
)
from .mps import MPSBackend
from .npu import NPUBackend
from .rocm import ROCmBackend
from .tpu import TPUBackend
from .xpu import XPUBackend

__all__ = [
    "HardwareBackend",
    "DeviceCapabilities",
    "MemorySnapshot",
    "FP32GradScaler",
    "FP8_DTYPES",
    "is_fp8_dtype",
    "is_inference_tensor",
    "is_bitsandbytes_tensor",
    "is_bitsandbytes_module",
    "probe_bitsandbytes_support",
    "is_custom_quantized_tensor",
    "probe_device_fp8_support",
    "quantize_weight_fp8_or_int8",
    "safe_to_device",
    "sanitize_module_no_fp64",
    "trim_process_working_set",
    "XPUBackend",
    "CUDABackend",
    "ROCmBackend",
    "TPUBackend",
    "NPUBackend",
    "HPUBackend",
    "MPSBackend",
    "CPUBackend",
    "BACKEND_REGISTRY",
    "AUTO_PROBE_ORDER",
    "VRAMState",
    "AutoHardwarePlan",
    "classify_vram_state",
    "select_optimal_dtype",
    "resolve_backend",
    "detect_and_configure",
]
