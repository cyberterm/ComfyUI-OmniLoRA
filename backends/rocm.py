"""
AMD ROCm HIP Hardware Backend (`torch.cuda` + `torch.version.hip`).
"""

from __future__ import annotations

import os
from typing import Dict
import torch

from .base import DeviceCapabilities
from .cuda import CUDABackend


class ROCmBackend(CUDABackend):
    """Hardware backend for AMD Instinct / Radeon GPUs via ROCm HIP (`torch.version.hip`)."""

    @property
    def backend_id(self) -> str:
        return "rocm"

    @classmethod
    def is_available(cls) -> bool:
        return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None)

    def configure_environment(self) -> Dict[str, str]:
        torch.set_default_dtype(torch.float32)
        applied: Dict[str, str] = {
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
            "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
        if self.is_available():
            try:
                self.device_index = self._valid_device_index()
                torch.cuda.set_device(self.device_index)
            except Exception:
                pass
        return applied

    def get_capabilities(self) -> DeviceCapabilities:
        if not self.is_available():
            return DeviceCapabilities(
                backend_name="AMD ROCm HIP (Offline)",
                device_type="cuda",
                device_name="AMD Radeon RX 7900 XTX (24GB)",
                device_index=self.device_index,
                total_vram_mb=24576.0,
                supports_bf16=True,
                supports_fp16=True,
                supports_tf32=False,
                supports_fp8=True,
                sdpa_backends=["aotriton_sdp", "math"],
                distributed_backend="rccl",
            )
        caps = super().get_capabilities()
        caps.backend_name = f"AMD ROCm (HIP {torch.version.hip})"
        caps.supports_tf32 = False
        caps.sdpa_backends = ["aotriton_sdp", "math"]
        caps.distributed_backend = "rccl"
        return caps

