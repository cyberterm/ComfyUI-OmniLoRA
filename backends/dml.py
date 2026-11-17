"""
Windows DirectML Hardware Backend (`torch_directml` / `dml`).
"""

from __future__ import annotations

import contextlib
from typing import Any, ContextManager
import torch

from .base import (
    DeviceCapabilities,
    HardwareBackend,
    MemorySnapshot,
    query_comfy_device_memory_mb,
    query_host_ram_mb,
)


class DirectMLBackend(HardwareBackend):
    """
    Hardware backend for Windows DirectML (`torch_directml`) supporting DX12 GPUs
    (AMD Radeon on Windows, Intel Arc on Windows, Qualcomm Adreno).
    Note: DirectML primarily accelerates FP16/FP32 rather than native BF16;
    `model_management` automatically selects `fp16` + `FP32GradScaler` when DirectML is detected.
    """

    @property
    def backend_id(self) -> str:
        return "dml"

    @property
    def device_type(self) -> str:
        return "privateuseone"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch_directml

            return bool(torch_directml.is_available())
        except ImportError:
            return False

    def get_device(self) -> torch.device:
        if self.is_available():
            import torch_directml

            try:
                count = int(torch_directml.device_count()) if hasattr(torch_directml, "device_count") else 1
                idx = max(0, min(int(self.device_index), max(0, count - 1)))
                return torch_directml.device(idx)
            except Exception:
                return torch_directml.device(0)
        return torch.device("cpu")

    def _query_dml_memory_mb(self) -> tuple[float, float, float]:
        """Return `(total_vram_mb, free_vram_mb, allocated_vram_mb)`."""
        if self.is_available():
            comfy_mem = query_comfy_device_memory_mb(self.get_device())
            if comfy_mem is not None and comfy_mem[0] > 0:
                total_c, free_c = comfy_mem
                return (total_c, free_c, max(0.0, round(total_c - free_c, 2)))
            total_ram, avail_ram = query_host_ram_mb()
            est_vram = min(16384.0, max(6144.0, round(total_ram * 0.5, 1)))
            free_vram = round(min(est_vram, avail_ram * 0.8), 1)
            return (est_vram, free_vram, 0.0)
        return (12288.0, 12288.0, 0.0)

    def get_capabilities(self) -> DeviceCapabilities:
        device_name = "DirectX 12 GPU (torch-directml)"
        if self.is_available():
            try:
                import torch_directml

                device_name = torch_directml.device_name(self.device_index)
            except Exception:
                pass
        total_mb, _, _ = self._query_dml_memory_mb()
        return DeviceCapabilities(
            backend_name="Windows DirectML (DX12)",
            device_type="dml",
            device_name=device_name,
            device_index=self.device_index,
            total_vram_mb=total_mb,
            supports_bf16=False,
            supports_fp16=True,
            supports_tf32=False,
            sdpa_backends=["math"],
            distributed_backend="gloo",
        )

    def synchronize(self) -> None:
        if self.is_available():
            try:
                with torch.inference_mode(False), torch.no_grad():
                    dev = self.get_device()
                    if dev.type != "cpu":
                        _ = torch.zeros(1, dtype=torch.float32, device=dev).to(device="cpu")
            except Exception:
                pass

    def empty_cache(self) -> None:
        import gc

        gc.collect()
        self.synchronize()

    def get_memory_snapshot(self) -> MemorySnapshot:
        total_mb, free_mb, used_mb = self._query_dml_memory_mb()
        return MemorySnapshot(used_mb, used_mb, used_mb, total_mb, free_mb)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def autocast_context(self, enabled: bool = True) -> ContextManager[Any]:
        # DirectML relies on explicit tensor dtype casting rather than CUDA autocast
        return contextlib.nullcontext()

