"""
Apple Silicon Metal Performance Shaders Backend (`torch.backends.mps`).
"""

from __future__ import annotations

import torch

from .base import (
    DeviceCapabilities,
    HardwareBackend,
    MemorySnapshot,
    query_comfy_device_memory_mb,
    query_host_ram_mb,
)


class MPSBackend(HardwareBackend):
    """Hardware backend for Apple Silicon Unified Memory GPUs via `torch.backends.mps`."""

    @property
    def backend_id(self) -> str:
        return "mps"

    @property
    def device_type(self) -> str:
        return "mps"

    @classmethod
    def is_available(cls) -> bool:
        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    def get_device(self) -> torch.device:
        return torch.device("mps", 0) if self.is_available() else torch.device("cpu")

    def _probe_bf16_support(self) -> bool:
        if not self.is_available():
            return True
        try:
            dev = self.get_device()
            a = torch.ones((2, 2), device=dev, dtype=torch.bfloat16)
            return bool(torch.matmul(a, a).shape == (2, 2))
        except Exception:
            return False

    def _query_unified_vram_mb(self) -> tuple[float, float]:
        if self.is_available():
            comfy_mem = query_comfy_device_memory_mb(self.get_device())
            if comfy_mem is not None and comfy_mem[0] > 0:
                return comfy_mem
            if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
                try:
                    rec_mb = float(torch.mps.recommended_max_memory()) / (1024**2)
                    if rec_mb > 0:
                        alloc_mb = (
                            float(torch.mps.current_allocated_memory()) / (1024**2)
                            if hasattr(torch.mps, "current_allocated_memory")
                            else 0.0
                        )
                        return (round(rec_mb, 1), round(max(0.0, rec_mb - alloc_mb), 1))
                except Exception:
                    pass
            total_ram, avail_ram = query_host_ram_mb()
            return (round(total_ram * 0.75, 1), round(avail_ram * 0.85, 1))
        return (36864.0, 36864.0)

    def get_capabilities(self) -> DeviceCapabilities:
        total_mb, _ = self._query_unified_vram_mb()
        return DeviceCapabilities(
            backend_name="Apple Metal Performance Shaders (MPS)",
            device_type="mps",
            device_name="Apple M-Series Unified GPU",
            device_index=0,
            total_vram_mb=total_mb,
            supports_bf16=self._probe_bf16_support(),
            supports_fp16=True,
            supports_tf32=False,
            sdpa_backends=["mps_sdpa", "math"],
            distributed_backend="gloo",
        )

    def synchronize(self) -> None:
        if self.is_available() and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    def empty_cache(self) -> None:
        if self.is_available() and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except Exception:
                pass

    def get_memory_snapshot(self) -> MemorySnapshot:
        total_mb, free_mb = self._query_unified_vram_mb()
        if self.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                alloc = torch.mps.current_allocated_memory() / (1024**2)
                driver = (
                    torch.mps.driver_allocated_memory() / (1024**2)
                    if hasattr(torch.mps, "driver_allocated_memory")
                    else alloc
                )
                return MemorySnapshot(
                    round(alloc, 2),
                    round(driver, 2),
                    round(driver, 2),
                    total_mb,
                    round(max(0.0, min(free_mb, total_mb - alloc)), 2),
                )
            except Exception:
                pass
        return MemorySnapshot(0.0, 0.0, 0.0, total_mb, free_mb)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            try:
                torch.mps.manual_seed(seed)
            except Exception:
                pass

