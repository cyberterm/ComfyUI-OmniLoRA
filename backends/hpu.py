"""
Intel Gaudi HPU Hardware Backend (`habana_frameworks.torch` / `torch.hpu`).
"""

from __future__ import annotations

from typing import Dict
import torch

from .base import (
    DeviceCapabilities,
    HardwareBackend,
    MemorySnapshot,
    probe_device_fp8_support,
    query_comfy_device_memory_mb,
)


class HPUBackend(HardwareBackend):
    """Hardware backend for Intel Gaudi2 / Gaudi3 AI Accelerators via `habana_frameworks.torch` (`hpu`)."""

    @property
    def backend_id(self) -> str:
        return "hpu"

    @property
    def device_type(self) -> str:
        return "hpu"

    @classmethod
    def is_available(cls) -> bool:
        if not hasattr(torch, "hpu"):
            try:
                import habana_frameworks.torch.core as htcore  # noqa: F401
            except ImportError:
                return False
        return bool(hasattr(torch, "hpu") and torch.hpu.is_available())

    def _valid_device_index(self) -> int:
        if not self.is_available():
            return self.device_index
        try:
            if hasattr(torch.hpu, "device_count"):
                count = int(torch.hpu.device_count())
                if count > 0:
                    return max(0, min(int(self.device_index), count - 1))
        except Exception:
            pass
        return 0

    def configure_environment(self) -> Dict[str, str]:
        torch.set_default_dtype(torch.float32)
        applied: Dict[str, str] = {
            "PT_HPU_LAZY_MODE": "0",
            "LOG_LEVEL_ALL": "4",
        }
        if self.is_available() and hasattr(torch.hpu, "set_device"):
            try:
                self.device_index = self._valid_device_index()
                torch.hpu.set_device(self.device_index)
            except Exception:
                pass
        return applied

    def get_device(self) -> torch.device:
        if self.is_available():
            return torch.device(f"hpu:{self._valid_device_index()}")
        return torch.device("cpu")

    def get_capabilities(self) -> DeviceCapabilities:
        total_mb = 98304.0
        dev_name = "Intel Gaudi2 AI Accelerator (96GB HBM2e)"
        idx = self._valid_device_index()
        if self.is_available():
            try:
                if hasattr(torch.hpu, "get_device_properties"):
                    props = torch.hpu.get_device_properties(idx)
                    dev_name = getattr(props, "name", dev_name)
                    prop_mem = getattr(props, "total_memory", 0)
                    if prop_mem > 0:
                        total_mb = round(prop_mem / (1024**2), 1)
            except Exception:
                pass
            if total_mb == 98304.0:
                comfy_mem = query_comfy_device_memory_mb(self.get_device())
                if comfy_mem is not None and comfy_mem[0] > 0:
                    total_mb = comfy_mem[0]
        return DeviceCapabilities(
            backend_name="Intel Gaudi SynapseAI (HPU)",
            device_type="hpu",
            device_name=dev_name,
            device_index=idx,
            total_vram_mb=total_mb,
            supports_bf16=True,
            supports_fp16=True,
            supports_tf32=False,
            supports_fp8=True if not self.is_available() else probe_device_fp8_support(self.get_device()),
            sdpa_backends=["fsdpa_hpu", "math"],
            distributed_backend="hccl",
        )

    def synchronize(self) -> None:
        if self.is_available():
            try:
                torch.hpu.synchronize()
            except Exception:
                pass

    def empty_cache(self) -> None:
        if self.is_available() and hasattr(torch.hpu, "empty_cache"):
            try:
                torch.hpu.empty_cache()
            except Exception:
                pass

    def get_memory_snapshot(self) -> MemorySnapshot:
        if not self.is_available():
            return MemorySnapshot(0.0, 0.0, 0.0, 98304.0, 98304.0)
        try:
            idx = self._valid_device_index()
            total_mb = self.get_capabilities().total_vram_mb
            alloc = torch.hpu.memory_allocated(idx) / (1024**2)
            res = (
                torch.hpu.memory_reserved(idx) / (1024**2)
                if hasattr(torch.hpu, "memory_reserved")
                else alloc
            )
            max_alloc = torch.hpu.max_memory_allocated(idx) / (1024**2)
            free_mb = max(0.0, total_mb - max(alloc, res))
            return MemorySnapshot(round(alloc, 2), round(res, 2), round(max_alloc, 2), round(total_mb, 2), round(free_mb, 2))
        except Exception:
            return MemorySnapshot(0.0, 0.0, 0.0, 98304.0, 98304.0)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.is_available() and hasattr(torch.hpu, "manual_seed_all"):
            try:
                torch.hpu.manual_seed_all(seed)
            except Exception:
                pass

