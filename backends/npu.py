"""
Huawei Ascend NPU Hardware Backend (`torch_npu` / `torch.npu`).
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


class NPUBackend(HardwareBackend):
    """Hardware backend for Huawei Ascend NPUs via `torch_npu` (`torch.npu`)."""

    @property
    def backend_id(self) -> str:
        return "npu"

    @property
    def device_type(self) -> str:
        return "npu"

    @classmethod
    def is_available(cls) -> bool:
        if not hasattr(torch, "npu"):
            try:
                import torch_npu  # noqa: F401
            except ImportError:
                return False
        return bool(hasattr(torch, "npu") and torch.npu.is_available())

    def _valid_device_index(self) -> int:
        if not self.is_available():
            return self.device_index
        try:
            count = int(torch.npu.device_count())
            if count > 0:
                return max(0, min(int(self.device_index), count - 1))
        except Exception:
            pass
        return 0

    def configure_environment(self) -> Dict[str, str]:
        torch.set_default_dtype(torch.float32)
        applied: Dict[str, str] = {
            "TASK_QUEUE_ENABLE": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        }
        if self.is_available():
            try:
                self.device_index = self._valid_device_index()
                torch.npu.set_device(self.device_index)
            except Exception:
                pass
        return applied

    def get_device(self) -> torch.device:
        if self.is_available():
            return torch.device(f"npu:{self._valid_device_index()}")
        return torch.device("cpu")

    def get_capabilities(self) -> DeviceCapabilities:
        if not self.is_available():
            return DeviceCapabilities(
                backend_name="Huawei Ascend CANN (Offline)",
                device_type="npu",
                device_name="Huawei Ascend 910B NPU (32GB HBM)",
                device_index=self.device_index,
                total_vram_mb=32768.0,
                supports_bf16=True,
                supports_fp16=True,
                supports_tf32=False,
                sdpa_backends=["npu_fusion_attention", "math"],
                distributed_backend="hccl",
            )
        idx = self._valid_device_index()
        dev_name = "Huawei Ascend 910B"
        total_mb = 32768.0
        try:
            props = torch.npu.get_device_properties(idx)
            dev_name = getattr(props, "name", dev_name)
            total_mb = round(getattr(props, "total_memory", 32 * 1024**3) / (1024**2), 1)
        except Exception:
            comfy_mem = query_comfy_device_memory_mb(self.get_device())
            if comfy_mem is not None and comfy_mem[0] > 0:
                total_mb = comfy_mem[0]
        return DeviceCapabilities(
            backend_name="Huawei Ascend NPU (CANN)",
            device_type="npu",
            device_name=dev_name,
            device_index=idx,
            total_vram_mb=total_mb,
            supports_bf16=True,
            supports_fp16=True,
            supports_tf32=False,
            supports_fp8=probe_device_fp8_support(self.get_device()),
            sdpa_backends=["npu_fusion_attention", "math"],
            distributed_backend="hccl",
        )

    def synchronize(self) -> None:
        if self.is_available():
            try:
                torch.npu.synchronize(self._valid_device_index())
            except Exception:
                pass

    def empty_cache(self) -> None:
        if self.is_available():
            try:
                torch.npu.synchronize(self._valid_device_index())
                torch.npu.empty_cache()
            except Exception:
                pass

    def get_memory_snapshot(self) -> MemorySnapshot:
        if not self.is_available():
            return MemorySnapshot(0.0, 0.0, 0.0, 32768.0, 32768.0)
        try:
            idx = self._valid_device_index()
            props = torch.npu.get_device_properties(idx)
            total = getattr(props, "total_memory", 32 * 1024**3) / (1024**2)
            alloc = torch.npu.memory_allocated(idx) / (1024**2)
            res = torch.npu.memory_reserved(idx) / (1024**2)
            max_alloc = torch.npu.max_memory_allocated(idx) / (1024**2)
            free_mb = max(0.0, total - max(alloc, res))
            if hasattr(torch.npu, "mem_get_info"):
                try:
                    free_bytes, total_bytes = torch.npu.mem_get_info(idx)
                    if total_bytes > 0:
                        total = total_bytes / (1024**2)
                        free_mb = min(total, (free_bytes / (1024**2)) + max(0.0, res - alloc))
                except Exception:
                    pass
            return MemorySnapshot(round(alloc, 2), round(res, 2), round(max_alloc, 2), round(total, 2), round(free_mb, 2))
        except Exception:
            return MemorySnapshot(0.0, 0.0, 0.0, 32768.0, 32768.0)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.is_available():
            try:
                torch.npu.manual_seed_all(seed)
            except Exception:
                pass

