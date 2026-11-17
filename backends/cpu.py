"""
Host CPU Reference Backend (`cpu` / OneDNN AVX512-BF16).
"""

from __future__ import annotations

import contextlib
from typing import Any, ContextManager
import torch

from .base import (
    DeviceCapabilities,
    HardwareBackend,
    MemorySnapshot,
    probe_device_fp8_support,
    query_host_ram_mb,
)


class CPUBackend(HardwareBackend):
    """Fallback reference backend for CPU execution and dry-run shape validation."""

    @property
    def backend_id(self) -> str:
        return "cpu"

    @property
    def device_type(self) -> str:
        return "cpu"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def get_device(self) -> torch.device:
        return torch.device("cpu")

    def get_capabilities(self) -> DeviceCapabilities:
        total_mb, _ = query_host_ram_mb()
        return DeviceCapabilities(
            backend_name="PyTorch CPU Reference (OneDNN / AVX512-BF16)",
            device_type="cpu",
            device_name="Host CPU",
            device_index=0,
            total_vram_mb=total_mb,
            supports_bf16=True,
            supports_fp16=True,
            supports_tf32=False,
            supports_fp8=probe_device_fp8_support(torch.device("cpu")),
            sdpa_backends=["cpp_sdpa", "math"],
            distributed_backend="gloo",
        )

    def synchronize(self) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def get_memory_snapshot(self) -> MemorySnapshot:
        total_mb, avail_mb = query_host_ram_mb()
        used_mb = max(0.0, round(total_mb - avail_mb, 2))
        return MemorySnapshot(used_mb, used_mb, used_mb, total_mb, avail_mb)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def autocast_context(self, enabled: bool = True) -> ContextManager[Any]:
        if not enabled or self.torch_dtype != torch.bfloat16:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)
