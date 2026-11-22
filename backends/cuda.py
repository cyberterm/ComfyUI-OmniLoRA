"""
NVIDIA CUDA Hardware Backend (`torch.cuda`).
"""

from __future__ import annotations

import logging
from typing import Dict
import torch

from ..diagnostics import report_exception
from .base import (
    DeviceCapabilities,
    HardwareBackend,
    MemorySnapshot,
    probe_device_fp8_support,
    query_comfy_device_memory_mb,
)


class CUDABackend(HardwareBackend):
    """Hardware backend for NVIDIA CUDA GPUs."""

    @property
    def backend_id(self) -> str:
        return "cuda"

    @property
    def device_type(self) -> str:
        return "cuda"

    @classmethod
    def is_available(cls) -> bool:
        return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None) is None)

    def _valid_device_index(self) -> int:
        if not self.is_available():
            return self.device_index
        try:
            count = int(torch.cuda.device_count())
            if count > 0:
                return max(0, min(int(self.device_index), count - 1))
        except (RuntimeError, ValueError, TypeError) as exc:
            report_exception("cuda.device_count", "torch.cuda.device_count() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")
        return 0

    def _probe_bf16_support(self) -> bool:
        if not self.is_available():
            return True
        try:
            idx = self._valid_device_index()
            props = torch.cuda.get_device_properties(idx)
            if int(getattr(props, "major", 8)) < 8:
                # Pre-Ampere CUDA GPUs (Volta sm_70, Turing sm_75 such as Tesla T4 / RTX 20-series,
                # Pascal sm_61) lack hardware BF16 Tensor Cores and only emulate BF16 in software
                # via FP32 CUDA cores (whereas PyTorch 2.2+ `is_bf16_supported()` defaults to
                # `including_emulation=True`). Returning False selects native FP16 Tensor Cores
                # (`fp16` + `FP32GradScaler`) on Turing/Volta for up to 8x faster matmuls & SDPA.
                return False
        except (RuntimeError, AttributeError, ValueError) as exc:
            report_exception("cuda.probe_bf16_props", "CUDA device properties probe failed", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                return bool(torch.cuda.is_bf16_supported(including_emulation=False))
            except TypeError:
                try:
                    return bool(torch.cuda.is_bf16_supported())
                except (RuntimeError, AttributeError) as exc:
                    report_exception("cuda.is_bf16_supported_legacy", "torch.cuda.is_bf16_supported() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")
            except (RuntimeError, AttributeError) as exc:
                report_exception("cuda.is_bf16_supported", "torch.cuda.is_bf16_supported(including_emulation=False) failed", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")
        try:
            dev = self.get_device()
            a = torch.ones((2, 2), device=dev, dtype=torch.bfloat16)
            return bool(torch.matmul(a, a).shape == (2, 2))
        except RuntimeError:
            return False

    def configure_environment(self) -> Dict[str, str]:
        torch.set_default_dtype(torch.float32)
        applied: Dict[str, str] = {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        if self.is_available():
            try:
                self.device_index = self._valid_device_index()
                torch.cuda.set_device(self.device_index)
            except RuntimeError as exc:
                report_exception("cuda.set_device", "torch.cuda.set_device() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")
        return applied

    def get_device(self) -> torch.device:
        if self.is_available():
            return torch.device(f"cuda:{self._valid_device_index()}")
        return torch.device("cpu")

    def get_capabilities(self) -> DeviceCapabilities:
        if not self.is_available():
            return DeviceCapabilities(
                backend_name="NVIDIA CUDA (Offline)",
                device_type="cuda",
                device_name="NVIDIA GeForce RTX 4090 (24GB)",
                device_index=self.device_index,
                total_vram_mb=24576.0,
                supports_bf16=True,
                supports_fp16=True,
                supports_tf32=True,
                supports_fp8=True,
                sdpa_backends=["flash_sdp", "mem_efficient_sdp", "math"],
                distributed_backend="nccl",
            )
        idx = self._valid_device_index()
        props = torch.cuda.get_device_properties(idx)
        total_mb = round(getattr(props, "total_memory", 24 * 1024**3) / (1024**2), 1)
        if total_mb <= 0:
            comfy_mem = query_comfy_device_memory_mb(self.get_device())
            total_mb = comfy_mem[0] if (comfy_mem is not None and comfy_mem[0] > 0) else 24576.0
        return DeviceCapabilities(
            backend_name="NVIDIA CUDA",
            device_type="cuda",
            device_name=getattr(props, "name", "NVIDIA CUDA GPU"),
            device_index=idx,
            total_vram_mb=total_mb,
            supports_bf16=self._probe_bf16_support(),
            supports_fp16=True,
            supports_tf32=bool(getattr(props, "major", 8) >= 8),
            supports_fp8=probe_device_fp8_support(self.get_device()),
            sdpa_backends=["flash_sdp", "mem_efficient_sdp", "math"],
            distributed_backend="nccl",
        )

    def synchronize(self) -> None:
        if self.is_available():
            try:
                torch.cuda.synchronize(self._valid_device_index())
            except RuntimeError as exc:
                report_exception("cuda.synchronize", "torch.cuda.synchronize() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")

    def empty_cache(self) -> None:
        if self.is_available():
            try:
                torch.cuda.synchronize(self._valid_device_index())
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                report_exception("cuda.empty_cache", "torch.cuda.empty_cache() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")

    def get_memory_snapshot(self) -> MemorySnapshot:
        if not self.is_available():
            return MemorySnapshot(0.0, 0.0, 0.0, 24576.0, 24576.0)
        try:
            idx = self._valid_device_index()
            props = torch.cuda.get_device_properties(idx)
            total_mb = round(getattr(props, "total_memory", 24 * 1024**3) / (1024**2), 2)
            alloc_mb = round(torch.cuda.memory_allocated(idx) / (1024**2), 2)
            res_mb = round(torch.cuda.memory_reserved(idx) / (1024**2), 2)
            max_mb = round(torch.cuda.max_memory_allocated(idx) / (1024**2), 2)
            free_mb = max(0.0, round(total_mb - max(alloc_mb, res_mb), 2))
            if hasattr(torch.cuda, "mem_get_info"):
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                    if total_bytes > 0:
                        total_mb = round(total_bytes / (1024**2), 2)
                        # Add back unused cached blocks inside PyTorch's caching allocator pool
                        cached_free_mb = max(0.0, res_mb - alloc_mb)
                        free_mb = round(min(total_mb, (free_bytes / (1024**2)) + cached_free_mb), 2)
                except RuntimeError as exc:
                    report_exception("cuda.mem_get_info", "torch.cuda.mem_get_info() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")
            return MemorySnapshot(
                allocated_mb=alloc_mb,
                reserved_mb=res_mb,
                max_allocated_mb=max_mb,
                total_vram_mb=total_mb,
                free_vram_mb=free_mb,
            )
        except Exception as exc:
            report_exception("cuda.memory_snapshot", "CUDA memory snapshot query failed; using fallback", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")
            return MemorySnapshot(0.0, 0.0, 0.0, 24576.0, 24576.0)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.is_available():
            try:
                torch.cuda.manual_seed_all(seed)
            except RuntimeError as exc:
                report_exception("cuda.manual_seed_all", "torch.cuda.manual_seed_all() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.cuda")

