"""
Intel XPU Hardware Backend (Target: Intel Arc A770 16GB / Battlemage / Data Center Max).

Designed for the `xpu` conda environment with native `torch.xpu` (PyTorch 2.4+)
and optional `intel_extension_for_pytorch` (IPEX) acceleration.
Defaults to `bf16` execution using Intel XMX (Xe Matrix Extensions) systolic arrays,
and strictly avoids `fp64` (`double`), which is not supported in hardware on consumer Arc GPUs.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

from ..diagnostics import report_exception
from .base import (
    DeviceCapabilities,
    HardwareBackend,
    MemorySnapshot,
    probe_device_fp8_support,
    query_comfy_device_memory_mb,
    sanitize_module_no_fp64,
)


class XPUBackend(HardwareBackend):
    """Hardware backend for Intel Arc & Data Center GPUs via `torch.xpu`."""

    @property
    def backend_id(self) -> str:
        return "xpu"

    @property
    def device_type(self) -> str:
        return "xpu"

    @classmethod
    def is_available(cls) -> bool:
        if not hasattr(torch, "xpu"):
            try:
                import intel_extension_for_pytorch as ipex  # noqa: F401
            except ImportError:
                return False
        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())

    def _valid_device_index(self) -> int:
        if not self.is_available():
            return self.device_index
        try:
            count = int(torch.xpu.device_count())
            if count > 0:
                return max(0, min(int(self.device_index), count - 1))
        except (RuntimeError, ValueError, TypeError) as exc:
            report_exception("xpu.device_count", "torch.xpu.device_count() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.xpu")
        return 0

    def _probe_bf16_support(self) -> bool:
        """
        Accurately probe whether `torch.bfloat16` is supported on this Intel XPU.

        Important: Do NOT rely on `props.has_bfloat16_conversions`, which reports
        `False` on Intel Arc Alchemist (A770/A750) even though XMX natively executes
        `torch.bfloat16` matmuls and `torch.xpu.is_bf16_supported()` is `True`.
        """
        if not self.is_available():
            return True

        if hasattr(torch.xpu, "is_bf16_supported"):
            try:
                if bool(torch.xpu.is_bf16_supported()):
                    return True
            except (RuntimeError, AttributeError) as exc:
                report_exception("xpu.is_bf16_supported", "torch.xpu.is_bf16_supported() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.xpu")

        try:
            dev = self.get_device()
            a = torch.ones((2, 2), device=dev, dtype=torch.bfloat16)
            b = (a + a).float()
            return bool(b.shape == (2, 2))
        except RuntimeError:
            return False

    def configure_environment(self) -> Dict[str, str]:
        """
        Configure Intel OneAPI / Level Zero / SYCL runtime profile
        for stable BF16/FP16 LoRA training across Intel XPU devices and guard against FP64
        and Windows WDDM TDR / Level Zero allocation limit crashes.
        """
        torch.set_default_dtype(torch.float32)
        applied: Dict[str, str] = {
            # Persist compiled SYCL/OpenCL kernels across runs to avoid startup JIT stalls
            "SYCL_CACHE_PERSISTENT": "1",
            # Use immediate command lists so Level Zero dispatches incrementally instead of
            # batching giant deferred command lists that trip the Windows 2s WDDM TDR timeout
            "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
            # Enable OneDNN primitive BF16 math mode on Xe Matrix Extensions (XMX)
            "ONEDNN_DEFAULT_FPMATH_MODE": "BF16",
            # Relax single allocation limits in Level Zero / Unified Runtime (both case variants)
            "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS": "1",
            "UR_L0_ENABLE_RELAXED_ALLOCATION_Limits": "1",
            "ZE_L0_ENABLE_RELAXED_ALLOCATION_LIMITS": "1",
            # Fallback DP emulation flags in case any 3rd-party kernel requires FP64
            "IGC_EnableDPEmulation": "1",
            "OverrideDefaultFP64Settings": "1",
        }

        if self.is_available():
            try:
                self.device_index = self._valid_device_index()
                torch.xpu.set_device(self.device_index)
            except RuntimeError as exc:
                report_exception("xpu.set_device", "torch.xpu.set_device() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.xpu")
        return applied

    def get_device(self) -> torch.device:
        if self.is_available():
            return torch.device(f"xpu:{self._valid_device_index()}")
        return torch.device("cpu")

    def get_capabilities(self) -> DeviceCapabilities:
        if not self.is_available():
            return DeviceCapabilities(
                backend_name="Intel XPU (Offline)",
                device_type="xpu",
                device_name="Intel(R) Arc(TM) / Xe XPU Graphics",
                device_index=self.device_index,
                total_vram_mb=16384.0,
                supports_bf16=True,
                supports_fp16=True,
                supports_tf32=False,
                supports_fp64=False,
                supports_fp8=True,
                sdpa_backends=["xpu_onednn_sdpa", "math"],
                distributed_backend="ccl",
                driver_info={
                    "conda_env": "xpu",
                    "xmx_engine": "Xe Matrix Extensions (BF16 Native)",
                    "fp64_hardware": "Disabled (Pure BF16/FP32 Pipeline)",
                    "runtime": "Intel OneAPI Level Zero / SYCL",
                },
            )

        idx = self._valid_device_index()
        props = torch.xpu.get_device_properties(idx)
        total_vram_mb = getattr(props, "total_memory", 16 * 1024**3) / (1024**2)
        if total_vram_mb <= 0:
            comfy_mem = query_comfy_device_memory_mb(self.get_device())
            total_vram_mb = comfy_mem[0] if (comfy_mem is not None and comfy_mem[0] > 0) else 16384.0
        device_name = getattr(props, "name", "Intel(R) XPU Device")
        supports_bf16 = self._probe_bf16_support()
        supports_fp64 = bool(getattr(props, "has_fp64", False))
        supports_fp8 = probe_device_fp8_support(self.get_device())

        return DeviceCapabilities(
            backend_name="Intel XPU (OneAPI / Xe)",
            device_type="xpu",
            device_name=device_name,
            device_index=idx,
            total_vram_mb=round(total_vram_mb, 1),
            supports_bf16=supports_bf16,
            supports_fp16=True,
            supports_tf32=False,
            supports_fp64=supports_fp64,
            supports_fp8=supports_fp8,
            sdpa_backends=["xpu_onednn_sdpa", "math"],
            distributed_backend="ccl",
            driver_info={
                "conda_env": "xpu",
                "torch_version": torch.__version__,
                "supports_fp64": str(supports_fp64),
                "supports_fp8": str(supports_fp8),
                "max_compute_units": str(getattr(props, "max_compute_units", 512)),
            },
        )

    def synchronize(self) -> None:
        if self.is_available():
            try:
                torch.xpu.synchronize(self._valid_device_index())
            except RuntimeError as exc:
                report_exception("xpu.synchronize", "torch.xpu.synchronize() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.xpu")

    def empty_cache(self) -> None:
        if self.is_available():
            try:
                torch.xpu.synchronize(self._valid_device_index())
                torch.xpu.empty_cache()
            except RuntimeError as exc:
                report_exception("xpu.empty_cache", "torch.xpu.empty_cache() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.xpu")

    def get_memory_snapshot(self) -> MemorySnapshot:
        if not self.is_available():
            return MemorySnapshot(
                allocated_mb=0.0,
                reserved_mb=0.0,
                max_allocated_mb=0.0,
                total_vram_mb=16384.0,
                free_vram_mb=16384.0,
            )
        try:
            idx = self._valid_device_index()
            allocated = torch.xpu.memory_allocated(idx) / (1024**2)
            reserved = torch.xpu.memory_reserved(idx) / (1024**2)
            max_alloc = torch.xpu.max_memory_allocated(idx) / (1024**2)
            props = torch.xpu.get_device_properties(idx)
            total = getattr(props, "total_memory", 16 * 1024**3) / (1024**2)
            if total <= 0:
                comfy_mem = query_comfy_device_memory_mb(self.get_device())
                total = comfy_mem[0] if (comfy_mem is not None and comfy_mem[0] > 0) else 16384.0
            free_mb = max(0.0, total - max(allocated, reserved))
            if hasattr(torch.xpu, "mem_get_info"):
                try:
                    free_bytes, total_bytes = torch.xpu.mem_get_info(idx)
                    if total_bytes > 0:
                        total = total_bytes / (1024**2)
                        cached_free = max(0.0, reserved - allocated)
                        free_mb = min(total, (free_bytes / (1024**2)) + cached_free)
                except RuntimeError as exc:
                    report_exception("xpu.mem_get_info", "torch.xpu.mem_get_info() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.xpu")
            return MemorySnapshot(
                allocated_mb=round(allocated, 2),
                reserved_mb=round(reserved, 2),
                max_allocated_mb=round(max_alloc, 2),
                total_vram_mb=round(total, 2),
                free_vram_mb=round(free_mb, 2),
            )
        except Exception as exc:
            report_exception("xpu.memory_snapshot", "XPU memory snapshot query failed; using fallback", exc, level=logging.DEBUG, once=True, logger_name="backends.xpu")
            return MemorySnapshot(0.0, 0.0, 0.0, 16384.0, 16384.0)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.is_available():
            try:
                torch.xpu.manual_seed_all(seed)
            except RuntimeError as exc:
                report_exception("xpu.manual_seed_all", "torch.xpu.manual_seed_all() failed", exc, level=logging.DEBUG, once=True, logger_name="backends.xpu")

    def optimize_module(self, module: nn.Module, mode: str = "eager", move_to_device: bool = True) -> nn.Module:
        module = sanitize_module_no_fp64(module)
        if move_to_device:
            module = module.to(device=self.get_device())
        if mode == "ipex":
            try:
                import intel_extension_for_pytorch as ipex

                module = ipex.optimize(module, dtype=self.torch_dtype, inplace=True)
            except ImportError:
                pass
        elif mode == "compile" and hasattr(torch, "compile"):
            module = torch.compile(module, backend="inductor")
        return module

