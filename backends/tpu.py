"""
Google Cloud TPU Hardware Backend (`torch_xla` / `xla:0` — TPU v5e-1, v4, v5p, v6e).
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Tuple
import torch

from .base import (
    DeviceCapabilities,
    HardwareBackend,
    MemorySnapshot,
    probe_device_fp8_support,
    query_comfy_device_memory_mb,
)


def _import_torch_xla() -> Tuple[Any, Any]:
    """Import `(torch_xla, xm)` lazily if `torch_xla` is installed."""
    import torch_xla  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    return torch_xla, xm


class TPUBackend(HardwareBackend):
    """
    Hardware backend for Google Cloud TPU accelerators (TPU v5e-1 16GB HBM, v4, v5p, v6e)
    via PyTorch/XLA (`torch_xla` / `xla:0`).
    """

    @property
    def backend_id(self) -> str:
        return "tpu"

    @property
    def device_type(self) -> str:
        return "xla"

    @classmethod
    def is_available(cls) -> bool:
        try:
            torch_xla, xm = _import_torch_xla()
            if hasattr(torch_xla, "device_count"):
                try:
                    if int(torch_xla.device_count()) > 0:
                        return True
                except Exception:
                    pass
            if hasattr(xm, "get_xla_supported_devices"):
                devs = xm.get_xla_supported_devices()
                return bool(devs and len(devs) > 0)
            return True
        except Exception:
            return False

    def _valid_device_index(self) -> int:
        if not self.is_available():
            return self.device_index
        try:
            torch_xla, xm = _import_torch_xla()
            if hasattr(torch_xla, "device_count"):
                count = int(torch_xla.device_count())
                if count > 0:
                    return max(0, min(int(self.device_index), count - 1))
            if hasattr(xm, "get_xla_supported_devices"):
                devs = xm.get_xla_supported_devices()
                if devs:
                    return max(0, min(int(self.device_index), len(devs) - 1))
        except Exception:
            pass
        return 0

    def configure_environment(self) -> Dict[str, str]:
        torch.set_default_dtype(torch.float32)
        # Keep XLA_USE_BF16=0 so FP32 master optimizer states (RAdamW exp_avg / exp_avg_sq)
        # and loss accumulators retain full 32-bit precision while autocast and bf16 weights
        # execute natively on the TPU bfloat16 Matrix Multiply Units (MXU).
        applied: Dict[str, str] = {
            "PJRT_DEVICE": "TPU",
            "XLA_USE_BF16": "0",
        }
        self.device_index = self._valid_device_index()
        return applied

    def get_device(self) -> torch.device:
        if self.is_available():
            idx = self._valid_device_index()
            try:
                torch_xla, xm = _import_torch_xla()
                if hasattr(torch_xla, "device"):
                    return torch_xla.device(idx)
                return xm.xla_device(idx)
            except Exception:
                return torch.device(f"xla:{idx}")
        return torch.device("cpu")

    def get_capabilities(self) -> DeviceCapabilities:
        # Default profile targets Google Cloud TPU v5e-1 (16 GB HBM, native BF16 MXU)
        total_mb = 16384.0
        dev_name = "Google Cloud TPU v5e-1 (16GB HBM)"
        idx = self._valid_device_index()
        if self.is_available():
            dev = self.get_device()
            try:
                torch_xla, xm = _import_torch_xla()
                if hasattr(xm, "get_memory_info"):
                    mem_info = xm.get_memory_info(dev)
                    if isinstance(mem_info, dict):
                        limit_b = int(
                            mem_info.get("bytes_limit", 0)
                            or mem_info.get("total_memory", 0)
                            or 0
                        )
                        if limit_b > 0:
                            total_mb = round(limit_b / (1024**2), 1)
                hw_kind = ""
                if hasattr(xm, "xla_device_hw"):
                    hw_kind = str(xm.xla_device_hw(dev) or "")
                if hasattr(torch_xla, "_XLAC") and hasattr(torch_xla._XLAC, "_xla_get_device_attributes"):
                    attrs = torch_xla._XLAC._xla_get_device_attributes(str(dev))
                    if isinstance(attrs, dict) and attrs.get("name"):
                        dev_name = f"Google Cloud TPU ({attrs['name']})"
                elif hw_kind:
                    dev_name = f"Google Cloud {hw_kind} ({total_mb / 1024.0:.0f}GB HBM)"
            except Exception:
                pass
            if total_mb == 16384.0:
                comfy_mem = query_comfy_device_memory_mb(dev)
                if comfy_mem is not None and comfy_mem[0] > 0:
                    total_mb = comfy_mem[0]
        return DeviceCapabilities(
            backend_name="Google Cloud TPU (PyTorch/XLA)",
            device_type="xla",
            device_name=dev_name,
            device_index=idx,
            total_vram_mb=total_mb,
            supports_bf16=True,
            supports_fp16=True,
            supports_tf32=False,
            supports_fp8=False if not self.is_available() else probe_device_fp8_support(self.get_device()),
            sdpa_backends=["xla_flash_attention", "math"],
            distributed_backend="xla",
        )

    @contextlib.contextmanager
    def autocast_context(self, enabled: bool = True) -> Iterator[None]:
        if not enabled or self.torch_dtype == torch.float32 or not self.is_available():
            yield
            return
        cm = None
        entered = False
        try:
            cm = torch.amp.autocast(
                device_type="xla",
                dtype=self.torch_dtype,
                enabled=True,
            )
            cm.__enter__()
            entered = True
        except Exception:
            entered = False
        try:
            yield
        finally:
            if entered and cm is not None:
                import sys

                cm.__exit__(*sys.exc_info())

    def synchronize(self) -> None:
        """
        Execute pending lazy XLA graph operations (`torch_xla.sync()` / `xm.mark_step()`)
        and wait for TPU device completion.
        """
        if self.is_available():
            try:
                torch_xla, xm = _import_torch_xla()
                if hasattr(torch_xla, "sync"):
                    torch_xla.sync()
                elif hasattr(xm, "mark_step"):
                    xm.mark_step()
                if hasattr(xm, "wait_device_ops"):
                    xm.wait_device_ops()
            except Exception:
                pass

    def empty_cache(self) -> None:
        if self.is_available():
            self.synchronize()

    def get_memory_snapshot(self) -> MemorySnapshot:
        caps = self.get_capabilities()
        if not self.is_available():
            return MemorySnapshot(
                allocated_mb=0.0,
                reserved_mb=0.0,
                max_allocated_mb=0.0,
                total_vram_mb=caps.total_vram_mb,
                free_vram_mb=caps.total_vram_mb,
            )
        allocated = 0.0
        peak = 0.0
        total = caps.total_vram_mb
        dev = self.get_device()
        try:
            _, xm = _import_torch_xla()
            if hasattr(xm, "get_memory_info"):
                mem_info = xm.get_memory_info(dev)
                if isinstance(mem_info, dict):
                    used_b = int(
                        mem_info.get("bytes_used", 0)
                        or mem_info.get("kb_total", 0) * 1024 - mem_info.get("kb_free", 0) * 1024
                        or 0
                    )
                    peak_b = int(mem_info.get("peak_bytes_used", used_b) or used_b)
                    limit_b = int(
                        mem_info.get("bytes_limit", 0)
                        or mem_info.get("total_memory", 0)
                        or 0
                    )
                    allocated = round(max(0.0, used_b / (1024**2)), 2)
                    peak = round(max(allocated, peak_b / (1024**2)), 2)
                    if limit_b > 0:
                        total = round(limit_b / (1024**2), 1)
        except Exception:
            pass

        free_mb = round(max(0.0, total - allocated), 1)
        comfy_mem = query_comfy_device_memory_mb(dev)
        if comfy_mem is not None and comfy_mem[1] > 0:
            free_mb = min(free_mb, comfy_mem[1]) if free_mb > 0 else comfy_mem[1]
        return MemorySnapshot(
            allocated_mb=allocated,
            reserved_mb=allocated,
            max_allocated_mb=peak,
            total_vram_mb=total,
            free_vram_mb=free_mb,
        )

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.is_available():
            try:
                _, xm = _import_torch_xla()
                if hasattr(xm, "set_rng_state"):
                    xm.set_rng_state(seed, str(self.get_device()))
            except Exception:
                pass

