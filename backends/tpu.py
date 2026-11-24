"""
Google Cloud TPU Hardware Backend:
- Native `torch.tpu` (`torch-tpu` / `tpu:0`, standard on modern Google Colab TPU runtimes)
- Legacy PyTorch/XLA (`torch_xla` / `xla:0`) fallback.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from typing import Any, Dict, Iterator, Tuple
import torch

from ..diagnostics import report_exception
from .base import (
    DeviceCapabilities,
    HardwareBackend,
    MemorySnapshot,
    probe_device_fp8_support,
    query_comfy_device_memory_mb,
)


def _has_native_torch_tpu() -> bool:
    """Return True if PyTorch exposes Google's native `torch.tpu` (`torch-tpu`) backend and a TPU is available."""
    accel = getattr(torch, "accelerator", None)
    if accel is not None:
        try:
            if callable(getattr(accel, "is_available", None)) and accel.is_available():
                cur = accel.current_accelerator() if callable(getattr(accel, "current_accelerator", None)) else None
                if cur is not None and getattr(cur, "type", str(cur)) == "tpu":
                    return True
        except Exception as exc:
            report_exception(
                "tpu.accelerator_is_available",
                "torch.accelerator TPU probe failed",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="backends.tpu",
            )
    return False


def _import_torch_xla() -> Tuple[Any, Any]:
    """Import `(torch_xla, xm)` lazily if legacy `torch_xla` is installed."""
    import torch_xla  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore

    return torch_xla, xm


class TPUBackend(HardwareBackend):
    """
    Hardware backend for Google Cloud TPU accelerators (TPU v5e-1 16GB HBM, v4, v5p, v6e).
    Automatically prefers native `torch.tpu` (`tpu:0`, `torch-tpu`) when available and falls
    back to legacy PyTorch/XLA (`torch_xla` / `xla:0`).
    """

    @property
    def backend_id(self) -> str:
        return "tpu"

    @property
    def device_type(self) -> str:
        return "tpu" if _has_native_torch_tpu() else "xla"

    @classmethod
    def is_available(cls) -> bool:
        if _has_native_torch_tpu():
            return True
        try:
            torch_xla, xm = _import_torch_xla()
            if hasattr(torch_xla, "device_count"):
                try:
                    if int(torch_xla.device_count()) > 0:
                        return True
                except (RuntimeError, ValueError, TypeError) as exc:
                    report_exception(
                        "tpu.device_count_avail",
                        "torch_xla.device_count() failed",
                        exc,
                        level=logging.DEBUG,
                        once=True,
                        logger_name="backends.tpu",
                    )
            if hasattr(xm, "get_xla_supported_devices"):
                devs = xm.get_xla_supported_devices()
                return bool(devs and len(devs) > 0)
            return True
        except ImportError:
            return False
        except Exception as exc:
            report_exception(
                "tpu.is_available",
                "TPU availability probe failed",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="backends.tpu",
            )
            return False

    def _valid_device_index(self) -> int:
        if _has_native_torch_tpu():
            try:
                accel = getattr(torch, "accelerator", None)
                if accel is not None and callable(getattr(accel, "device_count", None)):
                    count = int(accel.device_count())
                    if count > 0:
                        return max(0, min(int(self.device_index), count - 1))
            except Exception as exc:
                report_exception(
                    "tpu.native_valid_device_index",
                    "torch.accelerator.device_count() failed",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )
            return 0
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
        except Exception as exc:
            report_exception(
                "tpu.valid_device_index",
                "TPU device index query failed",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="backends.tpu",
            )
        return 0

    def configure_environment(self) -> Dict[str, str]:
        torch.set_default_dtype(torch.float32)
        if _has_native_torch_tpu():
            warnings.filterwarnings(
                "ignore",
                message=r"The \.grad attribute of a Tensor that is not a leaf Tensor is being accessed",
                category=UserWarning,
            )
            self.device_index = self._valid_device_index()
            return {"TPU_BACKEND": "torch.tpu"}
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
        if _has_native_torch_tpu():
            return torch.device(f"tpu:{self._valid_device_index()}")
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

        if _has_native_torch_tpu():
            dev = self.get_device()
            tpu_mod = getattr(torch, "tpu")
            try:
                if hasattr(tpu_mod, "get_device_name"):
                    name_val = tpu_mod.get_device_name(idx)
                    if name_val:
                        dev_name = f"Google Cloud TPU ({name_val})"
                if hasattr(tpu_mod, "get_device_properties"):
                    props = tpu_mod.get_device_properties(idx)
                    mem_b = int(
                        getattr(props, "total_memory", 0)
                        or getattr(props, "hbm_size", 0)
                        or getattr(props, "memory_size", 0)
                        or 0
                    )
                    if mem_b > 0:
                        total_mb = round(mem_b / (1024**2), 1)
            except Exception as exc:
                report_exception(
                    "tpu.native_capabilities_probe",
                    "torch.tpu capabilities query failed; using defaults",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )
            if total_mb == 16384.0:
                comfy_mem = query_comfy_device_memory_mb(dev)
                if comfy_mem is not None and comfy_mem[0] > 0:
                    total_mb = comfy_mem[0]
            return DeviceCapabilities(
                backend_name="Google Cloud TPU (torch.tpu)",
                device_type="tpu",
                device_name=dev_name,
                device_index=idx,
                total_vram_mb=total_mb,
                supports_bf16=True,
                supports_fp16=True,
                supports_tf32=False,
                supports_fp8=probe_device_fp8_support(dev),
                sdpa_backends=["tpu_flash_attention", "math"],
                distributed_backend="tpu",
            )

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
            except Exception as exc:
                report_exception(
                    "tpu.capabilities_probe",
                    "TPU capabilities query failed; using defaults",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )
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
                device_type=self.device_type,
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
        """Wait for pending TPU operations (`torch.accelerator.synchronize()` or `torch_xla.sync()`/`xm.mark_step()`)."""
        if _has_native_torch_tpu():
            try:
                accel = getattr(torch, "accelerator", None)
                if accel is not None and callable(getattr(accel, "synchronize", None)):
                    accel.synchronize()
                    return
                tpu_mod = getattr(torch, "tpu", None)
                if tpu_mod is not None and callable(getattr(tpu_mod, "synchronize", None)):
                    tpu_mod.synchronize()
            except Exception as exc:
                report_exception(
                    "tpu.native_synchronize",
                    "TPU synchronize() failed",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )
            return
        if self.is_available():
            try:
                torch_xla, xm = _import_torch_xla()
                if hasattr(torch_xla, "sync"):
                    torch_xla.sync()
                elif hasattr(xm, "mark_step"):
                    xm.mark_step()
                if hasattr(xm, "wait_device_ops"):
                    xm.wait_device_ops()
            except Exception as exc:
                report_exception(
                    "tpu.synchronize",
                    "TPU synchronize/mark_step failed",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )

    def empty_cache(self) -> None:
        if _has_native_torch_tpu():
            accel = getattr(torch, "accelerator", None)
            if accel is not None and callable(getattr(accel, "empty_cache", None)):
                try:
                    accel.empty_cache()
                    return
                except Exception as exc:
                    report_exception(
                        "tpu.native_empty_cache",
                        "torch.accelerator.empty_cache() failed",
                        exc,
                        level=logging.DEBUG,
                        once=True,
                        logger_name="backends.tpu",
                    )
            self.synchronize()
            return
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

        if _has_native_torch_tpu():
            tpu_mod = getattr(torch, "tpu")
            try:
                if hasattr(tpu_mod, "memory_allocated"):
                    allocated = round(max(0.0, float(tpu_mod.memory_allocated(dev)) / (1024**2)), 2)
                if hasattr(tpu_mod, "max_memory_allocated"):
                    peak = round(max(allocated, float(tpu_mod.max_memory_allocated(dev)) / (1024**2)), 2)
                if hasattr(tpu_mod, "mem_get_info"):
                    free_b, total_b = tpu_mod.mem_get_info(dev)
                    if total_b > 0:
                        total = round(float(total_b) / (1024**2), 1)
                        free_mb = round(max(0.0, float(free_b) / (1024**2)), 1)
                        return MemorySnapshot(
                            allocated_mb=allocated,
                            reserved_mb=allocated,
                            max_allocated_mb=peak,
                            total_vram_mb=total,
                            free_vram_mb=free_mb,
                        )
            except Exception as exc:
                report_exception(
                    "tpu.native_memory_snapshot",
                    "torch.tpu memory query failed",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )
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
        except Exception as exc:
            report_exception(
                "tpu.memory_snapshot",
                "TPU memory info query failed",
                exc,
                level=logging.DEBUG,
                once=True,
                logger_name="backends.tpu",
            )

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
        if _has_native_torch_tpu():
            try:
                tpu_mod = getattr(torch, "tpu")
                if hasattr(tpu_mod, "manual_seed_all"):
                    tpu_mod.manual_seed_all(seed)
                elif hasattr(tpu_mod, "manual_seed"):
                    tpu_mod.manual_seed(seed)
            except Exception as exc:
                report_exception(
                    "tpu.native_manual_seed_all",
                    "torch.tpu.manual_seed_all() failed",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )
            return
        if self.is_available():
            try:
                _, xm = _import_torch_xla()
                if hasattr(xm, "set_rng_state"):
                    xm.set_rng_state(seed, str(self.get_device()))
            except Exception as exc:
                report_exception(
                    "tpu.manual_seed_all",
                    "TPU set_rng_state() failed",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )

    def optimize_module(self, module: torch.nn.Module, mode: str = "eager", move_to_device: bool = True) -> torch.nn.Module:
        """
        Optimize a module for TPU execution. When `mode == "compile"` on native `torch.tpu`,
        attempts `torch.compile(..., backend="openxla")` (falling back to default `torch.compile`
        and then eager mode if compilation is unsupported for the graph).
        """
        from .base import sanitize_module_no_fp64

        module = sanitize_module_no_fp64(module)
        if move_to_device:
            module = module.to(device=self.get_device())
        if mode == "compile" and hasattr(torch, "compile"):
            try:
                if _has_native_torch_tpu():
                    return torch.compile(module, backend="openxla")
                return torch.compile(module)
            except Exception as exc:
                report_exception(
                    "tpu.optimize_module_compile",
                    "TPU torch.compile() failed; falling back to eager mode",
                    exc,
                    level=logging.WARNING,
                    once=True,
                    logger_name="backends.tpu",
                )
        return module


_COMFY_TPU_HOOKS_INSTALLED: bool = False


def install_comfy_tpu_hooks() -> bool:
    """
    Automatically hook ComfyUI's `comfy.model_management` onto Google's native `torch.tpu`
    (`tpu:0`) in `bfloat16` when running inside ComfyUI on a Cloud TPU runtime.

    ComfyUI imports `comfy.model_management` before loading `custom_nodes`, so TPU users
    should start vanilla ComfyUI with:
        `python main.py --cpu --use-pytorch-cross-attention`
    Once `omni_lora` is imported, this function automatically upgrades ComfyUI from `--cpu`
    to `tpu:0` in native `bfloat16` without needing an external launcher wrapper.

    No-op on non-TPU systems or when `OMNI_LORA_DISABLE_COMFY_TPU_HOOK=1` is set.
    """
    global _COMFY_TPU_HOOKS_INSTALLED
    if _COMFY_TPU_HOOKS_INSTALLED:
        return True
    if os.environ.get("OMNI_LORA_DISABLE_COMFY_TPU_HOOK", "").strip().lower() in ("1", "true", "yes"):
        return False
    has_native = _has_native_torch_tpu()
    has_xla = False
    xla_xm: Any = None
    xla_mod: Any = None
    if not has_native:
        try:
            xla_mod, xla_xm = _import_torch_xla()
            has_xla = TPUBackend.is_available()
        except ImportError:
            has_xla = False
    if not has_native and not has_xla:
        return False

    warnings.filterwarnings("ignore", message=r".*TorchTpu/ApiStage/ExperimentalApi.*", category=UserWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"The \.grad attribute of a Tensor that is not a leaf Tensor is being accessed",
        category=UserWarning,
    )

    try:
        import comfy.cli_args as cli_args  # type: ignore

        if "comfy.model_management" not in sys.modules:
            cli_args.args.cpu = True
        import comfy.model_management as mm  # type: ignore
    except ImportError:
        return False

    if has_native:
        tpu_dev = torch.device("tpu:0")
        tpu_autocast_type = "tpu"
        tpu_name = "Cloud TPU"
        tpu_mod = getattr(torch, "tpu", None)
        if tpu_mod is not None and callable(getattr(tpu_mod, "get_device_name", None)):
            try:
                name_val = tpu_mod.get_device_name(0)
                if name_val:
                    tpu_name = str(name_val)
            except Exception as exc:
                report_exception(
                    "tpu.comfy_hook_device_name",
                    "torch.tpu.get_device_name(0) failed during ComfyUI TPU hook setup",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )
        _sync_fn = lambda: torch.accelerator.synchronize()
    else:
        tpu_dev = (
            xla_mod.device(0)
            if (xla_mod is not None and hasattr(xla_mod, "device"))
            else xla_xm.xla_device(0)
        )
        tpu_autocast_type = "xla"
        tpu_name = "Cloud TPU (XLA)"
        tpu_mod = None
        _sync_fn = (
            (lambda: xla_mod.sync())
            if (xla_mod is not None and hasattr(xla_mod, "sync"))
            else (lambda: xla_xm.mark_step())
        )

    # Guard third-party nodes that call torch.cuda.empty_cache/synchronize blindly on non-CUDA builds
    if not torch.cuda.is_available():
        torch.cuda.empty_cache = _sync_fn  # type: ignore[assignment]
        torch.cuda.synchronize = lambda device=None: _sync_fn()  # type: ignore[assignment]

    cli_args.args.cpu = False
    mm.cpu_state = mm.CPUState.GPU
    mm.get_torch_device = lambda: tpu_dev
    mm.get_torch_device_name = lambda device=None: f"TPU ({tpu_name})"
    mm.get_autocast_device = lambda dev: "cpu" if getattr(dev, "type", str(dev)) == "cpu" else tpu_autocast_type
    # Keep text encoders on CPU by default on TPU because recurrent/quantized LLM encoders
    # (e.g. Qwen3.5 Gated DeltaNet) unroll Python token loops into massive XLA graphs that stall
    # compilation. Set OMNI_LORA_TPU_TEXT_ENCODER=1 to run standard CLIP/T5 encoders on tpu:0.
    te_on_tpu = os.environ.get("OMNI_LORA_TPU_TEXT_ENCODER", "").strip().lower() in ("1", "true", "yes")
    mm.text_encoder_device = (lambda: tpu_dev) if te_on_tpu else (lambda: torch.device("cpu"))
    mm.vae_device = lambda: tpu_dev
    mm.unet_offload_device = lambda: torch.device("cpu")
    mm.text_encoder_offload_device = lambda: torch.device("cpu")
    mm.vae_offload_device = lambda: torch.device("cpu")

    mm.should_use_bf16 = lambda device=None, **kwargs: True
    mm.should_use_fp16 = lambda device=None, **kwargs: False
    mm.unet_dtype = lambda device=None, model_dtype=torch.bfloat16, **kwargs: torch.bfloat16
    mm.text_encoder_dtype = lambda device=None: torch.bfloat16
    mm.vae_dtype = lambda device=None, allowed_dtypes=(): torch.bfloat16

    def _tpu_total_mem(device: Any = None, torch_total_too: bool = False) -> Any:
        if device is not None and getattr(device, "type", str(device)) == "cpu":
            import psutil

            total = int(psutil.virtual_memory().total)
            return (total, total) if torch_total_too else total
        total = 16 * 1024**3
        if tpu_mod is not None and callable(getattr(tpu_mod, "get_device_properties", None)):
            try:
                props = tpu_mod.get_device_properties(0)
                mem = (
                    getattr(props, "total_memory", 0)
                    or getattr(props, "hbm_size", 0)
                    or getattr(props, "memory_size", 0)
                    or 0
                )
                if mem > 0:
                    total = int(mem)
            except Exception as exc:
                report_exception(
                    "tpu.comfy_hook_total_mem",
                    "torch.tpu.get_device_properties(0) failed; using 16GB HBM default",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="backends.tpu",
                )
        return (total, total) if torch_total_too else total

    def _tpu_free_mem(device: Any = None, torch_free_too: bool = False) -> Any:
        if device is not None and getattr(device, "type", str(device)) == "cpu":
            import psutil

            free = int(psutil.virtual_memory().available)
            return (free, free) if torch_free_too else free
        free = int(_tpu_total_mem(tpu_dev) * 0.97)
        return (free, free) if torch_free_too else free

    mm.get_total_memory = _tpu_total_mem
    mm.get_free_memory = _tpu_free_mem
    mm.total_vram = _tpu_total_mem(tpu_dev) / (1024.0 * 1024.0)
    mm.vram_state = mm.VRAMState.HIGH_VRAM if mm.total_vram >= 24000.0 else mm.VRAMState.NORMAL_VRAM
    mm.soft_empty_cache = lambda force=False: _sync_fn()

    # Ensure ComfyUI uses native PyTorch SDPA instead of attention_sub_quad even if
    # --use-pytorch-cross-attention was omitted on the command line.
    try:
        import comfy.ldm.modules.attention as comfy_attn  # type: ignore

        attn_pt = getattr(comfy_attn, "attention_pytorch", None)
        if callable(attn_pt):
            comfy_attn.optimized_attention = attn_pt
            comfy_attn.optimized_attention_masked = attn_pt
    except ImportError:
        pass

    # Deduplicate default root StreamHandler attached by absl/torch_tpu alongside ComfyUI's logger
    root_logger = logging.getLogger()
    if len(root_logger.handlers) > 1:
        kept = [
            h
            for h in root_logger.handlers
            if not (type(h) is logging.StreamHandler and h.formatter is None)
        ]
        if kept and len(kept) < len(root_logger.handlers):
            root_logger.handlers[:] = kept

    _COMFY_TPU_HOOKS_INSTALLED = True
    logging.info(
        "[OmniLoRA] ComfyUI model_management hooked to %s (%s, %.0f MB HBM) in native bfloat16.",
        tpu_dev,
        tpu_name,
        mm.total_vram,
    )
    return True
