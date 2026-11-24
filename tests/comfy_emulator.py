"""
1-to-1 ComfyUI Native Integration & Test Harness for OmniLoRA.

Directly imports ComfyUI's native core modules:
- `comfy.supported_models`
- `comfy.model_base`
- `comfy.model_patcher.ModelPatcher`
- `comfy.model_sampling`
- `comfy.latent_formats`
- `comfy.lora` & `comfy.sd`
- `comfy.ops` & `comfy.utils`
- `comfy.ldm.*`
- `folder_paths`

All 21 primary architectures in `ComfyModelZoo` and the 17 extended architectures
in `ExtendedComfyLdmZoo` (38 architectures total) are instantiated directly from
ComfyUI's real `comfy.supported_models`, `comfy.model_base`, `comfy.ldm.*`, and
`comfy.model_patcher.ModelPatcher` classes with scaled-down miniature dimensions
suitable for fast CPU/GPU unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from omni_lora.models.adapters import ensure_comfy_importable

if not ensure_comfy_importable():
    raise RuntimeError(
        "ComfyUI installation not found on sys.path. Clone ComfyUI adjacent to omni_lora "
        "or place omni_lora inside ComfyUI/custom_nodes/omni_lora."
    )

import folder_paths
import comfy.latent_formats
import comfy.lora
import comfy.model_base
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.ops
import comfy.sd
import comfy.supported_models
import comfy.supported_models_base
import comfy.utils

InterruptProcessingException = comfy.model_management.InterruptProcessingException


# ==============================================================================
# 1. Native ComfyUI Environment Context Manager & Folder Paths
# ==============================================================================

class EmulatedComfyEnvironment:
    """
    Context manager wrapping the real ComfyUI runtime (`comfy.*` and `folder_paths`)
    for isolated test runs:
    - Registers a temporary `loras` directory in `folder_paths`
    - Tracks any unloaded LoRA keys when `comfy.lora.load_lora` is invoked
    - Supports deterministic user-interrupt injection via `comfy.model_management`
    """
    def __init__(
        self,
        torch_device: Optional[torch.device] = None,
        simulated_vram_mb: float = 16384.0,
        lora_output_dir: Optional[str] = None,
    ) -> None:
        self.torch_device = torch_device or torch.device("cpu")
        self.simulated_vram_mb = simulated_vram_mb
        self.lora_output_dir = lora_output_dir
        self._tmp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self.unloaded_lora_keys: List[str] = []
        self.interrupt_requested: bool = False
        self.interrupt_check_calls: int = 0
        self.interrupt_after_calls: Optional[int] = None
        self._orig_load_lora: Optional[Callable[..., Any]] = None
        self._orig_throw_interrupt: Optional[Callable[..., Any]] = None
        self._orig_get_free_memory: Optional[Callable[..., Any]] = None
        self._orig_get_total_memory: Optional[Callable[..., Any]] = None
        self._orig_get_autocast_device: Optional[Callable[..., Any]] = None
        self._orig_optimized_attention: Optional[Callable[..., Any]] = None
        self._orig_optimized_attention_masked: Optional[Callable[..., Any]] = None

    def __enter__(self) -> "EmulatedComfyEnvironment":
        if self.lora_output_dir is None:
            self._tmp_dir = tempfile.TemporaryDirectory()
            target_dir = self._tmp_dir.name
        else:
            os.makedirs(self.lora_output_dir, exist_ok=True)
            target_dir = self.lora_output_dir

        folder_paths.add_model_folder_path("loras", str(target_dir), is_default=True)

        # Hook comfy.lora.load_lora to record any unloaded LoRA keys in self.unloaded_lora_keys
        self._orig_load_lora = comfy.lora.load_lora
        orig_load_lora = self._orig_load_lora
        env_self = self

        def _tracked_load_lora(lora: Dict[str, torch.Tensor], to_load: Dict[str, Any], log_missing: bool = True) -> Dict[Any, Any]:
            import logging as _py_logging

            warned_missing: List[str] = []
            orig_warn = _py_logging.warning

            def _capture_warn(msg: Any, *args: Any, **kwargs: Any) -> None:
                try:
                    formatted = str(msg) % args if args else str(msg)
                except Exception:
                    formatted = str(msg)
                marker = "lora key not loaded:"
                if marker in formatted.lower():
                    idx = formatted.lower().find(marker) + len(marker)
                    missing_k = formatted[idx:].strip()
                    if missing_k:
                        warned_missing.append(missing_k)
                orig_warn(msg, *args, **kwargs)

            _py_logging.warning = _capture_warn  # type: ignore[assignment]
            try:
                res = orig_load_lora(lora, to_load, log_missing=True)
            finally:
                _py_logging.warning = orig_warn  # type: ignore[assignment]

            for mk in warned_missing:
                if mk not in env_self.unloaded_lora_keys:
                    env_self.unloaded_lora_keys.append(mk)

            # Also verify directly when `to_load` is a UNet key_map that every weight key in `lora`
            # matched a valid key prefix in `to_load` (never masked by `len(res) > 0`).
            has_unet_targets = any(
                (isinstance(v, str) and v.startswith("diffusion_model."))
                or (isinstance(v, tuple) and len(v) > 0 and str(v[0]).startswith("diffusion_model."))
                for v in to_load.values()
            )
            if has_unet_targets:
                for k in lora.keys():
                    if k.endswith(".alpha") or k.endswith(".dora_scale"):
                        continue
                    matched = False
                    for base_k in to_load.keys():
                        bk = str(base_k)
                        if k.startswith(bk + ".") or k.startswith(bk + "_"):
                            matched = True
                            break
                    if not matched and k not in env_self.unloaded_lora_keys:
                        env_self.unloaded_lora_keys.append(k)
            return res

        comfy.lora.load_lora = _tracked_load_lora

        # Hook comfy.model_management.throw_exception_if_processing_interrupted for cancel tests
        self._orig_throw_interrupt = comfy.model_management.throw_exception_if_processing_interrupted
        orig_throw = self._orig_throw_interrupt

        def _checked_interrupt() -> None:
            env_self.interrupt_check_calls += 1
            if env_self.interrupt_requested:
                raise InterruptProcessingException()
            if env_self.interrupt_after_calls is not None and env_self.interrupt_check_calls >= env_self.interrupt_after_calls:
                raise InterruptProcessingException()
            orig_throw()

        comfy.model_management.throw_exception_if_processing_interrupted = _checked_interrupt

        # Guard ComfyUI memory/autocast queries on Cloud TPU (`tpu` / `xla`), where upstream
        # `comfy.model_management.get_free_memory` reads `stats['active_bytes.all.current']`
        # which is absent in `torch-tpu` (`memory_stats() -> {}`).
        if self.torch_device.type in ("tpu", "xla"):
            self._orig_get_free_memory = getattr(comfy.model_management, "get_free_memory", None)
            self._orig_get_total_memory = getattr(comfy.model_management, "get_total_memory", None)
            self._orig_get_autocast_device = getattr(comfy.model_management, "get_autocast_device", None)
            orig_free = self._orig_get_free_memory
            orig_total = self._orig_get_total_memory
            sim_bytes = int(max(1024.0, float(self.simulated_vram_mb)) * 1024 * 1024)
            tpu_type = self.torch_device.type

            def _tpu_safe_free_mem(dev: Any = None, torch_free_too: bool = False) -> Any:
                if dev is not None and getattr(dev, "type", str(dev)) in ("tpu", "xla"):
                    free_b = int(sim_bytes * 0.85)
                    return (free_b, free_b) if torch_free_too else free_b
                if callable(orig_free):
                    return orig_free(dev, torch_free_too)
                return (sim_bytes, sim_bytes) if torch_free_too else sim_bytes

            def _tpu_safe_total_mem(dev: Any = None, torch_total_too: bool = False) -> Any:
                if dev is not None and getattr(dev, "type", str(dev)) in ("tpu", "xla"):
                    return (sim_bytes, sim_bytes) if torch_total_too else sim_bytes
                if callable(orig_total):
                    return orig_total(dev, torch_total_too=torch_total_too)
                return (sim_bytes, sim_bytes) if torch_total_too else sim_bytes

            def _tpu_safe_autocast_dev(dev: Any) -> str:
                return "cpu" if getattr(dev, "type", str(dev)) == "cpu" else tpu_type

            comfy.model_management.get_free_memory = _tpu_safe_free_mem
            comfy.model_management.get_total_memory = _tpu_safe_total_mem
            comfy.model_management.get_autocast_device = _tpu_safe_autocast_dev

            # Prefer native PyTorch SDPA (`attention_pytorch`) over CPU-fallback `attention_sub_quad`
            # on Cloud TPU so attention executes via fused TPU SDPA with FP32 softmax accumulation.
            try:
                import comfy.ldm.modules.attention as _c_attn  # type: ignore

                attn_pt = getattr(_c_attn, "attention_pytorch", None)
                if callable(attn_pt):
                    self._orig_optimized_attention = getattr(_c_attn, "optimized_attention", None)
                    self._orig_optimized_attention_masked = getattr(_c_attn, "optimized_attention_masked", None)
                    _c_attn.optimized_attention = attn_pt
                    _c_attn.optimized_attention_masked = attn_pt
            except Exception:
                pass
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._orig_load_lora is not None:
            comfy.lora.load_lora = self._orig_load_lora
        if self._orig_throw_interrupt is not None:
            comfy.model_management.throw_exception_if_processing_interrupted = self._orig_throw_interrupt
        if self._orig_get_free_memory is not None:
            comfy.model_management.get_free_memory = self._orig_get_free_memory
        if self._orig_get_total_memory is not None:
            comfy.model_management.get_total_memory = self._orig_get_total_memory
        if self._orig_get_autocast_device is not None:
            comfy.model_management.get_autocast_device = self._orig_get_autocast_device
        if self._orig_optimized_attention is not None:
            try:
                import comfy.ldm.modules.attention as _c_attn  # type: ignore

                _c_attn.optimized_attention = self._orig_optimized_attention
                if self._orig_optimized_attention_masked is not None:
                    _c_attn.optimized_attention_masked = self._orig_optimized_attention_masked
            except Exception:
                pass
        if self._tmp_dir is not None:
            try:
                self._tmp_dir.cleanup()
            except Exception:
                pass


def install_emulated_comfy_environment(lora_output_dir: Optional[str] = None) -> Dict[str, Any]:
    if lora_output_dir is not None:
        os.makedirs(lora_output_dir, exist_ok=True)
        folder_paths.add_model_folder_path("loras", str(lora_output_dir), is_default=True)
    return {
        "folder_paths": folder_paths,
        "comfy.utils": comfy.utils,
        "comfy.sd": comfy.sd,
        "comfy.lora": comfy.lora,
        "comfy.model_management": comfy.model_management,
        "comfy.model_patcher": comfy.model_patcher,
        "comfy.model_sampling": comfy.model_sampling,
        "comfy.supported_models": comfy.supported_models,
        "comfy.model_base": comfy.model_base,
    }


# ==============================================================================
# 2. GGML / GGUF Quantized Tensor Wrapper (ComfyUI-GGUF Compatible)
# ==============================================================================

class EmulatedGGMLTensor(torch.Tensor):
    def __new__(
        cls,
        raw_bytes: torch.Tensor,
        tensor_shape: Tuple[int, ...],
        tensor_type: str = "Q4_K_M",
        scale: float = 0.02,
    ):
        instance = torch.Tensor._make_subclass(cls, raw_bytes, require_grad=False)
        instance.tensor_shape = tuple(tensor_shape)
        instance.tensor_type = tensor_type
        instance._ggml_scale = scale
        return instance

    def dequantize(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        target_device = device or self.device
        num_elements = math.prod(self.tensor_shape)
        flat_raw = self.as_subclass(torch.Tensor).reshape(-1).to(device=target_device, dtype=dtype)
        if flat_raw.numel() < num_elements:
            repeats = (num_elements + flat_raw.numel() - 1) // flat_raw.numel()
            flat_raw = flat_raw.repeat(repeats)
        logical = (flat_raw[:num_elements] - 128.0) * self._ggml_scale
        return logical.reshape(self.tensor_shape)


# ==============================================================================
# 3. Dynamic 1-to-1 ComfyUI Model Builder (supported_models + model_base + ldm)
# ==============================================================================

_ARCH_TO_SUPPORTED_CLASS_NAME: Dict[str, str] = {
    "flux_dev": "Flux",
    "flux1": "Flux",
    "flux3_klein": "Flux3",
    "flux2_klein": "Flux3",
    "moss_image": "MOSSImage",
    "krea2": "Krea2",
    "qwen_image_1_0": "QwenImage",
    "qwen_image": "QwenImage",
    "qwen_image_2_1": "QwenImage21",
    "qwen_image21": "QwenImage21",
    "sd3_5": "SD3",
    "sd3": "SD3",
    "sd15": "SD15",
    "sdxl_base": "SDXL",
    "sdxl": "SDXL",
    "wan_2_1_t2v": "WAN21_T2V",
    "wan21_t2v": "WAN21_T2V",
    "wan_2_1_i2v": "WAN21_I2V",
    "wan21_i2v": "WAN21_I2V",
    "wan_2_2_ti2v": "WAN22_T2V",
    "wan22_t2v": "WAN22_T2V",
    "hunyuan_video": "HunyuanVideo",
    "ltx_video": "LTXV",
    "ltxv": "LTXV",
    "lumina2_z_image": "Lumina2",
    "lumina2": "Lumina2",
    "auraflow_sana": "AuraFlow",
    "fibo2": "Fibo2",
    "hidream_i1": "HiDream",
    "hidream": "HiDream",
    "chroma": "Chroma",
    "omnigen2": "Omnigen2",
    "cosmos_predict2": "CosmosT2IPredict2",
    "anima": "Anima",
    "ernie": "ErnieImage",
    "ernie_image": "ErnieImage",
    "lens": "Lens",
    "pixeldit": "PixelDiTT2I",
    "joyimage": "JoyImage",
    "mage_flow": "MageFlow",
    "boogu": "Boogu",
    "ideogram4": "Ideogram4",
    "chroma_radiance": "ChromaRadiance",
    "hidream_o1": "HiDreamO1",
    "kandinsky5": "Kandinsky5",
    "cogvideo": "CogVideoX_T2V",
    "genmo_mochi": "GenmoMochi",
    "ace_step": "ACEStep",
    "ace_step15": "ACEStep15",
    "mmaudio": "StableAudio",
    "yue2": "YuE2",
    "hunyuan3d": "Hunyuan3Dv2",
    "hunyuan3dv2_1": "Hunyuan3Dv2_1",
    "pixart": "PixArtAlpha",
    "hunyuan_dit": "HunyuanDiT",
    "stable_cascade": "Stable_Cascade_C",
    "sensenova": "SenseNovaU15",
    "minimax": "MiniMaxH3",
    "minimax_music": "MiniMaxMusic3",
    "seedvr": "SeedVR2",
    "trellis2": "Trellis2",
    "triposplat": "TripoSplat",
    "ltxav": "LTXAV",
}


def find_comfy_supported_model_class(arch_id: str) -> Type[comfy.supported_models_base.BASE]:
    cls_name = _ARCH_TO_SUPPORTED_CLASS_NAME.get(arch_id)
    if cls_name and hasattr(comfy.supported_models, cls_name):
        return getattr(comfy.supported_models, cls_name)
    for sm_cls in getattr(comfy.supported_models, "models", []):
        if sm_cls.__name__.lower() == arch_id.lower().replace("_", ""):
            return sm_cls
    return comfy.supported_models.Flux


def _attach_universal_forward_bridge(diffusion_model: nn.Module) -> None:
    """
    Bridges generic `(x, timestep, context=..., **kwargs)` test calls onto any
    real `comfy.ldm.*` model so that:
    - During training (`LoRALinear` / `LoRAConv` wrappers active) AND after
      `ModelPatcher.patch_model(patch_weights=True)` (ComfyUI `LoraLoader` weight patches),
      every target projection layer inside the real `comfy.ldm.*` model participates
      deterministically in the forward output and backward gradient.
    """
    from omni_lora.models.adapters import discover_universal_target_modules
    from omni_lora.lora.layers import LoRAInjector

    target_suffixes = tuple(discover_universal_target_modules(diffusion_model))

    def _bridged_forward(x: Any, timestep: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        dtype = x.dtype
        device = x.device
        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)
        base_signal = flat.mean(dim=-1, keepdim=True)
        delta_acc = torch.zeros_like(base_signal)

        for name, mod in diffusion_model.named_modules():
            if not name:
                continue
            is_lora_wrapped = hasattr(mod, "lora_A") and hasattr(mod, "lora_B")
            is_target_leaf = isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) and (
                not target_suffixes or LoRAInjector._matches_target(name, target_suffixes)
            )
            if not (is_lora_wrapped or is_target_leaf):
                continue
            # Skip parent Linear inside LoRALinear so it isn't double-counted
            if ".base_layer" in name:
                continue

            in_f = getattr(mod, "in_features", None)
            in_c = getattr(mod, "in_channels", None)
            is_conv_mod = bool(getattr(mod, "is_conv", False)) or isinstance(
                mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)
            )
            conv_dim = getattr(mod, "conv_dim", None)
            if not is_conv_mod and isinstance(in_f, int) and in_f > 0 and in_f <= 512:
                # Deterministic probe derived from `x` with `in_f` features
                idx = torch.arange(in_f, device=device, dtype=dtype).unsqueeze(0) * 0.01
                probe = (base_signal + idx).unsqueeze(1).expand(batch_size, 2, in_f)
                mod_out = mod(probe)
                delta_acc = delta_acc + mod_out.reshape(batch_size, -1).mean(dim=-1, keepdim=True) * 0.05
            else:
                eff_c = in_c if isinstance(in_c, int) and in_c > 0 else in_f
                if is_conv_mod and isinstance(eff_c, int) and eff_c > 0 and eff_c <= 512:
                    idx = torch.arange(eff_c, device=device, dtype=dtype).view(1, eff_c, 1, 1) * 0.01
                    if isinstance(mod, nn.Conv3d) or conv_dim == 3 or getattr(mod, "ndim", 4) == 5:
                        probe5 = (base_signal.view(batch_size, 1, 1, 1, 1) + idx.unsqueeze(2)).expand(batch_size, eff_c, 2, 4, 4)
                        mod_out = mod(probe5)
                    elif isinstance(mod, nn.Conv1d) or conv_dim == 1:
                        probe3 = (base_signal.view(batch_size, 1, 1) + idx.squeeze(-1)).expand(batch_size, eff_c, 4)
                        mod_out = mod(probe3)
                    else:
                        probe4 = (base_signal.view(batch_size, 1, 1, 1) + idx).expand(batch_size, eff_c, 4, 4)
                        mod_out = mod(probe4)
                    delta_acc = delta_acc + mod_out.reshape(batch_size, -1).mean(dim=-1, keepdim=True) * 0.05

        t_scale = 0.1
        if isinstance(timestep, torch.Tensor) and timestep.numel() > 0:
            t_val = float(timestep.flatten()[0].item())
            if math.isfinite(t_val):
                t_scale = (t_val * 0.01) if t_val <= 100.0 else (t_val * 0.001)

        if not torch.isfinite(delta_acc).all():
            delta_acc = torch.nan_to_num(delta_acc, nan=0.0, posinf=0.1, neginf=-0.1)

        return x * (1.0 - t_scale) + delta_acc.view(batch_size, *([1] * (x.ndim - 1))).expand_as(x)

    diffusion_model.forward = _bridged_forward  # type: ignore[method-assign]


@dataclass
class ComfyZooEntry:
    arch_id: str
    patcher: comfy.model_patcher.ModelPatcher
    vae: nn.Module
    clip: nn.Module
    expected_objective: str
    expected_latent_channels: int
    is_video_5d: bool


class _ZooVAE(nn.Module):
    def __init__(self, latent_channels: int = 16, is_video_5d: bool = False) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.is_video_5d = is_video_5d
        self.enc = nn.Conv2d(3, latent_channels, kernel_size=1)
        self.dec = nn.Conv2d(latent_channels, 3, kernel_size=1)

    def encode(self, pixel_samples: torch.Tensor) -> torch.Tensor:
        x = pixel_samples
        if x.ndim == 4 and x.shape[-1] in (3, 4) and x.shape[1] not in (3, 4):
            x = x[..., :3].permute(0, 3, 1, 2).contiguous()
        elif x.ndim == 4 and x.shape[1] == 4:
            x = x[:, :3]
        h = max(4, x.shape[-2] // 8)
        w = max(4, x.shape[-1] // 8)
        resized = F.interpolate(x.to(dtype=torch.float32), size=(h, w), mode="bilinear", align_corners=False)
        lat = self.enc(resized)
        if self.is_video_5d:
            return lat.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        return lat


class _ZooCLIP(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_stage_model = nn.Linear(embed_dim, embed_dim)
        self.patcher = comfy.model_patcher.ModelPatcher(
            self.cond_stage_model,
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )

    def clone(self, *args: Any, **kwargs: Any) -> "_ZooCLIP":
        c = _ZooCLIP(embed_dim=self.embed_dim)
        c.cond_stage_model = self.cond_stage_model
        c.patcher = self.patcher.clone() if self.patcher is not None else None
        return c

    def add_patches(self, patches: Any, strength_patch: float = 1.0, strength_model: float = 1.0) -> List[str]:
        if self.patcher is not None:
            return self.patcher.add_patches(patches, strength_patch, strength_model)
        return []

    def tokenize(self, text: str) -> Dict[str, Any]:
        gen = torch.Generator(device="cpu").manual_seed(abs(hash(text)) % (2**31))
        tokens = torch.randn(1, 8, self.embed_dim, generator=gen)
        return {"tokens": tokens, "text": text}

    def encode_from_tokens(self, tokenized: Dict[str, Any], return_pooled: bool = True) -> Any:
        tokens = tokenized["tokens"]
        cond = self.cond_stage_model(tokens)
        pooled = cond.mean(dim=1)
        if return_pooled:
            return cond, pooled
        return cond

    def encode_from_tokens_scheduled(self, tokenized: Dict[str, Any]) -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        cond, pooled = self.encode_from_tokens(tokenized, return_pooled=True)
        return [(cond, {"pooled_output": pooled})]


def build_native_comfy_patcher(
    arch_id: str,
    diffusion_model: nn.Module,
    unet_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    expected_objective: Optional[str] = None,
) -> comfy.model_patcher.ModelPatcher:
    """
    Wraps a real `comfy.ldm.*` `diffusion_model` in its genuine `comfy.supported_models.*`,
    `comfy.model_base.*`, `comfy.model_sampling.*`, `comfy.latent_formats.*`, and
    `comfy.model_patcher.ModelPatcher`.
    """
    target_device = device or torch.device("cpu")
    with torch.no_grad():
        for name, param in diffusion_model.named_parameters():
            if getattr(param, "is_meta", False):
                continue
            if param.is_floating_point():
                if "bias" in name:
                    torch.nn.init.zeros_(param)
                else:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
        for buffer in diffusion_model.buffers():
            if getattr(buffer, "is_meta", False):
                continue
            if buffer.is_floating_point():
                torch.nn.init.zeros_(buffer)

    diffusion_model.to(device=target_device, dtype=dtype)
    _attach_universal_forward_bridge(diffusion_model)

    supported_cls = find_comfy_supported_model_class(arch_id)
    merged_unet_config = dict(getattr(supported_cls, "unet_config", {}))
    if unet_config:
        merged_unet_config.update(unet_config)
    merged_unet_config["disable_unet_model_creation"] = True
    if "hidden_size" not in merged_unet_config:
        merged_unet_config["hidden_size"] = getattr(
            diffusion_model, "hidden_size", getattr(diffusion_model, "dim", 2604)
        )

    model_config = supported_cls(merged_unet_config)

    # Determine accurate model_type based on arch_id and expected_objective
    obj = expected_objective
    if not obj:
        if any(term in arch_id.lower() for term in ("sd15", "sdxl", "kolors", "hunyuan_dit")):
            obj = "ddpm_eps"
        else:
            obj = "rectified_flow"

    if obj in ("rectified_flow", "flow_matching"):
        model_type = comfy.model_base.ModelType.FLOW
    elif obj == "v_prediction":
        model_type = comfy.model_base.ModelType.V_PREDICTION
    elif obj == "edm":
        model_type = comfy.model_base.ModelType.EDM
    elif obj == "x0":
        model_type = comfy.model_base.ModelType.X0
    else:
        model_type = comfy.model_base.ModelType.EPS

    base_model_cls = getattr(comfy.model_base, supported_cls.__name__, comfy.model_base.BaseModel)
    try:
        base_model = base_model_cls(model_config, model_type=model_type, device=target_device)
    except Exception:
        base_model = comfy.model_base.BaseModel(model_config, model_type=model_type, device=target_device)

    base_model.diffusion_model = diffusion_model
    base_model.manual_cast_dtype = dtype

    def _safe_apply_model(x: torch.Tensor, t: torch.Tensor, c_concat: Any = None, c_crossattn: Any = None, control: Any = None, transformer_options: Any = None, **kwargs: Any) -> torch.Tensor:
        return base_model.diffusion_model(x, t, context=c_crossattn, transformer_options=transformer_options, **kwargs)

    base_model.apply_model = _safe_apply_model  # type: ignore[method-assign]

    return comfy.model_patcher.ModelPatcher(
        base_model,
        load_device=target_device,
        offload_device=target_device,
    )


# ==============================================================================
# 4. Real 1-to-1 ComfyModelZoo (21 Core Architectures from comfy.ldm.*)
# ==============================================================================

class ComfyModelZoo:
    ARCH_IDS: Tuple[str, ...] = (
        "flux_dev",
        "flux3_klein",
        "krea2",
        "moss_image",
        "qwen_image_1_0",
        "qwen_image_2_1",
        "sdxl_base",
        "sd15",
        "sd3_5",
        "wan_2_1_t2v",
        "wan_2_1_i2v",
        "wan_2_2_ti2v",
        "hunyuan_video",
        "ltx_video",
        "lumina2_z_image",
        "auraflow_sana",
        "fibo2",
        "hidream_i1",
        "chroma",
        "omnigen2",
        "cosmos_predict2",
    )

    @staticmethod
    def _instantiate_real_ldm(arch_id: str, dtype: torch.dtype = torch.float32) -> Tuple[nn.Module, int, str, bool]:
        ops = comfy.ops.manual_cast

        if arch_id in ("flux_dev", "flux1"):
            from comfy.ldm.flux.model import Flux
            dm = Flux(
                in_channels=16, out_channels=16, vec_in_dim=64, context_in_dim=128,
                hidden_size=64, mlp_ratio=2.0, num_heads=4, depth=2, depth_single_blocks=2,
                axes_dim=[4, 6, 6], theta=10000, patch_size=2, qkv_bias=True, guidance_embed=True,
                txt_ids_dims=[1, 2],
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", False

        if arch_id in ("flux3_klein", "flux2_klein"):
            from comfy.ldm.flux3.model import Flux3
            dm = Flux3(
                hidden_size=64, num_heads=4, depth=2, depth_single_blocks=2,
                fused_qkv=False, flow_map=True, dtype=dtype, device="cpu", operations=ops,
            )
            dm.txt_in = ops.Linear(128, 64, bias=False, dtype=dtype, device="cpu")
            return dm, 32, "rectified_flow", False

        if arch_id == "krea2":
            from comfy.ldm.krea2.model import SingleStreamDiT
            dm = SingleStreamDiT(
                features=64, tdim=32, txtdim=128, heads=4, kvheads=2, multiplier=2,
                layers=2, patch=2, channels=16, txtlayers=2, txtheads=4, txtkvheads=4,
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", False

        if arch_id == "moss_image":
            from comfy.ldm.moss_image.model import MOSSImage
            dm = MOSSImage(
                dim=64, ffn_dim=128, num_heads=4, num_layers=2, text_dim=128,
                latent_channels=128, fused_qkv=False, dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 128, "rectified_flow", False

        if arch_id in ("qwen_image_1_0", "qwen_image"):
            from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
            dm = QwenImageTransformer2DModel(
                patch_size=2, in_channels=64, out_channels=16, num_layers=2,
                attention_head_dim=16, num_attention_heads=4, joint_attention_dim=128,
                axes_dims_rope=(4, 6, 6), dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", True

        if arch_id in ("qwen_image_2_1", "qwen_image21"):
            from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
            dm = QwenImage21Transformer2DModel(
                in_channels=64, out_channels=64, num_layers=2,
                attention_head_dim=16, num_attention_heads=4, context_in_dim=128,
                axes_dims_rope=(4, 6, 6), dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", True

        if arch_id in ("sdxl_base", "sdxl"):
            from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
            dm = UNetModel(
                image_size=16, in_channels=4, model_channels=32, out_channels=4,
                num_res_blocks=1, channel_mult=[1, 2],
                num_heads=4, use_spatial_transformer=True,
                transformer_depth=[1, 1],
                transformer_depth_middle=1,
                transformer_depth_output=[1, 1, 1, 1],
                context_dim=128, adm_in_channels=128, num_classes="sequential",
                use_linear_in_transformer=True, dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 4, "ddpm_eps", False

        if arch_id == "sd15":
            from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
            dm = UNetModel(
                image_size=16, in_channels=4, model_channels=32, out_channels=4,
                num_res_blocks=1, channel_mult=[1, 2],
                num_heads=4, use_spatial_transformer=True,
                transformer_depth=[1, 1],
                transformer_depth_middle=1,
                transformer_depth_output=[1, 1, 1, 1],
                context_dim=64, dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 4, "ddpm_eps", False

        if arch_id in ("sd3_5", "sd3"):
            from comfy.ldm.modules.diffusionmodules.mmdit import OpenAISignatureMMDITWrapper
            dm = OpenAISignatureMMDITWrapper(
                input_size=16, patch_size=2, in_channels=16, depth=2, mlp_ratio=2.0,
                adm_in_channels=64, context_size=128, pos_embed_max_size=16,
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", False

        if arch_id in ("wan_2_1_t2v", "wan21_t2v"):
            from comfy.ldm.wan.model import WanModel
            dm = WanModel(
                model_type="t2v", patch_size=(1, 2, 2), in_dim=16, dim=64, ffn_dim=128,
                freq_dim=32, text_dim=128, out_dim=16, num_heads=4, num_layers=2,
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", True

        if arch_id in ("wan_2_1_i2v", "wan21_i2v"):
            from comfy.ldm.wan.model import WanModel
            dm = WanModel(
                model_type="i2v", patch_size=(1, 2, 2), in_dim=36, dim=64, ffn_dim=128,
                freq_dim=32, text_dim=128, out_dim=16, num_heads=4, num_layers=2,
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", True

        if arch_id in ("wan_2_2_ti2v", "wan22_t2v"):
            from comfy.ldm.wan.model import WanModel
            dm = WanModel(
                model_type="t2v", patch_size=(1, 2, 2), in_dim=48, dim=64, ffn_dim=128,
                freq_dim=32, text_dim=128, out_dim=48, num_heads=4, num_layers=2,
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 48, "rectified_flow", True

        if arch_id == "hunyuan_video":
            from comfy.ldm.hunyuan_video.model import HunyuanVideo
            dm = HunyuanVideo(
                in_channels=16, out_channels=16, vec_in_dim=64, context_in_dim=128,
                hidden_size=64, mlp_ratio=2.0, num_heads=4, depth=2, depth_single_blocks=2,
                axes_dim=[4, 6, 6], theta=256, patch_size=[1, 2, 2], qkv_bias=True,
                guidance_embed=True, byt5=False, meanflow=False, use_cond_type_embedding=False,
                vision_in_dim=None, meanflow_sum=False,
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", True

        if arch_id in ("ltx_video", "ltxv"):
            from comfy.ldm.lightricks.model import LTXVModel
            dm = LTXVModel(
                in_channels=128, cross_attention_dim=64, attention_head_dim=16,
                num_attention_heads=4, caption_channels=128, num_layers=2,
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 128, "rectified_flow", True

        if arch_id in ("lumina2_z_image", "lumina2"):
            from comfy.ldm.lumina.model import NextDiT
            dm = NextDiT(
                patch_size=2, in_channels=16, dim=64, n_layers=2, n_refiner_layers=1,
                n_heads=4, n_kv_heads=2, multiple_of=16, cap_feat_dim=128,
                axes_dims=(4, 6, 6), axes_lens=(4, 32, 32),
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", False

        if arch_id == "auraflow_sana":
            from comfy.ldm.aura.mmdit import MMDiT as AuraMMDiT
            dm = AuraMMDiT(
                in_channels=4, out_channels=4, patch_size=2, dim=64,
                n_layers=4, n_double_layers=2, n_heads=4, global_conddim=64,
                cond_seq_dim=128, max_seq=64, dtype=dtype, device="cpu", operations=ops,
            )
            for blk in dm.single_layers:
                if hasattr(blk, "modCX") and not hasattr(blk, "modX"):
                    blk.modX = blk.modCX
            return dm, 4, "rectified_flow", False

        if arch_id == "fibo2":
            from comfy.ldm.fibo2.model import Fibo2
            dm = Fibo2(
                in_channels=128, dim=64, n_heads=4, n_layers=2, n_refiner_layers=1,
                injection_layers=(0, 1), cap_feat_dim=128, perceiver_layers=1,
                max_gist_tokens=8, ffn_dim=128, dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 128, "rectified_flow", False

        if arch_id in ("hidream_i1", "hidream"):
            from comfy.ldm.hidream.model import HiDreamImageTransformer2DModel
            dm = HiDreamImageTransformer2DModel(
                patch_size=2, in_channels=16, out_channels=16, num_layers=2,
                num_single_layers=2, attention_head_dim=16, num_attention_heads=4,
                caption_channels=[128, 128], text_emb_dim=64, num_routed_experts=2,
                num_activated_experts=1, axes_dims_rope=(8, 8), max_resolution=(32, 32),
                llama_layers=[0, 1], dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", False

        if arch_id == "chroma":
            from comfy.ldm.chroma.model import Chroma
            dm = Chroma(
                in_channels=16, out_channels=16, context_in_dim=128, hidden_size=64,
                mlp_ratio=2.0, num_heads=4, depth=2, depth_single_blocks=2,
                axes_dim=[4, 6, 6], theta=10000, patch_size=2, qkv_bias=True,
                in_dim=16, out_dim=64, hidden_dim=64, n_layers=2,
                txt_ids_dims=[1, 2], vec_in_dim=64,
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", False

        if arch_id == "omnigen2":
            from comfy.ldm.omnigen.omnigen2 import OmniGen2Transformer2DModel
            dm = OmniGen2Transformer2DModel(
                patch_size=2, in_channels=16, out_channels=16, hidden_size=64,
                num_layers=2, num_refiner_layers=1, num_attention_heads=4,
                num_kv_heads=2, multiple_of=16, axes_dim_rope=(4, 6, 6),
                axes_lens=(32, 32, 32), text_feat_dim=128,
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", False

        if arch_id == "cosmos_predict2":
            from comfy.ldm.cosmos.predict2 import MiniTrainDIT
            dm = MiniTrainDIT(
                max_img_h=32, max_img_w=32, max_frames=8, in_channels=16,
                out_channels=16, patch_spatial=2, patch_temporal=1,
                concat_padding_mask=False, model_channels=64, num_blocks=2,
                num_heads=4, mlp_ratio=2.0, crossattn_emb_channels=128,
                pos_emb_cls="rope3d",
                dtype=dtype, device="cpu", operations=ops,
            )
            return dm, 16, "rectified_flow", True

        raise ValueError(f"Unknown ComfyModelZoo architecture: {arch_id}")

    @classmethod
    def build(cls, arch_id: str, dtype: torch.dtype = torch.bfloat16) -> ComfyZooEntry:
        dm, expected_ch, expected_obj, is_5d = cls._instantiate_real_ldm(arch_id, dtype=dtype)
        patcher = build_native_comfy_patcher(
            arch_id,
            dm,
            unet_config={"in_channels": expected_ch},
            dtype=dtype,
            expected_objective=expected_obj,
        )
        # Ensure latent_format reflects the model's true latent channel geometry and 5D state
        lf = getattr(patcher.model, "latent_format", None)
        if lf is not None:
            expected_ch = int(getattr(lf, "latent_channels", expected_ch))
            is_5d = int(getattr(lf, "latent_dimensions", 3 if is_5d else 2)) >= 3

        vae = _ZooVAE(latent_channels=expected_ch, is_video_5d=is_5d)
        clip = _ZooCLIP(embed_dim=128)
        return ComfyZooEntry(
            arch_id=arch_id,
            patcher=patcher,
            vae=vae,
            clip=clip,
            expected_objective=expected_obj,
            expected_latent_channels=expected_ch,
            is_video_5d=is_5d,
        )


# ==============================================================================
# 5. Extended ComfyUI LDM Model Zoo (29 Additional Real comfy.ldm.* Backbones)
# ==============================================================================

class ExtendedComfyLdmZoo:
    EXTENDED_ARCH_IDS: Tuple[str, ...] = (
        "ideogram4",
        "joyimage",
        "mage_flow",
        "boogu",
        "anima",
        "pixeldit",
        "ernie",
        "lens",
        "chroma_radiance",
        "hidream_o1",
        "kandinsky5",
        "cogvideo",
        "genmo_mochi",
        "ace_step",
        "mmaudio",
        "yue2",
        "hunyuan3d",
        "pixart",
        "hunyuan_dit",
        "stable_cascade",
        "sensenova",
        "minimax",
        "minimax_music",
        "seedvr",
        "trellis2",
        "triposplat",
        "hunyuan3dv2_1",
        "ltxav",
        "ace_step15",
    )

    @classmethod
    def build_backbone(cls, arch_id: str, dtype: torch.dtype = torch.float32) -> nn.Module:
        ops = comfy.ops.manual_cast

        if arch_id == "ideogram4":
            from comfy.ldm.ideogram4.model import Ideogram4Transformer2DModel
            return Ideogram4Transformer2DModel(
                in_channels=16, num_layers=2, num_attention_heads=4,
                attention_head_dim=16, intermediate_size=128, adaln_dim=64,
                llm_features_dim=64,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "joyimage":
            from comfy.ldm.joyimage.model import JoyImageTransformer3DModel
            return JoyImageTransformer3DModel(
                patch_size=[1, 2, 2], in_channels=16, hidden_size=64,
                num_attention_heads=4, text_dim=64, mlp_width_ratio=1.0,
                num_layers=2, rope_dim_list=[4, 6, 6], theta=256,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "mage_flow":
            from comfy.ldm.mage_flow.model import MageFlowTransformer2DModel
            return MageFlowTransformer2DModel(
                in_channels=16, out_channels=16, num_layers=2,
                attention_head_dim=16, num_attention_heads=4, joint_attention_dim=64,
                axes_dims_rope=(4, 6, 6),
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "boogu":
            from comfy.ldm.boogu.model import BooguTransformer2DModel
            return BooguTransformer2DModel(
                patch_size=2, in_channels=16, out_channels=16, hidden_size=64,
                num_layers=2, num_double_stream_layers=1, num_refiner_layers=1,
                num_attention_heads=4, num_kv_heads=2, multiple_of=16,
                axes_dim_rope=(4, 6, 6), axes_lens=(16, 16, 16), instruction_feat_dim=64,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "anima":
            from comfy.ldm.anima.model import Anima
            return Anima(
                max_img_h=32, max_img_w=32, max_frames=4,
                in_channels=16, out_channels=16, patch_spatial=2,
                patch_temporal=1, model_channels=64, num_blocks=2,
                num_heads=4, mlp_ratio=2.0, crossattn_emb_channels=64,
                pos_emb_cls="rope3d",
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "pixeldit":
            from comfy.ldm.pixeldit.model import PixDiT_T2I
            return PixDiT_T2I(
                in_channels=3, num_groups=4, hidden_size=64, pixel_hidden_size=16,
                pixel_attn_hidden_size=64, pixel_num_groups=4, patch_depth=2,
                pixel_depth=2, patch_size=2, txt_embed_dim=64, txt_max_length=64,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id in ("ernie", "ernie_image"):
            from comfy.ldm.ernie.model import ErnieImageModel
            return ErnieImageModel(
                hidden_size=64, num_attention_heads=4, num_layers=2,
                ffn_hidden_size=128, in_channels=16, out_channels=16,
                patch_size=2, text_in_dim=64, rope_theta=10000,
                rope_axes_dim=(4, 6, 6), dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "lens":
            from comfy.ldm.lens.model import LensTransformer2DModel
            return LensTransformer2DModel(
                patch_size=2, in_channels=16, out_channels=16,
                num_layers=2, attention_head_dim=16, num_attention_heads=4,
                enc_hidden_dim=64, axes_dims_rope=(4, 6, 6),
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "chroma_radiance":
            from comfy.ldm.chroma_radiance.model import ChromaRadiance
            return ChromaRadiance(
                in_channels=3, out_channels=3, context_in_dim=64,
                hidden_size=64, mlp_ratio=2.0, num_heads=4, depth=2,
                depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000,
                patch_size=2, qkv_bias=True, in_dim=16, out_dim=64,
                hidden_dim=64, n_layers=2, txt_ids_dims=[1, 2], vec_in_dim=64,
                nerf_hidden_size=32, nerf_mlp_ratio=2, nerf_depth=2, nerf_max_freqs=8,
                nerf_tile_size=0, nerf_final_head_type="linear",
                nerf_embedder_dtype=None, use_x0=False,
                use_sequential_txt_ids=False,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "hidream_o1":
            from comfy.ldm.hidream_o1.model import HiDreamO1Transformer
            return HiDreamO1Transformer(
                text_config_overrides={
                    "hidden_size": 64, "intermediate_size": 128,
                    "num_hidden_layers": 2, "num_attention_heads": 4,
                    "num_key_value_heads": 2, "head_dim": 16,
                    "vocab_size": 1000,
                },
                vision_config_overrides={
                    "hidden_size": 64, "intermediate_size": 128,
                    "num_hidden_layers": 2, "num_attention_heads": 4,
                    "out_hidden_size": 64,
                },
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "kandinsky5":
            from comfy.ldm.kandinsky5.model import Kandinsky5
            return Kandinsky5(
                in_visual_dim=16, in_text_dim=64, in_text_dim2=64,
                time_dim=64, out_visual_dim=16, patch_size=(1, 2, 2),
                model_dim=64, ff_dim=128, num_text_blocks=2,
                num_visual_blocks=2, axes_dims=(4, 6, 6),
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "cogvideo":
            from comfy.ldm.cogvideo.model import CogVideoXTransformer3DModel
            return CogVideoXTransformer3DModel(
                num_attention_heads=4, attention_head_dim=24,
                in_channels=16, out_channels=16, text_embed_dim=64,
                time_embed_dim=32, num_layers=2,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "genmo_mochi":
            from comfy.ldm.genmo.joint_model.asymm_models_joint import AsymmDiTJoint
            return AsymmDiTJoint(
                depth=2, patch_size=2, num_heads=4, hidden_size_x=64,
                hidden_size_y=64, mlp_ratio_x=192.0, mlp_ratio_y=2.0,
                in_channels=12, t5_feat_dim=64, qk_norm=True,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "ace_step":
            from comfy.ldm.ace.model import ACEStepTransformer2DModel
            return ACEStepTransformer2DModel(
                in_channels=8, out_channels=8, inner_dim=64, num_layers=2,
                attention_head_dim=16, num_attention_heads=4, mlp_ratio=2.0,
                speaker_embedding_dim=64, text_embedding_dim=64,
                lyric_encoder_vocab_size=64, lyric_hidden_size=64,
                ssl_names=[], ssl_latent_dims=[], patch_size=[2, 1],
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "mmaudio":
            from comfy.ldm.audio.dit import AudioDiffusionTransformer
            return AudioDiffusionTransformer(
                io_channels=16, embed_dim=64, cond_token_dim=64,
                depth=2, num_heads=4,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "yue2":
            from comfy.ldm.yue2.model import YuE2
            return YuE2(
                config={
                    "hidden_size": 64, "intermediate_size": 128,
                    "num_hidden_layers": 2, "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 256,
                },
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "hunyuan3d":
            from comfy.ldm.hunyuan3d.model import Hunyuan3Dv2
            return Hunyuan3Dv2(
                in_channels=16, context_in_dim=64, hidden_size=64,
                mlp_ratio=2.0, num_heads=4, depth=2, depth_single_blocks=2,
                qkv_bias=True, guidance_embed=True,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "pixart":
            from comfy.ldm.pixart.pixartms import PixArtMS
            return PixArtMS(
                input_size=16, patch_size=2, in_channels=4, hidden_size=96,
                depth=2, num_heads=4, mlp_ratio=2.0, caption_channels=64,
                micro_condition=False, dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "hunyuan_dit":
            from comfy.ldm.hydit.models import HunYuanDiT
            return HunYuanDiT(
                input_size=(16, 16), patch_size=2, in_channels=4, hidden_size=64,
                depth=4, num_heads=4, mlp_ratio=2.0, text_states_dim=64,
                text_states_dim_t5=64, text_len=16, text_len_t5=16,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "stable_cascade":
            from comfy.ldm.cascade.stage_c import StageC
            return StageC(
                c_in=16, c_out=16, c_r=16, patch_size=1, c_cond=64,
                c_hidden=[64, 64], nhead=[4, 4], blocks=[[1, 1], [1, 1]],
                block_repeat=[[1, 1], [1, 1]], level_config=["CTA", "CTA"],
                c_clip_text=64, c_clip_text_pooled=64, c_clip_img=64,
                c_clip_seq=2, switch_level=[False],
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "sensenova":
            import comfy.ldm.sensenova.model as _sn_mod
            orig_vals = {
                k: getattr(_sn_mod, k)
                for k in (
                    "HIDDEN_SIZE", "INTERMEDIATE_SIZE", "NUM_LAYERS",
                    "NUM_HEADS", "NUM_KV_HEADS", "HEAD_DIM", "VOCAB_SIZE",
                )
            }
            orig_lb_init = _sn_mod.LanguageBackbone.__init__
            orig_ve_init = _sn_mod.VisionEmbeddings.__init__
            orig_cd_init = _sn_mod.ConvDecoder.__init__

            def _mini_lb_init(self_lb: Any, device: Any = None, dtype: Any = None, operations: Any = None) -> None:
                nn.Module.__init__(self_lb)
                self_lb.embed_tokens = operations.Embedding(
                    _sn_mod.VOCAB_SIZE, _sn_mod.HIDDEN_SIZE, padding_idx=0, device=device, dtype=dtype
                )
                self_lb.layers = nn.ModuleList(
                    _sn_mod.DecoderLayer(device=device, dtype=dtype, operations=operations)
                    for _ in range(_sn_mod.NUM_LAYERS)
                )
                self_lb.norm = operations.RMSNorm(_sn_mod.HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype)
                self_lb.norm_mot_gen = operations.RMSNorm(_sn_mod.HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype)

            def _mini_ve_init(self_ve: Any, device: Any = None, dtype: Any = None, operations: Any = None) -> None:
                nn.Module.__init__(self_ve)
                self_ve.patch_embedding = operations.Conv2d(
                    3, 64, kernel_size=16, stride=16, device=device, dtype=dtype
                )
                self_ve.dense_embedding = operations.Conv2d(
                    64, _sn_mod.HIDDEN_SIZE, kernel_size=2, stride=2, device=device, dtype=dtype
                )
                self_ve.gelu = nn.GELU()

            def _mini_cd_init(self_cd: Any, device: Any = None, dtype: Any = None, operations: Any = None) -> None:
                nn.Module.__init__(self_cd)
                self_cd.ps1 = nn.PixelShuffle(2)
                self_cd.conv1 = operations.Conv2d(16, 16, kernel_size=3, padding=1, device=device, dtype=dtype)
                self_cd.act1 = nn.GELU()
                self_cd.ps2 = nn.PixelShuffle(2)
                self_cd.conv2 = operations.Conv2d(4, 192, kernel_size=3, padding=1, device=device, dtype=dtype)
                self_cd.ps3 = nn.PixelShuffle(8)

            try:
                _sn_mod.HIDDEN_SIZE = 64
                _sn_mod.INTERMEDIATE_SIZE = 128
                _sn_mod.NUM_LAYERS = 2
                _sn_mod.NUM_HEADS = 4
                _sn_mod.NUM_KV_HEADS = 2
                _sn_mod.HEAD_DIM = 16
                _sn_mod.VOCAB_SIZE = 256
                _sn_mod.LanguageBackbone.__init__ = _mini_lb_init  # type: ignore[method-assign]
                _sn_mod.VisionEmbeddings.__init__ = _mini_ve_init  # type: ignore[method-assign]
                _sn_mod.ConvDecoder.__init__ = _mini_cd_init  # type: ignore[method-assign]
                return _sn_mod.SenseNovaU15(dtype=dtype, device="cpu", operations=ops)
            finally:
                for k, v in orig_vals.items():
                    setattr(_sn_mod, k, v)
                _sn_mod.LanguageBackbone.__init__ = orig_lb_init  # type: ignore[method-assign]
                _sn_mod.VisionEmbeddings.__init__ = orig_ve_init  # type: ignore[method-assign]
                _sn_mod.ConvDecoder.__init__ = orig_cd_init  # type: ignore[method-assign]

        if arch_id == "minimax":
            from comfy.ldm.minimax.model import MiniMaxH3Model
            return MiniMaxH3Model(
                hidden_size=64, num_layers=2, token_refiner_num_layers=1,
                num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
                latents_dim=16, audio_latents_dim=16, patch_size=(1, 2, 2),
                text_dim=64, timestep_input_dim=32, time_embed_hidden_size=64,
                time_embed_dim=32, dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "minimax_music":
            import comfy.ldm.minimax_music.dit as _mm_dit
            orig_ct_init = _mm_dit.ContinuousTransformer.__init__
            orig_dt_init = _mm_dit.DiffusionTransformer.__init__
            orig_mm_init = _mm_dit.MiniMaxMusic3DiT.__init__

            def _mini_ct_init(self_ct: Any, dtype: Any = None, device: Any = None, operations: Any = None) -> None:
                nn.Module.__init__(self_ct)
                self_ct.project_in = operations.Linear(72, 64, bias=False, dtype=dtype, device=device)
                self_ct.project_out = operations.Linear(64, 16, bias=False, dtype=dtype, device=device)
                self_ct.rotary_pos_emb = _mm_dit.RotaryEmbedding(8, dtype, device)
                self_ct.layers = nn.ModuleList(
                    [_mm_dit.TransformerBlock(64, 16, 128, dtype, device, operations) for _ in range(2)]
                )

            def _mini_dt_init(self_dt: Any, dtype: Any = None, device: Any = None, operations: Any = None) -> None:
                nn.Module.__init__(self_dt)
                self_dt.transformer = _mm_dit.ContinuousTransformer(dtype, device, operations)
                self_dt.timestep_features = _mm_dit.FourierFeatures(1, 32, dtype, device)
                self_dt.to_timestep_embed = nn.Sequential(
                    operations.Linear(32, 64, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Linear(64, 64, dtype=dtype, device=device),
                )
                self_dt.preprocess_conv = operations.Conv1d(72, 72, 1, bias=False, dtype=dtype, device=device)
                self_dt.postprocess_conv = operations.Conv1d(16, 16, 1, bias=False, dtype=dtype, device=device)

            def _mini_mm_init(self_mm: Any, dtype: Any = None, device: Any = None, operations: Any = None, **kwargs: Any) -> None:
                nn.Module.__init__(self_mm)
                self_mm.dtype = dtype
                self_mm.latent_conditioners = nn.Sequential(
                    operations.Conv1d(64, 40, kernel_size=3, padding=1, dtype=dtype, device=device)
                )
                self_mm.diffusion_transformer = _mm_dit.DiffusionTransformer(dtype, device, operations)
                self_mm.cond_layer_logits = nn.Parameter(torch.empty(8, dtype=dtype, device=device))
                self_mm.cond_layer_scale = nn.Parameter(torch.empty(1, dtype=dtype, device=device))

            try:
                _mm_dit.ContinuousTransformer.__init__ = _mini_ct_init  # type: ignore[method-assign]
                _mm_dit.DiffusionTransformer.__init__ = _mini_dt_init  # type: ignore[method-assign]
                _mm_dit.MiniMaxMusic3DiT.__init__ = _mini_mm_init  # type: ignore[method-assign]
                return _mm_dit.MiniMaxMusic3DiT(dtype=dtype, device="cpu", operations=ops)
            finally:
                _mm_dit.ContinuousTransformer.__init__ = orig_ct_init  # type: ignore[method-assign]
                _mm_dit.DiffusionTransformer.__init__ = orig_dt_init  # type: ignore[method-assign]
                _mm_dit.MiniMaxMusic3DiT.__init__ = orig_mm_init  # type: ignore[method-assign]

        if arch_id == "seedvr":
            from comfy.ldm.seedvr.model import NaDiT
            return NaDiT(
                norm_eps=1e-5, num_layers=2, mlp_type="swiglu",
                vid_in_channels=16, vid_out_channels=16, vid_dim=64,
                txt_in_dim=64, heads=4, head_dim=16, mm_layers=1,
                expand_ratio=2, rope_dim=16,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "trellis2":
            from comfy.ldm.trellis2.model import Trellis2
            return Trellis2(
                resolution=4, in_channels=8, out_channels=8,
                model_channels=64, cond_channels=64, num_blocks=2,
                num_heads=4, mlp_ratio=2.0,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "triposplat":
            from comfy.ldm.triposplat.model import LatentSeqMMFlowModel
            return LatentSeqMMFlowModel(
                q_token_length=16, in_channels=16, model_channels=64,
                cond_channels=64, out_channels=16, num_blocks=2,
                num_refiner_blocks=1, num_heads=4, num_head_channels=16,
                cond2_channels=64, mlp_ratio=2,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "hunyuan3dv2_1":
            from comfy.ldm.hunyuan3dv2_1.hunyuandit import HunYuanDiTPlain
            return HunYuanDiTPlain(
                in_channels=16, hidden_size=64, context_dim=64,
                depth=4, num_heads=4, num_moe_layers=2,
                guidance_cond_proj_dim=64, num_experts=2, moe_top_k=1,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "ltxav":
            from comfy.ldm.lightricks.av_model import LTXAVModel
            return LTXAVModel(
                in_channels=16, audio_in_channels=16, cross_attention_dim=64,
                audio_cross_attention_dim=32, attention_head_dim=16,
                audio_attention_head_dim=8, num_attention_heads=4,
                audio_num_attention_heads=4, caption_channels=64, num_layers=2,
                connector_attention_head_dim=16, connector_num_attention_heads=4,
                connector_num_layers=1,
                dtype=dtype, device="cpu", operations=ops,
            )

        if arch_id == "ace_step15":
            from comfy.ldm.ace.ace_step15 import AceStepConditionGenerationModel
            return AceStepConditionGenerationModel(
                in_channels=16, hidden_size=64, text_hidden_dim=64,
                timbre_hidden_dim=16, audio_acoustic_hidden_dim=16,
                num_dit_layers=2, num_lyric_layers=1, num_timbre_layers=1,
                num_tokenizer_layers=1, num_heads=8, num_kv_heads=4,
                head_dim=8, intermediate_size=128, encoder_hidden_size=128,
                encoder_intermediate_size=256, encoder_num_heads=16, fsq_dim=64,
                dtype=dtype, device="cpu", operations=ops,
            )

        raise ValueError(f"Unknown ExtendedComfyLdmZoo architecture: {arch_id}")

    @classmethod
    def build(cls, arch_id: str, dtype: torch.dtype = torch.float32) -> comfy.model_patcher.ModelPatcher:
        dm = cls.build_backbone(arch_id, dtype=dtype)
        return build_native_comfy_patcher(arch_id, dm, dtype=dtype)


# ==============================================================================
# 6. Native ComfyUI KSampler & Loader Helpers
# ==============================================================================

class EmulatedKSampler:
    """
    Runs a deterministic sampling pass on a real `comfy.model_patcher.ModelPatcher`,
    applying any active `ModelPatcher` weight patches (`patch_model(patch_weights=True)`)
    before invoking `model.apply_model` and unpatching cleanly afterwards.
    """
    @staticmethod
    def sample(
        patcher: comfy.model_patcher.ModelPatcher,
        clip: Any = None,
        resolution: int = 64,
        steps: int = 2,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        target_dev = device or torch.device("cpu")
        lf = getattr(patcher.model, "latent_format", None)
        ch = int(getattr(lf, "latent_channels", 16))
        is_5d = int(getattr(lf, "latent_dimensions", 2)) >= 3
        h = max(4, resolution // 8)
        w = max(4, resolution // 8)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        shape = (1, ch, 2, h, w) if is_5d else (1, ch, h, w)
        dtype = getattr(patcher.model, "manual_cast_dtype", None) or torch.float32
        x = torch.randn(shape, generator=gen, dtype=torch.float32).to(device=target_dev, dtype=dtype)

        cond_tensor = None
        if clip is not None and hasattr(clip, "tokenize") and hasattr(clip, "encode_from_tokens"):
            tok = clip.tokenize("ksampler verification prompt")
            res = clip.encode_from_tokens(tok, return_pooled=True)
            cond_tensor = res[0] if isinstance(res, tuple) else res
            if isinstance(cond_tensor, torch.Tensor):
                cond_tensor = cond_tensor.to(device=target_dev, dtype=dtype)

        cloned = patcher.clone()
        try:
            try:
                cloned.patch_model(device_to=target_dev)
            except TypeError:
                cloned.patch_model()
            for step_idx in range(max(1, steps)):
                sigma_val = 1.0 - (step_idx / max(1, steps))
                t = torch.full((1,), sigma_val, dtype=dtype, device=target_dev)
                pred = cloned.model.apply_model(x, t, c_crossattn=cond_tensor)
                if isinstance(pred, torch.Tensor) and pred.shape == x.shape:
                    x = x - (pred - x) * (1.0 / max(1, steps))
        finally:
            try:
                cloned.unpatch_model(device_to=target_dev, unpatch_weights=True)
            except Exception:
                pass

        return x
