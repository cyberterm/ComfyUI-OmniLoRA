"""
ComfyUI Custom Nodes for OmniLoRA (`omni_lora/nodes.py`).

Provides 6 modular ComfyUI nodes under category `OmniLoRA/Training`:
1. `OmniLoRADatasetLoader` ("OmniLoRA Dataset & Prompt Loader"):
   - Loads training images + `.txt` captions from a folder AND/OR direct ComfyUI `IMAGE` / `MASK` inputs.
   - Applies `trigger_word`, `default_caption`, aspect-ratio bucketing / center-crop, `random_flip`, `caption_dropout`, `num_repeats`, and alpha/mask loss weighting.
2. `OmniLoRAPrecomputeConditioning` ("OmniLoRA VAE & CLIP Pre-Cache"):
   - Encodes all dataset images and captions using ComfyUI's loaded `VAE` and `CLIP`.
   - Captures full ComfyUI conditioning dictionaries (`return_dict=True`) and pads variable-length token sequences across batches.
   - Clones tensors outside `torch.inference_mode()` so autograd works cleanly.
   - Offloads `VAE` and `CLIP` from VRAM (`comfy.model_management.unload_all_models()`) so 100% of free VRAM is available for training.
3. `OmniLoRALoader` ("OmniLoRA Load LoRA Network"):
   - Loads an existing `.safetensors` or `.pt` LoRA network from `ComfyUI/models/loras/` (or a custom path) as a starting network (`OMNILORA_NETWORK`) with automatic `rank`, `alpha`, and `steps_trained` detection.
4. `OmniLoRATrainer` ("OmniLoRA Universal Trainer (All GPUs / XPU / CUDA / ROCm)"):
   - Queries ComfyUI's `MODEL` (`ModelPatcher` / `BaseModel`) directly for latent geometry, noise schedule, objective, and LoRA targets.
   - Optionally resumes from a connected `initial_lora` (`OMNILORA_NETWORK`), automatically adopting its `rank` and `alpha`.
   - Supports multiple FP64-free optimizers (`radamw`, `adamw`, `adafactor`, `sgd`), `grad_accum_steps=1` default, and optional periodic auto-checkpointing (`checkpoint_every_n_steps`).
   - Outputs a patched `MODEL`, a portable `OMNILORA_NETWORK` bundle, and a JSON `training_summary`.
5. `OmniLoRASave` ("OmniLoRA Save LoRA"):
   - Modularly saves any `OMNILORA_NETWORK` bundle to `ComfyUI/models/loras/` as `.safetensors` with complete ModelSpec/OmniLoRA metadata.
6. `OmniLoRALossTracker` ("OmniLoRA Loss Tracker"):
   - Mounts an interactive HTML5 canvas loss & telemetry dashboard on the ComfyUI canvas (`web/omni_lora_tracker.js`).
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backends import detect_and_configure, sanitize_module_no_fp64, trim_process_working_set
from .dataset import OmniLoRADataset, OmniLoRADatasetConfig
from .lora import LoRAConfig, LoRAInjector, release_streamed_weight_buffers
from .lora.layers import _is_conv_like, _is_linear_like, _strip_numeric_segments
from .models import (
    discover_universal_target_modules,
    install_comfy_runtime_hooks,
    resolve_model_adapter,
    unwrap_comfy_diffusion_model,
)
from .pipelines import LoRATrainingPipeline, TrainingConfig
from .tracker_store import TRACKER_STORE, register_omni_lora_routes


@dataclass
class CachedConditioningBundle:
    """CPU-pinned, autograd-ready pre-cached latents, optional control latents, and text embeddings."""

    batches: List[Dict[str, Any]]
    resolution: int
    batch_size: int
    trigger_word: str
    num_unique_images: int
    total_batches: int
    latent_scaled: bool = False
    random_flip: bool = False
    caption_dropout: float = 0.0
    uncond_prompt_embeds: Optional[torch.Tensor] = None
    uncond_pooled_output: Optional[torch.Tensor] = None
    uncond_extra_conds: Dict[str, Any] = field(default_factory=dict)
    has_control_latents: bool = False
    clip_used_control_images: bool = False
    num_controlled_images: int = 0


@dataclass
class OmniLoRANetworkBundle:
    """
    Portable in-memory LoRA network representation (`OMNILORA_NETWORK`) that moves
    seamlessly between `OmniLoRALoader`, `OmniLoRATrainer`, and `OmniLoRASave`.
    """

    state_dict: Dict[str, torch.Tensor]
    rank: int
    alpha: float
    model_architecture: str = "auto"
    steps_trained: int = 0
    trigger_word: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    source_name: str = "omni_lora"


def _try_comfy_unload_models() -> None:
    """
    Call ComfyUI's `comfy.model_management` to unload VAE/CLIP/prior models from VRAM,
    purge dead `LoadedModel` references (`cleanup_models_gc` / `cleanup_models`), run Python GC,
    and then release the GPU caching allocator (`soft_empty_cache(True)`).
    """
    had_loaded_models = False
    try:
        import comfy.model_management as mm  # type: ignore

        loaded_list = getattr(mm, "current_loaded_models", None)
        if isinstance(loaded_list, list) and len(loaded_list) > 0:
            had_loaded_models = True
            for i in range(len(loaded_list) - 1, -1, -1):
                loaded_m = loaded_list[i]
                try:
                    # ComfyUI's `LoadedModel.is_dead()` calls `self.real_model()`. Once `model_unload()`
                    # has been called on a `LoadedModel`, `self.real_model` is set to `None`.
                    # Ensure no unloaded entry with `real_model is None` ever remains in `current_loaded_models`.
                    if hasattr(loaded_m, "real_model") and getattr(loaded_m, "real_model") is None:
                        loaded_list.pop(i)
                        continue
                    if hasattr(loaded_m, "currently_used"):
                        loaded_m.currently_used = False
                except Exception as exc:
                    logging.debug(f"[OmniLoRA] Could not inspect loaded ComfyUI model entry: {exc}")

        mm.unload_all_models()

        if isinstance(loaded_list, list) and len(loaded_list) > 0:
            had_loaded_models = True
            while len(loaded_list) > 0:
                loaded_m = loaded_list.pop()
                try:
                    if (
                        hasattr(loaded_m, "model_unload")
                        and callable(loaded_m.model_unload)
                        and getattr(loaded_m, "real_model", True) is not None
                    ):
                        loaded_m.model_unload()
                except Exception as exc:
                    logging.debug(f"[OmniLoRA] Model unload entry cleanup warning: {exc}")

        if hasattr(mm, "cleanup_models_gc"):
            try:
                mm.cleanup_models_gc()
            except Exception as exc:
                logging.debug(f"[OmniLoRA] comfy.model_management.cleanup_models_gc failed: {exc}")
        if hasattr(mm, "cleanup_models"):
            try:
                mm.cleanup_models()
            except Exception as exc:
                logging.debug(f"[OmniLoRA] comfy.model_management.cleanup_models failed: {exc}")
    except ImportError:
        pass
    except Exception as exc:
        logging.debug(f"[OmniLoRA] Could not unload ComfyUI models: {exc}")
    gc.collect()
    try:
        import comfy.model_management as mm  # type: ignore

        try:
            mm.soft_empty_cache(True)
        except TypeError:
            mm.soft_empty_cache()
    except ImportError:
        pass
    except Exception as exc:
        logging.debug(f"[OmniLoRA] comfy.model_management.soft_empty_cache failed: {exc}")
    for dev_attr in ("xpu", "cuda", "mps", "npu", "hpu"):
        dev_mod = getattr(torch, dev_attr, None)
        if dev_mod is not None and hasattr(dev_mod, "empty_cache"):
            try:
                is_avail = (
                    dev_mod.is_available()
                    if hasattr(dev_mod, "is_available")
                    else (
                        bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
                        if dev_attr == "mps"
                        else False
                    )
                )
                if is_avail:
                    if hasattr(dev_mod, "synchronize"):
                        dev_mod.synchronize()
                    dev_mod.empty_cache()
            except Exception as exc:
                logging.debug(f"[OmniLoRA] Device {dev_attr} cache flush warning: {exc}")
    if had_loaded_models:
        trim_process_working_set()


def _check_comfy_interrupt() -> None:
    """Raise `comfy.model_management.InterruptProcessingException` immediately if the user pressed Cancel/Interrupt in ComfyUI."""
    try:
        import comfy.model_management as mm  # type: ignore

        if hasattr(mm, "throw_exception_if_processing_interrupted"):
            mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _resolve_comfy_loras_dir(fallback_dir: str = "./models/loras", create: bool = True) -> Path:
    """
    Resolve `ComfyUI/models/loras` via `folder_paths` when running inside ComfyUI,
    or via the parent ComfyUI root if located inside `ComfyUI/custom_nodes/<pack>`.
    Never creates a `models/` directory inside `ComfyUI/custom_nodes/`.
    """
    try:
        import folder_paths  # type: ignore

        paths = folder_paths.get_folder_paths("loras")
        if paths:
            out = Path(paths[0])
            if create:
                out.mkdir(parents=True, exist_ok=True)
            return out
    except Exception:
        pass

    pkg_dir = Path(__file__).resolve().parent
    if pkg_dir.parent.name.lower() == "custom_nodes":
        comfy_root = pkg_dir.parent.parent
        out = comfy_root / "models" / "loras"
        if create:
            out.mkdir(parents=True, exist_ok=True)
        return out

    out = Path(fallback_dir)
    try:
        resolved_parent = out.resolve().parent
        if resolved_parent.parent.name.lower() == "custom_nodes":
            out = resolved_parent.parent.parent / "models" / "loras"
    except Exception:
        pass
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_target_output_dir(custom_dir: str, default_dir: Path) -> Path:
    """
    Resolve an optional custom output/checkpoint directory:
    - Empty string -> `default_dir` (`ComfyUI/models/loras/`)
    - Absolute path or explicit relative path (`./...`, `../...`, `~/...`) -> resolved directly
    - Bare subfolder name (e.g. `my_run_ckpts`) -> resolved as a subfolder inside `default_dir`
      unless an existing folder with that name already exists relative to the current working directory.
    """
    clean = (custom_dir or "").strip().strip("\"'")
    if not clean:
        return default_dir
    expanded = Path(clean).expanduser()
    if expanded.is_absolute() or clean.startswith((".", "/", "\\", "~")) or expanded.exists():
        out = expanded.resolve()
    else:
        out = (default_dir / expanded).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _list_available_comfy_loras() -> List[str]:
    """Return available LoRA filenames from `ComfyUI/models/loras/` for dropdown selection without creating folders."""
    try:
        import folder_paths  # type: ignore

        files = list(folder_paths.get_filename_list("loras"))
        if files:
            return files
    except Exception:
        pass

    loras_dir = _resolve_comfy_loras_dir(create=False)
    found: List[str] = []
    if loras_dir.exists():
        for ext in ("*.safetensors", "*.pt", "*.ckpt", "*.bin"):
            for p in sorted(loras_dir.rglob(ext)):
                try:
                    found.append(str(p.relative_to(loras_dir)))
                except ValueError:
                    found.append(p.name)
    return found if found else ["(none)"]


def _save_lora_safetensors_or_pt(
    state_dict: Dict[str, torch.Tensor],
    out_path: Path,
    metadata: Dict[str, str],
) -> Path:
    """Save LoRA weights as `.safetensors` (preferred by ComfyUI) with `.pt` fallback."""
    clean_tensors: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        t = v.detach().cpu().contiguous()
        if t.dtype == torch.float64:
            t = t.to(dtype=torch.float32)
        clean_tensors[k] = t

    if out_path.suffix != ".safetensors":
        out_path = out_path.with_suffix(".safetensors")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file

        save_file(clean_tensors, str(out_path), metadata=metadata)
        return out_path
    except ImportError:
        pt_path = out_path.with_suffix(".pt")
        torch.save({"state_dict": clean_tensors, "metadata": metadata}, pt_path)
        return pt_path


def _load_lora_safetensors_or_pt(file_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Load a `.safetensors` or `.pt`/`.ckpt` LoRA file onto CPU along with its string metadata.
    Strictly sanitizes any `float64` tensors to `float32`.
    """
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"[OmniLoRA] LoRA file not found: {file_path}")

    state_dict: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, str] = {}

    if file_path.suffix.lower() == ".safetensors":
        try:
            from safetensors import safe_open  # type: ignore

            with safe_open(str(file_path), framework="pt", device="cpu") as f:
                raw_meta = f.metadata()
                if isinstance(raw_meta, dict):
                    metadata = {str(k): str(v) for k, v in raw_meta.items()}
                for key in f.keys():
                    t = f.get_tensor(key)
                    if isinstance(t, torch.Tensor):
                        if t.dtype == torch.float64:
                            t = t.to(dtype=torch.float32)
                        state_dict[key] = t.detach().cpu()
            return state_dict, metadata
        except ImportError:
            pass

    raw_obj: Any = None
    if file_path.suffix.lower() != ".safetensors":
        try:
            raw_obj = torch.load(str(file_path), map_location="cpu", weights_only=True)
        except Exception:
            raw_obj = None
    if raw_obj is None:
        try:
            import comfy.utils  # type: ignore

            raw_obj = comfy.utils.load_torch_file(str(file_path), safe_load=True)
        except Exception:
            raw_obj = torch.load(str(file_path), map_location="cpu", weights_only=True)

    if isinstance(raw_obj, dict):
        if "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
            raw_tensors = raw_obj["state_dict"]
            if isinstance(raw_obj.get("metadata"), dict):
                metadata = {str(k): str(v) for k, v in raw_obj["metadata"].items()}
        else:
            raw_tensors = raw_obj
        for k, v in raw_tensors.items():
            if isinstance(v, torch.Tensor):
                t = v.detach().cpu()
                if t.dtype == torch.float64:
                    t = t.to(dtype=torch.float32)
                state_dict[str(k)] = t

    return state_dict, metadata


def _coerce_lora_network_bundle(lora_obj: Any) -> OmniLoRANetworkBundle:
    """Coerce an `OmniLoRANetworkBundle` or raw state_dict into a validated `OmniLoRANetworkBundle`."""
    if isinstance(lora_obj, OmniLoRANetworkBundle):
        return lora_obj
    if isinstance(lora_obj, dict):
        raw_sd = lora_obj.get("state_dict", lora_obj) if isinstance(lora_obj.get("state_dict"), dict) else lora_obj
        raw_meta = lora_obj.get("metadata", {}) if isinstance(lora_obj.get("metadata"), dict) else {}
        info = LoRAInjector.inspect_lora_state_dict(raw_sd, metadata=raw_meta)
        return OmniLoRANetworkBundle(
            state_dict=raw_sd,
            rank=int(info["rank"]),
            alpha=float(info["alpha"]),
            model_architecture=str(info.get("model_architecture", "auto")),
            steps_trained=int(info.get("steps_trained", 0)),
            trigger_word=str(info.get("trigger_word", "")),
            metadata=raw_meta,
            source_name="dict_lora",
        )
    raise TypeError(f"[OmniLoRA] Expected OMNILORA_NETWORK bundle, got {type(lora_obj).__name__}")


def _pad_and_stack_sequence_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack a list of `[seq_len, dim]` tensors into `[B, max_seq_len, dim]`, padding
    shorter token sequences along `dim=0` if prompts in the batch have variable token counts.
    """
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0)
    max_seq = max(int(t.shape[0]) for t in tensors)
    if all(int(t.shape[0]) == max_seq for t in tensors):
        return torch.stack(tensors, dim=0)
    padded: List[torch.Tensor] = []
    for t in tensors:
        diff = max_seq - int(t.shape[0])
        if diff > 0:
            pad_shape = [0, 0] * (t.ndim - 1) + [0, diff]
            t = F.pad(t, pad_shape)
        padded.append(t)
    return torch.stack(padded, dim=0)


class OmniLoRADatasetLoader:
    """
    ComfyUI Node 1: Load and preprocess image + prompt pairs from a folder and/or direct ComfyUI IMAGE inputs.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "dataset_path": (
                    "STRING",
                    {
                        "default": "./datasets/train",
                        "multiline": False,
                        "tooltip": "Folder path containing training images (.png/.jpg/.jpeg/.webp/.avif/.bmp/.tiff/.heic) and matching .txt caption files.",
                    },
                ),
                "trigger_word": (
                    "STRING",
                    {
                        "default": "ohwx",
                        "multiline": False,
                        "tooltip": "Activation token automatically prepended to every training caption.",
                    },
                ),
                "default_caption": (
                    "STRING",
                    {
                        "default": "a high-resolution detailed photograph",
                        "multiline": True,
                        "tooltip": "Fallback caption used when an image does not have a matching .txt file.",
                    },
                ),
                "resolution": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 16, "step": 1},
                ),
                "num_repeats": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1},
                ),
                "center_crop": ("BOOLEAN", {"default": True}),
                "aspect_ratio_bucketing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically group portrait, square, and landscape images into aspect-preserving resolution buckets (multiples of 64) instead of forcing a 1:1 square crop.",
                    },
                ),
                "random_flip": ("BOOLEAN", {"default": True}),
                "caption_dropout": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Optional direct ComfyUI IMAGE batch connection (can be used instead of or in addition to dataset_path).",
                    },
                ),
                "custom_prompts": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional multiline prompts for connected `images` (one line per image in the batch).",
                    },
                ),
                "masks": (
                    "MASK",
                    {
                        "tooltip": "Optional ComfyUI MASK batch connection to weight training loss toward masked foreground regions.",
                    },
                ),
                "use_alpha_mask": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically weight training loss using RGBA PNG/WebP transparency channels, <stem>.mask.png sidecars, or connected ComfyUI MASK inputs.",
                    },
                ),
                "mask_unmasked_weight": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Minimum loss weight applied to unmasked background (mask=0) regions so the model preserves clean context without overfitting background (0.0 = strictly mask-only).",
                    },
                ),
                "control_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional folder path containing paired control/source images for image-editing models (matched by filename stem or mirrored subfolder path, any image extension). Leave empty for standard text-to-image training.",
                    },
                ),
                "control_resize_mode": (
                    ["match_target", "preserve_aspect"],
                    {
                        "default": "match_target",
                        "tooltip": "How control images are resized: 'match_target' matches each target image's exact bucket & crop framing; 'preserve_aspect' preserves each control image's own aspect ratio bucket.",
                    },
                ),
                "control_images": (
                    "IMAGE",
                    {
                        "tooltip": "Optional direct ComfyUI IMAGE batch of paired control/source images corresponding to connected `images`.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("OMNILORA_DATASET", "STRING")
    RETURN_NAMES = ("dataset", "dataset_info")
    FUNCTION = "load_dataset"
    CATEGORY = "OmniLoRA/Training"

    def load_dataset(
        self,
        dataset_path: str,
        trigger_word: str,
        default_caption: str,
        resolution: int,
        batch_size: int,
        num_repeats: int,
        center_crop: bool,
        random_flip: bool,
        caption_dropout: float,
        aspect_ratio_bucketing: bool = True,
        images: Optional[torch.Tensor] = None,
        custom_prompts: str = "",
        masks: Optional[torch.Tensor] = None,
        use_alpha_mask: bool = True,
        mask_unmasked_weight: float = 0.1,
        control_path: str = "",
        control_resize_mode: str = "match_target",
        control_images: Optional[torch.Tensor] = None,
    ) -> Tuple[OmniLoRADataset, str]:
        with torch.inference_mode(False):
            cfg = OmniLoRADatasetConfig(
                dataset_path=(dataset_path or "").strip().strip("\"'"),
                trigger_word=trigger_word,
                default_caption=default_caption,
                resolution=resolution,
                batch_size=batch_size,
                num_repeats=num_repeats,
                center_crop=center_crop,
                aspect_ratio_bucketing=aspect_ratio_bucketing,
                random_flip=random_flip,
                caption_dropout=caption_dropout,
                use_alpha_mask=bool(use_alpha_mask),
                mask_unmasked_weight=min(1.0, max(0.0, float(mask_unmasked_weight))),
                control_path=(control_path or "").strip().strip("\"'"),
                control_resize_mode=str(control_resize_mode or "match_target"),
            )
            dataset = OmniLoRADataset(
                config=cfg,
                comfy_images=images,
                custom_prompts=custom_prompts,
                comfy_masks=masks,
                comfy_control_images=control_images,
            )
            info_json = json.dumps(dataset.summary_dict(), indent=2)
            return (dataset, info_json)


class OmniLoRAPrecomputeConditioning:
    """
    ComfyUI Node 2: Encode all dataset images & prompts through ComfyUI's VAE & CLIP,
    clone tensors outside `torch.inference_mode()`, and offload VAE & CLIP from VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "dataset": ("OMNILORA_DATASET",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "offload_encoders_after_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unloads VAE and Text Encoder from VRAM after pre-caching to save VRAM.",
                    },
                ),
            },
            "optional": {
                "pass_control_to_clip": (
                    ["auto", "enabled", "disabled"],
                    {
                        "default": "auto",
                        "tooltip": "Pass paired control images into vision-language text encoders (VLM CLIPs that accept `images` in `clip.tokenize`). 'auto' detects support from the connected CLIP; 'disabled' skips passing images to CLIP.",
                    },
                ),
                "clip_vision_resolution": (
                    "INT",
                    {
                        "default": 384,
                        "min": 0,
                        "max": 2048,
                        "step": 32,
                        "tooltip": "Target square-area side length when resizing control images for vision-language text encoders (0 = keep control image resolution as-is). Preserves aspect ratio.",
                    },
                ),
                "clip_prompt_template": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional user-defined prompt wrapper template applied before CLIP encoding (supports `{prompt}` substitution, or `{}` as a ComfyUI `llama_template` override on LLM/VLM tokenizers). Leave empty to use captions and ComfyUI's native tokenizer template as-is.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("OMNILORA_CACHED_DATA",)
    RETURN_NAMES = ("cached_data",)
    FUNCTION = "precompute"
    CATEGORY = "OmniLoRA/Training"

    @staticmethod
    def _clip_supports_images_kwarg(clip: Any) -> bool:
        """
        Inspect a live ComfyUI `CLIP` and its underlying tokenizer(s) to determine whether
        `clip.tokenize(..., images=[...])` is natively supported (e.g. VLM text encoders
        such as Qwen-VL, Boogu, OmniGen2) without hardcoding any model or tokenizer names.
        """
        if clip is None:
            return False

        def _fn_accepts_param(fn: Any, param_name: str) -> bool:
            if not callable(fn):
                return False
            try:
                return param_name in inspect.signature(fn).parameters
            except (ValueError, TypeError):
                return False

        tokenizer = getattr(clip, "tokenizer", None)
        if tokenizer is None:
            return _fn_accepts_param(getattr(clip, "tokenize", None), "images")

        to_visit: List[Any] = [tokenizer]
        visited_ids: set[int] = set()
        while to_visit:
            cur = to_visit.pop()
            if cur is None or id(cur) in visited_ids:
                continue
            visited_ids.add(id(cur))
            if getattr(cur, "llama_template_images", None):
                return True
            if _fn_accepts_param(getattr(cur, "tokenize_with_weights", None), "images"):
                return True
            cur_dict = getattr(cur, "__dict__", None)
            if isinstance(cur_dict, dict):
                for sub_val in cur_dict.values():
                    if sub_val is not None and not isinstance(sub_val, (str, bytes, int, float, bool, list, tuple, dict, set, torch.Tensor)):
                        if hasattr(sub_val, "tokenize_with_weights") or getattr(sub_val, "llama_template_images", None):
                            to_visit.append(sub_val)
        return False

    @staticmethod
    def _clip_supports_llama_template_kwarg(clip: Any) -> bool:
        """Return True if the connected ComfyUI CLIP/tokenizer accepts `llama_template` in `tokenize`."""
        if clip is None:
            return False
        tok_fn = getattr(clip, "tokenize", None)
        if callable(tok_fn):
            try:
                if "llama_template" in inspect.signature(tok_fn).parameters:
                    return True
            except (ValueError, TypeError):
                pass
        tokenizer = getattr(clip, "tokenizer", None)
        if tokenizer is None:
            return False
        candidates: List[Any] = [tokenizer]
        cur_dict = getattr(tokenizer, "__dict__", None)
        if isinstance(cur_dict, dict):
            for v in cur_dict.values():
                if (
                    hasattr(v, "tokenize_with_weights")
                    or getattr(v, "llama_template", None)
                    or getattr(v, "llama_template_images", None)
                ):
                    candidates.append(v)
        for cand in candidates:
            if getattr(cand, "llama_template", None) or getattr(cand, "llama_template_images", None):
                return True
            fn = getattr(cand, "tokenize_with_weights", None)
            if callable(fn):
                try:
                    if "llama_template" in inspect.signature(fn).parameters:
                        return True
                except (ValueError, TypeError):
                    pass
        return False

    @staticmethod
    def _resize_control_for_clip_vision(ctrl_chw: torch.Tensor, vision_res: int) -> torch.Tensor:
        """
        Convert a normalized `[-1, 1]` `[3, H, W]` control tensor into a `[1, H', W', 3]` `[0, 1]`
        ComfyUI `IMAGE` tensor resized to approximately `vision_res * vision_res` area (preserving
        aspect ratio, rounded to multiples of 16) for vision-language text encoders.
        """
        with torch.inference_mode(False):
            img_01 = ((ctrl_chw.detach().to(device="cpu", dtype=torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0)
            if vision_res > 0:
                h, w = int(img_01.shape[1]), int(img_01.shape[2])
                target_area = float(vision_res * vision_res)
                scale = math.sqrt(target_area / float(max(1, h * w)))
                new_h = max(16, int(round((float(h) * scale) / 16.0)) * 16)
                new_w = max(16, int(round((float(w) * scale) / 16.0)) * 16)
                if new_h != h or new_w != w:
                    img_01 = F.interpolate(
                        img_01.unsqueeze(0),
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).clamp(0.0, 1.0)
            return img_01.permute(1, 2, 0).unsqueeze(0).contiguous()

    @staticmethod
    def _apply_prompt_template(
        text: str,
        template: str,
        supports_llama_template: bool,
    ) -> Tuple[str, Optional[str]]:
        """
        Apply an optional user-specified `clip_prompt_template`:
          - Empty template -> pass `text` verbatim (`llama_template=None`).
          - Contains `{prompt}` -> substitute `{prompt}` with `text`.
          - Contains `{}` and tokenizer supports `llama_template` -> pass as `llama_template` kwarg.
          - Otherwise -> prepend template before non-empty `text`.
        """
        tpl = (template or "")
        if not tpl.strip():
            return text, None
        if "{prompt}" in tpl:
            return tpl.replace("{prompt}", text), None
        if "{}" in tpl and supports_llama_template:
            return text, tpl
        if "{}" in tpl:
            return tpl.replace("{}", text), None
        if not text:
            return text, None
        return f"{tpl}{text}", None

    @staticmethod
    def _encode_single_prompt(
        clip: Any,
        text: str,
        control_images: Optional[List[torch.Tensor]] = None,
        llama_template: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Encode a single prompt with ComfyUI's `CLIP` object, preferring `return_dict=True`
        (and also supporting `encode_from_tokens_scheduled` / nested `[[cond, dict]]` outputs)
        so any model-specific conditioning keys (`attention_mask`, `clip_l`, `guidance`, etc.)
        are preserved alongside `cond` and `pooled_output`.
        Optionally forwards `images=control_images` and/or `llama_template=llama_template`
        to vision-language / LLM text encoders.
        """
        tok_kwargs: Dict[str, Any] = {}
        if control_images:
            tok_kwargs["images"] = control_images
        if llama_template is not None:
            tok_kwargs["llama_template"] = llama_template

        if tok_kwargs:
            try:
                tokens = clip.tokenize(text, **tok_kwargs)
            except TypeError:
                # Graceful fallback if the connected CLIP tokenizer does not accept `images` or `llama_template`
                fallback_kw = {k: v for k, v in tok_kwargs.items() if k != "images"}
                try:
                    tokens = clip.tokenize(text, **fallback_kw) if fallback_kw else clip.tokenize(text)
                except TypeError:
                    tokens = clip.tokenize(text)
        else:
            tokens = clip.tokenize(text)

        extra_dict: Dict[str, Any] = {}
        encoded: Any = None
        if hasattr(clip, "encode_from_tokens") and callable(clip.encode_from_tokens):
            try:
                encoded = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            except TypeError:
                try:
                    encoded = clip.encode_from_tokens(tokens, return_pooled=True)
                except TypeError:
                    encoded = clip.encode_from_tokens(tokens)
        elif hasattr(clip, "encode_from_tokens_scheduled") and callable(clip.encode_from_tokens_scheduled):
            encoded = clip.encode_from_tokens_scheduled(tokens)

        # Unwrap ComfyUI scheduled conditioning list format `[[cond_tensor, {"pooled_output": ..., ...}]]`
        while (
            isinstance(encoded, (tuple, list))
            and len(encoded) > 0
            and isinstance(encoded[0], (tuple, list))
            and len(encoded[0]) > 0
        ):
            encoded = encoded[0]

        if isinstance(encoded, (tuple, list)):
            cond = encoded[0]
            second = encoded[1] if len(encoded) > 1 else None
            if isinstance(second, dict):
                pooled = second.get("pooled_output")
                for k, v in second.items():
                    if k != "pooled_output":
                        extra_dict[k] = v
            else:
                pooled = second
        elif isinstance(encoded, dict):
            cond = encoded.get("cond", encoded.get("cross_attn"))
            pooled = encoded.get("pooled_output")
            for k, v in encoded.items():
                if k not in ("cond", "cross_attn", "pooled_output"):
                    extra_dict[k] = v
        else:
            cond = encoded
            pooled = None

        return cond, pooled if isinstance(pooled, torch.Tensor) else None, extra_dict

    @staticmethod
    def _stack_prompt_batch_outputs(
        batch_conds: List[torch.Tensor],
        batch_pools: List[torch.Tensor],
        batch_extra_lists: Dict[str, List[Any]],
        batch_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Stack per-sample CLIP outputs (`cond`, `pooled_output`, `extra_conds`) with sequence padding & mask synthesis."""
        orig_seq_lens = [
            int(t.shape[0]) if t.ndim == 2 else (int(t.shape[1]) if t.ndim >= 3 else 0)
            for t in batch_conds
        ]
        stacked_cond = _pad_and_stack_sequence_tensors(batch_conds)
        batch_seq_len = int(stacked_cond.shape[1]) if stacked_cond.ndim >= 2 else 0
        stacked_pool = torch.stack(batch_pools, dim=0) if (len(batch_pools) == batch_len and len(batch_pools) > 0) else None

        batch_extra: Dict[str, Any] = {}
        for ek, ev_list in batch_extra_lists.items():
            if len(ev_list) == batch_len and all(isinstance(v, torch.Tensor) for v in ev_list):
                single_list = [
                    (v[0] if v.ndim >= 2 and int(v.shape[0]) == 1 else v)
                    for v in ev_list
                ]
                if ek in ("attention_mask", "mask", "txt_mask") and all(v.ndim == 1 for v in single_list):
                    target_m_len = max(batch_seq_len, max(int(v.shape[0]) for v in single_list))
                    padded_masks = [
                        (F.pad(v, (0, target_m_len - int(v.shape[0])), value=0) if int(v.shape[0]) < target_m_len else v[:target_m_len])
                        for v in single_list
                    ]
                    batch_extra[ek] = torch.stack(padded_masks, dim=0)
                elif ek == "context_attention_bias" and all(v.ndim == 1 for v in single_list):
                    target_m_len = max(batch_seq_len, max(int(v.shape[0]) for v in single_list))
                    padded_biases = [
                        (F.pad(v, (0, target_m_len - int(v.shape[0])), value=-1e4) if int(v.shape[0]) < target_m_len else v[:target_m_len])
                        for v in single_list
                    ]
                    batch_extra[ek] = torch.stack(padded_biases, dim=0)
                elif all(v.shape == single_list[0].shape for v in single_list):
                    batch_extra[ek] = torch.stack(single_list, dim=0)
                elif all(v.ndim == 1 for v in single_list):
                    target_1d_len = max(int(v.shape[0]) for v in single_list)
                    padded_1d = [
                        (F.pad(v, (0, target_1d_len - int(v.shape[0])), value=0) if int(v.shape[0]) < target_1d_len else v[:target_1d_len])
                        for v in single_list
                    ]
                    batch_extra[ek] = torch.stack(padded_1d, dim=0)
                elif all(v.ndim >= 2 and v.shape[1:] == single_list[0].shape[1:] for v in single_list):
                    batch_extra[ek] = _pad_and_stack_sequence_tensors(single_list)
                else:
                    batch_extra[ek] = ev_list[-1]
            else:
                batch_extra[ek] = ev_list[-1]

        if (
            len(orig_seq_lens) > 1
            and batch_seq_len > 0
            and any(0 < s_len < batch_seq_len for s_len in orig_seq_lens)
        ):
            if not any(mk in batch_extra for mk in ("attention_mask", "mask", "txt_mask")):
                synth_mask = torch.zeros((len(orig_seq_lens), batch_seq_len), dtype=torch.long)
                for b_i, s_len in enumerate(orig_seq_lens):
                    synth_mask[b_i, : min(s_len, batch_seq_len)] = 1
                batch_extra["attention_mask"] = synth_mask
            if "context_attention_bias" not in batch_extra:
                src_mask = batch_extra.get("attention_mask")
                if isinstance(src_mask, torch.Tensor) and src_mask.ndim == 2 and int(src_mask.shape[1]) == batch_seq_len:
                    batch_extra["context_attention_bias"] = torch.where(
                        src_mask > 0,
                        torch.zeros_like(src_mask, dtype=stacked_cond.dtype),
                        torch.full_like(src_mask, -1e4, dtype=stacked_cond.dtype),
                    )
        return stacked_cond, stacked_pool, batch_extra

    def _encode_with_comfy_or_adapter(
        self,
        pixel_batches: List[torch.Tensor],
        prompt_batches: List[List[str]],
        dtype: torch.dtype,
        vae: Optional[Any],
        clip: Optional[Any],
        model: Optional[Any] = None,
        offload_between_encoders: bool = True,
        encode_flipped_latents: bool = False,
        mask_batches: Optional[List[Optional[torch.Tensor]]] = None,
        control_batches: Optional[List[List[Optional[torch.Tensor]]]] = None,
        control_source_batches: Optional[List[List[Optional[str]]]] = None,
        pass_control_to_clip: str = "auto",
        clip_vision_resolution: int = 384,
        clip_prompt_template: str = "",
        streaming_dataset: Optional[OmniLoRADataset] = None,
        streaming_batch_plans: Optional[List[List[Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], bool, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        if vae is None or not hasattr(vae, "encode") or clip is None or not hasattr(clip, "tokenize"):
            raise ValueError("Both 'vae' and 'clip' inputs are required for OmniLoRAPrecomputeConditioning.")

        use_streaming = streaming_dataset is not None and streaming_batch_plans is not None
        num_stage1 = len(streaming_batch_plans) if use_streaming and streaming_batch_plans is not None else len(pixel_batches)
        num_stage2 = len(streaming_batch_plans) if use_streaming and streaming_batch_plans is not None else len(prompt_batches)

        pbar: Optional[Any] = None
        try:
            import comfy.utils  # type: ignore

            pbar = comfy.utils.ProgressBar(num_stage1 + num_stage2)
        except Exception:
            pass

        latent_scaled = False
        inner_model = getattr(model, "model", model) if model is not None else None
        process_latent_in = getattr(inner_model, "process_latent_in", None) if inner_model is not None else None

        def _safe_process_latent_in(lat_tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
            if not callable(process_latent_in):
                return lat_tensor, False
            # Guard against stale `inner_model.latent_shapes` left on multimodal models (e.g.
            # MiniMaxH3 / LTXAV) by a previous ComfyUI KSampler run: single-stream image latents
            # must be scaled without multi-stream audio slice rescaling.
            had_stale_shapes = (
                inner_model is not None
                and getattr(inner_model, "latent_shapes", None) is not None
            )
            prev_shapes = getattr(inner_model, "latent_shapes", None) if had_stale_shapes else None
            if had_stale_shapes:
                try:
                    inner_model.latent_shapes = None
                except Exception:
                    had_stale_shapes = False
            try:
                return process_latent_in(lat_tensor), True
            except Exception:
                return lat_tensor, False
            finally:
                if had_stale_shapes:
                    try:
                        inner_model.latent_shapes = prev_shapes
                    except Exception:
                        pass

        def _downsample_single_mask_to_latent(mask_t: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
            with torch.inference_mode(False):
                m_cpu = mask_t.detach().to(device="cpu", dtype=torch.float32)
                if m_cpu.ndim == 2:
                    m_cpu = m_cpu.unsqueeze(0).unsqueeze(0)
                elif m_cpu.ndim == 3:
                    m_cpu = m_cpu.unsqueeze(0)
                if m_cpu.shape[-2:] != (lat_h, lat_w):
                    interp_mode = (
                        "area"
                        if (int(m_cpu.shape[-2]) >= lat_h and int(m_cpu.shape[-1]) >= lat_w)
                        else "bilinear"
                    )
                    if interp_mode == "bilinear":
                        m_lat = F.interpolate(m_cpu, size=(lat_h, lat_w), mode="bilinear", align_corners=False)
                    else:
                        m_lat = F.interpolate(m_cpu, size=(lat_h, lat_w), mode="area")
                else:
                    m_lat = m_cpu
                return m_lat.squeeze(0).clamp(0.0, 1.0).clone()

        resolved_ctrl_batches: List[List[Optional[torch.Tensor]]] = (
            list(control_batches)
            if (not use_streaming and control_batches is not None and len(control_batches) == len(pixel_batches))
            else ([] if use_streaming else [[None] * int(px.shape[0]) for px in pixel_batches])
        )
        resolved_ctrl_sources: List[List[Optional[str]]] = (
            list(control_source_batches)
            if (not use_streaming and control_source_batches is not None and len(control_source_batches) == len(pixel_batches))
            else ([] if use_streaming else [[None] * int(px.shape[0]) for px in pixel_batches])
        )

        # Stage 1: Encode target images AND optional control images one sample at a time ([1, H, W, C])
        # to minimize peak VAE VRAM AND peak CPU RAM (in streaming mode, only 1 decoded image pair
        # exists in host RAM at any moment), then immediately unload VAE before loading CLIP so both
        # encoders never coexist in VRAM.
        # IMPORTANT: Control latents are cached as RAW UNSCALED VAE outputs (no `process_latent_in`),
        # matching ComfyUI's `ReferenceLatent` and `InstructPixToPixConditioning` nodes, because
        # `BaseModel.extra_conds` / `concat_cond` applies `self.process_latent_in(...)` natively.
        cached_latents: List[torch.Tensor] = []
        cached_latents_flipped: List[Optional[torch.Tensor]] = []
        cached_ctrl_latents: List[Optional[List[Optional[torch.Tensor]]]] = []
        cached_ctrl_latents_flipped: List[Optional[List[Optional[torch.Tensor]]]] = []
        streamed_mask_batches: List[Optional[torch.Tensor]] = []
        pixel_hw_list: List[Tuple[int, int]] = []
        try:
            with torch.no_grad():
                for batch_i in range(num_stage1):
                    _check_comfy_interrupt()
                    if use_streaming and streaming_batch_plans is not None and streaming_dataset is not None:
                        chunk = streaming_batch_plans[batch_i]
                        bh, bw = int(chunk[0].bucket_size[0]), int(chunk[0].bucket_size[1])
                        pixel_hw_list.append((bh, bw))
                        batch_len = len(chunk)
                        pixels_batch_ref: Optional[torch.Tensor] = None
                        ctrl_px_list: List[Optional[torch.Tensor]] = []
                    else:
                        chunk = None
                        pixels_batch_ref = pixel_batches[batch_i]
                        if pixels_batch_ref.ndim >= 4:
                            pixel_hw_list.append((int(pixels_batch_ref.shape[-2]), int(pixels_batch_ref.shape[-1])))
                        else:
                            pixel_hw_list.append((0, 0))
                        batch_len = int(pixels_batch_ref.shape[0])
                        ctrl_px_list = resolved_ctrl_batches[batch_i]

                    sample_lats: List[torch.Tensor] = []
                    sample_lats_flip: List[torch.Tensor] = []
                    sample_ctrl_lats: List[Optional[torch.Tensor]] = []
                    sample_ctrl_lats_flip: List[Optional[torch.Tensor]] = []
                    sample_lat_masks: List[Optional[torch.Tensor]] = []
                    has_batch_ctrl = False
                    has_batch_mask = False

                    for idx in range(batch_len):
                        _check_comfy_interrupt()
                        raw_sample_mask: Optional[torch.Tensor] = None
                        if use_streaming and chunk is not None and streaming_dataset is not None:
                            px_chw, raw_sample_mask, ctrl_px = streaming_dataset.decode_sample_tensors(chunk[idx])
                            single_px = px_chw.unsqueeze(0)
                        else:
                            assert pixels_batch_ref is not None
                            single_px = pixels_batch_ref[idx : idx + 1]
                            ctrl_px = ctrl_px_list[idx] if idx < len(ctrl_px_list) else None

                        comfy_pixels = ((single_px.permute(0, 2, 3, 1) + 1.0) * 0.5).clamp(0.0, 1.0)
                        lat = vae.encode(comfy_pixels)
                        lat, did_scale = _safe_process_latent_in(lat)
                        if did_scale:
                            latent_scaled = True
                        with torch.inference_mode(False):
                            lat_cpu = lat.detach().to(device="cpu", dtype=dtype).clone()
                            sample_lats.append(lat_cpu)
                        if use_streaming:
                            if isinstance(raw_sample_mask, torch.Tensor) and raw_sample_mask.numel() > 0 and lat_cpu.ndim >= 4:
                                has_batch_mask = True
                                sample_lat_masks.append(
                                    _downsample_single_mask_to_latent(
                                        raw_sample_mask,
                                        int(lat_cpu.shape[-2]),
                                        int(lat_cpu.shape[-1]),
                                    )
                                )
                            else:
                                sample_lat_masks.append(None)
                        del lat, comfy_pixels

                        if encode_flipped_latents:
                            _check_comfy_interrupt()
                            flipped_px = torch.flip(single_px, dims=[-1])
                            comfy_pixels_flip = ((flipped_px.permute(0, 2, 3, 1) + 1.0) * 0.5).clamp(0.0, 1.0)
                            lat_flip = vae.encode(comfy_pixels_flip)
                            lat_flip, did_scale_flip = _safe_process_latent_in(lat_flip)
                            if did_scale_flip:
                                latent_scaled = True
                            with torch.inference_mode(False):
                                lat_flip_cpu = lat_flip.detach().to(device="cpu", dtype=dtype).clone()
                                sample_lats_flip.append(lat_flip_cpu)
                            del lat_flip, comfy_pixels_flip, flipped_px

                        if isinstance(ctrl_px, torch.Tensor) and ctrl_px.ndim >= 3 and ctrl_px.numel() > 0:
                            has_batch_ctrl = True
                            _check_comfy_interrupt()
                            single_ctrl = ctrl_px.unsqueeze(0) if ctrl_px.ndim == 3 else ctrl_px[0:1]
                            comfy_ctrl = ((single_ctrl.permute(0, 2, 3, 1) + 1.0) * 0.5).clamp(0.0, 1.0)
                            c_lat = vae.encode(comfy_ctrl)
                            with torch.inference_mode(False):
                                c_lat_cpu = c_lat.detach().to(device="cpu", dtype=dtype).clone()
                                sample_ctrl_lats.append(c_lat_cpu)
                            del c_lat, comfy_ctrl

                            if encode_flipped_latents:
                                _check_comfy_interrupt()
                                flipped_ctrl = torch.flip(single_ctrl, dims=[-1])
                                comfy_ctrl_flip = ((flipped_ctrl.permute(0, 2, 3, 1) + 1.0) * 0.5).clamp(0.0, 1.0)
                                c_lat_flip = vae.encode(comfy_ctrl_flip)
                                with torch.inference_mode(False):
                                    c_lat_flip_cpu = c_lat_flip.detach().to(device="cpu", dtype=dtype).clone()
                                    sample_ctrl_lats_flip.append(c_lat_flip_cpu)
                                del c_lat_flip, comfy_ctrl_flip, flipped_ctrl
                            del single_ctrl
                        else:
                            sample_ctrl_lats.append(None)
                            if encode_flipped_latents:
                                sample_ctrl_lats_flip.append(None)

                        if use_streaming:
                            del px_chw, raw_sample_mask, ctrl_px, single_px

                    with torch.inference_mode(False):
                        batch_lat = torch.cat(sample_lats, dim=0) if len(sample_lats) > 1 else sample_lats[0]
                        cached_latents.append(batch_lat)
                        if encode_flipped_latents and sample_lats_flip:
                            batch_lat_flip = (
                                torch.cat(sample_lats_flip, dim=0)
                                if len(sample_lats_flip) > 1
                                else sample_lats_flip[0]
                            )
                            cached_latents_flipped.append(batch_lat_flip)
                        else:
                            cached_latents_flipped.append(None)
                        cached_ctrl_latents.append(sample_ctrl_lats if has_batch_ctrl else None)
                        cached_ctrl_latents_flipped.append(
                            sample_ctrl_lats_flip if (has_batch_ctrl and encode_flipped_latents) else None
                        )
                        if use_streaming:
                            if has_batch_mask and batch_lat.ndim >= 4:
                                lat_h, lat_w = int(batch_lat.shape[-2]), int(batch_lat.shape[-1])
                                full_ones_m: Optional[torch.Tensor] = None
                                filled_masks: List[torch.Tensor] = []
                                for sm in sample_lat_masks:
                                    if sm is not None:
                                        filled_masks.append(sm)
                                    else:
                                        if full_ones_m is None:
                                            full_ones_m = torch.ones((1, lat_h, lat_w), dtype=torch.float32, device="cpu")
                                        filled_masks.append(full_ones_m)
                                streamed_mask_batches.append(torch.stack(filled_masks, dim=0).to(dtype=torch.float32))
                            else:
                                streamed_mask_batches.append(None)
                    if pbar is not None:
                        pbar.update(1)
        finally:
            if offload_between_encoders:
                _try_comfy_unload_models()

        # Stage 2: Encode prompts with deduplication cache so repeated captions (or identical
        # `(caption, control_image)` pairs on VLM text encoders) are encoded only once,
        # plus encode the unconditional empty prompt `""` once for dynamic per-step CFG caption dropout.
        pass_ctrl_mode = str(pass_control_to_clip or "auto").strip().lower()
        clip_supports_images = self._clip_supports_images_kwarg(clip)
        clip_supports_llama_tpl = self._clip_supports_llama_template_kwarg(clip)
        use_clip_vision = (
            True
            if pass_ctrl_mode == "enabled"
            else (False if pass_ctrl_mode == "disabled" else clip_supports_images)
        )
        if pass_ctrl_mode == "enabled" and not clip_supports_images:
            logging.warning(
                "[OmniLoRA] pass_control_to_clip='enabled' requested, but connected CLIP tokenizer does not "
                "declare an `images` parameter; attempting pass-through with safe text-only fallback."
            )

        cached_embeds: List[torch.Tensor] = []
        cached_pooled: List[Optional[torch.Tensor]] = []
        cached_extras: List[Dict[str, Any]] = []
        cached_embeds_flipped: List[Optional[torch.Tensor]] = []
        cached_pooled_flipped: List[Optional[torch.Tensor]] = []
        cached_extras_flipped: List[Optional[Dict[str, Any]]] = []
        prompt_cache: Dict[Tuple[str, str, bool, str], Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]] = {}
        uncond_embed: Optional[torch.Tensor] = None
        uncond_pooled: Optional[torch.Tensor] = None
        uncond_extra: Dict[str, Any] = {}
        clip_actually_used_control = False

        def _get_or_encode_prompt(
            text: str,
            ctrl_px: Optional[torch.Tensor] = None,
            ctrl_src_key: str = "",
            is_flipped: bool = False,
            apply_template: bool = True,
            ctrl_px_loader: Optional[Any] = None,
            has_ctrl: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
            nonlocal clip_actually_used_control
            formatted_text, llama_tpl = (
                self._apply_prompt_template(text, clip_prompt_template, clip_supports_llama_tpl)
                if apply_template
                else (text, None)
            )
            has_ctrl_img = bool(
                has_ctrl
                or (isinstance(ctrl_px, torch.Tensor) and ctrl_px.ndim >= 3 and ctrl_px.numel() > 0)
            )
            use_img_here = bool(use_clip_vision and has_ctrl_img)
            cache_key = (
                formatted_text,
                ctrl_src_key if use_img_here else "",
                bool(is_flipped) if use_img_here else False,
                llama_tpl or "",
            )
            if cache_key in prompt_cache:
                return prompt_cache[cache_key]

            clip_imgs: Optional[List[torch.Tensor]] = None
            if use_img_here:
                resolved_px = ctrl_px
                if resolved_px is None and callable(ctrl_px_loader):
                    resolved_px = ctrl_px_loader()
                if isinstance(resolved_px, torch.Tensor) and resolved_px.ndim >= 3 and resolved_px.numel() > 0:
                    c_in = torch.flip(resolved_px, dims=[-1]) if is_flipped else resolved_px
                    clip_imgs = [self._resize_control_for_clip_vision(c_in, int(clip_vision_resolution))]
                    clip_actually_used_control = True
                    del c_in
                del resolved_px

            cond, pooled, extra_dict = self._encode_single_prompt(
                clip,
                formatted_text,
                control_images=clip_imgs,
                llama_template=llama_tpl,
            )
            with torch.inference_mode(False):
                c_raw = cond.detach().to(device="cpu", dtype=dtype).clone()
                c_clean = c_raw[0] if c_raw.ndim == 3 else c_raw
                p_clean = None
                if isinstance(pooled, torch.Tensor):
                    p_raw = pooled.detach().to(device="cpu", dtype=dtype).clone()
                    p_clean = p_raw[0] if p_raw.ndim >= 2 else p_raw
                clean_extra: Dict[str, Any] = {}
                for ek, ev in extra_dict.items():
                    if isinstance(ev, torch.Tensor):
                        clean_extra[ek] = ev.detach().to(device="cpu").clone()
                    else:
                        clean_extra[ek] = ev
            prompt_cache[cache_key] = (c_clean, p_clean, clean_extra)
            return c_clean, p_clean, clean_extra

        try:
            with torch.no_grad():
                for batch_i in range(num_stage2):
                    _check_comfy_interrupt()
                    if use_streaming and streaming_batch_plans is not None and streaming_dataset is not None:
                        chunk = streaming_batch_plans[batch_i]
                        prompts = [s.caption for s in chunk]
                    else:
                        chunk = None
                        prompts = prompt_batches[batch_i]
                        ctrl_px_list = resolved_ctrl_batches[batch_i]
                        ctrl_src_list = resolved_ctrl_sources[batch_i]

                    batch_conds: List[torch.Tensor] = []
                    batch_pools: List[torch.Tensor] = []
                    batch_extra_lists: Dict[str, List[Any]] = {}
                    batch_has_vlm_ctrl = False

                    for p_i, text in enumerate(prompts):
                        _check_comfy_interrupt()
                        if use_streaming and chunk is not None and streaming_dataset is not None:
                            sample_obj = chunk[p_i]
                            c_has = bool(sample_obj.has_control)
                            c_src = (sample_obj.control_source_name or f"ctrl_{batch_i}_{p_i}") if c_has else ""
                            c_px = None
                            c_loader = (
                                (lambda _s=sample_obj: streaming_dataset.decode_sample_control_tensor(_s))
                                if c_has
                                else None
                            )
                        else:
                            c_px = ctrl_px_list[p_i] if p_i < len(ctrl_px_list) else None
                            c_has = bool(isinstance(c_px, torch.Tensor) and c_px.ndim >= 3 and c_px.numel() > 0)
                            c_src = (
                                (ctrl_src_list[p_i] or f"ctrl_{batch_i}_{p_i}")
                                if (p_i < len(ctrl_src_list) and c_has)
                                else ""
                            )
                            c_loader = None

                        if use_clip_vision and c_has:
                            batch_has_vlm_ctrl = True
                        c_clean, p_clean, clean_extra = _get_or_encode_prompt(
                            text,
                            ctrl_px=c_px,
                            ctrl_src_key=c_src,
                            is_flipped=False,
                            ctrl_px_loader=c_loader,
                            has_ctrl=c_has,
                        )
                        batch_conds.append(c_clean)
                        if isinstance(p_clean, torch.Tensor):
                            batch_pools.append(p_clean)
                        for ek, ev in clean_extra.items():
                            batch_extra_lists.setdefault(ek, []).append(ev)

                    with torch.inference_mode(False):
                        stacked_cond, stacked_pool, batch_extra = self._stack_prompt_batch_outputs(
                            batch_conds, batch_pools, batch_extra_lists, len(prompts)
                        )
                        cached_embeds.append(stacked_cond)
                        cached_pooled.append(stacked_pool)
                        cached_extras.append(batch_extra)

                    # When random_flip is active AND a VLM CLIP consumed the control image, also cache
                    # the horizontally flipped CLIP conditioning so target, control latent, and VLM
                    # visual tokens stay 100% geometrically aligned on flipped steps.
                    if encode_flipped_latents and batch_has_vlm_ctrl:
                        flip_conds: List[torch.Tensor] = []
                        flip_pools: List[torch.Tensor] = []
                        flip_extra_lists: Dict[str, List[Any]] = {}
                        for p_i, text in enumerate(prompts):
                            _check_comfy_interrupt()
                            if use_streaming and chunk is not None and streaming_dataset is not None:
                                sample_obj = chunk[p_i]
                                c_has = bool(sample_obj.has_control)
                                c_src = (sample_obj.control_source_name or f"ctrl_{batch_i}_{p_i}") if c_has else ""
                                c_px = None
                                c_loader = (
                                    (lambda _s=sample_obj: streaming_dataset.decode_sample_control_tensor(_s))
                                    if c_has
                                    else None
                                )
                            else:
                                c_px = ctrl_px_list[p_i] if p_i < len(ctrl_px_list) else None
                                c_has = bool(isinstance(c_px, torch.Tensor) and c_px.ndim >= 3 and c_px.numel() > 0)
                                c_src = (
                                    (ctrl_src_list[p_i] or f"ctrl_{batch_i}_{p_i}")
                                    if (p_i < len(ctrl_src_list) and c_has)
                                    else ""
                                )
                                c_loader = None
                            fc_clean, fp_clean, f_extra = _get_or_encode_prompt(
                                text,
                                ctrl_px=c_px,
                                ctrl_src_key=c_src,
                                is_flipped=True,
                                ctrl_px_loader=c_loader,
                                has_ctrl=c_has,
                            )
                            flip_conds.append(fc_clean)
                            if isinstance(fp_clean, torch.Tensor):
                                flip_pools.append(fp_clean)
                            for ek, ev in f_extra.items():
                                flip_extra_lists.setdefault(ek, []).append(ev)
                        with torch.inference_mode(False):
                            f_cond, f_pool, f_ext = self._stack_prompt_batch_outputs(
                                flip_conds, flip_pools, flip_extra_lists, len(prompts)
                            )
                            cached_embeds_flipped.append(f_cond)
                            cached_pooled_flipped.append(f_pool)
                            cached_extras_flipped.append(f_ext)
                    else:
                        cached_embeds_flipped.append(None)
                        cached_pooled_flipped.append(None)
                        cached_extras_flipped.append(None)

                    if pbar is not None:
                        pbar.update(1)

                # Encode the unconditional empty prompt `""` once so `OmniLoRATrainer` can swap it in
                # dynamically on any training step when `caption_dropout > 0.0`.
                _check_comfy_interrupt()
                u_cond, u_pool, u_ext = _get_or_encode_prompt("", ctrl_px=None, apply_template=False)
                with torch.inference_mode(False):
                    uncond_embed = u_cond.unsqueeze(0).clone() if u_cond.ndim == 2 else u_cond.clone()
                    if isinstance(u_pool, torch.Tensor):
                        uncond_pooled = u_pool.unsqueeze(0).clone() if u_pool.ndim == 1 else u_pool.clone()
                    uncond_extra = {
                        ek: (
                            (ev.unsqueeze(0).clone() if ev.ndim == 1 else ev.clone())
                            if isinstance(ev, torch.Tensor)
                            else ev
                        )
                        for ek, ev in u_ext.items()
                    }
        finally:
            prompt_cache.clear()
            if offload_between_encoders:
                _try_comfy_unload_models()

        # If `model` is connected to Node 2, validate VAE latent channels and CLIP width immediately
        if model is not None and cached_latents and cached_embeds:
            adapter = resolve_model_adapter(model_id="auto", comfy_model=model)
            backbone = adapter.diffusion_model
            if isinstance(backbone, nn.Module):
                adapter._validate_and_prepare_latents(
                    cached_latents[0],
                    backbone=backbone,
                    latent_scaled=latent_scaled,
                )
                adapter._validate_context_dim(cached_embeds[0], backbone=backbone)

        batches: List[Dict[str, Any]] = []
        if use_streaming:
            resolved_masks: List[Optional[torch.Tensor]] = streamed_mask_batches
        else:
            resolved_masks = (
                list(mask_batches)
                if mask_batches is not None and len(mask_batches) == len(cached_latents)
                else [None] * len(cached_latents)
            )
        lat_scale = int(
            (getattr(getattr(inner_model, "latent_format", None), "spacial_downscale_ratio", 0) if inner_model is not None else 0)
            or getattr(vae, "spacial_downscale_ratio", 0)
            or 8
        )
        for batch_idx, (
            lat,
            lat_flip,
            emb,
            pool,
            extra,
            emb_flip,
            pool_flip,
            extra_flip,
            raw_mask_b,
            ctrl_lat_b,
            ctrl_lat_flip_b,
        ) in enumerate(zip(
            cached_latents,
            cached_latents_flipped,
            cached_embeds,
            cached_pooled,
            cached_extras,
            cached_embeds_flipped,
            cached_pooled_flipped,
            cached_extras_flipped,
            resolved_masks,
            cached_ctrl_latents,
            cached_ctrl_latents_flipped,
        )):
            px_h, px_w = pixel_hw_list[batch_idx] if batch_idx < len(pixel_hw_list) else (0, 0)
            if px_h > 0 and px_w > 0:
                bucket_str = f"{px_w}x{px_h}"
            else:
                bucket_str = f"{int(lat.shape[-1]) * lat_scale}x{int(lat.shape[-2]) * lat_scale}"
            item: Dict[str, Any] = {
                "latents": lat,
                "prompt_embeds": emb,
                "latent_scaled": latent_scaled,
                "bucket": bucket_str,
            }
            if lat_flip is not None:
                item["latents_flipped"] = lat_flip
            if pool is not None:
                item["pooled_output"] = pool
            if extra:
                item["extra_conds"] = extra
            if emb_flip is not None:
                item["prompt_embeds_flipped"] = emb_flip
            if pool_flip is not None:
                item["pooled_output_flipped"] = pool_flip
            if extra_flip:
                item["extra_conds_flipped"] = extra_flip
            if ctrl_lat_b is not None:
                item["control_latents"] = ctrl_lat_b
            if ctrl_lat_flip_b is not None:
                item["control_latents_flipped"] = ctrl_lat_flip_b
            if isinstance(raw_mask_b, torch.Tensor) and raw_mask_b.numel() > 0 and lat.ndim >= 4:
                with torch.inference_mode(False):
                    m_cpu = raw_mask_b.detach().to(device="cpu", dtype=torch.float32)
                    if m_cpu.ndim == 3:
                        m_cpu = m_cpu.unsqueeze(1)
                    lat_h, lat_w = int(lat.shape[-2]), int(lat.shape[-1])
                    if m_cpu.shape[-2:] != (lat_h, lat_w):
                        interp_mode = (
                            "area"
                            if (int(m_cpu.shape[-2]) >= lat_h and int(m_cpu.shape[-1]) >= lat_w)
                            else "bilinear"
                        )
                        if interp_mode == "bilinear":
                            m_lat = F.interpolate(m_cpu, size=(lat_h, lat_w), mode="bilinear", align_corners=False)
                        else:
                            m_lat = F.interpolate(m_cpu, size=(lat_h, lat_w), mode="area")
                    else:
                        m_lat = m_cpu
                    m_lat_clean = m_lat.clamp(0.0, 1.0).clone()
                    item["loss_mask"] = m_lat_clean
                    if encode_flipped_latents:
                        item["loss_mask_flipped"] = torch.flip(m_lat_clean, dims=[-1]).clone()
            batches.append(item)

        self._last_clip_used_control_images = clip_actually_used_control
        return batches, latent_scaled, uncond_embed, uncond_pooled, uncond_extra

    def precompute(
        self,
        dataset: OmniLoRADataset,
        offload_encoders_after_cache: bool = True,
        vae: Optional[Any] = None,
        clip: Optional[Any] = None,
        model: Optional[Any] = None,
        pass_control_to_clip: str = "auto",
        clip_vision_resolution: int = 384,
        clip_prompt_template: str = "",
    ) -> Tuple[CachedConditioningBundle]:
        plan = detect_and_configure(preferred_backend="auto", requested_dtype="auto")
        dtype = plan.backend.torch_dtype

        # Encode unique samples once (temporarily setting num_repeats=1 if >1), then expand repeats in CPU RAM.
        # Prefer lazy streaming (`build_streaming_batch_plans`) so folder-backed datasets decode only
        # 1 image pair at a time in O(1) CPU RAM instead of materializing the whole dataset up front.
        orig_repeats = max(1, dataset.config.num_repeats)
        dataset.config.num_repeats = 1
        use_streaming_plans = hasattr(dataset, "build_streaming_batch_plans") and hasattr(
            dataset, "decode_sample_tensors"
        )
        unique_batch_plans: Optional[List[List[Any]]] = None
        unique_pixel_batches: List[torch.Tensor] = []
        unique_prompt_batches: List[List[str]] = []
        unique_mask_batches: List[Optional[torch.Tensor]] = []
        unique_control_batches: List[List[Optional[torch.Tensor]]] = []
        unique_control_source_batches: List[List[Optional[str]]] = []
        try:
            if use_streaming_plans:
                unique_batch_plans = dataset.build_streaming_batch_plans()
            elif hasattr(dataset, "build_batches_full"):
                (
                    unique_pixel_batches,
                    unique_prompt_batches,
                    unique_mask_batches,
                    unique_control_batches,
                    unique_control_source_batches,
                ) = dataset.build_batches_full()
            elif hasattr(dataset, "build_batches_with_masks"):
                unique_pixel_batches, unique_prompt_batches, unique_mask_batches = dataset.build_batches_with_masks()
                unique_control_batches = [[None] * int(px.shape[0]) for px in unique_pixel_batches]
                unique_control_source_batches = [[None] * int(px.shape[0]) for px in unique_pixel_batches]
            else:
                unique_pixel_batches, unique_prompt_batches = dataset.build_batches()
                unique_mask_batches = [None] * len(unique_pixel_batches)
                unique_control_batches = [[None] * int(px.shape[0]) for px in unique_pixel_batches]
                unique_control_source_batches = [[None] * int(px.shape[0]) for px in unique_pixel_batches]
        finally:
            dataset.config.num_repeats = orig_repeats

        self._last_clip_used_control_images = False
        (
            unique_cached_batches,
            latent_scaled,
            uncond_embed,
            uncond_pooled,
            uncond_extra,
        ) = self._encode_with_comfy_or_adapter(
            pixel_batches=unique_pixel_batches,
            prompt_batches=unique_prompt_batches,
            dtype=dtype,
            vae=vae,
            clip=clip,
            model=model,
            offload_between_encoders=offload_encoders_after_cache,
            encode_flipped_latents=bool(dataset.config.random_flip),
            mask_batches=unique_mask_batches,
            control_batches=unique_control_batches,
            control_source_batches=unique_control_source_batches,
            pass_control_to_clip=pass_control_to_clip,
            clip_vision_resolution=int(clip_vision_resolution),
            clip_prompt_template=str(clip_prompt_template or ""),
            streaming_dataset=dataset if use_streaming_plans else None,
            streaming_batch_plans=unique_batch_plans if use_streaming_plans else None,
        )
        has_ctrl_latents = any(
            isinstance(b.get("control_latents"), list) and any(c is not None for c in b["control_latents"])
            for b in unique_cached_batches
        )
        num_ctrl_imgs = int(getattr(dataset, "num_controlled_images", 0) or 0)
        del (
            unique_batch_plans,
            unique_pixel_batches,
            unique_prompt_batches,
            unique_mask_batches,
            unique_control_batches,
            unique_control_source_batches,
        )

        # Free uncompressed [3, H, W] float32 pixel & control tensors from host RAM now that latents are cached
        if hasattr(dataset, "release_pixel_memory"):
            dataset.release_pixel_memory()
        gc.collect()
        trim_process_working_set()

        cached_batches: List[Dict[str, Any]] = []
        for _ in range(orig_repeats):
            cached_batches.extend(unique_cached_batches)

        bundle = CachedConditioningBundle(
            batches=cached_batches,
            resolution=dataset.config.resolution,
            batch_size=dataset.config.batch_size,
            trigger_word=dataset.config.trigger_word,
            num_unique_images=dataset.num_unique_images,
            total_batches=len(cached_batches),
            latent_scaled=latent_scaled,
            random_flip=bool(dataset.config.random_flip),
            caption_dropout=float(dataset.config.caption_dropout),
            uncond_prompt_embeds=uncond_embed,
            uncond_pooled_output=uncond_pooled,
            uncond_extra_conds=uncond_extra,
            has_control_latents=has_ctrl_latents,
            clip_used_control_images=bool(getattr(self, "_last_clip_used_control_images", False)),
            num_controlled_images=num_ctrl_imgs,
        )
        return (bundle,)


class OmniLoRALoader:
    """
    ComfyUI Node 3: Load an existing LoRA `.safetensors` or `.pt` checkpoint as a starting
    `OMNILORA_NETWORK` so `OmniLoRATrainer` can seamlessly resume or continue training it.
    Automatically detects the loaded LoRA's `rank`, `alpha`, and prior `steps_trained`.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "lora_name": (
                    _list_available_comfy_loras(),
                    {
                        "tooltip": "Select an existing LoRA file from ComfyUI/models/loras/ to continue training.",
                    },
                ),
            },
            "optional": {
                "custom_lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional explicit file path to a .safetensors or .pt LoRA (overrides lora_name when non-empty).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("OMNILORA_NETWORK", "STRING")
    RETURN_NAMES = ("lora", "lora_info")
    FUNCTION = "load_lora"
    CATEGORY = "OmniLoRA/Training"

    def load_lora(
        self,
        lora_name: str,
        custom_lora_path: str = "",
    ) -> Tuple[OmniLoRANetworkBundle, str]:
        with torch.inference_mode(False):
            custom_clean = (custom_lora_path or "").strip().strip("\"'")
            resolved_path: Optional[Path] = None

            if custom_clean:
                resolved_path = Path(custom_clean).expanduser().resolve()
                if resolved_path.is_dir() and lora_name and lora_name != "(none)":
                    resolved_path = (resolved_path / lora_name).resolve()
            else:
                if not lora_name or lora_name == "(none)":
                    raise ValueError(
                        "[OmniLoRA] No LoRA file selected in OmniLoRALoader. "
                        "Place a .safetensors file in ComfyUI/models/loras/ or provide custom_lora_path."
                    )
                try:
                    import folder_paths  # type: ignore

                    full_p = folder_paths.get_full_path("loras", lora_name)
                    if full_p:
                        resolved_path = Path(full_p)
                except Exception:
                    pass
                if resolved_path is None:
                    resolved_path = _resolve_comfy_loras_dir() / lora_name

            state_dict, metadata = _load_lora_safetensors_or_pt(resolved_path)
            inspected = LoRAInjector.inspect_lora_state_dict(state_dict, metadata=metadata)

            bundle = OmniLoRANetworkBundle(
                state_dict=state_dict,
                rank=int(inspected["rank"]),
                alpha=float(inspected["alpha"]),
                model_architecture=str(inspected.get("model_architecture", "auto")),
                steps_trained=int(inspected.get("steps_trained", 0)),
                trigger_word=str(inspected.get("trigger_word", "")),
                metadata=metadata,
                source_name=resolved_path.stem,
            )

            logging.info(
                f"[OmniLoRA] Loaded starting LoRA '{resolved_path.name}': "
                f"rank={bundle.rank}, alpha={bundle.alpha}, modules={inspected['num_lora_modules']}, "
                f"prior_steps={bundle.steps_trained}."
            )

            info_summary = {
                "loaded_file": resolved_path.name,
                "loaded_path": str(resolved_path),
                "detected_rank": bundle.rank,
                "detected_alpha": bundle.alpha,
                "prior_steps_trained": bundle.steps_trained,
                "model_architecture": bundle.model_architecture,
                "trigger_word": bundle.trigger_word,
                "lora_module_count": inspected["num_lora_modules"],
                "tensor_count": len(state_dict),
            }
            return (bundle, json.dumps(info_summary, indent=2))


class OmniLoRATrainer:
    """
    ComfyUI Node 4: Universal Multi-Backend PyTorch LoRA Trainer (`xpu`, `rocm`, `cuda`, `npu`, `hpu`, `dml`, `mps`, `cpu`).
    - Exits ComfyUI's `@torch.inference_mode()` and enforces a dynamic VRAM headroom budget across any GPU size.
    - Optionally accepts `initial_lora` (`OMNILORA_NETWORK`) from `OmniLoRALoader` (or a previous trainer node)
      to seamlessly continue training an existing LoRA, automatically adopting its `rank` and `alpha`.
    - Supports multiple FP64-free optimizers (`radamw`, `adamw`, `adafactor`, `sgd`), LR schedulers (`cosine`, `rex`,
      `constant_with_warmup`, `constant`, `linear`, `cosine_with_restarts`), and optional periodic auto-checkpointing.
    - Applies `random_flip` and `caption_dropout` dynamically per training step using pre-cached flipped latents
      and unconditional prompt embeddings.
    - Outputs both a patched ComfyUI `MODEL` and a modular `OMNILORA_NETWORK` bundle for `OmniLoRASave` or chaining.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Connected ComfyUI MODEL from Load Diffusion Model / UNETLoader / CheckpointLoaderSimple.",
                    },
                ),
                "cached_data": ("OMNILORA_CACHED_DATA",),
                "model_architecture": (
                    ["auto"],
                    {
                        "default": "auto",
                        "tooltip": "Target diffusion/flow architecture (automatically configured from connected ComfyUI MODEL).",
                    },
                ),
                "backend": (
                    ["auto", "xpu", "rocm", "cuda", "tpu", "npu", "hpu", "dml", "mps", "cpu"],
                    {"default": "auto"},
                ),
                "reserve_vram_gb": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.5,
                        "tooltip": "Extra GPU VRAM (in GB) to reserve/leave free for the OS or display driver (recommended 0.5-1.5 GB if your monitor is connected to this GPU). Leave at 0.0 for maximum VRAM utilization.",
                    },
                ),
                "optimizer": (
                    ["radamw", "adamw", "adafactor", "sgd"],
                    {
                        "default": "radamw",
                        "tooltip": "Optimization algorithm (all FP64-free). RAdamW combines AdamW weight decay with variance warmup; Adafactor uses factored 2nd moments + stochastic rounding to cut optimizer VRAM by ~75-85%.",
                    },
                ),
                "rank": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "LoRA rank (automatically overridden when initial_lora is connected).",
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 32.0,
                        "min": 1.0,
                        "max": 128.0,
                        "step": 1.0,
                        "tooltip": "LoRA alpha scaling factor (automatically overridden when initial_lora is connected).",
                    },
                ),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.000001,
                        "max": 0.01,
                        "step": 0.000001,
                        "tooltip": "Peak base learning rate for LoRA_A (down-projection) weights.",
                    },
                ),
                "lr_scheduler": (
                    ["cosine", "rex", "constant_with_warmup", "constant", "linear", "cosine_with_restarts"],
                    {
                        "default": "cosine",
                        "tooltip": "Learning rate schedule across training steps. 'cosine' and 'rex' decay smoothly toward min_lr_ratio * learning_rate; 'constant_with_warmup' holds peak LR after warmup.",
                    },
                ),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 5000,
                        "step": 1,
                        "tooltip": "Number of initial steps to linearly warm up the learning rate from 0 to peak learning_rate (5% of total steps is optimal; automatically clamped to at most 50% of total steps).",
                    },
                ),
                "steps": ("INT", {"default": 1000, "min": 1, "max": 20000, "step": 1}),
                "grad_accum_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Gradient accumulation sub-steps per optimizer step (default 1).",
                    },
                ),
                "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                "checkpoint_every_n_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 20000,
                        "step": 1,
                        "tooltip": "Automatically save an intermediate checkpoint every N steps to custom_checkpoint_dir or ComfyUI/models/loras/ (0 = off by default).",
                    },
                ),
            },
            "optional": {
                "initial_lora": (
                    "OMNILORA_NETWORK",
                    {
                        "tooltip": "Optional starting LoRA network from OmniLoRA Load LoRA Network (or a previous trainer node) to continue training without starting over.",
                    },
                ),
                "checkpoint_prefix": (
                    "STRING",
                    {
                        "default": "omni_lora_ckpt",
                        "multiline": False,
                        "tooltip": "Filename prefix used when checkpoint_every_n_steps > 0.",
                    },
                ),
                "custom_checkpoint_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional custom output folder for periodic checkpoints when checkpoint_every_n_steps > 0 (defaults to ComfyUI/models/loras/ when empty; bare subfolder names resolve inside ComfyUI/models/loras/).",
                    },
                ),
                "lora_dtype": (
                    ["bf16", "fp32", "fp16"],
                    {
                        "default": "bf16",
                        "tooltip": "Precision of trainable LoRA adapter weights. bf16 is default (automatically falls back to fp16 if hardware does not support bf16); fp32 keeps master LoRA weights in 32-bit float.",
                    },
                ),
                "lora_plus_lr_ratio": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 16.0,
                        "step": 0.5,
                        "tooltip": "LoRA+ differential learning rate multiplier for LoRA_B (up-projection) relative to LoRA_A (1.0 = standard 1:1 LoRA LR; 2.0-4.0 = LoRA+ accelerated B matrix).",
                    },
                ),
                "min_lr_ratio": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Minimum learning rate at the end of decay schedules (cosine, rex, linear) as a fraction of peak learning_rate (default 0.1 = 10%).",
                    },
                ),
                "weight_decay": (
                    "FLOAT",
                    {
                        "default": 0.001,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.0005,
                        "tooltip": "Decoupled weight decay regularization factor for RAdamW / AdamW / SGD (0.001 = optimal light regularization for LoRA adapters).",
                    },
                ),
                "use_rslora": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Rank-Stabilized LoRA (rsLoRA): scales adapter output by alpha / sqrt(rank) instead of alpha / rank (recommended when rank >= 32).",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "Random seed for timestep sampling, noise generation, batch shuffling, and dynamic per-step caption dropout / horizontal flips.",
                    },
                ),
                "enable_target_module_filter": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Safety toggle for module filtering. When False (default), OmniLoRA automatically targets all optimal Attention/FFN blocks (and convs if enabled), ignoring any text in the filter box. When True, applies the filter patterns below.",
                    },
                ),
                "target_module_filter": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional comma-separated module path substrings to restrict LoRA injection to a specific sub-network (e.g. 'layers' vs 'text_embedder.blocks', or 'double_blocks' vs 'single_blocks'). Requires enable_target_module_filter to be True.",
                    },
                ),
                "min_timestep": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Minimum normalized noise/timestep fraction (0.0 = cleanest latent, 1.0 = pure noise). Useful for training dual-model or high/low-noise expert networks on separate timestep ranges (e.g. [0.0, 0.5] and [0.5, 1.0]).",
                    },
                ),
                "max_timestep": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Maximum normalized noise/timestep fraction (0.0 = cleanest latent, 1.0 = pure noise).",
                    },
                ),
                "min_snr_gamma": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Min-SNR gamma weighting (0.0 = disabled, standard 1.0 weight across all timesteps). Recommended 0.0 for likeness/character LoRAs; values like 5.0 are optional for large artistic style datasets.",
                    },
                ),
                "include_conv_layers": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "LoCon / Conv-LoRA mode: When False (default), adapts Attention, FeedForward/MLP, and block/skip projections (including SpatialTransformer proj_in/proj_out). When True, also adapts ResNet/Downsample/Upsample Convolution layers inside repeating blocks (excluding raw input/output stems).",
                    },
                ),
                "control_mode": (
                    ["auto", "reference_latents", "concat_latent_image"],
                    {
                        "default": "auto",
                        "tooltip": "How paired control images are routed into the diffusion model. 'auto' introspects the loaded ComfyUI model (uses reference_latents for token-concatenated edit DiTs, and concat_latent_image for channel-concatenated / image_cond edit models).",
                    },
                ),
                "reference_latents_method": (
                    ["auto", "offset", "index", "index_timestep_zero", "uxo"],
                    {
                        "default": "auto",
                        "tooltip": "Optional positional-encoding / timestep method for reference_latents models ('offset', 'index', 'index_timestep_zero', 'uxo'). 'auto' defers to ComfyUI's native model default.",
                    },
                ),
                "control_dropout": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Probability of randomly dropping the control image per sample during training (0.0 = always conditioned; e.g. 0.05-0.10 teaches CFG / unconditional fallback).",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MODEL", "OMNILORA_NETWORK", "STRING")
    RETURN_NAMES = ("model", "lora", "training_summary")
    OUTPUT_NODE = True
    FUNCTION = "train_lora"
    CATEGORY = "OmniLoRA/Training"

    @staticmethod
    @contextlib.contextmanager
    def _scoped_comfy_training_flags(model: Optional[Any] = None) -> Iterator[Callable[[], None]]:
        """
        Toggle ComfyUI's global `mm.in_training` and `cli_args.args.disable_comfy_compiler`
        for the active training session and deterministically restore their exact prior
        values on both normal completion and any pre-training setup exception.
        """
        orig_in_training: Optional[bool] = None
        orig_disable_compiler: Optional[bool] = None
        has_mm = False
        restored = False

        try:
            import comfy.model_management as mm  # type: ignore

            has_mm = True
            orig_in_training = getattr(mm, "in_training", False)
            mm.in_training = True
        except Exception:
            pass
        try:
            import comfy.cli_args as cli_args  # type: ignore

            if hasattr(cli_args, "args"):
                orig_disable_compiler = getattr(cli_args.args, "disable_comfy_compiler", None)
                cli_args.args.disable_comfy_compiler = True
                active_infer_attn: List[str] = []
                if bool(getattr(cli_args.args, "use_ck_attention", False)):
                    active_infer_attn.append("--use-ck-attention (Comfy Kitchen)")
                if bool(getattr(cli_args.args, "use_sage_attention", False)):
                    active_infer_attn.append("--use-sage-attention (SageAttention)")
                if (
                    model is not None
                    and isinstance(getattr(model, "model_options", None), dict)
                    and isinstance(model.model_options.get("transformer_options"), dict)
                    and "optimized_attention_override" in model.model_options["transformer_options"]
                ):
                    active_infer_attn.append("model_options.optimized_attention_override")
                if active_infer_attn:
                    logging.info(
                        f"[OmniLoRA] Detected inference-only attention configuration ({', '.join(active_infer_attn)}); "
                        "routing training attention and RoPE/AdaLN kernels through differentiable PyTorch SDPA."
                    )
        except Exception:
            pass

        def _restore_flags() -> None:
            nonlocal restored
            if restored:
                return
            restored = True
            if has_mm:
                try:
                    import comfy.model_management as mm  # type: ignore

                    mm.in_training = bool(orig_in_training)
                except Exception:
                    pass
            if orig_disable_compiler is not None:
                try:
                    import comfy.cli_args as cli_args  # type: ignore

                    if hasattr(cli_args, "args"):
                        cli_args.args.disable_comfy_compiler = orig_disable_compiler
                except Exception:
                    pass

        try:
            yield _restore_flags
        finally:
            _restore_flags()

    def train_lora(
        self,
        cached_data: CachedConditioningBundle,
        model: Optional[Any] = None,
        model_architecture: str = "auto",
        backend: str = "auto",
        reserve_vram_gb: float = 0.0,
        optimizer: str = "radamw",
        rank: int = 32,
        alpha: float = 32.0,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "cosine",
        warmup_steps: int = 50,
        steps: int = 1000,
        grad_accum_steps: int = 1,
        gradient_checkpointing: bool = True,
        checkpoint_every_n_steps: int = 0,
        initial_lora: Optional[Any] = None,
        checkpoint_prefix: str = "omni_lora_ckpt",
        custom_checkpoint_dir: str = "",
        lora_dtype: str = "bf16",
        lora_plus_lr_ratio: float = 1.0,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 0.001,
        use_rslora: bool = False,
        seed: int = 42,
        enable_target_module_filter: bool = False,
        target_module_filter: str = "",
        min_timestep: float = 0.0,
        max_timestep: float = 1.0,
        min_snr_gamma: float = 0.0,
        include_conv_layers: bool = False,
        control_mode: str = "auto",
        reference_latents_method: str = "auto",
        control_dropout: float = 0.0,
        unique_id: Optional[str] = None,
        precision: str = "auto",
        vram_mode: str = "auto",
    ) -> Tuple[Any, OmniLoRANetworkBundle, str]:
        # CRITICAL: Exit ComfyUI's global torch.inference_mode() so autograd can build computation graphs
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            self._scoped_comfy_training_flags(model) as _restore_comfy_training_flags,
        ):
            torch.set_default_dtype(torch.float32)

            install_comfy_runtime_hooks()

            resumed_bundle: Optional[OmniLoRANetworkBundle] = None
            resolved_rank = int(rank)
            resolved_alpha = float(alpha)
            resolved_rslora = bool(use_rslora)
            prior_steps = 0

            # Automatically carry over trained LoRA if `model` comes from an upstream `OmniLoRATrainer`
            # (e.g. `Trainer 1 -> KSampler -> Trainer 2`), even when `initial_lora` is not wired separately.
            effective_initial_lora = initial_lora
            if effective_initial_lora is None and model is not None:
                effective_initial_lora = getattr(model, "_omni_lora_bundle", None)
                if (
                    effective_initial_lora is None
                    and hasattr(model, "model_options")
                    and isinstance(getattr(model, "model_options", None), dict)
                ):
                    effective_initial_lora = model.model_options.get("_omni_lora_bundle")

            if effective_initial_lora is not None:
                resumed_bundle = _coerce_lora_network_bundle(effective_initial_lora)
                inspected_resume = LoRAInjector.inspect_lora_state_dict(
                    resumed_bundle.state_dict,
                    metadata=resumed_bundle.metadata,
                )
                resolved_rank = int(inspected_resume.get("rank", resumed_bundle.rank))
                resolved_alpha = float(inspected_resume.get("alpha", resumed_bundle.alpha))
                if bool(inspected_resume.get("use_rslora", False)):
                    resolved_rslora = True
                prior_steps = int(resumed_bundle.steps_trained)
                source_via = "initial_lora input" if initial_lora is not None else "connected MODEL"
                logging.info(
                    f"[OmniLoRA] Resuming from LoRA '{resumed_bundle.source_name}' (via {source_via}): "
                    f"using rank={resolved_rank}, alpha={resolved_alpha:g}, rslora={resolved_rslora}, prior_steps={prior_steps}."
                )

            # Support legacy workflows passing a string in `reserve_vram_gb` slot
            resolved_reserve_gb = 0.0
            if isinstance(reserve_vram_gb, (int, float)):
                resolved_reserve_gb = max(0.0, float(reserve_vram_gb))
            elif isinstance(reserve_vram_gb, str):
                try:
                    resolved_reserve_gb = max(0.0, float(reserve_vram_gb))
                except ValueError:
                    vram_mode = reserve_vram_gb

            plan = detect_and_configure(
                preferred_backend=backend,
                device_index=0,
                requested_dtype=precision,
                vram_mode=vram_mode,
            )

            req_lora_dtype = (lora_dtype or "bf16").strip().lower()
            req_prec = (precision or "auto").strip().lower()
            if req_lora_dtype in ("fp32", "float32"):
                effective_lora_dtype = "fp32"
            elif req_lora_dtype in ("fp16", "float16"):
                effective_lora_dtype = "fp16"
            elif req_prec in ("fp32", "float32", "fp16", "float16"):
                effective_lora_dtype = "fp32" if "32" in req_prec else "fp16"
            else:
                effective_lora_dtype = "bf16" if plan.capabilities.supports_bf16 else plan.resolved_dtype

            # Unload any active ComfyUI LoadedModels first so their `.backup` weights are restored
            # into `model.model` before we unpatch or clone `clean_base_patcher`.
            _try_comfy_unload_models()

            # Preserve a clean, pre-OmniLoRA `ModelPatcher` clone before any training or patching occurs
            # so chained `Trainer 1 -> KSampler -> Trainer 2` runs never stack duplicate LoRA patches.
            clean_base_patcher = None
            if model is not None:
                # If `model` (or its shared `BaseModel`) still holds baked weight patches from a prior
                # KSampler run, unpatch it first so training starts from clean base weights.
                inner_pre = getattr(model, "model", None)
                has_active_patches = bool(
                    getattr(model, "backup", None)
                    or getattr(model, "object_patches_backup", None)
                    or (inner_pre is not None and getattr(inner_pre, "current_weight_patches_uuid", None) is not None)
                )
                if hasattr(model, "unpatch_model") and callable(model.unpatch_model) and has_active_patches:
                    try:
                        model.unpatch_model(device_to=None, unpatch_weights=True)
                    except TypeError:
                        try:
                            model.unpatch_model()
                        except Exception:
                            pass
                    except Exception:
                        pass

                clean_base_patcher = getattr(model, "_omni_clean_base_patcher", None)
                if (
                    clean_base_patcher is None
                    and hasattr(model, "model_options")
                    and isinstance(getattr(model, "model_options", None), dict)
                ):
                    clean_base_patcher = model.model_options.get("_omni_clean_base_patcher")
                if clean_base_patcher is None:
                    clean_base_patcher = (
                        model.clone() if hasattr(model, "clone") and callable(model.clone) else model
                    )

            # Unwrap ComfyUI ModelPatcher (`model.model.diffusion_model`) on CPU first (`load_to_gpu=False`)
            # so `pipeline.setup()` can enforce the VRAM headroom budget without spiking GPU memory.
            external_backbone = unwrap_comfy_diffusion_model(model, load_to_gpu=False)
            if external_backbone is not None:
                for p in external_backbone.parameters():
                    patches_list = getattr(p, "patches", None)
                    if isinstance(patches_list, list):
                        if not hasattr(p, "_omni_base_patches_len"):
                            try:
                                setattr(p, "_omni_base_patches_len", 0 if effective_initial_lora is not None else len(patches_list))
                            except Exception:
                                pass
                        base_len = int(getattr(p, "_omni_base_patches_len", 0))
                        if len(patches_list) > base_len:
                            del patches_list[base_len:]
            if model is not None:
                inner = getattr(model, "model", model)
                if isinstance(inner, nn.Module) and inner is not external_backbone:
                    sanitize_module_no_fp64(inner)

            adapter = resolve_model_adapter(
                model_id=model_architecture,
                comfy_model=model if model is not None else external_backbone,
                gradient_checkpointing=gradient_checkpointing,
                min_timestep=float(min_timestep),
                max_timestep=float(max_timestep),
                include_conv=bool(include_conv_layers),
                min_snr_gamma=float(min_snr_gamma),
            )

            target_modules = list(adapter.spec.default_target_modules)
            if external_backbone is not None and not target_modules:
                target_modules = list(
                    discover_universal_target_modules(
                        external_backbone,
                        comfy_inner_model=getattr(model, "model", None),
                        include_conv=bool(include_conv_layers),
                    )
                )

            _IGNORED_FILTER_SENTINELS = (
                "0",
                "0.0",
                "none",
                "(none)",
                "all",
                "*",
                "",
                "false",
                "off",
                "disable",
                "disabled",
            )
            filter_tokens = []
            if bool(enable_target_module_filter):
                raw_filter = str(target_module_filter or "").strip()
                if raw_filter.lower() not in _IGNORED_FILTER_SENTINELS:
                    filter_tokens = [
                        tok.strip()
                        for tok in raw_filter.split(",")
                        if tok.strip() and tok.strip().lower() not in _IGNORED_FILTER_SENTINELS
                    ]
            if filter_tokens:
                def _matches_filter(mod_path: str, tokens: List[str]) -> bool:
                    mod_low = mod_path.lower()
                    for tok in tokens:
                        tok_low = tok.lower()
                        if tok_low in mod_low:
                            return True
                        if tok_low in ("conv", "convolution", "convolutions", "resnet", "resblock", "resblocks"):
                            if (
                                any(seg in mod_low for seg in ("in_layers", "out_layers", "skip_connection", "conv"))
                                or ".op." in mod_low
                                or mod_low.endswith(".op")
                                or mod_low == "op"
                            ):
                                return True
                    return False

                def _is_raw_conv_stem(mod_path: str) -> bool:
                    low = mod_path.lower()
                    stripped = _strip_numeric_segments(low)
                    return (
                        low in ("out", "conv_in", "conv_out", "input_blocks.0.0", "out.2")
                        or stripped in ("input_blocks", "middle_block", "output_blocks", "down_blocks", "mid_block", "up_blocks", "out", "conv_in", "conv_out")
                    )

                filtered_targets = [
                    mod_p
                    for mod_p in target_modules
                    if _matches_filter(mod_p, filter_tokens)
                ]
                if not filtered_targets and external_backbone is not None:
                    for mod_name, mod_obj in external_backbone.named_modules():
                        if not mod_name:
                            continue
                        if _is_linear_like(mod_obj) and _matches_filter(mod_name, filter_tokens):
                            filtered_targets.append(mod_name)
                    if bool(include_conv_layers) or not filtered_targets:
                        for mod_name, mod_obj in external_backbone.named_modules():
                            if not mod_name or _is_raw_conv_stem(mod_name):
                                continue
                            if _is_conv_like(mod_obj) and _matches_filter(mod_name, filter_tokens):
                                filtered_targets.append(mod_name)
                if not filtered_targets:
                    avail_prefixes = sorted({m.split(".")[0] for m in target_modules if m})
                    raise ValueError(
                        f"[OmniLoRA] target_module_filter='{target_module_filter}' did not match any target modules "
                        f"in {adapter.spec.model_id}. Available top-level sub-networks: {avail_prefixes}"
                    )
                target_modules = filtered_targets
            elif resumed_bundle is not None:
                inspected_init = LoRAInjector.inspect_lora_state_dict(
                    resumed_bundle.state_dict,
                    metadata=resumed_bundle.metadata,
                )
                backbone_mods = dict(external_backbone.named_modules()) if external_backbone is not None else {}
                for mod_p in inspected_init.get("target_modules", []):
                    if not mod_p or mod_p in target_modules:
                        continue
                    mod_obj = backbone_mods.get(mod_p)
                    if mod_obj is not None and _is_conv_like(mod_obj):
                        # Only inherit a resumed conv target if its saved lora_down tensor has valid 3D/4D/5D kernel shape (or 1x1)
                        w_obj = getattr(mod_obj, "weight", None)
                        down_key_matches = [
                            t
                            for k, t in resumed_bundle.state_dict.items()
                            if isinstance(t, torch.Tensor)
                            and LoRAInjector._strip_lora_key_prefix_and_suffix(k) == (mod_p, "down")
                        ]
                        if (
                            isinstance(w_obj, torch.Tensor)
                            and w_obj.ndim >= 3
                            and w_obj.numel() != w_obj.shape[0] * w_obj.shape[1]
                            and down_key_matches
                            and down_key_matches[0].ndim == 2
                        ):
                            continue
                    target_modules.append(mod_p)

            lora_cfg = LoRAConfig(
                rank=resolved_rank,
                alpha=resolved_alpha,
                lora_dtype=effective_lora_dtype,
                target_modules=tuple(target_modules),
                use_rslora=resolved_rslora,
                include_conv=bool(include_conv_layers),
            )
            has_control_latents = bool(getattr(cached_data, "has_control_latents", False)) or any(
                isinstance(b.get("control_latents"), list) and any(c is not None for c in b["control_latents"])
                for b in cached_data.batches
            )
            adapter.control_mode = str(control_mode or "auto")
            adapter.reference_latents_method = str(reference_latents_method or "auto")
            resolved_control_mode = (
                adapter.resolve_control_mode(external_backbone, requested_mode=control_mode)
                if has_control_latents
                else "none"
            )
            dynamic_ctrl_drop = max(0.0, min(1.0, float(control_dropout or 0.0)))

            loras_dir = _resolve_comfy_loras_dir()
            ckpt_target_dir = _resolve_target_output_dir(custom_checkpoint_dir, loras_dir)
            train_cfg = TrainingConfig(
                output_dir=str(ckpt_target_dir),
                backend=plan.backend.backend_id,
                device_index=0,
                dtype=plan.resolved_dtype,
                vram_mode=plan.vram_state.value,
                reserve_vram_gb=resolved_reserve_gb,
                model_id=adapter.spec.model_id,
                resolution=cached_data.resolution,
                batch_size=cached_data.batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                optimizer_type=optimizer,
                learning_rate=float(learning_rate),
                lora_plus_lr_ratio=max(1.0, float(lora_plus_lr_ratio)),
                lr_scheduler=str(lr_scheduler or "cosine"),
                warmup_steps=max(0, int(warmup_steps)),
                min_lr_ratio=min(1.0, max(0.0, float(min_lr_ratio))),
                weight_decay=max(0.0, float(weight_decay)),
                max_train_steps=steps,
                seed=int(seed),
                gradient_checkpointing=gradient_checkpointing,
                has_control_latents=has_control_latents,
                lora=lora_cfg,
            )

            pipeline = LoRATrainingPipeline(
                backend=plan.backend,
                model_adapter=adapter,
                config=train_cfg,
            )

            lora_state: Dict[str, torch.Tensor] = {}
            setup_info: Dict[str, Any] = {}
            last_metrics = None
            saved_checkpoints: List[str] = []
            last_completed_step = 0
            filtered_resumed_state: Dict[str, torch.Tensor] = (
                dict(resumed_bundle.state_dict) if resumed_bundle is not None else {}
            )

            try:
                setup_info = pipeline.setup(
                    external_backbone=external_backbone,
                    initial_lora_state=resumed_bundle.state_dict if resumed_bundle is not None else None,
                )
                if resumed_bundle is not None and pipeline.backbone is not None:
                    backbone_mods = dict(pipeline.backbone.named_modules())
                    invalid_resumed_prefixes: set[str] = set()
                    for rk, rt in resumed_bundle.state_dict.items():
                        mod_p, comp = LoRAInjector._strip_lora_key_prefix_and_suffix(rk)
                        if comp == "down" and isinstance(rt, torch.Tensor) and rt.ndim == 2:
                            mod_obj = backbone_mods.get(mod_p)
                            w_obj = getattr(mod_obj, "weight", None) if mod_obj is not None else None
                            if (
                                isinstance(w_obj, torch.Tensor)
                                and w_obj.ndim >= 3
                                and w_obj.numel() != w_obj.shape[0] * w_obj.shape[1]
                            ):
                                invalid_resumed_prefixes.add(mod_p)
                    filtered_resumed_state = {
                        rk: rv
                        for rk, rv in resumed_bundle.state_dict.items()
                        if LoRAInjector._strip_lora_key_prefix_and_suffix(rk)[0] not in invalid_resumed_prefixes
                    }
                vram_budget = setup_info.get("vram_budget", {})

                ui_pbar: Optional[Any] = None
                try:
                    import comfy.utils  # type: ignore

                    ui_pbar = comfy.utils.ProgressBar(steps, node_id=unique_id)
                except Exception:
                    pass

                weight_format = str(
                    vram_budget.get("weight_format")
                    or plan.base_weight_quantization
                    or plan.resolved_dtype
                )
                steps_label = (
                    f"{steps} steps (cumulative {prior_steps + steps})"
                    if prior_steps > 0
                    else f"{steps} steps"
                )
                resident_mb = float(vram_budget.get("resident_model_mb", 0.0))
                offloaded_mb = float(vram_budget.get("offloaded_model_mb", 0.0))
                total_vram_mb = float(vram_budget.get("total_vram_mb", 0.0))
                opt_mb = float(vram_budget.get("optimizer_allowance_mb", 0.0))
                act_mb = float(vram_budget.get("activation_allowance_mb", 0.0))
                ckpt_in_mb = float(vram_budget.get("ckpt_inputs_mb", 0.0))
                res_gb = resident_mb / 1024.0
                stream_gb = offloaded_mb / 1024.0
                total_gb = total_vram_mb / 1024.0
                vram_placement = (
                    f"{res_gb:.2f} GB in VRAM + {stream_gb:.2f} GB CPU-streamed"
                    if stream_gb >= 0.05
                    else f"{res_gb:.2f} GB in VRAM (100% GPU resident)"
                )
                ledger_suffix = (
                    f" [ledger: opt={opt_mb:.0f}MB, act={act_mb:.0f}MB, ckpt_in={ckpt_in_mb:.0f}MB]"
                    if opt_mb > 0.0 or act_mb > 0.0
                    else ""
                )

                ckpt_blocks = int(setup_info.get("checkpointed_blocks", 0))
                ckpt_label = f"on({ckpt_blocks})" if ckpt_blocks > 0 else ("on" if setup_info.get("gradient_checkpointing") else "off")
                sched_label = str(setup_info.get("lr_scheduler", lr_scheduler))
                warmup_eff = int(setup_info.get("warmup_steps", 0))
                if warmup_eff > 0:
                    sched_label = f"{sched_label}(warmup={warmup_eff})"

                target_summary = setup_info.get("target_summary") or {}
                summary_lines = list(target_summary.get("summary_lines") or [])
                if summary_lines:
                    trainable_count = int(setup_info.get("trainable_parameters", 0))
                    total_count = int(setup_info.get("total_parameters", 0))
                    trainable_pct = float(setup_info.get("trainable_ratio_pct", 0.0))
                    if trainable_count > 0 and total_count > 0:
                        def _fmt_params(n: int) -> str:
                            if n >= 1_000_000_000:
                                return f"{n / 1e9:.2f}B"
                            if n >= 1_000_000:
                                return f"{n / 1e6:.2f}M"
                            if n >= 1_000:
                                return f"{n / 1e3:.1f}K"
                            return str(n)

                        total_adapted = int(target_summary.get("total", 0))
                        type_bits = []
                        if int(target_summary.get("linear", 0)):
                            type_bits.append(f"{int(target_summary['linear'])} Linear")
                        if int(target_summary.get("pointwise_conv", 0)):
                            type_bits.append(f"{int(target_summary['pointwise_conv'])} 1x1 Conv")
                        if int(target_summary.get("spatial_conv", 0)):
                            type_bits.append(f"{int(target_summary['spatial_conv'])} Conv")
                        type_desc = f" ({', '.join(type_bits)})" if type_bits else ""
                        header_line = (
                            f"Target modules: {total_adapted} adapted{type_desc} | "
                            f"{_fmt_params(trainable_count)}/{_fmt_params(total_count)} trainable ({trainable_pct:.2f}%):"
                        )
                    else:
                        header_line = summary_lines[0]
                    logging.info(f"[OmniLoRA] {header_line}")
                    for detail_line in summary_lines[1:]:
                        logging.info(f"[OmniLoRA] {detail_line}")

                ctrl_log_suffix = (
                    f" | control={resolved_control_mode}"
                    f"{f'({reference_latents_method})' if (resolved_control_mode == 'reference_latents' and str(reference_latents_method or 'auto').lower() != 'auto') else ''}"
                    f"{f' drop={dynamic_ctrl_drop:g}' if dynamic_ctrl_drop > 0.0 else ''}"
                    if has_control_latents
                    else ""
                )
                logging.info(
                    f"[OmniLoRA] Training {adapter.spec.model_id} on {plan.backend.backend_id.upper()} | "
                    f"{steps_label} | compute={plan.resolved_dtype}, weights={weight_format}, lora={effective_lora_dtype} | "
                    f"opt={optimizer.upper()} lr={learning_rate:g} sched={sched_label} rank={resolved_rank} alpha={resolved_alpha:g} accum={grad_accum_steps} ckpt={ckpt_label}{ctrl_log_suffix} | "
                    f"Model: {vram_placement}{ledger_suffix} ({total_gb:.2f} GB GPU)"
                )

                start_train_time = time.perf_counter()
                num_batches = max(1, len(cached_data.batches))
                accum_steps = max(1, grad_accum_steps)
                ema_step_sec: Optional[float] = None
                ema_loss: Optional[float] = None
                ckpt_interval = max(0, int(checkpoint_every_n_steps))
                clean_ckpt_prefix = (checkpoint_prefix or "omni_lora_ckpt").strip().strip("\"'") or "omni_lora_ckpt"
                step_width = len(str(steps))

                # Broadcast live training startup to frontend loss tracker
                prompt_server = None
                try:
                    import server  # type: ignore

                    if hasattr(server, "PromptServer") and server.PromptServer.instance is not None:
                        prompt_server = server.PromptServer.instance
                        register_omni_lora_routes(prompt_server)
                except Exception:
                    prompt_server = None

                run_timestamp = int(time.time())
                run_id = f"{clean_ckpt_prefix}_{run_timestamp}"
                run_title = f"{clean_ckpt_prefix} ({adapter.spec.model_id}, r{resolved_rank})"
                start_payload = {
                    "node_id": str(unique_id or ""),
                    "run_id": run_id,
                    "run_title": run_title,
                    "max_steps": steps,
                    "lr": float(learning_rate),
                    "optimizer": optimizer,
                    "model_id": adapter.spec.model_id,
                    "rank": resolved_rank,
                    "alpha": resolved_alpha,
                    "prior_steps": prior_steps,
                }
                TRACKER_STORE.start_run(start_payload)
                if prompt_server is not None:
                    try:
                        prompt_server.send_sync("omni_lora_training_start", start_payload)
                    except Exception:
                        pass

                # Per-pass batch shuffling (with boundary de-duplication so the first batch of a
                # new pass never immediately repeats the last batch of the previous pass).
                batch_rng = random.Random(int(train_cfg.seed) + int(prior_steps))
                batch_deck: List[int] = list(range(num_batches))
                if num_batches > 1:
                    batch_rng.shuffle(batch_deck)
                deck_pos = 0
                last_batch_idx: Optional[int] = None

                dynamic_flip_enabled = bool(getattr(cached_data, "random_flip", False))
                dynamic_cap_drop = max(0.0, min(1.0, float(getattr(cached_data, "caption_dropout", 0.0) or 0.0)))
                uncond_emb = getattr(cached_data, "uncond_prompt_embeds", None)
                uncond_pool = getattr(cached_data, "uncond_pooled_output", None)
                uncond_extra = getattr(cached_data, "uncond_extra_conds", None) or {}

                for step_idx in range(1, steps + 1):
                    _check_comfy_interrupt()
                    micro_batches: List[Dict[str, Any]] = []
                    for _ in range(accum_steps):
                        if deck_pos >= num_batches:
                            deck_pos = 0
                            if num_batches > 1:
                                batch_rng.shuffle(batch_deck)
                                if last_batch_idx is not None and batch_deck[0] == last_batch_idx:
                                    batch_deck[0], batch_deck[-1] = batch_deck[-1], batch_deck[0]
                        batch_idx = batch_deck[deck_pos]
                        deck_pos += 1
                        last_batch_idx = batch_idx
                        batch_dict = cached_data.batches[batch_idx]

                        latents_t = batch_dict["latents"]
                        latents_flip_t = batch_dict.get("latents_flipped")
                        mask_t = batch_dict.get("loss_mask")
                        mask_flip_t = batch_dict.get("loss_mask_flipped")
                        prompt_t = batch_dict["prompt_embeds"]
                        prompt_flip_t = batch_dict.get("prompt_embeds_flipped")
                        pooled_t = (
                            batch_dict["pooled_output"]
                            if "pooled_output" in batch_dict and isinstance(batch_dict["pooled_output"], torch.Tensor)
                            else None
                        )
                        pooled_flip_t = (
                            batch_dict["pooled_output_flipped"]
                            if "pooled_output_flipped" in batch_dict and isinstance(batch_dict["pooled_output_flipped"], torch.Tensor)
                            else None
                        )
                        extra_t = batch_dict.get("extra_conds")
                        extra_flip_t = batch_dict.get("extra_conds_flipped")
                        raw_ctrl_b = batch_dict.get("control_latents")
                        ctrl_list: Optional[List[Optional[torch.Tensor]]] = (
                            list(raw_ctrl_b) if isinstance(raw_ctrl_b, list) else None
                        )
                        raw_ctrl_flip_b = batch_dict.get("control_latents_flipped")
                        ctrl_flip_list: Optional[List[Optional[torch.Tensor]]] = (
                            list(raw_ctrl_flip_b) if isinstance(raw_ctrl_flip_b, list) else None
                        )

                        # 1. If a bucket batch contains multiple samples (batch_size > 1), shuffle
                        # the within-batch sample order on each draw (keeping targets, controls, masks, and prompts in lockstep).
                        bsz = int(latents_t.shape[0]) if isinstance(latents_t, torch.Tensor) and latents_t.ndim >= 1 else 1
                        perm_indices: Optional[List[int]] = None
                        if bsz > 1 and isinstance(prompt_t, torch.Tensor) and int(prompt_t.shape[0]) == bsz:
                            perm_indices = list(range(bsz))
                            batch_rng.shuffle(perm_indices)
                            latents_t = latents_t[perm_indices]
                            if isinstance(latents_flip_t, torch.Tensor) and int(latents_flip_t.shape[0]) == bsz:
                                latents_flip_t = latents_flip_t[perm_indices]
                            if isinstance(mask_t, torch.Tensor) and int(mask_t.shape[0]) == bsz:
                                mask_t = mask_t[perm_indices]
                            if isinstance(mask_flip_t, torch.Tensor) and int(mask_flip_t.shape[0]) == bsz:
                                mask_flip_t = mask_flip_t[perm_indices]
                            prompt_t = prompt_t[perm_indices]
                            if isinstance(prompt_flip_t, torch.Tensor) and int(prompt_flip_t.shape[0]) == bsz:
                                prompt_flip_t = prompt_flip_t[perm_indices]
                            if isinstance(pooled_t, torch.Tensor) and pooled_t.ndim >= 1 and int(pooled_t.shape[0]) == bsz:
                                pooled_t = pooled_t[perm_indices]
                            if isinstance(pooled_flip_t, torch.Tensor) and pooled_flip_t.ndim >= 1 and int(pooled_flip_t.shape[0]) == bsz:
                                pooled_flip_t = pooled_flip_t[perm_indices]
                            if isinstance(extra_t, dict):
                                perm_extra: Dict[str, Any] = {}
                                for ek, ev in extra_t.items():
                                    if isinstance(ev, torch.Tensor) and ev.ndim >= 1 and int(ev.shape[0]) == bsz:
                                        perm_extra[ek] = ev[perm_indices]
                                    else:
                                        perm_extra[ek] = ev
                                extra_t = perm_extra
                            if isinstance(extra_flip_t, dict):
                                perm_extra_f: Dict[str, Any] = {}
                                for ek, ev in extra_flip_t.items():
                                    if isinstance(ev, torch.Tensor) and ev.ndim >= 1 and int(ev.shape[0]) == bsz:
                                        perm_extra_f[ek] = ev[perm_indices]
                                    else:
                                        perm_extra_f[ek] = ev
                                extra_flip_t = perm_extra_f
                            if ctrl_list is not None and len(ctrl_list) == bsz:
                                ctrl_list = [ctrl_list[idx_p] for idx_p in perm_indices]
                            if ctrl_flip_list is not None and len(ctrl_flip_list) == bsz:
                                ctrl_flip_list = [ctrl_flip_list[idx_p] for idx_p in perm_indices]

                        # 2. Dynamic per-sample horizontal flip (`random_flip`) using pre-cached VAE flipped latents,
                        # synchronized flipped loss_mask, synchronized flipped control latents, and (for VLM CLIPs)
                        # synchronized flipped prompt embeddings.
                        if dynamic_flip_enabled and isinstance(latents_flip_t, torch.Tensor) and latents_flip_t.shape == latents_t.shape:
                            if bsz == 1:
                                flip_flags = [batch_rng.random() < 0.5]
                                if flip_flags[0]:
                                    latents_t = latents_flip_t
                                    if isinstance(mask_t, torch.Tensor):
                                        mask_t = (
                                            mask_flip_t
                                            if isinstance(mask_flip_t, torch.Tensor) and mask_flip_t.shape == mask_t.shape
                                            else torch.flip(mask_t, dims=[-1])
                                        )
                                    if ctrl_list is not None and ctrl_flip_list is not None and len(ctrl_flip_list) == 1:
                                        ctrl_list = list(ctrl_flip_list)
                                    if isinstance(prompt_flip_t, torch.Tensor):
                                        prompt_t = prompt_flip_t
                                        if isinstance(pooled_flip_t, torch.Tensor):
                                            pooled_t = pooled_flip_t
                                        if isinstance(extra_flip_t, dict):
                                            extra_t = extra_flip_t
                            else:
                                flip_flags = [batch_rng.random() < 0.5 for _ in range(bsz)]
                                if all(flip_flags):
                                    latents_t = latents_flip_t
                                    if isinstance(mask_t, torch.Tensor) and int(mask_t.shape[0]) == bsz:
                                        mask_t = (
                                            mask_flip_t
                                            if isinstance(mask_flip_t, torch.Tensor) and mask_flip_t.shape == mask_t.shape
                                            else torch.flip(mask_t, dims=[-1])
                                        )
                                    if ctrl_list is not None and ctrl_flip_list is not None and len(ctrl_flip_list) == bsz:
                                        ctrl_list = list(ctrl_flip_list)
                                    if isinstance(prompt_flip_t, torch.Tensor) and int(prompt_flip_t.shape[0]) == bsz:
                                        prompt_t = prompt_flip_t
                                        if isinstance(pooled_flip_t, torch.Tensor) and int(pooled_flip_t.shape[0]) == bsz:
                                            pooled_t = pooled_flip_t
                                        if isinstance(extra_flip_t, dict):
                                            extra_t = extra_flip_t
                                elif any(flip_flags):
                                    latents_t = torch.stack(
                                        [latents_flip_t[i] if flip_flags[i] else latents_t[i] for i in range(bsz)],
                                        dim=0,
                                    )
                                    if isinstance(mask_t, torch.Tensor) and int(mask_t.shape[0]) == bsz:
                                        if isinstance(mask_flip_t, torch.Tensor) and mask_flip_t.shape == mask_t.shape:
                                            mask_t = torch.stack(
                                                [mask_flip_t[i] if flip_flags[i] else mask_t[i] for i in range(bsz)],
                                                dim=0,
                                            )
                                        else:
                                            mask_t = torch.stack(
                                                [torch.flip(mask_t[i], dims=[-1]) if flip_flags[i] else mask_t[i] for i in range(bsz)],
                                                dim=0,
                                            )
                                    if ctrl_list is not None and ctrl_flip_list is not None and len(ctrl_flip_list) == bsz:
                                        ctrl_list = [
                                            ctrl_flip_list[i] if flip_flags[i] else ctrl_list[i]
                                            for i in range(bsz)
                                        ]
                                    if isinstance(prompt_flip_t, torch.Tensor) and prompt_flip_t.shape == prompt_t.shape:
                                        prompt_t = torch.stack(
                                            [prompt_flip_t[i] if flip_flags[i] else prompt_t[i] for i in range(bsz)],
                                            dim=0,
                                        )
                                        if (
                                            isinstance(pooled_t, torch.Tensor)
                                            and isinstance(pooled_flip_t, torch.Tensor)
                                            and pooled_flip_t.shape == pooled_t.shape
                                        ):
                                            pooled_t = torch.stack(
                                                [pooled_flip_t[i] if flip_flags[i] else pooled_t[i] for i in range(bsz)],
                                                dim=0,
                                            )
                                        if isinstance(extra_t, dict) and isinstance(extra_flip_t, dict):
                                            blended_extra: Dict[str, Any] = {}
                                            for ek, ev in extra_t.items():
                                                ev_f = extra_flip_t.get(ek)
                                                if (
                                                    isinstance(ev, torch.Tensor)
                                                    and isinstance(ev_f, torch.Tensor)
                                                    and ev_f.shape == ev.shape
                                                    and int(ev.shape[0]) == bsz
                                                ):
                                                    blended_extra[ek] = torch.stack(
                                                        [ev_f[i] if flip_flags[i] else ev[i] for i in range(bsz)],
                                                        dim=0,
                                                    )
                                                else:
                                                    blended_extra[ek] = ev
                                            extra_t = blended_extra

                        # 3. Dynamic per-sample CFG caption dropout (`caption_dropout`) using pre-cached unconditional prompt
                        if dynamic_cap_drop > 0.0 and isinstance(prompt_t, torch.Tensor):
                            drop_flags = [batch_rng.random() < dynamic_cap_drop for _ in range(bsz)]
                            if any(drop_flags):
                                u_single: Optional[torch.Tensor] = None
                                if isinstance(uncond_emb, torch.Tensor) and uncond_emb.shape[-1] == prompt_t.shape[-1]:
                                    u_single = uncond_emb[0:1] if uncond_emb.ndim == 3 else uncond_emb.unsqueeze(0)

                                if bsz == 1:
                                    # For single-sample batches (`bsz == 1`), preserve `u_single`'s exact unpadded
                                    # token sequence length so RoPE text_len and unmasked text layers match `""`.
                                    if u_single is not None:
                                        prompt_t = u_single.to(dtype=prompt_t.dtype, device=prompt_t.device)
                                    else:
                                        prompt_t = torch.zeros_like(prompt_t)
                                    if isinstance(pooled_t, torch.Tensor):
                                        if isinstance(uncond_pool, torch.Tensor) and uncond_pool.shape[-1] == pooled_t.shape[-1]:
                                            u_p = uncond_pool[0:1] if uncond_pool.ndim >= 2 else uncond_pool.unsqueeze(0)
                                            pooled_t = u_p.to(dtype=pooled_t.dtype, device=pooled_t.device)
                                        else:
                                            pooled_t = torch.zeros_like(pooled_t)
                                    if isinstance(extra_t, dict) and uncond_extra:
                                        merged_extra = dict(extra_t)
                                        for ek, ev in uncond_extra.items():
                                            merged_extra[ek] = ev
                                        extra_t = merged_extra
                                else:
                                    matched_u_emb: Optional[torch.Tensor] = None
                                    target_seq = int(prompt_t.shape[1])
                                    u_seq = int(u_single.shape[1]) if u_single is not None else 0
                                    if u_single is not None:
                                        if u_seq == target_seq:
                                            matched_u_emb = u_single.to(dtype=prompt_t.dtype, device=prompt_t.device)
                                        elif u_seq < target_seq:
                                            matched_u_emb = F.pad(u_single, (0, 0, 0, target_seq - u_seq)).to(
                                                dtype=prompt_t.dtype, device=prompt_t.device
                                            )
                                        else:
                                            matched_u_emb = u_single[:, :target_seq, :].to(
                                                dtype=prompt_t.dtype, device=prompt_t.device
                                            )
                                    prompt_t = prompt_t.clone()
                                    u_row = matched_u_emb[0] if matched_u_emb is not None else torch.zeros_like(prompt_t[0])
                                    for i, is_dropped in enumerate(drop_flags):
                                        if is_dropped:
                                            prompt_t[i] = u_row
                                    if isinstance(pooled_t, torch.Tensor) and int(pooled_t.shape[0]) == bsz:
                                        pooled_t = pooled_t.clone()
                                        if isinstance(uncond_pool, torch.Tensor) and uncond_pool.shape[-1] == pooled_t.shape[-1]:
                                            u_p_row = (uncond_pool[0] if uncond_pool.ndim >= 2 else uncond_pool).to(
                                                dtype=pooled_t.dtype, device=pooled_t.device
                                            )
                                        else:
                                            u_p_row = torch.zeros_like(pooled_t[0])
                                        for i, is_dropped in enumerate(drop_flags):
                                            if is_dropped:
                                                pooled_t[i] = u_p_row
                                    if isinstance(extra_t, dict):
                                        merged_extra = dict(extra_t)
                                        for ek, ev in extra_t.items():
                                            u_ev = uncond_extra.get(ek) if uncond_extra else None
                                            if (
                                                isinstance(ev, torch.Tensor)
                                                and ev.ndim >= 1
                                                and int(ev.shape[0]) == bsz
                                            ):
                                                ev_cloned = ev.clone()
                                                if isinstance(u_ev, torch.Tensor):
                                                    u_ev_row = u_ev[0] if u_ev.ndim == ev.ndim else u_ev
                                                    if u_ev_row.ndim == 1 and ev_cloned[0].ndim == 1 and int(u_ev_row.shape[0]) != int(ev_cloned[0].shape[0]):
                                                        t_len = int(ev_cloned[0].shape[0])
                                                        u_len = int(u_ev_row.shape[0])
                                                        pad_val = -1e4 if ek == "context_attention_bias" else 0
                                                        u_ev_row = (
                                                            F.pad(u_ev_row, (0, t_len - u_len), value=pad_val)
                                                            if u_len < t_len
                                                            else u_ev_row[:t_len]
                                                        )
                                                    if u_ev_row.shape == ev_cloned[0].shape:
                                                        u_ev_row = u_ev_row.to(dtype=ev_cloned.dtype, device=ev_cloned.device)
                                                        for i, is_dropped in enumerate(drop_flags):
                                                            if is_dropped:
                                                                ev_cloned[i] = u_ev_row
                                                        merged_extra[ek] = ev_cloned
                                                elif ek in ("attention_mask", "mask", "txt_mask", "context_attention_bias") and ev_cloned.ndim == 2 and u_seq > 0:
                                                    t_len = int(ev_cloned.shape[1])
                                                    valid_u = min(u_seq, t_len)
                                                    for i, is_dropped in enumerate(drop_flags):
                                                        if is_dropped:
                                                            if ek == "context_attention_bias":
                                                                ev_cloned[i, :valid_u] = 0.0
                                                                ev_cloned[i, valid_u:] = -1e4
                                                            else:
                                                                ev_cloned[i, :valid_u] = 1
                                                                ev_cloned[i, valid_u:] = 0
                                                    merged_extra[ek] = ev_cloned
                                        extra_t = merged_extra

                        # 4. Dynamic per-sample control dropout (`control_dropout`) for CFG / unconditional edit training
                        ctrl_drop_flags: Optional[List[bool]] = None
                        if ctrl_list is not None:
                            ctrl_drop_flags = [c is None for c in ctrl_list]
                            if dynamic_ctrl_drop > 0.0:
                                for i in range(len(ctrl_list)):
                                    if ctrl_list[i] is not None and batch_rng.random() < dynamic_ctrl_drop:
                                        ctrl_list[i] = None
                                        ctrl_drop_flags[i] = True

                        lat_scale = (
                            int(getattr(adapter.latent_format, "spacial_downscale_ratio", 8) or 8)
                            if getattr(adapter, "latent_format", None) is not None
                            else 8
                        )
                        cur_batch_bucket = batch_dict.get(
                            "bucket",
                            f"{int(latents_t.shape[-1]) * lat_scale}x{int(latents_t.shape[-2]) * lat_scale}" if latents_t.ndim >= 4 else "default"
                        )
                        item_batch: Dict[str, Any] = {
                            "latents": latents_t,
                            "prompt_embeds": prompt_t,
                            "latent_scaled": batch_dict.get("latent_scaled", cached_data.latent_scaled),
                            "bucket": cur_batch_bucket,
                        }
                        if isinstance(mask_t, torch.Tensor):
                            item_batch["loss_mask"] = mask_t
                        if pooled_t is not None:
                            item_batch["pooled_output"] = pooled_t
                        if extra_t:
                            item_batch["extra_conds"] = extra_t
                        if ctrl_list is not None and any(c is not None for c in ctrl_list):
                            item_batch["control_latents"] = ctrl_list
                            item_batch["control_mode"] = str(control_mode or "auto")
                            item_batch["reference_latents_method"] = str(reference_latents_method or "auto")
                        if ctrl_drop_flags is not None and any(ctrl_drop_flags):
                            item_batch["control_dropped"] = (
                                ctrl_drop_flags[0] if len(ctrl_drop_flags) == 1 else ctrl_drop_flags
                            )
                            item_batch["control_mode"] = str(control_mode or "auto")
                        micro_batches.append(item_batch)

                    last_metrics = pipeline.train_step(step_idx, micro_batches=micro_batches)
                    last_completed_step = step_idx
                    step_duration = float(last_metrics.step_time_ms) / 1000.0 if last_metrics else 0.0

                    # Exclude Step 1's one-time kernel compilation warm-up from the steady-state EMA speed
                    if step_idx == 1:
                        ema_step_sec = step_duration
                    elif step_idx == 2:
                        ema_step_sec = step_duration
                    else:
                        assert ema_step_sec is not None
                        ema_step_sec = 0.7 * ema_step_sec + 0.3 * step_duration

                    loss_val = float(last_metrics.loss) if last_metrics else 0.0
                    cur_lr = float(last_metrics.lr) if last_metrics else float(learning_rate)
                    ema_loss = loss_val if ema_loss is None else (0.85 * ema_loss + 0.15 * loss_val)
                    alloc_mb = float(last_metrics.allocated_vram_mb) if last_metrics else 0.0
                    res_mb = float(last_metrics.reserved_vram_mb) if last_metrics else 0.0

                    if ui_pbar is not None:
                        ui_pbar.update_absolute(step_idx, steps)

                    remaining_steps = steps - step_idx
                    eta_sec = int(round(remaining_steps * (ema_step_sec or step_duration)))
                    if eta_sec >= 3600:
                        eta_str = f"{eta_sec // 3600}h {(eta_sec % 3600) // 60:02d}m"
                    elif eta_sec >= 60:
                        eta_str = f"{eta_sec // 60:02d}m {eta_sec % 60:02d}s"
                    else:
                        eta_str = f"{eta_sec:02d}s"

                    active_gb = (alloc_mb if alloc_mb > 0.0 else resident_mb) / 1024.0
                    pool_gb = (max(alloc_mb, res_mb) if max(alloc_mb, res_mb) > 0.0 else resident_mb) / 1024.0
                    vram_str = (
                        f"{active_gb:.2f} / {total_gb:.2f} GB (pool {pool_gb:.2f} GB)"
                        if total_gb > 0
                        else f"{active_gb:.2f} GB (pool {pool_gb:.2f} GB)"
                    )
                    logging.info(
                        f"[OmniLoRA] Step {step_idx:>{step_width}}/{steps} ({step_idx / steps * 100:5.1f}%) | "
                        f"loss: {loss_val:.4f} (avg {ema_loss:.4f}) | "
                        f"lr: {cur_lr:.2e} | "
                        f"speed: {step_duration:.2f}s/step | "
                        f"ETA: {eta_str} | "
                        f"VRAM: {vram_str}"
                    )

                    # Persist step metrics in server store & stream to frontend in-canvas loss tracker
                    step_t = float(getattr(last_metrics, "timestep", 0.5)) if last_metrics else 0.5
                    step_bucket = str(micro_batches[0].get("bucket", "default") if micro_batches else "default")
                    elapsed_train_sec = max(0.0, time.perf_counter() - start_train_time)
                    step_payload = {
                        "node_id": str(unique_id or ""),
                        "run_id": run_id,
                        "run_title": run_title,
                        "step": step_idx,
                        "max_steps": steps,
                        "loss": round(loss_val, 6),
                        "ema_loss": round(ema_loss if ema_loss is not None else loss_val, 6),
                        "lr": cur_lr,
                        "timestep": round(step_t, 4),
                        "bucket": step_bucket,
                        "step_sec": round(step_duration, 3),
                        "rel_time_sec": round(elapsed_train_sec, 3),
                        "wall_time": round(time.time(), 3),
                        "eta_sec": eta_sec,
                        "vram_gb": round(active_gb, 2),
                    }
                    TRACKER_STORE.record_step(step_payload)
                    if prompt_server is not None:
                        try:
                            prompt_server.send_sync("omni_lora_step_metric", step_payload)
                        except Exception:
                            pass

                    manual_ckpt = TRACKER_STORE.consume_checkpoint_request(str(unique_id or ""))
                    if (
                        (ckpt_interval > 0 and step_idx % ckpt_interval == 0) or manual_ckpt
                    ) and pipeline.backbone is not None:
                        cum_step = prior_steps + step_idx
                        ckpt_state = dict(filtered_resumed_state)
                        ckpt_state.update(
                            LoRAInjector.extract_lora_state_dict(
                                pipeline.backbone,
                                comfy_model=model,
                            )
                        )
                        ckpt_meta = {
                            "modelspec.architecture": adapter.spec.model_id,
                            "modelspec.objective": adapter.spec.objective_type,
                            "omni_lora.backend": plan.backend.backend_id,
                            "omni_lora.dtype": plan.resolved_dtype,
                            "omni_lora.lora_dtype": effective_lora_dtype,
                            "omni_lora.optimizer": optimizer.lower(),
                            "omni_lora.lr_scheduler": str(setup_info.get("lr_scheduler", lr_scheduler)),
                            "omni_lora.rank": str(resolved_rank),
                            "omni_lora.alpha": str(resolved_alpha),
                            "omni_lora.use_rslora": str(resolved_rslora),
                            "omni_lora.include_conv_layers": str(bool(include_conv_layers)),
                            "omni_lora.trigger_word": cached_data.trigger_word,
                            "omni_lora.steps": str(cum_step),
                        }
                        ckpt_file_path = (
                            ckpt_target_dir / f"{clean_ckpt_prefix}_step_{cum_step:05d}.safetensors"
                        ).resolve()
                        ckpt_file_path.parent.mkdir(parents=True, exist_ok=True)
                        ckpt_path = _save_lora_safetensors_or_pt(
                            ckpt_state,
                            ckpt_file_path,
                            metadata=ckpt_meta,
                        )
                        saved_checkpoints.append(ckpt_path.name)
                        logging.info(f"[OmniLoRA] Auto-checkpoint saved at step {step_idx} -> {ckpt_path}")

                total_train_time = time.perf_counter() - start_train_time
                avg_step_time = total_train_time / max(1, steps)
                final_avg_loss = float(ema_loss) if ema_loss is not None else 0.0
                logging.info(
                    f"[OmniLoRA] Training completed in {total_train_time:.2f}s "
                    f"({steps} steps, avg {avg_step_time:.2f}s/step, avg loss {final_avg_loss:.4f})."
                )

                complete_payload = {
                    "node_id": str(unique_id or ""),
                    "run_id": run_id,
                    "run_title": run_title,
                    "final_step": steps,
                    "final_loss": round(float(last_metrics.loss) if last_metrics else 0.0, 6),
                    "final_ema_loss": round(final_avg_loss, 6),
                    "total_time_sec": round(total_train_time, 2),
                }
                TRACKER_STORE.complete_run(complete_payload)
                if prompt_server is not None:
                    try:
                        prompt_server.send_sync("omni_lora_training_complete", complete_payload)
                    except Exception:
                        pass

                assert pipeline.backbone is not None
                lora_state = dict(filtered_resumed_state)
                lora_state.update(
                    LoRAInjector.extract_lora_state_dict(
                        pipeline.backbone,
                        comfy_model=model,
                    )
                )
            except BaseException as exc:
                completed_step_idx = last_completed_step
                emergency_ckpt_path_str = ""
                if completed_step_idx > 0 and pipeline.backbone is not None:
                    try:
                        emerg_state: Dict[str, torch.Tensor] = dict(filtered_resumed_state)
                        emerg_state.update(
                            LoRAInjector.extract_lora_state_dict(
                                pipeline.backbone,
                                comfy_model=model,
                            )
                        )
                        if emerg_state:
                            emerg_cum_step = prior_steps + completed_step_idx
                            emerg_meta = {
                                "modelspec.architecture": adapter.spec.model_id,
                                "modelspec.objective": adapter.spec.objective_type,
                                "omni_lora.backend": plan.backend.backend_id,
                                "omni_lora.dtype": plan.resolved_dtype,
                                "omni_lora.lora_dtype": effective_lora_dtype,
                                "omni_lora.optimizer": optimizer.lower(),
                                "omni_lora.lr_scheduler": str(setup_info.get("lr_scheduler", lr_scheduler)),
                                "omni_lora.rank": str(resolved_rank),
                                "omni_lora.alpha": str(resolved_alpha),
                                "omni_lora.use_rslora": str(resolved_rslora),
                                "omni_lora.include_conv_layers": str(bool(include_conv_layers)),
                                "omni_lora.trigger_word": cached_data.trigger_word,
                                "omni_lora.steps": str(emerg_cum_step),
                                "omni_lora.interrupted": "true",
                            }
                            emerg_file = (
                                ckpt_target_dir / f"{clean_ckpt_prefix}_interrupted_step_{emerg_cum_step:05d}.safetensors"
                            ).resolve()
                            emerg_file.parent.mkdir(parents=True, exist_ok=True)
                            saved_emerg = _save_lora_safetensors_or_pt(
                                emerg_state,
                                emerg_file,
                                metadata=emerg_meta,
                            )
                            emergency_ckpt_path_str = str(saved_emerg)
                            logging.info(
                                f"[OmniLoRA] Emergency interrupt checkpoint saved at step {completed_step_idx} -> {saved_emerg}"
                            )
                    except Exception as emerg_exc:
                        logging.warning(f"[OmniLoRA] Could not write emergency checkpoint: {emerg_exc}")

                interrupt_payload = {
                    "node_id": str(unique_id or ""),
                    "run_id": locals().get("run_id", ""),
                    "step": completed_step_idx,
                    "reason": str(exc),
                    "emergency_checkpoint": emergency_ckpt_path_str,
                }
                TRACKER_STORE.interrupt_run(interrupt_payload)
                if prompt_server is not None:
                    try:
                        prompt_server.send_sync("omni_lora_training_interrupted", interrupt_payload)
                    except Exception:
                        pass
                logging.error(f"[OmniLoRA] Training interrupted or failed ({type(exc).__name__}: {exc})")
                raise
            finally:
                # CRITICAL: Synchronize the accelerator stream first, disable gradient checkpointing,
                # unwrap `LoRALinear` back to original `base_layer` in-place, offload all resident
                # weights back to `offload_device` (CPU), and empty the GPU caching allocator (even
                # on user Interrupt or OOM) so zero untracked tensors remain on the GPU.
                try:
                    plan.backend.synchronize()
                except Exception as sync_exc:
                    logging.debug(f"[OmniLoRA] Pre-cleanup backend synchronize warning: {sync_exc}")
                offload_dev = torch.device("cpu")
                if model is not None:
                    raw_off = getattr(model, "offload_device", None) or getattr(
                        getattr(model, "model", None), "offload_device", None
                    )
                    if isinstance(raw_off, torch.device):
                        offload_dev = raw_off
                    elif isinstance(raw_off, str):
                        try:
                            offload_dev = torch.device(raw_off)
                        except Exception:
                            offload_dev = torch.device("cpu")
                if pipeline.backbone is not None:
                    try:
                        pipeline.disable_gradient_checkpointing()
                        LoRAInjector.uninject(pipeline.backbone, offload_device=offload_dev)
                        pipeline.backbone.eval()
                    except Exception as uninj_exc:
                        logging.warning(f"[OmniLoRA] Warning during uninject cleanup: {uninj_exc}")
                if model is not None:
                    try:
                        inner_m = getattr(model, "model", None)
                        if inner_m is not None:
                            if hasattr(inner_m, "device"):
                                inner_m.device = offload_dev
                            if hasattr(inner_m, "model_loaded_weight_memory"):
                                inner_m.model_loaded_weight_memory = 0
                            if hasattr(inner_m, "model_offload_buffer_memory"):
                                inner_m.model_offload_buffer_memory = 0
                            if hasattr(inner_m, "current_weight_patches_uuid"):
                                inner_m.current_weight_patches_uuid = None
                            if hasattr(inner_m, "model_lowvram"):
                                inner_m.model_lowvram = False
                            if hasattr(inner_m, "lowvram_patch_counter"):
                                inner_m.lowvram_patch_counter = 0
                            if isinstance(inner_m, nn.Module):
                                for sub in inner_m.modules():
                                    if hasattr(sub, "comfy_patched_weights"):
                                        try:
                                            delattr(sub, "comfy_patched_weights")
                                        except Exception:
                                            setattr(sub, "comfy_patched_weights", False)
                                    if hasattr(sub, "_v_signature"):
                                        try:
                                            setattr(sub, "_v_signature", None)
                                        except Exception:
                                            pass
                                    comb = getattr(sub, "_combined_linear", None)
                                    if comb is not None and hasattr(comb, "_v_weight"):
                                        try:
                                            setattr(comb, "_v_weight", None)
                                        except Exception:
                                            pass
                        if hasattr(model, "model_loaded_weight_memory"):
                            model.model_loaded_weight_memory = 0
                    except Exception as reset_exc:
                        logging.warning(f"[OmniLoRA] Warning resetting ModelPatcher residency metadata: {reset_exc}")
                pipeline.optimizer = None
                pipeline.lr_scheduler = None
                pipeline.grad_scaler = None
                try:
                    pipeline.cleanup()
                except Exception as clean_exc:
                    logging.warning(f"[OmniLoRA] Warning during pipeline cleanup: {clean_exc}")
                release_streamed_weight_buffers()
                gc.collect()
                try:
                    plan.backend.synchronize()
                    plan.backend.empty_cache()
                except Exception as cache_exc:
                    logging.debug(f"[OmniLoRA] Post-cleanup cache release warning: {cache_exc}")
                _try_comfy_unload_models()
                _restore_comfy_training_flags()

            install_comfy_runtime_hooks()

            cumulative_steps = prior_steps + steps
            resolved_trigger = cached_data.trigger_word or (resumed_bundle.trigger_word if resumed_bundle else "")
            bundle_metadata: Dict[str, str] = {}
            if resumed_bundle is not None and resumed_bundle.metadata:
                bundle_metadata.update(resumed_bundle.metadata)
            bundle_metadata.update({
                "modelspec.architecture": adapter.spec.model_id,
                "modelspec.objective": adapter.spec.objective_type,
                "omni_lora.backend": plan.backend.backend_id,
                "omni_lora.dtype": plan.resolved_dtype,
                "omni_lora.lora_dtype": effective_lora_dtype,
                "omni_lora.optimizer": optimizer.lower(),
                "omni_lora.lr_scheduler": str(setup_info.get("lr_scheduler", lr_scheduler)),
                "omni_lora.rank": str(resolved_rank),
                "omni_lora.alpha": str(resolved_alpha),
                "omni_lora.use_rslora": str(resolved_rslora),
                "omni_lora.include_conv_layers": str(bool(include_conv_layers)),
                "omni_lora.trigger_word": resolved_trigger,
                "omni_lora.steps": str(cumulative_steps),
            })

            lora_bundle = OmniLoRANetworkBundle(
                state_dict=lora_state,
                rank=resolved_rank,
                alpha=resolved_alpha,
                model_architecture=adapter.spec.model_id,
                steps_trained=cumulative_steps,
                trigger_word=resolved_trigger,
                metadata=bundle_metadata,
                source_name=(
                    f"{resumed_bundle.source_name}_plus_{steps}steps"
                    if resumed_bundle is not None
                    else f"{adapter.spec.model_id}_omni_lora"
                ),
            )
            OmniLoRATrainer._LAST_TRAINED_BUNDLE = lora_bundle

            out_model = model
            if model is not None:
                source_patcher = clean_base_patcher if clean_base_patcher is not None else model
                base_patcher = (
                    source_patcher.clone()
                    if hasattr(source_patcher, "clone") and callable(source_patcher.clone)
                    else source_patcher
                )
                try:
                    import comfy.sd  # type: ignore

                    out_model, _ = comfy.sd.load_lora_for_models(base_patcher, None, lora_state, 1.0, 0.0)
                except (ImportError, ModuleNotFoundError):
                    out_model = base_patcher
                except Exception as patch_exc:
                    logging.warning(
                        f"[OmniLoRA] Failed to apply trained LoRA onto output MODEL via "
                        f"comfy.sd.load_lora_for_models ({type(patch_exc).__name__}: {patch_exc}); "
                        "returning base MODEL without attached LoRA patches (use the LORA output or Load LoRA node)."
                    )
                    out_model = base_patcher

                if out_model is not source_patcher:
                    try:
                        setattr(out_model, "_omni_lora_bundle", lora_bundle)
                        setattr(out_model, "_omni_clean_base_patcher", source_patcher)
                    except Exception:
                        pass
                    if hasattr(out_model, "model_options") and isinstance(getattr(out_model, "model_options", None), dict):
                        out_model.model_options["_omni_lora_bundle"] = lora_bundle
                        out_model.model_options["_omni_clean_base_patcher"] = source_patcher

            summary = {
                "resolved_model_architecture": adapter.spec.model_id,
                "training_objective": adapter.spec.objective_type,
                "detected_backend": plan.backend.backend_id,
                "device_name": plan.capabilities.device_name,
                "vram_state": plan.vram_state.value,
                "vram_budget": setup_info.get("vram_budget", {}),
                "resolved_dtype": plan.resolved_dtype,
                "lora_dtype": effective_lora_dtype,
                "base_weight_quantization": plan.base_weight_quantization,
                "optimizer": optimizer.lower(),
                "learning_rate": float(learning_rate),
                "lr_scheduler": setup_info.get("lr_scheduler", lr_scheduler),
                "warmup_steps": setup_info.get("warmup_steps", 0),
                "min_lr_ratio": setup_info.get("min_lr_ratio", min_lr_ratio),
                "lora_plus_lr_ratio": setup_info.get("lora_plus_lr_ratio", lora_plus_lr_ratio),
                "weight_decay": setup_info.get("weight_decay", weight_decay),
                "rank": resolved_rank,
                "alpha": resolved_alpha,
                "use_rslora": resolved_rslora,
                "include_conv_layers": bool(include_conv_layers),
                "seed": int(seed),
                "random_flip": dynamic_flip_enabled,
                "caption_dropout": dynamic_cap_drop,
                "has_control_latents": has_control_latents,
                "control_mode_resolved": resolved_control_mode,
                "reference_latents_method": str(reference_latents_method or "auto"),
                "control_dropout": dynamic_ctrl_drop,
                "clip_used_control_images": bool(getattr(cached_data, "clip_used_control_images", False)),
                "resumed_from_lora": resumed_bundle.source_name if resumed_bundle is not None else None,
                "resumed_lora_layers": setup_info.get("resumed_lora_layers", 0),
                "steps_this_session": steps,
                "cumulative_steps_trained": cumulative_steps,
                "grad_accum_steps": grad_accum_steps,
                "auto_checkpoints_saved": saved_checkpoints,
                "checkpoint_dir": str(ckpt_target_dir),
                "fp64_free_verified": True,
                "grad_scaler_active": bool(last_metrics and last_metrics.grad_scaler_active),
                "final_loss": last_metrics.loss if last_metrics else 0.0,
                "average_loss": round(float(ema_loss), 6) if ema_loss is not None else 0.0,
                "dataset_unique_images": cached_data.num_unique_images,
                "trigger_word": resolved_trigger,
                "target_module_filter": ",".join(filter_tokens),
                "min_timestep": float(getattr(adapter, "min_timestep", min_timestep)),
                "max_timestep": float(getattr(adapter, "max_timestep", max_timestep)),
                "injected_layers": setup_info.get("injected_layer_count", 0),
                "trainable_parameters": setup_info.get("trainable_parameters", 0),
            }
            return (out_model, lora_bundle, json.dumps(summary, indent=2))


class OmniLoRASave:
    """
    ComfyUI Node 5: Modular Save LoRA node (`OmniLoRA Save LoRA`).
    Takes an `OMNILORA_NETWORK` bundle from `OmniLoRATrainer` (or `OmniLoRALoader`), writes it
    as `.safetensors` to `ComfyUI/models/loras/` with full metadata, and passes the `lora` bundle
    through so it can be chained further if desired.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "lora": (
                    "OMNILORA_NETWORK",
                    {
                        "tooltip": "Trained or loaded LoRA network bundle from OmniLoRA Universal Trainer or OmniLoRA Load LoRA Network.",
                    },
                ),
                "output_lora_name": (
                    "STRING",
                    {
                        "default": "omni_lora_adapter",
                        "multiline": False,
                        "tooltip": "Filename (without extension) saved into ComfyUI/models/loras/.",
                    },
                ),
                "append_step_count": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, appends _step_<cumulative_steps> to the saved .safetensors filename.",
                    },
                ),
            },
            "optional": {
                "custom_output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional custom output folder (defaults to ComfyUI/models/loras/ when empty).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("OMNILORA_NETWORK", "STRING", "STRING")
    RETURN_NAMES = ("lora", "lora_filename", "saved_path")
    OUTPUT_NODE = True
    FUNCTION = "save_lora"
    CATEGORY = "OmniLoRA/Training"

    def save_lora(
        self,
        lora: Any,
        output_lora_name: str = "omni_lora_adapter",
        append_step_count: bool = False,
        custom_output_dir: str = "",
    ) -> Tuple[OmniLoRANetworkBundle, str, str]:
        with torch.inference_mode(False):
            bundle = _coerce_lora_network_bundle(lora)
            target_dir = _resolve_target_output_dir(custom_output_dir, _resolve_comfy_loras_dir())

            base_name = (output_lora_name or "").strip().strip("\"'") or bundle.source_name or "omni_lora_adapter"
            if base_name.lower().endswith(".safetensors"):
                base_name = base_name[: -len(".safetensors")]
            elif base_name.lower().endswith(".pt"):
                base_name = base_name[: -len(".pt")]

            if append_step_count and bundle.steps_trained > 0:
                final_stem = f"{base_name}_step_{bundle.steps_trained:05d}"
            else:
                final_stem = base_name

            save_meta = dict(bundle.metadata or {})
            save_meta.setdefault("modelspec.architecture", bundle.model_architecture)
            save_meta.setdefault("omni_lora.rank", str(bundle.rank))
            save_meta.setdefault("omni_lora.alpha", str(bundle.alpha))
            save_meta.setdefault("omni_lora.steps", str(bundle.steps_trained))
            if bundle.trigger_word:
                save_meta.setdefault("omni_lora.trigger_word", bundle.trigger_word)

            saved_path = _save_lora_safetensors_or_pt(
                bundle.state_dict,
                target_dir / f"{final_stem}.safetensors",
                metadata=save_meta,
            )

            logging.info(
                f"[OmniLoRA] Saved LoRA network '{saved_path.name}' "
                f"(rank={bundle.rank}, alpha={bundle.alpha}, steps={bundle.steps_trained}) -> {saved_path}"
            )
            return (bundle, saved_path.name, str(saved_path))


class OmniLoRALossTracker:
    """
    ComfyUI Node 6: Dedicated Full-Canvas Real-Time Loss & Telemetry Tracker (`OmniLoRA Loss Tracker`).
    - Mounts an interactive HTML5 canvas dashboard directly on the ComfyUI canvas.
    - Features real-time loss curves, dynamic EMA smoothing, multi-run overlays,
      noise bin filtering, aspect ratio bucketing, and one-click CSV/PNG export.
    - Freely resizable on canvas; optionally connect `training_summary` from
      OmniLoRATrainer to monitor and review training runs.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "training_summary": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional connection from OmniLoRATrainer's training_summary output to associate this tracker with that training node.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "track"
    CATEGORY = "OmniLoRA/Training"

    def track(
        self,
        training_summary: str = "",
        unique_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        snap = TRACKER_STORE.get_snapshot()
        return {
            "ui": {
                "summary": [training_summary],
                "runs": snap.get("runs", []),
            }
        }


NODE_CLASS_MAPPINGS = {
    "OmniLoRADatasetLoader": OmniLoRADatasetLoader,
    "OmniLoRAPrecomputeConditioning": OmniLoRAPrecomputeConditioning,
    "OmniLoRALoader": OmniLoRALoader,
    "OmniLoRATrainer": OmniLoRATrainer,
    "OmniLoRASave": OmniLoRASave,
    "OmniLoRALossTracker": OmniLoRALossTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniLoRADatasetLoader": "OmniLoRA Dataset & Prompt Loader",
    "OmniLoRAPrecomputeConditioning": "OmniLoRA VAE & CLIP Pre-Cache",
    "OmniLoRALoader": "OmniLoRA Load LoRA Network",
    "OmniLoRATrainer": "OmniLoRA Universal Trainer (All GPUs / XPU / CUDA / ROCm)",
    "OmniLoRASave": "OmniLoRA Save LoRA",
    "OmniLoRALossTracker": "OmniLoRA Loss Tracker",
}


