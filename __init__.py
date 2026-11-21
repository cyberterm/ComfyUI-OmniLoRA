"""
ComfyUI-OmniLoRA: Universal Multi-Backend PyTorch LoRA Training Extension.

Drop into `ComfyUI/custom_nodes/omni_lora` or install via Comfy Registry (`comfy node install omni-lora`).
Queries ComfyUI's native `ModelPatcher`, `BaseModel`, `model_sampling`, `latent_format`, and `extra_conds`
directly so any current or future ComfyUI diffusion/flow model works automatically.
Supports Intel Arc XPU (`torch.xpu` in native `bf16` with strict FP64-free guards),
AMD ROCm, NVIDIA CUDA, Google Cloud TPU (`torch_xla`), Huawei Ascend NPU, Intel Gaudi HPU, Windows DirectML, Apple MPS, and CPU.
"""

import os as _os
from pathlib import Path as _Path

import torch as _torch

# PyTorch caches this flag on the first SDPA eligibility check, so it must be set at import time,
# before any attention runs. Without it, archs AOTriton lists as experimental fall back to math SDPA.
if getattr(_torch.version, "hip", None):
    _os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

_PKG_DIR = str(_Path(__file__).resolve().parent)
if "_PKG_DIR" in globals():
    if "__path__" not in globals() or not isinstance(globals().get("__path__"), list):
        __path__ = [_PKG_DIR]
    elif _PKG_DIR not in __path__:
        __path__.insert(0, _PKG_DIR)
    if globals().get("__spec__") is not None and getattr(__spec__, "submodule_search_locations", None) is None:
        __spec__.submodule_search_locations = list(__path__)

from .models import install_comfy_runtime_hooks
from .nodes import (
    OmniLoRADatasetLoader,
    OmniLoRALoader,
    OmniLoRALossTracker,
    OmniLoRANetworkBundle,
    OmniLoRAPrecomputeConditioning,
    OmniLoRASave,
    OmniLoRATrainer,
)
from .tracker_store import register_omni_lora_routes

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

# Register non-intrusive ComfyUI LoraLoader cleanup hook
install_comfy_runtime_hooks()

# Register /omni_lora/metrics HTTP catch-up route on PromptServer (for idle/sleep recovery)
register_omni_lora_routes()

WEB_DIRECTORY = "./web"

__version__ = "1.1.0"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "OmniLoRADatasetLoader",
    "OmniLoRALoader",
    "OmniLoRALossTracker",
    "OmniLoRANetworkBundle",
    "OmniLoRAPrecomputeConditioning",
    "OmniLoRASave",
    "OmniLoRATrainer",
    "WEB_DIRECTORY",
]

