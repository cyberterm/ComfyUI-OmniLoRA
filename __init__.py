"""
ComfyUI-OmniLoRA: Universal Multi-Backend PyTorch LoRA Training Extension.

Drop into `ComfyUI/custom_nodes/omni_lora` or install via Comfy Registry (`comfy node install omni-lora`).
Queries ComfyUI's native `ModelPatcher`, `BaseModel`, `model_sampling`, `latent_format`, and `extra_conds`
directly so any current or future ComfyUI diffusion/flow model works automatically.
Supports Intel Arc XPU (`torch.xpu` in native `bf16` with strict FP64-free guards),
AMD ROCm, NVIDIA CUDA, Google Cloud TPU (`torch_xla`), Huawei Ascend NPU, Intel Gaudi HPU, Windows DirectML, Apple MPS, and CPU.
"""

from .models import install_comfy_runtime_hooks
from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    OmniLoRADatasetLoader,
    OmniLoRALoader,
    OmniLoRALossTracker,
    OmniLoRANetworkBundle,
    OmniLoRAPrecomputeConditioning,
    OmniLoRASave,
    OmniLoRATrainer,
)
from .tracker_store import register_omni_lora_routes

# Register non-intrusive ComfyUI LoraLoader cleanup hook
install_comfy_runtime_hooks()

# Register /omni_lora/metrics HTTP catch-up route on PromptServer (for idle/sleep recovery)
register_omni_lora_routes()

WEB_DIRECTORY = "./web"

__version__ = "1.0.0"

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

