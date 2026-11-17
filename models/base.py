"""
Abstract Model Adapter Layer.

Allows OmniLoRA to dynamically adapt to any generative image/video model
loaded in ComfyUI while keeping the hardware backend and LoRA trainer pipeline
completely model-agnostic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelSpec:
    """Metadata and default LoRA injection targets queried from a ComfyUI model."""

    model_id: str
    display_name: str
    architecture_type: str
    default_resolution: int
    latent_channels: int
    patch_size: int
    hidden_size: int
    double_blocks: int
    single_blocks: int
    default_target_modules: Tuple[str, ...]
    recommended_vram_gb_bf16: float
    objective_type: str = "rectified_flow"
    text_dim: int = 2048
    supports_video_5d: bool = False
    requires_pooled_conditioning: bool = False
    requires_guidance_embed: bool = False


class BaseModelAdapter(abc.ABC):
    """Base interface for generative model adapters in OmniLoRA."""

    last_timestep: float = 0.5

    @property
    @abc.abstractmethod
    def spec(self) -> ModelSpec:
        """Return architectural metadata and discovered LoRA targets."""

    @abc.abstractmethod
    def build_backbone(self, dtype: torch.dtype = torch.bfloat16, device: torch.device = torch.device("cpu")) -> nn.Module:
        """Return the ComfyUI diffusion_model backbone (or a lightweight fallback for offline tests)."""

    @abc.abstractmethod
    def compute_training_loss(
        self,
        backbone: nn.Module,
        batch: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Execute a single forward pass using ComfyUI's native model_sampling
        and extra_conds, returning the scalar loss tensor.
        """

    @abc.abstractmethod
    def create_synthetic_batch(
        self,
        batch_size: int,
        resolution: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """Create a deterministic synthetic batch for dry-run validation & XPU benchmarking."""
