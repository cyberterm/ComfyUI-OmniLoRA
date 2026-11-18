"""
ComfyUI-Native Universal Model Adapter for OmniLoRA (`omni_lora/models/adapters.py`).

Communicates directly with ComfyUI's runtime objects (`ModelPatcher`, `BaseModel`,
`model_sampling`, `latent_format`, `model_type`, and `extra_conds`) so that any
current or future model supported by ComfyUI works automatically without hardcoded
per-model configuration files.

Strictly 100% FP64-Free (`bf16` / `fp16` / `fp32` only) and zero per-step GPU-CPU
synchronizations for Intel Arc XPU and multi-vendor GPUs.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_comfy_importable() -> bool:
    """
    Ensure ComfyUI's `comfy` package (`comfy.model_base`, `comfy.supported_models`,
    `comfy.model_detection`, `comfy.lora`, `comfy.latent_formats`) is importable
    both when running inside `ComfyUI/custom_nodes/omni_lora` and when running
    standalone alongside a sibling `ComfyUI/` checkout.
    """
    if "comfy" in sys.modules:
        return True
    try:
        import comfy  # type: ignore # noqa: F401
        return True
    except Exception:
        pass

    here = Path(__file__).resolve()
    # Check standard custom_nodes hierarchy (`ComfyUI/custom_nodes/omni_lora/omni_lora/models/adapters.py`)
    # as well as sibling `ComfyUI/` checkouts (`<workspace>/ComfyUI`).
    for ancestor in here.parents:
        for candidate in (ancestor, ancestor / "ComfyUI"):
            comfy_pkg = candidate / "comfy"
            if comfy_pkg.is_dir() and (comfy_pkg / "model_base.py").is_file():
                cand_str = str(candidate)
                if cand_str not in sys.path:
                    sys.path.insert(0, cand_str)
                try:
                    import comfy  # type: ignore # noqa: F401
                    return True
                except Exception:
                    return False
    return False


from ..backends.base import (
    install_dml_fp64_scalar_guards,
    is_inference_tensor,
    safe_to_device,
    sanitize_module_no_fp64,
)
from ..lora.layers import (
    LoRAInjector,
    LoRALinear,
    _UNET_BLOCK_CONTAINER_NAMES,
    _collect_repeating_block_module_ids,
    _is_conv_like,
    _is_linear_like,
    _is_pointwise_conv,
    _strip_numeric_segments,
    classify_block_structural_roles,
    trace_single_block_dataflow,
)
from .base import BaseModelAdapter, ModelSpec


# ============================================================================
# 1. Lightweight Synthetic Fallback Backbone (Offline Unit Tests Only)
# ============================================================================

class ToyBackbone(nn.Module):
    """Minimal, FP64-free synthetic fallback backbone used only for offline dry-runs when no ComfyUI MODEL is connected."""

    def __init__(self, in_channels: int = 16, context_dim: int = 2048) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = 2
        self.hidden_size = 64
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.context_in = nn.Linear(context_dim, 64)
        self.attn_proj = nn.Linear(64, 64)
        self.conv_out = nn.Conv2d(64, in_channels, kernel_size=1)
        with torch.no_grad():
            for mod in (self.conv_in, self.context_in, self.attn_proj, self.conv_out):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)
            out_2d = self._forward_2d(x_2d, timestep, context)
            return out_2d.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return self._forward_2d(x, timestep, context)

    def _forward_2d(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.conv_in(x.contiguous())
        b, c, height, width = h.shape
        tokens = h.permute(0, 2, 3, 1).contiguous().reshape(b, height * width, c)
        if context is not None:
            ctx = self.context_in(context).mean(dim=1, keepdim=True)
            tokens = tokens + ctx
        tokens = tokens + self.attn_proj(tokens)
        out_h = tokens.reshape(b, height, width, c).permute(0, 3, 1, 2).contiguous()
        return self.conv_out(out_h)


# ============================================================================
# 2. Universal ComfyUI Native Model Adapter & Resilient Transformer Options
# ============================================================================

_SAMPLE_SIGMAS_MULT_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


def _is_omni_training_active() -> bool:
    """Return True when OmniLoRA autograd training is active (`grad` enabled or `mm.in_training`)."""
    if torch.is_grad_enabled():
        return True
    mm_mod = sys.modules.get("comfy.model_management")
    if mm_mod is not None and bool(getattr(mm_mod, "in_training", False)):
        return True
    return False


def _unwrap_attn_tensor(tensor: Any) -> Any:
    """Unwrap a ComfyUI `AttentionTensorContainer` without consuming raw tensors."""
    if isinstance(tensor, torch.Tensor):
        return tensor
    if hasattr(tensor, "take") and callable(tensor.take):
        return tensor.take()
    return tensor


def _normalize_sdpa_attn_mask(
    mask: Optional[torch.Tensor],
    q: Any,
    k: Any,
    skip_reshape: bool = True,
) -> Optional[torch.Tensor]:
    """
    Ensure `attn_mask` satisfies PyTorch `F.scaled_dot_product_attention` dtype and rank rules:
      1. Convert integer `0/1` token masks (`int64`/`int32`/`uint8`) to `torch.bool` (`0 -> False`,
         `!= 0 -> True`), since SDPA strictly rejects non-bool integer masks.
      2. Reshape batched 2D key masks `[B, L_k]` (`B > 1` and `B != L_q`) to `[B, 1, 1, L_k]`
         before ComfyUI's `attention_pytorch` unsqueezes 2D masks into `[1, 1, dim0, dim1]`.
    """
    if mask is None or not isinstance(mask, torch.Tensor):
        return mask
    if mask.dtype != torch.bool and not mask.is_floating_point():
        mask = mask.to(dtype=torch.bool)
    if mask.ndim == 2 and isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
        if skip_reshape and q.ndim == 4 and k.ndim == 4:
            b_sz, q_len, k_len = int(q.shape[0]), int(q.shape[-2]), int(k.shape[-2])
        elif not skip_reshape and q.ndim == 3 and k.ndim == 3:
            b_sz, q_len, k_len = int(q.shape[0]), int(q.shape[1]), int(k.shape[1])
        else:
            b_sz, q_len, k_len = 0, 0, 0
        if b_sz > 1 and int(mask.shape[0]) == b_sz and int(mask.shape[1]) == k_len and b_sz != q_len:
            mask = mask[:, None, None, :]
    return mask


def _run_pytorch_sdpa_attention(
    q: Any,
    k: Any,
    v: Any,
    heads: int,
    mask: Optional[torch.Tensor] = None,
    attn_precision: Any = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Execute differentiable PyTorch SDPA (`attention_pytorch` / `F.scaled_dot_product_attention`)
    using ComfyUI's exact `(q, k, v, heads, mask, skip_reshape, skip_output_reshape)` contract.
    Includes a guard for `skip_output_reshape=True` when `b > 32768` (`SDP_BATCH_LIMIT`).
    """
    q = _unwrap_attn_tensor(q)
    k = _unwrap_attn_tensor(k)
    v = _unwrap_attn_tensor(v)
    mask = _normalize_sdpa_attn_mask(mask, q, k, skip_reshape=skip_reshape)

    b_dim = int(q.shape[0]) if isinstance(q, torch.Tensor) and q.ndim >= 1 else 1
    comfy_attn = sys.modules.get("comfy.ldm.modules.attention")
    attn_pt = getattr(comfy_attn, "attention_pytorch", None) if comfy_attn is not None else None
    if callable(attn_pt) and not (skip_output_reshape and b_dim > 32768):
        raw_pt = getattr(attn_pt, "__wrapped__", None)
        if callable(raw_pt):
            return raw_pt(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        call_kw = dict(kwargs)
        call_kw.pop("preferred_attention", None)
        call_kw["_inside_attn_wrapper"] = True
        return attn_pt(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **call_kw,
        )

    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if skip_reshape:
        b, q_heads, _, dim_head = q.shape
    else:
        b = int(q.shape[0])
        dim_head = int(q.shape[-1]) // int(heads)
        q_heads = int(heads)
        k_heads = max(1, int(k.shape[-1]) // dim_head) if enable_gqa else q_heads
        v_heads = max(1, int(v.shape[-1]) // dim_head) if enable_gqa else q_heads
        q = q.reshape(b, -1, q_heads, dim_head).transpose(1, 2)
        k = k.reshape(b, -1, k_heads, dim_head).transpose(1, 2)
        v = v.reshape(b, -1, v_heads, dim_head).transpose(1, 2)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    if k.shape[1] != q.shape[1]:
        rep_k = max(1, int(q.shape[1]) // max(1, int(k.shape[1])))
        rep_v = max(1, int(q.shape[1]) // max(1, int(v.shape[1])))
        if mask is not None:
            k = k.repeat_interleave(rep_k, dim=1)
            v = v.repeat_interleave(rep_v, dim=1)
            enable_gqa = False
        else:
            enable_gqa = True
    else:
        enable_gqa = False

    sdpa_kwargs: Dict[str, Any] = {}
    if "scale" in kwargs and kwargs["scale"] is not None:
        sdpa_kwargs["scale"] = kwargs["scale"]
    if enable_gqa:
        sdpa_kwargs["enable_gqa"] = True

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        **sdpa_kwargs,
    )
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, int(heads) * dim_head)
    return out


def _omni_training_attention_override(orig_func: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
    """
    ComfyUI `transformer_options["optimized_attention_override"]` callback used during OmniLoRA
    training so `--use-ck-attention`, `--use-sage-attention`, and checkpoint `ComfyAttention`
    settings transparently execute via differentiable PyTorch SDPA.
    """
    return _run_pytorch_sdpa_attention(*args, **kwargs)


def _build_training_sample_sigmas(
    timesteps: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    num_steps: int = 20,
) -> torch.Tensor:
    """
    Construct a 1D descending schedule tensor `sample_sigmas` of length `num_steps + 1` (default 21)
    on `timesteps.device` with zero GPU-CPU syncs and zero `linspace`/`argmin`/`sort` kernel dependencies.

    Guarantees that `timesteps[0]` is an exact interior entry at index `i = 1` followed
    by strictly smaller sigmas (`0.90 * ref_t`, ...) down to `0.0`, so step-lookahead / flow-map models
    (such as Flux 3 Klein: `i = int((sample_sigmas - timestep[0]).abs().argmin()); target = sample_sigmas[min(i + 1, ...)]`)
    always resolve `i = 1` and `target = sample_sigmas[2] = 0.90 * timestep[0] < timestep[0]`.
    """
    device = timesteps.device
    ref_t = timesteps.detach().flatten()[0].to(dtype=torch.float32).clamp(min=1e-4)

    n = max(4, int(num_steps))
    cache_key = (str(device), n)
    multipliers = _SAMPLE_SIGMAS_MULT_CACHE.get(cache_key)
    if multipliers is None or multipliers.device != device:
        tail_len = n - 1
        mult_list = [1.25, 1.0] + [0.90 * (1.0 - float(k) / float(max(1, tail_len - 1))) for k in range(tail_len)]
        multipliers = safe_to_device(torch.tensor(mult_list, dtype=torch.float32), device=device, dtype=torch.float32)
        _SAMPLE_SIGMAS_MULT_CACHE[cache_key] = multipliers
    return ref_t * multipliers


class OmniTransformerOptions(dict):  # type: ignore[type-arg]
    """
    Copy-safe, resilient `transformer_options` dictionary passed to ComfyUI models during training.
    1. Pre-populates all standard ComfyUI sampler/patcher keys (`sample_sigmas`, `sigmas`,
       `cond_or_uncond`, `patches`, `patches_replace`, `wrappers`, `callbacks`).
    2. Preserves its class across `transformer_options = transformer_options.copy()` calls inside
       `comfy/ldm/*/model.py` and `WrapperExecutor`.
    3. Provides safe fallback defaults via `__missing__` if a new or custom model accesses an
       unanticipated `transformer_options["..."]` key via `[]` instead of `.get()`.
    """

    def copy(self) -> "OmniTransformerOptions":
        return OmniTransformerOptions(self)

    def __missing__(self, key: str) -> Any:
        k_lower = str(key).lower()
        if "sample_sigma" in k_lower:
            return self.get("sample_sigmas")
        if "sigma" in k_lower or "timestep" in k_lower:
            return self.get("sigmas", self.get("sample_sigmas"))
        if "cond_or_uncond" in k_lower:
            return [0]
        if any(w in k_lower for w in ("patch", "wrapper", "callback", "replace", "option", "dict")):
            val: Dict[str, Any] = {}
            self[key] = val
            return val
        if any(w in k_lower for w in ("list", "uuid", "indices", "adain")):
            lst: List[Any] = []
            self[key] = lst
            return lst
        if any(w in k_lower for w in ("scale", "strength", "multiplier", "factor")):
            return 1.0
        if any(w in k_lower for w in ("step", "index", "idx", "total")):
            return 0
        return None


class ComfyModelAdapter(BaseModelAdapter):
    """
    Universal ComfyUI Native Model Adapter.

    Queries latent geometry (`latent_format`), noise schedule (`model_sampling`),
    training objective (`model_type` / `model_sampling`), conditioning dispatch (`extra_conds`),
    and LoRA target layers directly from the connected ComfyUI `ModelPatcher` / `BaseModel`.
    """

    MODEL_ID = "comfy_model"

    def __init__(
        self,
        comfy_model: Optional[Any] = None,
        model_id: str = "auto",
        gradient_checkpointing: bool = True,
        guidance_scale: float = 3.5,
        min_timestep: float = 0.0,
        max_timestep: float = 1.0,
        include_conv: bool = False,
        **kwargs: Any,
    ) -> None:
        self.comfy_model = comfy_model
        self.inner_model = getattr(comfy_model, "model", comfy_model) if comfy_model is not None else None
        self.diffusion_model = (
            self._get_comfy_object("diffusion_model")
            or getattr(self.inner_model, "diffusion_model", None)
            or getattr(comfy_model, "diffusion_model", None)
            or (self.inner_model if isinstance(self.inner_model, nn.Module) else None)
        )

        self.model_config = (
            self._get_comfy_object("model_config")
            or getattr(self.inner_model, "model_config", None)
            or getattr(comfy_model, "model_config", None)
        )
        self.unet_config: Dict[str, Any] = (
            dict(getattr(self.model_config, "unet_config", None) or {})
            if self.model_config is not None
            else {}
        )
        self.model_sampling = (
            self._get_comfy_object("model_sampling")
            or getattr(self.inner_model, "model_sampling", None)
        )
        if self.model_sampling is not None and isinstance(self.model_sampling, nn.Module):
            sanitize_module_no_fp64(self.model_sampling)
        self.model_type = (
            self._get_comfy_object("model_type")
            or getattr(self.inner_model, "model_type", None)
        )
        self.latent_format = (
            self._get_comfy_object("latent_format")
            or getattr(self.inner_model, "latent_format", None)
            or (getattr(self.model_config, "latent_format", None) if self.model_config is not None else None)
        )
        self._gradient_checkpointing = gradient_checkpointing
        self.guidance_scale = float(guidance_scale)
        t_lo = max(0.0, min(1.0, float(min_timestep)))
        t_hi = max(0.0, min(1.0, float(max_timestep)))
        if t_hi <= t_lo:
            t_lo, t_hi = 0.0, 1.0
        self.min_timestep: float = t_lo
        self.max_timestep: float = t_hi
        raw_min_snr = kwargs.get("min_snr_gamma", 0.0)
        self.min_snr_gamma: float = float(0.0 if raw_min_snr is None else raw_min_snr)
        self.loss_type: str = str(kwargs.get("loss_type", "l2") or "l2").lower()
        self.huber_c: float = float(kwargs.get("huber_c", 0.1) or 0.1)
        self._cached_sigmas: Optional[torch.Tensor] = None
        self._cached_log_sigmas: Optional[torch.Tensor] = None
        self._cached_latent_channels: Dict[int, int] = {}
        self._cached_raw_input_channels: Dict[int, Optional[int]] = {}
        self._cached_context_dim: Dict[int, Optional[int]] = {}
        self._cached_vector_dim: Dict[int, Optional[int]] = {}
        self._cached_forward_sig: Dict[int, Tuple[set[str], bool]] = {}

        cls_name = ""
        if self.diffusion_model is not None:
            cls_name = self.diffusion_model.__class__.__name__
        elif self.inner_model is not None:
            cls_name = self.inner_model.__class__.__name__
        elif comfy_model is not None:
            cls_name = comfy_model.__class__.__name__

        native_info = resolve_comfy_native_architecture(
            comfy_model if comfy_model is not None else (self.inner_model or self.diffusion_model),
            unet_config=self.unet_config,
        )
        if self.model_config is None and native_info.get("model_config") is not None:
            self.model_config = native_info["model_config"]
        if not self.unet_config and isinstance(native_info.get("unet_config"), dict):
            self.unet_config = dict(native_info["unet_config"])
        if self.latent_format is None and native_info.get("latent_format") is not None:
            self.latent_format = native_info["latent_format"]
        if self.model_type is None and native_info.get("model_type") is not None:
            self.model_type = native_info["model_type"]
        if self.model_sampling is None and native_info.get("model_sampling") is not None:
            self.model_sampling = native_info["model_sampling"]
            if isinstance(self.model_sampling, nn.Module):
                sanitize_module_no_fp64(self.model_sampling)

        detected_arch = str(native_info.get("canonical_id") or "") or detect_comfy_model_architecture(
            self.diffusion_model or self.inner_model,
            unet_config=self.unet_config,
        )
        resolved_id = (
            model_id.strip().lower()
            if model_id and model_id.strip().lower() != "auto"
            else (detected_arch or cls_name.lower() or "comfy_model")
        )
        self._model_id = resolved_id

        # 1. Derive training objective from ComfyUI's ModelType and model_sampling MRO
        objective_type = self._resolve_objective_type(resolved_id)

        # 2. Derive latent channels & 2D/3D video dimensionality from ComfyUI's latent_format & diffusion_model
        latent_channels = 16
        supports_video_5d = False
        if self.latent_format is not None:
            latent_channels = int(getattr(self.latent_format, "latent_channels", 16))
            latent_dimensions = int(getattr(self.latent_format, "latent_dimensions", 2))
            supports_video_5d = latent_dimensions == 3
            if isinstance(self.diffusion_model, nn.Module):
                # Cross-check with backbone if a custom loader attached a generic 16-ch/3D latent_format
                # to a 64-ch/2D model (e.g., QwenImage21 loaded via a generic QwenImage wrapper)
                if not self._infer_backbone_accepts_5d(self.diffusion_model, default=supports_video_5d):
                    supports_video_5d = False
                inferred_c = self._infer_backbone_latent_channels(
                    self.diffusion_model,
                    fallback_channels=latent_channels,
                    ignore_latent_format=True,
                )
                if inferred_c != latent_channels and self._backbone_strictly_requires_channels(
                    self.diffusion_model, inferred_c
                ):
                    latent_channels = inferred_c
        elif self.diffusion_model is not None:
            latent_channels = self._infer_backbone_latent_channels(self.diffusion_model, fallback_channels=16)
            supports_video_5d = self._infer_backbone_accepts_5d(self.diffusion_model, default=False)

        # 3. Discover LoRA target modules from ComfyUI's live diffusion_model
        self.include_conv = bool(include_conv)
        default_targets: Tuple[str, ...] = ()
        if isinstance(self.diffusion_model, nn.Module):
            default_targets = discover_universal_target_modules(
                self.diffusion_model,
                comfy_inner_model=self.inner_model,
                include_conv=self.include_conv,
            )

        # 4. Discover expected text conditioning dimension from the backbone
        text_dim = 2048
        if isinstance(self.diffusion_model, nn.Module):
            inferred_ctx = self._infer_backbone_context_dim(self.diffusion_model)
            if inferred_ctx is not None:
                text_dim = inferred_ctx

        # 5. Check whether the model actually uses pooled/vector conditioning (`y`)
        # Note: In ComfyUI's UNetModel (SD1.5/SD2.1), `num_classes` is None and passing `y`
        # triggers `AssertionError: must specify y if and only if the model is class-conditional`.
        num_classes = getattr(self.diffusion_model, "num_classes", None)
        requires_pooled = bool(
            hasattr(self.diffusion_model, "vector_in")
            or hasattr(self.diffusion_model, "y_embedder")
            or hasattr(self.diffusion_model, "p_embedder")
            or (hasattr(self.diffusion_model, "label_emb") and num_classes is not None)
        )

        raw_patch_size = getattr(
            self.diffusion_model,
            "patch_size",
            getattr(
                self.diffusion_model,
                "patch_spatial",
                self.unet_config.get("patch_size", self.unet_config.get("patch_spatial", 2)),
            ),
        )
        resolved_patch_size = (
            int(raw_patch_size[0])
            if isinstance(raw_patch_size, (tuple, list)) and len(raw_patch_size) > 0
            else (int(raw_patch_size) if isinstance(raw_patch_size, int) else 2)
        )

        self._spec = ModelSpec(
            model_id=resolved_id,
            display_name=f"ComfyUI ({cls_name or resolved_id})",
            architecture_type=getattr(self.model_type, "name", objective_type),
            default_resolution=1024 if not supports_video_5d else 512,
            latent_channels=latent_channels,
            patch_size=resolved_patch_size,
            hidden_size=getattr(self.diffusion_model, "hidden_size", 256)
            if isinstance(getattr(self.diffusion_model, "hidden_size", None), int)
            else 256,
            double_blocks=2,
            single_blocks=2,
            default_target_modules=default_targets,
            recommended_vram_gb_bf16=12.0,
            objective_type=objective_type,
            text_dim=text_dim,
            supports_video_5d=supports_video_5d,
            requires_pooled_conditioning=requires_pooled,
            requires_guidance_embed=hasattr(self.diffusion_model, "guidance_in"),
        )

    def _get_comfy_object(self, name: str) -> Any:
        """Query `comfy_model.get_model_object(name)` safely if `comfy_model` is a ComfyUI `ModelPatcher`."""
        if self.comfy_model is not None and hasattr(self.comfy_model, "get_model_object"):
            try:
                obj = self.comfy_model.get_model_object(name)
                if obj is not None:
                    return obj
            except Exception:
                pass
        return None

    def _infer_backbone_accepts_5d(self, backbone: nn.Module, default: bool = False) -> bool:
        """
        Determine structurally whether `backbone.forward` expects 5D `[B, C, T, H, W]` video/3D-patched
        latents or 4D `[B, C, H, W]` image latents across any ComfyUI model:
          - If `patch_size` is a 3-tuple `(pt, ph, pw)` or the backbone contains `nn.Conv3d`, `patch_temporal` + `patch_spatial`
            (e.g. Cosmos Predict2 / Anima `MiniTrainDIT`), or a 5D `process_img` / `_forward` (e.g. `QwenImageTransformer2DModel`,
            `Krea2` `SingleStreamDiT`), it accepts 5D latents.
          - Models whose `_forward` and `process_img` unpack 4D `[B, C, H, W]` (e.g. `Flux`, `Flux2`, `Chroma`, `MageFlow`) or
            where `patch_size` is None or 1 with 2D `img_in` (`in_channels == out_channels`) expect 4D latents.
        """
        if hasattr(backbone, "patch_temporal") and hasattr(backbone, "patch_spatial"):
            return True
        p = getattr(backbone, "patch_size", self.unet_config.get("patch_size", None))
        if isinstance(p, (tuple, list)) and len(p) == 3:
            return True
        if any(isinstance(m, nn.Conv3d) for m in backbone.modules()):
            return True
        if hasattr(backbone, "process_img"):
            # Distinguish 5D process_img / _forward models (QwenImage: `bs, c, t, h, w = x.shape` in process_img;
            # Krea2 SingleStreamDiT: `temporal = x.ndim == 5; b5, c5, t5, h5, w5 = x.shape` in _forward)
            # from 4D image DiTs (Flux / Flux2 / Chroma / LongCatImage / MageFlow: `bs, c, h, w = x.shape`).
            if hasattr(backbone, "double_blocks") and hasattr(backbone, "single_blocks"):
                return False
            inner_fwd = getattr(backbone, "_forward", getattr(backbone, "forward", None))
            fwd_code = getattr(inner_fwd, "__code__", None)
            if fwd_code is not None:
                fwd_vars = getattr(fwd_code, "co_varnames", ())
                if any(v in fwd_vars for v in ("temporal", "t5", "b5")):
                    return True
            proc_fn = getattr(backbone, "process_img", None)
            proc_code = getattr(proc_fn, "__code__", None)
            if proc_code is not None:
                varnames = getattr(proc_code, "co_varnames", ())
                if "t" in varnames or "t_len" in varnames or "frames" in varnames:
                    return True
                if any(v in varnames for v in ("h_orig", "w_orig", "h", "w")):
                    return False
            cls_mod = f"{backbone.__class__.__module__}.{backbone.__class__.__name__}".lower()
            if any(k in cls_mod for k in ("qwen_image", "krea", "wan", "hunyuan_video", "ltx", "cosmos", "mochi", "cogvideo")):
                return True
            if any(k in cls_mod for k in ("flux", "chroma", "mage_flow", "longcat")):
                return False
            if self.latent_format is not None:
                return int(getattr(self.latent_format, "latent_dimensions", 2)) >= 3
            return True
        if p in (None, 1):
            img_in = getattr(backbone, "img_in", None)
            in_f = _get_linear_in_features(img_in)
            out_ch = getattr(backbone, "out_channels", self.unet_config.get("out_channels", None))
            if isinstance(in_f, int) and in_f > 0 and out_ch == in_f:
                return False
        return default

    def _backbone_strictly_requires_channels(self, backbone: nn.Module, inferred_c: int) -> bool:
        """
        Return True when `backbone` structurally consumes unpatched `inferred_c`-channel latents
        directly into `img_in` (with `patch_size == 1`, or `patch_size is None` and no internal `process_img` patch packing),
        or when `backbone` explicitly declares `latents_dim == inferred_c` (e.g. `MiniMaxH3Model` where `latents_dim=24`
        defines the video stream channel count while `MiniMaxH3AV.latent_channels=32` is the max across video/audio).
        """
        latents_dim = getattr(backbone, "latents_dim", self.unet_config.get("latents_dim", None))
        if isinstance(latents_dim, int) and latents_dim > 0 and latents_dim == inferred_c:
            return True
        p = getattr(backbone, "patch_size", self.unet_config.get("patch_size", None))
        if p == 1 or (p is None and not hasattr(backbone, "process_img")):
            img_in = getattr(backbone, "img_in", None)
            in_f = _get_linear_in_features(img_in)
            if isinstance(in_f, int) and in_f == inferred_c:
                return True
        return False

    def _resolve_objective_type(self, resolved_id: str) -> str:
        """Query ComfyUI's `model_type` and `model_sampling` class hierarchy to determine the training objective."""
        if self.model_type is not None:
            type_name = getattr(self.model_type, "name", str(self.model_type)).upper()
            if "FLOW" in type_name or "CONST" in type_name:
                return "rectified_flow"
            if "V_PREDICTION" in type_name:
                return "v_prediction"
            if "EDM" in type_name:
                return "edm"
            if "X0" in type_name:
                return "x0"
            if "EPS" in type_name or "CASCADE" in type_name:
                return "ddpm_eps"

        if self.model_sampling is not None:
            mro_names = [cls.__name__.upper() for cls in self.model_sampling.__class__.__mro__]
            if any("CONST" in n or "FLOW" in n for n in mro_names):
                return "rectified_flow"
            if any("V_PREDICTION" in n for n in mro_names):
                return "v_prediction"
            if any(n == "EDM" or "CONTINUOUSEDM" in n for n in mro_names):
                return "edm"
            if any(n == "X0" for n in mro_names):
                return "x0"
            if any("EPS" in n for n in mro_names):
                return "ddpm_eps"

        if "sd15" in resolved_id or "sdxl" in resolved_id or "kolors" in resolved_id or "hunyuan_dit" in resolved_id:
            return "ddpm_eps"
        return "rectified_flow"

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def build_backbone(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ) -> nn.Module:
        if self.diffusion_model is not None and isinstance(self.diffusion_model, nn.Module):
            return self.diffusion_model.to(device=device, dtype=dtype)
        return ToyBackbone(
            in_channels=self.spec.latent_channels,
            context_dim=self.spec.text_dim,
        ).to(device=device, dtype=dtype)

    def _infer_backbone_latent_channels(
        self,
        backbone: Optional[nn.Module] = None,
        fallback_channels: Optional[int] = None,
        ignore_latent_format: bool = False,
    ) -> int:
        """
        Inspect ComfyUI's `latent_format`, `model_config.unet_config`, and `diffusion_model`
        to determine expected input latent channels across all supported ComfyUI architectures.
        """
        if not ignore_latent_format and self.latent_format is not None:
            lf_ch = getattr(self.latent_format, "latent_channels", None)
            if isinstance(lf_ch, int) and lf_ch > 0:
                if backbone is None or not self._backbone_strictly_requires_channels(
                    backbone,
                    self._infer_backbone_latent_channels(backbone, fallback_channels=lf_ch, ignore_latent_format=True),
                ):
                    return lf_ch

        default_c = fallback_channels if (isinstance(fallback_channels, int) and fallback_channels > 0) else 16
        if backbone is None:
            return default_c

        bid = id(backbone)
        if bid in self._cached_latent_channels:
            return self._cached_latent_channels[bid]

        resolved_c: Optional[int] = None
        latents_dim = getattr(backbone, "latents_dim", self.unet_config.get("latents_dim", None))
        if isinstance(latents_dim, int) and latents_dim > 0:
            resolved_c = int(latents_dim)

        out_ch = getattr(
            backbone,
            "out_channels",
            getattr(backbone, "out_dim", self.unet_config.get("out_channels", self.unet_config.get("out_dim", None))),
        )
        in_ch = getattr(
            backbone,
            "in_channels",
            getattr(backbone, "in_dim", self.unet_config.get("in_channels", self.unet_config.get("in_dim", None))),
        )
        explicit_patch_size = getattr(backbone, "patch_size", self.unet_config.get("patch_size", None))

        # If a model explicitly declares `in_channels` and `out_channels` where `in_channels` is
        # patch-packed (e.g. QwenImage 1.0: in_channels=64, out_channels=16, patch_size=2),
        # `out_channels` is the unpacked VAE latent channel count.
        if (
            resolved_c is None
            and isinstance(out_ch, int)
            and out_ch in (3, 4, 8, 12, 16, 24, 32, 48, 64, 128)
            and isinstance(in_ch, int)
            and in_ch != out_ch
        ):
            resolved_c = int(out_ch)

        img_in = getattr(backbone, "img_in", None) or getattr(backbone, "patch_embed", None)
        if resolved_c is None and img_in is not None:
            if hasattr(img_in, "in_channels") and isinstance(img_in.in_channels, int) and img_in.in_channels > 0:
                resolved_c = int(img_in.in_channels)
            elif hasattr(img_in, "proj") and hasattr(img_in.proj, "in_channels"):
                resolved_c = int(img_in.proj.in_channels)
            else:
                in_f = _get_linear_in_features(img_in)
                if isinstance(in_f, int) and in_f > 0:
                    # Only divide by `patch_size^2` when the backbone explicitly defines `patch_size > 1`
                    # and does NOT set `out_channels == in_f` (e.g. QwenImage21 has no patch_size and
                    # consumes 64-channel latents directly, whereas Flux/Fibo2/QwenImage 1.0 have patch_size=2).
                    if explicit_patch_size is not None and (out_ch != in_f or in_f == 128):
                        p_area = (
                            int(explicit_patch_size[0]) * int(explicit_patch_size[1])
                            if isinstance(explicit_patch_size, (tuple, list)) and len(explicit_patch_size) >= 2
                            else (int(explicit_patch_size[-1]) ** 2 if isinstance(explicit_patch_size, (tuple, list)) else int(explicit_patch_size) ** 2)
                        )
                        if p_area > 1 and in_f % p_area == 0 and (in_f // p_area) in (3, 4, 8, 12, 16, 24, 32, 48, 64, 128):
                            resolved_c = in_f // p_area
                    if resolved_c is None and in_f in (3, 4, 8, 12, 16, 24, 32, 48, 64, 128):
                        resolved_c = in_f

        if resolved_c is None:
            emb_in = getattr(backbone, "emb_in", None)
            if isinstance(emb_in, nn.ModuleDict) and len(emb_in) > 0:
                first_emb = emb_in["image"] if "image" in emb_in else next(iter(emb_in.values()))
                in_f = _get_linear_in_features(first_emb)
                if isinstance(in_f, int) and in_f > 0:
                    p_area = 4 if (explicit_patch_size in (None, 2) and in_f in (64, 128)) else 1
                    resolved_c = in_f // p_area

        if resolved_c is None:
            x_embedder = getattr(backbone, "x_embedder", None)
            if x_embedder is not None and hasattr(x_embedder, "proj") and hasattr(x_embedder.proj, "in_channels"):
                resolved_c = int(x_embedder.proj.in_channels)

        if resolved_c is None and isinstance(in_ch, int) and in_ch > 0:
            p = explicit_patch_size if isinstance(explicit_patch_size, int) else 1
            if in_ch == 64 and p == 2 and out_ch != 64 and hasattr(backbone, "process_img"):
                resolved_c = 16
            elif in_ch == 128 and (hasattr(backbone, "emb_in") or p == 2):
                resolved_c = 32
            else:
                resolved_c = int(in_ch)

        final_c = resolved_c if resolved_c is not None else default_c
        self._cached_latent_channels[bid] = final_c
        return final_c

    def _infer_backbone_context_dim(self, backbone: nn.Module) -> Optional[int]:
        """Inspect a live ComfyUI `diffusion_model` (or `unet_config`) to find its expected text conditioning feature width."""
        bid = id(backbone)
        if bid in self._cached_context_dim:
            return self._cached_context_dim[bid]

        for norm_attr in ("llm_cond_norm", "context_norm", "txt_norm"):
            norm_mod = getattr(backbone, norm_attr, None)
            if norm_mod is not None:
                ns = getattr(norm_mod, "normalized_shape", None)
                if isinstance(ns, (tuple, list, torch.Size)) and len(ns) == 1 and int(ns[0]) > 0:
                    self._cached_context_dim[bid] = int(ns[0])
                    return int(ns[0])

        text_embedder = getattr(backbone, "text_embedder", None)
        if text_embedder is not None:
            proj = getattr(text_embedder, "proj", None)
            in_f = _get_linear_in_features(proj)
            if in_f is not None:
                self._cached_context_dim[bid] = in_f
                return in_f

        for attr in (
            "context_in",
            "context_embedder",
            "txt_embed",
            "cond_proj",
            "condition_proj",
            "text_proj",
            "caption_projection",
            "txt_in",
            "llm_proj",
        ):
            mod = getattr(backbone, attr, None)
            in_f = _get_linear_in_features(mod)
            if in_f is not None:
                self._cached_context_dim[bid] = in_f
                return in_f

        text_emb = getattr(backbone, "text_embedding", None)
        if isinstance(text_emb, nn.Sequential) and len(text_emb) > 0:
            in_f = _get_linear_in_features(text_emb[0])
            if in_f is not None:
                self._cached_context_dim[bid] = in_f
                return in_f

        for name, mod in backbone.named_modules():
            if any(
                name.endswith(suffix)
                for suffix in (
                    "attn2.to_k",
                    "cross_attn.k",
                    "cross_attn.k_proj",
                    "cross_attn.to_k",
                    "cross_attn.to_kv",
                    "txt_attn.qkv",
                    "context_block.attn.qkv",
                )
            ):
                in_f = _get_linear_in_features(mod)
                if in_f is not None:
                    self._cached_context_dim[bid] = in_f
                    return in_f

        for cfg_key in (
            "context_in_dim",
            "joint_attention_dim",
            "caption_channels",
            "cross_attention_dim",
            "context_dim",
            "text_dim",
            "text_in_dim",
            "llm_dim",
        ):
            cfg_val = getattr(backbone, cfg_key, self.unet_config.get(cfg_key))
            if isinstance(cfg_val, (tuple, list)) and cfg_val:
                non_zero = [int(v) for v in cfg_val if isinstance(v, int) and v > 0]
                if non_zero:
                    cfg_val = non_zero[-1]
            if isinstance(cfg_val, int) and cfg_val > 0:
                self._cached_context_dim[bid] = cfg_val
                return cfg_val

        self._cached_context_dim[bid] = None
        return None

    def _infer_backbone_vector_dim(self, backbone: nn.Module) -> Optional[int]:
        """
        Inspect a live ComfyUI `diffusion_model` (or `unet_config`) to find its expected pooled/ADM
        vector conditioning width (`y` / `pooled_projections`, e.g. 2816 for SDXL `label_emb`,
        768 for Flux `vector_in`, 2048 for SD3 `y_embedder`, or HiDream `p_embedder`).
        """
        bid = id(backbone)
        if bid in self._cached_vector_dim:
            return self._cached_vector_dim[bid]

        for cfg_key in ("adm_in_channels", "vec_in_dim", "pooled_projection_dim"):
            cfg_val = getattr(backbone, cfg_key, None)
            if isinstance(cfg_val, int) and cfg_val > 0:
                self._cached_vector_dim[bid] = cfg_val
                return cfg_val

        for attr in ("vector_in", "y_embedder", "p_embedder", "label_emb"):
            mod = getattr(backbone, attr, None)
            if mod is None:
                continue
            for sub in (
                mod,
                getattr(mod, "in_layer", None),
                getattr(mod, "mlp", None),
                getattr(mod, "linear_1", None),
            ):
                if sub is None:
                    continue
                in_f = _get_linear_in_features(sub)
                if in_f is not None:
                    self._cached_vector_dim[bid] = in_f
                    return in_f
                if isinstance(sub, (nn.Sequential, nn.ModuleList)) and len(sub) > 0:
                    first = sub[0]
                    in_f = _get_linear_in_features(first)
                    if in_f is not None:
                        self._cached_vector_dim[bid] = in_f
                        return in_f
                    if isinstance(first, (nn.Sequential, nn.ModuleList)) and len(first) > 0:
                        in_f = _get_linear_in_features(first[0])
                        if in_f is not None:
                            self._cached_vector_dim[bid] = in_f
                            return in_f

        for cfg_key in ("adm_in_channels", "vec_in_dim", "pooled_projection_dim"):
            cfg_val = self.unet_config.get(cfg_key)
            if isinstance(cfg_val, int) and cfg_val > 0:
                self._cached_vector_dim[bid] = cfg_val
                return cfg_val

        self._cached_vector_dim[bid] = None
        return None

    def _prepare_vector_embeds(
        self,
        y: torch.Tensor,
        backbone: nn.Module,
        spatial_h: int = 1024,
        spatial_w: int = 1024,
    ) -> torch.Tensor:
        """
        Ensure vector/ADM conditioning tensor `y` matches the input dimension of `backbone.label_emb`,
        `backbone.vector_in`, or `backbone.y_embedder`. When `backbone` is an SDXL-family UNet whose
        `label_emb` expects `pooled_output` + 256-dim sinusoidal `Timestep` ADM embeddings
        (`[height, width, crop_h, crop_w, target_height, target_width]` = 1536 dims) and only
        `pooled_output` (`[B, 1280]`) was provided, synthesizes exact sinusoidal ADM embeddings
        matching ComfyUI's `SDXL.encode_adm`.
        """
        expected_v = self._infer_backbone_vector_dim(backbone)
        if expected_v is None or y.ndim < 2:
            return y
        actual_v = int(y.shape[-1])
        if actual_v == expected_v:
            return y
        if expected_v > actual_v:
            diff = expected_v - actual_v
            if diff in (1536, 1280, 768, 512, 256) and diff % 256 == 0 and y.ndim == 2:
                try:
                    num_adm_tokens = diff // 256
                    adm_scalars = [
                        float(spatial_h),
                        float(spatial_w),
                        0.0,
                        0.0,
                        float(spatial_h),
                        float(spatial_w),
                    ][:num_adm_tokens]
                    half = 128
                    freqs = torch.exp(
                        -math.log(10000.0)
                        * torch.arange(start=0, end=half, dtype=torch.float32)
                        / float(half)
                    )
                    t_vals = torch.tensor(adm_scalars, dtype=torch.float32)[:, None] * freqs[None, :]
                    adm_1d = torch.cat([torch.cos(t_vals), torch.sin(t_vals)], dim=-1).reshape(1, diff)
                    adm_batch = safe_to_device(adm_1d, device=y.device, dtype=y.dtype).repeat(y.shape[0], 1)
                    return torch.cat([y, adm_batch], dim=-1)
                except Exception:
                    pass
            return F.pad(y, (0, diff))
        return y[..., :expected_v]

    def _infer_backbone_raw_input_channels(self, backbone: nn.Module) -> Optional[int]:
        """
        Determine the unpacked input channel count directly expected by the first layer of `backbone`
        so that Inpaint / I2V / Control / Edit models (`in_channels > latent_channels`, e.g. 9 vs 4
        for SDXL-Inpaint, 36 vs 16 for Wan-I2V, 32/96 vs 16 for Flux Control/Fill) can have their
        extra `c_concat` conditioning channels populated automatically.
        """
        bid = id(backbone)
        if bid in self._cached_raw_input_channels:
            return self._cached_raw_input_channels[bid]

        raw_c: Optional[int] = None
        explicit_patch_size = getattr(backbone, "patch_size", self.unet_config.get("patch_size", None))

        # 1. Check OpenAI / SD1.5 / SDXL UNet `input_blocks[0][0]` or `conv_in` or `patch_embedding`
        for conv_attr in ("conv_in", "patch_embedding"):
            conv_mod = getattr(backbone, conv_attr, None)
            if conv_mod is not None and isinstance(getattr(conv_mod, "in_channels", None), int):
                raw_c = int(conv_mod.in_channels)
                break

        if raw_c is None:
            input_blocks = getattr(backbone, "input_blocks", None)
            if isinstance(input_blocks, (nn.ModuleList, nn.Sequential)) and len(input_blocks) > 0:
                b0 = input_blocks[0]
                first_conv = b0[0] if isinstance(b0, (nn.ModuleList, nn.Sequential)) and len(b0) > 0 else b0
                if isinstance(getattr(first_conv, "in_channels", None), int):
                    raw_c = int(first_conv.in_channels)

        # 2. Check DiT / MMDiT `x_embedder`, `patch_embed`, `img_in`, `init_x_linear`, or `emb_in`
        if raw_c is None:
            for emb_attr in ("x_embedder", "patch_embed", "img_in", "init_x_linear", "patchify_proj"):
                emb_mod = getattr(backbone, emb_attr, None)
                if emb_mod is None:
                    continue
                if isinstance(getattr(emb_mod, "in_channels", None), int) and int(emb_mod.in_channels) > 0:
                    raw_c = int(emb_mod.in_channels)
                    break
                proj = getattr(emb_mod, "proj", None)
                if proj is not None and isinstance(getattr(proj, "in_channels", None), int) and int(proj.in_channels) > 0:
                    raw_c = int(proj.in_channels)
                    break
                in_f = _get_linear_in_features(emb_mod)
                if isinstance(in_f, int) and in_f > 0:
                    p_area = 1
                    if explicit_patch_size is not None:
                        p_area = (
                            int(explicit_patch_size[0]) * int(explicit_patch_size[1])
                            if isinstance(explicit_patch_size, (tuple, list)) and len(explicit_patch_size) >= 2
                            else (
                                int(explicit_patch_size[-1]) ** 2
                                if isinstance(explicit_patch_size, (tuple, list))
                                else int(explicit_patch_size) ** 2
                            )
                        )
                    if p_area > 1 and in_f % p_area == 0:
                        raw_c = in_f // p_area
                    else:
                        raw_c = in_f
                    break

        if raw_c is None:
            emb_in = getattr(backbone, "emb_in", None)
            if isinstance(emb_in, nn.ModuleDict) and len(emb_in) > 0:
                first_emb = emb_in["image"] if "image" in emb_in else next(iter(emb_in.values()))
                in_f = _get_linear_in_features(first_emb)
                if isinstance(in_f, int) and in_f > 0:
                    p_area = 4 if (explicit_patch_size in (None, 2) and in_f in (64, 128)) else 1
                    raw_c = in_f // p_area

        if raw_c is None:
            in_ch = getattr(backbone, "in_channels", self.unet_config.get("in_channels", None))
            if isinstance(in_ch, int) and in_ch > 0:
                p = explicit_patch_size if isinstance(explicit_patch_size, int) else 1
                if in_ch >= 64 and p == 2 and (hasattr(backbone, "process_img") or hasattr(backbone, "emb_in")) and in_ch % 4 == 0:
                    raw_c = in_ch // 4
                else:
                    raw_c = int(in_ch)

        if raw_c is None:
            latents_dim = getattr(backbone, "latents_dim", self.unet_config.get("latents_dim", None))
            if isinstance(latents_dim, int) and latents_dim > 0:
                raw_c = int(latents_dim)

        self._cached_raw_input_channels[bid] = raw_c
        return raw_c

    def _prepare_concat_input_channels(
        self,
        noisy_latents: torch.Tensor,
        backbone: nn.Module,
        c_concat: Optional[torch.Tensor] = None,
        clean_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply ComfyUI's `c_concat` channel concatenation and automatically pad any remaining
        conditioning channels required by Inpaint / I2V / Control / Edit models.
        """
        if isinstance(c_concat, torch.Tensor):
            cc = c_concat.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
            if cc.ndim == noisy_latents.ndim - 1 and noisy_latents.ndim == 5:
                cc = cc.unsqueeze(2).expand(-1, -1, noisy_latents.shape[2], -1, -1)
            if cc.shape[0] == 1 and noisy_latents.shape[0] > 1:
                cc = cc.repeat(noisy_latents.shape[0], *([1] * (cc.ndim - 1)))
            if cc.shape[2:] == noisy_latents.shape[2:]:
                noisy_latents = torch.cat([noisy_latents, cc], dim=1)

        raw_in_c = self._infer_backbone_raw_input_channels(backbone)
        actual_c = int(noisy_latents.shape[1])
        if raw_in_c is not None and raw_in_c > actual_c:
            extra_c = raw_in_c - actual_c
            if (
                isinstance(clean_latents, torch.Tensor)
                and clean_latents.shape == noisy_latents.shape
                and extra_c >= actual_c
            ):
                ref_cond = clean_latents.detach().to(device=noisy_latents.device, dtype=noisy_latents.dtype)
                rem_c = extra_c - actual_c
                if rem_c > 0:
                    pad_shape = list(noisy_latents.shape)
                    pad_shape[1] = rem_c
                    zeros = torch.zeros(pad_shape, device=noisy_latents.device, dtype=noisy_latents.dtype)
                    noisy_latents = torch.cat([noisy_latents, ref_cond, zeros], dim=1)
                else:
                    noisy_latents = torch.cat([noisy_latents, ref_cond], dim=1)
            else:
                pad_shape = list(noisy_latents.shape)
                pad_shape[1] = extra_c
                zeros = torch.zeros(pad_shape, device=noisy_latents.device, dtype=noisy_latents.dtype)
                noisy_latents = torch.cat([noisy_latents, zeros], dim=1)
        return noisy_latents

    @staticmethod
    def _adapt_latent_channels_spatial(
        latents: torch.Tensor,
        expected_c: int,
    ) -> Optional[torch.Tensor]:
        """
        Automatically adapt between unpacked (`[B, C, H, W]`, e.g. 16-ch @ 8x) and patch-packed
        (`[B, C * r^2, H // r, W // r]`, e.g. 64-ch @ 16x) latent formats using exact, invertible
        `F.pixel_unshuffle` / `F.pixel_shuffle` space-to-depth operations when `expected_c` and
        `actual_c` differ by an integer square patch factor `r^2` (`r in (2, 4)`).
        """
        actual_c = int(latents.shape[1])
        if actual_c == expected_c:
            return latents
        # Only adapt between modern 16+/32+ channel latents and their 2x2/4x4 patch-packed 64/128 channel counterparts;
        # never convert 4-channel SD1.5/SDXL latents into 16-channel Flux/MOSS latents.
        if min(actual_c, expected_c) < 16:
            return None

        is_5d = latents.ndim == 5
        if is_5d:
            b, c, t, h, w = latents.shape
            lat_4d = latents.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        elif latents.ndim == 4:
            lat_4d = latents
            b, c, t, h, w = latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3]
        else:
            return None

        adapted_4d: Optional[torch.Tensor] = None
        if expected_c > actual_c and expected_c % actual_c == 0:
            ratio = expected_c // actual_c
            r = int(round(ratio ** 0.5))
            if r in (2, 4) and r * r == ratio and h >= 1 and w >= 1:
                pad_h = (r - (h % r)) % r
                pad_w = (r - (w % r)) % r
                if pad_h > 0 or pad_w > 0:
                    lat_4d = F.pad(lat_4d, (0, pad_w, 0, pad_h), mode="replicate")
                adapted_4d = F.pixel_unshuffle(lat_4d, r)
        elif actual_c > expected_c and actual_c % expected_c == 0:
            ratio = actual_c // expected_c
            r = int(round(ratio ** 0.5))
            if r in (2, 4) and r * r == ratio:
                adapted_4d = F.pixel_shuffle(lat_4d, r)

        if adapted_4d is None:
            return None
        if is_5d:
            new_h, new_w = int(adapted_4d.shape[-2]), int(adapted_4d.shape[-1])
            return adapted_4d.reshape(b, t, expected_c, new_h, new_w).permute(0, 2, 1, 3, 4)
        return adapted_4d

    def _validate_and_prepare_latents(
        self,
        latents: torch.Tensor,
        backbone: nn.Module,
        latent_scaled: bool = True,
    ) -> torch.Tensor:
        """
        Apply ComfyUI's `process_latent_in` if not yet scaled during pre-caching, adjust 4D/5D
        video axis if needed, verify (or auto-adapt 2x2/4x4 space-to-depth) VAE latent channels,
        and align spatial dimensions `(H, W)` to the backbone's patchifier stride (`patch_size`).
        """
        scaled_ok = bool(latent_scaled)
        if not scaled_ok:
            process_in = getattr(self.inner_model, "process_latent_in", None) if self.inner_model is not None else None
            if callable(process_in):
                try:
                    latents = process_in(latents)
                    scaled_ok = True
                except Exception:
                    pass
            elif self.latent_format is not None and hasattr(self.latent_format, "process_in"):
                try:
                    latents = self.latent_format.process_in(latents)
                    scaled_ok = True
                except Exception:
                    pass

        expected_c = self._infer_backbone_latent_channels(backbone, fallback_channels=int(latents.shape[1]))
        actual_c = int(latents.shape[1])

        if actual_c != expected_c:
            adapted = self._adapt_latent_channels_spatial(latents, expected_c)
            if adapted is not None:
                latents = adapted
                actual_c = int(latents.shape[1])

        if not scaled_ok:
            process_in = getattr(self.inner_model, "process_latent_in", None) if self.inner_model is not None else None
            if callable(process_in):
                try:
                    latents = process_in(latents)
                except Exception:
                    pass
            elif self.latent_format is not None and hasattr(self.latent_format, "process_in"):
                try:
                    latents = self.latent_format.process_in(latents)
                except Exception:
                    pass

        accepts_5d = self._infer_backbone_accepts_5d(backbone, default=self.spec.supports_video_5d)
        if accepts_5d and latents.ndim == 4:
            latents = latents.unsqueeze(2)
        elif not accepts_5d and latents.ndim == 5:
            latents = latents[:, :, 0, :, :]

        if actual_c != expected_c:
            raise ValueError(
                f"VAE latent channel mismatch: connected VAE produced {actual_c}-channel latents "
                f"(shape {tuple(latents.shape)}), but the connected ComfyUI model "
                f"('{self.spec.display_name}') expects {expected_c}-channel latents. "
                f"Please connect the matching VAELoader for this model."
            )

        if latents.ndim >= 4:
            raw_ps = getattr(
                backbone,
                "patch_size",
                getattr(
                    backbone,
                    "patch_spatial",
                    self.unet_config.get("patch_size", self.unet_config.get("patch_spatial", self.spec.patch_size)),
                ),
            )
            if isinstance(raw_ps, (tuple, list)) and len(raw_ps) >= 2:
                stride_h = max(1, int(raw_ps[-2]))
                stride_w = max(1, int(raw_ps[-1]))
            elif isinstance(raw_ps, int) and raw_ps > 1:
                stride_h = stride_w = int(raw_ps)
            else:
                stride_h = stride_w = max(2, int(self.spec.patch_size or 2))
            if hasattr(backbone, "input_blocks") and hasattr(backbone, "output_blocks"):
                stride_h = max(stride_h, 4)
                stride_w = max(stride_w, 4)
            lat_h, lat_w = int(latents.shape[-2]), int(latents.shape[-1])
            pad_h = (stride_h - (lat_h % stride_h)) % stride_h
            pad_w = (stride_w - (lat_w % stride_w)) % stride_w
            if pad_h > 0 or pad_w > 0:
                latents = F.pad(latents, (0, pad_w, 0, pad_h))
        return latents

    def _prepare_prompt_embeds(self, prompt_embeds: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
        """
        Validate and automatically adapt the connected CLIP text encoder's embedding dimension
        against the diffusion model's text input projection or conditioning norm (handling both
        integer-multiple multi-layer widths like 4096 <-> 53248 and cross-encoder widths like 768 -> 4096).
        """
        expected_dim = self._infer_backbone_context_dim(backbone)
        if expected_dim is None or prompt_embeds.ndim < 2:
            return prompt_embeds
        return _adapt_context_width(prompt_embeds, expected_dim, model_name=self.spec.display_name)

    def _validate_context_dim(self, prompt_embeds: torch.Tensor, backbone: nn.Module) -> None:
        """
        Ensure the connected CLIP text encoder's embedding dimension can be adapted to the
        diffusion model's text input projection.
        """
        _ = self._prepare_prompt_embeds(prompt_embeds, backbone)

    def _get_forward_signature_info(self, backbone: nn.Module) -> Tuple[set[str], bool]:
        """
        Cache `inspect.signature` parameter metadata for `backbone.forward` and (when present)
        `backbone._forward` (used by `comfy.patcher_extension.WrapperExecutor` in Flux 3, Flux,
        Qwen, Wan, LTXV, etc.).
        """
        bid = id(backbone)
        if bid in self._cached_forward_sig:
            return self._cached_forward_sig[bid]
        param_names: set[str] = set()
        accepts_var_kwargs = True
        try:
            sig_fwd = inspect.signature(backbone.forward)
            param_names.update(sig_fwd.parameters.keys())
            fwd_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_fwd.parameters.values()
            )
            inner_fwd = getattr(backbone, "_forward", None)
            if callable(inner_fwd):
                sig_inner = inspect.signature(inner_fwd)
                param_names.update(sig_inner.parameters.keys())
                inner_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_inner.parameters.values()
                )
                accepts_var_kwargs = fwd_var_kw and inner_var_kw
            else:
                accepts_var_kwargs = fwd_var_kw
        except (ValueError, TypeError):
            param_names = set()
            accepts_var_kwargs = True
        param_names.discard("self")
        self._cached_forward_sig[bid] = (param_names, accepts_var_kwargs)
        return param_names, accepts_var_kwargs

    def _build_transformer_options(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        existing: Optional[Dict[str, Any]] = None,
    ) -> OmniTransformerOptions:
        """
        Construct a complete ComfyUI `transformer_options` dictionary populated with
        `sample_sigmas`, `sigmas`, `cond_or_uncond`, and patch/wrapper registries.
        """
        opts = OmniTransformerOptions()
        if self.comfy_model is not None:
            model_opts = getattr(self.comfy_model, "model_options", None)
            if isinstance(model_opts, dict):
                base_to = model_opts.get("transformer_options")
                if isinstance(base_to, dict):
                    opts.update(base_to)
        if isinstance(existing, dict):
            opts.update(existing)

        bsz = int(noisy_latents.shape[0]) if noisy_latents.ndim >= 1 else 1
        opts["sample_sigmas"] = _build_training_sample_sigmas(timesteps=timesteps, sigma=sigma)
        opts["sigmas"] = (
            sigma.detach().to(device=timesteps.device, dtype=timesteps.dtype)
            if isinstance(sigma, torch.Tensor)
            else timesteps.detach()
        )
        opts.setdefault("cond_or_uncond", [0] * bsz)
        opts.setdefault("patches", {})
        opts.setdefault("patches_replace", {})
        opts.setdefault("wrappers", {})
        opts.setdefault("callbacks", {})
        if _is_omni_training_active():
            # Route all attention calls carrying `transformer_options` through differentiable PyTorch SDPA
            # even if ComfyUI was launched with `--use-ck-attention` / `--use-sage-attention`, the
            # checkpoint embeds a `ComfyAttention` config, or a workflow node patched `model_options`.
            opts["optimized_attention_override"] = _omni_training_attention_override
        return opts

    def _forward_backbone(
        self,
        backbone: nn.Module,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_output: Optional[torch.Tensor] = None,
        extra_cond_dict: Optional[Dict[str, Any]] = None,
        sigma: Optional[torch.Tensor] = None,
        clean_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dispatch forward pass through ComfyUI's native `encode_adm`, `extra_conds`, `c_concat`,
        `process_timestep`, `transformer_options`, and `diffusion_model` without mutating inference tensors.
        """
        prompt_embeds = self._prepare_prompt_embeds(prompt_embeds, backbone)
        param_names, accepts_var_kwargs = self._get_forward_signature_info(backbone)

        kwargs: Dict[str, Any] = {}
        c_concat_tensor: Optional[torch.Tensor] = None
        bsz = int(noisy_latents.shape[0]) if noisy_latents.ndim >= 1 else 1

        spacial_ratio = (
            int(getattr(self.latent_format, "spacial_downscale_ratio", 8))
            if self.latent_format is not None
            else 8
        )
        spatial_h = int(noisy_latents.shape[-2]) * spacial_ratio if noisy_latents.ndim >= 4 else 1024
        spatial_w = int(noisy_latents.shape[-1]) * spacial_ratio if noisy_latents.ndim >= 4 else 1024

        # 1. Ask ComfyUI's native BaseModel.encode_adm(...) & BaseModel.extra_conds(...) to construct conditioning
        extra_conds_succeeded = False
        if self.inner_model is not None and hasattr(self.inner_model, "extra_conds"):
            try:
                expected_v = self._infer_backbone_vector_dim(backbone)
                cond_pooled = pooled_output
                if (
                    isinstance(cond_pooled, torch.Tensor)
                    and cond_pooled.ndim >= 2
                    and expected_v is not None
                    and expected_v > 1536
                    and hasattr(self.inner_model, "embedder")
                ):
                    expected_clip_pooled = expected_v - 1536
                    actual_clip_pooled = int(cond_pooled.shape[-1])
                    if actual_clip_pooled < expected_clip_pooled:
                        cond_pooled = F.pad(cond_pooled, (0, expected_clip_pooled - actual_clip_pooled))
                    elif actual_clip_pooled > expected_clip_pooled and actual_clip_pooled != expected_v:
                        cond_pooled = cond_pooled[..., :expected_clip_pooled]

                cond_in: Dict[str, Any] = {
                    "cross_attn": prompt_embeds,
                    "c_crossattn": prompt_embeds,
                    "pooled_output": cond_pooled,
                    "noise": noisy_latents,
                    "latent_image": clean_latents if isinstance(clean_latents, torch.Tensor) else noisy_latents,
                    "device": noisy_latents.device,
                    "width": spatial_w,
                    "height": spatial_h,
                    "crop_w": 0,
                    "crop_h": 0,
                    "target_width": spatial_w,
                    "target_height": spatial_h,
                    "aesthetic_score": 6.0,
                    "fps_id": 6,
                    "motion_bucket_id": 127,
                    "augmentation_level": 0.0,
                    "noise_aug_strength": 0.0,
                    "conditioning_scale": 1.0,
                    "seed": 0,
                    "prompt_type": "positive",
                }
                if self.spec.requires_guidance_embed or hasattr(backbone, "guidance_in"):
                    cond_in["guidance"] = self.guidance_scale
                if extra_cond_dict:
                    for ek, ev in extra_cond_dict.items():
                        if ev is None:
                            continue
                        if ek not in cond_in or cond_in[ek] is None:
                            ev_for_cond = ev
                            # `comfy.model_base.Anima.extra_conds` assumes `t5xxl_ids` and `t5xxl_weights`
                            # are 1D `(L_t5,)` tensors and unconditionally calls `.unsqueeze(0)` on them.
                            if ek in ("t5xxl_ids", "t5xxl_weights") and isinstance(ev, torch.Tensor) and ev.ndim >= 2:
                                ev_for_cond = ev[0] if int(ev.shape[0]) >= 1 else ev.squeeze(0)
                            cond_in[ek] = (
                                safe_to_device(ev_for_cond, device=noisy_latents.device, dtype=noisy_latents.dtype)
                                if isinstance(ev_for_cond, torch.Tensor) and torch.is_floating_point(ev_for_cond)
                                else (
                                    safe_to_device(ev_for_cond, device=noisy_latents.device)
                                    if isinstance(ev_for_cond, torch.Tensor)
                                    else ev_for_cond
                                )
                            )

                if (
                    "adm_encoded" not in cond_in
                    and hasattr(self.inner_model, "encode_adm")
                    and callable(self.inner_model.encode_adm)
                ):
                    try:
                        adm = self.inner_model.encode_adm(**cond_in)
                        if isinstance(adm, torch.Tensor):
                            cond_in["adm_encoded"] = safe_to_device(
                                adm,
                                device=noisy_latents.device,
                                dtype=noisy_latents.dtype,
                            )
                    except Exception:
                        pass

                built_conds = self.inner_model.extra_conds(**cond_in)
                extra_conds_succeeded = isinstance(built_conds, dict)
                for k, v in built_conds.items():
                    val = None
                    if hasattr(v, "process_cond") and callable(v.process_cond):
                        try:
                            processed = v.process_cond(batch_size=bsz, device=noisy_latents.device)
                            val = getattr(processed, "cond", processed)
                        except Exception:
                            val = getattr(v, "cond", v)
                    else:
                        val = getattr(v, "cond", v)

                    if isinstance(val, torch.Tensor):
                        val = val.clone() if (torch.is_grad_enabled() and val.requires_grad) else val.detach().clone()
                        if torch.is_floating_point(val):
                            val = safe_to_device(val, device=noisy_latents.device, dtype=noisy_latents.dtype)
                        else:
                            val = safe_to_device(val, device=noisy_latents.device)
                        if val.ndim >= 2 and val.shape[0] == 1 and bsz > 1 and k != "sample_sigmas":
                            val = val.repeat(bsz, *([1] * (val.ndim - 1)))

                    if k == "c_crossattn":
                        if "context" in param_names or accepts_var_kwargs:
                            kwargs["context"] = val
                    elif k == "c_concat" and isinstance(val, torch.Tensor) and "c_concat" not in param_names:
                        c_concat_tensor = val
                    elif k in param_names or accepts_var_kwargs:
                        kwargs[k] = val
            except Exception:
                pass

        if c_concat_tensor is None and extra_cond_dict and isinstance(extra_cond_dict.get("c_concat"), torch.Tensor):
            c_concat_tensor = extra_cond_dict["c_concat"]

        if extra_cond_dict:
            # When `BaseModel.extra_conds` succeeded, it has already authoritatively filtered, renamed,
            # or upscaled conditioning keys for `backbone.forward` (e.g., `Flux.extra_conds` intentionally
            # drops text-only `int64` `attention_mask` when `attention_mask_img_shape` is None because
            # `Flux._forward` passes `kwargs.get("attention_mask")` verbatim to joint SDPA; `HunyuanDiT`
            # maps `attention_mask` -> `text_embedding_mask`; `FluxSchnell` strips `guidance`).
            # Only fall back to forwarding raw `extra_cond_dict` keys when `extra_conds` did not run,
            # and never leak mask/bias keys blindly through `**kwargs` if not explicitly declared.
            if not extra_conds_succeeded:
                _explicit_only_mask_keys = (
                    "attention_mask",
                    "attention_mask_img_shape",
                    "attn_mask",
                    "mask",
                    "txt_mask",
                    "context_attention_bias",
                )
                for ek, ev in extra_cond_dict.items():
                    if ev is None or ek in kwargs or ek in ("c_concat", "cross_attn", "c_crossattn", "pooled_output"):
                        continue
                    if ek in param_names or (accepts_var_kwargs and ek not in _explicit_only_mask_keys):
                        kwargs[ek] = (
                            safe_to_device(ev, device=noisy_latents.device, dtype=noisy_latents.dtype)
                            if isinstance(ev, torch.Tensor) and torch.is_floating_point(ev)
                            else (safe_to_device(ev, device=noisy_latents.device) if isinstance(ev, torch.Tensor) else ev)
                        )

            # Ensure Anima's `t5xxl_ids` (`[B, L_t5]` int) and `t5xxl_weights` (`[B, L_t5, 1]` float)
            # preserve exact per-sample batch contents and tensor ranks when passed to `Anima.forward`.
            raw_t5_ids = extra_cond_dict.get("t5xxl_ids")
            if isinstance(raw_t5_ids, torch.Tensor) and (
                "t5xxl_ids" in kwargs
                or "t5xxl_ids" in param_names
                or (not extra_conds_succeeded and accepts_var_kwargs)
            ):
                if not torch.is_inference_mode_enabled() or "t5xxl_ids" in kwargs:
                    t5_ids = safe_to_device(raw_t5_ids.detach().clone(), device=noisy_latents.device)
                    if torch.is_floating_point(t5_ids):
                        t5_ids = t5_ids.to(dtype=torch.long)
                    if t5_ids.ndim == 1:
                        t5_ids = t5_ids.unsqueeze(0)
                    elif t5_ids.ndim >= 3 and int(t5_ids.shape[0]) == 1:
                        t5_ids = t5_ids.squeeze(0)
                    if t5_ids.ndim == 2 and int(t5_ids.shape[0]) == 1 and bsz > 1:
                        t5_ids = t5_ids.repeat(bsz, 1)
                    kwargs["t5xxl_ids"] = t5_ids

            raw_t5_w = extra_cond_dict.get("t5xxl_weights")
            if isinstance(raw_t5_w, torch.Tensor) and (
                "t5xxl_weights" in kwargs
                or "t5xxl_weights" in param_names
                or (not extra_conds_succeeded and accepts_var_kwargs)
            ):
                if not torch.is_inference_mode_enabled() or "t5xxl_weights" in kwargs:
                    t5_w = safe_to_device(raw_t5_w.detach().clone(), device=noisy_latents.device, dtype=noisy_latents.dtype)
                    if t5_w.ndim == 1:
                        t5_w = t5_w.unsqueeze(0).unsqueeze(-1)
                    elif t5_w.ndim == 2:
                        t5_w = t5_w.unsqueeze(-1)
                    elif t5_w.ndim == 4 and int(t5_w.shape[0]) == 1:
                        t5_w = t5_w.squeeze(0)
                    if t5_w.ndim == 3 and int(t5_w.shape[0]) == 1 and bsz > 1:
                        t5_w = t5_w.repeat(bsz, 1, 1)
                    kwargs["t5xxl_weights"] = t5_w

        # 2. Ensure primary text conditioning is mapped to the parameter name expected by backbone.forward
        if not any(k in kwargs for k in ("context", "prompt_embeds", "encoder_hidden_states", "txt")):
            if "context" in param_names:
                kwargs["context"] = prompt_embeds
            elif "prompt_embeds" in param_names:
                kwargs["prompt_embeds"] = prompt_embeds
            elif "encoder_hidden_states" in param_names:
                kwargs["encoder_hidden_states"] = prompt_embeds
            elif "txt" in param_names:
                kwargs["txt"] = prompt_embeds
            elif accepts_var_kwargs:
                kwargs["context"] = prompt_embeds

        # 3. Supply pooled/ADM `y` if `extra_conds` did not already build it AND the model requires pooled conditioning
        has_vec_embedder = any(
            hasattr(backbone, attr) for attr in ("vector_in", "y_embedder", "p_embedder", "label_emb")
        )
        backbone_accepts_y = not (
            hasattr(backbone, "num_classes")
            and getattr(backbone, "num_classes", None) is None
            and not has_vec_embedder
        )
        if not backbone_accepts_y:
            kwargs.pop("y", None)
        elif (
            pooled_output is not None
            and (self.spec.requires_pooled_conditioning or has_vec_embedder)
            and "y" not in kwargs
            and "pooled_projections" not in kwargs
        ):
            if "y" in param_names or accepts_var_kwargs:
                kwargs["y"] = pooled_output
            elif "pooled_projections" in param_names:
                kwargs["pooled_projections"] = pooled_output
        elif (
            "y" not in kwargs
            and "pooled_projections" not in kwargs
            and getattr(backbone, "num_classes", None) is not None
            and ("y" in param_names or accepts_var_kwargs)
        ):
            expected_v = self._infer_backbone_vector_dim(backbone) or 2816
            base_v = expected_v - 1536 if expected_v > 1536 else expected_v
            kwargs["y"] = torch.zeros((bsz, base_v), device=noisy_latents.device, dtype=noisy_latents.dtype)

        # Adapt vector conditioning (`y` / `pooled_projections`) width to match `label_emb` / `vector_in` / `y_embedder`
        for vec_key in ("y", "pooled_projections"):
            if isinstance(kwargs.get(vec_key), torch.Tensor):
                kwargs[vec_key] = self._prepare_vector_embeds(
                    kwargs[vec_key],
                    backbone,
                    spatial_h=spatial_h,
                    spatial_w=spatial_w,
                )

        # 4. Supply guidance embedding if required by the model (e.g. Flux / Hunyuan / Wan guidance-distilled)
        if (
            (self.spec.requires_guidance_embed or hasattr(backbone, "guidance_in"))
            and "guidance" in param_names
            and "guidance" not in kwargs
        ):
            if noisy_latents.device.type in ("privateuseone", "dml"):
                kwargs["guidance"] = safe_to_device(
                    torch.full((noisy_latents.shape[0],), self.guidance_scale, dtype=torch.float32),
                    device=noisy_latents.device,
                    dtype=noisy_latents.dtype,
                )
            else:
                kwargs["guidance"] = torch.full(
                    (noisy_latents.shape[0],),
                    self.guidance_scale,
                    device=noisy_latents.device,
                    dtype=noisy_latents.dtype,
                )

        # 5. Apply `c_concat` and pad any extra input channels required by Inpaint / I2V / Control / Edit models
        noisy_latents = self._prepare_concat_input_channels(
            noisy_latents,
            backbone=backbone,
            c_concat=c_concat_tensor,
            clean_latents=clean_latents,
        )

        # 6. Apply ComfyUI BaseModel.process_timestep if implemented on `self.inner_model`
        if self.inner_model is not None and hasattr(self.inner_model, "process_timestep") and callable(self.inner_model.process_timestep):
            try:
                processed_t = self.inner_model.process_timestep(timesteps, x=noisy_latents, **kwargs)
                if isinstance(processed_t, torch.Tensor):
                    timesteps = processed_t.to(device=noisy_latents.device, dtype=timesteps.dtype)
            except Exception:
                pass

        # 7. Always populate `transformer_options` with `sample_sigmas`, `sigmas`, and patch registries
        # whenever `transformer_options` is accepted by `backbone.forward` / `_forward`
        if "transformer_options" in param_names or "transformer_options" in kwargs:
            kwargs["transformer_options"] = self._build_transformer_options(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                sigma=sigma,
                existing=kwargs.get("transformer_options"),
            )

        # Remove positional argument names from kwargs to avoid TypeError: got multiple values for argument
        for pos_key in ("x", "timestep", "timesteps"):
            kwargs.pop(pos_key, None)

        is_dml_dev = noisy_latents.device.type in ("privateuseone", "dml")
        dml_infer_ctx = (
            torch.inference_mode(False)
            if (is_dml_dev and torch.is_inference_mode_enabled())
            else contextlib.nullcontext()
        )
        with dml_infer_ctx:
            if is_inference_tensor(noisy_latents):
                noisy_latents = noisy_latents.detach().clone()
            if is_inference_tensor(timesteps):
                timesteps = timesteps.detach().clone()
            if isinstance(prompt_embeds, torch.Tensor) and is_inference_tensor(prompt_embeds):
                prompt_embeds = prompt_embeds.detach().clone()

            for k, v in list(kwargs.items()):
                if isinstance(v, torch.Tensor) and is_inference_tensor(v):
                    kwargs[k] = v.detach().clone()

            mm_mod_for_train: Optional[Any] = None
            prev_mm_in_training: Optional[bool] = None
            if torch.is_grad_enabled():
                if not noisy_latents.requires_grad:
                    noisy_latents = noisy_latents.detach().requires_grad_(True)
                mp_mod = sys.modules.get("comfy.model_prefetch")
                attn_mod = sys.modules.get("comfy.ldm.modules.attention")
                if (
                    (mp_mod is not None and not getattr(getattr(mp_mod, "prefetch_queue_pop", None), "_omni_patched", False))
                    or (attn_mod is not None and not getattr(attn_mod, "_omni_attn_hooks_installed", False))
                ):
                    install_comfy_runtime_hooks()
                _rebind_comfy_attention_modules(backbone)
                if hasattr(backbone, "reset_prefix_cache") and callable(backbone.reset_prefix_cache):
                    try:
                        backbone.reset_prefix_cache(False)
                    except TypeError:
                        try:
                            backbone.reset_prefix_cache()
                        except Exception:
                            pass
                    except Exception:
                        pass
                mm_mod_for_train = sys.modules.get("comfy.model_management")
                if mm_mod_for_train is not None:
                    prev_mm_in_training = bool(getattr(mm_mod_for_train, "in_training", False))
                    if not prev_mm_in_training:
                        try:
                            mm_mod_for_train.in_training = True
                        except Exception:
                            mm_mod_for_train = None
                minimax_mod = sys.modules.get("comfy.ldm.minimax.model")
                if minimax_mod is not None:
                    try:
                        ms_fn = getattr(minimax_mod, "_mod_scale_shift", None)
                        if callable(ms_fn) and not getattr(ms_fn, "_omni_autograd_safe", False):
                            def _omni_mod_scale_shift(h: Any, shift: Any, scale: Any, segments: Any) -> Any:
                                if not torch.is_grad_enabled():
                                    for a, b, row in segments:
                                        h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
                                    return h
                                pieces = []
                                cur = 0
                                for a, b, row in sorted(segments, key=lambda s: int(s[0])):
                                    if a > cur:
                                        pieces.append(h[cur:a])
                                    pieces.append(h[a:b] * (1.0 + scale[row].to(h.dtype)) + shift[row].to(h.dtype))
                                    cur = b
                                if cur < h.shape[0]:
                                    pieces.append(h[cur:])
                                return torch.cat(pieces, dim=0) if pieces else h
                            _omni_mod_scale_shift._omni_autograd_safe = True  # type: ignore[attr-defined]
                            setattr(minimax_mod, "_mod_scale_shift", _omni_mod_scale_shift)
                        mg_fn = getattr(minimax_mod, "_mod_gate", None)
                        if callable(mg_fn) and not getattr(mg_fn, "_omni_autograd_safe", False):
                            def _omni_mod_gate(x: Any, gate: Any, other: Any, segments: Any) -> Any:
                                if not torch.is_grad_enabled():
                                    for a, b, row in segments:
                                        x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
                                    return x
                                pieces = []
                                cur = 0
                                for a, b, row in sorted(segments, key=lambda s: int(s[0])):
                                    if a > cur:
                                        pieces.append(x[cur:a])
                                    pieces.append(x[a:b] + other[a:b] * gate[row].to(x.dtype))
                                    cur = b
                                if cur < x.shape[0]:
                                    pieces.append(x[cur:])
                                return torch.cat(pieces, dim=0) if pieces else x
                            _omni_mod_gate._omni_autograd_safe = True  # type: ignore[attr-defined]
                            setattr(minimax_mod, "_mod_gate", _omni_mod_gate)
                    except Exception:
                        pass

            backbone_input: Any = noisy_latents
            latent_shapes = kwargs.pop("latent_shapes", None)
            if latent_shapes is not None:
                try:
                    import comfy.utils as _cu  # type: ignore

                    if hasattr(_cu, "unpack_latents"):
                        backbone_input = _cu.unpack_latents(noisy_latents, latent_shapes)
                except Exception:
                    backbone_input = noisy_latents
            elif (
                isinstance(backbone_input, torch.Tensor)
                and isinstance(getattr(backbone, "audio_latents_dim", None), int)
                and int(getattr(backbone, "audio_latents_dim", 0)) > 0
                and getattr(backbone, "audio_patch_proj", None) is not None
            ):
                # Dual-stream video+audio backbones (e.g. `MiniMaxH3Model`, whereas `MiniMaxH3ImageModel`
                # sets `audio_latents_dim = None` and takes a single tensor) unpack `video_x, audio_x = x[0], x[1]`
                # inside `forward`/`_forward`. When training on a single video/image latent stream without packed
                # `latent_shapes`, synthesize the companion silent audio latent stream matching `MiniMaxH3AV`.
                video_in = backbone_input.unsqueeze(2) if backbone_input.ndim == 4 else backbone_input
                audio_c = int(getattr(backbone, "audio_latents_dim", 32))
                t_tokens = int(video_in.shape[2]) if video_in.ndim >= 5 else 1
                frames_per_token = (1, 4, 4, 4, 4)
                frame_count = sum(frames_per_token[i % len(frames_per_token)] for i in range(max(1, t_tokens)))
                audio_t = max(1, int(round(frame_count * (5.0 / 3.0))))
                dummy_audio = torch.zeros(
                    (int(video_in.shape[0]), audio_c, 2, audio_t),
                    device=video_in.device,
                    dtype=video_in.dtype,
                )
                backbone_input = [video_in, dummy_audio]

            try:
                if kwargs:
                    try:
                        out = backbone(backbone_input, timesteps, **kwargs)
                    except TypeError:
                        if param_names:
                            filtered_kw = {k: v for k, v in kwargs.items() if k in param_names}
                            out = backbone(backbone_input, timesteps, **filtered_kw)
                        else:
                            raise
                else:
                    out = backbone(backbone_input, timesteps, prompt_embeds)
            finally:
                if mm_mod_for_train is not None and prev_mm_in_training is False:
                    try:
                        mm_mod_for_train.in_training = False
                    except Exception:
                        pass

            if isinstance(out, (tuple, list)):
                if (
                    latent_shapes is not None
                    and len(out) > 1
                    and all(isinstance(t, torch.Tensor) for t in out)
                ):
                    try:
                        import comfy.utils as _cu  # type: ignore

                        if hasattr(_cu, "pack_latents"):
                            out = _cu.pack_latents(list(out))[0]
                        else:
                            out = out[0]
                    except Exception:
                        out = out[0]
                else:
                    out = out[0]
            elif isinstance(out, dict):
                out = out.get("sample", next(iter(out.values())))

            return out

    @classmethod
    def _align_prediction_to_target(cls, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Ensure `pred` returned by `backbone.forward` matches `target.shape` when a model
        adds/removes a singleton temporal axis (`[B, C, 1, H, W]` <-> `[B, C, H, W]`),
        packs/unpacks spatial patches (`16ch <-> 64ch`), or outputs extra auxiliary channels.
        """
        if pred.shape == target.shape:
            return pred
        if pred.ndim == 5 and target.ndim == 4 and pred.shape[2] == 1:
            pred = pred[:, :, 0, :, :]
        elif pred.ndim == 4 and target.ndim == 5 and target.shape[2] == 1:
            pred = pred.unsqueeze(2)
        if pred.shape != target.shape and pred.ndim in (4, 5) and target.ndim == pred.ndim:
            if int(pred.shape[1]) != int(target.shape[1]):
                adapted = cls._adapt_latent_channels_spatial(pred, int(target.shape[1]))
                if adapted is not None:
                    pred = adapted
                elif int(pred.shape[1]) > int(target.shape[1]) and pred.shape[2:] == target.shape[2:]:
                    pred = pred[:, : int(target.shape[1])]
            if int(pred.shape[-2]) > int(target.shape[-2]) or int(pred.shape[-1]) > int(target.shape[-1]):
                pred = pred[..., : int(target.shape[-2]), : int(target.shape[-1])]
        return pred

    def _get_device_sigmas(self, sampling: Any, device: torch.device) -> Optional[torch.Tensor]:
        """Return `sampling.sigmas` cached on `device` in `float32` with zero CPU-GPU syncs."""
        sigmas = getattr(sampling, "sigmas", None)
        if not isinstance(sigmas, torch.Tensor) or sigmas.numel() <= 1:
            return None
        if (
            self._cached_sigmas is None
            or self._cached_sigmas.device != device
            or self._cached_sigmas.numel() != sigmas.numel()
        ):
            self._cached_sigmas = safe_to_device(sigmas.detach(), device=device, dtype=torch.float32)
        return self._cached_sigmas

    def _compute_device_timesteps(
        self,
        sampling: Any,
        sigma: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute `sampling.timestep(sigma)` on `device` without copying `sigma` back to CPU when
        `sampling` is a `ModelSamplingDiscrete` whose `log_sigmas` buffer resides on CPU.
        """
        log_sigmas = getattr(sampling, "log_sigmas", None)
        if (
            isinstance(log_sigmas, torch.Tensor)
            and log_sigmas.device != device
            and device.type not in ("privateuseone", "dml")
        ):
            if (
                self._cached_log_sigmas is None
                or self._cached_log_sigmas.device != device
                or self._cached_log_sigmas.numel() != log_sigmas.numel()
            ):
                self._cached_log_sigmas = safe_to_device(log_sigmas.detach(), device=device, dtype=torch.float32)
            try:
                orig_log_sigmas = log_sigmas
                sampling.log_sigmas = self._cached_log_sigmas
                try:
                    t_out = sampling.timestep(sigma)
                finally:
                    sampling.log_sigmas = orig_log_sigmas
                return safe_to_device(t_out, device=device, dtype=dtype)
            except Exception:
                pass
        return safe_to_device(sampling.timestep(sigma), device=device, dtype=dtype)

    def compute_training_loss(
        self,
        backbone: nn.Module,
        batch: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if device.type in ("privateuseone", "dml") and dtype == torch.bfloat16:
            dtype = torch.float16
        raw_latents = safe_to_device(batch["latents"], device=device, dtype=dtype)
        latent_scaled = bool(batch.get("latent_scaled", True))
        latents = self._validate_and_prepare_latents(
            raw_latents,
            backbone=backbone,
            latent_scaled=latent_scaled,
        )

        prompt_embeds = safe_to_device(batch["prompt_embeds"], device=device, dtype=dtype)
        pooled_output = batch.get("pooled_output")
        if isinstance(pooled_output, torch.Tensor):
            pooled_output = safe_to_device(pooled_output, device=device, dtype=dtype)
        elif self.spec.requires_pooled_conditioning:
            pooled_output = prompt_embeds.to(dtype=torch.float32).mean(dim=1).to(dtype=dtype)
        else:
            pooled_output = None

        extra_cond_dict = batch.get("extra_conds")

        bsz = latents.shape[0]
        broadcast_shape = [bsz] + [1] * (latents.ndim - 1)
        is_flow_objective = self.spec.objective_type in ("rectified_flow", "flow_matching")
        min_snr_gamma = float(getattr(self, "min_snr_gamma", 0.0))
        snr_weight: Optional[torch.Tensor] = None
        t_lo = float(getattr(self, "min_timestep", 0.0))
        t_hi = float(getattr(self, "max_timestep", 1.0))
        t_span = max(1e-5, t_hi - t_lo)

        if self.model_sampling is not None:
            sampling = self.model_sampling
            sigmas_dev = self._get_device_sigmas(sampling, device=device)
            if sigmas_dev is not None:
                max_idx = sigmas_dev.numel() - 1
                if is_flow_objective:
                    # Logit-normal timestep sampling for Rectified Flow / Flow Matching,
                    # mapped into [min_timestep, max_timestep] for single or split-stage models.
                    u_base = torch.sigmoid(torch.randn(bsz, device=device, dtype=torch.float32))
                    u_norm = t_lo + u_base * t_span
                else:
                    u_base = torch.rand(bsz, device=device, dtype=torch.float32)
                    u_norm = t_lo + u_base * t_span
                self.last_timestep = float(u_norm.mean().item())
                pos = (u_norm * float(max_idx)).clamp(0.0, max(0.0, float(max_idx) - 1e-5))
                idx0 = pos.long().clamp(0, max(0, max_idx - 1))
                frac = pos - idx0.to(dtype=torch.float32)
                s0 = sigmas_dev[idx0]
                s1 = sigmas_dev[idx0 + 1]
                sigma = (s0 + (s1 - s0) * frac).clamp(min=1e-5)
            else:
                if is_flow_objective:
                    u_base = torch.sigmoid(torch.randn(bsz, device=device, dtype=torch.float32))
                    flow_shift = float(getattr(sampling, "shift", 1.0) or 1.0)
                    if flow_shift > 0.0 and abs(flow_shift - 1.0) > 1e-4:
                        u_base = (flow_shift * u_base) / (1.0 + (flow_shift - 1.0) * u_base)
                else:
                    u_base = torch.rand(bsz, device=device, dtype=torch.float32)
                u_norm = (t_lo + u_base * t_span).clamp(1e-4, 0.999)
                self.last_timestep = float(u_norm.mean().item())
                sigma = u_norm

            noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32).to(dtype=latents.dtype)

            # Pass broadcast-shaped sigma [B, 1, 1, 1] to noise_scaling so both CONST and EPS
            # (which computes `noise * sigma` without reshaping) work for any batch_size >= 1.
            sigma_b = sigma.view(*broadcast_shape)
            if callable(getattr(sampling, "noise_scaling", None)):
                noisy_latents = sampling.noise_scaling(sigma_b, noise, latents).to(dtype=dtype)
            elif is_flow_objective:
                noisy_latents = (sigma_b * noise + (1.0 - sigma_b) * latents).to(dtype=dtype)
            else:
                noisy_latents = (latents + noise * sigma_b).to(dtype=dtype)

            if callable(getattr(sampling, "calculate_input", None)):
                xc = sampling.calculate_input(sigma_b, noisy_latents).to(dtype=dtype)
            elif is_flow_objective:
                xc = noisy_latents
            else:
                xc = (noisy_latents / torch.sqrt(1.0 + sigma_b * sigma_b)).to(dtype=dtype)
            t_dtype = dtype if device.type in ("privateuseone", "dml") else torch.float32
            timesteps = self._compute_device_timesteps(sampling, sigma, device=device, dtype=t_dtype).view(bsz)

            if self.spec.objective_type == "ddpm_eps":
                target = noise
                if min_snr_gamma > 0.0:
                    sig_c = sigma.to(dtype=torch.float32).clamp(min=1e-4)
                    snr = (sig_c * sig_c).reciprocal()
                    snr_weight = torch.clamp(snr, max=min_snr_gamma) / snr.clamp(min=1e-5)
            elif self.spec.objective_type == "v_prediction":
                # Compute in float32 so `sig_exp * sig_exp` never overflows float16 (65504.0) when sigma > 255
                sig_exp = sigma_b.to(dtype=torch.float32)
                lat_f32 = latents.to(dtype=torch.float32)
                noise_f32 = noise.to(dtype=torch.float32)
                target = (noise_f32 - sig_exp * lat_f32) / torch.sqrt(sig_exp * sig_exp + 1.0)
                if min_snr_gamma > 0.0:
                    sig_c = sigma.to(dtype=torch.float32).clamp(min=1e-4)
                    snr = (sig_c * sig_c).reciprocal()
                    snr_weight = torch.clamp(snr, max=min_snr_gamma) / (snr + 1.0)
            elif self.spec.objective_type == "edm":
                sig_exp = sigma_b.to(dtype=torch.float32)
                lat_f32 = latents.to(dtype=torch.float32)
                noise_f32 = noise.to(dtype=torch.float32)
                sigma_data = float(getattr(sampling, "sigma_data", 1.0))
                sd_sq = sigma_data * sigma_data
                target = (sig_exp * lat_f32 - sd_sq * noise_f32) / (
                    sigma_data * torch.sqrt(sig_exp * sig_exp + sd_sq)
                )
            elif self.spec.objective_type == "x0":
                target = latents
            else:
                target = noise.to(dtype=torch.float32) - latents.to(dtype=torch.float32)

            pred = self._forward_backbone(
                backbone=backbone,
                noisy_latents=xc,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_output=pooled_output,
                extra_cond_dict=extra_cond_dict,
                sigma=sigma,
                clean_latents=latents,
            )
        elif self.spec.objective_type == "ddpm_eps":
            # Fallback DDPM Epsilon-Prediction when model_sampling is not present
            u_base = torch.rand(bsz, device=device, dtype=torch.float32)
            u = (t_lo + u_base * t_span).clamp(1e-4, 0.9999)
            self.last_timestep = float(u.mean().item())
            timesteps = (u * 1000.0).to(dtype=dtype)
            cos_u = torch.cos(u * 1.5707963)
            alpha_bar = cos_u * cos_u
            one_minus_ab = (1.0 - alpha_bar).clamp(min=1e-5)
            sqrt_alpha_bar = torch.sqrt(alpha_bar.clamp(min=1e-5)).to(dtype=dtype).view(*broadcast_shape)
            sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_ab).to(dtype=dtype).view(*broadcast_shape)

            snr = alpha_bar.clamp(min=1e-5) / one_minus_ab
            if min_snr_gamma > 0.0:
                snr_weight = torch.clamp(snr, max=min_snr_gamma) / snr.clamp(min=1e-5)

            noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32).to(dtype=latents.dtype)
            noisy_latents = sqrt_alpha_bar * latents + sqrt_one_minus_alpha_bar * noise
            target = noise

            pred = self._forward_backbone(
                backbone=backbone,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_output=pooled_output,
                extra_cond_dict=extra_cond_dict,
                sigma=u.to(dtype=dtype),
                clean_latents=latents,
            )
        else:
            # Fallback Rectified Flow Velocity Matching when model_sampling is not present
            u_base = torch.sigmoid(torch.randn(bsz, device=device, dtype=torch.float32))
            u_norm = (t_lo + u_base * t_span).clamp(1e-4, 0.9999)
            self.last_timestep = float(u_norm.mean().item())
            timesteps = u_norm.to(dtype=dtype)
            t_expand = timesteps.view(*broadcast_shape)

            noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32).to(dtype=latents.dtype)
            noisy_latents = latents + t_expand * (noise - latents)
            target = noise - latents

            pred = self._forward_backbone(
                backbone=backbone,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_output=pooled_output,
                extra_cond_dict=extra_cond_dict,
                sigma=timesteps,
                clean_latents=latents,
            )

        pred = self._align_prediction_to_target(pred, target)
        diff = pred.to(dtype=torch.float32) - target.to(dtype=torch.float32)
        loss_type = str(getattr(self, "loss_type", "l2") or "l2").lower()
        if loss_type == "huber":
            huber_c = float(getattr(self, "huber_c", 0.1) or 0.1)
            loss_elem = torch.sqrt(diff * diff + (huber_c * huber_c)) - huber_c
        elif loss_type == "l1":
            loss_elem = torch.abs(diff)
        else:
            # Standard diffusion L2 (MSE) loss
            loss_elem = diff * diff

        # Optional spatial mask / alpha-channel loss weighting (`loss_mask` in [0, 1])
        raw_loss_mask = batch.get("loss_mask")
        if isinstance(raw_loss_mask, torch.Tensor) and raw_loss_mask.numel() > 0 and loss_elem.ndim >= 4:
            mask_dev = (
                torch.device("cpu")
                if (raw_loss_mask.device.type == "cpu" or device.type in ("privateuseone", "dml"))
                else device
            )
            m = raw_loss_mask.to(device=mask_dev, dtype=torch.float32)
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(1)
            elif m.ndim > 4:
                m = m.view(int(m.shape[0]), 1, int(m.shape[-2]), int(m.shape[-1]))
            if int(m.shape[0]) == 1 and int(loss_elem.shape[0]) > 1:
                m = m.expand(int(loss_elem.shape[0]), -1, -1, -1)

            lat_h, lat_w = int(loss_elem.shape[-2]), int(loss_elem.shape[-1])
            if int(m.shape[-2]) != lat_h or int(m.shape[-1]) != lat_w:
                if int(m.shape[-2]) >= lat_h and int(m.shape[-1]) >= lat_w:
                    m = F.interpolate(m, size=(lat_h, lat_w), mode="area")
                else:
                    m = F.interpolate(m, size=(lat_h, lat_w), mode="bilinear", align_corners=False)

            m = m.clamp(min=0.0)
            m_mean = m.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
            m_norm = safe_to_device(m / m_mean, device=device, dtype=torch.float32)
            if loss_elem.ndim == 5:
                m_norm = m_norm.unsqueeze(2)
            loss_elem = loss_elem * m_norm

        if snr_weight is not None:
            reduce_dims = list(range(1, loss_elem.ndim))
            per_sample_loss = loss_elem.mean(dim=reduce_dims) if reduce_dims else loss_elem
            return (per_sample_loss * snr_weight.to(dtype=torch.float32)).mean()
        return loss_elem.mean()

    def create_synthetic_batch(
        self,
        batch_size: int,
        resolution: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        latent_size = max(4, resolution // 8)
        if self.spec.supports_video_5d:
            latents = torch.randn(
                batch_size,
                self.spec.latent_channels,
                2,
                latent_size,
                latent_size,
                device=device,
                dtype=torch.float32,
            ).to(dtype=dtype)
        else:
            latents = torch.randn(
                batch_size,
                self.spec.latent_channels,
                latent_size,
                latent_size,
                device=device,
                dtype=torch.float32,
            ).to(dtype=dtype)
        prompt_embeds = torch.randn(
            batch_size, 77, self.spec.text_dim, device=device, dtype=torch.float32
        ).to(dtype=dtype)
        batch: Dict[str, Any] = {
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "latent_scaled": True,
        }
        if self.spec.requires_pooled_conditioning:
            batch["pooled_output"] = prompt_embeds.to(dtype=torch.float32).mean(dim=1).to(dtype=dtype)
        return batch


# ============================================================================
# 3. ComfyUI Model Inspection, Target Discovery & Runtime Hooks
# ============================================================================

def _get_linear_in_features(mod: Any) -> Optional[int]:
    """Extract `in_features` from `nn.Linear`, `LoRALinear`, GGUF `GGMLOps.Linear`, or `comfy.ops` Linear modules."""
    if mod is None:
        return None
    in_f = getattr(mod, "in_features", None)
    if isinstance(in_f, int) and in_f > 0:
        return in_f
    weight = getattr(mod, "weight", None)
    t_shape = getattr(weight, "tensor_shape", None) or getattr(weight, "orig_shape", None)
    if isinstance(t_shape, (tuple, list, torch.Size)) and len(t_shape) >= 2 and int(t_shape[1]) > 0:
        return int(t_shape[1])
    if isinstance(weight, torch.Tensor) and weight.ndim == 2 and weight.shape[1] > 0:
        return int(weight.shape[1])
    base = getattr(mod, "base_layer", None)
    if base is not None:
        return _get_linear_in_features(base)
    return None


_LEGACY_MOSS_LAYER_RE = re.compile(
    r"^diffusion_model\.layers_(\d+)_(proj_qkv|proj_o|proj_ffn_up_gate|proj_ffn_down)\.(lora_down\.weight|lora_up\.weight|alpha)$"
)
_LEGACY_MOSS_TEXT_BLOCK_RE = re.compile(
    r"^diffusion_model\.text_embedder_blocks_(\d+)_(qkv|proj|ffn_gate|ffn_up|ffn_down)\.(lora_down\.weight|lora_up\.weight|alpha)$"
)


def normalize_legacy_lora_key(key: str) -> str:
    """Convert legacy underscored keys to ComfyUI dotted format."""
    m1 = _LEGACY_MOSS_LAYER_RE.match(key)
    if m1:
        idx, proj_name, suffix = m1.groups()
        return f"diffusion_model.layers.{idx}.{proj_name}.{suffix}"
    m2 = _LEGACY_MOSS_TEXT_BLOCK_RE.match(key)
    if m2:
        idx, proj_name, suffix = m2.groups()
        return f"diffusion_model.text_embedder.blocks.{idx}.{proj_name}.{suffix}"
    return key


_WARNED_CONTEXT_ADAPTATIONS: set[Tuple[int, int]] = set()
_CACHED_DM_EXPECTED_CONTEXT_DIM: Dict[int, Optional[int]] = {}


def _adapt_context_width(
    input_t: torch.Tensor,
    expected_dim: int,
    model_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Adapt the feature width (last dimension) of text conditioning tensor `input_t` to `expected_dim`:
    - Exact integer multiples (`expected_dim % actual_dim == 0`, e.g. `4096 -> 53248 = 13 * 4096`)
      are tiled cleanly across the feature axis.
    - Non-integer widths (e.g. `768 -> 4096`, `2048 -> 4096`, `2560 -> 4096`, `3584 -> 4096`)
      repeat `ceil(expected_dim / actual_dim)` times and slice `[..., :expected_dim]` (or slice
      `[..., :expected_dim]` when `actual_dim > expected_dim`), emitting a one-time warning when
      non-integer adaptation occurs so workflows never crash on mismatched CLIPLoader connections.
    """
    if not isinstance(input_t, torch.Tensor) or input_t.ndim < 2 or expected_dim <= 0:
        return input_t
    actual_dim = int(input_t.shape[-1])
    if actual_dim <= 0 or actual_dim == expected_dim:
        return input_t
    if expected_dim > actual_dim:
        if expected_dim % actual_dim == 0:
            ratio = expected_dim // actual_dim
            return input_t.repeat(*([1] * (input_t.ndim - 1)), ratio)
        repeats = (expected_dim + actual_dim - 1) // actual_dim
        pair = (actual_dim, expected_dim)
        if pair not in _WARNED_CONTEXT_ADAPTATIONS:
            _WARNED_CONTEXT_ADAPTATIONS.add(pair)
            label = f" for '{model_name}'" if model_name else ""
            logging.warning(
                "[OmniLoRA] Connected CLIP produced %d-dim text embeddings, while model%s expects %d-dim; "
                "automatically adapting conditioning width (%d -> %d). For optimal quality, select the matching "
                "text encoder in CLIPLoader.",
                actual_dim,
                label,
                expected_dim,
                actual_dim,
                expected_dim,
            )
        return input_t.repeat(*([1] * (input_t.ndim - 1)), repeats)[..., :expected_dim]
    if actual_dim % expected_dim == 0:
        return input_t[..., -expected_dim:]
    pair = (actual_dim, expected_dim)
    if pair not in _WARNED_CONTEXT_ADAPTATIONS:
        _WARNED_CONTEXT_ADAPTATIONS.add(pair)
        label = f" for '{model_name}'" if model_name else ""
        logging.warning(
            "[OmniLoRA] Connected CLIP produced %d-dim text embeddings, while model%s expects %d-dim; "
            "automatically slicing conditioning width (%d -> %d).",
            actual_dim,
            label,
            expected_dim,
            actual_dim,
            expected_dim,
        )
    return input_t[..., :expected_dim]


def _infer_dm_expected_context_dim(
    backbone: Optional[nn.Module],
    unet_config: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Infer the expected text conditioning feature width for a ComfyUI `diffusion_model`."""
    if backbone is None:
        return None
    if hasattr(backbone, "_omni_expected_context_dim"):
        return getattr(backbone, "_omni_expected_context_dim")

    for norm_attr in ("llm_cond_norm", "context_norm", "txt_norm"):
        norm_mod = getattr(backbone, norm_attr, None)
        if norm_mod is not None:
            ns = getattr(norm_mod, "normalized_shape", None)
            if isinstance(ns, (tuple, list, torch.Size)) and len(ns) == 1 and int(ns[0]) > 0:
                setattr(backbone, "_omni_expected_context_dim", int(ns[0]))
                return int(ns[0])

    text_embedder = getattr(backbone, "text_embedder", None)
    if text_embedder is not None:
        proj = getattr(text_embedder, "proj", None)
        in_f = _get_linear_in_features(proj)
        if in_f is not None:
            setattr(backbone, "_omni_expected_context_dim", in_f)
            return in_f

    for attr in ("context_in", "context_embedder", "txt_embed", "cond_proj", "caption_projection", "txt_in", "llm_proj"):
        mod = getattr(backbone, attr, None)
        in_f = _get_linear_in_features(mod)
        if in_f is not None:
            setattr(backbone, "_omni_expected_context_dim", in_f)
            return in_f

    text_emb = getattr(backbone, "text_embedding", None)
    if isinstance(text_emb, nn.Sequential) and len(text_emb) > 0:
        in_f = _get_linear_in_features(text_emb[0])
        if in_f is not None:
            setattr(backbone, "_omni_expected_context_dim", in_f)
            return in_f

    for name, mod in backbone.named_modules():
        if any(
            name.endswith(suffix)
            for suffix in (
                "attn2.to_k",
                "cross_attn.k",
                "cross_attn.k_proj",
                "cross_attn.to_k",
                "cross_attn.to_kv",
                "txt_attn.qkv",
                "context_block.attn.qkv",
            )
        ):
            in_f = _get_linear_in_features(mod)
            if in_f is not None:
                setattr(backbone, "_omni_expected_context_dim", in_f)
                return in_f

    if isinstance(unet_config, dict):
        for cfg_key in (
            "context_in_dim",
            "joint_attention_dim",
            "caption_channels",
            "cross_attention_dim",
            "context_dim",
            "llm_dim",
        ):
            cfg_val = unet_config.get(cfg_key)
            if isinstance(cfg_val, (tuple, list)) and cfg_val:
                non_zero = [int(v) for v in cfg_val if isinstance(v, int) and v > 0]
                if non_zero:
                    cfg_val = non_zero[-1]
            if isinstance(cfg_val, int) and cfg_val > 0:
                setattr(backbone, "_omni_expected_context_dim", cfg_val)
                return cfg_val

    setattr(backbone, "_omni_expected_context_dim", None)
    return None


def _adapt_norm_input_last_dim(
    input_t: torch.Tensor,
    normalized_shape: Any,
) -> torch.Tensor:
    """
    Zero-overhead conditioning width adapter for `rms_norm` / `layer_norm`:
    When a model's conditioning norm (such as `Ideogram4.llm_cond_norm` with `normalized_shape=[53248]`,
    `13 * 4096`) receives a single-layer Qwen embedding (`[B, seq, 4096]`) or narrower CLIP embedding,
    automatically adapts the last dimension so training and KSampler inference succeed.
    """
    if (
        isinstance(input_t, torch.Tensor)
        and input_t.ndim >= 2
        and isinstance(normalized_shape, (tuple, list, torch.Size))
        and len(normalized_shape) == 1
    ):
        expected_dim = int(normalized_shape[0])
        actual_dim = int(input_t.shape[-1])
        if actual_dim != expected_dim and actual_dim > 0 and expected_dim > 0:
            return _adapt_context_width(input_t, expected_dim)
    return input_t


def install_comfy_runtime_hooks() -> None:
    """
    Install non-intrusive ComfyUI runtime hooks so that:
    1. If a cached `diffusion_model` still has any residual `LoRALinear` wrappers, they are
       unwrapped before `LoraLoader` inspects `model.state_dict()`.
    2. `comfy.lora.model_lora_keys_unet` registers both unprefixed (`<module>`) and prefixed
       (`diffusion_model.<module>`) keys for all direct and tuple-sliced weights (including
       GGUF `tensor_shape`-aware `.img_mlp.gate_layer` / `.proj` slices).
    3. `torch.nn.functional.rms_norm` and `layer_norm` automatically adapt integer-multiple
       multi-layer conditioning embeddings (`4096 <-> 53248 = 13 * 4096`).
    """
    install_dml_fp64_scalar_guards()

    def _pure_fp32_rms_norm(
        x: torch.Tensor,
        normalized_shape: Any,
        weight: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        ndim_norm = len(normalized_shape) if isinstance(normalized_shape, (tuple, list, torch.Size)) else 1
        dims = tuple(range(-ndim_norm, 0))
        eps_val = float(eps) if eps is not None else 1e-6
        x_f32 = x.to(dtype=torch.float32)
        ms = (x_f32 * x_f32).mean(dim=dims, keepdim=True)
        normed = (x_f32 / torch.sqrt(ms + eps_val)).to(dtype=x.dtype)
        if weight is not None:
            normed = normed * safe_to_device(weight, device=x.device, dtype=x.dtype)
        return normed

    # 1. Install RMSNorm / LayerNorm integer-multiple conditioning width adapter + DirectML fallbacks
    # Never overwrite hardened `_omni_dml_safe` wrappers installed by `LoRATrainingPipeline`.
    try:
        if hasattr(F, "rms_norm") and not getattr(F.rms_norm, "_omni_dml_safe", False):
            orig_f_rms_norm = getattr(F, "_omni_orig_rms_norm", F.rms_norm)
            setattr(F, "_omni_orig_rms_norm", orig_f_rms_norm)

            def _omni_f_rms_norm(
                input: torch.Tensor,
                normalized_shape: List[int],
                weight: Optional[torch.Tensor] = None,
                eps: Optional[float] = None,
            ) -> torch.Tensor:
                adapted = _adapt_norm_input_last_dim(input, normalized_shape)
                if getattr(adapted.device, "type", "cpu") in ("privateuseone", "dml"):
                    return _pure_fp32_rms_norm(adapted, normalized_shape, weight=weight, eps=eps)
                try:
                    return orig_f_rms_norm(adapted, normalized_shape, weight=weight, eps=eps)
                except (RuntimeError, NotImplementedError):
                    return _pure_fp32_rms_norm(adapted, normalized_shape, weight=weight, eps=eps)

            F.rms_norm = _omni_f_rms_norm

        if hasattr(torch, "rms_norm") and not getattr(torch.rms_norm, "_omni_dml_safe", False):
            orig_t_rms_norm = getattr(torch, "_omni_orig_rms_norm", torch.rms_norm)
            setattr(torch, "_omni_orig_rms_norm", orig_t_rms_norm)

            def _omni_t_rms_norm(
                input: torch.Tensor,
                normalized_shape: Any,
                weight: Optional[torch.Tensor] = None,
                eps: Optional[float] = None,
            ) -> torch.Tensor:
                adapted = _adapt_norm_input_last_dim(input, normalized_shape)
                if getattr(adapted.device, "type", "cpu") in ("privateuseone", "dml"):
                    return _pure_fp32_rms_norm(adapted, normalized_shape, weight=weight, eps=eps)
                try:
                    return orig_t_rms_norm(adapted, normalized_shape, weight, eps)
                except (RuntimeError, NotImplementedError):
                    return _pure_fp32_rms_norm(adapted, normalized_shape, weight=weight, eps=eps)

            torch.rms_norm = _omni_t_rms_norm  # type: ignore[assignment]

        if hasattr(F, "pixel_unshuffle") and not getattr(F.pixel_unshuffle, "_omni_dml_safe", False):
            orig_pixel_unshuffle = getattr(F, "_omni_orig_pixel_unshuffle", F.pixel_unshuffle)
            setattr(F, "_omni_orig_pixel_unshuffle", orig_pixel_unshuffle)

            def _omni_pixel_unshuffle_5d(tensor: torch.Tensor, downscale_factor: int) -> torch.Tensor:
                ctx = (
                    torch.inference_mode(False)
                    if torch.is_inference_mode_enabled()
                    else contextlib.nullcontext()
                )
                with ctx:
                    r = int(downscale_factor)
                    b, c, h, w = tensor.shape
                    if is_inference_tensor(tensor) or not tensor.is_contiguous() or tensor.storage_offset() != 0:
                        tensor = tensor.clone()
                    # Merge (B * C) so intermediate tensor is 5D (within DirectML's 5D tensor descriptor limit), never 6D
                    x_5d = tensor.reshape(b * c, h // r, r, w // r, r).permute(0, 2, 4, 1, 3).contiguous()
                    return x_5d.reshape(b, c * r * r, h // r, w // r)

            def _omni_pixel_unshuffle(input: torch.Tensor, downscale_factor: int) -> torch.Tensor:
                if (
                    isinstance(input, torch.Tensor)
                    and getattr(input.device, "type", "cpu") in ("privateuseone", "dml")
                    and input.ndim == 4
                ):
                    return _omni_pixel_unshuffle_5d(input, downscale_factor)
                try:
                    return orig_pixel_unshuffle(input, downscale_factor)
                except (RuntimeError, NotImplementedError):
                    if isinstance(input, torch.Tensor) and input.ndim == 4:
                        return _omni_pixel_unshuffle_5d(input, downscale_factor)
                    raise

            F.pixel_unshuffle = _omni_pixel_unshuffle

        if hasattr(F, "pixel_shuffle") and not getattr(F.pixel_shuffle, "_omni_dml_safe", False):
            orig_pixel_shuffle = getattr(F, "_omni_orig_pixel_shuffle", F.pixel_shuffle)
            setattr(F, "_omni_orig_pixel_shuffle", orig_pixel_shuffle)

            def _omni_pixel_shuffle_5d(tensor: torch.Tensor, upscale_factor: int) -> torch.Tensor:
                ctx = (
                    torch.inference_mode(False)
                    if torch.is_inference_mode_enabled()
                    else contextlib.nullcontext()
                )
                with ctx:
                    r = int(upscale_factor)
                    b, c_in, h, w = tensor.shape
                    c_out = c_in // (r * r)
                    if is_inference_tensor(tensor) or not tensor.is_contiguous() or tensor.storage_offset() != 0:
                        tensor = tensor.clone()
                    # Merge (B * C_out) so intermediate tensor is 5D (within DirectML's 5D tensor descriptor limit), never 6D
                    x_5d = tensor.reshape(b * c_out, r, r, h, w).permute(0, 3, 1, 4, 2).contiguous()
                    return x_5d.reshape(b, c_out, h * r, w * r)

            def _omni_pixel_shuffle(input: torch.Tensor, upscale_factor: int) -> torch.Tensor:
                if (
                    isinstance(input, torch.Tensor)
                    and getattr(input.device, "type", "cpu") in ("privateuseone", "dml")
                    and input.ndim == 4
                ):
                    return _omni_pixel_shuffle_5d(input, upscale_factor)
                try:
                    return orig_pixel_shuffle(input, upscale_factor)
                except (RuntimeError, NotImplementedError):
                    if isinstance(input, torch.Tensor) and input.ndim == 4:
                        return _omni_pixel_shuffle_5d(input, upscale_factor)
                    raise

            F.pixel_shuffle = _omni_pixel_shuffle

        if hasattr(F, "conv3d") and not getattr(F.conv3d, "_omni_dml_safe", False):
            orig_conv3d = getattr(F, "_omni_orig_conv3d", F.conv3d)
            setattr(F, "_omni_orig_conv3d", orig_conv3d)

            def _omni_conv3d(
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                stride: Any = 1,
                padding: Any = 0,
                dilation: Any = 1,
                groups: int = 1,
            ) -> torch.Tensor:
                if (
                    isinstance(input, torch.Tensor)
                    and getattr(input.device, "type", "cpu") in ("privateuseone", "dml")
                    and input.ndim == 5
                    and isinstance(weight, torch.Tensor)
                    and weight.ndim == 5
                    and isinstance(padding, (int, tuple, list))
                ):
                    ctx = (
                        torch.inference_mode(False)
                        if torch.is_inference_mode_enabled()
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        if is_inference_tensor(input) or not input.is_contiguous() or input.storage_offset() != 0:
                            input = input.clone()
                        if weight.device != input.device or weight.dtype != input.dtype:
                            weight = safe_to_device(weight, device=input.device, dtype=input.dtype)
                        if is_inference_tensor(weight) or not weight.is_contiguous() or weight.storage_offset() != 0:
                            weight = weight.clone()
                        if isinstance(bias, torch.Tensor):
                            if bias.device != input.device or bias.dtype != input.dtype:
                                bias = safe_to_device(bias, device=input.device, dtype=input.dtype)
                            if is_inference_tensor(bias) or not bias.is_contiguous() or bias.storage_offset() != 0:
                                bias = bias.clone()
                        s_tup = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
                        p_tup = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
                        d_tup = (dilation, dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
                        s_t, s_h, s_w = int(s_tup[0]), int(s_tup[1]), int(s_tup[2])
                        p_t, p_h, p_w = int(p_tup[0]), int(p_tup[1]), int(p_tup[2])
                        d_t, d_h, d_w = int(d_tup[0]), int(d_tup[1]), int(d_tup[2])
                        k_t = int(weight.shape[2])
                        b, c, t_len, h, w = input.shape
                        if k_t == 1 and s_t == 1 and p_t == 0:
                            x_2d = input.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t_len, c, h, w)
                            w_2d = weight.squeeze(2).clone()
                            out_2d = F.conv2d(
                                x_2d,
                                w_2d,
                                bias=bias,
                                stride=(s_h, s_w),
                                padding=(p_h, p_w),
                                dilation=(d_h, d_w),
                                groups=groups,
                            )
                            _, out_c, out_h, out_w = out_2d.shape
                            return out_2d.reshape(b, t_len, out_c, out_h, out_w).permute(0, 2, 1, 3, 4).contiguous()
                        if p_t > 0:
                            input = F.pad(input, (0, 0, 0, 0, p_t, p_t))
                        t_padded = int(input.shape[2])
                        eff_k_t = (k_t - 1) * d_t + 1
                        t_out_len = max(1, (t_padded - eff_k_t) // s_t + 1)
                        frames: List[torch.Tensor] = []
                        for t_idx in range(t_out_len):
                            t_start = t_idx * s_t
                            acc: Optional[torch.Tensor] = None
                            for kt_idx in range(k_t):
                                x_slice = input[:, :, t_start + kt_idx * d_t, :, :].clone()
                                w_slice = weight[:, :, kt_idx, :, :].clone()
                                conv_step = F.conv2d(
                                    x_slice,
                                    w_slice,
                                    bias=bias if kt_idx == 0 else None,
                                    stride=(s_h, s_w),
                                    padding=(p_h, p_w),
                                    dilation=(d_h, d_w),
                                    groups=groups,
                                )
                                acc = conv_step if acc is None else (acc + conv_step)
                            assert acc is not None
                            frames.append(acc.unsqueeze(2))
                        return torch.cat(frames, dim=2)
                return orig_conv3d(input, weight, bias, stride, padding, dilation, groups)

            F.conv3d = _omni_conv3d
    except Exception:
        pass

    # 2. Install universal `model_lora_keys_unet` hook (always refreshes to latest implementation)
    try:
        import comfy.lora  # type: ignore

        if hasattr(comfy.lora, "model_lora_keys_unet"):
            _orig_model_lora_keys_unet = getattr(
                comfy.lora,
                "_omni_orig_model_lora_keys_unet",
                comfy.lora.model_lora_keys_unet,
            )
            setattr(comfy.lora, "_omni_orig_model_lora_keys_unet", _orig_model_lora_keys_unet)

            def _omni_model_lora_keys_unet(
                model: Any,
                key_map: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                if key_map is None:
                    key_map = {}

                res_map = _orig_model_lora_keys_unet(model, key_map)
                try:
                    # Strip any `.base_layer` artifact keys/values if model has injected LoRA layers
                    clean_map: Dict[str, Any] = {}
                    for k, v in res_map.items():
                        ck = k.replace(".base_layer", "")
                        if isinstance(v, tuple):
                            cv = (v[0].replace(".base_layer.weight", ".weight") if isinstance(v[0], str) else v[0], v[1])
                        elif isinstance(v, str):
                            cv = v.replace(".base_layer.weight", ".weight")
                        else:
                            cv = v
                        clean_map[ck] = cv
                        clean_map[k] = v
                    res_map = clean_map

                    sd = model.state_dict() if hasattr(model, "state_dict") and callable(model.state_dict) else {}

                    # Ensure every `.weight` in `model.state_dict()` (including GGUF 1D packed weights)
                    # AND fused `.img_mlp.gate_up` layers have bidirectional `<module>` and
                    # `diffusion_model.<module>` entries in `res_map`.
                    for raw_k, w_tensor in sd.items():
                        if not isinstance(raw_k, str):
                            continue
                        clean_k = raw_k.replace(".base_layer.weight", ".weight")
                        if not clean_k.endswith(".weight"):
                            continue
                        short_mod = clean_k.removeprefix("diffusion_model.")[: -len(".weight")]
                        full_mod = f"diffusion_model.{short_mod}"
                        canonical_w = clean_k if clean_k.startswith("diffusion_model.") else f"diffusion_model.{clean_k}"
                        res_map.setdefault(short_mod, canonical_w)
                        res_map.setdefault(full_mod, canonical_w)

                        if clean_k.endswith(".img_mlp.gate_up.weight"):
                            t_shape = getattr(w_tensor, "tensor_shape", getattr(w_tensor, "orig_shape", None))
                            if isinstance(t_shape, (tuple, list, torch.Size)) and len(t_shape) >= 2:
                                out_f = int(t_shape[0])
                            elif isinstance(w_tensor, torch.Tensor) and w_tensor.ndim >= 2:
                                out_f = int(w_tensor.shape[0])
                            else:
                                continue
                            if out_f > 0 and out_f % 2 == 0:
                                half = out_f // 2
                                short_base = clean_k.removeprefix("diffusion_model.")[: -len(".gate_up.weight")]
                                res_map[f"{short_base}.gate_layer"] = (canonical_w, (0, 0, half))
                                res_map[f"diffusion_model.{short_base}.gate_layer"] = (canonical_w, (0, 0, half))
                                res_map[f"{short_base}.proj"] = (canonical_w, (0, half, half))
                                res_map[f"diffusion_model.{short_base}.proj"] = (canonical_w, (0, half, half))

                    # Register bidirectional `<module>` <-> `diffusion_model.<module>` aliases for all entries
                    for k, v in list(res_map.items()):
                        if isinstance(k, str) and not k.startswith("lora_"):
                            short_k = k[len("diffusion_model.") :] if k.startswith("diffusion_model.") else k
                            full_k = f"diffusion_model.{short_k}"
                            res_map.setdefault(short_k, v)
                            res_map.setdefault(full_k, v)
                        raw_k = v[0] if isinstance(v, tuple) and len(v) > 0 else v
                        if isinstance(raw_k, str) and raw_k.startswith("diffusion_model.") and raw_k.endswith(".weight"):
                            short_w = raw_k[len("diffusion_model.") : -len(".weight")]
                            full_w = raw_k[: -len(".weight")]
                            res_map.setdefault(short_w, raw_k)
                            res_map.setdefault(full_w, raw_k)
                except Exception:
                    pass
                return res_map

            comfy.lora.model_lora_keys_unet = _omni_model_lora_keys_unet
            comfy.lora._omni_lora_keys_patched = True

            try:
                import comfy.sd  # type: ignore

                if hasattr(comfy.sd, "model_lora_keys_unet"):
                    comfy.sd.model_lora_keys_unet = _omni_model_lora_keys_unet
            except Exception:
                pass
    except Exception:
        pass

    # 3. Guard `comfy_aimdo.malloc_graph` and `comfy.model_prefetch` so ComfyUI's inference-only
    # C memory graph compiler (`control.lib.malloc_graph_pop`) never runs or raises
    # `RuntimeError: aimdo memory compile error` (`rogues: N`) when autograd saves activations during training.
    def _is_omni_training_active() -> bool:
        if torch.is_grad_enabled():
            return True
        mm_mod = sys.modules.get("comfy.model_management")
        if mm_mod is not None and bool(getattr(mm_mod, "in_training", False)):
            return True
        return False

    def _patch_malloc_graph_obj(mg_target: Any) -> None:
        if mg_target is None:
            return
        is_cls = isinstance(mg_target, type)
        if not is_cls:
            _patch_malloc_graph_obj(type(mg_target))
        target_dict = getattr(mg_target, "__dict__", {})
        if isinstance(target_dict, dict) and target_dict.get("_omni_aimdo_patched", False):
            return
        try:
            setattr(mg_target, "_omni_aimdo_patched", True)
        except Exception:
            return

        if hasattr(mg_target, "_call"):
            orig_call = target_dict.get("_omni_orig_call") if isinstance(target_dict, dict) else None
            if orig_call is None:
                orig_call = getattr(mg_target, "_call")
            try:
                setattr(mg_target, "_omni_orig_call", orig_call)

                def _safe_aimdo_call(*args: Any, **kwargs: Any) -> Any:
                    if _is_omni_training_active():
                        return 2
                    try:
                        return orig_call(*args, **kwargs)
                    except RuntimeError as exc:
                        if "aimdo" in str(exc).lower() or "compile error" in str(exc).lower():
                            return 2
                        raise

                setattr(mg_target, "_call", _safe_aimdo_call)
            except Exception:
                pass

        for method_name in ("iterate", "push", "pop", "begin", "end", "pause", "resume"):
            if not hasattr(mg_target, method_name):
                continue
            orig_m = target_dict.get(f"_omni_orig_{method_name}") if isinstance(target_dict, dict) else None
            if orig_m is None:
                orig_m = getattr(mg_target, method_name)
            try:
                setattr(mg_target, f"_omni_orig_{method_name}", orig_m)

                def _make_safe_mg_method(orig_fn: Any) -> Any:
                    def _wrapped(*a: Any, **kw: Any) -> Any:
                        if _is_omni_training_active():
                            return False
                        try:
                            return orig_fn(*a, **kw)
                        except RuntimeError as exc:
                            if "aimdo" in str(exc).lower() or "compile error" in str(exc).lower():
                                return False
                            raise

                    return _wrapped

                setattr(mg_target, method_name, _make_safe_mg_method(orig_m))
            except Exception:
                pass

    _aimdo_mg = sys.modules.get("comfy_aimdo.malloc_graph")
    if _aimdo_mg is None:
        try:
            import comfy_aimdo.malloc_graph as _aimdo_mg  # type: ignore
        except Exception:
            _aimdo_mg = None
    if _aimdo_mg is not None:
        _patch_malloc_graph_obj(getattr(_aimdo_mg, "MallocGraph", None))
        _patch_malloc_graph_obj(getattr(_aimdo_mg, "malloc_graph", None))

    _mp = sys.modules.get("comfy.model_prefetch")
    if _mp is None:
        try:
            import comfy.model_prefetch as _mp  # type: ignore
        except Exception:
            _mp = None

    if _mp is not None:
        mg_inst = getattr(_mp, "malloc_graph", None)
        if mg_inst is not None:
            _patch_malloc_graph_obj(type(mg_inst))
            _patch_malloc_graph_obj(mg_inst)

        if hasattr(_mp, "malloc_graph_enabled") and not getattr(_mp.malloc_graph_enabled, "_omni_patched", False):
            _orig_mg_enabled = getattr(_mp, "_omni_orig_malloc_graph_enabled", _mp.malloc_graph_enabled)
            setattr(_mp, "_omni_orig_malloc_graph_enabled", _orig_mg_enabled)

            def _omni_malloc_graph_enabled(*args: Any, **kwargs: Any) -> bool:
                if _is_omni_training_active():
                    return False
                return bool(_orig_mg_enabled(*args, **kwargs))

            setattr(_omni_malloc_graph_enabled, "_omni_patched", True)
            _mp.malloc_graph_enabled = _omni_malloc_graph_enabled

        for _mp_fn_name in ("malloc_graph_begin", "malloc_graph_end"):
            if hasattr(_mp, _mp_fn_name) and not getattr(getattr(_mp, _mp_fn_name), "_omni_patched", False):
                _orig_mp_fn = getattr(_mp, f"_omni_orig_{_mp_fn_name}", getattr(_mp, _mp_fn_name))
                setattr(_mp, f"_omni_orig_{_mp_fn_name}", _orig_mp_fn)

                def _make_mp_noop(orig_f: Any) -> Any:
                    def _wrapped_mp(*args: Any, **kwargs: Any) -> Any:
                        if _is_omni_training_active():
                            return None
                        try:
                            return orig_f(*args, **kwargs)
                        except RuntimeError as exc:
                            if "aimdo" in str(exc).lower() or "compile error" in str(exc).lower():
                                return None
                            raise

                    setattr(_wrapped_mp, "_omni_patched", True)
                    return _wrapped_mp

                setattr(_mp, _mp_fn_name, _make_mp_noop(_orig_mp_fn))

        if hasattr(_mp, "prefetch_queue_pop") and not getattr(_mp.prefetch_queue_pop, "_omni_patched", False):
            _orig_pq_pop = getattr(_mp, "_omni_orig_prefetch_queue_pop", _mp.prefetch_queue_pop)
            setattr(_mp, "_omni_orig_prefetch_queue_pop", _orig_pq_pop)

            def _omni_prefetch_queue_pop(*args: Any, **kwargs: Any) -> Any:
                if _is_omni_training_active():
                    if "malloc_scope" in kwargs or "enable_graph" in kwargs:
                        kwargs = dict(kwargs)
                        if "malloc_scope" in kwargs:
                            kwargs["malloc_scope"] = None
                        if "enable_graph" in kwargs:
                            kwargs["enable_graph"] = False
                try:
                    return _orig_pq_pop(*args, **kwargs)
                except RuntimeError as exc:
                    if "aimdo" in str(exc).lower() or "compile error" in str(exc).lower():
                        return None
                    raise

            setattr(_omni_prefetch_queue_pop, "_omni_patched", True)
            _mp.prefetch_queue_pop = _omni_prefetch_queue_pop

    # 4. Install `comfy.model_base.BaseModel.extra_conds` context width adapter so KSampler /
    # downstream ComfyUI sampling automatically adapts `cross_attn` / `c_crossattn` width when needed.
    try:
        import comfy.model_base as _mb  # type: ignore

        base_cls = getattr(_mb, "BaseModel", None)
        if base_cls is not None and hasattr(base_cls, "extra_conds") and not getattr(base_cls, "_omni_extra_conds_patched", False):
            _orig_extra_conds = getattr(base_cls, "_omni_orig_extra_conds", base_cls.extra_conds)
            setattr(base_cls, "_omni_orig_extra_conds", _orig_extra_conds)

            def _omni_extra_conds(self: Any, **kwargs: Any) -> Dict[str, Any]:
                try:
                    dm = getattr(self, "diffusion_model", None)
                    mc = getattr(self, "model_config", None)
                    ucfg = getattr(mc, "unet_config", None) if mc is not None else None
                    exp_dim = _infer_dm_expected_context_dim(dm, ucfg if isinstance(ucfg, dict) else None)
                    if exp_dim is not None and exp_dim > 0:
                        for ck in ("cross_attn", "c_crossattn"):
                            ct = kwargs.get(ck)
                            if isinstance(ct, torch.Tensor) and ct.ndim >= 2 and int(ct.shape[-1]) != exp_dim:
                                kwargs = dict(kwargs)
                                kwargs[ck] = _adapt_context_width(ct, exp_dim, model_name=type(dm).__name__ if dm is not None else None)
                except Exception:
                    pass
                return _orig_extra_conds(self, **kwargs)

            setattr(base_cls, "extra_conds", _omni_extra_conds)
            setattr(base_cls, "_omni_extra_conds_patched", True)
    except Exception:
        pass

    # 5. Normalize Qwen3.5 / MOSS-Image text encoder state_dicts in `comfy.sd.detect_te_model` and
    # `comfy.sd.load_text_encoder_state_dicts` when checkpoints use `model.layers.0.linear_attn.A_log`,
    # `layers.0.linear_attn.A_log`, or `(text_encoders.)qwen35_*.transformer.model.layers.0.linear_attn.A_log`
    # instead of `model.language_model.layers.0.linear_attn.A_log`, preventing fallback to SD1ClipModel (768-dim).
    try:
        import comfy.sd as _csd  # type: ignore

        def _normalize_qwen35_sd_in_place(sd: Any) -> Any:
            if not isinstance(sd, dict) or "model.language_model.layers.0.linear_attn.A_log" in sd:
                return sd
            target_suffix = "layers.0.linear_attn.A_log"
            matched_prefix: Optional[str] = None
            if "model.layers.0.linear_attn.A_log" in sd:
                matched_prefix = "model."
            elif "layers.0.linear_attn.A_log" in sd:
                matched_prefix = ""
            else:
                for k in sd.keys():
                    if isinstance(k, str) and k.endswith(target_suffix):
                        matched_prefix = k[: -len(target_suffix)]
                        break
            if matched_prefix is None:
                return sd
            norm_key = f"{matched_prefix}layers.0.input_layernorm.weight"
            if norm_key not in sd:
                return sd
            plen = len(matched_prefix)
            remapped: Dict[str, Any] = {}
            for k, v in sd.items():
                if isinstance(k, str) and k.startswith(matched_prefix):
                    sub = k[plen:]
                    if sub.startswith(("layers.", "embed_tokens.", "norm.")):
                        remapped[f"model.language_model.{sub}"] = v
                    else:
                        remapped[k] = v
                else:
                    remapped[k] = v
            sd.clear()
            sd.update(remapped)
            return sd

        if hasattr(_csd, "detect_te_model") and not getattr(_csd.detect_te_model, "_omni_patched", False):
            _orig_detect_te = getattr(_csd, "_omni_orig_detect_te_model", _csd.detect_te_model)
            setattr(_csd, "_omni_orig_detect_te_model", _orig_detect_te)

            def _omni_detect_te_model(sd: Any) -> Any:
                _normalize_qwen35_sd_in_place(sd)
                return _orig_detect_te(sd)

            setattr(_omni_detect_te_model, "_omni_patched", True)
            _csd.detect_te_model = _omni_detect_te_model

        if hasattr(_csd, "load_text_encoder_state_dicts") and not getattr(_csd.load_text_encoder_state_dicts, "_omni_patched", False):
            _orig_load_te_sds = getattr(_csd, "_omni_orig_load_text_encoder_state_dicts", _csd.load_text_encoder_state_dicts)
            setattr(_csd, "_omni_orig_load_text_encoder_state_dicts", _orig_load_te_sds)

            def _omni_load_text_encoder_state_dicts(state_dicts: Any = [], *args: Any, **kwargs: Any) -> Any:
                if isinstance(state_dicts, (list, tuple)):
                    for sd_item in state_dicts:
                        _normalize_qwen35_sd_in_place(sd_item)
                return _orig_load_te_sds(state_dicts, *args, **kwargs)

            setattr(_omni_load_text_encoder_state_dicts, "_omni_patched", True)
            _csd.load_text_encoder_state_dicts = _omni_load_text_encoder_state_dicts
    except Exception:
        pass

    # 6. Guard `comfy.weight_adapter` LoRAAdapter.calculate_weight so legacy 2D `[rank, in_c]`
    # conv LoRA weights loaded onto 3D/4D/5D convolution layers (`[out_c, in_c, kH, kW]`) are
    # automatically upgraded at the center spatial tap instead of failing `.reshape(weight.shape)`.
    try:
        import comfy.weight_adapter as _cwa  # type: ignore

        adapter_classes: List[Any] = []
        if hasattr(_cwa, "LoRAAdapter"):
            adapter_classes.append(getattr(_cwa, "LoRAAdapter"))
        if hasattr(_cwa, "adapters") and isinstance(_cwa.adapters, (list, tuple)):
            for ad_cls in _cwa.adapters:
                if ad_cls not in adapter_classes and getattr(ad_cls, "name", "") == "lora":
                    adapter_classes.append(ad_cls)

        for lora_ad_cls in adapter_classes:
            if hasattr(lora_ad_cls, "calculate_weight") and not getattr(
                lora_ad_cls, "_omni_calc_weight_patched", False
            ):
                _orig_ad_calc = getattr(lora_ad_cls, "_omni_orig_calculate_weight", lora_ad_cls.calculate_weight)
                setattr(lora_ad_cls, "_omni_orig_calculate_weight", _orig_ad_calc)

                def _make_safe_lora_calc(orig_fn: Any) -> Any:
                    def _omni_ad_calculate_weight(
                        self: Any,
                        weight: torch.Tensor,
                        key: str,
                        *args: Any,
                        **kwargs: Any,
                    ) -> Any:
                        try:
                            w_tuple = getattr(self, "weights", None)
                            if (
                                isinstance(weight, torch.Tensor)
                                and weight.ndim >= 3
                                and isinstance(w_tuple, (tuple, list))
                                and len(w_tuple) >= 2
                            ):
                                up_t, down_t = w_tuple[0], w_tuple[1]
                                if (
                                    isinstance(up_t, torch.Tensor)
                                    and isinstance(down_t, torch.Tensor)
                                    and down_t.ndim == 2
                                    and up_t.shape[0] == weight.shape[0]
                                    and down_t.shape[1] == weight.shape[1]
                                    and weight.numel() != up_t.shape[0] * down_t.shape[1]
                                ):
                                    padded_down = torch.zeros(
                                        (down_t.shape[0], weight.shape[1], *weight.shape[2:]),
                                        dtype=down_t.dtype,
                                        device=down_t.device,
                                    )
                                    center_idx = (slice(None), slice(None)) + tuple(
                                        k // 2 for k in weight.shape[2:]
                                    )
                                    padded_down[center_idx] = down_t
                                    reshaped_up = up_t.reshape(
                                        up_t.shape[0], up_t.shape[1], *([1] * (weight.ndim - 2))
                                    )
                                    new_weights = list(w_tuple)
                                    new_weights[0] = reshaped_up
                                    new_weights[1] = padded_down
                                    self.weights = tuple(new_weights) if isinstance(w_tuple, tuple) else new_weights
                        except Exception:
                            pass
                        return orig_fn(self, weight, key, *args, **kwargs)

                    return _omni_ad_calculate_weight

                setattr(lora_ad_cls, "calculate_weight", _make_safe_lora_calc(_orig_ad_calc))
                setattr(lora_ad_cls, "_omni_calc_weight_patched", True)
    except Exception:
        pass

    # 7. Guard inference-only ComfyUI attention backends (`attention_comfy_kitchen_int8`,
    # `attention_comfy_kitchen_sol`, `attention_sage`, `attention3_sage`), pre-loaded `ComfyAttention`
    # instances, and direct `comfy_kitchen` / `comfy.quant_ops.ck` non-autograd kernels (`int8_attention`,
    # `prequantize_int8_attention`, `sol_attn`, `rms_rope`, `rms_rope1`, `rms_rope_split_half`,
    # `apply_rope`, `apply_rope1`, `apply_rope_split_half`, `adaln`, `rms_adaln`) during training.
    def _sdpa_4d_fallback(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: Optional[float] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn_mask is not None:
            attn_mask = _normalize_sdpa_attn_mask(attn_mask, q, k, skip_reshape=True)
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
        sdpa_kw: Dict[str, Any] = {}
        if scale is not None:
            sdpa_kw["scale"] = scale
        if k.ndim == 4 and q.ndim == 4 and k.shape[1] != q.shape[1]:
            rep_k = max(1, int(q.shape[1]) // max(1, int(k.shape[1])))
            rep_v = max(1, int(q.shape[1]) // max(1, int(v.shape[1])))
            k = k.repeat_interleave(rep_k, dim=1)
            v = v.repeat_interleave(rep_v, dim=1)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            **sdpa_kw,
        )

    def _rms_norm_t(
        x_t: torch.Tensor,
        w_t: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        w_cast: Optional[torch.Tensor] = None
        if isinstance(w_t, torch.Tensor):
            w_cast = safe_to_device(w_t, device=x_t.device, dtype=x_t.dtype)
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x_t, (int(x_t.shape[-1]),), weight=w_cast, eps=float(eps))
        x_f32 = x_t.to(dtype=torch.float32)
        ms = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
        out_t = (x_f32 * torch.rsqrt(ms + float(eps))).to(dtype=x_t.dtype)
        if w_cast is not None:
            out_t = out_t * w_cast
        return out_t

    def _torch_apply_rope_interleaved1(x_t: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        freqs = pe
        x_ = x_t.to(dtype=freqs.dtype).reshape(*x_t.shape[:-1], -1, 1, 2)
        if x_.ndim >= 4 and freqs.ndim >= 4 and x_.shape[2] != 1 and freqs.shape[2] != 1 and x_.shape[2] != freqs.shape[2]:
            freqs = freqs[:, :, : x_.shape[2]]
        x_out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
        return x_out.reshape(*x_t.shape).to(dtype=x_t.dtype)

    def _torch_apply_rope_split_half1(
        x_t: torch.Tensor,
        pe: torch.Tensor,
        rot_dim: Optional[int] = None,
    ) -> torch.Tensor:
        total_d = int(x_t.shape[-1])
        pe_rot_d = int(pe.shape[-3]) * 2 if pe.ndim >= 3 else total_d
        r_dim = int(rot_dim) if rot_dim is not None else min(total_d, pe_rot_d)
        if 0 < r_dim < total_d:
            head_part = x_t[..., :r_dim]
            tail_part = x_t[..., r_dim:]
            rot_head = _torch_apply_rope_split_half1(head_part, pe, rot_dim=r_dim)
            return torch.cat((rot_head, tail_part), dim=-1)
        x_ = x_t.to(dtype=pe.dtype).reshape(*x_t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        x_out = pe[..., 0] * x_[..., 0] + pe[..., 1] * x_[..., 1]
        return x_out.movedim(-1, -2).reshape(*x_t.shape).to(dtype=x_t.dtype)

    def _torch_rms_rope_interleaved(
        q: torch.Tensor,
        k: torch.Tensor,
        pe: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_n = _rms_norm_t(q, q_scale, eps=eps)
        k_n = _rms_norm_t(k, k_scale, eps=eps)
        return _torch_apply_rope_interleaved1(q_n, pe), _torch_apply_rope_interleaved1(k_n, pe)

    def _torch_rms_rope_split_half(
        q: torch.Tensor,
        k: torch.Tensor,
        pe: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        eps: float = 1e-6,
        rot_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_n = _rms_norm_t(q, q_scale, eps=eps)
        k_n = _rms_norm_t(k, k_scale, eps=eps)
        return (
            _torch_apply_rope_split_half1(q_n, pe, rot_dim=rot_dim),
            _torch_apply_rope_split_half1(k_n, pe, rot_dim=rot_dim),
        )

    try:
        comfy_attn = sys.modules.get("comfy.ldm.modules.attention")
        if comfy_attn is None and "comfy.ldm" in sys.modules:
            try:
                import comfy.ldm.modules.attention as comfy_attn  # type: ignore
            except Exception:
                comfy_attn = None

        if comfy_attn is not None:
            import functools

            target_attn_names = (
                "attention_comfy_kitchen_int8",
                "attention_comfy_kitchen_sol",
                "attention_sage",
                "attention3_sage",
            )
            replaced_map: Dict[int, Tuple[Any, Any]] = {}

            for fn_name in target_attn_names:
                cur_fn = getattr(comfy_attn, fn_name, None)
                if cur_fn is None:
                    continue

                orig_fn = getattr(cur_fn, "_omni_orig_attn_fn", cur_fn)
                if not getattr(cur_fn, "_omni_train_attn_safe", False):
                    orig_container = getattr(orig_fn, "container_function", None)
                    if callable(orig_container) and not getattr(orig_container, "_omni_train_attn_safe", False):
                        def _make_safe_container(orig_c: Any) -> Any:
                            def _safe_container(*c_args: Any, **c_kwargs: Any) -> Any:
                                if _is_omni_training_active():
                                    return _run_pytorch_sdpa_attention(*c_args, **c_kwargs)
                                return orig_c(*c_args, **c_kwargs)

                            setattr(_safe_container, "_omni_train_attn_safe", True)
                            setattr(_safe_container, "_omni_orig_attn_fn", orig_c)
                            return _safe_container

                        safe_c = _make_safe_container(orig_container)
                        try:
                            setattr(orig_fn, "container_function", safe_c)
                        except Exception:
                            pass

                    def _make_safe_attn_fn(orig_f: Any) -> Any:
                        @functools.wraps(orig_f)
                        def _safe_attn(*a: Any, **kw: Any) -> Any:
                            if _is_omni_training_active():
                                return _run_pytorch_sdpa_attention(*a, **kw)
                            return orig_f(*a, **kw)

                        setattr(_safe_attn, "_omni_train_attn_safe", True)
                        setattr(_safe_attn, "_omni_orig_attn_fn", orig_f)
                        if getattr(orig_f, "container_function", None) is not None:
                            setattr(_safe_attn, "container_function", getattr(orig_f, "container_function"))
                        return _safe_attn

                    safe_fn = _make_safe_attn_fn(orig_fn)
                    setattr(comfy_attn, fn_name, safe_fn)
                    replaced_map[id(orig_fn)] = (orig_fn, safe_fn)
                else:
                    replaced_map[id(orig_fn)] = (orig_fn, cur_fn)

            if replaced_map:
                setattr(comfy_attn, "_omni_replaced_attn_map", replaced_map)
                reg_map = getattr(comfy_attn, "REGISTERED_ATTENTION_FUNCTIONS", None)
                if isinstance(reg_map, dict):
                    for reg_k, reg_v in list(reg_map.items()):
                        pair = replaced_map.get(id(reg_v))
                        if pair is not None:
                            reg_map[reg_k] = pair[1]

                for global_attr in ("optimized_attention", "optimized_attention_masked"):
                    g_val = getattr(comfy_attn, global_attr, None)
                    pair = replaced_map.get(id(g_val))
                    if pair is not None:
                        setattr(comfy_attn, global_attr, pair[1])

                # Rebind already-imported `from comfy.ldm.modules.attention import optimized_attention`
                # references across loaded `comfy.*` submodules (e.g. ernie, mmdit, wan, qwen_image21).
                for mod_name, mod_obj in list(sys.modules.items()):
                    if not isinstance(mod_name, str) or not mod_name.startswith("comfy.") or mod_obj is None:
                        continue
                    for attr_n in ("optimized_attention", "optimized_attention_masked"):
                        try:
                            cur_ref = getattr(mod_obj, attr_n, None)
                        except Exception:
                            continue
                        pair = replaced_map.get(id(cur_ref))
                        if pair is not None:
                            try:
                                setattr(mod_obj, attr_n, pair[1])
                            except Exception:
                                pass

            setattr(comfy_attn, "_omni_attn_hooks_installed", True)
    except Exception:
        pass

    try:
        import functools as _ft

        ck_targets: List[Any] = []
        ck_mod = sys.modules.get("comfy_kitchen")
        if ck_mod is not None:
            ck_targets.append(ck_mod)
        qops_mod = sys.modules.get("comfy.quant_ops")
        if qops_mod is not None and getattr(qops_mod, "ck", None) is not None:
            qops_ck = getattr(qops_mod, "ck")
            if qops_ck not in ck_targets:
                ck_targets.append(qops_ck)

        for ck_obj in ck_targets:
            if hasattr(ck_obj, "int8_attention") and not getattr(ck_obj.int8_attention, "_omni_train_safe", False):
                _orig_ck_int8 = getattr(ck_obj, "_omni_orig_int8_attention", ck_obj.int8_attention)
                setattr(ck_obj, "_omni_orig_int8_attention", _orig_ck_int8)

                def _make_ck_int8_wrapper(orig_f: Any) -> Any:
                    def _safe_ck_int8(
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        scale: Optional[float] = None,
                        attn_mask: Optional[torch.Tensor] = None,
                        *args: Any,
                        **kwargs: Any,
                    ) -> torch.Tensor:
                        if _is_omni_training_active():
                            return _sdpa_4d_fallback(q, k, v, scale=scale, attn_mask=attn_mask)
                        return orig_f(q, k, v, scale=scale, attn_mask=attn_mask, *args, **kwargs)

                    setattr(_safe_ck_int8, "_omni_train_safe", True)
                    return _safe_ck_int8

                setattr(ck_obj, "int8_attention", _make_ck_int8_wrapper(_orig_ck_int8))

            if (
                hasattr(ck_obj, "prequantize_int8_attention")
                and hasattr(ck_obj, "int8_attention_from_prequantized")
                and not getattr(ck_obj.prequantize_int8_attention, "_omni_train_safe", False)
            ):
                _orig_preq = getattr(ck_obj, "_omni_orig_prequantize_int8_attention", ck_obj.prequantize_int8_attention)
                _orig_from_preq = getattr(
                    ck_obj, "_omni_orig_int8_attention_from_prequantized", ck_obj.int8_attention_from_prequantized
                )
                setattr(ck_obj, "_omni_orig_prequantize_int8_attention", _orig_preq)
                setattr(ck_obj, "_omni_orig_int8_attention_from_prequantized", _orig_from_preq)

                def _make_preq_pair(orig_p: Any, orig_fp: Any) -> Tuple[Any, Any]:
                    def _safe_preq(
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        scale: Optional[float] = None,
                        attn_mask: Optional[torch.Tensor] = None,
                        *args: Any,
                        **kwargs: Any,
                    ) -> Any:
                        if _is_omni_training_active():
                            return {"_omni_sdpa_qkv": (q, k, v, scale, attn_mask)}
                        return orig_p(q, k, v, scale=scale, attn_mask=attn_mask, *args, **kwargs)

                    def _safe_from_preq(quantized: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                        if isinstance(quantized, dict) and "_omni_sdpa_qkv" in quantized:
                            q, k, v, scale, attn_mask = quantized["_omni_sdpa_qkv"]
                            return _sdpa_4d_fallback(q, k, v, scale=scale, attn_mask=attn_mask)
                        return orig_fp(quantized, *args, **kwargs)

                    setattr(_safe_preq, "_omni_train_safe", True)
                    setattr(_safe_from_preq, "_omni_train_safe", True)
                    return _safe_preq, _safe_from_preq

                safe_preq, safe_from_preq = _make_preq_pair(_orig_preq, _orig_from_preq)
                setattr(ck_obj, "prequantize_int8_attention", safe_preq)
                setattr(ck_obj, "int8_attention_from_prequantized", safe_from_preq)

            if hasattr(ck_obj, "sol_attn") and not getattr(ck_obj.sol_attn, "_omni_train_safe", False):
                _orig_sol = getattr(ck_obj, "_omni_orig_sol_attn", ck_obj.sol_attn)
                setattr(ck_obj, "_omni_orig_sol_attn", _orig_sol)

                def _make_sol_wrapper(orig_f: Any) -> Any:
                    def _safe_sol(
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        tau: float = 1.0,
                        scale: Optional[float] = None,
                        *args: Any,
                        **kwargs: Any,
                    ) -> torch.Tensor:
                        if _is_omni_training_active():
                            base_scale = float(scale) if scale is not None else float(int(q.shape[-1]) ** -0.5)
                            eff_scale = base_scale / max(1e-6, float(tau or 1.0))
                            out_4d = _sdpa_4d_fallback(
                                q.transpose(1, 2),
                                k.transpose(1, 2),
                                v.transpose(1, 2),
                                scale=eff_scale,
                                attn_mask=None,
                            )
                            return out_4d.transpose(1, 2)
                        return orig_f(q, k, v, tau=tau, scale=scale, *args, **kwargs)

                    setattr(_safe_sol, "_omni_train_safe", True)
                    return _safe_sol

                setattr(ck_obj, "sol_attn", _make_sol_wrapper(_orig_sol))

            if hasattr(ck_obj, "rms_rope") and not getattr(ck_obj.rms_rope, "_omni_train_safe", False):
                _orig_rms_rope = getattr(ck_obj, "_omni_orig_rms_rope", ck_obj.rms_rope)
                setattr(ck_obj, "_omni_orig_rms_rope", _orig_rms_rope)

                def _make_rms_rope_wrapper(orig_f: Any) -> Any:
                    def _safe_rms_rope(
                        q: torch.Tensor,
                        k: torch.Tensor,
                        pe: torch.Tensor,
                        q_scale: Optional[torch.Tensor] = None,
                        k_scale: Optional[torch.Tensor] = None,
                        eps: float = 1e-6,
                        *args: Any,
                        **kwargs: Any,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
                        if _is_omni_training_active():
                            eff_eps = float(kwargs.get("epsilon", eps))
                            return _torch_rms_rope_interleaved(q, k, pe, q_scale, k_scale, eps=eff_eps)
                        return orig_f(q, k, pe, q_scale, k_scale, eps, *args, **kwargs)

                    setattr(_safe_rms_rope, "_omni_train_safe", True)
                    return _safe_rms_rope

                setattr(ck_obj, "rms_rope", _make_rms_rope_wrapper(_orig_rms_rope))

            if hasattr(ck_obj, "rms_rope1") and not getattr(ck_obj.rms_rope1, "_omni_train_safe", False):
                _orig_rms_rope1 = getattr(ck_obj, "_omni_orig_rms_rope1", ck_obj.rms_rope1)
                setattr(ck_obj, "_omni_orig_rms_rope1", _orig_rms_rope1)

                def _make_rms_rope1_wrapper(orig_f: Any) -> Any:
                    def _safe_rms_rope1(
                        x: torch.Tensor,
                        pe: torch.Tensor,
                        scale: Optional[torch.Tensor] = None,
                        eps: float = 1e-6,
                        *args: Any,
                        **kwargs: Any,
                    ) -> torch.Tensor:
                        if _is_omni_training_active():
                            eff_eps = float(kwargs.get("epsilon", eps))
                            return _torch_apply_rope_interleaved1(_rms_norm_t(x, scale, eps=eff_eps), pe)
                        return orig_f(x, pe, scale, eps, *args, **kwargs)

                    setattr(_safe_rms_rope1, "_omni_train_safe", True)
                    return _safe_rms_rope1

                setattr(ck_obj, "rms_rope1", _make_rms_rope1_wrapper(_orig_rms_rope1))

            for split_fn_name in ("rms_rope_split_half", "rms_rope_split_half_"):
                if hasattr(ck_obj, split_fn_name) and not getattr(getattr(ck_obj, split_fn_name), "_omni_train_safe", False):
                    _orig_split = getattr(ck_obj, f"_omni_orig_{split_fn_name}", getattr(ck_obj, split_fn_name))
                    setattr(ck_obj, f"_omni_orig_{split_fn_name}", _orig_split)
                    is_inplace_split = split_fn_name.endswith("_")

                    def _make_rms_rope_split_wrapper(orig_f: Any, inplace_flag: bool) -> Any:
                        def _safe_rms_rope_split(
                            q: torch.Tensor,
                            k: torch.Tensor,
                            pe: torch.Tensor,
                            q_scale: Optional[torch.Tensor] = None,
                            k_scale: Optional[torch.Tensor] = None,
                            eps: float = 1e-6,
                            *args: Any,
                            **kwargs: Any,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
                            if _is_omni_training_active():
                                eff_eps = float(kwargs.get("epsilon", eps))
                                rot_dim = kwargs.get("rot_dim", args[0] if len(args) > 0 else None)
                                rq, rk = _torch_rms_rope_split_half(
                                    q, k, pe, q_scale, k_scale, eps=eff_eps, rot_dim=rot_dim
                                )
                                if inplace_flag and not (q.requires_grad or k.requires_grad):
                                    q.copy_(rq)
                                    k.copy_(rk)
                                    return q, k
                                return rq, rk
                            return orig_f(q, k, pe, q_scale, k_scale, eps, *args, **kwargs)

                        setattr(_safe_rms_rope_split, "_omni_train_safe", True)
                        return _safe_rms_rope_split

                    setattr(ck_obj, split_fn_name, _make_rms_rope_split_wrapper(_orig_split, is_inplace_split))

            if hasattr(ck_obj, "apply_rope_split_half") and not getattr(ck_obj.apply_rope_split_half, "_omni_train_safe", False):
                _orig_apply_split = getattr(ck_obj, "_omni_orig_apply_rope_split_half", ck_obj.apply_rope_split_half)
                setattr(ck_obj, "_omni_orig_apply_rope_split_half", _orig_apply_split)

                def _make_apply_rope_split_wrapper(orig_f: Any) -> Any:
                    def _safe_apply_rope_split(
                        xq: torch.Tensor,
                        xk: torch.Tensor,
                        freqs_cis: torch.Tensor,
                        *args: Any,
                        **kwargs: Any,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
                        if _is_omni_training_active():
                            rot_dim = kwargs.get("rot_dim", args[0] if len(args) > 0 else None)
                            return (
                                _torch_apply_rope_split_half1(xq, freqs_cis, rot_dim=rot_dim),
                                _torch_apply_rope_split_half1(xk, freqs_cis, rot_dim=rot_dim),
                            )
                        return orig_f(xq, xk, freqs_cis, *args, **kwargs)

                    setattr(_safe_apply_rope_split, "_omni_train_safe", True)
                    return _safe_apply_rope_split

                setattr(ck_obj, "apply_rope_split_half", _make_apply_rope_split_wrapper(_orig_apply_split))

            if hasattr(ck_obj, "apply_rope") and not getattr(ck_obj.apply_rope, "_omni_train_safe", False):
                _orig_apply_rope = getattr(ck_obj, "_omni_orig_apply_rope", ck_obj.apply_rope)
                setattr(ck_obj, "_omni_orig_apply_rope", _orig_apply_rope)

                def _make_apply_rope_wrapper(orig_f: Any) -> Any:
                    def _safe_apply_rope(
                        xq: torch.Tensor,
                        xk: torch.Tensor,
                        freqs_cis: torch.Tensor,
                        *args: Any,
                        **kwargs: Any,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
                        if _is_omni_training_active():
                            return (
                                _torch_apply_rope_interleaved1(xq, freqs_cis),
                                _torch_apply_rope_interleaved1(xk, freqs_cis),
                            )
                        return orig_f(xq, xk, freqs_cis, *args, **kwargs)

                    setattr(_safe_apply_rope, "_omni_train_safe", True)
                    return _safe_apply_rope

                setattr(ck_obj, "apply_rope", _make_apply_rope_wrapper(_orig_apply_rope))

            if hasattr(ck_obj, "apply_rope1") and not getattr(ck_obj.apply_rope1, "_omni_train_safe", False):
                _orig_apply_rope1 = getattr(ck_obj, "_omni_orig_apply_rope1", ck_obj.apply_rope1)
                setattr(ck_obj, "_omni_orig_apply_rope1", _orig_apply_rope1)

                def _make_apply_rope1_wrapper(orig_f: Any) -> Any:
                    def _safe_apply_rope1(
                        x: torch.Tensor,
                        freqs_cis: torch.Tensor,
                        *args: Any,
                        **kwargs: Any,
                    ) -> torch.Tensor:
                        if _is_omni_training_active():
                            return _torch_apply_rope_interleaved1(x, freqs_cis)
                        return orig_f(x, freqs_cis, *args, **kwargs)

                    setattr(_safe_apply_rope1, "_omni_train_safe", True)
                    return _safe_apply_rope1

                setattr(ck_obj, "apply_rope1", _make_apply_rope1_wrapper(_orig_apply_rope1))

            if hasattr(ck_obj, "adaln") and not getattr(ck_obj.adaln, "_omni_train_safe", False):
                _orig_adaln = getattr(ck_obj, "_omni_orig_adaln", ck_obj.adaln)
                setattr(ck_obj, "_omni_orig_adaln", _orig_adaln)

                def _make_adaln_wrapper(orig_f: Any) -> Any:
                    def _safe_adaln(
                        x: torch.Tensor,
                        scale: torch.Tensor,
                        shift: torch.Tensor,
                        eps: float = 1e-6,
                        *args: Any,
                        **kwargs: Any,
                    ) -> torch.Tensor:
                        if _is_omni_training_active():
                            normed = F.layer_norm(x, (int(x.shape[-1]),), eps=float(eps or 1e-6))
                            return torch.addcmul(shift, normed, 1.0 + scale)
                        return orig_f(x, scale, shift, eps, *args, **kwargs)

                    setattr(_safe_adaln, "_omni_train_safe", True)
                    return _safe_adaln

                setattr(ck_obj, "adaln", _make_adaln_wrapper(_orig_adaln))

            if hasattr(ck_obj, "rms_adaln") and not getattr(ck_obj.rms_adaln, "_omni_train_safe", False):
                _orig_rms_adaln = getattr(ck_obj, "_omni_orig_rms_adaln", ck_obj.rms_adaln)
                setattr(ck_obj, "_omni_orig_rms_adaln", _orig_rms_adaln)

                def _make_rms_adaln_wrapper(orig_f: Any) -> Any:
                    def _safe_rms_adaln(
                        x: torch.Tensor,
                        scale: torch.Tensor,
                        shift: torch.Tensor,
                        eps: float = 1e-6,
                        *args: Any,
                        **kwargs: Any,
                    ) -> torch.Tensor:
                        if _is_omni_training_active():
                            normed = _rms_norm_t(x, None, eps=float(eps or 1e-6))
                            return torch.addcmul(shift, normed, 1.0 + scale)
                        return orig_f(x, scale, shift, eps, *args, **kwargs)

                    setattr(_safe_rms_adaln, "_omni_train_safe", True)
                    return _safe_rms_adaln

                setattr(ck_obj, "rms_adaln", _make_rms_adaln_wrapper(_orig_rms_adaln))
    except Exception:
        pass


def _rebind_comfy_attention_modules(root: Optional[nn.Module]) -> None:
    """
    Ensure any pre-loaded `ComfyAttention` submodule on `root` whose `.function` (or
    `functools.partial.func`) bound an unpatched `attention_comfy_kitchen_*` / `attention_sage`
    callable before `install_comfy_runtime_hooks()` is rebound to the training-safe wrapper.
    """
    if root is None or not isinstance(root, nn.Module):
        return
    comfy_attn = sys.modules.get("comfy.ldm.modules.attention")
    replaced_map = getattr(comfy_attn, "_omni_replaced_attn_map", None) if comfy_attn is not None else None
    if not isinstance(replaced_map, dict) or not replaced_map:
        return
    import functools

    for mod in root.modules():
        candidates = [mod]
        ca = getattr(mod, "comfy_attention", None)
        if ca is not None and ca is not mod:
            candidates.append(ca)
        for obj in candidates:
            fn = getattr(obj, "function", None)
            if fn is None or getattr(fn, "_omni_train_attn_safe", False):
                continue
            pair = replaced_map.get(id(fn))
            if pair is not None:
                try:
                    setattr(obj, "function", pair[1])
                except Exception:
                    pass
            elif isinstance(fn, functools.partial):
                inner_pair = replaced_map.get(id(fn.func))
                if inner_pair is not None:
                    try:
                        new_partial = functools.partial(inner_pair[1], *fn.args, **(fn.keywords or {}))
                        setattr(new_partial, "_omni_train_attn_safe", True)
                        setattr(obj, "function", new_partial)
                    except Exception:
                        pass


def unwrap_comfy_diffusion_model(
    comfy_model: Any,
    load_to_gpu: bool = False,
    memory_required_mb: float = 4608.0,
) -> Optional[nn.Module]:
    """Unwrap a ComfyUI `ModelPatcher` (`MODEL`) to extract the underlying `diffusion_model` (`nn.Module`)."""
    install_comfy_runtime_hooks()
    if comfy_model is None:
        return None

    # Only call comfy.model_management.load_models_gpu on an actual ModelPatcher when explicitly requested,
    # passing `memory_required` so ComfyUI reserves VRAM headroom for activations and optimizer states.
    if load_to_gpu and not isinstance(comfy_model, nn.Module) and hasattr(comfy_model, "model"):
        try:
            import comfy.model_management as mm  # type: ignore

            mem_req_bytes = int(max(2048.0, float(memory_required_mb)) * (1024**2))
            try:
                mm.load_models_gpu([comfy_model], memory_required=mem_req_bytes)
            except TypeError:
                mm.load_models_gpu([comfy_model])
        except Exception:
            pass

    candidates: List[Any] = []
    if hasattr(comfy_model, "get_model_object") and callable(comfy_model.get_model_object):
        try:
            dm = comfy_model.get_model_object("diffusion_model")
            if dm is not None:
                candidates.append(dm)
        except Exception:
            pass

    if hasattr(comfy_model, "model"):
        inner = comfy_model.model
        if hasattr(inner, "diffusion_model"):
            candidates.append(inner.diffusion_model)
        if hasattr(inner, "transformer"):
            candidates.append(inner.transformer)
        candidates.append(inner)

    if hasattr(comfy_model, "diffusion_model"):
        candidates.append(comfy_model.diffusion_model)
    if hasattr(comfy_model, "transformer"):
        candidates.append(comfy_model.transformer)
    candidates.append(comfy_model)

    for cand in candidates:
        if isinstance(cand, nn.Module):
            has_children = any(True for _ in cand.children())
            has_params = any(True for _ in cand.parameters())
            if has_children or has_params:
                _rebind_comfy_attention_modules(cand)
                LoRAInjector.capture_comfy_lora_key_map(cand, comfy_model=comfy_model)
                return cand

    return None


_COMFY_SUPPORTED_MODEL_CANONICAL_MAP: Dict[str, str] = {
    "mossimage": "moss_image",
    "krea2": "krea2",
    "flux3": "flux3",
    "flux3klein": "flux3",
    "mockflux3klein": "flux3",
    "flux2": "flux2",
    "flux": "flux_dev",
    "fluxschnell": "flux_dev",
    "fluxinpaint": "flux_dev",
    "fibo2": "fibo2",
    "chroma": "chroma",
    "chromaradiance": "chroma_radiance",
    "hidream": "hidream",
    "hidreamo1": "hidream_o1",
    "sd3": "sd3_5",
    "sdxl": "sdxl_base",
    "sdxlrefiner": "sdxl_base",
    "sdxl_instructpix2pix": "sdxl_base",
    "ssd1b": "sdxl_base",
    "sd15": "sd15",
    "sd15_instructpix2pix": "sd15",
    "sd20": "sd15",
    "sd21unclipl": "sd15",
    "sd21uncliph": "sd15",
    "qwenimage21": "qwen_image21",
    "qwenimage": "qwen_image",
    "lumina2": "lumina2",
    "zimage": "lumina2",
    "zimagepixelspace": "lumina2",
    "omnigen2": "omnigen2",
    "auraflow": "auraflow",
    "pixartalpha": "pixart",
    "pixartsigma": "pixart",
    "hunyuandit": "hunyuan_dit",
    "hunyuandit1": "hunyuan_dit",
    "stable_cascade_c": "stable_cascade",
    "stable_cascade_b": "stable_cascade",
    "stablecascade_c": "stable_cascade",
    "stablecascade_b": "stable_cascade",
    "ideogram4": "ideogram4",
    "boogu": "boogu",
    "joyimage": "joyimage",
    "mageflow": "mage_flow",
    "anima": "anima",
    "pixelditt2i": "pixeldit",
    "pid": "pixeldit",
    "pixeldit": "pixeldit",
    "ernieimage": "ernie",
    "lens": "lens",
    "wan22_t2v": "wan22",
    "wan22_camera": "wan22",
    "wan22_s2v": "wan22",
    "wan22_animate": "wan22",
    "wan22_relayvsr": "wan22",
    "wan22_wandancer": "wan22",
    "wan22": "wan22",
    "wan21_t2v": "wan_t2v",
    "wan21_i2v": "wan_t2v",
    "wan21_funcontrol2v": "wan_t2v",
    "wan21_vace": "wan_t2v",
    "wan21_camera": "wan_t2v",
    "wan21_humo": "wan_t2v",
    "wan21_causalar_t2v": "wan_t2v",
    "wan21": "wan_t2v",
    "hunyuanvideo": "hunyuan_video",
    "hunyuanvideoi2v": "hunyuan_video",
    "hunyuanvideo15": "hunyuan_video",
    "hunyuanvideo15_sr_distilled": "hunyuan_video",
    "hunyuanvideoskyreelsi2v": "hunyuan_video",
    "hunyuanimage21": "hunyuan_video",
    "ltxv": "ltx_video",
    "ltxav": "ltxav",
    "cosmost2v": "cosmos",
    "cosmosi2v": "cosmos",
    "cosmost2ipredict2": "cosmos",
    "cosmosi2vpredict2": "cosmos",
    "cosmosvideo": "cosmos",
    "cosmospredict2": "cosmos",
    "kandinsky5": "kandinsky5",
    "kandinsky5image": "kandinsky5",
    "cogvideox": "cogvideo",
    "cogvideox_t2v": "cogvideo",
    "cogvideox_i2v": "cogvideo",
    "cogvideox_inpaint": "cogvideo",
    "genmomochi": "genmo_mochi",
    "acestep": "ace_step",
    "acestep15": "ace_step15",
    "stableaudio": "mmaudio",
    "stableaudio3": "mmaudio",
    "yue2": "yue2",
    "hunyuan3dv2": "hunyuan3d",
    "hunyuan3dv2_1": "hunyuan3dv2_1",
    "hunyuan3dv2mini": "hunyuan3d",
    "sensenovau15": "sensenova",
    "sensenova": "sensenova",
    "minimaxh3": "minimax",
    "minimax": "minimax",
    "minimaxmusic3": "minimax_music",
    "minimax_music": "minimax_music",
    "seedvr2": "seedvr",
    "seedvr": "seedvr",
    "trellis2": "trellis2",
    "triposplat": "triposplat",
}


_COMFY_LDM_MODULE_TO_CANONICAL: Dict[str, str] = {
    "comfy.ldm.moss_image": "moss_image",
    "comfy.ldm.krea2": "krea2",
    "comfy.ldm.flux3": "flux3",
    "comfy.ldm.flux": "flux_dev",
    "comfy.ldm.fibo2": "fibo2",
    "comfy.ldm.chroma_radiance": "chroma_radiance",
    "comfy.ldm.chroma": "chroma",
    "comfy.ldm.hidream_o1": "hidream_o1",
    "comfy.ldm.hidream": "hidream",
    "comfy.ldm.qwen_image21": "qwen_image21",
    "comfy.ldm.qwen_image": "qwen_image",
    "comfy.ldm.lumina": "lumina2",
    "comfy.ldm.omnigen": "omnigen2",
    "comfy.ldm.aura": "auraflow",
    "comfy.ldm.wan": "wan_t2v",
    "comfy.ldm.hunyuan_video": "hunyuan_video",
    "comfy.ldm.lightricks.av_model": "ltxav",
    "comfy.ldm.lightricks": "ltx_video",
    "comfy.ldm.cosmos": "cosmos",
    "comfy.ldm.ideogram4": "ideogram4",
    "comfy.ldm.boogu": "boogu",
    "comfy.ldm.joyimage": "joyimage",
    "comfy.ldm.mage_flow": "mage_flow",
    "comfy.ldm.anima": "anima",
    "comfy.ldm.pixeldit": "pixeldit",
    "comfy.ldm.ernie": "ernie",
    "comfy.ldm.lens": "lens",
    "comfy.ldm.kandinsky5": "kandinsky5",
    "comfy.ldm.cogvideo": "cogvideo",
    "comfy.ldm.genmo": "genmo_mochi",
    "comfy.ldm.ace.ace_step15": "ace_step15",
    "comfy.ldm.ace": "ace_step",
    "comfy.ldm.mmaudio": "mmaudio",
    "comfy.ldm.audio": "mmaudio",
    "comfy.ldm.yue2": "yue2",
    "comfy.ldm.hunyuan3dv2_1": "hunyuan3dv2_1",
    "comfy.ldm.hunyuan3d": "hunyuan3d",
    "comfy.ldm.hydit": "hunyuan_dit",
    "comfy.ldm.hunyuandit": "hunyuan_dit",
    "comfy.ldm.pixart": "pixart",
    "comfy.ldm.cascade": "stable_cascade",
    "comfy.ldm.sensenova": "sensenova",
    "comfy.ldm.minimax_music": "minimax_music",
    "comfy.ldm.minimax": "minimax",
    "comfy.ldm.seedvr": "seedvr",
    "comfy.ldm.trellis2": "trellis2",
    "comfy.ldm.triposplat": "triposplat",
    "comfy.ldm.modules.diffusionmodules.mmdit": "sd3_5",
    "comfy.ldm.modules.diffusionmodules.openaimodel": "sdxl_base",
}


def resolve_comfy_native_architecture(
    comfy_model_or_backbone: Any,
    unet_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Directly query ComfyUI's native runtime modules (`comfy.model_base`,
    `comfy.supported_models`, `comfy.model_detection`, and `comfy.latent_formats`)
    to identify the exact model architecture, `model_config`, `unet_config`,
    `latent_format`, `model_type`, and `model_sampling` without string guessing.
    """
    ensure_comfy_importable()

    result: Dict[str, Any] = {
        "canonical_id": None,
        "comfy_model_base_class": None,
        "comfy_supported_model_class": None,
        "comfy_ldm_class": None,
        "model_config": None,
        "unet_config": dict(unet_config) if isinstance(unet_config, dict) else {},
        "latent_format": None,
        "model_type": None,
        "model_sampling": None,
    }
    if comfy_model_or_backbone is None:
        return result

    inner_model = getattr(comfy_model_or_backbone, "model", comfy_model_or_backbone)
    backbone = getattr(inner_model, "diffusion_model", None)
    if backbone is None and isinstance(comfy_model_or_backbone, nn.Module):
        backbone = comfy_model_or_backbone

    model_cfg = (
        getattr(inner_model, "model_config", None)
        or getattr(comfy_model_or_backbone, "model_config", None)
    )

    # 1. Inspect `backbone` (`comfy.ldm.*` class identity)
    if isinstance(backbone, nn.Module):
        b_mod = getattr(backbone.__class__, "__module__", "")
        b_cls = backbone.__class__.__name__
        result["comfy_ldm_class"] = f"{b_mod}.{b_cls}"
        for mod_prefix, canon_id in _COMFY_LDM_MODULE_TO_CANONICAL.items():
            if b_mod == mod_prefix or b_mod.startswith(f"{mod_prefix}."):
                if mod_prefix == "comfy.ldm.hidream" and "hidreamo1" in b_cls.lower():
                    result["canonical_id"] = "hidream_o1"
                elif mod_prefix == "comfy.ldm.modules.diffusionmodules.openaimodel":
                    # Distinguish SD1.5 (no adm/label_emb) vs SDXL (num_classes == "sequential")
                    result["canonical_id"] = (
                        "sdxl_base" if getattr(backbone, "num_classes", None) == "sequential" else "sd15"
                    )
                elif mod_prefix == "comfy.ldm.wan":
                    in_d = getattr(backbone, "in_dim", getattr(getattr(backbone, "patch_embedding", None), "in_channels", 16))
                    result["canonical_id"] = "wan22" if int(in_d or 16) == 48 else "wan_t2v"
                else:
                    result["canonical_id"] = canon_id
                break

    # 2. If a bare `diffusion_model` (`nn.Module`) was provided without `model_config`,
    # invoke ComfyUI's native `comfy.model_detection.model_config_from_unet` or match `comfy.supported_models.models`.
    if model_cfg is None and isinstance(backbone, nn.Module):
        try:
            import comfy.model_detection as _cmd  # type: ignore

            sd = backbone.state_dict()
            if sd:
                detected_cfg = _cmd.model_config_from_unet(sd, "", use_base_if_no_match=False)
                if detected_cfg is not None:
                    model_cfg = detected_cfg
        except Exception:
            pass

        if model_cfg is None and result["canonical_id"] is not None:
            try:
                import comfy.supported_models as _csm  # type: ignore

                target_canon = result["canonical_id"]
                for sm_cls in getattr(_csm, "models", []):
                    sm_name_l = sm_cls.__name__.lower()
                    if _COMFY_SUPPORTED_MODEL_CANONICAL_MAP.get(sm_name_l) == target_canon:
                        base_ucfg = dict(getattr(sm_cls, "unet_config", {}) or {})
                        base_ucfg.update(result["unet_config"])
                        base_ucfg["disable_unet_model_creation"] = True
                        model_cfg = sm_cls(base_ucfg)
                        break
            except Exception:
                pass

    if model_cfg is not None:
        result["model_config"] = model_cfg
        cfg_cls_name = model_cfg.__class__.__name__
        result["comfy_supported_model_class"] = f"{model_cfg.__class__.__module__}.{cfg_cls_name}"
        raw_ucfg = getattr(model_cfg, "unet_config", None)
        if isinstance(raw_ucfg, dict) and raw_ucfg:
            merged_ucfg = dict(raw_ucfg)
            merged_ucfg.update(result["unet_config"])
            result["unet_config"] = merged_ucfg
        lf = getattr(model_cfg, "latent_format", None)
        if lf is not None:
            result["latent_format"] = lf
            if isinstance(lf, type):
                try:
                    result["latent_format"] = lf()
                except Exception:
                    pass
        if result["model_type"] is None and hasattr(model_cfg, "model_type") and callable(model_cfg.model_type):
            try:
                result["model_type"] = model_cfg.model_type({})
            except Exception:
                pass
        mapped = _COMFY_SUPPORTED_MODEL_CANONICAL_MAP.get(cfg_cls_name.lower())
        if mapped:
            result["canonical_id"] = mapped

    # 3. Inspect `inner_model` against `comfy.model_base.BaseModel` subclasses & `ModelType`
    try:
        import comfy.model_base as _cmb  # type: ignore

        if isinstance(inner_model, _cmb.BaseModel):
            mb_cls_name = inner_model.__class__.__name__
            result["comfy_model_base_class"] = f"{inner_model.__class__.__module__}.{mb_cls_name}"
            if getattr(inner_model, "model_type", None) is not None:
                result["model_type"] = inner_model.model_type
            if getattr(inner_model, "latent_format", None) is not None:
                result["latent_format"] = inner_model.latent_format
            if getattr(inner_model, "model_sampling", None) is not None:
                result["model_sampling"] = inner_model.model_sampling
            if result["canonical_id"] is None:
                mapped_mb = _COMFY_SUPPORTED_MODEL_CANONICAL_MAP.get(mb_cls_name.lower())
                if mapped_mb:
                    result["canonical_id"] = mapped_mb

        if result["model_sampling"] is None and model_cfg is not None:
            m_type = result["model_type"] if result["model_type"] is not None else _cmb.ModelType.EPS
            result["model_sampling"] = _cmb.model_sampling(model_cfg, m_type)
    except Exception:
        pass

    # 4. Check `unet_config["image_model"]` or `unet_config["audio_model"]` populated by ComfyUI's `supported_models.py`
    ucfg = result["unet_config"]
    if result["canonical_id"] is None and isinstance(ucfg, dict):
        for k_field in ("image_model", "audio_model"):
            val = ucfg.get(k_field)
            if isinstance(val, str) and val.strip():
                norm_val = val.strip().lower()
                result["canonical_id"] = _COMFY_SUPPORTED_MODEL_CANONICAL_MAP.get(norm_val, norm_val)
                break

    return result


def detect_comfy_model_architecture(
    comfy_model_or_backbone: Any,
    unet_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Inspect a ComfyUI `ModelPatcher`, `comfy.model_base.BaseModel`, `comfy.supported_models` config,
    `unet_config`, or `comfy.ldm.*` `diffusion_model` via `resolve_comfy_native_architecture`
    and return its canonical architecture ID.
    """
    if comfy_model_or_backbone is None and not unet_config:
        return "auto"

    native_info = resolve_comfy_native_architecture(comfy_model_or_backbone, unet_config=unet_config)
    if isinstance(native_info.get("canonical_id"), str) and native_info["canonical_id"]:
        return str(native_info["canonical_id"])

    if comfy_model_or_backbone is None:
        return "auto"

    backbone = (
        comfy_model_or_backbone
        if isinstance(comfy_model_or_backbone, nn.Module)
        else (unwrap_comfy_diffusion_model(comfy_model_or_backbone, load_to_gpu=False) or comfy_model_or_backbone)
    )
    cls_name = backbone.__class__.__name__.lower()
    if cls_name.startswith("mock"):
        unmocked = cls_name[4:]
        if unmocked in _COMFY_SUPPORTED_MODEL_CANONICAL_MAP:
            return _COMFY_SUPPORTED_MODEL_CANONICAL_MAP[unmocked]
    return _COMFY_SUPPORTED_MODEL_CANONICAL_MAP.get(cls_name, cls_name or "auto")


def is_optimal_lora_target_path(
    module_name: str,
    structural_roles: Optional[Dict[str, str]] = None,
    backbone: Optional[nn.Module] = None,
    model: Optional[nn.Module] = None,
) -> bool:
    """
    Return True if `module_name` is an optimal LoRA target (Self/Cross/Joint Attention,
    FeedForward/MLP/SwiGLU/MoE-Expert, or repeating-block Linear / 1x1 `block_projection`
    such as SpatialTransformer `proj_in` / `proj_out`), evaluated structurally via
    `classify_block_structural_roles`:
      - Returns True for `"attention"`, `"ffn"`, `"attention_or_ffn"`, and `"block_projection"` roles.
      - Returns False for `"stem_or_head"`, `"modulation"` (AdaLN / timestep modulations),
        and `"moe_router"` (`ff_i.gate`) roles.
    """
    if not module_name or not _strip_numeric_segments(module_name.replace("__", ".")):
        return False

    if backbone is None and model is not None:
        backbone = model

    if structural_roles is None and isinstance(backbone, nn.Module):
        structural_roles = classify_block_structural_roles(backbone)

    if structural_roles is not None:
        role = structural_roles.get(module_name)
        if role is None:
            norm_name = module_name.replace("__", ".")
            for path, r in structural_roles.items():
                if LoRAInjector._matches_target(path, (norm_name,)):
                    role = r
                    break
        if role is not None:
            return role in ("attention", "ffn", "attention_or_ffn", "block_projection")
        if backbone is not None:
            return False

    return True


def discover_universal_target_modules(
    backbone: nn.Module,
    comfy_inner_model: Optional[Any] = None,
    include_conv: bool = False,
) -> Tuple[str, ...]:
    """
    Universal ComfyUI LoRA target discovery:
    1. Queries `comfy.lora.model_lora_keys_unet` (if available) so any module exposed by ComfyUI's
       own LoRA loader (including tuple-sliced keys like QwenImage21's `img_mlp.gate_up`) is recognized.
    2. Classifies every Linear-like and pointwise 1x1 `block_projection` module by its structural role
       in the residual stream topology via `classify_block_structural_roles(backbone)` (distinguishing
       Attention, FFN/MoE-Expert, and block/skip projections like SpatialTransformer `proj_in`/`proj_out`
       from AdaLN modulations, MoE router gates `ff_i.gate`, and top-level stems/heads without string guessing).
    3. Strips numeric block indices (`layers.0.proj_qkv` -> `layers.proj_qkv`) so all Attention,
       MLP/expert, and block projections inside repeating transformer/UNet blocks are adapted by default.
    4. When `include_conv=True` (LoCon / Conv-LoRA mode), also discovers spatial ResNet/Downsample/Upsample
       Convolution layers inside repeating blocks (e.g., `in_layers.2`, `out_layers.3`, `skip_connection`,
       `op`, `conv`), while still excluding raw input/output stems.
    """
    comfy_valid_paths: Optional[set[str]] = None
    if comfy_inner_model is not None:
        try:
            import comfy.lora  # type: ignore

            if hasattr(comfy.lora, "model_lora_keys_unet"):
                key_map = comfy.lora.model_lora_keys_unet(comfy_inner_model, {})
                paths: set[str] = set()
                for v in key_map.values():
                    raw_k = v[0] if isinstance(v, tuple) and len(v) > 0 else v
                    if isinstance(raw_k, str) and raw_k.endswith(".weight"):
                        clean_w = raw_k.replace(".base_layer.weight", ".weight")
                        paths.add(clean_w[: -len(".weight")].removeprefix("diffusion_model."))
                if paths:
                    comfy_valid_paths = paths
        except Exception:
            comfy_valid_paths = None

    block_module_ids = _collect_repeating_block_module_ids(backbone)
    structural_roles = classify_block_structural_roles(backbone)

    discovered: List[str] = []
    seen: set[str] = set()

    for _, _, name, module in LoRAInjector._iter_injectable_modules(backbone):
        if not name:
            continue
        if comfy_valid_paths is not None and name not in comfy_valid_paths:
            continue
        if block_module_ids and id(module) not in block_module_ids:
            continue
        is_pointwise_block_proj = (
            _is_pointwise_conv(module) and structural_roles.get(name) == "block_projection"
        )
        if (
            _is_linear_like(module)
            or (isinstance(module, LoRALinear) and not getattr(module, "is_conv", False))
            or is_pointwise_block_proj
        ):
            if not is_optimal_lora_target_path(name, structural_roles=structural_roles):
                continue
            stripped = _strip_numeric_segments(name)
            if stripped and stripped not in seen:
                seen.add(stripped)
                discovered.append(stripped)

    if include_conv:
        for _, _, name, module in LoRAInjector._iter_injectable_modules(backbone):
            if not name:
                continue
            if comfy_valid_paths is not None and name not in comfy_valid_paths:
                continue
            if block_module_ids and id(module) not in block_module_ids:
                continue
            is_conv_mod = _is_conv_like(module) or (isinstance(module, LoRALinear) and getattr(module, "is_conv", False))
            if is_conv_mod:
                stripped = _strip_numeric_segments(name)
                if (
                    stripped
                    and stripped.lower() not in (_UNET_BLOCK_CONTAINER_NAMES | {"out", "conv_in", "conv_out"})
                    and stripped not in seen
                ):
                    seen.add(stripped)
                    discovered.append(stripped)

    if not discovered:
        for _, _, name, module in LoRAInjector._iter_injectable_modules(backbone):
            if _is_linear_like(module) and name:
                if not is_optimal_lora_target_path(name, structural_roles=structural_roles):
                    continue
                stripped = _strip_numeric_segments(name)
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    discovered.append(stripped)

    if not discovered:
        for _, _, name, module in LoRAInjector._iter_injectable_modules(backbone):
            if (_is_linear_like(module) or _is_conv_like(module)) and name:
                stripped = _strip_numeric_segments(name)
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    discovered.append(stripped)

    return tuple(discovered)


def resolve_model_adapter(
    model_id: str = "auto",
    comfy_model: Optional[Any] = None,
    **kwargs: Any,
) -> ComfyModelAdapter:
    """Instantiate a universal ComfyModelAdapter driven by the connected ComfyUI MODEL."""
    return ComfyModelAdapter(comfy_model=comfy_model, model_id=model_id, **kwargs)


# Backwards-compatible aliases pointing to the single universal ComfyModelAdapter
UniversalModelAdapter = ComfyModelAdapter
MossImageAdapter = ComfyModelAdapter
FluxModelAdapter = ComfyModelAdapter
SD3ModelAdapter = ComfyModelAdapter
SDXLModelAdapter = ComfyModelAdapter
SD15ModelAdapter = ComfyModelAdapter
WanVideoAdapter = ComfyModelAdapter
HunyuanVideoAdapter = ComfyModelAdapter
LTXVideoAdapter = ComfyModelAdapter
QwenImageAdapter = ComfyModelAdapter


class _DynamicModelSpecDict(dict):  # type: ignore[type-arg]
    """On-demand ModelSpec lookup that works for any model identifier without hardcoded tables."""

    def __missing__(self, key: str) -> ModelSpec:
        return ComfyModelAdapter(model_id=str(key)).spec


MODEL_REGISTRY: Dict[str, Type[ComfyModelAdapter]] = {
    "auto": ComfyModelAdapter,
    "comfy_model": ComfyModelAdapter,
    "moss_image": ComfyModelAdapter,
    "krea2": ComfyModelAdapter,
    "flux3": ComfyModelAdapter,
    "flux2": ComfyModelAdapter,
    "flux_dev": ComfyModelAdapter,
    "fibo2": ComfyModelAdapter,
    "hidream": ComfyModelAdapter,
    "hidream_o1": ComfyModelAdapter,
    "chroma": ComfyModelAdapter,
    "chroma_radiance": ComfyModelAdapter,
    "sd3_5": ComfyModelAdapter,
    "sdxl_base": ComfyModelAdapter,
    "sd15": ComfyModelAdapter,
    "wan_t2v": ComfyModelAdapter,
    "wan21": ComfyModelAdapter,
    "wan22": ComfyModelAdapter,
    "hunyuan_video": ComfyModelAdapter,
    "ltx_video": ComfyModelAdapter,
    "qwen_image": ComfyModelAdapter,
    "qwen_image21": ComfyModelAdapter,
    "lumina2": ComfyModelAdapter,
    "omnigen2": ComfyModelAdapter,
    "cosmos": ComfyModelAdapter,
    "auraflow": ComfyModelAdapter,
    "ideogram4": ComfyModelAdapter,
    "boogu": ComfyModelAdapter,
    "joyimage": ComfyModelAdapter,
    "mage_flow": ComfyModelAdapter,
    "anima": ComfyModelAdapter,
    "pixeldit": ComfyModelAdapter,
    "ernie": ComfyModelAdapter,
    "lens": ComfyModelAdapter,
    "kandinsky5": ComfyModelAdapter,
    "cogvideo": ComfyModelAdapter,
    "genmo_mochi": ComfyModelAdapter,
    "ace_step": ComfyModelAdapter,
    "ace_step15": ComfyModelAdapter,
    "mmaudio": ComfyModelAdapter,
    "yue2": ComfyModelAdapter,
    "hunyuan3d": ComfyModelAdapter,
    "hunyuan3dv2_1": ComfyModelAdapter,
    "pixart": ComfyModelAdapter,
    "hunyuan_dit": ComfyModelAdapter,
    "stable_cascade": ComfyModelAdapter,
    "sensenova": ComfyModelAdapter,
    "minimax": ComfyModelAdapter,
    "minimax_music": ComfyModelAdapter,
    "seedvr": ComfyModelAdapter,
    "trellis2": ComfyModelAdapter,
    "triposplat": ComfyModelAdapter,
    "ltxav": ComfyModelAdapter,
}

MODEL_SPECS: Dict[str, ModelSpec] = _DynamicModelSpecDict(
    {k: ComfyModelAdapter(model_id=k).spec for k in MODEL_REGISTRY}
)
