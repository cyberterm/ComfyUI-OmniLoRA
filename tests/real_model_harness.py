"""
Unmocked Real-Forward Model Harness & 7-Invariant Contract Verifier for OmniLoRA.

Unlike `comfy_emulator.py`'s bridged zoo (which replaces `diffusion_model.forward` with a
synthetic per-layer probe for fast plumbing sweeps), this harness NEVER monkey-patches
`diffusion_model.forward` or `base_model.apply_model`:

  1. Instantiates genuine `comfy.ldm.*` backbones wrapped in their genuine `comfy.model_base.*`
     subclasses (`extra_conds`, `encode_adm`, `process_latent_in`, `process_timestep`, `apply_model`).
  2. Uses realistic per-architecture `_ContractCLIP` and `_ContractVAE` fixtures that emit real
     multi-width / connector / extra-cond payloads (e.g. LTX 2.5 `cross_attention_dim +
     audio_cross_attention_dim` embeddings, `unprocessed_ltxav_embeds=True`, MiniMax H3
     `text_token_tags` + 24ch video latents, Anima `t5xxl_ids` + `t5xxl_weights`) with unequal
     caption sequence lengths across batch items so `batch_size=2` exercises `attention_mask`
     synthesis, padding/trimming, and post-connector length reconciliation.
  3. Attaches `StrictComfySpy` so ANY silent exception inside `extra_conds`, `process_latent_in`,
     or `process_timestep` fails the test immediately instead of passing via silent fallback.
  4. Enforces 7 hard invariants per architecture:
       (1) Full real-forward pipeline with `batch_size=1` and `batch_size=2` (unequal prompts)
           inside `torch.inference_mode()` with zero silent conditioning/scaling failures and
           strict `comfy.model_management.in_training` hygiene.
       (2) 100% non-zero LoRA gradient coverage (zero dead LoRA adapters).
       (3) Exact gradient parity between `gradient_checkpointing=True` and `False`.
       (4) FP8 `get_dtype_inference()` resilience on `preprocess_text_embeds` models.
       (5) Dirty post-KSampler `inner_model.latent_shapes` isolation on multimodal AV models.
       (6) Clean uninject with bit-tight baseline real-forward restoration.
       (7) Native `comfy.sd.load_lora_for_models` zero-missing-keys round-trip + real patched
           forward delta verification.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import functools
import json
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from omni_lora.models.adapters import ensure_comfy_importable

if not ensure_comfy_importable():
    raise RuntimeError(
        "ComfyUI installation not found on sys.path. Clone ComfyUI adjacent to omni_lora "
        "or place omni_lora inside ComfyUI/custom_nodes/omni_lora."
    )

import comfy.latent_formats
import comfy.lora
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.ops
import comfy.sd
import comfy.supported_models
import comfy.supported_models_base
import comfy.utils


# ==============================================================================
# 1. Realistic Contract Fixtures (VAE, CLIP, Weight Init, Unmocked Patcher)
# ==============================================================================

def _initialize_real_comfy_module(module: nn.Module, seed: int = 1234) -> None:
    """
    Deterministically initialize `torch.empty`-allocated ComfyUI `manual_cast` parameters
    WITHOUT destroying valid precomputed buffers (such as RoPE `inv_freq`, `freqs_cis`,
    sinusoidal tables, or causal masks).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    norm_types = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    rms_norm_cls = getattr(nn, "RMSNorm", None)
    if rms_norm_cls is not None:
        norm_types = norm_types + (rms_norm_cls,)

    norm_param_ids: set[int] = set()
    for sub in module.modules():
        cls_name = sub.__class__.__name__.lower()
        if isinstance(sub, norm_types) or "rmsnorm" in cls_name or "layernorm" in cls_name or "groupnorm" in cls_name:
            w = getattr(sub, "weight", None)
            if isinstance(w, nn.Parameter):
                norm_param_ids.add(id(w))

    with torch.no_grad():
        for name, param in module.named_parameters():
            if getattr(param, "is_meta", False) or not param.is_floating_point():
                continue
            low = name.lower()
            if id(param) in norm_param_ids:
                param.fill_(1.0)
            elif low.endswith(".bias") or low == "bias":
                param.zero_()
            else:
                vals = torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.02
                param.copy_(vals.to(device=param.device, dtype=param.dtype))

        # Sanitize uninitialized/non-finite buffers (e.g. `rope.inv_freq` and `adaln_t_table`
        # allocated via `torch.empty` in MiniMax H3 / H3-Image) while preserving precomputed buffers.
        for name, buf in module.named_buffers():
            if getattr(buf, "is_meta", False) or not buf.is_floating_point():
                continue
            low = name.lower()
            if "adaln_t_table" in low:
                vals = torch.randn(buf.shape, generator=gen, dtype=torch.float32) * 0.5
                buf.copy_(vals.to(device=buf.device, dtype=buf.dtype))
                continue
            is_uninit_rope = "inv_freq" in low and (
                not torch.isfinite(buf).all()
                or float(buf.abs().max().item()) > 1e4
                or float(buf.abs().max().item()) < 1e-6
            )
            if is_uninit_rope:
                if buf.ndim == 1 and buf.numel() > 0:
                    idx = torch.arange(buf.numel(), dtype=torch.float32)
                    freqs = 1.0 / (10000.0 ** (idx / max(1.0, float(buf.numel()))))
                    buf.copy_(freqs.to(device=buf.device, dtype=buf.dtype))
                else:
                    buf.fill_(0.01)
            elif not torch.isfinite(buf).all():
                buf.copy_(torch.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0))


class _ContractVAE(nn.Module):
    """Deterministic VAE fixture matching an architecture's real latent channels & 4D/5D layout."""

    def __init__(
        self,
        latent_channels: int = 16,
        is_video_5d: bool = False,
        spatial_ratio: int = 8,
        temporal_frames: int = 2,
        max_spatial: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.is_video_5d = bool(is_video_5d)
        self.spatial_ratio = max(1, int(spatial_ratio))
        self.temporal_frames = max(1, int(temporal_frames))
        self.max_spatial = int(max_spatial) if max_spatial is not None else None
        self.enc = nn.Conv2d(3, self.latent_channels, kernel_size=1)
        with torch.no_grad():
            gen = torch.Generator(device="cpu").manual_seed(77)
            self.enc.weight.copy_(torch.randn(self.enc.weight.shape, generator=gen) * 0.25)
            if self.enc.bias is not None:
                self.enc.bias.zero_()

    def encode(self, pixel_samples: torch.Tensor) -> torch.Tensor:
        x = pixel_samples
        if x.ndim == 4 and x.shape[-1] in (3, 4) and x.shape[1] not in (3, 4):
            x = x[..., :3].permute(0, 3, 1, 2).contiguous()
        elif x.ndim == 4 and x.shape[1] == 4:
            x = x[:, :3]
        h = max(4, int(x.shape[-2]) // self.spatial_ratio)
        w = max(4, int(x.shape[-1]) // self.spatial_ratio)
        if self.max_spatial is not None:
            h = min(self.max_spatial, h)
            w = min(self.max_spatial, w)
        resized = F.interpolate(x.to(dtype=torch.float32), size=(h, w), mode="bilinear", align_corners=False)
        lat = self.enc(resized)
        if self.is_video_5d:
            frames = [lat * (1.0 + 0.05 * i) for i in range(self.temporal_frames)]
            return torch.stack(frames, dim=2)
        return lat


class _ContractCLIP(nn.Module):
    """
    Realistic ComfyUI CLIP fixture that:
      - Produces exact architecture-specific embedding widths (`embed_dim`) and optional `pooled_dim`.
      - Varies token sequence length deterministically from caption word count so `batch_size=2`
        genuinely exercises `OmniLoRAPrecomputeConditioning` padding & `attention_mask` synthesis.
      - Emits real model-specific conditioning extras via `extra_cond_factory(text, seq_len)`.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        pooled_dim: Optional[int] = None,
        extra_cond_factory: Optional[Callable[[str, int], Dict[str, Any]]] = None,
        base_seq_len: int = 5,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.pooled_dim = int(pooled_dim) if pooled_dim is not None else int(embed_dim)
        self.extra_cond_factory = extra_cond_factory
        self.base_seq_len = max(2, int(base_seq_len))
        self.cond_stage_model = nn.Linear(self.embed_dim, self.embed_dim)
        with torch.no_grad():
            gen = torch.Generator(device="cpu").manual_seed(99)
            self.cond_stage_model.weight.copy_(
                torch.eye(self.embed_dim) * 0.5
                + torch.randn(self.embed_dim, self.embed_dim, generator=gen) * 0.05
            )
            if self.cond_stage_model.bias is not None:
                self.cond_stage_model.bias.zero_()
        self.patcher = comfy.model_patcher.ModelPatcher(
            self.cond_stage_model,
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )

    def clone(self, *args: Any, **kwargs: Any) -> "_ContractCLIP":
        c = _ContractCLIP(
            embed_dim=self.embed_dim,
            pooled_dim=self.pooled_dim,
            extra_cond_factory=self.extra_cond_factory,
            base_seq_len=self.base_seq_len,
        )
        c.cond_stage_model = self.cond_stage_model
        c.patcher = self.patcher.clone() if self.patcher is not None else None
        return c

    def add_patches(self, patches: Any, strength_patch: float = 1.0, strength_model: float = 1.0) -> List[str]:
        if self.patcher is not None:
            return self.patcher.add_patches(patches, strength_patch, strength_model)
        return []

    def tokenize(self, text: str) -> Dict[str, Any]:
        words = [w for w in str(text).strip().split() if w]
        # Deterministic unequal lengths across short vs long prompts to trigger attention_mask batching
        seq_len = self.base_seq_len + min(12, len(words))
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(str(text))) % (2**31 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed + 17)
        tokens = torch.randn(1, seq_len, self.embed_dim, generator=gen, dtype=torch.float32) * 0.5
        return {"tokens": tokens, "text": str(text), "seq_len": seq_len}

    def encode_from_tokens(self, tokenized: Dict[str, Any], return_pooled: bool = True, return_dict: bool = False) -> Any:
        tokens = tokenized["tokens"]
        seq_len = int(tokens.shape[1])
        cond = self.cond_stage_model(tokens)
        pooled_raw = cond.mean(dim=1)
        if self.pooled_dim == self.embed_dim:
            pooled = pooled_raw
        elif self.pooled_dim < self.embed_dim:
            pooled = pooled_raw[:, : self.pooled_dim]
        else:
            reps = (self.pooled_dim + self.embed_dim - 1) // self.embed_dim
            pooled = pooled_raw.repeat(1, reps)[:, : self.pooled_dim]

        extras: Dict[str, Any] = {"pooled_output": pooled}
        if callable(self.extra_cond_factory):
            extra_out = self.extra_cond_factory(str(tokenized.get("text", "")), seq_len)
            if isinstance(extra_out, dict):
                extras.update(extra_out)

        if return_dict:
            out_dict = dict(extras)
            out_dict["cond"] = cond
            return out_dict
        if return_pooled:
            return cond, pooled
        return cond

    def encode_from_tokens_scheduled(self, tokenized: Dict[str, Any]) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
        res = self.encode_from_tokens(tokenized, return_pooled=True, return_dict=True)
        cond = res.pop("cond")
        return [(cond, res)]


def build_real_comfy_patcher(
    supported_cls: type,
    base_model_cls: type,
    diffusion_model: nn.Module,
    unet_config: Optional[Dict[str, Any]] = None,
    model_type: Optional[comfy.model_base.ModelType] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    latent_format: Optional[Any] = None,
    sampling_settings: Optional[Dict[str, Any]] = None,
    base_model_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 1234,
) -> comfy.model_patcher.ModelPatcher:
    """
    Construct a 100% genuine, UNMOCKED ComfyUI `ModelPatcher` around a real `comfy.ldm.*`
    `diffusion_model` and real `comfy.model_base.*` wrapper — NEVER monkey-patching
    `diffusion_model.forward` or `base_model.apply_model`.
    """
    target_device = device or torch.device("cpu")
    _initialize_real_comfy_module(diffusion_model, seed=seed)
    diffusion_model.to(device=target_device, dtype=dtype)
    diffusion_model.requires_grad_(False)
    diffusion_model.eval()
    try:
        diffusion_model.dtype = dtype  # type: ignore[attr-defined]
    except Exception:
        pass

    merged_unet_config = dict(getattr(supported_cls, "unet_config", {}))
    if unet_config:
        merged_unet_config.update(unet_config)
    merged_unet_config["disable_unet_model_creation"] = True

    model_config = supported_cls(merged_unet_config)
    if sampling_settings:
        try:
            merged_ss = dict(getattr(model_config, "sampling_settings", {}) or {})
            merged_ss.update(sampling_settings)
            model_config.sampling_settings = merged_ss
        except Exception:
            pass
    if latent_format is not None:
        model_config.latent_format = latent_format

    eff_type = model_type
    if eff_type is None:
        eff_type = getattr(model_config, "model_type", None) or comfy.model_base.ModelType.FLOW

    bm_kwargs = dict(base_model_kwargs or {})
    base_model = base_model_cls(model_config, model_type=eff_type, device=target_device, **bm_kwargs)
    base_model.diffusion_model = diffusion_model
    base_model.manual_cast_dtype = dtype
    if latent_format is not None:
        base_model.latent_format = latent_format

    patcher = comfy.model_patcher.ModelPatcher(
        base_model,
        load_device=target_device,
        offload_device=target_device,
    )
    base_model.current_patcher = patcher
    return patcher


# ==============================================================================
# 2. StrictComfySpy — Zero-Silent-Fallback Instrumentation
# ==============================================================================

class StrictComfySpy:
    """
    Transparent instrumentation context manager attached to a real `BaseModel` + `diffusion_model`.
    Records every call and every exception in `extra_conds`, `process_latent_in`, `process_timestep`,
    and `diffusion_model.forward`, plus `comfy.model_management.in_training` state inside each.
    """

    def __init__(self, inner_model: Any) -> None:
        self.inner_model = inner_model
        self.backbone = getattr(inner_model, "diffusion_model", None)
        self.extra_conds_calls: int = 0
        self.extra_conds_errors: List[Tuple[str, str]] = []
        self.in_training_in_extra_conds: List[bool] = []
        self.process_latent_in_calls: int = 0
        self.process_latent_in_errors: List[Tuple[str, str]] = []
        self.process_timestep_calls: int = 0
        self.process_timestep_errors: List[Tuple[str, str]] = []
        self.forward_calls: int = 0
        self.in_training_in_forward: List[bool] = []
        self._saved: List[Tuple[Any, str, bool, Any]] = []

    def _patch_attr(self, obj: Any, attr: str, wrapper: Any) -> None:
        had_dict = attr in obj.__dict__
        orig = obj.__dict__.get(attr)
        self._saved.append((obj, attr, had_dict, orig))
        obj.__dict__[attr] = wrapper

    def __enter__(self) -> "StrictComfySpy":
        im = self.inner_model
        if hasattr(im, "extra_conds") and callable(im.extra_conds):
            orig_ec = im.extra_conds

            def _spied_extra_conds(*args: Any, **kwargs: Any) -> Any:
                self.extra_conds_calls += 1
                self.in_training_in_extra_conds.append(bool(getattr(comfy.model_management, "in_training", False)))
                try:
                    return orig_ec(*args, **kwargs)
                except Exception as exc:
                    self.extra_conds_errors.append((type(exc).__name__, str(exc)))
                    raise

            self._patch_attr(im, "extra_conds", _spied_extra_conds)

        if hasattr(im, "process_latent_in") and callable(im.process_latent_in):
            orig_pli = im.process_latent_in

            def _spied_process_latent_in(*args: Any, **kwargs: Any) -> Any:
                self.process_latent_in_calls += 1
                try:
                    return orig_pli(*args, **kwargs)
                except Exception as exc:
                    self.process_latent_in_errors.append((type(exc).__name__, str(exc)))
                    raise

            self._patch_attr(im, "process_latent_in", _spied_process_latent_in)

        if hasattr(im, "process_timestep") and callable(im.process_timestep):
            orig_pt = im.process_timestep

            def _spied_process_timestep(*args: Any, **kwargs: Any) -> Any:
                self.process_timestep_calls += 1
                try:
                    return orig_pt(*args, **kwargs)
                except Exception as exc:
                    self.process_timestep_errors.append((type(exc).__name__, str(exc)))
                    raise

            self._patch_attr(im, "process_timestep", _spied_process_timestep)

        if self.backbone is not None and hasattr(self.backbone, "forward"):
            orig_fwd = self.backbone.forward

            @functools.wraps(orig_fwd)
            def _spied_forward(*args: Any, **kwargs: Any) -> Any:
                self.forward_calls += 1
                self.in_training_in_forward.append(bool(getattr(comfy.model_management, "in_training", False)))
                return orig_fwd(*args, **kwargs)

            self._patch_attr(self.backbone, "forward", _spied_forward)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for obj, attr, had_dict, orig in reversed(self._saved):
            if had_dict:
                obj.__dict__[attr] = orig
            else:
                obj.__dict__.pop(attr, None)
        self._saved.clear()

    def assert_no_silent_errors(self, test_case: unittest.TestCase, model_id: str) -> None:
        test_case.assertEqual(
            self.extra_conds_errors,
            [],
            f"[{model_id}] Silent exception(s) raised inside `inner_model.extra_conds`: {self.extra_conds_errors}",
        )
        test_case.assertEqual(
            self.process_latent_in_errors,
            [],
            f"[{model_id}] Silent exception(s) raised inside `inner_model.process_latent_in`: {self.process_latent_in_errors}",
        )
        test_case.assertEqual(
            self.process_timestep_errors,
            [],
            f"[{model_id}] Silent exception(s) raised inside `inner_model.process_timestep`: {self.process_timestep_errors}",
        )


# ==============================================================================
# 3. Real-Forward Model Registry (22 Unmocked ComfyUI Architectures)
# ==============================================================================

@dataclass
class RealModelContractSpec:
    model_id: str
    patcher: comfy.model_patcher.ModelPatcher
    vae: nn.Module
    clip: nn.Module
    expected_objective: str
    expected_latent_channels: int
    is_video_5d: bool
    allow_unused_prefixes: Tuple[str, ...] = ()
    supports_fp8_extra_conds: bool = False
    is_dual_stream_av: bool = False
    requires_batch_size_one_native: bool = False
    resolution: int = 64


class RealForwardModelRegistry:
    """
    Factory for 22 miniature, 100% UNMOCKED ComfyUI (`comfy.ldm.*` + `comfy.model_base.*`)
    architectures whose real `forward()`, `extra_conds()`, `process_latent_in()`, and
    `process_timestep()` execute end-to-end on CPU in milliseconds.
    """

    MODEL_IDS: Tuple[str, ...] = (
        "flux_dev",
        "flux3_klein",
        "sd15",
        "sdxl",
        "sd3_5",
        "qwen_image_1_0",
        "qwen_image_2_1",
        "wan_2_1_t2v",
        "wan_2_1_i2v",
        "hunyuan_video",
        "ltxv",
        "ltxav_2_0",
        "ltxav_2_5",
        "ltxav_2_5_first_linear",
        "minimax_h3_video",
        "minimax_h3_image",
        "anima",
        "lumina2",
        "auraflow",
        "moss_image",
        "krea2",
        "cosmos_predict2",
        # --- Stage 1 Full Unmocked Image & Video Expansion (+22 architectures -> 44 total) ---
        "wan_2_2_ti2v",
        "fibo2",
        "hidream_i1",
        "chroma",
        "chroma_radiance",
        "omnigen2",
        "boogu",
        "flux2",
        "ideogram4",
        "joyimage",
        "mage_flow",
        "pixeldit",
        "ernie",
        "lens",
        "kandinsky5_video",
        "kandinsky5_image",
        "cogvideox",
        "genmo_mochi",
        "pixart",
        "hunyuan_dit",
        "stable_cascade",
        "hunyuan_image_2_1",
    )

    @classmethod
    def build(
        cls,
        model_id: str,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> RealModelContractSpec:
        ops = comfy.ops.manual_cast
        dev = device or torch.device("cpu")

        if model_id == "flux_dev":
            from comfy.ldm.flux.model import Flux
            dm = Flux(
                in_channels=16, out_channels=16, vec_in_dim=64, context_in_dim=64,
                hidden_size=64, mlp_ratio=2.0, num_heads=4, depth=1, depth_single_blocks=1,
                axes_dim=[4, 6, 6], theta=10000, patch_size=2, qkv_bias=True, guidance_embed=True,
                txt_ids_dims=[1, 2], dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Flux,
                comfy.model_base.Flux,
                dm,
                unet_config={"in_channels": 16, "out_channels": 16, "context_in_dim": 64, "vec_in_dim": 64, "guidance_embed": True},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=False,
            )

        if model_id == "flux3_klein":
            from comfy.ldm.flux.layers import EmbedND
            from comfy.ldm.flux3.model import Flux3
            dm = Flux3(
                hidden_size=64, num_heads=4, depth=1, depth_single_blocks=1,
                fused_qkv=False, flow_map=True, dtype=dtype, device="cpu", operations=ops,
            )
            # Align RoPE axes_dim to tiny head_dim (64 // 4 = 16) and text dim to 64
            dm.pe_embedder = EmbedND(dim=16, theta=10000, axes_dim=[4, 4, 4, 4])
            dm.txt_in = ops.Linear(64, 64, bias=False, dtype=dtype, device="cpu")
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Flux3,
                comfy.model_base.Flux3,
                dm,
                unet_config={"in_channels": 128, "hidden_size": 64, "num_heads": 4},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            # `Flux3` inherits `Flux2` latent format (`latent_channels=128`, 4D `[B, 128, H, W]`)
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=False, spatial_ratio=16),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=False,
                allow_unused_prefixes=(
                    "content_mode_blocks.image_cond.",
                    "early_stream_modulations.image_cond.",
                    "single_stream_modulations.image_cond.",
                    "emb_in.image_cond",
                ),
            )

        if model_id == "sd15":
            from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
            dm = UNetModel(
                image_size=16, in_channels=4, model_channels=32, out_channels=4,
                num_res_blocks=1, channel_mult=[1, 2], num_heads=4,
                use_spatial_transformer=True, transformer_depth=[1, 1],
                transformer_depth_middle=1, transformer_depth_output=[1, 1, 1, 1],
                context_dim=64, dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.SD15,
                comfy.model_base.BaseModel,
                dm,
                unet_config={"in_channels": 4, "context_dim": 64, "model_channels": 32},
                model_type=comfy.model_base.ModelType.EPS,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(4, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="ddpm_eps",
                expected_latent_channels=4,
                is_video_5d=False,
            )

        if model_id == "sdxl":
            from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
            # Real `comfy.model_base.SDXL.encode_adm` concatenates 1280 pooled + 6*256=1536 sinusoidal = 2816
            dm = UNetModel(
                image_size=16, in_channels=4, model_channels=32, out_channels=4,
                num_res_blocks=1, channel_mult=[1, 2], num_heads=4,
                use_spatial_transformer=True, transformer_depth=[1, 1],
                transformer_depth_middle=1, transformer_depth_output=[1, 1, 1, 1],
                context_dim=64, adm_in_channels=2816, num_classes="sequential",
                use_linear_in_transformer=True, dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.SDXL,
                comfy.model_base.SDXL,
                dm,
                unet_config={"in_channels": 4, "context_dim": 64, "adm_in_channels": 2816},
                model_type=comfy.model_base.ModelType.EPS,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(4, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=1280),
                expected_objective="ddpm_eps",
                expected_latent_channels=4,
                is_video_5d=False,
            )

        if model_id == "sd3_5":
            from comfy.ldm.modules.diffusionmodules.mmdit import OpenAISignatureMMDITWrapper
            dm = OpenAISignatureMMDITWrapper(
                input_size=16, patch_size=2, in_channels=16, depth=2, mlp_ratio=2.0,
                adm_in_channels=64, context_size=64, pos_embed_max_size=16,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.SD3,
                comfy.model_base.SD3,
                dm,
                unet_config={"in_channels": 16, "context_size": 64, "adm_in_channels": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=False,
            )

        if model_id == "qwen_image_1_0":
            from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
            dm = QwenImageTransformer2DModel(
                patch_size=2, in_channels=64, out_channels=16, num_layers=2,
                attention_head_dim=16, num_attention_heads=4, joint_attention_dim=64,
                axes_dims_rope=(4, 6, 6), dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.QwenImage,
                comfy.model_base.QwenImage,
                dm,
                unet_config={"in_channels": 64, "out_channels": 16, "joint_attention_dim": 64},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            # In QwenImageTransformer2DModel, the final block's text output (`to_add_out` / `txt_mlp`)
            # is not consumed by `norm_out`/`proj_out`.
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=1),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=True,
                allow_unused_prefixes=(
                    "transformer_blocks.1.attn.add_q_proj",
                    "transformer_blocks.1.attn.to_add_out",
                    "transformer_blocks.1.txt_mlp.",
                ),
            )

        if model_id == "qwen_image_2_1":
            from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
            dm = QwenImage21Transformer2DModel(
                in_channels=64, out_channels=64, num_layers=2,
                attention_head_dim=16, num_attention_heads=4, context_in_dim=64,
                axes_dims_rope=(4, 6, 6), dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.QwenImage21,
                comfy.model_base.QwenImage21,
                dm,
                unet_config={"in_channels": 64, "out_channels": 64, "context_in_dim": 64},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(64, is_video_5d=False, spatial_ratio=16),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=64,
                is_video_5d=False,
            )

        if model_id == "wan_2_1_t2v":
            from comfy.ldm.wan.model import WanModel
            dm = WanModel(
                model_type="t2v", patch_size=(1, 2, 2), in_dim=16, dim=64, ffn_dim=128,
                freq_dim=32, text_dim=64, out_dim=16, num_heads=4, num_layers=2,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.WAN21_T2V,
                comfy.model_base.WAN21,
                dm,
                unet_config={"in_dim": 16, "out_dim": 16, "text_dim": 64, "model_type": "t2v"},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=2),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=True,
            )

        if model_id == "wan_2_1_i2v":
            from comfy.ldm.wan.model import WanModel
            dm = WanModel(
                model_type="i2v", patch_size=(1, 2, 2), in_dim=36, dim=64, ffn_dim=128,
                freq_dim=32, text_dim=64, out_dim=16, num_heads=4, num_layers=1,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.WAN21_I2V,
                comfy.model_base.WAN21,
                dm,
                unet_config={"in_dim": 36, "out_dim": 16, "text_dim": 64, "model_type": "i2v"},
                model_type=comfy.model_base.ModelType.FLOW,
                base_model_kwargs={"image_to_video": True},
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=2),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=True,
                # `k_img` / `v_img` are only invoked when optional `clip_vision_output` is supplied
                allow_unused_prefixes=("blocks.0.cross_attn.k_img", "blocks.0.cross_attn.v_img"),
            )

        if model_id == "hunyuan_video":
            from comfy.ldm.hunyuan_video.model import HunyuanVideo
            dm = HunyuanVideo(
                in_channels=16, out_channels=16, vec_in_dim=64, context_in_dim=64,
                hidden_size=64, mlp_ratio=2.0, num_heads=4, depth=1, depth_single_blocks=1,
                axes_dim=[4, 6, 6], theta=256, patch_size=[1, 2, 2], qkv_bias=True,
                guidance_embed=True, byt5=False, meanflow=False, use_cond_type_embedding=False,
                vision_in_dim=None, meanflow_sum=False,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.HunyuanVideo,
                comfy.model_base.HunyuanVideo,
                dm,
                unet_config={"in_channels": 16, "vec_in_dim": 64, "context_in_dim": 64, "guidance_embed": True},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=2),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=True,
            )

        if model_id == "ltxv":
            from comfy.ldm.lightricks.model import LTXVModel
            dm = LTXVModel(
                in_channels=128, cross_attention_dim=64, attention_head_dim=16,
                num_attention_heads=4, caption_channels=64, num_layers=1,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.LTXV,
                comfy.model_base.LTXV,
                dm,
                unet_config={"in_channels": 128, "cross_attention_dim": 64, "caption_channels": 64},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=True, spatial_ratio=16, temporal_frames=2),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=True,
            )

        if model_id == "ltxav_2_0":
            from comfy.ldm.lightricks.av_model import LTXAVModel
            dm = LTXAVModel(
                in_channels=128, audio_in_channels=128,
                cross_attention_dim=64, audio_cross_attention_dim=32,
                attention_head_dim=16, audio_attention_head_dim=8,
                num_attention_heads=4, audio_num_attention_heads=4,
                caption_channels=64, num_layers=1,
                caption_proj_before_connector=False,
                connector_attention_head_dim=16, connector_num_attention_heads=4,
                audio_connector_attention_head_dim=16, audio_connector_num_attention_heads=4,
                connector_num_layers=1,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.LTXAV,
                comfy.model_base.LTXAV,
                dm,
                unet_config={
                    "in_channels": 128,
                    "audio_in_channels": 128,
                    "cross_attention_dim": 64,
                    "audio_cross_attention_dim": 32,
                    "caption_channels": 64,
                    "caption_proj_before_connector": False,
                },
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=True, spatial_ratio=16, temporal_frames=2),
                clip=_ContractCLIP(
                    embed_dim=64,
                    pooled_dim=64,
                    extra_cond_factory=lambda _t, _l: {"unprocessed_ltxav_embeds": True},
                ),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=True,
                supports_fp8_extra_conds=True,
                is_dual_stream_av=True,
            )

        if model_id == "ltxav_2_5":
            from comfy.ldm.lightricks.av_model import LTXAVModel
            # Mirrors LTX 2.3 / 2.4 / 2.5 architecture: `caption_proj_before_connector=True`,
            # gated attention (`to_gate_logits`), cross-attention AdaLN, prompt AdaLN, and
            # dual-width (`cross_attention_dim + audio_cross_attention_dim = 64 + 32 = 96`) CLIP output.
            dm = LTXAVModel(
                in_channels=128, audio_in_channels=128,
                cross_attention_dim=64, audio_cross_attention_dim=32,
                attention_head_dim=16, audio_attention_head_dim=8,
                num_attention_heads=4, audio_num_attention_heads=4,
                caption_channels=48, num_layers=1,
                apply_gated_attention=True,
                caption_proj_before_connector=True,
                caption_projection_first_linear=False,
                cross_attention_adaln=True,
                use_prompt_adaln_single=True,
                ff_bias=False,
                audio_ff_bias=False,
                connector_apply_gated_attention=True,
                connector_attention_head_dim=16, connector_num_attention_heads=4,
                audio_connector_attention_head_dim=8, audio_connector_num_attention_heads=4,
                connector_num_layers=1,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.LTXAV,
                comfy.model_base.LTXAV,
                dm,
                unet_config={
                    "in_channels": 128,
                    "audio_in_channels": 128,
                    "cross_attention_dim": 64,
                    "audio_cross_attention_dim": 32,
                    "caption_channels": 48,
                    "caption_proj_before_connector": True,
                    "caption_projection_first_linear": False,
                    "apply_gated_attention": True,
                    "cross_attention_adaln": True,
                },
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=True, spatial_ratio=16, temporal_frames=2),
                clip=_ContractCLIP(
                    embed_dim=96,  # cross_attention_dim (64) + audio_cross_attention_dim (32)
                    pooled_dim=64,
                    extra_cond_factory=lambda _t, _l: {"unprocessed_ltxav_embeds": True},
                ),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=True,
                supports_fp8_extra_conds=True,
                is_dual_stream_av=True,
            )

        if model_id == "ltxav_2_5_first_linear":
            from comfy.ldm.lightricks.av_model import LTXAVModel
            # Variant where `caption_projection_first_linear=True` (raw `caption_channels` enters
            # `NormSingleLinearTextProjection` before `Embeddings1DConnector`).
            dm = LTXAVModel(
                in_channels=128, audio_in_channels=128,
                cross_attention_dim=64, audio_cross_attention_dim=32,
                attention_head_dim=16, audio_attention_head_dim=8,
                num_attention_heads=4, audio_num_attention_heads=4,
                caption_channels=48, num_layers=1,
                apply_gated_attention=True,
                caption_proj_before_connector=True,
                caption_projection_first_linear=True,
                cross_attention_adaln=True,
                use_prompt_adaln_single=True,
                ff_bias=False,
                audio_ff_bias=False,
                connector_apply_gated_attention=True,
                connector_attention_head_dim=16, connector_num_attention_heads=4,
                audio_connector_attention_head_dim=8, audio_connector_num_attention_heads=4,
                connector_num_layers=1,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.LTXAV,
                comfy.model_base.LTXAV,
                dm,
                unet_config={
                    "in_channels": 128,
                    "audio_in_channels": 128,
                    "cross_attention_dim": 64,
                    "audio_cross_attention_dim": 32,
                    "caption_channels": 48,
                    "caption_proj_before_connector": True,
                    "caption_projection_first_linear": True,
                },
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=True, spatial_ratio=16, temporal_frames=2),
                clip=_ContractCLIP(
                    embed_dim=48,  # raw `caption_channels` before NormSingleLinearTextProjection
                    pooled_dim=48,
                    extra_cond_factory=lambda _t, _l: {"unprocessed_ltxav_embeds": True},
                ),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=True,
                supports_fp8_extra_conds=True,
                is_dual_stream_av=True,
            )

        if model_id == "minimax_h3_video":
            from comfy.ldm.minimax.model import MiniMaxH3Model
            dm = MiniMaxH3Model(
                hidden_size=64, num_layers=1, token_refiner_num_layers=1,
                num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
                latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2),
                text_dim=32, timestep_input_dim=32, time_embed_hidden_size=64,
                time_embed_dim=32, rope_inv_freq_len=2,
                sigma_shift_video=5.0, sigma_shift_audio=1.0,
                gate_compress=True, pdd_steps=4,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.MiniMaxH3,
                comfy.model_base.MiniMaxH3,
                dm,
                unet_config={
                    "image_model": "minimax_h3",
                    "latents_dim": 24,
                    "audio_latents_dim": 32,
                    "gate_compress": True,
                    "pdd_steps": 4,
                    "sigma_shift_video": 5.0,
                    "sigma_shift_audio": 1.0,
                },
                model_type=comfy.model_base.ModelType.FLOW_AV,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(24, is_video_5d=True, temporal_frames=2),
                clip=_ContractCLIP(
                    embed_dim=32,
                    pooled_dim=32,
                    extra_cond_factory=lambda _t, seq_len: {
                        "minimax_token_tags": torch.zeros((1, seq_len), dtype=torch.long)
                    },
                ),
                expected_objective="rectified_flow",
                expected_latent_channels=24,
                is_video_5d=True,
                supports_fp8_extra_conds=True,
                is_dual_stream_av=True,
                requires_batch_size_one_native=True,
            )

        if model_id == "minimax_h3_image":
            from comfy.ldm.minimax.model import MiniMaxH3ImageModel
            pdd_sigmas = [1.0, 0.85, 0.60, 0.30, 0.0]
            dm = MiniMaxH3ImageModel(
                pdd_sigmas=pdd_sigmas, pdd_steps=4,
                hidden_size=64, num_layers=1, token_refiner_num_layers=1,
                num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
                latents_dim=24, patch_size=(1, 2, 2),
                text_dim=32, time_embed_dim=16, adaln_curve_grid=8,
                gate_compress=True, rope_inv_freq_len=2,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.MiniMaxH3Image,
                comfy.model_base.MiniMaxH3Image,
                dm,
                unet_config={
                    "image_model": "minimax_h3_image",
                    "latents_dim": 24,
                    "pdd_steps": 4,
                    "pdd_sigmas": pdd_sigmas,
                    "gate_compress": True,
                    "adaln_curve_grid": 8,
                },
                model_type=comfy.model_base.ModelType.FLOW,
                sampling_settings={"pdd_sigmas": pdd_sigmas},
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(24, is_video_5d=True, temporal_frames=1),
                clip=_ContractCLIP(
                    embed_dim=32,
                    pooled_dim=32,
                    extra_cond_factory=lambda _t, seq_len: {
                        "minimax_token_tags": torch.zeros((1, seq_len), dtype=torch.long)
                    },
                ),
                expected_objective="rectified_flow",
                expected_latent_channels=24,
                is_video_5d=True,
                supports_fp8_extra_conds=True,
                requires_batch_size_one_native=True,
            )

        if model_id == "anima":
            from comfy.ldm.anima.model import Anima, LLMAdapter
            dm = Anima(
                max_img_h=32, max_img_w=32, max_frames=4,
                in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
                concat_padding_mask=False, model_channels=64, num_blocks=1,
                num_heads=4, mlp_ratio=2.0, crossattn_emb_channels=64,
                pos_emb_cls="rope3d", dtype=dtype, device="cpu", operations=ops,
            )
            # Replace default 1024-dim 6-layer LLMAdapter with a miniature 64-dim 1-layer LLMAdapter
            dm.llm_adapter = LLMAdapter(
                source_dim=64, target_dim=64, model_dim=64, num_layers=1, num_heads=4,
                device="cpu", dtype=dtype, operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Anima,
                comfy.model_base.Anima,
                dm,
                unet_config={"in_channels": 16, "out_channels": 16, "crossattn_emb_channels": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=1),
                clip=_ContractCLIP(
                    embed_dim=64,
                    pooled_dim=64,
                    extra_cond_factory=lambda _t, seq_len: {
                        "t5xxl_ids": torch.arange(1, seq_len + 1, dtype=torch.long),
                        "t5xxl_weights": torch.ones((seq_len,), dtype=torch.float32),
                    },
                ),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=True,
                supports_fp8_extra_conds=True,
            )

        if model_id == "lumina2":
            from comfy.ldm.lumina.model import NextDiT
            dm = NextDiT(
                patch_size=2, in_channels=16, dim=64, n_layers=1, n_refiner_layers=1,
                n_heads=4, n_kv_heads=2, multiple_of=16, cap_feat_dim=64,
                axes_dims=(4, 6, 6), axes_lens=(4, 32, 32),
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Lumina2,
                comfy.model_base.Lumina2,
                dm,
                unet_config={"in_channels": 16, "cap_feat_dim": 64, "dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=False,
            )

        if model_id == "auraflow":
            from comfy.ldm.aura.mmdit import MMDiT as AuraMMDiT
            dm = AuraMMDiT(
                in_channels=4, out_channels=4, patch_size=2, dim=64,
                n_layers=2, n_double_layers=1, n_heads=4, global_conddim=64,
                cond_seq_dim=64, max_seq=64, dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.AuraFlow,
                comfy.model_base.AuraFlow,
                dm,
                unet_config={"in_channels": 4, "cond_seq_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(4, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=4,
                is_video_5d=False,
            )

        if model_id == "moss_image":
            from comfy.ldm.moss_image.model import MOSSImage
            dm = MOSSImage(
                dim=64, ffn_dim=128, num_heads=4, num_layers=1, text_dim=64,
                latent_channels=128, fused_qkv=False, dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.MOSSImage,
                comfy.model_base.MOSSImage,
                dm,
                unet_config={"latent_channels": 128, "text_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=False, spatial_ratio=16),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=False,
            )

        if model_id == "krea2":
            from comfy.ldm.krea2.model import SingleStreamDiT
            dm = SingleStreamDiT(
                features=64, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=2,
                layers=1, patch=2, channels=16, txtlayers=1, txtheads=4, txtkvheads=4,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Krea2,
                comfy.model_base.Krea2,
                dm,
                unet_config={"channels": 16, "txtdim": 64, "txtlayers": 1},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=1),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=True,
            )

        if model_id == "cosmos_predict2":
            from comfy.ldm.cosmos.predict2 import MiniTrainDIT
            dm = MiniTrainDIT(
                max_img_h=32, max_img_w=32, max_frames=8, in_channels=16, out_channels=16,
                patch_spatial=2, patch_temporal=1, concat_padding_mask=False,
                model_channels=64, num_blocks=1, num_heads=4, mlp_ratio=2.0,
                crossattn_emb_channels=64, pos_emb_cls="rope3d",
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.CosmosT2IPredict2,
                comfy.model_base.CosmosPredict2,
                dm,
                unet_config={"in_channels": 16, "out_channels": 16, "crossattn_emb_channels": 64},
                model_type=comfy.model_base.ModelType.FLOW_COSMOS,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=2),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=True,
            )

        if model_id == "wan_2_2_ti2v":
            from comfy.ldm.wan.model import WanModel
            dm = WanModel(
                model_type="t2v", patch_size=(1, 2, 2), text_len=16, in_dim=48,
                dim=64, ffn_dim=128, freq_dim=32, text_dim=64, out_dim=48,
                num_heads=4, num_layers=1, dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.WAN22_T2V,
                comfy.model_base.WAN22,
                dm,
                unet_config={"image_model": "wan2.1", "model_type": "t2v", "in_dim": 48, "out_dim": 48, "text_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(48, is_video_5d=True, temporal_frames=2),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=48,
                is_video_5d=True,
            )

        if model_id == "fibo2":
            from comfy.ldm.fibo2.model import Fibo2
            # Fibo2 hardcodes `self.rope_axes_dim = [32, 48, 48]` (sum=128) -> head_dim = dim // n_heads = 128,
            # `gist_count(...)` enforces a minimum of 256 gist tokens, and `context` is chunked along dim=-1
            # into `1 + len(injection_layers)` bundles of width `cap_feat_dim` (3 * 64 = 192).
            dm = Fibo2(
                in_channels=128, dim=256, n_heads=2, n_layers=2, n_refiner_layers=1,
                injection_layers=(0, 1), cap_feat_dim=64, perceiver_layers=1,
                max_gist_tokens=256, ffn_dim=256, dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Fibo2,
                comfy.model_base.Fibo2,
                dm,
                unet_config={"image_model": "fibo2", "in_channels": 128, "cap_feat_dim": 64},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=False, spatial_ratio=16),
                clip=_ContractCLIP(embed_dim=192, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=False,
            )

        if model_id == "hidream_i1":
            from comfy.ldm.hidream.model import HiDreamImageTransformer2DModel
            dm = HiDreamImageTransformer2DModel(
                patch_size=2, in_channels=16, out_channels=16, num_layers=2,
                num_single_layers=2, attention_head_dim=16, num_attention_heads=4,
                caption_channels=[64, 64], text_emb_dim=64,
                num_routed_experts=2, num_activated_experts=2,
                axes_dims_rope=(4, 6, 6), max_resolution=(16, 16),
                llama_layers=[0, 1, 2, 3],
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.HiDream,
                comfy.model_base.HiDream,
                dm,
                unet_config={"image_model": "hidream", "in_channels": 16, "out_channels": 16},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )

            def _hidream_extras(text: str, _seq_len: int) -> Dict[str, Any]:
                seed = sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % (2**31 - 1)
                gen = torch.Generator(device="cpu").manual_seed(seed + 409)
                llama = torch.randn(1, 4, 8, 64, generator=gen, dtype=torch.float32) * 0.5
                return {"conditioning_llama3": llama}

            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64, extra_cond_factory=_hidream_extras),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=False,
            )

        if model_id == "chroma":
            from comfy.ldm.chroma.model import Chroma
            # Chroma expects patchified `in_channels = latent_channels * patch_size^2 = 16 * 4 = 64`
            # and `in_dim = 64` (timestep[16] + guidance[16] + mod_index[32]).
            dm = Chroma(
                in_channels=64, out_channels=64, context_in_dim=64, hidden_size=64,
                mlp_ratio=2.0, num_heads=4, depth=1, depth_single_blocks=1,
                axes_dim=[4, 6, 6], theta=10000, patch_size=2, qkv_bias=True,
                in_dim=64, out_dim=64, hidden_dim=64, n_layers=2,
                txt_ids_dims=[1, 2], vec_in_dim=64,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Chroma,
                comfy.model_base.Chroma,
                dm,
                unet_config={"image_model": "chroma", "in_channels": 64, "out_channels": 64, "context_in_dim": 64},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=False,
            )

        if model_id == "chroma_radiance":
            from comfy.ldm.chroma_radiance.model import ChromaRadiance
            dm = ChromaRadiance(
                in_channels=3, out_channels=3, context_in_dim=64,
                hidden_size=64, mlp_ratio=2.0, num_heads=4, depth=1,
                depth_single_blocks=1, axes_dim=[4, 6, 6], theta=10000,
                patch_size=2, qkv_bias=True, in_dim=64, out_dim=64,
                hidden_dim=64, n_layers=2, txt_ids_dims=[1, 2], vec_in_dim=64,
                nerf_hidden_size=32, nerf_mlp_ratio=2, nerf_depth=1, nerf_max_freqs=4,
                nerf_tile_size=0, nerf_final_head_type="linear",
                nerf_embedder_dtype=None, use_x0=False,
                use_sequential_txt_ids=False,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.ChromaRadiance,
                comfy.model_base.ChromaRadiance,
                dm,
                unet_config={"image_model": "chroma_radiance", "in_channels": 3, "out_channels": 3, "context_in_dim": 64},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(3, is_video_5d=False, spatial_ratio=1, max_spatial=8),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=3,
                is_video_5d=False,
            )

        if model_id == "omnigen2":
            from comfy.ldm.omnigen.omnigen2 import OmniGen2Transformer2DModel
            dm = OmniGen2Transformer2DModel(
                patch_size=2, in_channels=16, out_channels=16, hidden_size=64,
                num_layers=2, num_refiner_layers=1, num_attention_heads=4,
                num_kv_heads=2, multiple_of=16, axes_dim_rope=(4, 6, 6),
                axes_lens=(32, 32, 32), text_feat_dim=64,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Omnigen2,
                comfy.model_base.Omnigen2,
                dm,
                unet_config={"image_model": "omnigen2", "in_channels": 16, "out_channels": 16, "text_feat_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=False),
                clip=_ContractCLIP(
                    embed_dim=64,
                    pooled_dim=64,
                    extra_cond_factory=lambda _t, l: {"attention_mask": torch.ones((1, l), dtype=torch.long)},
                ),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=False,
                # `ref_image_refiner` only executes when reference-image conditioning (`ref_latents`) is supplied.
                allow_unused_prefixes=("ref_image_refiner.",),
            )

        if model_id == "boogu":
            from comfy.ldm.boogu.model import BooguTransformer2DModel
            dm = BooguTransformer2DModel(
                patch_size=2, in_channels=16, out_channels=16, hidden_size=64,
                num_layers=2, num_double_stream_layers=1, num_refiner_layers=1,
                num_attention_heads=4, num_kv_heads=2, multiple_of=16,
                axes_dim_rope=(4, 6, 6), axes_lens=(32, 32, 32), instruction_feat_dim=64,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Boogu,
                comfy.model_base.Boogu,
                dm,
                unet_config={"image_model": "boogu", "in_channels": 16, "out_channels": 16},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=False),
                clip=_ContractCLIP(
                    embed_dim=64,
                    pooled_dim=64,
                    extra_cond_factory=lambda _t, l: {"attention_mask": torch.ones((1, l), dtype=torch.long)},
                ),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=False,
                # `ref_image_refiner` only executes when reference-image conditioning (`ref_latents`) is supplied.
                allow_unused_prefixes=("ref_image_refiner.",),
            )

        if model_id == "flux2":
            from comfy.ldm.flux.model import Flux
            dm = Flux(
                in_channels=128, out_channels=128, vec_in_dim=None, context_in_dim=64,
                hidden_size=64, mlp_ratio=2.0, num_heads=4, depth=1, depth_single_blocks=1,
                axes_dim=[4, 4, 4, 4], theta=2000, patch_size=1, qkv_bias=False,
                guidance_embed=True, global_modulation=True, mlp_silu_act=True,
                ops_bias=False, txt_ids_dims=[3],
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Flux2,
                comfy.model_base.Flux2,
                dm,
                unet_config={
                    "image_model": "flux2",
                    "in_channels": 128,
                    "out_channels": 128,
                    "hidden_size": 64,
                    "context_in_dim": 64,
                },
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=False, spatial_ratio=16),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=False,
            )

        if model_id == "ideogram4":
            from comfy.ldm.ideogram4.model import Ideogram4Transformer2DModel
            dm = Ideogram4Transformer2DModel(
                in_channels=128, num_layers=2, num_attention_heads=4,
                attention_head_dim=16, intermediate_size=128, adaln_dim=64,
                llm_features_dim=64, mrope_section=(2, 3, 3),
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Ideogram4,
                comfy.model_base.Ideogram4,
                dm,
                unet_config={"image_model": "ideogram4", "in_channels": 128, "llm_features_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=False, spatial_ratio=16),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=False,
            )

        if model_id == "joyimage":
            from comfy.ldm.joyimage.model import JoyImageTransformer3DModel
            dm = JoyImageTransformer3DModel(
                patch_size=[1, 2, 2], in_channels=16, hidden_size=64,
                num_attention_heads=4, text_dim=64, mlp_width_ratio=1.0,
                num_layers=2, rope_dim_list=[4, 6, 6], theta=256,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.JoyImage,
                comfy.model_base.JoyImage,
                dm,
                unet_config={"image_model": "joyimage", "in_channels": 16, "text_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=1),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=True,
                # Final dual-stream block discards text-stream output (same pattern as QwenImage 1.0 / MageFlow / Lens).
                allow_unused_prefixes=(
                    "double_blocks.1.txt_mlp.",
                    "double_blocks.1.attn.txt_attn_proj",
                ),
            )

        if model_id == "mage_flow":
            from comfy.ldm.mage_flow.model import MageFlowTransformer2DModel
            dm = MageFlowTransformer2DModel(
                in_channels=128, out_channels=128, num_layers=2,
                attention_head_dim=16, num_attention_heads=4, joint_attention_dim=64,
                axes_dims_rope=(4, 6, 6),
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.MageFlow,
                comfy.model_base.MageFlow,
                dm,
                unet_config={"image_model": "mage_flow", "in_channels": 128, "out_channels": 128, "joint_attention_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=False, spatial_ratio=16),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=False,
                # Final dual-stream block discards text stream output (same as QwenImage 1.0)
                allow_unused_prefixes=(
                    "transformer_blocks.1.attn.add_q_proj",
                    "transformer_blocks.1.attn.to_add_out",
                    "transformer_blocks.1.txt_mlp.",
                ),
            )

        if model_id == "pixeldit":
            from comfy.ldm.pixeldit.model import PixDiT_T2I
            dm = PixDiT_T2I(
                in_channels=3, num_groups=4, hidden_size=64, pixel_hidden_size=16,
                pixel_attn_hidden_size=64, pixel_num_groups=4, patch_depth=2,
                pixel_depth=2, patch_size=2, txt_embed_dim=64, txt_max_length=64,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.PixelDiTT2I,
                comfy.model_base.PixelDiTT2I,
                dm,
                unet_config={"image_model": "pixeldit_t2i", "in_channels": 3, "txt_embed_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(3, is_video_5d=False, spatial_ratio=1, max_spatial=8),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=3,
                is_video_5d=False,
                # Final patch-level DiT block discards `y_emb`; only visual tokens `s` feed `pixel_blocks`.
                allow_unused_prefixes=(
                    "patch_blocks.1.attn.proj_y",
                    "patch_blocks.1.mlp_y.",
                ),
            )

        if model_id == "ernie":
            from comfy.ldm.ernie.model import ErnieImageModel
            dm = ErnieImageModel(
                hidden_size=64, num_attention_heads=4, num_layers=2,
                ffn_hidden_size=128, in_channels=128, out_channels=128,
                patch_size=1, text_in_dim=64, rope_theta=256,
                rope_axes_dim=(4, 6, 6), dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.ErnieImage,
                comfy.model_base.ErnieImage,
                dm,
                unet_config={"image_model": "ernie", "in_channels": 128, "out_channels": 128, "text_in_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=False, spatial_ratio=16),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=False,
            )

        if model_id == "lens":
            from comfy.ldm.lens.model import LensTransformer2DModel
            dm = LensTransformer2DModel(
                patch_size=2, in_channels=128, out_channels=32,
                num_layers=2, attention_head_dim=16, num_attention_heads=4,
                enc_hidden_dim=32, axes_dims_rope=(4, 6, 6),
                multi_layer_encoder_feature=True, selected_layer_index=(0, 1),
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Lens,
                comfy.model_base.Lens,
                dm,
                unet_config={"image_model": "lens", "in_channels": 128, "out_channels": 32, "enc_hidden_dim": 32},
                model_type=comfy.model_base.ModelType.FLUX,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(128, is_video_5d=False, spatial_ratio=16),
                # 2 selected layers * enc_hidden_dim(32) = 64
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=128,
                is_video_5d=False,
                allow_unused_prefixes=(
                    "transformer_blocks.1.attn.add_q_proj",
                    "transformer_blocks.1.attn.to_add_out",
                    "transformer_blocks.1.txt_mlp.",
                ),
            )

        if model_id == "kandinsky5_video":
            from comfy.ldm.kandinsky5.model import Kandinsky5
            # Kandinsky5 video concatenates (noise[16], zero_img[16], mask[1]) = 33 channels -> 33*1*2*2 = 132
            dm = Kandinsky5(
                in_visual_dim=33, in_text_dim=64, in_text_dim2=64,
                time_dim=64, out_visual_dim=16, visual_embed_dim=132,
                patch_size=(1, 2, 2), model_dim=64, ff_dim=128,
                num_text_blocks=1, num_visual_blocks=2, axes_dims=(4, 6, 6),
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Kandinsky5,
                comfy.model_base.Kandinsky5,
                dm,
                unet_config={"image_model": "kandinsky5", "in_visual_dim": 33, "out_visual_dim": 16, "visual_embed_dim": 132},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=2),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=True,
            )

        if model_id == "kandinsky5_image":
            from comfy.ldm.kandinsky5.model import Kandinsky5
            dm = Kandinsky5(
                in_visual_dim=16, in_text_dim=64, in_text_dim2=64,
                time_dim=64, out_visual_dim=16, visual_embed_dim=64,
                patch_size=(1, 2, 2), model_dim=64, ff_dim=128,
                num_text_blocks=1, num_visual_blocks=2, axes_dims=(4, 6, 6),
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Kandinsky5Image,
                comfy.model_base.Kandinsky5Image,
                dm,
                unet_config={"image_model": "kandinsky5", "in_visual_dim": 16, "out_visual_dim": 16, "visual_embed_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=16,
                is_video_5d=False,
            )

        if model_id == "cogvideox":
            from comfy.ldm.cogvideo.model import CogVideoXTransformer3DModel
            dm = CogVideoXTransformer3DModel(
                num_attention_heads=4, attention_head_dim=24,
                in_channels=16, out_channels=16, text_embed_dim=64,
                time_embed_dim=32, num_layers=2,
                sample_width=8, sample_height=8, sample_frames=5,
                patch_size=2, temporal_compression_ratio=4, max_text_seq_length=32,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.CogVideoX_T2V,
                comfy.model_base.CogVideoX,
                dm,
                unet_config={"image_model": "cogvideox", "in_channels": 16, "out_channels": 16, "num_attention_heads": 4},
                model_type=comfy.model_base.ModelType.V_PREDICTION_DDPM,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=True, temporal_frames=2),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="v_prediction",
                expected_latent_channels=16,
                is_video_5d=True,
            )

        if model_id == "genmo_mochi":
            from comfy.ldm.genmo.joint_model.asymm_models_joint import AsymmDiTJoint
            # ComfyUI's `AsymmetricJointBlock` enforces `assert int(hidden_size_x * mlp_ratio_x) == int(1536 * 8)` (12288).
            dm = AsymmDiTJoint(
                depth=2, patch_size=2, num_heads=4, hidden_size_x=64,
                hidden_size_y=64, mlp_ratio_x=192.0, mlp_ratio_y=2.0,
                in_channels=12, use_t5=True, t5_feat_dim=64, learn_sigma=False,
                qk_norm=True, dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.GenmoMochi,
                comfy.model_base.GenmoMochi,
                dm,
                unet_config={"image_model": "mochi_preview", "in_channels": 12},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(12, is_video_5d=True, temporal_frames=2),
                clip=_ContractCLIP(
                    embed_dim=64,
                    pooled_dim=64,
                    extra_cond_factory=lambda _t, l: {"attention_mask": torch.ones((1, l), dtype=torch.bool)},
                ),
                expected_objective="rectified_flow",
                expected_latent_channels=12,
                is_video_5d=True,
            )

        if model_id == "pixart":
            from comfy.ldm.pixart.pixartms import PixArtMS
            # `PixArtMS.forward_orig` computes `base_size = (round(c_res / 64) * 64) // patch_size`;
            # spatial latent size >= 34 ensures `round(c_res / 64) == 1` (`base_size == 32`, avoiding ZeroDivisionError).
            dm = PixArtMS(
                input_size=34, patch_size=2, in_channels=4, hidden_size=64,
                depth=2, num_heads=4, mlp_ratio=2.0, caption_channels=64,
                pe_interpolation=1.0, micro_condition=False,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.PixArtSigma,
                comfy.model_base.PixArt,
                dm,
                unet_config={"image_model": "pixart_sigma", "in_channels": 4, "caption_channels": 64},
                model_type=comfy.model_base.ModelType.EPS,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(4, is_video_5d=False, spatial_ratio=1, max_spatial=34),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="ddpm_eps",
                expected_latent_channels=4,
                is_video_5d=False,
            )

        if model_id == "hunyuan_dit":
            from comfy.ldm.hydit.models import HunYuanDiT
            dm = HunYuanDiT(
                input_size=(8, 8), patch_size=2, in_channels=4, hidden_size=64,
                depth=4, num_heads=4, mlp_ratio=2.0, text_states_dim=64,
                text_states_dim_t5=64, text_len=4, text_len_t5=8,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.HunyuanDiT,
                comfy.model_base.HunyuanDiT,
                dm,
                unet_config={"image_model": "hydit", "in_channels": 4, "text_states_dim": 64},
                model_type=comfy.model_base.ModelType.V_PREDICTION,
                device=dev,
                dtype=dtype,
            )

            def _hydit_extras(text: str, seq_len: int) -> Dict[str, Any]:
                seed = sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % (2**31 - 1)
                gen = torch.Generator(device="cpu").manual_seed(seed + 521)
                return {
                    "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
                    "conditioning_mt5xl": torch.randn(1, 8, 64, generator=gen, dtype=torch.float32) * 0.5,
                    "attention_mask_mt5xl": torch.ones((1, 8), dtype=torch.long),
                }

            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(4, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64, extra_cond_factory=_hydit_extras),
                expected_objective="v_prediction",
                expected_latent_channels=4,
                is_video_5d=False,
            )

        if model_id == "stable_cascade":
            from comfy.ldm.cascade.stage_c import StageC
            # `model_base.StableCascade_C.extra_conds` synthesizes `clip_img = torch.zeros((1, 1, 768))`
            dm = StageC(
                c_in=16, c_out=16, c_r=16, patch_size=1, c_cond=64,
                c_hidden=[64, 64], nhead=[4, 4], blocks=[[1, 1], [1, 1]],
                block_repeat=[[1, 1], [1, 1]], level_config=["CTA", "CTA"],
                c_clip_text=64, c_clip_text_pooled=64, c_clip_img=768,
                c_clip_seq=2, switch_level=[False],
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.Stable_Cascade_C,
                comfy.model_base.StableCascade_C,
                dm,
                unet_config={"stable_cascade_stage": "c", "c_in": 16, "c_out": 16},
                model_type=comfy.model_base.ModelType.STABLE_CASCADE,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(16, is_video_5d=False),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="ddpm_eps",
                expected_latent_channels=16,
                is_video_5d=False,
            )

        if model_id == "hunyuan_image_2_1":
            from comfy.ldm.hunyuan_video.model import HunyuanVideo
            dm = HunyuanVideo(
                in_channels=64, out_channels=64, vec_in_dim=None, context_in_dim=64,
                hidden_size=64, mlp_ratio=2.0, num_heads=4, depth=1, depth_single_blocks=1,
                axes_dim=[8, 8], theta=256, patch_size=[1, 1], qkv_bias=True, guidance_embed=True,
                byt5=False, meanflow=False, use_cond_type_embedding=False,
                vision_in_dim=None, meanflow_sum=False,
                dtype=dtype, device="cpu", operations=ops,
            )
            patcher = build_real_comfy_patcher(
                comfy.supported_models.HunyuanImage21,
                comfy.model_base.HunyuanImage21,
                dm,
                unet_config={"image_model": "hunyuan_video", "vec_in_dim": None, "in_channels": 64, "out_channels": 64, "context_in_dim": 64},
                model_type=comfy.model_base.ModelType.FLOW,
                device=dev,
                dtype=dtype,
            )
            return RealModelContractSpec(
                model_id=model_id,
                patcher=patcher,
                vae=_ContractVAE(64, is_video_5d=False, spatial_ratio=16),
                clip=_ContractCLIP(embed_dim=64, pooled_dim=64),
                expected_objective="rectified_flow",
                expected_latent_channels=64,
                is_video_5d=False,
            )

        raise ValueError(f"Unknown RealForwardModelRegistry model_id: {model_id}")


# ==============================================================================
# 4. Deterministic Real-Forward Helper & 7-Invariant Contract Verifier
# ==============================================================================

def _run_real_forward_probe(
    patcher: comfy.model_patcher.ModelPatcher,
    adapter: Any,
    batch: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    patch_model_weights: bool = False,
) -> torch.Tensor:
    """
    Execute the REAL unmocked `diffusion_model.forward()` via `adapter._forward_backbone` under
    `torch.no_grad()` on a cloned `ModelPatcher` (optionally applying ComfyUI LoRA patches via
    `cloned.patch_model()`), returning the raw model prediction tensor for exact parity checks.
    """
    cloned = patcher.clone()
    backbone = cloned.model.diffusion_model
    lat = adapter._validate_and_prepare_latents(
        batch["latents"].to(device=device, dtype=dtype),
        backbone=backbone,
        latent_scaled=bool(batch.get("latent_scaled", True)),
    )
    pe = batch["prompt_embeds"].to(device=device, dtype=dtype)
    pooled = batch.get("pooled_output")
    if isinstance(pooled, torch.Tensor):
        pooled = pooled.to(device=device, dtype=dtype)
    extra = batch.get("extra_conds")
    bsz = int(lat.shape[0])
    # Single-item slice if model requires batch_size == 1 natively (or for multi-row attn_mask)
    if bsz > 1 and adapter._requires_batch_size_one_for_batch(backbone, batch):
        sub = adapter._slice_single_item_batch(batch, 0, bsz)
        return _run_real_forward_probe(
            patcher=patcher,
            adapter=adapter,
            batch=sub,
            device=device,
            dtype=dtype,
            patch_model_weights=patch_model_weights,
        )
    sigma = torch.full((bsz,), 0.5, device=device, dtype=torch.float32)
    sampling = adapter.model_sampling
    if sampling is not None:
        timesteps = adapter._compute_device_timesteps(sampling, sigma, device=device, dtype=torch.float32).view(bsz)
    else:
        timesteps = sigma

    orig_dev: Optional[torch.device] = None
    try:
        first_p = next(backbone.parameters(), None)
        if first_p is not None:
            orig_dev = first_p.device
    except Exception:
        orig_dev = None

    moved_backbone = False
    try:
        if patch_model_weights:
            try:
                cloned.patch_model(device_to=device)
            except TypeError:
                cloned.patch_model()
        if orig_dev is None or orig_dev != device:
            backbone.to(device=device)
            moved_backbone = True
        # Seed CPU + active backend RNG deterministically so joint audio-video models that
        # synthesize companion audio noise during forward (e.g. MiniMax H3) produce bit-identical
        # outputs across baseline, post-uninject, and ComfyUI-patched parity probes.
        saved_cpu_rng = torch.random.get_rng_state()
        try:
            torch.manual_seed(1337)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed(1337)
            elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.manual_seed(1337)
            elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(1337)
            with torch.no_grad():
                out = adapter._forward_backbone(
                    backbone=backbone,
                    noisy_latents=lat,
                    timesteps=timesteps,
                    prompt_embeds=pe,
                    pooled_output=pooled,
                    extra_cond_dict=extra,
                    sigma=sigma,
                    clean_latents=lat,
                )
                return out.detach().clone()
        finally:
            torch.random.set_rng_state(saved_cpu_rng)
    finally:
        if patch_model_weights:
            try:
                cloned.unpatch_model(device_to=orig_dev or torch.device("cpu"), unpatch_weights=True)
            except Exception:
                pass
        elif moved_backbone and orig_dev is not None:
            try:
                backbone.to(device=orig_dev)
            except Exception:
                pass


def _collect_seeded_lora_grads(
    adapter: Any,
    backbone: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    gradient_checkpointing: bool,
    seed: int = 2026,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Run a single deterministic real forward+backward pass under `_inplace_safe_outer_autograd_context`
    (optionally activating `LoRATrainingPipeline._enable_gradient_checkpointing`) and return
    `(loss_value, {param_name: grad_clone})`.
    """
    from omni_lora.lora.layers import LoRALinear
    from omni_lora.pipelines.lora_trainer import (
        LoRATrainingPipeline,
        _inplace_safe_outer_autograd_context,
    )

    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.weight.grad = None
            m.lora_B.weight.grad = None

    restore_ckpt_fn = None
    if gradient_checkpointing:
        restore_ckpt_fn = LoRATrainingPipeline._enable_gradient_checkpointing(backbone)

    try:
        torch.manual_seed(seed)
        with _inplace_safe_outer_autograd_context(gradient_checkpointing=gradient_checkpointing):
            loss = adapter.compute_training_loss(
                backbone=backbone,
                batch=batch,
                device=device,
                dtype=dtype,
            )
            loss.backward()
        grads: Dict[str, torch.Tensor] = {}
        for name, mod in backbone.named_modules():
            if isinstance(mod, LoRALinear):
                if mod.lora_A.weight.grad is not None:
                    grads[f"{name}.lora_A"] = mod.lora_A.weight.grad.detach().clone()
                if mod.lora_B.weight.grad is not None:
                    grads[f"{name}.lora_B"] = mod.lora_B.weight.grad.detach().clone()
        return float(loss.detach().item()), grads
    finally:
        if callable(restore_ckpt_fn):
            restore_ckpt_fn()
        for m in backbone.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.weight.grad = None
                m.lora_B.weight.grad = None


def verify_real_model_contract(
    test_case: unittest.TestCase,
    spec: RealModelContractSpec,
    env: Any,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> None:
    """
    Execute all 7 hard invariants on a genuine, unmocked `RealModelContractSpec`:
      1. End-to-end `batch_size=2` (unequal prompt lengths) inside `torch.inference_mode()` with
         `StrictComfySpy` asserting 0 silent exceptions and proper `in_training` lifecycle.
      2. 100% non-zero LoRA gradient coverage (zero dead LoRAs outside `allow_unused_prefixes`).
      3. `gradient_checkpointing=True` vs `gradient_checkpointing=False` exact gradient parity.
      4. FP8 `get_dtype_inference()` resilience (when `spec.supports_fp8_extra_conds`).
      5. Dirty post-KSampler `inner_model.latent_shapes` isolation (when `spec.is_dual_stream_av`).
      6. Clean uninject with zero residual `LoRALinear` wrappers and exact real-forward parity.
      7. Native `comfy.sd.load_lora_for_models` zero-missing-keys round-trip + real patched forward delta.
    """
    from omni_lora import (
        OmniLoRADatasetLoader,
        OmniLoRALoader,
        OmniLoRAPrecomputeConditioning,
        OmniLoRASave,
        OmniLoRATrainer,
    )
    from omni_lora.lora import LoRAConfig, LoRAInjector, LoRALinear
    from omni_lora.models import resolve_model_adapter

    dev = device or torch.device("cpu")
    dm = spec.patcher.model.diffusion_model
    inner = spec.patcher.model

    adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)
    test_case.assertEqual(adapter.spec.objective_type, spec.expected_objective, f"[{spec.model_id}] objective")
    test_case.assertEqual(
        adapter.spec.latent_channels,
        spec.expected_latent_channels,
        f"[{spec.model_id}] latent_channels",
    )
    test_case.assertEqual(adapter.spec.supports_video_5d, spec.is_video_5d, f"[{spec.model_id}] supports_video_5d")

    dataset_node = OmniLoRADatasetLoader()
    precache_node = OmniLoRAPrecomputeConditioning()
    trainer_node = OmniLoRATrainer()
    save_node = OmniLoRASave()
    loader_node = OmniLoRALoader()

    gen = torch.Generator(device="cpu").manual_seed(314)
    rgb_images = torch.rand(2, spec.resolution, spec.resolution, 3, generator=gen, dtype=torch.float32)

    with torch.inference_mode():
        # Build a `batch_size=2` dataset with UNEQUAL prompt lengths so `attention_mask` is synthesized
        dataset_bs2, _ = dataset_node.load_dataset(
            dataset_path="",
            trigger_word="ohwx",
            default_caption="subject",
            resolution=spec.resolution,
            batch_size=2,
            num_repeats=1,
            center_crop=True,
            aspect_ratio_bucketing=False,
            random_flip=False,
            caption_dropout=0.0,
            images=rgb_images,
            custom_prompts="short caption\na much longer detailed multi word training prompt for sequence padding",
        )
        dataset_bs1, _ = dataset_node.load_dataset(
            dataset_path="",
            trigger_word="ohwx",
            default_caption="subject",
            resolution=spec.resolution,
            batch_size=1,
            num_repeats=1,
            center_crop=True,
            aspect_ratio_bucketing=False,
            random_flip=False,
            caption_dropout=0.0,
            images=rgb_images[:1],
            custom_prompts="single item prompt",
        )

        with StrictComfySpy(inner) as pre_spy:
            (cached_bs2,) = precache_node.precompute(
                dataset=dataset_bs2,
                offload_encoders_after_cache=False,
                vae=spec.vae,
                clip=spec.clip,
                model=spec.patcher,
            )
            (cached_bs1,) = precache_node.precompute(
                dataset=dataset_bs1,
                offload_encoders_after_cache=False,
                vae=spec.vae,
                clip=spec.clip,
                model=spec.patcher,
            )
        pre_spy.assert_no_silent_errors(test_case, spec.model_id)

        # Invariant 5 (Part A): Dirty post-KSampler `latent_shapes` must NOT alter single-stream precompute scaling
        if spec.is_dual_stream_av:
            stale_shapes = [(1, spec.expected_latent_channels, 2, 4, 4), (1, 32, 2, 16)]
            inner.latent_shapes = stale_shapes
            try:
                (cached_dirty,) = precache_node.precompute(
                    dataset=dataset_bs1,
                    offload_encoders_after_cache=False,
                    vae=spec.vae,
                    clip=spec.clip,
                    model=spec.patcher,
                )
                test_case.assertEqual(
                    inner.latent_shapes,
                    stale_shapes,
                    f"[{spec.model_id}] precompute failed to restore stale inner_model.latent_shapes",
                )
                test_case.assertTrue(
                    torch.allclose(
                        cached_bs1.batches[0]["latents"],
                        cached_dirty.batches[0]["latents"],
                        atol=1e-5,
                    ),
                    f"[{spec.model_id}] stale inner_model.latent_shapes poisoned precompute process_latent_in!",
                )
            finally:
                inner.latent_shapes = None

    # Capture REAL unmocked baseline output before training
    probe_batch = cached_bs1.batches[0]
    baseline_out = _run_real_forward_probe(spec.patcher, adapter, probe_batch, dev, dtype, patch_model_weights=False)
    test_case.assertTrue(torch.isfinite(baseline_out).all(), f"[{spec.model_id}] baseline real forward non-finite")

    # Invariants 2 & 3: 100% non-zero gradient coverage + `grad_ckpt=True` vs `False` gradient parity
    # (executed outside `inference_mode` just like inside `LoRATrainingPipeline.train`)
    with torch.inference_mode(False):
        batch_bs2 = cached_bs2.batches[0]
        injected = LoRAInjector.inject(
            dm,
            LoRAConfig(rank=4, alpha=4.0, target_modules=adapter.spec.default_target_modules),
            lora_dtype=dtype,
        )
        test_case.assertGreater(len(injected), 0, f"[{spec.model_id}] injected 0 LoRA layers")
        try:
            # Give lora_B small non-zero weights so both lora_A and lora_B receive gradients on step 1
            init_gen = torch.Generator(device="cpu").manual_seed(555)
            with torch.no_grad():
                for mod in dm.modules():
                    if isinstance(mod, LoRALinear):
                        b_vals = torch.randn(mod.lora_B.weight.shape, generator=init_gen, dtype=torch.float32) * 0.02
                        mod.lora_B.weight.copy_(b_vals.to(device=mod.lora_B.weight.device, dtype=mod.lora_B.weight.dtype))

            with StrictComfySpy(inner) as grad_spy:
                loss_no_ckpt, grads_no_ckpt = _collect_seeded_lora_grads(
                    adapter, dm, batch_bs2, dev, dtype, gradient_checkpointing=False, seed=2026
                )
                loss_ckpt, grads_ckpt = _collect_seeded_lora_grads(
                    adapter, dm, batch_bs2, dev, dtype, gradient_checkpointing=True, seed=2026
                )
            grad_spy.assert_no_silent_errors(test_case, spec.model_id)
            test_case.assertFalse(
                bool(getattr(comfy.model_management, "in_training", False)),
                f"[{spec.model_id}] comfy.model_management.in_training leaked True after forward!",
            )
            test_case.assertTrue(
                all(grad_spy.in_training_in_forward),
                f"[{spec.model_id}] comfy.model_management.in_training was False inside backbone.forward!",
            )
            if grad_spy.extra_conds_calls > 0:
                test_case.assertTrue(
                    all(grad_spy.in_training_in_extra_conds),
                    f"[{spec.model_id}] comfy.model_management.in_training was False inside extra_conds!",
                )

            test_case.assertTrue(math.isfinite(loss_no_ckpt), f"[{spec.model_id}] non-finite loss (grad_ckpt=False)")
            test_case.assertTrue(math.isfinite(loss_ckpt), f"[{spec.model_id}] non-finite loss (grad_ckpt=True)")
            test_case.assertAlmostEqual(
                loss_no_ckpt,
                loss_ckpt,
                places=3,
                msg=f"[{spec.model_id}] loss mismatch between grad_ckpt=False ({loss_no_ckpt}) and True ({loss_ckpt})",
            )

            # Invariant 2: EVERY injected LoRA layer (not explicitly allowlisted) MUST have non-zero grad!
            dead_layers: List[str] = []
            for mod_name in injected:
                if any(mod_name.startswith(pref) or mod_name == pref for pref in spec.allow_unused_prefixes):
                    continue
                g_a = grads_no_ckpt.get(f"{mod_name}.lora_A")
                g_b = grads_no_ckpt.get(f"{mod_name}.lora_B")
                a_ok = isinstance(g_a, torch.Tensor) and torch.isfinite(g_a).all() and float(g_a.abs().sum().item()) > 0.0
                b_ok = isinstance(g_b, torch.Tensor) and torch.isfinite(g_b).all() and float(g_b.abs().sum().item()) > 0.0
                if not (a_ok and b_ok):
                    dead_layers.append(mod_name)
            test_case.assertEqual(
                dead_layers,
                [],
                f"[{spec.model_id}] Dead zero-gradient LoRA layer(s) detected: {dead_layers}",
            )

            # Invariant 3: Exact gradient parity between `grad_ckpt=False` and `grad_ckpt=True`
            mismatched_grads: List[str] = []
            for key, g_ref in grads_ckpt.items():
                g_no = grads_no_ckpt.get(key)
                if g_no is None or not torch.allclose(g_ref.float(), g_no.float(), rtol=1e-2, atol=1e-3):
                    max_diff = float((g_ref.float() - g_no.float()).abs().max().item()) if g_no is not None else float("inf")
                    mismatched_grads.append(f"{key}(max_diff={max_diff:.3e})")
            test_case.assertEqual(
                mismatched_grads,
                [],
                f"[{spec.model_id}] Gradient mismatch between grad_ckpt=True and False (possible in-place corruption): "
                f"{mismatched_grads[:6]}",
            )

            # Invariant 4: FP8 `get_dtype_inference()` resilience during `extra_conds` + `preprocess_text_embeds`
            if spec.supports_fp8_extra_conds and hasattr(torch, "float8_e4m3fn"):
                orig_get_dtype_inf = inner.__dict__.get("get_dtype_inference")
                had_get_dtype_inf = "get_dtype_inference" in inner.__dict__
                inner.__dict__["get_dtype_inference"] = lambda: torch.float8_e4m3fn
                try:
                    with StrictComfySpy(inner) as fp8_spy:
                        fp8_loss, _ = _collect_seeded_lora_grads(
                            adapter, dm, batch_bs2, dev, dtype, gradient_checkpointing=False, seed=2027
                        )
                    fp8_spy.assert_no_silent_errors(test_case, f"{spec.model_id}[fp8_infer_dtype]")
                    test_case.assertTrue(
                        math.isfinite(fp8_loss),
                        f"[{spec.model_id}] FP8 get_dtype_inference training produced non-finite loss",
                    )
                finally:
                    if had_get_dtype_inf:
                        inner.__dict__["get_dtype_inference"] = orig_get_dtype_inf
                    else:
                        inner.__dict__.pop("get_dtype_inference", None)

            # Invariant 5 (Part B): Dirty post-KSampler `latent_shapes` during real training step
            if spec.is_dual_stream_av:
                stale_shapes = [(1, spec.expected_latent_channels, 2, 4, 4), (1, 32, 2, 16)]
                inner.latent_shapes = stale_shapes
                try:
                    with StrictComfySpy(inner) as dirty_spy:
                        dirty_loss, _ = _collect_seeded_lora_grads(
                            adapter, dm, batch_bs2, dev, dtype, gradient_checkpointing=False, seed=2026
                        )
                    dirty_spy.assert_no_silent_errors(test_case, f"{spec.model_id}[stale_latent_shapes]")
                    test_case.assertTrue(math.isfinite(dirty_loss))
                    test_case.assertEqual(
                        inner.latent_shapes,
                        stale_shapes,
                        f"[{spec.model_id}] training forward failed to restore stale inner_model.latent_shapes",
                    )
                finally:
                    inner.latent_shapes = None
        finally:
            LoRAInjector.uninject(dm)

    # Invariants 1, 6, 7: Full ComfyUI node pipeline (`OmniLoRATrainer` -> `OmniLoRASave` ->
    # `OmniLoRALoader` -> `comfy.sd.load_lora_for_models`) inside `torch.inference_mode()`!
    env.unloaded_lora_keys.clear()
    with torch.inference_mode():
        with StrictComfySpy(inner) as train_spy:
            model_out, lora_bundle, summary_json = trainer_node.train_lora(
                cached_data=cached_bs2,
                model=spec.patcher,
                model_architecture="auto",
                backend="cpu",
                vram_mode="auto",
                reserve_vram_gb=1.0,
                optimizer="adamw",
                rank=4,
                alpha=4.0,
                learning_rate=1e-2,
                lr_scheduler="constant",
                warmup_steps=0,
                steps=2,
                grad_accum_steps=1,
                gradient_checkpointing=False,
                lora_dtype="fp32",
                lora_plus_lr_ratio=1.0,
                use_rslora=False,
                seed=42,
            )
        train_spy.assert_no_silent_errors(test_case, f"{spec.model_id}[train_lora]")
        summary = json.loads(summary_json)
        test_case.assertGreater(summary["injected_layers"], 0)
        test_case.assertGreater(len(lora_bundle.state_dict), 0)

        # Invariant 6: Zero residual `LoRALinear` wrappers and exact baseline real-forward restoration
        for mod in dm.modules():
            test_case.assertNotIsInstance(
                mod,
                LoRALinear,
                f"[{spec.model_id}] residual LoRALinear left on diffusion_model after training!",
            )
        post_uninject_out = _run_real_forward_probe(
            spec.patcher, adapter, probe_batch, dev, dtype, patch_model_weights=False
        )
        test_case.assertTrue(
            torch.allclose(baseline_out, post_uninject_out, atol=1e-5),
            f"[{spec.model_id}] base diffusion_model drifted after uninject!",
        )

        # Invariant 7: Save -> reload -> `comfy.sd.load_lora_for_models` (0 missing keys) + real patched delta
        _, saved_filename, _ = save_node.save_lora(
            lora=lora_bundle,
            output_lora_name=f"contract_{spec.model_id}",
            append_step_count=True,
        )
        reloaded_bundle, _ = loader_node.load_lora(lora_name=saved_filename)
        patched_model, _ = comfy.sd.load_lora_for_models(
            spec.patcher,
            spec.clip,
            reloaded_bundle.state_dict,
            strength_model=1.0,
            strength_clip=1.0,
        )
        test_case.assertEqual(
            env.unloaded_lora_keys,
            [],
            f"[{spec.model_id}] Unloaded LoRA keys in ComfyUI LoraLoader: {env.unloaded_lora_keys}",
        )
        assert patched_model is not None
        test_case.assertGreater(len(patched_model.patches), 0)

        patched_out = _run_real_forward_probe(
            patched_model, adapter, probe_batch, dev, dtype, patch_model_weights=True
        )
        direct_out = _run_real_forward_probe(
            model_out, adapter, probe_batch, dev, dtype, patch_model_weights=True
        )
        test_case.assertTrue(torch.isfinite(patched_out).all(), f"[{spec.model_id}] patched real forward non-finite")
        test_case.assertTrue(
            torch.allclose(patched_out, direct_out, atol=1e-4),
            f"[{spec.model_id}] OmniLoRATrainer MODEL output differed from ComfyUI LoraLoader!",
        )
        test_case.assertFalse(
            torch.allclose(baseline_out, patched_out, atol=1e-7),
            f"[{spec.model_id}] Trained LoRA had zero measurable effect on real diffusion_model forward!",
        )


def verify_real_model_on_backend(
    spec: RealModelContractSpec,
    env: Any,
    backend_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Execute an unmocked real-forward training + real ComfyUI LoraLoader verification directly on
    the active accelerator (`xpu`, `cuda`, `rocm`, `tpu`, `mps`, `cpu`) in its native precision
    (`bf16` / `fp16` / `fp32`), returning structured metrics for the cloud verification table.

    Unlike mocked/bridged benchmarks, every forward, backward, gradient checkpoint recompute,
    and post-training ComfyUI `ModelPatcher.patch_model()` probe runs the genuine `comfy.ldm.*`
    and `comfy.model_base.*` code paths on the real hardware backend.
    """
    import time

    from omni_lora.lora import LoRALinear
    from omni_lora.models import resolve_model_adapter
    from omni_lora.nodes import (
        OmniLoRADatasetLoader,
        OmniLoRAPrecomputeConditioning,
        OmniLoRATrainer,
    )

    t0 = time.perf_counter()
    env.unloaded_lora_keys.clear()
    inner = spec.patcher.model
    dm = inner.diffusion_model

    adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)
    assert adapter.spec.objective_type == spec.expected_objective, (
        f"[{spec.model_id}] objective mismatch: got {adapter.spec.objective_type}, expected {spec.expected_objective}"
    )
    assert adapter.spec.latent_channels == spec.expected_latent_channels, (
        f"[{spec.model_id}] channels mismatch: got {adapter.spec.latent_channels}, expected {spec.expected_latent_channels}"
    )

    dataset_node = OmniLoRADatasetLoader()
    precache_node = OmniLoRAPrecomputeConditioning()
    trainer_node = OmniLoRATrainer()

    gen = torch.Generator(device="cpu").manual_seed(909)
    imgs = torch.rand(2, 64, 64, 3, generator=gen, dtype=torch.float32)

    with torch.inference_mode():
        ds, _ = dataset_node.load_dataset(
            dataset_path="",
            trigger_word="ohwx",
            default_caption="subject",
            resolution=64,
            batch_size=2,
            num_repeats=1,
            center_crop=True,
            aspect_ratio_bucketing=False,
            random_flip=False,
            caption_dropout=0.0,
            images=imgs,
            custom_prompts="short ohwx prompt\na much longer descriptive ohwx prompt with extra words",
            use_alpha_mask=False,
        )
        (cached,) = precache_node.precompute(
            dataset=ds,
            offload_encoders_after_cache=False,
            vae=spec.vae,
            clip=spec.clip,
            model=spec.patcher,
        )

        probe_batch = cached.batches[0]
        with torch.inference_mode(False):
            baseline_out = _run_real_forward_probe(
                spec.patcher, adapter, probe_batch, device, dtype, patch_model_weights=False
            )
        assert bool(torch.isfinite(baseline_out).all()), f"[{spec.model_id}] non-finite baseline on {device}/{dtype}"

        lora_dtype_str = "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32")
        # Use AdamW (normalized first-step update `±lr`, exercising the same `_fp32_master_step`
        # FP32 master-weight path on BF16/FP16) so 2-step smoke runs produce deltas comfortably above
        # a BF16 weight ULP (`~1.2e-4` at `|W| ~ 0.02`); RAdam intentionally runs unrectified SGD
        # (`rho_t <= 5`) for steps 1-4 and is separately verified over multi-step schedules.
        with StrictComfySpy(inner) as spy:
            model_out, lora_bundle, summary_json = trainer_node.train_lora(
                cached_data=cached,
                model=spec.patcher,
                model_architecture="auto",
                backend=backend_id,
                vram_mode="auto",
                reserve_vram_gb=1.0,
                optimizer="adamw",
                rank=4,
                alpha=4.0,
                learning_rate=1e-2,
                lr_scheduler="constant",
                warmup_steps=0,
                steps=2,
                grad_accum_steps=1,
                gradient_checkpointing=True,
                lora_dtype=lora_dtype_str,
                lora_plus_lr_ratio=1.0,
                use_rslora=True,
                seed=42,
            )

        if spy.extra_conds_errors or spy.process_latent_in_errors or spy.process_timestep_errors:
            raise AssertionError(
                f"[{spec.model_id}] silent ComfyUI errors on {backend_id}/{dtype}: "
                f"extra_conds={spy.extra_conds_errors}, "
                f"process_latent_in={spy.process_latent_in_errors}, "
                f"process_timestep={spy.process_timestep_errors}"
            )
        assert spy.forward_calls > 0, f"[{spec.model_id}] diffusion_model.forward was never called on {backend_id}/{dtype}!"

        summary = json.loads(summary_json)
        assert int(summary["injected_layers"]) > 0, f"[{spec.model_id}] injected 0 LoRA layers"
        assert math.isfinite(float(summary["final_loss"])), f"[{spec.model_id}] non-finite loss: {summary['final_loss']}"
        for mod in dm.modules():
            assert not isinstance(mod, LoRALinear), f"[{spec.model_id}] residual LoRALinear left after training!"

        lora_up_max = max(
            (float(v.detach().float().abs().max().item()) for k, v in lora_bundle.state_dict.items() if "lora_up" in k and isinstance(v, torch.Tensor)),
            default=0.0,
        )

        patched, _ = comfy.sd.load_lora_for_models(
            spec.patcher, spec.clip, lora_bundle.state_dict, strength_model=1.0, strength_clip=1.0
        )
        assert patched is not None, f"[{spec.model_id}] load_lora_for_models returned None"
        assert env.unloaded_lora_keys == [], f"[{spec.model_id}] unloaded LoRA keys: {env.unloaded_lora_keys}"

        with torch.inference_mode(False):
            patched_out = _run_real_forward_probe(
                patched, adapter, probe_batch, device, dtype, patch_model_weights=True
            )
        assert bool(torch.isfinite(patched_out).all()), f"[{spec.model_id}] non-finite patched output on {device}/{dtype}"
        drift = float((patched_out.float() - baseline_out.float()).abs().max().item())
        assert drift > 1e-6, (
            f"[{spec.model_id}] trained LoRA had zero measurable effect on {device}/{dtype} "
            f"(drift={drift}, lora_up_max={lora_up_max:.3e})"
        )

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "model_id": spec.model_id,
        "objective": str(summary["training_objective"]),
        "latent_channels": int(spec.expected_latent_channels),
        "injected_layers": int(summary["injected_layers"]),
        "final_loss": float(summary["final_loss"]),
        "drift": drift,
        "elapsed_ms": dt_ms,
    }

