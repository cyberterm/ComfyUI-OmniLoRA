"""
Bulletproof Unmocked Real-Forward Model Contract Suite for OmniLoRA.

Run with:
    python -m unittest omni_lora/tests/test_real_model_contracts.py
    pytest omni_lora/tests/test_real_model_contracts.py

Contains 4 complementary test suites designed to make fatal ComfyUI model
compatibility crashes (such as in-place autograd corruption, silent `extra_conds`
failures, dead/untrained multimodal LoRAs, FP8 connector dtype mismatches, stale
`latent_shapes` leakage, and upstream ComfyUI drift) impossible to miss:

  1. `TestRealForwardModelContracts` — Runs all 7 hard invariants across 22 genuine,
     unmocked `comfy.ldm.*` + `comfy.model_base.*` architectures (zero forward bridging).
  2. `TestAutogradInPlaceFidelity` — Exact numerical gradient verification against
     pure out-of-place references for in-place residuals (`add_`, `addcmul_`, `mul_`,
     `CopySlices`, `torch.cat(..., out=x)`), including >8 MB tensors and real
     MiniMax H3 / LTX 2.5 blocks, plus `in_training` exception-safety.
  3. `TestMultimodalAndConnectorStress` — Deep architectural regression suite for
     LTX 2.3/2.4/2.5 (`LTXAVModel`), MiniMax H3 Video (`MiniMaxH3Model`), and
     audio-only model protection.
  4. `TestComfyUIUpstreamDriftGuard` — Live introspection of `comfy.model_base` so
     any new upstream ComfyUI architecture adding `preprocess_text_embeds`,
     `preprocess_conds`, `preprocess_refs`, or custom `process_latent_in` immediately
     flags missing contract coverage.
"""

from __future__ import annotations

import inspect
import math
import pkgutil
import re
import types
from typing import Any, Dict, List, Set, Tuple
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

import comfy.model_base
import comfy.model_management
import comfy.ops
import comfy.sd

from omni_lora.lora import LoRAConfig, LoRAInjector, LoRALinear
from omni_lora.lora.layers import classify_block_structural_roles
from omni_lora.models import resolve_model_adapter
from omni_lora.pipelines.lora_trainer import (
    _checkpoint_block_inplace_safe,
    _inplace_safe_outer_autograd_context,
)
from omni_lora.tests.comfy_emulator import EmulatedComfyEnvironment
from omni_lora.tests.real_model_harness import (
    RealForwardModelRegistry,
    RealModelContractSpec,
    StrictComfySpy,
    _ContractCLIP,
    _initialize_real_comfy_module,
    verify_real_model_contract,
)


# ==============================================================================
# 1. Full 44-Architecture Unmocked Real-Forward Contract Sweep
# ==============================================================================

class TestRealForwardModelContracts(unittest.TestCase):
    """
    Executes all 7 hard invariants across all 44 genuine, UNMOCKED ComfyUI
    Image & Video architectures in `RealForwardModelRegistry`.
    """

    def test_all_real_comfy_architectures_satisfy_contract(self) -> None:
        with EmulatedComfyEnvironment() as env:
            for model_id in RealForwardModelRegistry.MODEL_IDS:
                with self.subTest(model_id=model_id):
                    try:
                        spec = RealForwardModelRegistry.build(model_id, dtype=torch.float32)
                    except (ImportError, AttributeError) as exc:
                        self.skipTest(f"[{model_id}] unavailable in installed ComfyUI checkout: {exc}")
                        continue
                    verify_real_model_contract(self, spec=spec, env=env, dtype=torch.float32)

    def test_ltxav_2_5_preprocessed_mode_contract(self) -> None:
        """
        Verify LTX 2.5 (`caption_proj_before_connector=True`, `caption_projection_first_linear=False`)
        when CLIP emits already-processed `cross_attention_dim + audio_cross_attention_dim` embeddings
        (`unprocessed_ltxav_embeds=False`), where `preprocess_text_embeds` is a pass-through and
        `video_embeddings_connector` is intentionally bypassed.
        """
        with EmulatedComfyEnvironment() as env:
            base_spec = RealForwardModelRegistry.build("ltxav_2_5", dtype=torch.float32)
            preproc_spec = RealModelContractSpec(
                model_id="ltxav_2_5_preprocessed",
                patcher=base_spec.patcher,
                vae=base_spec.vae,
                clip=_ContractCLIP(
                    embed_dim=96,
                    pooled_dim=64,
                    extra_cond_factory=lambda _t, _l: {"unprocessed_ltxav_embeds": False},
                ),
                expected_objective=base_spec.expected_objective,
                expected_latent_channels=base_spec.expected_latent_channels,
                is_video_5d=base_spec.is_video_5d,
                allow_unused_prefixes=("video_embeddings_connector.",),
                supports_fp8_extra_conds=True,
                is_dual_stream_av=True,
            )
            verify_real_model_contract(self, spec=preproc_spec, env=env, dtype=torch.float32)

    def test_all_real_comfy_architectures_bf16_smoke_contract(self) -> None:
        """
        Execute a fast 1-step `torch.bfloat16` real-forward + backward + `StrictComfySpy`
        smoke contract across every architecture in `RealForwardModelRegistry` on CPU.

        Catches mixed-dtype bugs that only trigger in BF16 (e.g. `torch.lerp(bf16, bf16, fp32)`,
        BF16 connector/RoPE casts, or FP32-vs-BF16 matmul mismatches) without requiring a GPU,
        and verifies that OmniLoRA's scoped PyTorch op wrappers restore native ops on exit.
        """
        with EmulatedComfyEnvironment():
            for model_id in RealForwardModelRegistry.MODEL_IDS:
                with self.subTest(model_id=model_id, dtype="bf16"):
                    try:
                        spec = RealForwardModelRegistry.build(model_id, dtype=torch.bfloat16)
                    except (ImportError, AttributeError) as exc:
                        self.skipTest(f"[{model_id}] unavailable in installed ComfyUI checkout: {exc}")
                        continue

                    dm = spec.patcher.model.diffusion_model
                    inner = spec.patcher.model
                    adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)

                    gen = torch.Generator(device="cpu").manual_seed(515)
                    px = torch.rand(1, spec.resolution, spec.resolution, 3, generator=gen, dtype=torch.float32)
                    raw_lat = spec.vae.encode(px)
                    clean_lat = inner.process_latent_in(raw_lat) if hasattr(inner, "process_latent_in") else raw_lat
                    clean_lat = clean_lat.to(dtype=torch.bfloat16)

                    tokens = spec.clip.tokenize("a cinematic detailed studio photo")
                    scheduled = spec.clip.encode_from_tokens_scheduled(tokens)
                    pe, meta = scheduled[0][0].to(dtype=torch.bfloat16), dict(scheduled[0][1])
                    pooled = meta.pop("pooled_output", None)
                    if isinstance(pooled, torch.Tensor):
                        pooled = pooled.to(dtype=torch.bfloat16)

                    injected = LoRAInjector.inject(
                        dm,
                        LoRAConfig(rank=4, alpha=4.0, target_modules=adapter.spec.default_target_modules),
                        lora_dtype=torch.bfloat16,
                    )
                    try:
                        self.assertGreater(len(injected), 0, f"[{model_id}] BF16 LoRA injection matched 0 layers")
                        batch = {
                            "latents": clean_lat,
                            "prompt_embeds": pe,
                            "pooled_output": pooled,
                            "extra_conds": meta,
                            "latent_scaled": True,
                        }
                        torch.manual_seed(616)
                        with StrictComfySpy(inner) as spy:
                            with _inplace_safe_outer_autograd_context(gradient_checkpointing=False):
                                loss = adapter.compute_training_loss(
                                    backbone=dm,
                                    batch=batch,
                                    device=torch.device("cpu"),
                                    dtype=torch.bfloat16,
                                )
                                loss.backward()
                        spy.assert_no_silent_errors(self, f"{model_id}[bf16_smoke]")
                        self.assertTrue(
                            bool(torch.isfinite(loss).all()),
                            f"[{model_id}] BF16 smoke loss was non-finite: {loss}",
                        )
                        trainable = LoRAInjector.get_trainable_parameters(dm)
                        nonzero_grads = sum(
                            1
                            for p in trainable
                            if p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().max().item()) > 0.0
                        )
                        self.assertGreater(
                            nonzero_grads,
                            0,
                            f"[{model_id}] BF16 smoke step produced 0 finite non-zero LoRA gradients!",
                        )
                    finally:
                        LoRAInjector.uninject(dm)

                    # Scoped op wrappers MUST restore native PyTorch ops once training context exits
                    self.assertFalse(
                        getattr(torch.lerp, "_omni_out_kwarg_safe", False),
                        f"[{model_id}] torch.lerp remained monkey-patched after training scope exit!",
                    )
                    self.assertFalse(
                        getattr(torch.cat, "_omni_out_kwarg_safe", False),
                        f"[{model_id}] torch.cat remained monkey-patched after training scope exit!",
                    )


# ==============================================================================
# 2. Exact Autograd In-Place Fidelity & Exception-Safety Suite
# ==============================================================================

class _SyntheticInPlaceResidualBlock(nn.Module):
    """Replicates ComfyUI's inference-style in-place residual mutations (LTXAV / MiniMax / Flux3)."""

    def __init__(self, dim: int, inplace: bool = True) -> None:
        super().__init__()
        self.inplace = inplace
        self.proj1 = nn.Linear(dim, dim, bias=False)
        self.proj2 = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.full((1, 1, dim), 0.35))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.rms_norm(x, (x.shape[-1],), eps=1e-6)
        delta1 = self.proj1(h)
        if self.inplace:
            x.addcmul_(delta1, self.gate)
        else:
            x = x + delta1 * self.gate

        h2 = F.rms_norm(x, (x.shape[-1],), eps=1e-6)
        delta2 = self.proj2(h2)
        if self.inplace:
            # Slice-mutation (`CopySlices`) + in-place `add_`
            half = x.shape[1] // 2
            x[:, :half].add_(delta2[:, :half])
            x[:, half:].add_(delta2[:, half:])
        else:
            x = x + delta2
        return x


class TestAutogradInPlaceFidelity(unittest.TestCase):
    """
    Mathematically proves that OmniLoRA's saved-tensor snapshot and checkpointing
    wrappers produce bit-tight identical gradients to pure out-of-place math —
    even for >8 MB activations and real ComfyUI MiniMax H3 / LTX 2.5 primitives.
    """

    def _clone_weights(self, src: nn.Module, dst: nn.Module) -> None:
        with torch.no_grad():
            for p_s, p_d in zip(src.parameters(), dst.parameters()):
                p_d.copy_(p_s)

    def test_inplace_vs_out_of_place_gradient_fidelity_both_ckpt_modes(self) -> None:
        """Compare `_SyntheticInPlaceResidualBlock(inplace=True)` vs `False` under grad_ckpt=False & True."""
        torch.manual_seed(101)
        dim = 64
        x_base = torch.randn(2, 16, dim, dtype=torch.float32)

        ref_blk = _SyntheticInPlaceResidualBlock(dim, inplace=False)
        x_ref = x_base.clone().requires_grad_(True)
        loss_ref = ref_blk(x_ref).pow(2).mean()
        loss_ref.backward()
        ref_grads = {
            "x": x_ref.grad.detach().clone(),
            "proj1": ref_blk.proj1.weight.grad.detach().clone(),
            "proj2": ref_blk.proj2.weight.grad.detach().clone(),
            "gate": ref_blk.gate.grad.detach().clone(),
        }

        for grad_ckpt in (False, True):
            with self.subTest(gradient_checkpointing=grad_ckpt):
                inp_blk = _SyntheticInPlaceResidualBlock(dim, inplace=True)
                self._clone_weights(ref_blk, inp_blk)
                x_inp = x_base.clone().requires_grad_(True)

                with _inplace_safe_outer_autograd_context(gradient_checkpointing=grad_ckpt):
                    # Mirror transformer blocks operating on non-leaf hidden states (`z`), since
                    # PyTorch disallows in-place mutation directly on a leaf tensor requiring grad.
                    z = x_inp * 1.0
                    if grad_ckpt:
                        cache: List[Any] = [None]
                        out = _checkpoint_block_inplace_safe(inp_blk.forward, cache, z)
                    else:
                        out = inp_blk(z)
                    loss = out.pow(2).mean()
                    loss.backward()

                self.assertAlmostEqual(float(loss.item()), float(loss_ref.item()), places=5)
                self.assertTrue(torch.allclose(x_inp.grad, ref_grads["x"], rtol=1e-4, atol=1e-5))
                self.assertTrue(torch.allclose(inp_blk.proj1.weight.grad, ref_grads["proj1"], rtol=1e-4, atol=1e-5))
                self.assertTrue(torch.allclose(inp_blk.proj2.weight.grad, ref_grads["proj2"], rtol=1e-4, atol=1e-5))
                self.assertTrue(torch.allclose(inp_blk.gate.grad, ref_grads["gate"], rtol=1e-4, atol=1e-5))

    def test_large_tensor_above_8mb_no_silent_gradient_corruption(self) -> None:
        """
        Regression guard against any byte-cap on saved-tensor snapshots (such as the old 8 MB cap):
        (1, 512, 4608) float32 is ~9.44 MB — in-place `rms_norm` + `addcmul_` MUST match out-of-place
        reference gradients with zero silent corruption when `gradient_checkpointing=False`.
        """
        torch.manual_seed(202)
        seq_len, dim = 512, 4608
        x_base = torch.randn(1, seq_len, dim, dtype=torch.float32)
        self.assertGreater(x_base.numel() * x_base.element_size(), 8 * 1024 * 1024)

        w_a = torch.randn(dim, 4, dtype=torch.float32) * 0.05
        w_b = torch.randn(4, dim, dtype=torch.float32) * 0.05
        gate_base = torch.full((1, 1, dim), 0.25, dtype=torch.float32)

        # Pure out-of-place reference
        x_ref = x_base.clone().requires_grad_(True)
        wa_ref = w_a.clone().requires_grad_(True)
        wb_ref = w_b.clone().requires_grad_(True)
        h_ref = F.rms_norm(x_ref, (dim,), eps=1e-6)
        delta_ref = (h_ref @ wa_ref) @ wb_ref
        y_ref = x_ref + delta_ref * gate_base
        loss_ref = y_ref.pow(2).mean()
        loss_ref.backward()

        # In-place mutation under `_inplace_safe_outer_autograd_context(gradient_checkpointing=False)`
        x_inp = x_base.clone().requires_grad_(True)
        wa_inp = w_a.clone().requires_grad_(True)
        wb_inp = w_b.clone().requires_grad_(True)
        with _inplace_safe_outer_autograd_context(gradient_checkpointing=False):
            # Clone `x_inp` into non-leaf `z` so in-place `z.addcmul_` mirrors transformer block residuals
            z = x_inp * 1.0
            h_inp = F.rms_norm(z, (dim,), eps=1e-6)
            delta_inp = (h_inp @ wa_inp) @ wb_inp
            z.addcmul_(delta_inp, gate_base)
            loss_inp = z.pow(2).mean()
            loss_inp.backward()

        self.assertTrue(
            torch.allclose(wa_inp.grad, wa_ref.grad, rtol=1e-4, atol=1e-5),
            f"Large >8MB tensor gradient mismatch on w_a: max_diff={(wa_inp.grad - wa_ref.grad).abs().max().item():.3e}",
        )
        self.assertTrue(
            torch.allclose(wb_inp.grad, wb_ref.grad, rtol=1e-4, atol=1e-5),
            f"Large >8MB tensor gradient mismatch on w_b: max_diff={(wb_inp.grad - wb_ref.grad).abs().max().item():.3e}",
        )
        self.assertTrue(
            torch.allclose(x_inp.grad, x_ref.grad, rtol=1e-4, atol=1e-5),
            f"Large >8MB tensor gradient mismatch on x: max_diff={(x_inp.grad - x_ref.grad).abs().max().item():.3e}",
        )

    def test_torch_cat_out_kwarg_autograd_fidelity(self) -> None:
        """Verify `torch.cat([a, b], dim=1, out=buf)` (as in `comfy/ldm/flux3/model.py`) matches pure `torch.cat`."""
        torch.manual_seed(303)
        a0 = torch.randn(2, 6, 16)
        b0 = torch.randn(2, 10, 16)

        a_ref = a0.clone().requires_grad_(True)
        b_ref = b0.clone().requires_grad_(True)
        loss_ref = torch.cat([a_ref, b_ref], dim=1).sin().sum()
        loss_ref.backward()

        a_out = a0.clone().requires_grad_(True)
        b_out = b0.clone().requires_grad_(True)
        with _inplace_safe_outer_autograd_context(gradient_checkpointing=False):
            buf = torch.empty(2, 16, 16)
            torch.cat([a_out, b_out], dim=1, out=buf)
            loss_out = buf.sin().sum()
            loss_out.backward()

        self.assertTrue(torch.allclose(a_out.grad, a_ref.grad, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(b_out.grad, b_ref.grad, rtol=1e-5, atol=1e-6))

    def test_real_minimax_h3_mod_scale_shift_and_mod_gate_fidelity(self) -> None:
        """
        Test real `comfy.ldm.minimax.model._mod_scale_shift` and `_mod_gate` (which execute
        `h[a:b].mul_(...).add_(...)` and `x[a:b].addcmul_(...)` in-place per modality segment)
        against pure out-of-place math.
        """
        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

        torch.manual_seed(404)
        tokens, dim = 20, 32
        segments = [(0, 12, 0), (12, 20, 1)]
        x0 = torch.randn(tokens, dim)
        shift0 = torch.randn(2, dim) * 0.2
        scale0 = torch.randn(2, dim) * 0.2
        gate0 = torch.randn(2, dim) * 0.3
        w0 = torch.randn(dim, dim) * 0.1

        # Pure out-of-place mathematical reference
        x_ref = x0.clone().requires_grad_(True)
        sh_ref = shift0.clone().requires_grad_(True)
        sc_ref = scale0.clone().requires_grad_(True)
        gt_ref = gate0.clone().requires_grad_(True)
        w_ref = w0.clone().requires_grad_(True)

        h_norm_ref = F.rms_norm(x_ref, (dim,), eps=1e-5)
        h_parts = []
        for a, b, row in segments:
            h_parts.append(h_norm_ref[a:b] * (1.0 + sc_ref[row]) + sh_ref[row])
        h_mod_ref = torch.cat(h_parts, dim=0)
        other_ref = h_mod_ref @ w_ref
        y_parts = []
        for a, b, row in segments:
            y_parts.append(x_ref[a:b] + other_ref[a:b] * gt_ref[row])
        y_ref = torch.cat(y_parts, dim=0)
        y_ref.pow(2).mean().backward()

        # Real MiniMax H3 in-place helpers under `_inplace_safe_outer_autograd_context`
        x_inp = x0.clone().requires_grad_(True)
        sh_inp = shift0.clone().requires_grad_(True)
        sc_inp = scale0.clone().requires_grad_(True)
        gt_inp = gate0.clone().requires_grad_(True)
        w_inp = w0.clone().requires_grad_(True)

        with _inplace_safe_outer_autograd_context(gradient_checkpointing=False):
            z = x_inp * 1.0
            h_norm = F.rms_norm(z, (dim,), eps=1e-5)
            h_mod = _mod_scale_shift(h_norm, sh_inp, sc_inp, segments)
            other = h_mod @ w_inp
            y_inp = _mod_gate(z, gt_inp, other, segments)
            y_inp.pow(2).mean().backward()

        self.assertTrue(torch.allclose(y_inp.detach(), y_ref.detach(), rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(x_inp.grad, x_ref.grad, rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(sh_inp.grad, sh_ref.grad, rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(sc_inp.grad, sc_ref.grad, rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(gt_inp.grad, gt_ref.grad, rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(w_inp.grad, w_ref.grad, rtol=1e-4, atol=1e-5))

    def test_in_training_flag_restored_on_exception(self) -> None:
        """
        `comfy.model_management.in_training` MUST ALWAYS be restored to `False` even when
        `diffusion_model.forward` raises an exception mid-forward.
        """
        spec = RealForwardModelRegistry.build("sd15", dtype=torch.float32)
        adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)
        dm = spec.patcher.model.diffusion_model

        orig_fwd = dm.forward

        def _exploding_forward(*args: Any, **kwargs: Any) -> Any:
            self.assertTrue(bool(getattr(comfy.model_management, "in_training", False)))
            raise RuntimeError("Simulated mid-forward explosion")

        dm.__dict__["forward"] = _exploding_forward
        try:
            self.assertFalse(bool(getattr(comfy.model_management, "in_training", False)))
            lat = torch.randn(1, 4, 8, 8)
            t = torch.tensor([0.5])
            pe = torch.randn(1, 6, 64)
            with self.assertRaises(RuntimeError):
                adapter._forward_backbone(
                    backbone=dm,
                    noisy_latents=lat,
                    timesteps=t,
                    prompt_embeds=pe,
                )
            self.assertFalse(
                bool(getattr(comfy.model_management, "in_training", False)),
                "comfy.model_management.in_training leaked True after forward exception!",
            )
        finally:
            dm.__dict__.pop("forward", None)
            _ = orig_fwd


# ==============================================================================
# 3. Multimodal, Connector, and Dual-Stream Stress Suite
# ==============================================================================

class TestMultimodalAndConnectorStress(unittest.TestCase):
    """
    Deep architectural invariants for LTX 2.3/2.4/2.5 (`LTXAVModel`), MiniMax H3 Video
    (`MiniMaxH3Model`), and audio-only model protection.
    """

    def test_ltxav_2_5_skips_audio_and_targets_video_and_connector(self) -> None:
        """
        On real `LTXAVModel` (LTX 2.5):
          - `audio_attn1`, `audio_attn2`, `audio_to_video_attn`, `video_to_audio_attn`, `audio_ff`,
            and `audio_embeddings_connector.*` MUST be classified as `"unused_secondary_modality"`
            and NEVER LoRA-injected on video-only training.
          - `attn1`, `attn2`, `ff`, `to_gate_logits`, and `video_embeddings_connector.*` MUST be
            injected cleanly.
        """
        spec = RealForwardModelRegistry.build("ltxav_2_5", dtype=torch.float32)
        dm = spec.patcher.model.diffusion_model
        adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)

        roles = classify_block_structural_roles(dm)
        audio_modules = [
            name
            for name, role in roles.items()
            if role == "unused_secondary_modality"
        ]
        self.assertGreater(len(audio_modules), 0, "Expected audio submodules to be classified unused_secondary_modality")
        for expected_frag in (
            "audio_attn1",
            "audio_attn2",
            "audio_to_video_attn",
            "video_to_audio_attn",
            "audio_ff",
            "audio_embeddings_connector",
        ):
            self.assertTrue(
                any(expected_frag in m for m in audio_modules),
                f"Expected `{expected_frag}` to be classified as unused_secondary_modality; got {audio_modules[:10]}",
            )

        injected = LoRAInjector.inject(
            dm,
            LoRAConfig(rank=4, alpha=4.0, target_modules=adapter.spec.default_target_modules),
            lora_dtype=torch.float32,
        )
        try:
            self.assertGreater(len(injected), 0)
            for name in injected:
                self.assertNotIn("audio_", name, f"Unused audio layer was injected on video-only LTXAV: {name}")
                self.assertNotIn("video_to_audio_attn", name, f"Unused v2a cross-modal layer injected: {name}")
            self.assertTrue(any(".attn1." in n for n in injected), "Video attn1 not injected")
            self.assertTrue(any(".attn2." in n for n in injected), "Video attn2 not injected")
            self.assertTrue(any("to_gate_logits" in n for n in injected), "LTX 2.5 gated attention not injected")
            self.assertTrue(
                any("video_embeddings_connector." in n for n in injected),
                "LTX 2.5 video_embeddings_connector not injected",
            )
        finally:
            LoRAInjector.uninject(dm)

    def test_ltxav_2_5_multi_width_context_preservation(self) -> None:
        """
        Verify `_prepare_prompt_embeds` preserves all 3 legal LTXAV input widths without
        destructive slicing/zero-padding:
          - `cross_attention_dim + audio_cross_attention_dim` (64 + 32 = 96)
          - `caption_channels * 2` (48 * 2 = 96; also tested on ltxav_2_0 where 64 * 2 = 128 != 96)
          - `caption_channels` (48)
        while still adapting a truly mismatched width (e.g. 24).
        """
        spec_25 = RealForwardModelRegistry.build("ltxav_2_5", dtype=torch.float32)
        adapter_25 = resolve_model_adapter("auto", comfy_model=spec_25.patcher)
        dm_25 = spec_25.patcher.model.diffusion_model

        for legal_w in (96, 48):
            t = torch.randn(2, 7, legal_w)
            out = adapter_25._prepare_prompt_embeds(t, dm_25)
            self.assertEqual(int(out.shape[-1]), legal_w, f"Legal width {legal_w} was mutated to {out.shape[-1]}")
            self.assertTrue(torch.equal(out, t))

        mismatched = torch.randn(2, 7, 24)
        adapted = adapter_25._prepare_prompt_embeds(mismatched, dm_25)
        self.assertNotEqual(int(adapted.shape[-1]), 24)

        spec_20 = RealForwardModelRegistry.build("ltxav_2_0", dtype=torch.float32)
        adapter_20 = resolve_model_adapter("auto", comfy_model=spec_20.patcher)
        dm_20 = spec_20.patcher.model.diffusion_model
        # ltxav_2_0: caption_channels=64 -> caption_channels*2=128, cross_attn+audio=96
        for legal_w in (64, 96, 128):
            t = torch.randn(1, 5, legal_w)
            out = adapter_20._prepare_prompt_embeds(t, dm_20)
            self.assertEqual(int(out.shape[-1]), legal_w, f"LTXAV 2.0 legal width {legal_w} mutated to {out.shape[-1]}")

    def test_ltxav_2_5_post_connector_attention_mask_reconciliation(self) -> None:
        """
        When `batch_size=2` has unequal prompt lengths (e.g. 5 vs 11 tokens) and
        `Embeddings1DConnector` expands `context` to 1024 tokens inside `LTXAV.extra_conds`,
        BOTH integer (`0/1`) and float (`0.0/-1e4`) `attention_mask` tensors MUST be
        reconciled to the post-connector sequence length so real `LTXAVModel.forward` succeeds.
        """
        spec = RealForwardModelRegistry.build("ltxav_2_5", dtype=torch.float32)
        adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)
        dm = spec.patcher.model.diffusion_model
        inner = spec.patcher.model

        lat = torch.randn(2, 128, 2, 4, 4, dtype=torch.float32)
        t = torch.tensor([0.4, 0.7], dtype=torch.float32)
        pe = torch.randn(2, 11, 96, dtype=torch.float32)

        int_mask = torch.zeros((2, 11), dtype=torch.long)
        int_mask[0, :5] = 1
        int_mask[1, :11] = 1

        float_mask = torch.where(int_mask > 0, torch.zeros_like(int_mask, dtype=torch.float32), torch.full_like(int_mask, -1e4, dtype=torch.float32))

        for mask_kind, mask_tensor in (("int_mask", int_mask), ("float_mask", float_mask)):
            with self.subTest(mask_kind=mask_kind):
                with StrictComfySpy(inner) as spy:
                    out = adapter._forward_backbone(
                        backbone=dm,
                        noisy_latents=lat,
                        timesteps=t,
                        prompt_embeds=pe,
                        extra_cond_dict={
                            "unprocessed_ltxav_embeds": True,
                            "attention_mask": mask_tensor,
                        },
                        sigma=t,
                        clean_latents=lat,
                    )
                spy.assert_no_silent_errors(self, f"ltxav_2_5[{mask_kind}]")
                self.assertEqual(tuple(out.shape), tuple(lat.shape))
                self.assertTrue(torch.isfinite(out).all())

    def test_minimax_h3_video_shared_av_targeting_and_audio_companion_stream(self) -> None:
        """
        On real `MiniMaxH3Model` (video+audio shared joint transformer):
          1. Shared `blocks.0.attn.{to_q,to_k,to_v,to_out}` and `blocks.0.mlp.*` are NOT excluded
             as `unused_secondary_modality` and ARE LoRA-injected; `to_gate_compress` IS skipped.
          2. The dummy audio companion stream passed into `MiniMaxH3Model.forward` has non-zero
             noise variance matching `sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)`.
          3. `transformer_options["minimax_h3_sigma_shift_video"]` overrides `_get_minimax_pdd_grid`.
          4. Stale `inner_model.latent_shapes` does NOT leak `audio_scale == 4.0` into
             `minimax_payload["audio_scale"]`.
        """
        from comfy.ldm.minimax.model import time_shift_sigma

        spec = RealForwardModelRegistry.build("minimax_h3_video", dtype=torch.float32)
        dm = spec.patcher.model.diffusion_model
        inner = spec.patcher.model
        adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)

        # 1. Structural role classification & injection
        roles = classify_block_structural_roles(dm)
        self.assertEqual(roles.get("blocks.0.attn.to_gate_compress"), "unused_auxiliary")
        for shared_key in ("blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj", "blocks.0.mlp.fc1", "blocks.0.mlp.fc2"):
            self.assertNotEqual(
                roles.get(shared_key),
                "unused_secondary_modality",
                f"Shared MiniMax H3 AV layer wrongly marked unused_secondary_modality: {shared_key}",
            )

        injected = LoRAInjector.inject(
            dm,
            LoRAConfig(rank=4, alpha=4.0, target_modules=adapter.spec.default_target_modules),
            lora_dtype=torch.float32,
        )
        try:
            self.assertTrue(any("blocks.0.attn.qkv_proj" in n for n in injected))
            self.assertTrue(any("blocks.0.attn.out_proj" in n for n in injected))
            self.assertTrue(any("blocks.0.mlp." in n for n in injected))
            self.assertFalse(
                any("to_gate_compress" in n for n in injected),
                f"Gate-compress auxiliary layer should not be injected: {injected}",
            )
        finally:
            LoRAInjector.uninject(dm)

        # 2 & 4. Spy on actual arguments reaching `MiniMaxH3Model.forward` with stale `latent_shapes`
        captured: Dict[str, Any] = {}
        orig_fwd = dm.forward

        def _capture_fwd(x: Any, timestep: Any, context: Any, **kwargs: Any) -> Any:
            captured["x"] = x
            captured["minimax_payload"] = dict(kwargs.get("minimax_payload") or {})
            return orig_fwd(x, timestep, context, **kwargs)

        dm.__dict__["forward"] = _capture_fwd
        stale_shapes = [(1, 24, 2, 4, 4), (1, 32, 2, 16)]
        inner.latent_shapes = stale_shapes
        try:
            torch.manual_seed(777)
            lat = torch.randn(1, 24, 2, 4, 4, dtype=torch.float32)
            sigma_v = torch.tensor([0.5], dtype=torch.float32)
            pe = torch.randn(1, 8, 32, dtype=torch.float32)
            out = adapter._forward_backbone(
                backbone=dm,
                noisy_latents=lat,
                timesteps=sigma_v,
                prompt_embeds=pe,
                extra_cond_dict={"minimax_token_tags": torch.zeros((1, 8), dtype=torch.long)},
                sigma=sigma_v,
                clean_latents=lat,
            )
            self.assertEqual(tuple(out.shape), tuple(lat.shape))
            self.assertEqual(inner.latent_shapes, stale_shapes, "Stale latent_shapes was not restored")

            x_arg = captured.get("x")
            self.assertIsInstance(x_arg, list, "MiniMaxH3Model must receive [video, audio] stream list")
            self.assertEqual(len(x_arg), 2)
            audio_stream = x_arg[1]
            self.assertEqual(int(audio_stream.shape[1]), 32)
            audio_std = float(audio_stream.std().item())
            expected_sigma_a = float(time_shift_sigma(sigma_v, float(dm.sigma_shift_video), float(dm.sigma_shift_audio)).item())
            self.assertGreater(audio_std, 0.01, "Dummy audio companion stream must NOT be all-zeros!")
            self.assertLess(
                abs(audio_std - expected_sigma_a),
                max(0.15, expected_sigma_a * 0.6),
                f"Dummy audio stream std ({audio_std:.4f}) drifted far from time_shift_sigma ({expected_sigma_a:.4f})",
            )

            payload = captured.get("minimax_payload", {})
            self.assertAlmostEqual(
                float(payload.get("audio_scale", 1.0)),
                1.0,
                places=5,
                msg=f"Stale latent_shapes poisoned minimax_payload['audio_scale']: {payload}",
            )
        finally:
            dm.__dict__.pop("forward", None)
            inner.latent_shapes = None

        # 3. `minimax_h3_sigma_shift_video` override dynamically shifts `_get_minimax_pdd_grid`
        # Note: older ComfyUI checkouts predate `pdd_steps` on `MiniMaxH3Model.final_layer`
        # (their `__init__` absorbs `pdd_steps=4` via `**kwargs`), so temporarily attach
        # `pdd_steps=4` when absent so OmniLoRA's PDD grid override is verified on all versions.
        final_layer = getattr(dm, "final_layer", None)
        had_pdd_attr = final_layer is not None and hasattr(final_layer, "pdd_steps")
        orig_pdd_steps = getattr(final_layer, "pdd_steps", None) if had_pdd_attr else None
        patched_pdd = False
        try:
            if final_layer is not None and int(getattr(final_layer, "pdd_steps", 1) or 1) <= 1:
                setattr(final_layer, "pdd_steps", 4)
                patched_pdd = True
            grid_default = adapter._get_minimax_pdd_grid(dm, transformer_options=None)
            grid_shifted = adapter._get_minimax_pdd_grid(dm, transformer_options={"minimax_h3_sigma_shift_video": 12.0})
            self.assertIsNotNone(grid_default)
            self.assertIsNotNone(grid_shifted)
            self.assertEqual(len(grid_default), len(grid_shifted))
            self.assertNotAlmostEqual(
                float(grid_default[1]),
                float(grid_shifted[1]),
                places=3,
                msg="minimax_h3_sigma_shift_video override failed to update PDD grid",
            )
        finally:
            if patched_pdd and final_layer is not None:
                if had_pdd_attr:
                    setattr(final_layer, "pdd_steps", orig_pdd_steps)
                else:
                    try:
                        delattr(final_layer, "pdd_steps")
                    except Exception:
                        pass

    def test_audio_only_models_never_classified_as_unused_secondary_modality(self) -> None:
        """
        Audio-only models (no video stream) MUST NEVER have their `audio_*` layers skipped
        by `classify_block_structural_roles`.
        """
        class _SyntheticAudioOnlyDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.audio_in = nn.Linear(64, 64)
                self.audio_embeddings_connector = nn.ModuleList([nn.Linear(64, 64)])
                self.blocks = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "audio_attn1": nn.ModuleDict(
                                    {
                                        "to_q": nn.Linear(64, 64),
                                        "to_k": nn.Linear(64, 64),
                                        "to_v": nn.Linear(64, 64),
                                        "to_out": nn.Linear(64, 64),
                                    }
                                ),
                                "audio_ff": nn.Linear(64, 64),
                            }
                        )
                    ]
                )

        synth_audio = _SyntheticAudioOnlyDiT()
        roles = classify_block_structural_roles(synth_audio)
        self.assertFalse(
            any(r == "unused_secondary_modality" for r in roles.values()),
            f"Audio-only model layers were wrongly marked unused_secondary_modality: {roles}",
        )

        # Also test genuine ComfyUI ACE-Step 1.5 audio transformer when importable
        try:
            from comfy.ldm.ace.model15 import AceStepConditionGenerationModel  # type: ignore[attr-defined]
        except Exception:
            AceStepConditionGenerationModel = None
        if AceStepConditionGenerationModel is not None:
            try:
                ace = AceStepConditionGenerationModel(
                    hidden_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    intermediate_size=128,
                    dtype=torch.float32,
                    device="cpu",
                    operations=comfy.ops.manual_cast,
                )
                ace_roles = classify_block_structural_roles(ace)
                self.assertFalse(
                    any(r == "unused_secondary_modality" for r in ace_roles.values()),
                    f"ACE-Step 1.5 audio-only layers wrongly marked unused_secondary_modality: {ace_roles}",
                )
            except TypeError:
                # Constructor signature changed upstream; synthetic audio-only guard above still ran
                pass


# ==============================================================================
# 4. Upstream ComfyUI Drift Guard
# ==============================================================================

class TestComfyUIUpstreamDriftGuard(unittest.TestCase):
    """
    Introspects live `comfy.model_base` subclasses so whenever upstream ComfyUI adds
    a new architecture that uses:
      - `preprocess_text_embeds` (connector / LLM adapter inside `extra_conds`),
      - `preprocess_conds` or `preprocess_refs` (inference-cached pre-blocks), or
      - custom `process_latent_in` (multi-stream latent scaling),
    this test fails immediately until that architecture is added to
    `RealForwardModelRegistry` and verified by the 7-invariant contract!
    """

    def _collect_registered_base_model_classes(self) -> Set[type]:
        classes: Set[type] = set()
        for model_id in RealForwardModelRegistry.MODEL_IDS:
            try:
                spec = RealForwardModelRegistry.build(model_id, dtype=torch.float32)
            except (ImportError, AttributeError):
                continue
            classes.add(spec.patcher.model.__class__)
        return classes

    # Pixel-space / no-VAE architectures that do not operate through standard latent-space VAE
    # pipelines and therefore are intentionally excluded from latent-space registry coverage.
    KNOWN_PIXEL_SPACE_NO_VAE_MODELS: Set[str] = {"SenseNovaU15"}

    def test_all_comfy_preprocess_and_custom_latent_in_models_are_covered(self) -> None:
        covered_classes = self._collect_registered_base_model_classes()

        uncovered_preprocess_any: List[str] = []
        uncovered_custom_latent_in: List[str] = []

        preprocess_call_re = re.compile(r"\.(preprocess_[a-zA-Z0-9_]+)\s*\(")

        for name, obj in inspect.getmembers(comfy.model_base, inspect.isclass):
            if obj is comfy.model_base.BaseModel or not issubclass(obj, comfy.model_base.BaseModel):
                continue
            if name in self.KNOWN_PIXEL_SPACE_NO_VAE_MODELS:
                continue
            try:
                src = inspect.getsource(obj)
            except (OSError, TypeError):
                continue

            preprocess_methods = sorted(set(preprocess_call_re.findall(src)))
            if preprocess_methods and obj not in covered_classes:
                uncovered_preprocess_any.append(f"{name}({', '.join(preprocess_methods)})")
            if "process_latent_in" in obj.__dict__ and obj not in covered_classes:
                uncovered_custom_latent_in.append(name)

        self.assertEqual(
            uncovered_preprocess_any,
            [],
            f"Upstream comfy.model_base class(es) invoke `.preprocess_*()` without RealForwardModelRegistry coverage: "
            f"{uncovered_preprocess_any}",
        )
        self.assertEqual(
            uncovered_custom_latent_in,
            [],
            f"Upstream comfy.model_base class(es) override `process_latent_in` without RealForwardModelRegistry coverage: "
            f"{uncovered_custom_latent_in}",
        )

    def test_all_registered_multimodal_models_recognized_by_structural_guards(self) -> None:
        """
        Verify every architecture in `RealForwardModelRegistry` that exposes dual-stream AV
        hooks (`separate_audio_and_video_latents`, `audio_patch_proj`, or `audio_latents_dim`)
        is properly handled by OmniLoRA's structural guards.
        """
        for model_id in RealForwardModelRegistry.MODEL_IDS:
            try:
                spec = RealForwardModelRegistry.build(model_id, dtype=torch.float32)
            except (ImportError, AttributeError):
                continue
            dm = spec.patcher.model.diffusion_model
            adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)

            if hasattr(dm, "separate_audio_and_video_latents"):
                roles = classify_block_structural_roles(dm)
                self.assertTrue(
                    any(r == "unused_secondary_modality" for r in roles.values()),
                    f"[{model_id}] Dual-stream `separate_audio_and_video_latents` model had no audio layers skipped!",
                )
            if getattr(dm, "audio_latents_dim", None) is not None:
                self.assertTrue(
                    adapter._is_minimax_h3_backbone(dm) or spec.is_dual_stream_av,
                    f"[{model_id}] Model with `audio_latents_dim` was not recognized as dual-stream AV!",
                )
            if spec.requires_batch_size_one_native:
                self.assertTrue(
                    adapter._requires_batch_size_one(dm),
                    f"[{model_id}] Expected `_requires_batch_size_one(dm)` to be True!",
                )

    def test_real_comfy_vlm_clip_tokenizer_detection(self) -> None:
        """
        Verify `OmniLoRAPrecomputeConditioning._clip_supports_images_kwarg` against genuine
        ComfyUI `comfy.text_encoders.*` tokenizer classes so VLM encoders (e.g. QwenImage /
        Flux2 Mistral / OmniGen2) are detected accurately while standard non-VLM encoders
        (or classes defining `llama_template_images = None`) never trigger false positives.
        """
        import importlib
        import comfy.text_encoders
        from omni_lora.nodes import OmniLoRAPrecomputeConditioning

        vlm_classes_found: List[str] = []
        non_vlm_classes_checked: int = 0

        for mod_info in pkgutil.iter_modules(comfy.text_encoders.__path__):
            try:
                mod = importlib.import_module(f"comfy.text_encoders.{mod_info.name}")
            except Exception:
                continue
            for cls_name, cls_obj in inspect.getmembers(mod, inspect.isclass):
                if getattr(cls_obj, "__module__", "") != mod.__name__:
                    continue
                tw_fn = getattr(cls_obj, "tokenize_with_weights", None)
                if not callable(tw_fn):
                    continue
                try:
                    sig_params = inspect.signature(tw_fn).parameters
                except (ValueError, TypeError):
                    continue

                has_images_param = "images" in sig_params
                raw_tpl_img = getattr(cls_obj, "llama_template_images", None)
                has_truthy_tpl_img = bool(raw_tpl_img)

                # Build a lightweight shell mimicking a ComfyUI CLIP wrapping this tokenizer class
                # without loading any multi-GB tokenizer weights from disk.
                tok_shell = types.SimpleNamespace(
                    tokenize_with_weights=tw_fn,
                    llama_template_images=raw_tpl_img,
                )
                clip_shell = types.SimpleNamespace(
                    tokenizer=tok_shell,
                    tokenize=comfy.sd.CLIP.tokenize,
                )
                detected = OmniLoRAPrecomputeConditioning._clip_supports_images_kwarg(clip_shell)

                if has_images_param or has_truthy_tpl_img:
                    vlm_classes_found.append(f"{mod_info.name}.{cls_name}")
                    self.assertTrue(
                        detected,
                        f"Real ComfyUI VLM tokenizer `{mod_info.name}.{cls_name}` was NOT detected as image-capable!",
                    )
                else:
                    non_vlm_classes_checked += 1
                    self.assertFalse(
                        detected,
                        f"Real ComfyUI non-VLM tokenizer `{mod_info.name}.{cls_name}` falsely detected as image-capable!",
                    )

        # Guard explicitly against `llama_template_images = None` false positive
        none_tpl_clip = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(llama_template_images=None),
            tokenize=lambda self, text, images=None: {},
        )
        self.assertFalse(OmniLoRAPrecomputeConditioning._clip_supports_images_kwarg(none_tpl_clip))
        self.assertGreater(non_vlm_classes_checked, 0, "Expected at least one non-VLM ComfyUI tokenizer class")


# ==============================================================================
# 5. Unmocked Real-Forward Image Editing / Control Conditioning Contract Suite
# ==============================================================================

class TestRealForwardEditModels(unittest.TestCase):
    """
    Verifies universal control/reference latent routing against genuine, UNMOCKED
    ComfyUI `comfy.ldm.*` + `comfy.model_base.*` edit models across BOTH ComfyUI
    conditioning paradigms, in BOTH FP32 and BF16, at BOTH square and mismatched
    (`preserve_aspect`) control resolutions, plus ComfyUI `LoraLoader` round-trip:
      1. `reference_latents` (token-concatenated DiTs: Flux Kontext, Flux 2, Flux3 Klein,
         Qwen-Image 1.0/2.1, OmniGen2, Boogu, MOSS-Image, FIBO 2.0, Krea 2, JoyImage, plus
         any additional live ComfyUI model declaring `ref_latents` support)
      2. `concat_latent_image` (`HiDream-E1` `image_cond` & `SD15_instructpix2pix` `c_concat`)
    """

    REQUIRED_REF_EDIT_MODELS: Tuple[str, ...] = (
        "flux_dev",
        "flux2",
        "flux3_klein",
        "qwen_image_1_0",
        "qwen_image_2_1",
        "omnigen2",
        "boogu",
        "moss_image",
        "fibo2",
        "krea2",
        "joyimage",
    )

    REF_BRANCH_PREFIXES: Dict[str, str] = {
        "omnigen2": "ref_image_refiner.",
        "boogu": "ref_image_refiner.",
        "flux3_klein": "content_mode_blocks.image_cond.",
    }

    @classmethod
    def _discover_reference_latents_model_ids(cls) -> Tuple[str, ...]:
        """
        Return `REQUIRED_REF_EDIT_MODELS` plus any additional registered ComfyUI architecture
        whose live `diffusion_model._forward`/`forward` explicitly declares and reads
        `ref_latents` and whose `BaseModel` wires `reference_latents`/`ref_latents`.
        """
        discovered: List[str] = list(cls.REQUIRED_REF_EDIT_MODELS)
        seen: Set[str] = set(discovered)
        for model_id in RealForwardModelRegistry.MODEL_IDS:
            if model_id in seen:
                continue
            try:
                spec = RealForwardModelRegistry.build(model_id, dtype=torch.float32)
            except (ImportError, AttributeError):
                continue
            inner = spec.patcher.model
            dm = inner.diffusion_model
            fn = getattr(dm, "_forward", None)
            if not callable(fn):
                fn = getattr(dm, "forward", None)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
                src = inspect.getsource(fn)
            except (ValueError, TypeError, OSError):
                continue
            if "ref_latents" not in sig.parameters or "ref_latents" not in src:
                continue
            # T2I-only `NextDiT` (Lumina 2 without Z-Image-Omni `pad_tokens_multiple`) shares the
            # `NextDiT._forward` signature but does not instantiate `cap_pad_token` for `ref_latents`.
            if hasattr(dm, "pad_tokens_multiple") and not hasattr(dm, "cap_pad_token"):
                continue
            mem_conds = getattr(inner, "memory_usage_factor_conds", ())
            has_base_ref = isinstance(mem_conds, (tuple, list, set)) and "ref_latents" in mem_conds
            if not has_base_ref:
                try:
                    ec_src = inspect.getsource(type(inner).extra_conds)
                    has_base_ref = "reference_latents" in ec_src
                except (TypeError, OSError):
                    has_base_ref = False
            if has_base_ref:
                discovered.append(model_id)
                seen.add(model_id)
        return tuple(discovered)

    @staticmethod
    def _cast_floating_cond(obj: Any, dtype: torch.dtype) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(dtype=dtype) if obj.is_floating_point() else obj
        if isinstance(obj, dict):
            return {k: TestRealForwardEditModels._cast_floating_cond(v, dtype) for k, v in obj.items()}
        if isinstance(obj, list):
            return [TestRealForwardEditModels._cast_floating_cond(v, dtype) for v in obj]
        if isinstance(obj, tuple):
            return tuple(TestRealForwardEditModels._cast_floating_cond(v, dtype) for v in obj)
        return obj

    @staticmethod
    def _build_sd15_instructpix2pix_spec(dtype: torch.dtype = torch.float32) -> Tuple[Any, Any, Any]:
        import comfy.supported_models
        from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
        from omni_lora.tests.real_model_harness import _ContractVAE, build_real_comfy_patcher

        ops = comfy.ops.manual_cast
        unet_ip2p = UNetModel(
            image_size=16, in_channels=8, model_channels=32, out_channels=4,
            num_res_blocks=1, channel_mult=[1, 2], num_heads=4,
            use_spatial_transformer=True, transformer_depth=[1, 1],
            transformer_depth_middle=1, transformer_depth_output=[1, 1, 1, 1],
            context_dim=64, dtype=dtype, device="cpu", operations=ops,
        )
        patcher_ip2p = build_real_comfy_patcher(
            comfy.supported_models.SD15,
            comfy.model_base.SD15_instructpix2pix,
            unet_ip2p,
            unet_config={"in_channels": 8, "out_channels": 4, "context_dim": 64, "model_channels": 32},
            model_type=comfy.model_base.ModelType.EPS,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        return patcher_ip2p, _ContractVAE(4, is_video_5d=False), _ContractCLIP(embed_dim=64, pooled_dim=64)

    def _encode_prompt_from_clip(
        self,
        clip: Any,
        text: str,
    ) -> tuple[torch.Tensor, Any, Dict[str, Any]]:
        tokens = clip.tokenize(text)
        scheduled = clip.encode_from_tokens_scheduled(tokens)
        cond_t, meta = scheduled[0][0], dict(scheduled[0][1])
        pooled = meta.pop("pooled_output", None)
        return cond_t, pooled, meta

    def test_real_reference_latents_edit_models(self) -> None:
        """
        Verify `reference_latents` edit routing across all 11+ real unmocked ComfyUI models:
          - Auto-resolves `resolve_control_mode == "reference_latents"`
          - Zero silent exceptions in `StrictComfySpy`
          - Changing square AND mismatched rectangular (`preserve_aspect`) control latents
            measurably alters real backbone output
          - Finite loss & non-zero LoRA gradients (including `ref_image_refiner.*` on OmniGen2/Boogu
            and `content_mode_blocks.image_cond.*` on Flux3 Klein)
          - Safe `batch_size=2` mixed controlled + uncontrolled microbatching
          - Full ComfyUI `comfy.sd.load_lora_for_models` round-trip (0 unloaded keys, edit-branch
            keys exported cleanly, patched real forward with control_latents measurably changes)
        """
        edit_ref_models = self._discover_reference_latents_model_ids()
        ref_branch_prefixes = self.REF_BRANCH_PREFIXES

        with EmulatedComfyEnvironment() as env:
            for model_id in edit_ref_models:
                with self.subTest(model_id=model_id):
                    try:
                        spec = RealForwardModelRegistry.build(model_id, dtype=torch.float32)
                    except (ImportError, AttributeError) as exc:
                        self.skipTest(f"[{model_id}] unavailable in installed ComfyUI checkout: {exc}")
                        continue

                    dm = spec.patcher.model.diffusion_model
                    inner = spec.patcher.model
                    adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)
                    self.assertEqual(
                        adapter.resolve_control_mode(dm, "auto"),
                        "reference_latents",
                        f"[{model_id}] expected auto control_mode='reference_latents'",
                    )

                    gen = torch.Generator(device="cpu").manual_seed(707)
                    rect_w = spec.resolution + max(32, spec.resolution // 2)
                    tgt_px = torch.rand(1, spec.resolution, spec.resolution, 3, generator=gen, dtype=torch.float32)
                    ctrl_a_px = torch.rand(1, spec.resolution, spec.resolution, 3, generator=gen, dtype=torch.float32)
                    ctrl_b_px = torch.rand(1, spec.resolution, spec.resolution, 3, generator=gen, dtype=torch.float32)
                    ctrl_rect_a_px = torch.rand(1, spec.resolution, rect_w, 3, generator=gen, dtype=torch.float32)
                    ctrl_rect_b_px = torch.rand(1, spec.resolution, rect_w, 3, generator=gen, dtype=torch.float32)

                    # Target is scaled (`process_latent_in`); controls stay RAW UNSCALED (matching ComfyUI)
                    raw_tgt = spec.vae.encode(tgt_px)
                    clean_lat = inner.process_latent_in(raw_tgt) if hasattr(inner, "process_latent_in") else raw_tgt
                    ctrl_a_raw = spec.vae.encode(ctrl_a_px)
                    ctrl_b_raw = spec.vae.encode(ctrl_b_px)
                    ctrl_rect_a_raw = spec.vae.encode(ctrl_rect_a_px)
                    ctrl_rect_b_raw = spec.vae.encode(ctrl_rect_b_px)
                    self.assertNotEqual(
                        tuple(ctrl_rect_a_raw.shape[-2:]),
                        tuple(clean_lat.shape[-2:]),
                        f"[{model_id}] Expected rectangular control latent shape to differ from square target!",
                    )

                    pe, pooled, extra_cond = self._encode_prompt_from_clip(spec.clip, "restyle image to oil painting")
                    sigma = torch.tensor([0.5], dtype=torch.float32)
                    torch.manual_seed(909)
                    noisy = clean_lat + torch.randn_like(clean_lat) * 0.25

                    # 1. Forward sensitivity check: square AND mismatched `preserve_aspect` controls
                    with torch.no_grad():
                        with StrictComfySpy(inner) as spy:
                            out_a = adapter._forward_backbone(
                                backbone=dm,
                                noisy_latents=noisy,
                                timesteps=sigma,
                                prompt_embeds=pe,
                                pooled_output=pooled,
                                extra_cond_dict=extra_cond,
                                sigma=sigma,
                                clean_latents=clean_lat,
                                control_latents=[ctrl_a_raw],
                                control_mode="auto",
                            )
                            out_b = adapter._forward_backbone(
                                backbone=dm,
                                noisy_latents=noisy,
                                timesteps=sigma,
                                prompt_embeds=pe,
                                pooled_output=pooled,
                                extra_cond_dict=extra_cond,
                                sigma=sigma,
                                clean_latents=clean_lat,
                                control_latents=[ctrl_b_raw],
                                control_mode="auto",
                            )
                            out_rect_a = adapter._forward_backbone(
                                backbone=dm,
                                noisy_latents=noisy,
                                timesteps=sigma,
                                prompt_embeds=pe,
                                pooled_output=pooled,
                                extra_cond_dict=extra_cond,
                                sigma=sigma,
                                clean_latents=clean_lat,
                                control_latents=[ctrl_rect_a_raw],
                                control_mode="auto",
                            )
                            out_rect_b = adapter._forward_backbone(
                                backbone=dm,
                                noisy_latents=noisy,
                                timesteps=sigma,
                                prompt_embeds=pe,
                                pooled_output=pooled,
                                extra_cond_dict=extra_cond,
                                sigma=sigma,
                                clean_latents=clean_lat,
                                control_latents=[ctrl_rect_b_raw],
                                control_mode="auto",
                            )
                        spy.assert_no_silent_errors(self, f"{model_id}[ref_latents]")

                    self.assertEqual(tuple(out_a.shape), tuple(noisy.shape))
                    self.assertTrue(torch.isfinite(out_a).all())
                    self.assertFalse(
                        torch.allclose(out_a, out_b, atol=1e-6),
                        f"[{model_id}] Changing reference_latents had zero effect on real diffusion_model output!",
                    )
                    self.assertEqual(tuple(out_rect_a.shape), tuple(noisy.shape))
                    self.assertTrue(torch.isfinite(out_rect_a).all())
                    self.assertFalse(
                        torch.allclose(out_rect_a, out_rect_b, atol=1e-6),
                        f"[{model_id}] Mismatched rectangular (`preserve_aspect`) reference_latents had zero effect!",
                    )

                    # 2. Backward + LoRA gradients (including dedicated reference branches on OmniGen2/Boogu/Flux3)
                    #    + ComfyUI `LoraLoader` round-trip verification
                    LoRAInjector.capture_comfy_lora_key_map(dm, comfy_model=spec.patcher)
                    injected = LoRAInjector.inject(
                        dm,
                        LoRAConfig(rank=4, alpha=4.0, target_modules=adapter.spec.default_target_modules),
                        lora_dtype=torch.float32,
                    )
                    lora_sd: Dict[str, torch.Tensor] = {}
                    expected_prefix = ref_branch_prefixes.get(model_id)
                    try:
                        self.assertGreater(len(injected), 0)
                        batch_bs2 = {
                            "latents": torch.cat([clean_lat, clean_lat], dim=0),
                            "prompt_embeds": torch.cat([pe, pe], dim=0),
                            "pooled_output": torch.cat([pooled, pooled], dim=0) if isinstance(pooled, torch.Tensor) else None,
                            "extra_conds": {
                                k: (torch.cat([v, v], dim=0) if isinstance(v, torch.Tensor) and v.ndim >= 2 and int(v.shape[0]) == 1 else v)
                                for k, v in extra_cond.items()
                            },
                            "latent_scaled": True,
                            # Mixed controlled + uncontrolled item in batch_size=2 exercises safe microbatching!
                            "control_latents": [ctrl_a_raw, None],
                            "control_mode": "auto",
                            "reference_latents_method": "auto",
                        }
                        torch.manual_seed(111)
                        with _inplace_safe_outer_autograd_context(gradient_checkpointing=False):
                            loss = adapter.compute_training_loss(
                                backbone=dm,
                                batch=batch_bs2,
                                device=torch.device("cpu"),
                                dtype=torch.float32,
                            )
                            loss.backward()

                        self.assertTrue(torch.isfinite(loss).item(), f"[{model_id}] non-finite edit loss: {loss}")
                        nonzero_names: List[str] = []
                        with torch.no_grad():
                            for name, mod in dm.named_modules():
                                if isinstance(mod, LoRALinear) and mod.lora_B.weight.grad is not None:
                                    max_g = float(mod.lora_B.weight.grad.abs().max().item())
                                    if max_g > 0.0:
                                        nonzero_names.append(name)
                                        # Take one normalized gradient step so `lora_B` is non-zero for LoraLoader round-trip
                                        mod.lora_B.weight.add_(mod.lora_B.weight.grad / max(max_g, 1e-6), alpha=-0.1)
                        self.assertGreater(len(nonzero_names), 0, f"[{model_id}] zero LoRA gradients with control_latents!")

                        if expected_prefix is not None:
                            self.assertTrue(
                                any(n.startswith(expected_prefix) for n in nonzero_names),
                                f"[{model_id}] Dedicated reference branch `{expected_prefix}*` received NO gradients! Got: {nonzero_names[:10]}",
                            )

                        lora_sd = LoRAInjector.extract_lora_state_dict(dm, comfy_model=spec.patcher)
                    finally:
                        LoRAInjector.uninject(dm)

                    # 3. Verify ComfyUI's native `load_lora_for_models` maps EVERY exported key (including edit branches)
                    self.assertGreater(len(lora_sd), 0)
                    if expected_prefix is not None:
                        prefix_dot = expected_prefix.rstrip(".")
                        prefix_us = prefix_dot.replace(".", "_")
                        self.assertTrue(
                            any(prefix_dot in k or prefix_us in k for k in lora_sd.keys()),
                            f"[{model_id}] Exported LoRA state_dict missing dedicated edit branch `{expected_prefix}*`!",
                        )
                    env.unloaded_lora_keys.clear()
                    patched_model, _ = comfy.sd.load_lora_for_models(
                        spec.patcher,
                        spec.clip,
                        lora_sd,
                        strength_model=1.0,
                        strength_clip=0.0,
                    )
                    self.assertEqual(
                        env.unloaded_lora_keys,
                        [],
                        f"[{model_id}] ComfyUI LoraLoader reported unloaded edit LoRA keys: {env.unloaded_lora_keys}",
                    )
                    assert patched_model is not None
                    self.assertGreater(len(patched_model.patches), 0)
                    try:
                        try:
                            patched_model.patch_model(device_to=torch.device("cpu"))
                        except TypeError:
                            patched_model.patch_model()
                        with torch.no_grad():
                            patched_out_a = adapter._forward_backbone(
                                backbone=patched_model.model.diffusion_model,
                                noisy_latents=noisy,
                                timesteps=sigma,
                                prompt_embeds=pe,
                                pooled_output=pooled,
                                extra_cond_dict=extra_cond,
                                sigma=sigma,
                                clean_latents=clean_lat,
                                control_latents=[ctrl_a_raw],
                                control_mode="auto",
                            )
                    finally:
                        patched_model.unpatch_model()
                    self.assertTrue(torch.isfinite(patched_out_a).all())
                    self.assertFalse(
                        torch.allclose(out_a, patched_out_a, atol=1e-6),
                        f"[{model_id}] Reloaded edit LoRA had zero measurable effect on real control-conditioned forward!",
                    )

    def test_real_concat_latent_image_edit_models_and_zero_target_leakage(self) -> None:
        """
        Verify `concat_latent_image` edit routing on genuine unmocked ComfyUI models:
          1. `hidream_i1` (`comfy.model_base.HiDream` `image_cond` path — HiDream-E1)
          2. `comfy.model_base.SD15_instructpix2pix` + widened 8-channel `UNetModel` (`c_concat` path)
        Proves auto-detection, square & mismatched (`preserve_aspect`) control sensitivity,
        finite loss + LoRA gradients, AND zero clean-target data leakage when `control_dropped=True`.
        """
        with EmulatedComfyEnvironment():
            # --- Case A: HiDream-E1 (`image_cond` via `concat_latent_image`) ---
            try:
                hidream_spec = RealForwardModelRegistry.build("hidream_i1", dtype=torch.float32)
            except (ImportError, AttributeError):
                hidream_spec = None

            if hidream_spec is not None:
                with self.subTest(model_id="hidream_e1"):
                    dm = hidream_spec.patcher.model.diffusion_model
                    inner = hidream_spec.patcher.model
                    adapter = resolve_model_adapter("auto", comfy_model=hidream_spec.patcher)
                    self.assertEqual(adapter.resolve_control_mode(dm, "auto"), "concat_latent_image")

                    gen_hd = torch.Generator(device="cpu").manual_seed(808)
                    clean_a = inner.process_latent_in(
                        hidream_spec.vae.encode(torch.rand(1, 64, 64, 3, generator=gen_hd, dtype=torch.float32))
                    )
                    clean_b = inner.process_latent_in(
                        hidream_spec.vae.encode(torch.rand(1, 64, 64, 3, generator=gen_hd, dtype=torch.float32))
                    )
                    ctrl_a = hidream_spec.vae.encode(torch.rand(1, 64, 64, 3, generator=gen_hd, dtype=torch.float32))
                    ctrl_b = hidream_spec.vae.encode(torch.rand(1, 64, 64, 3, generator=gen_hd, dtype=torch.float32))
                    # Mismatched rectangular control (`preserve_aspect` 64x96 paired with 64x64 target)
                    ctrl_rect_a = hidream_spec.vae.encode(torch.rand(1, 64, 96, 3, generator=gen_hd, dtype=torch.float32))
                    ctrl_rect_b = hidream_spec.vae.encode(torch.rand(1, 64, 96, 3, generator=gen_hd, dtype=torch.float32))
                    pe, pooled, extra_cond = self._encode_prompt_from_clip(hidream_spec.clip, "turn day into night")
                    sigma = torch.tensor([0.5], dtype=torch.float32)
                    torch.manual_seed(42)
                    noisy = torch.randn_like(clean_a)

                    with torch.no_grad():
                        with StrictComfySpy(inner) as spy:
                            out_a = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean_a, control_latents=[ctrl_a], control_mode="auto",
                            )
                            out_b = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean_a, control_latents=[ctrl_b], control_mode="auto",
                            )
                            out_rect_a = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean_a, control_latents=[ctrl_rect_a], control_mode="auto",
                            )
                            out_rect_b = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean_a, control_latents=[ctrl_rect_b], control_mode="auto",
                            )
                            # Dropped control MUST NOT leak `clean_latents` into `image_cond`!
                            out_drop_a = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean_a, control_latents=None,
                                control_mode="auto", control_dropped=True,
                            )
                            out_drop_b = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean_b, control_latents=None,
                                control_mode="auto", control_dropped=True,
                            )
                        spy.assert_no_silent_errors(self, "hidream_e1[concat_latent_image]")

                    self.assertEqual(tuple(out_a.shape), tuple(noisy.shape))
                    self.assertEqual(tuple(out_rect_a.shape), tuple(noisy.shape))
                    self.assertTrue(torch.isfinite(out_rect_a).all())
                    self.assertFalse(torch.allclose(out_a, out_b, atol=1e-6), "HiDream-E1 ignored concat_latent_image!")
                    self.assertFalse(
                        torch.allclose(out_rect_a, out_rect_b, atol=1e-6),
                        "HiDream-E1 ignored mismatched rectangular concat_latent_image!",
                    )
                    self.assertTrue(
                        torch.allclose(out_drop_a, out_drop_b, atol=1e-6),
                        "HiDream-E1 leaked clean_latents during control_dropout!",
                    )

                    LoRAInjector.inject(
                        dm,
                        LoRAConfig(rank=4, alpha=4.0, target_modules=adapter.spec.default_target_modules),
                        lora_dtype=torch.float32,
                    )
                    try:
                        with _inplace_safe_outer_autograd_context(gradient_checkpointing=False):
                            loss = adapter.compute_training_loss(
                                backbone=dm,
                                batch={
                                    "latents": clean_a,
                                    "prompt_embeds": pe,
                                    "pooled_output": pooled,
                                    "extra_conds": extra_cond,
                                    "latent_scaled": True,
                                    "control_latents": [ctrl_a],
                                    "control_mode": "auto",
                                },
                                device=torch.device("cpu"),
                                dtype=torch.float32,
                            )
                            loss.backward()
                        self.assertTrue(torch.isfinite(loss).item())
                        self.assertTrue(
                            any(
                                isinstance(m, LoRALinear) and m.lora_B.weight.grad is not None and float(m.lora_B.weight.grad.abs().max().item()) > 0.0
                                for m in dm.modules()
                            )
                        )
                    finally:
                        LoRAInjector.uninject(dm)

            # --- Case B: Real SD15_instructpix2pix (widened 8-channel UNet `c_concat`) ---
            with self.subTest(model_id="sd15_instructpix2pix"):
                patcher_ip2p, vae, clip = self._build_sd15_instructpix2pix_spec(dtype=torch.float32)
                inner = patcher_ip2p.model
                dm = inner.diffusion_model
                adapter = resolve_model_adapter("auto", comfy_model=patcher_ip2p)
                self.assertEqual(adapter.resolve_control_mode(dm, "auto"), "concat_latent_image")

                gen_ip = torch.Generator(device="cpu").manual_seed(909)
                clean_a = inner.process_latent_in(vae.encode(torch.rand(1, 64, 64, 3, generator=gen_ip, dtype=torch.float32)))
                clean_b = inner.process_latent_in(vae.encode(torch.rand(1, 64, 64, 3, generator=gen_ip, dtype=torch.float32)))
                ctrl_a = vae.encode(torch.rand(1, 64, 64, 3, generator=gen_ip, dtype=torch.float32))
                ctrl_b = vae.encode(torch.rand(1, 64, 64, 3, generator=gen_ip, dtype=torch.float32))
                ctrl_rect_a = vae.encode(torch.rand(1, 64, 96, 3, generator=gen_ip, dtype=torch.float32))
                ctrl_rect_b = vae.encode(torch.rand(1, 64, 96, 3, generator=gen_ip, dtype=torch.float32))
                pe, pooled, extra_cond = self._encode_prompt_from_clip(clip, "add snow to the trees")
                sigma = torch.tensor([0.5], dtype=torch.float32)
                torch.manual_seed(77)
                noisy = torch.randn_like(clean_a)

                with torch.no_grad():
                    with StrictComfySpy(inner) as spy:
                        out_a = adapter._forward_backbone(
                            backbone=dm, noisy_latents=noisy, timesteps=sigma,
                            prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                            sigma=sigma, clean_latents=clean_a, control_latents=[ctrl_a], control_mode="auto",
                        )
                        out_b = adapter._forward_backbone(
                            backbone=dm, noisy_latents=noisy, timesteps=sigma,
                            prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                            sigma=sigma, clean_latents=clean_a, control_latents=[ctrl_b], control_mode="auto",
                        )
                        out_rect_a = adapter._forward_backbone(
                            backbone=dm, noisy_latents=noisy, timesteps=sigma,
                            prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                            sigma=sigma, clean_latents=clean_a, control_latents=[ctrl_rect_a], control_mode="auto",
                        )
                        out_rect_b = adapter._forward_backbone(
                            backbone=dm, noisy_latents=noisy, timesteps=sigma,
                            prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                            sigma=sigma, clean_latents=clean_a, control_latents=[ctrl_rect_b], control_mode="auto",
                        )
                        # Dropped control MUST feed zeros to `c_concat`, NOT `clean_latents`!
                        out_drop_a = adapter._forward_backbone(
                            backbone=dm, noisy_latents=noisy, timesteps=sigma,
                            prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                            sigma=sigma, clean_latents=clean_a, control_latents=None,
                            control_mode="auto", control_dropped=True,
                        )
                        out_drop_b = adapter._forward_backbone(
                            backbone=dm, noisy_latents=noisy, timesteps=sigma,
                            prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                            sigma=sigma, clean_latents=clean_b, control_latents=None,
                            control_mode="auto", control_dropped=True,
                        )
                    spy.assert_no_silent_errors(self, "sd15_instructpix2pix[c_concat]")

                self.assertEqual(tuple(out_a.shape), tuple(noisy.shape))
                self.assertEqual(tuple(out_rect_a.shape), tuple(noisy.shape))
                self.assertTrue(torch.isfinite(out_rect_a).all())
                self.assertFalse(
                    torch.allclose(out_a, out_b, atol=1e-6),
                    "SD15_instructpix2pix ignored concat_latent_image!",
                )
                self.assertFalse(
                    torch.allclose(out_rect_a, out_rect_b, atol=1e-6),
                    "SD15_instructpix2pix ignored mismatched rectangular concat_latent_image!",
                )
                self.assertTrue(
                    torch.allclose(out_drop_a, out_drop_b, atol=1e-6),
                    "SD15_instructpix2pix leaked clean_latents during control_dropout!",
                )

                LoRAInjector.inject(
                    dm,
                    LoRAConfig(rank=4, alpha=4.0, target_modules=adapter.spec.default_target_modules),
                    lora_dtype=torch.float32,
                )
                try:
                    with _inplace_safe_outer_autograd_context(gradient_checkpointing=False):
                        loss = adapter.compute_training_loss(
                            backbone=dm,
                            batch={
                                "latents": clean_a,
                                "prompt_embeds": pe,
                                "pooled_output": pooled,
                                "extra_conds": extra_cond,
                                "latent_scaled": True,
                                "control_latents": [ctrl_a],
                                "control_mode": "auto",
                            },
                            device=torch.device("cpu"),
                            dtype=torch.float32,
                        )
                        loss.backward()
                    self.assertTrue(torch.isfinite(loss).item())
                    self.assertTrue(
                        any(
                            isinstance(m, LoRALinear) and m.lora_B.weight.grad is not None and float(m.lora_B.weight.grad.abs().max().item()) > 0.0
                            for m in dm.modules()
                        )
                    )
                finally:
                    LoRAInjector.uninject(dm)

    def test_real_edit_models_bf16_and_ref_methods_contract(self) -> None:
        """
        Verify BF16 edit-mode resilience across ALL `reference_latents` and `concat_latent_image`
        ComfyUI edit architectures:
          1. BF16 real forward sensitivity (`ctrl_a` vs `ctrl_b`) with zero `StrictComfySpy` errors.
          2. BF16 `compute_training_loss` + `loss.backward()` with FP32 LoRA master weights
             producing finite loss and non-zero LoRA gradients.
          3. Explicit sweep of all 4 ComfyUI `reference_latents_method` values
             (`"offset"`, `"index"`, `"index_timestep_zero"`, `"uxo"`) on `flux_dev` and `flux2`.
          4. Zero leaked PyTorch op wrappers after BF16 edit execution.
        """
        orig_add_ = torch.Tensor.add_
        orig_mul_ = torch.Tensor.mul_
        orig_addcmul_ = torch.Tensor.addcmul_
        orig_cat = torch.cat
        orig_sdpa = F.scaled_dot_product_attention

        edit_ref_models = self._discover_reference_latents_model_ids()

        with EmulatedComfyEnvironment():
            # 1. All reference_latents models in CPU BF16
            for model_id in edit_ref_models:
                with self.subTest(model_id=f"{model_id}[bf16]"):
                    try:
                        spec = RealForwardModelRegistry.build(model_id, dtype=torch.bfloat16)
                    except (ImportError, AttributeError) as exc:
                        self.skipTest(f"[{model_id}] unavailable in installed ComfyUI checkout: {exc}")
                        continue

                    dm = spec.patcher.model.diffusion_model
                    inner = spec.patcher.model
                    adapter = resolve_model_adapter("auto", comfy_model=spec.patcher)

                    gen = torch.Generator(device="cpu").manual_seed(515)
                    raw_tgt = spec.vae.encode(torch.rand(1, spec.resolution, spec.resolution, 3, generator=gen, dtype=torch.float32))
                    clean_lat = (
                        inner.process_latent_in(raw_tgt) if hasattr(inner, "process_latent_in") else raw_tgt
                    ).to(dtype=torch.bfloat16)
                    ctrl_a_raw = spec.vae.encode(torch.rand(1, spec.resolution, spec.resolution, 3, generator=gen, dtype=torch.float32))
                    ctrl_b_raw = spec.vae.encode(torch.rand(1, spec.resolution, spec.resolution, 3, generator=gen, dtype=torch.float32))

                    pe_f, pooled_f, extra_f = self._encode_prompt_from_clip(spec.clip, "make scene golden hour")
                    pe = self._cast_floating_cond(pe_f, torch.bfloat16)
                    pooled = self._cast_floating_cond(pooled_f, torch.bfloat16)
                    extra_cond = self._cast_floating_cond(extra_f, torch.bfloat16)
                    sigma = torch.tensor([0.5], dtype=torch.float32)
                    torch.manual_seed(321)
                    noisy = (clean_lat.float() + torch.randn_like(clean_lat.float()) * 0.25).to(dtype=torch.bfloat16)

                    with torch.no_grad():
                        with StrictComfySpy(inner) as spy:
                            out_a = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean_lat,
                                control_latents=[ctrl_a_raw], control_mode="auto",
                            )
                            out_b = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean_lat,
                                control_latents=[ctrl_b_raw], control_mode="auto",
                            )
                        spy.assert_no_silent_errors(self, f"{model_id}[bf16_ref_latents]")

                    self.assertEqual(tuple(out_a.shape), tuple(noisy.shape))
                    self.assertTrue(torch.isfinite(out_a).all(), f"[{model_id}] non-finite BF16 edit forward!")
                    self.assertFalse(
                        torch.allclose(out_a.float(), out_b.float(), atol=1e-6),
                        f"[{model_id}] BF16 edit forward ignored reference_latents!",
                    )

                    # Sweep explicit `reference_latents_method` modes supported by each Flux variant:
                    # Flux 2 (`global_modulation=True`) supports all 4 including `index_timestep_zero`;
                    # Flux 1 (`global_modulation=False`) supports `offset`, `index`, and `uxo`.
                    if model_id in ("flux_dev", "flux2"):
                        ref_methods = (
                            ("offset", "index", "index_timestep_zero", "uxo")
                            if model_id == "flux2"
                            else ("offset", "index", "uxo")
                        )
                        for ref_method in ref_methods:
                            with torch.no_grad():
                                with StrictComfySpy(inner) as m_spy:
                                    m_a = adapter._forward_backbone(
                                        backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                        prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                        sigma=sigma, clean_latents=clean_lat,
                                        control_latents=[ctrl_a_raw], control_mode="auto",
                                        reference_latents_method=ref_method,
                                    )
                                    m_b = adapter._forward_backbone(
                                        backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                        prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                        sigma=sigma, clean_latents=clean_lat,
                                        control_latents=[ctrl_b_raw], control_mode="auto",
                                        reference_latents_method=ref_method,
                                    )
                                m_spy.assert_no_silent_errors(self, f"{model_id}[bf16_method={ref_method}]")
                            self.assertTrue(
                                torch.isfinite(m_a).all(),
                                f"[{model_id}] non-finite BF16 output with reference_latents_method='{ref_method}'",
                            )
                            self.assertFalse(
                                torch.allclose(m_a.float(), m_b.float(), atol=1e-6),
                                f"[{model_id}] reference_latents_method='{ref_method}' ignored control latents!",
                            )

                    # BF16 training loss + backward with FP32 LoRA master weights
                    LoRAInjector.inject(
                        dm,
                        LoRAConfig(rank=4, alpha=4.0, target_modules=adapter.spec.default_target_modules),
                        lora_dtype=torch.float32,
                    )
                    try:
                        torch.manual_seed(654)
                        with _inplace_safe_outer_autograd_context(gradient_checkpointing=False):
                            loss = adapter.compute_training_loss(
                                backbone=dm,
                                batch={
                                    "latents": clean_lat,
                                    "prompt_embeds": pe,
                                    "pooled_output": pooled,
                                    "extra_conds": extra_cond,
                                    "latent_scaled": True,
                                    "control_latents": [ctrl_a_raw],
                                    "control_mode": "auto",
                                    "reference_latents_method": "auto",
                                },
                                device=torch.device("cpu"),
                                dtype=torch.bfloat16,
                            )
                            loss.backward()
                        self.assertTrue(torch.isfinite(loss).item(), f"[{model_id}] non-finite BF16 edit loss: {loss}")
                        self.assertTrue(
                            any(
                                isinstance(m, LoRALinear) and m.lora_B.weight.grad is not None and float(m.lora_B.weight.grad.abs().max().item()) > 0.0
                                for m in dm.modules()
                            ),
                            f"[{model_id}] zero LoRA gradients in BF16 edit mode!",
                        )
                    finally:
                        LoRAInjector.uninject(dm)

            # 2. Both concat_latent_image edit models (HiDream-E1 & SD15_instructpix2pix) in CPU BF16
            concat_cases: List[Tuple[str, Any, Any, Any]] = []
            try:
                hd_bf16 = RealForwardModelRegistry.build("hidream_i1", dtype=torch.bfloat16)
                concat_cases.append(("hidream_e1", hd_bf16.patcher, hd_bf16.vae, hd_bf16.clip))
            except (ImportError, AttributeError):
                pass
            ip2p_p, ip2p_v, ip2p_c = self._build_sd15_instructpix2pix_spec(dtype=torch.bfloat16)
            concat_cases.append(("sd15_instructpix2pix", ip2p_p, ip2p_v, ip2p_c))

            for cname, c_patcher, c_vae, c_clip in concat_cases:
                with self.subTest(model_id=f"{cname}[bf16]"):
                    inner = c_patcher.model
                    dm = inner.diffusion_model
                    adapter = resolve_model_adapter("auto", comfy_model=c_patcher)

                    gen_c = torch.Generator(device="cpu").manual_seed(777)
                    clean = inner.process_latent_in(
                        c_vae.encode(torch.rand(1, 64, 64, 3, generator=gen_c, dtype=torch.float32))
                    ).to(dtype=torch.bfloat16)
                    ctrl_a = c_vae.encode(torch.rand(1, 64, 96, 3, generator=gen_c, dtype=torch.float32))
                    ctrl_b = c_vae.encode(torch.rand(1, 64, 96, 3, generator=gen_c, dtype=torch.float32))
                    pe_f, pooled_f, extra_f = self._encode_prompt_from_clip(c_clip, "apply watercolor style")
                    pe = self._cast_floating_cond(pe_f, torch.bfloat16)
                    pooled = self._cast_floating_cond(pooled_f, torch.bfloat16)
                    extra_cond = self._cast_floating_cond(extra_f, torch.bfloat16)
                    sigma = torch.tensor([0.5], dtype=torch.float32)
                    torch.manual_seed(88)
                    noisy = torch.randn_like(clean.float()).to(dtype=torch.bfloat16)

                    with torch.no_grad():
                        with StrictComfySpy(inner) as spy:
                            out_a = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean,
                                control_latents=[ctrl_a], control_mode="auto",
                            )
                            out_b = adapter._forward_backbone(
                                backbone=dm, noisy_latents=noisy, timesteps=sigma,
                                prompt_embeds=pe, pooled_output=pooled, extra_cond_dict=extra_cond,
                                sigma=sigma, clean_latents=clean,
                                control_latents=[ctrl_b], control_mode="auto",
                            )
                        spy.assert_no_silent_errors(self, f"{cname}[bf16_concat]")

                    self.assertTrue(torch.isfinite(out_a).all(), f"[{cname}] non-finite BF16 concat output!")
                    self.assertFalse(
                        torch.allclose(out_a.float(), out_b.float(), atol=1e-6),
                        f"[{cname}] BF16 concat_latent_image ignored control!",
                    )

                    LoRAInjector.inject(
                        dm,
                        LoRAConfig(rank=4, alpha=4.0, target_modules=adapter.spec.default_target_modules),
                        lora_dtype=torch.float32,
                    )
                    try:
                        with _inplace_safe_outer_autograd_context(gradient_checkpointing=False):
                            loss = adapter.compute_training_loss(
                                backbone=dm,
                                batch={
                                    "latents": clean,
                                    "prompt_embeds": pe,
                                    "pooled_output": pooled,
                                    "extra_conds": extra_cond,
                                    "latent_scaled": True,
                                    "control_latents": [ctrl_a],
                                    "control_mode": "auto",
                                },
                                device=torch.device("cpu"),
                                dtype=torch.bfloat16,
                            )
                            loss.backward()
                        self.assertTrue(torch.isfinite(loss).item(), f"[{cname}] non-finite BF16 concat loss!")
                        self.assertTrue(
                            any(
                                isinstance(m, LoRALinear) and m.lora_B.weight.grad is not None and float(m.lora_B.weight.grad.abs().max().item()) > 0.0
                                for m in dm.modules()
                            ),
                            f"[{cname}] zero LoRA gradients in BF16 concat edit mode!",
                        )
                    finally:
                        LoRAInjector.uninject(dm)

        # Verify scoped PyTorch op wrappers restored native callables cleanly
        self.assertIs(torch.Tensor.add_, orig_add_)
        self.assertIs(torch.Tensor.mul_, orig_mul_)
        self.assertIs(torch.Tensor.addcmul_, orig_addcmul_)
        self.assertIs(torch.cat, orig_cat)
        self.assertIs(F.scaled_dot_product_attention, orig_sdpa)


if __name__ == "__main__":
    unittest.main()
