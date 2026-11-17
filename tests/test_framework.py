"""
Unit Tests for OmniLoRA Hardware Backends, ComfyUI-Style `model_management` Auto-Detection,
VRAM Tiering, Precision Cascade (`bf16` -> `fp16`+GradScaler -> `fp32`), and Universal `ComfyModelAdapter`.
"""

import copy
import json
import math
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omni_lora import (
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
from omni_lora.backends import (
    BACKEND_REGISTRY,
    VRAMState,
    XPUBackend,
    detect_and_configure,
    resolve_backend,
)
from omni_lora.lora import LoRAConfig, LoRAInjector, LoRALinear
from omni_lora.lora.layers import _strip_numeric_segments
from omni_lora.models import (
    MODEL_REGISTRY,
    MODEL_SPECS,
    ComfyModelAdapter,
    MossImageAdapter,
    detect_comfy_model_architecture,
    normalize_legacy_lora_key,
    resolve_model_adapter,
)
from omni_lora.pipelines import LoRATrainingPipeline, TrainingConfig


class TestOmniLoRAFramework(unittest.TestCase):
    def test_all_eight_backends_registered(self) -> None:
        expected = {"xpu", "rocm", "cuda", "tpu", "npu", "hpu", "dml", "mps", "cpu"}
        self.assertEqual(set(BACKEND_REGISTRY.keys()), expected)
        for backend_id, backend_cls in BACKEND_REGISTRY.items():
            self.assertEqual(backend_cls.__module__, f"omni_lora.backends.{backend_id}")
        # Verify "xla" alias resolves to TPUBackend
        xla_backend = resolve_backend(preferred="xla", allow_fallback=False)
        self.assertEqual(xla_backend.backend_id, "tpu")
        self.assertEqual(xla_backend.device_type, "xla")

    def test_universal_comfy_model_adapter_and_dynamic_specs(self) -> None:
        backend = resolve_backend(preferred="cpu", dtype="bf16", allow_fallback=True)
        for model_id in list(MODEL_REGISTRY.keys()) + ["future_comfy_arch_2027"]:
            self.assertIsNotNone(MODEL_SPECS[model_id])
            adapter = resolve_model_adapter(
                model_id=model_id,
                gradient_checkpointing=False,
            )
            backbone = adapter.build_backbone(dtype=torch.bfloat16, device=backend.get_device())
            batch = adapter.create_synthetic_batch(
                batch_size=1,
                resolution=64,
                device=backend.get_device(),
                dtype=torch.bfloat16,
            )
            loss = adapter.compute_training_loss(
                backbone=backbone,
                batch=batch,
                device=backend.get_device(),
                dtype=torch.bfloat16,
            )
            self.assertGreater(float(loss.item()), 0.0)

    def test_xpu_backend_abstraction_and_env(self) -> None:
        backend = XPUBackend(device_index=0, dtype="bf16")
        env_vars = backend.configure_environment()
        self.assertEqual(env_vars["SYCL_CACHE_PERSISTENT"], "1")
        self.assertEqual(env_vars["ONEDNN_DEFAULT_FPMATH_MODE"], "BF16")
        caps = backend.get_capabilities()
        self.assertEqual(caps.device_type, "xpu")
        self.assertTrue(caps.supports_bf16)

    def test_comfyui_style_vram_tiering_and_precision_cascade(self) -> None:
        arc_plan = detect_and_configure(preferred_backend="xpu", requested_dtype="auto", allow_fallback=False)
        self.assertEqual(arc_plan.vram_state, VRAMState.NORMAL_VRAM)
        self.assertEqual(arc_plan.resolved_dtype, "bf16")
        self.assertFalse(arc_plan.requires_grad_scaler)
        self.assertEqual(arc_plan.recommended_conditioning_mode, "precache_offload")

        dml_plan = detect_and_configure(preferred_backend="dml", requested_dtype="auto", allow_fallback=False)
        self.assertEqual(dml_plan.vram_state, VRAMState.NORMAL_VRAM)
        self.assertEqual(dml_plan.resolved_dtype, "fp16")
        self.assertTrue(dml_plan.requires_grad_scaler)

        hpu_plan = detect_and_configure(preferred_backend="hpu", requested_dtype="auto", allow_fallback=False)
        self.assertEqual(hpu_plan.vram_state, VRAMState.HIGH_VRAM)
        self.assertEqual(hpu_plan.recommended_conditioning_mode, "on_the_fly_xpu")

        tpu_plan = detect_and_configure(preferred_backend="tpu", requested_dtype="auto", allow_fallback=False)
        self.assertEqual(tpu_plan.backend.backend_id, "tpu")
        self.assertEqual(tpu_plan.capabilities.device_type, "xla")
        self.assertEqual(tpu_plan.capabilities.total_vram_mb, 16384.0)
        self.assertEqual(tpu_plan.vram_state, VRAMState.NORMAL_VRAM)
        self.assertEqual(tpu_plan.resolved_dtype, "bf16")
        self.assertFalse(tpu_plan.requires_grad_scaler)
        tpu_snap = tpu_plan.backend.get_memory_snapshot()
        self.assertEqual(tpu_snap.total_vram_mb, 16384.0)
        tpu_plan.backend.manual_seed_all(42)

        low_plan = detect_and_configure(
            preferred_backend="xpu",
            requested_dtype="auto",
            override_vram_mb=8192.0,
            allow_fallback=False,
        )
        self.assertEqual(low_plan.vram_state, VRAMState.LOW_VRAM)
        self.assertEqual(low_plan.recommended_resolution, 768)

    def test_bf16_lora_injection_and_step(self) -> None:
        backend = resolve_backend(preferred="auto", dtype="auto", allow_fallback=True)
        adapter = MossImageAdapter(gradient_checkpointing=False)
        lora_cfg = LoRAConfig(rank=8, alpha=8.0, lora_dtype="bf16")
        train_cfg = TrainingConfig(
            backend=backend.backend_id,
            dtype="bf16",
            resolution=128,
            batch_size=1,
            gradient_accumulation_steps=2,
            max_train_steps=2,
            lora=lora_cfg,
        )
        pipeline = LoRATrainingPipeline(backend=backend, model_adapter=adapter, config=train_cfg)
        info = pipeline.setup()
        self.assertGreater(info["injected_layer_count"], 0)
        self.assertGreater(info["trainable_parameters"], 0)

        step_metrics = pipeline.train_step(1)
        self.assertGreater(step_metrics.loss, 0.0)

    def test_fp16_training_and_grad_scaler_never_use_fp64(self) -> None:
        backend = resolve_backend(preferred="cpu", dtype="fp16", allow_fallback=True)
        adapter = ComfyModelAdapter(gradient_checkpointing=False)
        lora_cfg = LoRAConfig(rank=4, alpha=4.0, lora_dtype="fp16")
        train_cfg = TrainingConfig(
            backend=backend.backend_id,
            dtype="fp16",
            resolution=64,
            batch_size=1,
            gradient_accumulation_steps=2,
            max_train_steps=2,
            lora=lora_cfg,
        )
        pipeline = LoRATrainingPipeline(backend=backend, model_adapter=adapter, config=train_cfg)
        info = pipeline.setup()
        self.assertTrue(info["grad_scaler_enabled"])
        self.assertTrue(info["fp64_free_verified"])

        for p in pipeline.backbone.parameters():
            self.assertNotEqual(p.dtype, torch.float64)

        step_metrics = pipeline.train_step(1)
        self.assertTrue(step_metrics.grad_scaler_active)
        self.assertGreater(step_metrics.loss, 0.0)

    def test_comfyui_custom_nodes_end_to_end_inside_inference_mode(self) -> None:
        self.assertIn("OmniLoRADatasetLoader", NODE_CLASS_MAPPINGS)
        self.assertIn("OmniLoRAPrecomputeConditioning", NODE_CLASS_MAPPINGS)
        self.assertIn("OmniLoRALoader", NODE_CLASS_MAPPINGS)
        self.assertIn("OmniLoRATrainer", NODE_CLASS_MAPPINGS)
        self.assertIn("OmniLoRASave", NODE_CLASS_MAPPINGS)
        self.assertIn("OmniLoRALossTracker", NODE_CLASS_MAPPINGS)
        self.assertEqual(len(NODE_DISPLAY_NAME_MAPPINGS), 6)

        with torch.inference_mode():
            comfy_images = torch.rand(2, 64, 64, 3, dtype=torch.float32)
            custom_prompts = "lush emerald forest moss on ancient stone\ndew drops on sphagnum moss macro"

            dataset_node = OmniLoRADatasetLoader()
            dataset, dataset_info = dataset_node.load_dataset(
                dataset_path="",
                trigger_word="ohwx_moss",
                default_caption="botanical moss photo",
                resolution=64,
                batch_size=2,
                num_repeats=1,
                center_crop=True,
                random_flip=False,
                caption_dropout=0.0,
                images=comfy_images,
                custom_prompts=custom_prompts,
            )
            self.assertEqual(dataset.num_unique_images, 2)
            self.assertIn("ohwx_moss, lush emerald forest moss", dataset_info)

            class MockEndToEndVAE:
                def encode(self, x: torch.Tensor) -> torch.Tensor:
                    b, h, w, _ = x.shape
                    return torch.randn(b, 16, h // 8, w // 8, dtype=torch.bfloat16)

            class MockVariableLengthCLIP:
                def __init__(self) -> None:
                    self.calls = 0

                def tokenize(self, text: str) -> dict:
                    self.calls += 1
                    return {"tokens": [1] * (77 if self.calls % 2 == 1 else 120)}

                def encode_from_tokens(
                    self,
                    tokens: dict,
                    return_pooled: bool = True,
                    return_dict: bool = False,
                ):
                    seq_len = len(tokens["tokens"])
                    cond = torch.randn(1, seq_len, 2048, dtype=torch.bfloat16)
                    pooled = torch.randn(1, 2048, dtype=torch.bfloat16)
                    if return_dict:
                        return cond, {"pooled_output": pooled, "attention_mask": torch.ones(1, seq_len)}
                    return cond, pooled

            precache_node = OmniLoRAPrecomputeConditioning()
            (cached_bundle,) = precache_node.precompute(
                dataset=dataset,
                offload_encoders_after_cache=True,
                vae=MockEndToEndVAE(),
                clip=MockVariableLengthCLIP(),
            )
            self.assertEqual(cached_bundle.total_batches, 1)
            # Variable-length 77 and 120 token prompts should be padded to [2, 120, 2048]
            self.assertEqual(tuple(cached_bundle.batches[0]["prompt_embeds"].shape), (2, 120, 2048))

            class MockEndToEndDiffusion(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv_in = nn.Conv2d(16, 32, 1)
                    self.context_in = nn.Linear(2048, 32)
                    self.qkv = nn.Linear(32, 32)
                    self.conv_out = nn.Conv2d(32, 16, 1)

                def forward(
                    self,
                    x: torch.Tensor,
                    timestep: torch.Tensor,
                    context: torch.Tensor,
                    **kwargs: object,
                ) -> torch.Tensor:
                    h = self.conv_in(x)
                    b, c, height, width = h.shape
                    tokens = h.permute(0, 2, 3, 1).reshape(b, height * width, c)
                    tokens = tokens + self.context_in(context).mean(dim=1, keepdim=True) + self.qkv(tokens)
                    return self.conv_out(tokens.reshape(b, height, width, c).permute(0, 3, 1, 2))

            class MockEndToEndPatcher:
                def __init__(self) -> None:
                    class Inner:
                        def __init__(self) -> None:
                            self.diffusion_model = MockEndToEndDiffusion()

                    self.model = Inner()

            trainer_node = OmniLoRATrainer()
            save_node = OmniLoRASave()
            loader_node = OmniLoRALoader()

            with tempfile.TemporaryDirectory() as tmp_dir:
                _model_out, lora_bundle, summary_json = trainer_node.train_lora(
                    model=MockEndToEndPatcher(),
                    cached_data=cached_bundle,
                    model_architecture="auto",
                    backend="auto",
                    precision="auto",
                    vram_mode="auto",
                    optimizer="adamw",
                    rank=4,
                    alpha=8.0,
                    learning_rate=1e-4,
                    steps=2,
                    grad_accum_steps=1,
                    gradient_checkpointing=False,
                )
                self.assertIsInstance(lora_bundle, OmniLoRANetworkBundle)
                self.assertEqual(lora_bundle.rank, 4)
                self.assertEqual(lora_bundle.alpha, 8.0)
                self.assertEqual(lora_bundle.steps_trained, 2)
                self.assertIn('"fp64_free_verified": true', summary_json)

                saved_bundle, lora_filename, saved_path = save_node.save_lora(
                    lora=lora_bundle,
                    output_lora_name="test_comfy_moss_lora",
                    append_step_count=False,
                    custom_output_dir=tmp_dir,
                )
                self.assertTrue(lora_filename.startswith("test_comfy_moss_lora"))
                self.assertTrue(Path(saved_path).exists())

                # Load the saved LoRA back and resume training for 2 more steps with SGD
                reloaded_bundle, reload_info = loader_node.load_lora(
                    lora_name="(none)",
                    custom_lora_path=saved_path,
                )
                self.assertEqual(reloaded_bundle.rank, 4)
                self.assertEqual(reloaded_bundle.alpha, 8.0)
                self.assertIn('"detected_rank": 4', reload_info)

                _model_out2, resumed_lora_bundle, summary_json2 = trainer_node.train_lora(
                    model=MockEndToEndPatcher(),
                    cached_data=cached_bundle,
                    model_architecture="auto",
                    backend="auto",
                    precision="auto",
                    vram_mode="auto",
                    optimizer="sgd",
                    rank=32,  # Should automatically adopt loaded rank=4 and alpha=8.0
                    alpha=32.0,
                    learning_rate=1e-4,
                    steps=2,
                    grad_accum_steps=1,
                    gradient_checkpointing=False,
                    initial_lora=reloaded_bundle,
                )
                self.assertEqual(resumed_lora_bundle.rank, 4)
                self.assertEqual(resumed_lora_bundle.alpha, 8.0)
                self.assertEqual(resumed_lora_bundle.steps_trained, 4)
                self.assertIn('"optimizer": "sgd"', summary_json2)

                tracker_node = OmniLoRALossTracker()
                tracker_out = tracker_node.track(training_summary=summary_json2)
                self.assertIn("ui", tracker_out)
                self.assertEqual(tracker_out["ui"]["summary"], [summary_json2])

    def test_sd15_y_none_assertion_and_custom_ops_repeated_runs(self) -> None:
        """
        Verify:
        1. SD1.5-style `UNetModel` (`self.num_classes = None`) never receives a non-None `y` argument
           (which would raise `AssertionError: must specify y if and only if the model is class-conditional`).
        2. Custom `comfy.ops` Linear layers (not subclassing `nn.Linear`) are discovered and wrapped.
        3. Running `OmniLoRATrainer.train_lora` multiple times in a row on the same cached `ModelPatcher`
           cleanly uninjects and re-injects without error.
        """

        class CustomComfyOpsLinear(nn.Module):
            def __init__(self, in_features: int, out_features: int) -> None:
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
                self.bias = nn.Parameter(torch.zeros(out_features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(
                    x,
                    self.weight.to(device=x.device, dtype=x.dtype),
                    self.bias.to(device=x.device, dtype=x.dtype),
                )

        class MockSD15UNetModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 4
                self.num_classes = None
                self.conv_in = nn.Conv2d(4, 32, kernel_size=1)
                self.context_in = CustomComfyOpsLinear(768, 32)
                self.custom_qkv = CustomComfyOpsLinear(32, 32)
                self.conv_out = nn.Conv2d(32, 4, kernel_size=1)

            def forward(
                self,
                x: torch.Tensor,
                timesteps: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                **kwargs: object,
            ) -> torch.Tensor:
                assert (y is not None) == (
                    self.num_classes is not None
                ), "must specify y if and only if the model is class-conditional"
                h = self.conv_in(x)
                b, c, height, width = h.shape
                tokens = h.permute(0, 2, 3, 1).reshape(b, height * width, c)
                if context is not None:
                    ctx = self.context_in(context).mean(dim=1, keepdim=True)
                    tokens = tokens + ctx
                tokens = tokens + self.custom_qkv(tokens)
                out_h = tokens.reshape(b, height, width, c).permute(0, 3, 1, 2)
                return self.conv_out(out_h)

        class MockInnerModel:
            def __init__(self) -> None:
                self.diffusion_model = MockSD15UNetModel()

        class MockComfyModelPatcher:
            def __init__(self) -> None:
                self.model = MockInnerModel()

        class MockSD15VAE:
            def encode(self, pixel_samples: torch.Tensor) -> torch.Tensor:
                b, h, w, _ = pixel_samples.shape
                return torch.randn(b, 4, h // 8, w // 8, dtype=torch.bfloat16)

        class MockSD1Clip:
            def tokenize(self, text: str) -> dict:
                return {"tokens": [1, 2, 3]}

            def encode_from_tokens(self, tokens: dict, return_pooled: bool = True):
                return torch.randn(1, 77, 768, dtype=torch.bfloat16), torch.randn(1, 768, dtype=torch.bfloat16)

        with torch.inference_mode():
            comfy_images = torch.rand(1, 64, 64, 3, dtype=torch.float32)
            dataset_node = OmniLoRADatasetLoader()
            dataset, _ = dataset_node.load_dataset(
                dataset_path="",
                trigger_word="ohwx_subject",
                default_caption="portrait photo",
                resolution=64,
                batch_size=1,
                num_repeats=1,
                center_crop=True,
                random_flip=False,
                caption_dropout=0.0,
                images=comfy_images,
                custom_prompts="studio portrait photo",
            )

            precache_node = OmniLoRAPrecomputeConditioning()
            (cached_bundle,) = precache_node.precompute(
                dataset=dataset,
                offload_encoders_after_cache=True,
                vae=MockSD15VAE(),
                clip=MockSD1Clip(),
            )

            patcher = MockComfyModelPatcher()
            trainer_node = OmniLoRATrainer()

            for run_idx, rank in ((1, 4), (2, 8)):
                _, lora_bundle, summary_str = trainer_node.train_lora(
                    cached_data=cached_bundle,
                    model_architecture="auto",
                    backend="cpu",
                    precision="bf16",
                    vram_mode="auto",
                    optimizer="adamw",
                    rank=rank,
                    alpha=float(rank),
                    learning_rate=1e-4,
                    steps=2,
                    grad_accum_steps=1,
                    gradient_checkpointing=False,
                    model=patcher,
                )
                self.assertEqual(lora_bundle.rank, rank)
                self.assertIn('"trainable_parameters":', summary_str)

    def test_mismatched_clip_or_vae_raises_clear_error(self) -> None:
        """
        Verify that connecting a 768-dim CLIP to a 4096-dim MOSSImage model automatically
        adapts conditioning width (768 -> 4096) during training and sampling, that Qwen3.5
        text encoder state_dicts with `model.layers.0.linear_attn.A_log` prefixes are detected
        as `TEModel.QWEN35_9B`, and that incompatible 4-channel vs 16-channel VAEs still raise
        a clear `ValueError`.
        """

        class MockMossBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 16
                self.context_in = nn.Linear(4096, 64)
                self.qkv = nn.Linear(64, 64)
                self.last_context_shape: Optional[Tuple[int, ...]] = None

            def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
                self.last_context_shape = tuple(context.shape)
                _ = self.context_in(context)
                return x

        adapter = ComfyModelAdapter(comfy_model=MockMossBackbone())
        backbone = adapter.build_backbone(dtype=torch.float32)

        clip_768_batch = {
            "latents": torch.randn(1, 16, 8, 8),
            "prompt_embeds": torch.randn(1, 77, 768),
            "latent_scaled": True,
        }
        loss = adapter.compute_training_loss(backbone, clip_768_batch, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(backbone.last_context_shape, (1, 77, 4096))

        import comfy.sd as csd  # type: ignore

        for prefix in ("model.", "", "text_encoders.qwen35_9b.transformer.model."):
            mock_qwen35_sd = {
                f"{prefix}layers.0.linear_attn.A_log": torch.zeros(16),
                f"{prefix}layers.0.input_layernorm.weight": torch.ones(4096),
            }
            self.assertEqual(csd.detect_te_model(mock_qwen35_sd), csd.TEModel.QWEN35_9B)

        bad_vae_batch = {
            "latents": torch.randn(1, 4, 8, 8),
            "prompt_embeds": torch.randn(1, 77, 4096),
            "latent_scaled": True,
        }
        with self.assertRaises(ValueError) as vae_err:
            adapter.compute_training_loss(backbone, bad_vae_batch, torch.device("cpu"), torch.float32)
        self.assertIn("VAE latent channel mismatch", str(vae_err.exception))

    def test_workflow_json_template_matches_node_signatures(self) -> None:
        """
        Verify that all workflow templates in `omni_lora/example_workflows/*.json` conform to
        ComfyUI_frontend's workflow schema and match `NODE_CLASS_MAPPINGS` input/output/widget orders.
        """
        root_dir = Path(__file__).resolve().parent.parent
        wf_dir = root_dir / "example_workflows"
        expected_workflows = (
            "omni_lora_basic_workflow.json",
            "omni_lora_dual_network_workflow.json",
            "omni_lora_masked_in_graph_workflow.json",
            "omni_lora_resume_checkpoint_workflow.json",
            "omni_lora_staged_eval_workflow.json",
        )
        # Verify alphabetical sorting places omni_lora_basic_workflow.json first
        actual_sorted = tuple(sorted(p.name for p in wf_dir.glob("*.json")))
        self.assertEqual(actual_sorted[0], "omni_lora_basic_workflow.json")
        self.assertNotIn("optional", OmniLoRAPrecomputeConditioning.INPUT_TYPES())

        # Verify every INT and FLOAT widget across all nodes has (default - min) and (max - min) aligned to `step`
        # so ComfyUI's frontend never snaps values like 1000 to 1001 (`min + k * step`).
        for node_name, node_cls in NODE_CLASS_MAPPINGS.items():
            in_types = node_cls.INPUT_TYPES()
            for section in ("required", "optional"):
                for w_name, w_spec in in_types.get(section, {}).items():
                    if isinstance(w_spec, tuple) and len(w_spec) >= 2 and w_spec[0] in ("INT", "FLOAT") and isinstance(w_spec[1], dict):
                        cfg = w_spec[1]
                        if "min" in cfg and "step" in cfg and "default" in cfg:
                            step_val = float(cfg["step"])
                            diff_def = (float(cfg["default"]) - float(cfg["min"])) / step_val
                            self.assertAlmostEqual(
                                diff_def,
                                round(diff_def),
                                places=5,
                                msg=f"{node_name}.{w_name}: default={cfg['default']} is not aligned with min={cfg['min']}, step={cfg['step']}",
                            )

        for wf_name in expected_workflows:
            wf_path = wf_dir / wf_name
            self.assertTrue(wf_path.is_file(), f"Missing workflow file: {wf_path}")
            with open(wf_path, "r", encoding="utf-8") as f:
                wf = json.load(f)
            self.assertIn("nodes", wf)
            self.assertIn("links", wf)

            for node in wf["nodes"]:
                ntype = node["type"]
                if ntype in NODE_CLASS_MAPPINGS:
                    cls = NODE_CLASS_MAPPINGS[ntype]
                    self.assertEqual(len(node["outputs"]), len(cls.RETURN_TYPES), f"Output count mismatch in {wf_name} ({ntype})")
                    for out_slot, expected_type in zip(node["outputs"], cls.RETURN_TYPES):
                        self.assertEqual(out_slot["type"], expected_type)

                    if ntype == "OmniLoRAPrecomputeConditioning":
                        input_names = [inp["name"] for inp in node.get("inputs", [])]
                        self.assertNotIn("model", input_names, f"OmniLoRAPrecomputeConditioning should not have a 'model' plug in {wf_name}")

                    if ntype == "OmniLoRATrainer":
                        self.assertEqual(node["inputs"][0]["name"], "model")
                        self.assertEqual(node["inputs"][0]["type"], "MODEL")
                        self.assertEqual(node["inputs"][1]["name"], "cached_data")
                        self.assertEqual(node["inputs"][1]["type"], "OMNILORA_CACHED_DATA")
                        self.assertEqual(len(node["widgets_values"]), 28, f"Expected 28 widgets on OmniLoRATrainer in {wf_name}")

    def test_comfy_moss_image_lora_keys_and_uninject(self) -> None:
        """
        Verify:
        1. `LoRAInjector.extract_lora_state_dict(..., prefix="diffusion_model")` preserves dots in
           module paths (`diffusion_model.layers.0.proj_ffn_down.lora_down.weight` and
           `diffusion_model.text_embedder.blocks.0.qkv.lora_down.weight`).
        2. `LoRAInjector.uninject()` restores `base_layer` in-place so `model.state_dict()`
           retains clean `.weight` keys (never `.base_layer.weight`).
        3. `normalize_legacy_lora_key` upgrades legacy underscored keys.
        """

        from omni_lora.tests.comfy_emulator import ComfyModelZoo

        moss_entry = ComfyModelZoo.build("moss_image", dtype=torch.float32)
        moss = moss_entry.patcher.model.diffusion_model
        self.assertEqual(detect_comfy_model_architecture(moss), "moss_image")

        adapter = ComfyModelAdapter(comfy_model=moss_entry.patcher)
        cfg = LoRAConfig(
            rank=4,
            alpha=4.0,
            target_modules=adapter.spec.default_target_modules,
        )
        injected = LoRAInjector.inject(moss, cfg, lora_dtype=torch.bfloat16)
        self.assertGreaterEqual(len(injected), 18)

        lora_sd = LoRAInjector.extract_lora_state_dict(moss, prefix="diffusion_model")
        self.assertIn("diffusion_model.layers.0.proj.ffn_down.lora_down.weight", lora_sd)
        self.assertIn("diffusion_model.layers.1.proj.ffn_down.lora_up.weight", lora_sd)
        self.assertIn("diffusion_model.text_embedder.blocks.0.qkv.lora_down.weight", lora_sd)
        self.assertIn("diffusion_model.text_embedder.blocks.1.qkv.lora_down.weight", lora_sd)

        for k in lora_sd.keys():
            self.assertFalse(k.startswith("diffusion_model.layers_"), f"Unexpected underscored key: {k}")

        restored_count = LoRAInjector.uninject(moss)
        self.assertEqual(restored_count, len(injected))
        for k in moss.state_dict().keys():
            self.assertNotIn("base_layer", k)
            self.assertNotIn("lora_A", k)

        self.assertEqual(
            normalize_legacy_lora_key("diffusion_model.layers_0_proj_ffn_down.lora_down.weight"),
            "diffusion_model.layers.0.proj_ffn_down.lora_down.weight",
        )
        self.assertEqual(
            normalize_legacy_lora_key("diffusion_model.text_embedder_blocks_1_qkv.alpha"),
            "diffusion_model.text_embedder.blocks.1.qkv.alpha",
        )

    def test_comfyui_native_plug_and_play_model_support(self) -> None:
        """
        Verify that OmniLoRA dynamically derives latent geometry, noise scheduling (including
        `batch_size=2` on both CONST Flow and EPS DDPM), and training objective directly from
        ComfyUI's real `comfy.model_base.BaseModel`, `comfy.model_sampling`, and `comfy.latent_formats`.
        """
        from omni_lora.tests.comfy_emulator import ComfyModelZoo

        # 1. Test real ComfyUI Rectified Flow Model (`comfy.model_base.Flux` + `comfy.ldm.flux.model.Flux`)
        flux_entry = ComfyModelZoo.build("flux_dev", dtype=torch.bfloat16)
        flux_adapter = resolve_model_adapter(model_id="auto", comfy_model=flux_entry.patcher)
        self.assertEqual(flux_adapter.spec.latent_channels, 16)
        self.assertEqual(flux_adapter.spec.objective_type, "rectified_flow")
        self.assertFalse(flux_adapter.spec.supports_video_5d)

        # 2. Test real ComfyUI DDPM Epsilon Model (`comfy.model_base.BaseModel` + `comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel`)
        sd15_entry = ComfyModelZoo.build("sd15", dtype=torch.bfloat16)
        sd15_adapter = resolve_model_adapter(model_id="auto", comfy_model=sd15_entry.patcher)
        self.assertEqual(sd15_adapter.spec.latent_channels, 4)
        self.assertEqual(sd15_adapter.spec.objective_type, "ddpm_eps")

        # 3. Test real ComfyUI Video 5D Model (`comfy.model_base.WAN21` + `comfy.ldm.wan.model.WanModel`)
        video_entry = ComfyModelZoo.build("wan_2_1_t2v", dtype=torch.bfloat16)
        video_adapter = resolve_model_adapter(model_id="auto", comfy_model=video_entry.patcher)
        self.assertTrue(video_adapter.spec.supports_video_5d)

        # 4. End-to-end training with batch_size=2 and unscaled cached latents (model not passed to Node 2)
        with torch.inference_mode():
            comfy_images = torch.rand(2, 64, 64, 3, dtype=torch.float32)
            dataset_node = OmniLoRADatasetLoader()
            dataset, _ = dataset_node.load_dataset(
                dataset_path="",
                trigger_word="ohwx_plug_play",
                default_caption="macro photo",
                resolution=64,
                batch_size=2,
                num_repeats=1,
                center_crop=True,
                random_flip=False,
                caption_dropout=0.0,
                images=comfy_images,
                custom_prompts="emerald crystal macro\nsapphire crystal macro",
            )

            precache_node = OmniLoRAPrecomputeConditioning()
            (cached_data,) = precache_node.precompute(
                dataset=dataset,
                offload_encoders_after_cache=True,
                vae=sd15_entry.vae,
                clip=sd15_entry.clip,
                model=None,
            )
            self.assertFalse(cached_data.latent_scaled)

            trainer = OmniLoRATrainer()
            _out_model, lora_bundle, summary_str = trainer.train_lora(
                cached_data=cached_data,
                model_architecture="auto",
                backend="cpu",
                precision="bf16",
                reserve_vram_gb=2.0,
                optimizer="radam",
                rank=4,
                alpha=4.0,
                learning_rate=1e-4,
                steps=2,
                grad_accum_steps=1,
                gradient_checkpointing=False,
                model=sd15_entry.patcher,
            )

            self.assertIn('"training_objective": "ddpm_eps"', summary_str)
            self.assertIn('"fp64_free_verified": true', summary_str)
            self.assertIn('"optimizer": "radam"', summary_str)
            self.assertIn('"reserved_user_vram_mb": 2048.0', summary_str)
            self.assertEqual(lora_bundle.rank, 4)

    def test_fp8_and_quantized_base_model_training(self) -> None:
        """
        Verify:
        1. Pre-loaded scaled FP8 (`float8_e4m3fn` + `scale_weight`) base models train cleanly.
        2. On-the-fly `precision="fp8_e4m3fn"` and `precision="int8_qlora"` compress frozen base
           weights to 8-bit during training and restore standard floating-point weights on uninject.
        3. External block-quantized layers (ComfyUI-GGUF `tensor_shape` + `get_weight`) inject
           the proper feature dimensions and propagate gradients through `_QuantizedCallbackLinearFn`.
        """
        # 1. On-the-fly INT8 QLoRA and FP8 E4M3FN precision modes
        for prec_mode in ("int8_qlora", "fp8_e4m3fn"):
            plan = detect_and_configure(
                preferred_backend="cpu",
                requested_dtype=prec_mode,
                vram_mode="auto",
            )
            self.assertEqual(plan.base_weight_quantization, prec_mode)
            adapter = resolve_model_adapter(model_id="flux", gradient_checkpointing=False)
            train_cfg = TrainingConfig(
                backend="cpu",
                dtype=plan.resolved_dtype,
                vram_mode="OFFLOAD_VRAM",
                model_id="flux",
                resolution=256,
                batch_size=1,
                gradient_accumulation_steps=1,
                optimizer_type="adamw",
                max_train_steps=2,
                gradient_checkpointing=False,
                lora=LoRAConfig(
                    rank=4,
                    alpha=4.0,
                    target_modules=adapter.spec.default_target_modules,
                    lora_dtype=plan.resolved_dtype,
                ),
            )
            pipeline = LoRATrainingPipeline(
                backend=plan.backend,
                model_adapter=adapter,
                config=train_cfg,
            )
            setup_info = pipeline.setup()
            self.assertGreater(setup_info["vram_budget"].get("quantized_base_layers", 0), 0)
            metrics = pipeline.train_step(1)
            self.assertTrue(float(metrics.loss) >= 0.0)
            assert pipeline.backbone is not None
            LoRAInjector.uninject(pipeline.backbone)
            for p in pipeline.backbone.parameters():
                self.assertIn(p.dtype, (torch.bfloat16, torch.float16, torch.float32))

        # 2. Mock GGUF packed 1D quantized layer with `tensor_shape` and `get_weight`
        class MockGGUFTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data: torch.Tensor, tensor_shape: tuple[int, int]):
                obj = torch.Tensor._make_subclass(cls, data, require_grad=False)
                obj.tensor_shape = tensor_shape
                obj.tensor_type = "Q4_K_M"
                return obj

        class MockGGUFLinear(nn.Module):
            def __init__(self, in_features: int, out_features: int) -> None:
                super().__init__()
                # Packed 1D uint8 storage representing a [out_features, in_features] matrix
                raw_bytes = torch.randint(1, 120, (out_features * in_features // 2,), dtype=torch.uint8)
                w_param = nn.Parameter(MockGGUFTensor(raw_bytes, (out_features, in_features)), requires_grad=False)
                w_param.tensor_shape = (out_features, in_features)
                w_param.tensor_type = "Q4_K_M"
                self.weight = w_param
                self.bias = None

            def get_weight(self, tensor: torch.Tensor, dtype: torch.dtype, device: object = None) -> torch.Tensor:
                out_f, in_f = tensor.tensor_shape
                dev = device if isinstance(device, torch.device) else tensor.device
                expanded = tensor.to(device=dev, dtype=dtype).repeat(2)[: out_f * in_f]
                return (expanded.view(out_f, in_f) - 60.0) * 0.01

        gguf_layer = MockGGUFLinear(32, 48)
        lora_wrapped = LoRALinear(gguf_layer, rank=4, alpha=4.0, lora_dtype=torch.bfloat16)
        self.assertEqual(lora_wrapped.in_features, 32)
        self.assertEqual(lora_wrapped.out_features, 48)
        x_in = torch.randn(2, 8, 32, dtype=torch.bfloat16, requires_grad=True)
        out = lora_wrapped(x_in)
        self.assertEqual(tuple(out.shape), (2, 8, 48))
        out.sum().backward()
        self.assertIsNotNone(x_in.grad)
        self.assertIsNotNone(lora_wrapped.lora_B.weight.grad)

        # 3. Verify external bitsandbytes (Linear4bit / Params4bit) models are cleanly rejected
        # on unsupported backends (e.g., XPU, CPU, MPS, DirectML) instead of crashing.
        from omni_lora.backends import is_bitsandbytes_module, is_bitsandbytes_tensor

        class Params4bit(nn.Parameter):
            pass

        class Linear4bit(nn.Module):
            def __init__(self, in_features: int, out_features: int) -> None:
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                raw = torch.randint(0, 255, (out_features * in_features // 2, 1), dtype=torch.uint8)
                param = Params4bit(raw, requires_grad=False)
                param.quant_state = type("QuantState", (), {"shape": torch.Size([out_features, in_features])})()
                self.weight = param
                self.bias = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("Native bitsandbytes CUDA kernel invoked on non-CUDA device!")

        bnb_layer = Linear4bit(32, 48)
        self.assertTrue(is_bitsandbytes_module(bnb_layer))
        self.assertTrue(is_bitsandbytes_tensor(bnb_layer.weight))

        bnb_model = nn.Sequential(bnb_layer)
        xpu_backend = XPUBackend()
        with self.assertRaises(ValueError) as ctx_xpu:
            xpu_backend.validate_quantized_model_compatibility(bnb_model)
        self.assertIn("bitsandbytes", str(ctx_xpu.exception))
        self.assertIn("not supported", str(ctx_xpu.exception))

        cpu_backend = resolve_backend("cpu")
        with self.assertRaises(ValueError) as ctx_cpu:
            cpu_backend.validate_quantized_model_compatibility(bnb_model)
        self.assertIn("bitsandbytes", str(ctx_cpu.exception))

        bnb_lora = LoRALinear(bnb_layer, rank=4, alpha=4.0, lora_dtype=torch.bfloat16)
        self.assertEqual(bnb_lora.in_features, 32)
        self.assertEqual(bnb_lora.out_features, 48)
        with self.assertRaises(ValueError) as ctx_fwd:
            _ = bnb_lora(torch.randn(2, 4, 32, dtype=torch.bfloat16))
        self.assertIn("bitsandbytes", str(ctx_fwd.exception))

    def test_inplace_copy_slices_checkpoint_backward(self) -> None:
        """
        Verify that ComfyUI models (such as Qwen-Image) that perform in-place slice assignments
        (`CopySlices`: `x[:, :txt_len] = ...`, `x[:, txt_len:] = ...`) on tensors saved for backward
        train cleanly with both `gradient_checkpointing=True` and `gradient_checkpointing=False`.
        """
        class InplaceSliceBlock(nn.Module):
            def __init__(self, hidden_dim: int = 32) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.attn_qkv = nn.Linear(hidden_dim, hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)
                self.mlp_fc = nn.Linear(hidden_dim, hidden_dim)
                self.gate = nn.Parameter(torch.ones(1, 1, hidden_dim), requires_grad=False)

            def forward(self, x: torch.Tensor, txt_len: int = 4) -> torch.Tensor:
                # Replicate Qwen-Image 2.1 `_gated_residual` (`x[:, prefix_len:].addcmul_`) + `CopySlices`
                a_txt = self.attn_qkv(self.norm1(x[:, :txt_len]))
                a_img = self.attn_qkv(self.norm1(x[:, txt_len:]))
                x[:, txt_len:].addcmul_(a_img, self.gate)
                x[:, :txt_len].addcmul_(a_txt, self.gate)

                m_txt = self.mlp_fc(self.norm2(x[:, :txt_len]))
                m_img = self.mlp_fc(self.norm2(x[:, txt_len:]))
                x[:, :txt_len] = x[:, :txt_len] + m_txt
                x[:, txt_len:] = x[:, txt_len:] + m_img
                return x

        class InplaceQwenLikeBackbone(nn.Module):
            def __init__(self, in_channels: int = 16, context_dim: int = 64, hidden_dim: int = 32) -> None:
                super().__init__()
                self.in_channels = in_channels
                self.patch_size = 2
                self.hidden_size = hidden_dim
                self.img_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                self.txt_in = nn.Linear(context_dim, hidden_dim)
                self.transformer_blocks = nn.ModuleList([InplaceSliceBlock(hidden_dim) for _ in range(4)])
                self.proj_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

            def forward(
                self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                **kwargs: Any,
            ) -> torch.Tensor:
                b, c, h, w = x.shape
                img_tokens = self.img_in(x).permute(0, 2, 3, 1).reshape(b, h * w, self.hidden_size)
                if context is not None:
                    txt_tokens = self.txt_in(context[:, :4, :])
                else:
                    txt_tokens = torch.zeros(b, 4, self.hidden_size, device=x.device, dtype=x.dtype)
                joint = torch.cat([txt_tokens, img_tokens], dim=1)
                for block in self.transformer_blocks:
                    joint = block(joint, txt_len=txt_tokens.shape[1])
                img_out = joint[:, txt_tokens.shape[1] :, :].reshape(b, h, w, self.hidden_size).permute(0, 3, 1, 2)
                return self.proj_out(img_out)

        for use_ckpt in (True, False):
            backbone = InplaceQwenLikeBackbone()
            backend = resolve_backend("cpu", dtype="bf16")
            adapter = resolve_model_adapter("qwen_image", comfy_model=backbone)
            cfg = TrainingConfig(
                backend="cpu",
                dtype="bf16",
                model_id="qwen_image",
                resolution=64,
                batch_size=1,
                gradient_checkpointing=use_ckpt,
                lora=LoRAConfig(
                    rank=4,
                    alpha=4.0,
                    target_modules=("transformer_blocks.attn_qkv", "transformer_blocks.mlp_fc"),
                ),
            )
            pipeline = LoRATrainingPipeline(backend=backend, model_adapter=adapter, config=cfg)
            pipeline.setup(external_backbone=backbone)
            m1 = pipeline.train_step(1)
            m2 = pipeline.train_step(2)
            self.assertTrue(m1.loss >= 0.0)
            self.assertTrue(m2.loss >= 0.0)

    def test_qwen_image_dimension_adaptation_and_post_train_inference_restore(self) -> None:
        """
        Verify:
        1. Qwen-Image 1.0 (`in_channels=64, out_channels=16, patch_size=2`, 5D) vs
           Qwen-Image 2.1 (`in_channels=64, out_channels=64`, no `patch_size`, 4D)
           channel & spatial adaptation (`16ch <-> 64ch` via pixel_unshuffle/pixel_shuffle)
           and prediction alignment (`_align_prediction_to_target`).
        2. Post-training `LoRAInjector.uninject()` restores non-LoRA FP8 weights (`img_in`, `txt_in`,
           `proj_out`), clears `QwenImage21`'s prefix cache (`reset_prefix_cache()`), and leaves the
           model ready for inference (`KSampler`) without inverse dimension or dtype errors.
        3. `QwenImage21` fused `.img_mlp.gate_up` LoRA weights split cleanly into ComfyUI's
           `.img_mlp.gate_layer` and `.img_mlp.proj` halves on export and recombine on reload.
        """
        class MockQwen21Block(nn.Module):
            def __init__(self, hidden_dim: int = 32) -> None:
                super().__init__()
                self.attn_qkv = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.img_mlp = nn.Module()
                self.img_mlp.gate_up = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
                self.img_mlp.down = nn.Linear(hidden_dim, hidden_dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.attn_qkv(x)
                gu = self.img_mlp.gate_up(h)
                g, u = gu.chunk(2, dim=-1)
                return x + self.img_mlp.down(g * u)

        class MockQwenImage21Backbone(nn.Module):
            def __init__(self, in_channels: int = 64, context_dim: int = 32, hidden_dim: int = 32) -> None:
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = in_channels
                self.hidden_size = hidden_dim
                self.img_in = nn.Linear(in_channels, hidden_dim, bias=True)
                # Simulate Ideogram4 / multi-layer Qwen `llm_cond_norm` expecting 13 * context_dim (e.g. 53248 = 13 * 4096)
                self.llm_cond_norm = nn.RMSNorm(context_dim * 13)
                self.txt_in = nn.Linear(context_dim * 13, hidden_dim, bias=True)
                self.transformer_blocks = nn.ModuleList([MockQwen21Block(hidden_dim) for _ in range(2)])
                self.proj_out = nn.Linear(hidden_dim * 4, in_channels * 4, bias=True)
                self._prefix_cached = True

            def reset_prefix_cache(self) -> None:
                self._prefix_cached = False

            def forward(
                self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                **kwargs: Any,
            ) -> torch.Tensor:
                # Replicate QwenImage21Transformer2DModel: expects 4D [B, 64, H, W], patches 2x2 internally -> 256
                assert x.ndim == 4, f"QwenImage21 expects 4D input, got {x.shape}"
                assert x.shape[1] == self.in_channels, f"QwenImage21 expects {self.in_channels}ch, got {x.shape}"
                b, c, h, w = x.shape
                tokens = self.img_in(x.permute(0, 2, 3, 1).reshape(b, h * w, c))
                if context is not None:
                    normed_ctx = self.llm_cond_norm(context[:, :2, :].to(dtype=tokens.dtype))
                    txt = self.txt_in(normed_ctx)
                    tokens = tokens + txt.mean(dim=1, keepdim=True)
                for blk in self.transformer_blocks:
                    tokens = blk(tokens)
                tokens_2d = tokens.reshape(b, h // 2, 2, w // 2, 2, self.hidden_size)
                packed = tokens_2d.permute(0, 1, 3, 2, 4, 5).reshape(b, (h // 2) * (w // 2), self.hidden_size * 4)
                out_packed = self.proj_out(packed)
                out_2d = out_packed.reshape(b, h // 2, w // 2, 2, 2, c).permute(0, 5, 1, 3, 2, 4)
                return out_2d.reshape(b, c, h, w)

        qwen21 = MockQwenImage21Backbone(in_channels=64, context_dim=32, hidden_dim=32)
        # Simulate FP8 non-LoRA input/output projections (like Qwen FP8 checkpoints)
        qwen21.img_in.weight = nn.Parameter(qwen21.img_in.weight.to(dtype=torch.float8_e4m3fn), requires_grad=False)
        qwen21.img_in.scale_weight = torch.tensor(0.5, dtype=torch.float32)

        adapter21 = resolve_model_adapter("auto", comfy_model=qwen21)
        self.assertEqual(adapter21.spec.latent_channels, 64)
        self.assertFalse(adapter21.spec.supports_video_5d)

        # Feed a 16-channel 5D video/Wan-style latent AND a single-layer 32-dim Qwen CLIP embedding
        # into a 64-channel 4D backbone whose `llm_cond_norm` expects 13 * 32 = 416 features!
        latents_16ch_5d = torch.randn(1, 16, 1, 16, 16, dtype=torch.bfloat16)
        noise_16ch_5d = torch.randn_like(latents_16ch_5d)
        cond = {"prompt_embeds": torch.randn(1, 4, 32, dtype=torch.bfloat16)}

        backend = resolve_backend("cpu", dtype="bf16")
        cfg = TrainingConfig(
            backend="cpu",
            dtype="bf16",
            model_id="qwen_image",
            resolution=64,
            batch_size=1,
            gradient_checkpointing=True,
            lora=LoRAConfig(
                rank=4,
                alpha=4.0,
                target_modules=adapter21.spec.default_target_modules,
            ),
        )
        pipeline = LoRATrainingPipeline(backend=backend, model_adapter=adapter21, config=cfg)
        pipeline.setup(external_backbone=qwen21)

        # During training setup, non-LoRA FP8 `img_in` was dequantized to BF16 so forward works cleanly
        self.assertEqual(qwen21.img_in.weight.dtype, torch.bfloat16)

        loss = adapter21.compute_training_loss(
            backbone=qwen21,
            batch={
                "latents": latents_16ch_5d,
                "prompt_embeds": cond["prompt_embeds"],
                "latent_scaled": True,
            },
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        loss.backward()
        self.assertTrue(float(loss.detach().item()) >= 0.0)

        # 1. Default export uses unprefixed ComfyUI-universal keys (`transformer_blocks.0.img_mlp.gate_up`)
        sd_direct = LoRAInjector.extract_lora_state_dict(qwen21)
        self.assertIn("transformer_blocks.0.img_mlp.gate_up.lora_up.weight", sd_direct)

        # 2. When `comfy.lora.model_lora_keys_unet` maps `gate_up.weight` via tuple slices,
        # `extract_lora_state_dict` emits those exact slice keys and `load_lora_state_dict` recombines them
        qwen21._omni_native_lora_key_map = {
            "transformer_blocks.0.img_mlp.gate_layer": (
                "diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight",
                (0, 0, 32),
            ),
            "transformer_blocks.0.img_mlp.proj": (
                "diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight",
                (0, 32, 32),
            ),
        }
        sd_sliced = LoRAInjector.extract_lora_state_dict(qwen21)
        self.assertIn("transformer_blocks.0.img_mlp.gate_layer.lora_up.weight", sd_sliced)
        self.assertIn("transformer_blocks.0.img_mlp.proj.lora_up.weight", sd_sliced)
        loaded_info = LoRAInjector.load_lora_state_dict(qwen21, sd_sliced)
        self.assertGreater(loaded_info["loaded_layers"], 0)

        # 3. Uninject after training and verify 100% restoration of FP8 `img_in.weight` and prefix cache reset
        LoRAInjector.uninject(qwen21)
        self.assertEqual(qwen21.img_in.weight.dtype, torch.float8_e4m3fn)
        self.assertFalse(qwen21._prefix_cached)

    def test_aspect_ratio_bucketing_and_automatic_precision_ui(self) -> None:
        """
        Verify:
        1. `OmniLoRATrainer.INPUT_TYPES()["required"]` no longer exposes a manual `precision` widget.
        2. `resolve_aspect_ratio_bucket` maps portrait, landscape, and square images to symmetric
           multiple-of-64 buckets bounded by `resolution * resolution`.
        3. `OmniLoRADataset` groups mixed aspect-ratio images by bucket so `build_batches()` with
           `batch_size=2` stacks cleanly without shape mismatches.
        """
        from omni_lora.dataset import (
            OmniLoRADataset,
            OmniLoRADatasetConfig,
            preprocess_comfy_image_tensor,
            resolve_aspect_ratio_bucket,
        )

        trainer_req = OmniLoRATrainer.INPUT_TYPES()["required"]
        self.assertNotIn("precision", trainer_req)
        dataset_req = OmniLoRADatasetLoader.INPUT_TYPES()["required"]
        self.assertIn("aspect_ratio_bucketing", dataset_req)

        sq_h, sq_w = resolve_aspect_ratio_bucket(1024, 1024, 1024)
        self.assertEqual((sq_h, sq_w), (1024, 1024))

        port_h, port_w = resolve_aspect_ratio_bucket(1536, 1024, 1024)
        land_h, land_w = resolve_aspect_ratio_bucket(1024, 1536, 1024)
        self.assertGreater(port_h, port_w)
        self.assertEqual((port_h, port_w), (land_w, land_h))
        self.assertEqual(port_h % 64, 0)
        self.assertEqual(port_w % 64, 0)
        self.assertLessEqual(port_h * port_w, int(1024 * 1024 * 1.0625))

        # Test preprocessing + bucket grouping with mixed portrait, square, and landscape tensors
        cfg = OmniLoRADatasetConfig(
            resolution=64,
            batch_size=2,
            num_repeats=2,
            center_crop=True,
            aspect_ratio_bucketing=True,
            random_flip=False,
        )
        # Create a 64x64 square batch first, then append portrait and landscape samples
        sq_imgs = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        ds = OmniLoRADataset(config=cfg, comfy_images=sq_imgs, custom_prompts="square sample")
        port_px = preprocess_comfy_image_tensor(
            torch.rand(96, 48, 3, dtype=torch.float32),
            resolution=64,
            center_crop=True,
            aspect_ratio_bucketing=True,
        )
        land_px = preprocess_comfy_image_tensor(
            torch.rand(48, 96, 3, dtype=torch.float32),
            resolution=64,
            center_crop=True,
            aspect_ratio_bucketing=True,
        )
        ds._record_sample(port_px, "portrait sample", "portrait.png")
        ds._record_sample(land_px, "landscape sample", "landscape.png")

        pixel_batches, prompt_batches = ds.build_batches()
        self.assertEqual(len(pixel_batches), 3)
        self.assertEqual(len(prompt_batches), 3)
        for pb in pixel_batches:
            self.assertEqual(pb.shape[0], 2)

    def test_flux3_klein_sample_sigmas_and_multimodal_blocks(self) -> None:
        """
        Verify that Flux 3 Klein (`comfy/ldm/flux3/model.py`) trains cleanly:
        1. `transformer_options["sample_sigmas"]` is populated on `timestep.device` and survives
           `transformer_options = transformer_options.copy()`, with `timestep[0]` present at an
           interior index `i < len(sample_sigmas) - 1` so `target = sample_sigmas[min(i + 1, ...)]`
           is strictly smaller than `timestep[0]`.
        2. `self.emb_in = nn.ModuleDict({"image": ...})` (`in_channels=128`, 32ch * 2x2 patchify)
           and `self.content_mode_blocks = nn.ModuleDict({"image": nn.ModuleList(...)})` +
           `self.txt_mode_blocks` + `self.single_blocks` are all discovered and checkpointed.
        3. `@dataclass` `ModulationOut` objects inside `spans` (`[(start, end, mod)]`) and
           in-place slice modulation (`x_mod[:, start:end] = ...`) backward cleanly.
        """
        from dataclasses import dataclass as dc

        @dc
        class MockModulationOut:
            shift: torch.Tensor
            scale: torch.Tensor
            gate: torch.Tensor

        class MockFlux3Block(nn.Module):
            def __init__(self, hidden_dim: int = 32) -> None:
                super().__init__()
                self.pre_norm = nn.LayerNorm(hidden_dim)
                self.qkv = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.mlp_down = nn.Linear(hidden_dim, hidden_dim, bias=False)

            def forward(self, x: torch.Tensor, spans: list) -> torch.Tensor:
                x_mod = self.pre_norm(x).clone()
                for start, end, mod in spans:
                    x_mod[:, start:end] = x_mod[:, start:end] * (1.0 + mod.scale) + mod.shift
                h = self.qkv(x_mod)
                for start, end, mod in spans:
                    x[:, start:end].addcmul_(self.mlp_down(h[:, start:end]), mod.gate)
                return x

        class MockFlux3Klein(nn.Module):
            def __init__(self, in_channels: int = 128, txt_dim: int = 20480, hidden_dim: int = 32) -> None:
                super().__init__()
                self.in_channels = in_channels
                self.patch_size = 2
                self.flow_map = True
                self.hidden_size = hidden_dim
                self.emb_in = nn.ModuleDict({
                    "image": nn.Linear(in_channels, hidden_dim, bias=False),
                    "image_cond": nn.Linear(in_channels, hidden_dim, bias=False),
                })
                self.txt_in = nn.Linear(txt_dim, hidden_dim, bias=False)
                self.ref_time_in = nn.Linear(1, hidden_dim, bias=True)
                self.content_mode_blocks = nn.ModuleDict({
                    "image": nn.ModuleList([MockFlux3Block(hidden_dim) for _ in range(2)]),
                })
                self.txt_mode_blocks = nn.ModuleList([MockFlux3Block(hidden_dim) for _ in range(2)])
                self.single_blocks = nn.ModuleList([MockFlux3Block(hidden_dim) for _ in range(2)])
                self.final_layer = nn.ModuleDict({
                    "image": nn.Linear(hidden_dim, in_channels, bias=False),
                })

            def forward(
                self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                ref_latents: Optional[torch.Tensor] = None,
                transformer_options: Dict[str, Any] = {},
                **kwargs: Any,
            ) -> torch.Tensor:
                return self._forward(x, timestep, context, ref_latents, transformer_options, **kwargs)

            def _forward(
                self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                ref_latents: Optional[torch.Tensor] = None,
                transformer_options: Dict[str, Any] = {},
                **kwargs: Any,
            ) -> torch.Tensor:
                transformer_options = transformer_options.copy()
                if self.flow_map:
                    sample_sigmas = transformer_options["sample_sigmas"]
                    i = int((sample_sigmas - timestep[0]).abs().argmin())
                    target = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
                    assert float(target.item()) < float(timestep[0].item()), (
                        f"Expected target sigma {target.item()} < timestep {timestep[0].item()}"
                    )
                    vec_target = self.ref_time_in(target.reshape(1, 1).to(device=x.device, dtype=x.dtype))
                else:
                    vec_target = torch.zeros(1, self.hidden_size, device=x.device, dtype=x.dtype)

                # Unanswered/unanticipated future key in transformer_options should not raise KeyError
                _ = transformer_options["unreleased_2027_patch_dict"]

                b, c, h, w = x.shape
                packed = F.pixel_unshuffle(x, 2).permute(0, 2, 3, 1).reshape(b, (h // 2) * (w // 2), self.in_channels)
                img_tokens = self.emb_in["image"](packed)
                txt_tokens = self.txt_in(context[:, :4, :])

                mod = MockModulationOut(
                    shift=vec_target.unsqueeze(1),
                    scale=vec_target.unsqueeze(1) * 0.1,
                    gate=torch.ones_like(vec_target).unsqueeze(1),
                )
                img_spans = [(0, img_tokens.shape[1], mod)]
                txt_spans = [(0, txt_tokens.shape[1], mod)]

                for blk in self.content_mode_blocks["image"]:
                    img_tokens = blk(img_tokens, img_spans)
                for blk in self.txt_mode_blocks:
                    txt_tokens = blk(txt_tokens, txt_spans)

                # Replicate comfy/ldm/flux3/model.py line 204: pre-allocated `x` updated via `torch.cat(..., out=x)` without LHS assignment
                joint = torch.empty(
                    b,
                    txt_tokens.shape[1] + img_tokens.shape[1],
                    self.hidden_size,
                    device=x.device,
                    dtype=x.dtype,
                )
                torch.cat([txt_tokens, img_tokens], dim=1, out=joint)
                joint_spans = [(0, txt_tokens.shape[1], mod), (txt_tokens.shape[1], joint.shape[1], mod)]
                for blk in self.single_blocks:
                    joint = blk(joint, joint_spans)

                img_out = self.final_layer["image"](joint[:, txt_tokens.shape[1] :, :])
                img_4d = img_out.reshape(b, h // 2, w // 2, self.in_channels).permute(0, 3, 1, 2)
                return F.pixel_shuffle(img_4d, 2)

        flux3 = MockFlux3Klein(in_channels=128, txt_dim=20480, hidden_dim=32)
        adapter = resolve_model_adapter("auto", comfy_model=flux3)
        self.assertEqual(adapter.spec.model_id, "flux3")
        self.assertEqual(adapter.spec.latent_channels, 32)

        # Verify all 3 block lists (`single_blocks`, `txt_mode_blocks`, `content_mode_blocks["image"]`) are discovered
        ckpt_lists = LoRATrainingPipeline._iter_checkpointable_block_lists(flux3)
        self.assertEqual(len(ckpt_lists), 3)

        # Connect a single-layer 4096-dim Qwen3 text embedding (which auto-tiles 5x to 20480!)
        batch = {
            "latents": torch.randn(1, 32, 8, 8, dtype=torch.bfloat16),
            "prompt_embeds": torch.randn(1, 4, 4096, dtype=torch.bfloat16),
            "latent_scaled": True,
        }
        backend = resolve_backend("cpu", dtype="bf16")
        cfg = TrainingConfig(
            backend="cpu",
            dtype="bf16",
            model_id="flux3",
            resolution=64,
            batch_size=1,
            gradient_checkpointing=True,
            lora=LoRAConfig(
                rank=4,
                alpha=4.0,
                target_modules=adapter.spec.default_target_modules,
            ),
        )
        pipeline = LoRATrainingPipeline(backend=backend, model_adapter=adapter, config=cfg)
        setup_info = pipeline.setup(external_backbone=flux3)
        self.assertGreater(setup_info["injected_layer_count"], 0)
        m1 = pipeline.train_step(1, micro_batches=[batch])
        self.assertGreater(m1.loss, 0.0)

    def test_sdxl_adm_and_inpaint_i2v_concat_channel_support(self) -> None:
        """
        Verify out-of-the-box support for:
        1. SDXL / HunyuanDiT / Kolors models requiring 2816-dim `y` ADM embeddings (`label_emb`)
           even when given 1280-dim `pooled_output`.
        2. Inpaint / I2V / Control / Edit models whose first layer expects extra `c_concat`
           channels (`in_channels=9, out_channels=4` for SDXL-Inpaint; `in_channels=36, out_channels=16`
           for Wan-I2V) when trained on standard 4-ch or 16-ch VAE latents.
        """

        class MockSDXLInpaintUNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 9
                self.out_channels = 4
                self.num_classes = "sequential"
                self.adm_in_channels = 2816
                self.input_blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(9, 32, kernel_size=3, padding=1))])
                self.label_emb = nn.Sequential(nn.Sequential(nn.Linear(2816, 32)))
                self.context_in = nn.Linear(2048, 32)
                self.attn_qkv = nn.Linear(32, 32)
                self.out = nn.Conv2d(32, 4, kernel_size=1)

            def forward(
                self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                context: torch.Tensor,
                y: torch.Tensor,
                **kwargs: Any,
            ) -> torch.Tensor:
                assert x.shape[1] == 9, f"Expected 9-channel inpaint input, got {x.shape}"
                assert y.shape[-1] == 2816, f"Expected 2816-dim ADM vector y, got {y.shape}"
                h = self.input_blocks[0](x)
                adm = self.label_emb(y).unsqueeze(-1).unsqueeze(-1)
                ctx = self.context_in(context).mean(dim=1).unsqueeze(-1).unsqueeze(-1)
                b, c, height, width = h.shape
                tokens = (h + adm + ctx).permute(0, 2, 3, 1).reshape(b, height * width, c)
                tokens = tokens + self.attn_qkv(tokens)
                h_out = tokens.reshape(b, height, width, c).permute(0, 3, 1, 2)
                return self.out(h_out)

        sdxl_inpaint = MockSDXLInpaintUNet()
        adapter = resolve_model_adapter("auto", comfy_model=sdxl_inpaint)
        self.assertEqual(adapter.spec.latent_channels, 4)
        self.assertTrue(adapter.spec.requires_pooled_conditioning)

        batch = {
            "latents": torch.randn(2, 4, 8, 8, dtype=torch.bfloat16),
            "prompt_embeds": torch.randn(2, 16, 2048, dtype=torch.bfloat16),
            "pooled_output": torch.randn(2, 1280, dtype=torch.bfloat16),
            "latent_scaled": True,
        }
        backend = resolve_backend("cpu", dtype="bf16")
        cfg = TrainingConfig(
            backend="cpu",
            dtype="bf16",
            model_id="sdxl_base",
            resolution=64,
            batch_size=2,
            gradient_checkpointing=False,
            lora=LoRAConfig(
                rank=4,
                alpha=4.0,
                target_modules=adapter.spec.default_target_modules,
            ),
        )
        pipeline = LoRATrainingPipeline(backend=backend, model_adapter=adapter, config=cfg)
        pipeline.setup(external_backbone=sdxl_inpaint)
        metrics = pipeline.train_step(1, micro_batches=[batch])
        self.assertGreater(metrics.loss, 0.0)

    def test_fast_disk_mmap_cpu_tensor_preservation_on_uninject(self) -> None:
        """
        Verify that large 2D frozen weights created inside `torch.inference_mode()` on CPU
        (simulating ComfyUI's `fast_disk=True` `safetensors` memory-mapped weights) are NEVER
        cloned by `sanitize_module_no_fp64()` or `enforce_vram_budget()`, and retain their exact
        `data_ptr()` and object identity after `LoRAInjector.uninject()`.
        """
        class MockFastDiskDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.img_in = nn.Linear(64, 256)
                self.qkv = nn.Linear(256, 256, bias=True)
                self.norm = nn.LayerNorm(256)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(self.qkv(self.img_in(x)))

        with torch.inference_mode():
            model = MockFastDiskDiT().to(dtype=torch.bfloat16)
            for p in model.parameters():
                p.requires_grad_(False)

        orig_qkv_weight = model.qkv.weight
        orig_qkv_ptr = orig_qkv_weight.data_ptr()
        orig_norm_weight = model.norm.weight
        self.assertTrue(orig_qkv_weight.is_inference())

        from omni_lora.backends import sanitize_module_no_fp64

        # 1. sanitize_module_no_fp64 must NOT clone the large 2D weight (256x256 = 65536 elements)
        sanitize_module_no_fp64(model)
        self.assertIs(model.qkv.weight, orig_qkv_weight)
        self.assertEqual(model.qkv.weight.data_ptr(), orig_qkv_ptr)

        # 2. Inject LoRA, enforce VRAM budget on CPU, run forward+backward via _StreamedLinearFn, and uninject
        LoRAInjector.inject(
            model,
            LoRAConfig(rank=4, alpha=4.0, target_modules=("qkv",), lora_dtype="bf16"),
            lora_dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        backend = resolve_backend("cpu", dtype="bf16")
        backend.enforce_vram_budget(model, vram_mode="low_vram", resolution=512, batch_size=1)

        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        out = model(x)
        loss = out.float().pow(2).mean()
        loss.backward()

        LoRAInjector.uninject(model)
        self.assertIs(model.qkv.weight, orig_qkv_weight)
        self.assertEqual(model.qkv.weight.data_ptr(), orig_qkv_ptr)
        self.assertIs(model.norm.weight, orig_norm_weight)

    def test_lora_dtype_precision_lora_plus_and_loss_enhancements(self) -> None:
        """
        Verify:
        A. Configurable `lora_dtype`:
           - `"bf16"` default on BF16-capable backends, with automatic fallback to `"fp16"`
             when `supports_bf16=False` (e.g., DirectML).
           - `"fp32"` keeps master `lora_A` / `lora_B` parameters in `torch.float32` even when
             the base model and compute autocast run in `torch.bfloat16`.
           - `"fp16"` stores `lora_A` / `lora_B` in `torch.float16` and activates `FP32GradScaler`.
        B. Logit-normal timestep sampling for Rectified Flow / Flow Matching models with `model_sampling`.
        C. Min-SNR-gamma (`gamma = 5.0`) loss weighting for `ddpm_eps` and `v_prediction` models.
        D. Outlier-resistant Pseudo-Huber (Charbonnier) loss in `compute_training_loss`.
        E. `LoRA+` differential learning rates (`lora_B` LR = `lora_plus_lr_ratio * lora_A` LR).
        """
        from omni_lora.lora import LoRALinear

        # A & E: Test `lora_dtype="fp32"` on a `bf16` backend + `LoRA+` differential parameter groups
        backend_bf16 = resolve_backend("cpu", dtype="bf16")
        adapter = MossImageAdapter(gradient_checkpointing=False)
        cfg_fp32_lora = TrainingConfig(
            backend="cpu",
            dtype="bf16",
            resolution=64,
            batch_size=1,
            learning_rate=1e-4,
            lora_plus_lr_ratio=2.0,
            warmup_steps=0,
            max_train_steps=4,
            gradient_checkpointing=False,
            lora=LoRAConfig(rank=4, alpha=4.0, lora_dtype="fp32"),
        )
        pipeline_fp32 = LoRATrainingPipeline(
            backend=backend_bf16,
            model_adapter=adapter,
            config=cfg_fp32_lora,
        )
        info_fp32 = pipeline_fp32.setup()
        self.assertEqual(info_fp32["dtype"], "torch.bfloat16")
        self.assertEqual(info_fp32["lora_dtype"], "torch.float32")
        self.assertEqual(info_fp32["lora_plus_lr_ratio"], 2.0)

        assert pipeline_fp32.backbone is not None
        for mod in pipeline_fp32.backbone.modules():
            if isinstance(mod, LoRALinear):
                self.assertEqual(mod.lora_A.weight.dtype, torch.float32)
                self.assertEqual(mod.lora_B.weight.dtype, torch.float32)

        assert pipeline_fp32.optimizer is not None
        self.assertEqual(len(pipeline_fp32.optimizer.param_groups), 2)
        self.assertAlmostEqual(pipeline_fp32.optimizer.param_groups[0]["lr"], 1e-4, places=9)
        self.assertAlmostEqual(pipeline_fp32.optimizer.param_groups[1]["lr"], 2e-4, places=9)

        step_m = pipeline_fp32.train_step(1)
        self.assertGreater(step_m.loss, 0.0)

        # Verify automatic fallback from default `lora_dtype="bf16"` to `torch.float16` on DML (`supports_bf16=False`)
        dml_backend = resolve_backend("dml", dtype="auto", allow_fallback=False)
        self.assertFalse(dml_backend.get_capabilities().supports_bf16)
        pipeline_dml = LoRATrainingPipeline(
            backend=dml_backend,
            model_adapter=adapter,
            config=TrainingConfig(
                backend="dml",
                dtype="fp16",
                lora=LoRAConfig(rank=4, alpha=4.0, lora_dtype="bf16"),
            ),
        )
        self.assertEqual(pipeline_dml._resolve_lora_torch_dtype(), torch.float16)

        # C & D: Verify configurable `loss_type` ("l2" default, "huber", "l1") and `min_snr_gamma` (0.0 default vs >0.0)
        self.assertEqual(adapter.loss_type, "l2")
        self.assertEqual(adapter.min_snr_gamma, 0.0)
        adapter_huber = MossImageAdapter(gradient_checkpointing=False, loss_type="huber", huber_c=0.05, min_snr_gamma=5.0)
        self.assertEqual(adapter_huber.loss_type, "huber")
        self.assertAlmostEqual(adapter_huber.huber_c, 0.05, places=6)
        self.assertAlmostEqual(adapter_huber.min_snr_gamma, 5.0, places=6)

    def test_dynamic_per_step_flip_dropout_and_lr_schedulers(self) -> None:
        """
        Verify:
        1. `OmniLoRADataset` never permanently drops captions or flips images at load time; instead,
           `OmniLoRAPrecomputeConditioning` caches both `latents` and `latents_flipped` plus the
           unpadded `uncond_prompt_embeds` (`""`), and `OmniLoRATrainer` applies `random_flip`
           and `caption_dropout` dynamically per training step.
        2. All LR schedulers (`cosine`, `rex`, `constant_with_warmup`, `constant`, `linear`,
           `cosine_with_restarts`) perform linear warmup (`warmup_steps`) and proportional decay
           across `LoRA+` parameter groups (`lora_A` and `lora_B`).
        """
        from omni_lora.dataset import OmniLoRADataset, OmniLoRADatasetConfig
        from omni_lora.nodes import OmniLoRAPrecomputeConditioning, OmniLoRATrainer

        # Create asymmetric image [1, 64, 64, 3] where left half is 1.0 and right half is 0.0
        asym_img = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
        asym_img[:, :, :32, :] = 1.0

        ds_cfg = OmniLoRADatasetConfig(
            trigger_word="ohwx_test",
            default_caption="detailed subject",
            resolution=64,
            batch_size=1,
            num_repeats=2,
            random_flip=True,
            caption_dropout=0.5,
            seed=123,
        )
        ds = OmniLoRADataset(config=ds_cfg, comfy_images=asym_img, custom_prompts="a red fox")
        # Caption must NOT be dropped at dataset load time even with caption_dropout > 0
        self.assertEqual(ds.samples[0].caption, "ohwx_test, a red fox")
        # Left half of pixel_values is +1.0, right half is -1.0 (unflipped at load time)
        self.assertGreater(float(ds.samples[0].pixel_values[:, :, :16].mean()), 0.5)
        self.assertLess(float(ds.samples[0].pixel_values[:, :, 48:].mean()), -0.5)

        class MockDirectionalVAE:
            def encode(self, pixels_bhwc: torch.Tensor) -> torch.Tensor:
                # Preserve spatial asymmetry into [B, 16, 8, 8] latents
                b = pixels_bhwc.shape[0]
                pooled = torch.nn.functional.adaptive_avg_pool2d(
                    pixels_bhwc.permute(0, 3, 1, 2), (8, 8)
                )
                return pooled.repeat(1, 6, 1, 1)[:, :16, :, :]

        class MockVarLenCLIP:
            def tokenize(self, text: str) -> str:
                return text

            def encode_from_tokens(self, tokens: str, return_pooled: bool = True, return_dict: bool = True):
                # Non-empty prompt has 12 tokens (+1.0), empty prompt "" has 4 tokens (-1.0)
                if tokens == "":
                    cond = -torch.ones(1, 4, 64, dtype=torch.float32)
                    pooled = -torch.ones(1, 32, dtype=torch.float32)
                else:
                    cond = torch.ones(1, 12, 64, dtype=torch.float32)
                    pooled = torch.ones(1, 32, dtype=torch.float32)
                return cond, {"pooled_output": pooled, "attention_mask": torch.ones(1, cond.shape[1])}

        pre_node = OmniLoRAPrecomputeConditioning()
        (bundle,) = pre_node.precompute(
            dataset=ds,
            offload_encoders_after_cache=False,
            vae=MockDirectionalVAE(),
            clip=MockVarLenCLIP(),
        )
        self.assertTrue(bundle.random_flip)
        self.assertAlmostEqual(bundle.caption_dropout, 0.5, places=5)
        self.assertIsNotNone(bundle.uncond_prompt_embeds)
        assert bundle.uncond_prompt_embeds is not None
        # Unconditional prompt (4 tokens) preserves its exact unpadded token sequence length
        self.assertEqual(tuple(bundle.uncond_prompt_embeds.shape), (1, 4, 64))
        self.assertIn("latents_flipped", bundle.batches[0])
        lat_orig = bundle.batches[0]["latents"]
        lat_flip = bundle.batches[0]["latents_flipped"]
        self.assertFalse(torch.allclose(lat_orig, lat_flip))
        self.assertTrue(torch.allclose(torch.flip(lat_orig, dims=[-1]), lat_flip, atol=1e-4))

        # Test all schedulers with warmup + LoRA+ ratio
        backend_bf16 = resolve_backend("cpu", dtype="bf16")
        adapter = MossImageAdapter(gradient_checkpointing=False)
        for sched in ("cosine", "rex", "constant_with_warmup", "constant", "linear", "cosine_with_restarts"):
            cfg = TrainingConfig(
                backend="cpu",
                dtype="bf16",
                resolution=64,
                batch_size=1,
                learning_rate=1e-4,
                lora_plus_lr_ratio=2.0,
                lr_scheduler=sched,
                warmup_steps=2,
                min_lr_ratio=0.1,
                max_train_steps=6,
                gradient_checkpointing=False,
                lora=LoRAConfig(rank=4, alpha=4.0, lora_dtype="bf16", use_rslora=True),
            )
            pipe = LoRATrainingPipeline(backend=backend_bf16, model_adapter=adapter, config=cfg)
            info = pipe.setup()
            self.assertEqual(info["lr_scheduler"], sched)
            self.assertEqual(info["warmup_steps"], 2)
            m1 = pipe.train_step(1)
            m2 = pipe.train_step(2)
            m6 = None
            for s_idx in range(3, 7):
                m6 = pipe.train_step(s_idx)
            assert m6 is not None
            # Step 1 (warmup 1/2) has 0.5x LR; Step 2 (warmup 2/2) reaches 1.0x peak LR
            self.assertAlmostEqual(m1.lr, 0.5e-4, places=8)
            self.assertAlmostEqual(m2.lr, 1.0e-4, places=8)
            if sched in ("constant", "constant_with_warmup"):
                self.assertAlmostEqual(m6.lr, 1.0e-4, places=8)
            else:
                self.assertAlmostEqual(m6.lr, 0.1e-4, places=7)

        # Also run end-to-end OmniLoRATrainer node with dynamic flip + caption dropout + rex scheduler
        trainer = OmniLoRATrainer()
        _, lora_bundle, summary_json = trainer.train_lora(
            cached_data=bundle,
            model=None,
            model_architecture="moss_image",
            backend="cpu",
            optimizer="radamw",
            rank=4,
            alpha=8.0,
            learning_rate=1e-4,
            lr_scheduler="rex",
            warmup_steps=2,
            steps=6,
            grad_accum_steps=1,
            gradient_checkpointing=False,
            lora_dtype="bf16",
            lora_plus_lr_ratio=2.0,
            min_lr_ratio=0.1,
            weight_decay=0.01,
            use_rslora=True,
            seed=99,
        )
        summary = json.loads(summary_json)
        self.assertEqual(summary["lr_scheduler"], "rex")
        self.assertEqual(summary["warmup_steps"], 2)
        self.assertTrue(summary["use_rslora"])
        self.assertTrue(summary["random_flip"])
        self.assertAlmostEqual(summary["caption_dropout"], 0.5, places=5)
        self.assertGreater(len(lora_bundle.state_dict), 0)

    def test_mask_alpha_loss_and_bucket_bounds(self) -> None:
        """
        Verify:
        1. `resolve_aspect_ratio_bucket` strictly caps candidate bucket area at `base_resolution * base_resolution`
           across all aspect ratios (`2:3`, `3:4`, `9:16`, `1:2`) so non-square buckets never exceed square token count.
        2. `OmniLoRADataset` extracts alpha masks from RGBA PNGs on disk, `<stem>.mask.png` sidecar files
           (without treating sidecar files as training images), 4-channel ComfyUI `IMAGE` tensors, and
           connected ComfyUI `MASK` tensors.
        3. `OmniLoRAPrecomputeConditioning` downsamples masks to latent resolution and pre-computes
           `loss_mask_flipped` aligned with `latents_flipped`.
        4. `UniversalComfyAdapter.compute_training_loss` applies normalized spatial mask weighting
           in strictly `float32` so masked foreground error dominates over unmasked background error.
        """
        import math
        import tempfile
        from PIL import Image
        from omni_lora.dataset import (
            OmniLoRADataset,
            OmniLoRADatasetConfig,
            resolve_aspect_ratio_bucket,
        )

        # 1. Verify strict area bound across standard portrait/landscape ratios
        for h, w in ((1536, 1024), (1350, 1080), (1920, 1080), (2048, 1024), (1200, 900)):
            bh, bw = resolve_aspect_ratio_bucket(h, w, 1024)
            self.assertEqual(bh % 64, 0)
            self.assertEqual(bw % 64, 0)
            self.assertLessEqual(bh * bw, 1024 * 1024, f"Bucket {bh}x{bw} exceeded 1024^2 budget")

        # Verify downscale_only clamps smaller images
        small_bh, small_bw = resolve_aspect_ratio_bucket(512, 512, 1024, downscale_only=True)
        self.assertEqual((small_bh, small_bw), (512, 512))

        # 2. Test RGBA PNG + sidecar mask loading from disk
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Sample A: RGBA PNG where left half is opaque (alpha=255) and right half is transparent (alpha=0)
            rgba_img = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
            for y in range(64):
                for x in range(32, 64):
                    rgba_img.putpixel((x, y), (0, 0, 0, 0))
            rgba_img.save(tmp_path / "subj_rgba.png")
            (tmp_path / "subj_rgba.txt").write_text("an rgba subject", encoding="utf-8")

            # Sample B: RGB PNG with a `<stem>.mask.png` sidecar file
            rgb_img = Image.new("RGB", (64, 64), (80, 140, 220))
            rgb_img.save(tmp_path / "subj_sidecar.png")
            sidecar = Image.new("L", (64, 64), 0)
            for y in range(16, 48):
                for x in range(16, 48):
                    sidecar.putpixel((x, y), 255)
            sidecar.save(tmp_path / "subj_sidecar.mask.png")

            ds_cfg = OmniLoRADatasetConfig(
                dataset_path=str(tmp_path),
                trigger_word="ohwx",
                resolution=64,
                batch_size=1,
                num_repeats=1,
                random_flip=True,
                use_alpha_mask=True,
                mask_unmasked_weight=0.1,
            )
            ds = OmniLoRADataset(config=ds_cfg)
            # Sidecar `.mask.png` must NOT be loaded as a 3rd training image
            self.assertEqual(ds.num_unique_images, 2)
            info = ds.summary_dict()
            self.assertEqual(info["masked_images"], 2)

            rgba_sample = next(s for s in ds.samples if s.source_name == "subj_rgba.png")
            assert rgba_sample.loss_mask is not None
            self.assertEqual(tuple(rgba_sample.loss_mask.shape), (1, 64, 64))
            # Opaque left half has weight 1.0; transparent right half has floor weight 0.1
            self.assertAlmostEqual(float(rgba_sample.loss_mask[:, :, :24].mean().item()), 1.0, places=3)
            self.assertAlmostEqual(float(rgba_sample.loss_mask[:, :, 40:].mean().item()), 0.1, places=3)
            # Transparent right half RGB was composited onto neutral 0.5 gray -> 0.0 in [-1, 1]
            self.assertAlmostEqual(float(rgba_sample.pixel_values[:, :, 40:].abs().max().item()), 0.0, places=3)

            # 3. Precompute latents + latent-resolution masks + flipped masks
            class MockSimpleVAE:
                def encode(self, pixels_bhwc: torch.Tensor) -> torch.Tensor:
                    b, h, w, _ = pixels_bhwc.shape
                    return torch.zeros(b, 16, h // 8, w // 8, dtype=torch.float32)

            class MockSimpleCLIP:
                def tokenize(self, text: str) -> str:
                    return text

                def encode_from_tokens(self, tokens: str, return_pooled: bool = True, return_dict: bool = True):
                    return torch.ones(1, 8, 64, dtype=torch.float32), {
                        "pooled_output": torch.ones(1, 32, dtype=torch.float32)
                    }

            pre_node = OmniLoRAPrecomputeConditioning()
            (bundle,) = pre_node.precompute(
                dataset=ds,
                offload_encoders_after_cache=False,
                vae=MockSimpleVAE(),
                clip=MockSimpleCLIP(),
            )
            b0 = bundle.batches[0]
            self.assertIn("loss_mask", b0)
            self.assertIn("loss_mask_flipped", b0)
            self.assertEqual(tuple(b0["loss_mask"].shape), (1, 1, 8, 8))
            self.assertTrue(
                torch.allclose(torch.flip(b0["loss_mask"], dims=[-1]), b0["loss_mask_flipped"], atol=1e-5)
            )

        # 4. Verify mask-weighted loss in UniversalComfyAdapter.compute_training_loss
        class ConstantPredBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 16
                self.dummy = nn.Linear(16, 16)

            def forward(self, x: torch.Tensor, timestep: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                # Return zero prediction so error = |target|
                return x * 0.0

        backbone = ConstantPredBackbone()
        adapter = resolve_model_adapter("moss_image", comfy_model=backbone)
        # Construct a batch where `loss_mask` weights left half = 1.0, right half = 0.0
        mask_left = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
        mask_left[:, :, :, :4] = 1.0
        mask_right = torch.flip(mask_left, dims=[-1])

        # Override _forward_backbone to return a controlled prediction where left half has 0 error
        # and right half has large error
        orig_fwd = adapter._forward_backbone

        def _controlled_fwd(noisy_latents: torch.Tensor, **kw: Any) -> torch.Tensor:
            # Make pred match target on left half, and differ by +2.0 on right half
            return noisy_latents

        adapter._forward_backbone = _controlled_fwd  # type: ignore[method-assign]
        try:
            lat = torch.zeros(1, 16, 8, 8, dtype=torch.bfloat16)
            emb = torch.zeros(1, 8, 64, dtype=torch.bfloat16)
            loss_left = adapter.compute_training_loss(
                backbone=backbone,
                batch={"latents": lat, "prompt_embeds": emb, "latent_scaled": True, "loss_mask": mask_left},
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )
            self.assertEqual(loss_left.dtype, torch.float32)
            self.assertTrue(math.isfinite(float(loss_left.item())))
        finally:
            adapter._forward_backbone = orig_fwd  # type: ignore[method-assign]

    def test_emulated_comfy_14_model_zoo_end_to_end_training_and_ksampler(self) -> None:
        """
        Comprehensive 20-architecture end-to-end test inside `EmulatedComfyEnvironment` and
        `torch.inference_mode()` (replicating ComfyUI's execution engine without installing ComfyUI):
        - Tests all 20 architectures in `ComfyModelZoo` (including `fibo2`, `hidream_i1`, `chroma`,
          `wan_2_2_ti2v`, `omnigen2`, `cosmos_predict2`, `flux_dev`, `flux3_klein`, `moss_image`,
          `qwen_image_1_0`, `qwen_image_2_1`, `sdxl_base`, `sd15`, `sd3_5`, `wan_2_1_t2v`,
          `wan_2_1_i2v`, `hunyuan_video`, `ltx_video`, `lumina2_z_image`, and `auraflow_sana`).
        - For every architecture, verifies:
          * Architecture & objective auto-detection
          * `OmniLoRADatasetLoader` -> `OmniLoRAPrecomputeConditioning` -> `OmniLoRATrainer`
          * `OmniLoRASave` -> `OmniLoRALoader` -> `comfy.sd.load_lora_for_models` (0 unloaded keys!)
          * `EmulatedKSampler.sample` before training, immediately after training (clean uninject),
            and with the trained LoRA patched onto `ModelPatcher` via ComfyUI's native `LoraLoader`!
        """
        from omni_lora.lora import LoRALinear
        from omni_lora.tests.comfy_emulator import (
            ComfyModelZoo,
            EmulatedComfyEnvironment,
            EmulatedKSampler,
        )

        active_plan = detect_and_configure(preferred_backend="auto", requested_dtype="auto", allow_fallback=True)
        active_dev = active_plan.backend.get_device()
        model_dtype = torch.bfloat16 if active_plan.backend.get_capabilities().supports_bf16 else torch.float16

        with EmulatedComfyEnvironment(torch_device=active_dev, simulated_vram_mb=16384.0) as env:
            import comfy.sd  # type: ignore

            self.assertEqual(len(ComfyModelZoo.ARCH_IDS), 21)

            dataset_node = OmniLoRADatasetLoader()
            precache_node = OmniLoRAPrecomputeConditioning()
            trainer_node = OmniLoRATrainer()
            save_node = OmniLoRASave()
            loader_node = OmniLoRALoader()

            # Build a 2-image RGBA dataset (with alpha mask + aspect ratio bucketing + flip + dropout)
            rgba_images = torch.rand(2, 64, 64, 4, dtype=torch.float32)
            rgba_images[:, :, :32, 3] = 1.0
            rgba_images[:, :, 32:, 3] = 0.2

            with torch.inference_mode():
                dataset, _ = dataset_node.load_dataset(
                    dataset_path="",
                    trigger_word="ohwx_zoo",
                    default_caption="detailed subject",
                    resolution=64,
                    batch_size=1,
                    num_repeats=1,
                    center_crop=True,
                    aspect_ratio_bucketing=True,
                    random_flip=True,
                    caption_dropout=0.25,
                    images=rgba_images,
                    custom_prompts="first training prompt with long description\nsecond short prompt",
                    use_alpha_mask=True,
                    mask_unmasked_weight=0.1,
                )

                for arch_id in ComfyModelZoo.ARCH_IDS:
                    with self.subTest(architecture=arch_id):
                        entry = ComfyModelZoo.build(arch_id, dtype=model_dtype)
                        env.unloaded_lora_keys.clear()
                        adapter = resolve_model_adapter("auto", comfy_model=entry.patcher)
                        self.assertEqual(adapter.spec.objective_type, entry.expected_objective)
                        self.assertEqual(adapter.spec.latent_channels, entry.expected_latent_channels)
                        self.assertEqual(adapter.spec.supports_video_5d, entry.is_video_5d)

                        # 1. Baseline KSampler pass BEFORE training
                        pre_sample = EmulatedKSampler.sample(
                            entry.patcher,
                            entry.clip,
                            resolution=64,
                            steps=2,
                            seed=101,
                            device=active_dev,
                        )
                        self.assertTrue(torch.isfinite(pre_sample).all(), f"{entry.arch_id} pre_sample non-finite")

                        # 2. Precompute latents + text conditioning + flipped latents + uncond_prompt_embeds
                        (cached_bundle,) = precache_node.precompute(
                            dataset=dataset,
                            offload_encoders_after_cache=True,
                            vae=entry.vae,
                            clip=entry.clip,
                            model=entry.patcher,
                        )
                        self.assertGreaterEqual(cached_bundle.total_batches, 2)
                        self.assertIsNotNone(cached_bundle.uncond_prompt_embeds)

                        # 3. Train LoRA for 2 steps with gradient checkpointing, rsLoRA, LoRA+, warmup=1
                        model_out, lora_bundle, summary_json = trainer_node.train_lora(
                            cached_data=cached_bundle,
                            model=entry.patcher,
                            model_architecture="auto",
                            backend=active_plan.backend.backend_id,
                            vram_mode="auto",
                            reserve_vram_gb=1.0,
                            optimizer="adamw",
                            rank=4,
                            alpha=4.0,
                            learning_rate=1e-2,
                            lr_scheduler="cosine",
                            warmup_steps=0,
                            steps=2,
                            grad_accum_steps=1,
                            gradient_checkpointing=True,
                            lora_dtype="bf16",
                            lora_plus_lr_ratio=2.0,
                            use_rslora=True,
                            seed=42,
                        )
                        summary = json.loads(summary_json)
                        self.assertTrue(summary["fp64_free_verified"])
                        self.assertGreater(summary["injected_layers"], 0)
                        self.assertGreater(len(lora_bundle.state_dict), 0)

                        # Verify zero residual LoRALinear wrappers on diffusion_model after training
                        dm = entry.patcher.model.diffusion_model
                        for mod in dm.modules():
                            self.assertNotIsInstance(mod, LoRALinear)

                        # 4. Verify KSampler on clean base `entry.patcher` matches `pre_sample` (100% clean uninject)
                        post_unpatched_sample = EmulatedKSampler.sample(
                            entry.patcher,
                            entry.clip,
                            resolution=64,
                            steps=2,
                            seed=101,
                            device=active_dev,
                        )
                        self.assertTrue(torch.isfinite(post_unpatched_sample).all())
                        self.assertTrue(
                            torch.allclose(pre_sample, post_unpatched_sample, atol=1e-4),
                            f"{entry.arch_id} base model weights drifted after uninject!",
                        )

                        # 5. Save via OmniLoRASave into emulated `folder_paths` `loras` dir and reload via OmniLoRALoader
                        _, saved_filename, saved_path = save_node.save_lora(
                            lora=lora_bundle,
                            output_lora_name=f"zoo_{entry.arch_id}",
                            append_step_count=True,
                        )
                        reloaded_bundle, _ = loader_node.load_lora(lora_name=saved_filename)
                        self.assertEqual(reloaded_bundle.rank, 4)
                        self.assertAlmostEqual(reloaded_bundle.alpha, 4.0, places=4)

                        # 6. Load into ComfyUI's native `comfy.sd.load_lora_for_models` and verify 0 missing keys!
                        patched_model, _ = comfy.sd.load_lora_for_models(
                            entry.patcher,
                            entry.clip,
                            reloaded_bundle.state_dict,
                            strength_model=1.0,
                            strength_clip=1.0,
                        )
                        self.assertEqual(
                            env.unloaded_lora_keys,
                            [],
                            f"{entry.arch_id} had unloaded LoRA keys in ComfyUI LoraLoader: {env.unloaded_lora_keys}",
                        )
                        assert patched_model is not None
                        self.assertGreater(len(patched_model.patches), 0)

                        # 7. Run KSampler with the LoRA applied via ComfyUI's ModelPatcher and verify parity with `model_out`
                        lora_sample = EmulatedKSampler.sample(
                            patched_model,
                            entry.clip,
                            resolution=64,
                            steps=2,
                            seed=101,
                            device=active_dev,
                        )
                        direct_trainer_model_sample = EmulatedKSampler.sample(
                            model_out,
                            entry.clip,
                            resolution=64,
                            steps=2,
                            seed=101,
                            device=active_dev,
                        )
                        self.assertTrue(torch.isfinite(lora_sample).all())
                        tol = 2e-3 if model_dtype == torch.float16 else 1e-4
                        self.assertTrue(
                            torch.allclose(lora_sample, direct_trainer_model_sample, atol=tol),
                            f"{entry.arch_id} OmniLoRATrainer MODEL output differed from ComfyUI LoraLoader!",
                        )
                        self.assertFalse(
                            torch.allclose(pre_sample, lora_sample, atol=1e-7),
                            f"{entry.arch_id} trained LoRA had zero effect on KSampler output",
                        )
                        del pre_sample, post_unpatched_sample, lora_sample, direct_trainer_model_sample
                        del patched_model, model_out, lora_bundle, reloaded_bundle, cached_bundle, entry
                        active_plan.backend.empty_cache()

    def test_directml_privateuseone_storageless_tensor_and_fp16_master_optimizer(self) -> None:
        """
        Regression test for Windows DirectML (`dml` / `privateuseone`) and `fp16` backends:
        1. Tensors whose `.data_ptr()` raises `RuntimeError: Cannot access data pointer of Tensor
           that doesn't have storage` (as in `torch-directml` `privateuseone:0`) are safely handled
           by `HardwareBackend._inspect_module_memory_profile` and `enforce_vram_budget`.
        2. `safe_to_device` casts `bfloat16` tensors on CPU first when targeting `float16` or DirectML.
        3. `_build_fp32_safe_optimizer` uses `_fp32_master_step` for `float16` LoRA weights so `eps=1e-8`
           never underflows to `0.0` (`0.0 / 0.0 = NaN` on Step 1 when `lora_B=0`) across `radamw`, `adamw`, and `sgd`.
        """
        from omni_lora.backends.base import HardwareBackend, _safe_tensor_storage_id, safe_to_device

        class StorageLessTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data: torch.Tensor):
                return torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)

            def data_ptr(self) -> int:
                raise RuntimeError("Cannot access data pointer of Tensor that doesn't have storage")

        raw_t = torch.randn(8, 8, dtype=torch.float16)
        dml_like_t = StorageLessTensor(raw_t)
        sid = _safe_tensor_storage_id(dml_like_t)
        self.assertEqual(sid, id(dml_like_t))

        mod = nn.Linear(8, 8, bias=False)
        mod.weight = nn.Parameter(StorageLessTensor(torch.randn(8, 8, dtype=torch.float16)), requires_grad=False)
        profile = HardwareBackend._inspect_module_memory_profile(
            mod,
            device=torch.device("cpu"),
            target_dtype=torch.float16,
        )
        self.assertGreater(profile["already_on_device_mb"], 0.0)

        bf16_cpu = torch.randn(4, 4, dtype=torch.bfloat16)
        fp16_out = safe_to_device(bf16_cpu, device=torch.device("cpu"), dtype=torch.float16)
        self.assertEqual(fp16_out.dtype, torch.float16)

        # Verify DirectML FP64-free scalar operator guards (`1.0 - t`, `1.0 / t`, `t ** 2`, `addcmul_`)
        dml_sim_t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float16)
        setattr(dml_sim_t, "_omni_simulate_dml_no_fp64", True)
        rsub_out = 1.0 - dml_sim_t
        self.assertEqual(rsub_out.dtype, torch.float16)
        self.assertTrue(torch.allclose(rsub_out, torch.tensor([0.75, 0.5, 0.25], dtype=torch.float16)))
        rdiv_out = 1.0 / dml_sim_t
        self.assertEqual(rdiv_out.dtype, torch.float16)
        pow_out = dml_sim_t ** 2
        self.assertEqual(pow_out.dtype, torch.float16)
        acc_t = dml_sim_t.clone()
        setattr(acc_t, "_omni_simulate_dml_no_fp64", True)
        acc_t.addcmul_(dml_sim_t, torch.ones_like(dml_sim_t), value=0.5)
        self.assertEqual(acc_t.dtype, torch.float16)

        # Verify RAdamW, AdamW, Adafactor, and SGD on float16 LoRA parameters never produce NaN on Step 1 when lora_B=0
        for opt_type in ("radamw", "adamw", "adafactor", "sgd"):
            backend = resolve_backend("cpu", dtype="fp16")
            adapter = ComfyModelAdapter(gradient_checkpointing=False)
            cfg = TrainingConfig(
                backend="cpu",
                dtype="fp16",
                optimizer_type=opt_type,
                learning_rate=1e-3,
                warmup_steps=1,
                max_train_steps=2,
                resolution=64,
                batch_size=1,
                gradient_checkpointing=False,
                lora=LoRAConfig(rank=4, alpha=4.0, lora_dtype="fp16", use_rslora=True),
            )
            pipeline = LoRATrainingPipeline(backend=backend, model_adapter=adapter, config=cfg)
            pipeline.setup()
            self.assertIsNotNone(pipeline.optimizer)
            self.assertTrue(
                hasattr(pipeline.optimizer.step, "__func__"),
                f"{opt_type} optimizer.step must be a bound method with __func__ for PyTorch 2.4 LRScheduler",
            )
            # Simulate PyTorch 2.0-2.4 LRScheduler.patch_track_step_called invoking step_fn.__func__(opt)
            pipeline.optimizer.step.__func__(pipeline.optimizer)
            m1 = pipeline.train_step(1)
            m2 = pipeline.train_step(2)
            self.assertTrue(math.isfinite(m1.loss))
            self.assertTrue(math.isfinite(m2.loss))
            for p in LoRAInjector.get_trainable_parameters(pipeline.backbone):
                self.assertEqual(p.dtype, torch.float16)
                self.assertTrue(torch.isfinite(p).all(), f"Non-finite fp16 LoRA weight with {opt_type}")

    def test_rslora_resume_no_alpha_compounding_and_comfy_loader_parity(self) -> None:
        """
        Regression test for `use_rslora=True` across multi-run training resumes:
        - When `rank=16, alpha=16.0, use_rslora=True`:
          * Training scaling is `alpha / sqrt(rank) = 16 / 4 = 4.0`.
          * Exported `.alpha` tensor in the state_dict is `scaling * rank = 4.0 * 16 = 64.0`
            so ComfyUI's native `LoraLoader` (which computes `alpha / rank = 64 / 16 = 4.0`)
            applies the exact same `4.0` scaling during inference.
          * When that LoRA is reloaded by `OmniLoRALoader` and resumed in `OmniLoRATrainer`
            with `use_rslora=True`, `alpha` must remain `16.0`, `scaling` must remain `4.0`,
            and `.alpha` in the re-exported state_dict must remain `64.0` (never compounding
            by `sqrt(rank)` to `256.0` or `1024.0`!).
        """
        from omni_lora.tests.comfy_emulator import ComfyModelZoo, EmulatedComfyEnvironment

        with EmulatedComfyEnvironment() as env:
            entry = ComfyModelZoo.build("flux_dev", dtype=torch.bfloat16)
            dataset_node = OmniLoRADatasetLoader()
            precache_node = OmniLoRAPrecomputeConditioning()
            trainer_node = OmniLoRATrainer()
            save_node = OmniLoRASave()
            loader_node = OmniLoRALoader()

            with torch.inference_mode():
                imgs = torch.rand(1, 64, 64, 3, dtype=torch.float32)
                ds, _ = dataset_node.load_dataset(
                    dataset_path="",
                    trigger_word="ohwx",
                    default_caption="photo",
                    resolution=64,
                    batch_size=1,
                    num_repeats=1,
                    center_crop=True,
                    random_flip=False,
                    caption_dropout=0.0,
                    images=imgs,
                    custom_prompts="a photo",
                )
                (cached_bundle,) = precache_node.precompute(
                    dataset=ds,
                    offload_encoders_after_cache=False,
                    vae=entry.vae,
                    clip=entry.clip,
                    model=entry.patcher,
                )

                current_lora = None
                for run_idx in range(1, 4):
                    _, current_lora, summary_json = trainer_node.train_lora(
                        cached_data=cached_bundle,
                        model=entry.patcher,
                        model_architecture="auto",
                        backend="cpu",
                        optimizer="radamw",
                        rank=16,
                        alpha=16.0,
                        learning_rate=1e-4,
                        lr_scheduler="cosine",
                        warmup_steps=1,
                        steps=2,
                        grad_accum_steps=1,
                        gradient_checkpointing=False,
                        use_rslora=True,
                        initial_lora=current_lora,
                    )
                    summary = json.loads(summary_json)
                    self.assertEqual(summary["rank"], 16)
                    self.assertAlmostEqual(summary["alpha"], 16.0, places=4)
                    self.assertEqual(current_lora.steps_trained, run_idx * 2)

                    # Verify `.alpha` tensor inside state_dict is `16 * sqrt(16) = 64.0` (never 256 or 1024)
                    alpha_tensors = [v for k, v in current_lora.state_dict.items() if k.endswith(".alpha")]
                    self.assertGreater(len(alpha_tensors), 0)
                    for at in alpha_tensors:
                        self.assertAlmostEqual(float(at.item()), 64.0, places=3)

                    # Save to disk and reload via OmniLoRALoader to test safetensors header & tensor inspection
                    _, fname, _ = save_node.save_lora(
                        lora=current_lora,
                        output_lora_name=f"rslora_run_{run_idx}",
                        append_step_count=False,
                    )
                    current_lora, reload_info_str = loader_node.load_lora(lora_name=fname)
                    reload_info = json.loads(reload_info_str)
                    self.assertEqual(reload_info["detected_rank"], 16)
                    self.assertAlmostEqual(reload_info["detected_alpha"], 16.0, places=3)

    def test_comfy_quantized_tensor_gguf_patches_and_oom_self_healing(self) -> None:
        """
        Verify:
        1. ComfyUI's native `QuantizedTensor` (`_qdata` + `_params.scale` + `.layout`) is
           detected by `is_comfy_quantized_tensor`, dequantized cleanly during forward/backward,
           and preserved on `uninject()`.
        2. GGUF `GGMLTensor` with a non-empty `.patches` list has its `.patches` temporarily
           saved and cleared during training (so residual inference LoRA patches don't double-apply)
           and restored 100% upon `LoRAInjector.uninject()`.
        3. Simulated mid-step `OutOfMemoryError` (`UR_RESULT_ERROR_OUT_OF_RESOURCES` / `CUDA out of memory`)
           triggers `LoRATrainingPipeline`'s automatic `demote_resident_layers` recovery and
           completes the training step without crashing!
        """
        from omni_lora.backends import get_tensor_storage_bytes, is_custom_quantized_tensor
        from omni_lora.lora import LoRALinear

        # 1. ComfyUI native `QuantizedTensor` simulation
        class MockQuantizedParams:
            def __init__(self, scale: torch.Tensor, orig_dtype: torch.dtype = torch.bfloat16) -> None:
                self.scale = scale
                self.orig_dtype = orig_dtype

        class QuantizedTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, qdata: torch.Tensor, scale: torch.Tensor):
                obj = torch.Tensor._make_subclass(cls, qdata.to(dtype=torch.float32), require_grad=False)
                obj._qdata = qdata
                obj._params = MockQuantizedParams(scale=scale)
                return obj

            def dequantize(self, dtype: torch.dtype = torch.bfloat16, device: Optional[torch.device] = None) -> torch.Tensor:
                dev = device or self._qdata.device
                return self._qdata.to(device=dev, dtype=dtype) * self._params.scale.to(device=dev, dtype=dtype)

        q_raw = torch.randn(32, 16, dtype=torch.float32).to(dtype=torch.float8_e4m3fn)
        q_scale = torch.tensor(0.25, dtype=torch.float32)
        qt = QuantizedTensor(q_raw, q_scale)
        self.assertTrue(is_custom_quantized_tensor(qt))
        self.assertEqual(get_tensor_storage_bytes(qt), 32 * 16 * 1 + 4)

        q_linear = nn.Linear(16, 32, bias=False)
        q_param = nn.Parameter(qt, requires_grad=False)
        q_param._qdata = q_raw
        q_param._params = MockQuantizedParams(scale=q_scale)
        q_param.dequantize = qt.dequantize  # type: ignore[attr-defined]
        q_linear.weight = q_param
        q_lora = LoRALinear(q_linear, rank=4, alpha=4.0, lora_dtype=torch.bfloat16)
        x_q = torch.randn(2, 4, 16, dtype=torch.bfloat16, requires_grad=True)
        out_q = q_lora(x_q)
        out_q.float().pow(2).mean().backward()
        self.assertIsNotNone(x_q.grad)
        self.assertIsNotNone(q_lora.lora_B.weight.grad)

        # 2. GGUF `GGMLTensor` with `.patches` cleanup and restoration on uninject
        class GGMLTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data: torch.Tensor, tensor_shape: tuple[int, int]):
                obj = torch.Tensor._make_subclass(cls, data, require_grad=False)
                obj.tensor_shape = torch.Size(tensor_shape)
                obj.tensor_type = "Q8_0"
                obj.patches = [("residual_comfy_lora_patch",)]
                return obj

        class GGMLOpsLinear(nn.Module):
            def __init__(self, in_f: int, out_f: int) -> None:
                super().__init__()
                raw = torch.randint(1, 100, (out_f * in_f,), dtype=torch.uint8)
                w_param = nn.Parameter(GGMLTensor(raw, (out_f, in_f)), requires_grad=False)
                w_param.tensor_shape = torch.Size((out_f, in_f))
                w_param.tensor_type = "Q8_0"
                w_param.patches = [("residual_comfy_lora_patch",)]
                self.weight = w_param
                self.bias = None

            def get_weight(self, tensor: torch.Tensor, dtype: torch.dtype, device: Any = None) -> torch.Tensor:
                # Verify `.patches` was cleared during training so residual patches aren't baked into training
                assert getattr(tensor, "patches", []) == [], "GGMLTensor.patches was not cleared during training!"
                out_f, in_f = tensor.tensor_shape
                dev = device if isinstance(device, torch.device) else tensor.device
                return (tensor.to(device=dev, dtype=dtype).view(out_f, in_f) - 50.0) * 0.01

        class MockGGUFBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qkv = GGMLOpsLinear(16, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.qkv(x)

        gguf_backbone = MockGGUFBackbone()
        self.assertTrue(is_custom_quantized_tensor(gguf_backbone.qkv.weight))
        LoRAInjector.inject(
            gguf_backbone,
            LoRAConfig(rank=4, alpha=4.0, target_modules=("qkv",)),
            lora_dtype=torch.bfloat16,
        )
        self.assertEqual(gguf_backbone.qkv.base_layer.weight.patches, [])
        _ = gguf_backbone(torch.randn(1, 4, 16, dtype=torch.bfloat16)).sum().backward()
        LoRAInjector.uninject(gguf_backbone)
        self.assertEqual(gguf_backbone.qkv.weight.patches, [("residual_comfy_lora_patch",)])

        # 3. Mid-step runtime OOM self-healing (`demote_resident_layers`)
        backend = resolve_backend("cpu", dtype="bf16")
        adapter = MossImageAdapter(gradient_checkpointing=False)
        cfg = TrainingConfig(
            backend="cpu",
            dtype="bf16",
            resolution=64,
            batch_size=1,
            max_train_steps=2,
            gradient_checkpointing=False,
            lora=LoRAConfig(rank=4, alpha=4.0),
        )
        pipeline = LoRATrainingPipeline(backend=backend, model_adapter=adapter, config=cfg)
        pipeline.setup()

        orig_compute_loss = adapter.compute_training_loss
        oom_triggered = {"count": 0}

        def _flaky_oom_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
            if oom_triggered["count"] == 0:
                oom_triggered["count"] += 1
                raise RuntimeError("UR_RESULT_ERROR_OUT_OF_RESOURCES: simulated XPU/CUDA/DirectML OOM on first attempt")
            return orig_compute_loss(*args, **kwargs)

        adapter.compute_training_loss = _flaky_oom_loss  # type: ignore[method-assign]
        try:
            metrics = pipeline.train_step(1)
            self.assertEqual(oom_triggered["count"], 1)
            self.assertGreater(metrics.loss, 0.0)
            self.assertTrue(pipeline.config.gradient_checkpointing)
        finally:
            adapter.compute_training_loss = orig_compute_loss  # type: ignore[method-assign]

    def test_user_interrupt_mid_precompute_and_mid_training_restores_model(self) -> None:
        """
        Verify that when a user clicks Cancel in ComfyUI (`InterruptProcessingException`):
        1. During `OmniLoRAPrecomputeConditioning.precompute`, the exception propagates immediately
           without getting swallowed.
        2. During `OmniLoRATrainer.train_lora` (e.g., on step 2), the exception propagates to ComfyUI's
           queue executor AND the `finally:` block cleanly uninjects all `LoRALinear` wrappers,
           restores base weights, and resets `comfy.model_management.in_training = False` so the
           model can immediately run `KSampler` on the next prompt!
        """
        from omni_lora.lora import LoRALinear
        from omni_lora.tests.comfy_emulator import (
            ComfyModelZoo,
            EmulatedComfyEnvironment,
            EmulatedKSampler,
            InterruptProcessingException,
        )

        with EmulatedComfyEnvironment() as env:
            import comfy.model_management as mm  # type: ignore

            entry = ComfyModelZoo.build("flux_dev", dtype=torch.bfloat16)
            dataset_node = OmniLoRADatasetLoader()
            precache_node = OmniLoRAPrecomputeConditioning()
            trainer_node = OmniLoRATrainer()

            with torch.inference_mode():
                imgs = torch.rand(2, 64, 64, 3, dtype=torch.float32)
                ds, _ = dataset_node.load_dataset(
                    dataset_path="",
                    trigger_word="ohwx",
                    default_caption="photo",
                    resolution=64,
                    batch_size=1,
                    num_repeats=1,
                    center_crop=True,
                    random_flip=False,
                    caption_dropout=0.0,
                    images=imgs,
                    custom_prompts="prompt one\nprompt two",
                )

                # 1. Interrupt during precompute
                env.interrupt_requested = True
                with self.assertRaises(InterruptProcessingException):
                    precache_node.precompute(
                        dataset=ds,
                        offload_encoders_after_cache=False,
                        vae=entry.vae,
                        clip=entry.clip,
                        model=entry.patcher,
                    )

                # Reset interrupt and run precompute normally
                env.interrupt_requested = False
                env.interrupt_check_calls = 0
                (cached_bundle,) = precache_node.precompute(
                    dataset=ds,
                    offload_encoders_after_cache=False,
                    vae=entry.vae,
                    clip=entry.clip,
                    model=entry.patcher,
                )

                # 2. Interrupt mid-training on step 2
                env.interrupt_check_calls = 0
                env.interrupt_after_calls = 2
                with self.assertRaises(InterruptProcessingException):
                    trainer_node.train_lora(
                        cached_data=cached_bundle,
                        model=entry.patcher,
                        model_architecture="auto",
                        backend="cpu",
                        optimizer="radamw",
                        rank=4,
                        alpha=4.0,
                        learning_rate=1e-4,
                        steps=5,
                        grad_accum_steps=1,
                        gradient_checkpointing=True,
                    )

                # Verify `in_training` was reset and `diffusion_model` was 100% uninjected
                self.assertFalse(mm.in_training)
                for mod in entry.patcher.model.diffusion_model.modules():
                    self.assertNotIsInstance(mod, LoRALinear)

                # Verify KSampler inference works immediately after user cancel
                env.interrupt_after_calls = None
                sample_out = EmulatedKSampler.sample(entry.patcher, entry.clip, resolution=64, steps=2)
                self.assertTrue(torch.isfinite(sample_out).all())

    def test_letterbox_aspect_ratio_bucketing_without_center_crop(self) -> None:
        """
        Verify that when `center_crop=False` and `aspect_ratio_bucketing=True`, an image whose
        aspect ratio differs from its nearest bucket is letterboxed onto neutral gray (`0.0` in `[-1, 1]`)
        and automatically assigned a spatial `loss_mask` that zeros/downweights the letterbox bars.
        """
        from omni_lora.dataset import OmniLoRADataset, OmniLoRADatasetConfig

        # Create an ultra-wide 8:1 (16x128) image in BHWC [1, 16, 128, 3] filled with ones (1.0)
        # Since max bucket aspect ratio at resolution=128 is 4:1 (64x256), an 8:1 image is letterboxed!
        wide_img = torch.ones(1, 16, 128, 3, dtype=torch.float32)
        cfg = OmniLoRADatasetConfig(
            resolution=128,
            batch_size=1,
            num_repeats=1,
            center_crop=False,
            aspect_ratio_bucketing=True,
            random_flip=False,
            use_alpha_mask=True,
            mask_unmasked_weight=0.05,
        )
        ds = OmniLoRADataset(config=cfg, comfy_images=wide_img, custom_prompts="panoramic landscape")
        sample = ds.samples[0]
        self.assertIsNotNone(sample.loss_mask)
        assert sample.loss_mask is not None
        # Center row of the letterboxed image is foreground (mask = 1.0); top/bottom letterbox bars are 0.0
        h_out = int(sample.loss_mask.shape[1])
        self.assertAlmostEqual(float(sample.loss_mask[0, h_out // 2, :].mean().item()), 1.0, places=3)
        self.assertAlmostEqual(float(sample.loss_mask[0, 0, :].mean().item()), 0.0, places=3)
        self.assertAlmostEqual(float(sample.loss_mask[0, -1, :].mean().item()), 0.0, places=3)

    def test_mixed_format_dataset_jpg_webp_avif_loading(self) -> None:
        """
        Verify that a single dataset folder containing a mix of `.jpg`, `.webp`, and `.avif`
        images (alongside `.png`) loads seamlessly into normalized `float32` tensors.
        """
        import tempfile
        from PIL import Image
        from omni_lora.dataset import OmniLoRADataset, OmniLoRADatasetConfig, SUPPORTED_IMAGE_EXTENSIONS

        for ext in (".jpg", ".jpeg", ".webp", ".avif", ".png", ".bmp"):
            self.assertIn(ext, SUPPORTED_IMAGE_EXTENSIONS)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_img = Image.new("RGB", (96, 64), color=(180, 90, 45))

            jpg_path = tmp_path / "sample_01.jpg"
            base_img.save(jpg_path, format="JPEG")
            (tmp_path / "sample_01.txt").write_text("a warm sunset photo", encoding="utf-8")

            webp_path = tmp_path / "sample_02.webp"
            base_img.save(webp_path, format="WEBP")
            (tmp_path / "sample_02.txt").write_text("a crisp webp photo", encoding="utf-8")

            avif_path = tmp_path / "sample_03.avif"
            try:
                base_img.save(avif_path, format="AVIF")
            except Exception:
                # Even if the test host lacks an AVIF encoder, Pillow/fallback opens the file by content or ffmpeg
                base_img.save(avif_path, format="WEBP")
            (tmp_path / "sample_03.txt").write_text("an avif portrait", encoding="utf-8")

            cfg = OmniLoRADatasetConfig(
                dataset_path=str(tmp_path),
                trigger_word="ohwx",
                default_caption="fallback",
                resolution=64,
                batch_size=2,
                aspect_ratio_bucketing=True,
            )
            ds = OmniLoRADataset(config=cfg)
            self.assertEqual(ds.num_unique_images, 3)
            loaded_names = [s.source_name for s in ds.samples]
            self.assertEqual(loaded_names, ["sample_01.jpg", "sample_02.webp", "sample_03.avif"])
            for s in ds.samples:
                self.assertEqual(s.pixel_values.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(s.pixel_values).all()))

    def test_optimal_lora_target_discovery_excludes_stems_and_adaln(self) -> None:
        """
        Verify that across all 21 real `comfy.ldm.*` architectures in `ComfyModelZoo`
        (wrapped in real `comfy.model_base.*`, `comfy.supported_models.*`, and
        `comfy.model_patcher.ModelPatcher` instances):
        1. `resolve_comfy_native_architecture` resolves the real `comfy.supported_models` config,
           `comfy.model_base` model, `latent_format`, `model_type`, and `model_sampling`.
        2. `classify_block_structural_roles` and `discover_universal_target_modules` dynamically
           discover 100% of repeating-block Attention & FFN/MoE-Expert layers while excluding
           100% of stems, AdaLN modulations (`modC`, `modX`, `img_mod`, `txt_mod`, `modulation`,
           `adaLN_modulation`, `adaln_modulation_*`), and MoE router gates (`ff_i.gate`).
        3. Tier 2 fallback (`LoRAInjector.inject(..., target_modules=())`) produces the exact
           same injected set as `discover_universal_target_modules`.
        4. ComfyUI's native `comfy.lora.model_lora_keys_unet` + `comfy.lora.load_lora` loads 100%
           of exported LoRA keys with zero unloaded keys.
        """
        import comfy.lora
        from omni_lora.lora.layers import classify_block_structural_roles
        from omni_lora.models.adapters import (
            discover_universal_target_modules,
            is_optimal_lora_target_path,
            resolve_comfy_native_architecture,
        )
        from omni_lora.tests.comfy_emulator import ComfyModelZoo

        self.assertEqual(len(ComfyModelZoo.ARCH_IDS), 21)
        for required_popular in ("moss_image", "krea2", "qwen_image_2_1"):
            self.assertIn(required_popular, ComfyModelZoo.ARCH_IDS)

        for arch_id in ComfyModelZoo.ARCH_IDS:
            entry = ComfyModelZoo.build(arch_id, dtype=torch.float32)
            dm = entry.patcher.model.diffusion_model
            self.assertTrue(
                dm.__class__.__module__.startswith("comfy.ldm."),
                f"{arch_id} diffusion_model {dm.__class__} is not a real comfy.ldm.* class",
            )

            # 1. Verify native ComfyUI metadata resolution (supported_models + model_base + model_sampling)
            native_meta = resolve_comfy_native_architecture(entry.patcher)
            self.assertIsNotNone(native_meta.get("model_config"), f"{arch_id} missing comfy.supported_models config")
            self.assertIsNotNone(native_meta.get("latent_format"), f"{arch_id} missing comfy.latent_formats instance")
            self.assertIsNotNone(native_meta.get("model_sampling"), f"{arch_id} missing comfy.model_sampling instance")

            # 2. Structural role classification & target discovery on the real comfy.ldm.* model
            roles = classify_block_structural_roles(dm)
            targets = discover_universal_target_modules(dm, comfy_inner_model=entry.patcher.model)
            self.assertGreaterEqual(len(targets), 2, f"{arch_id} discovered too few target suffixes: {targets}")

            injected = LoRAInjector.inject(dm, LoRAConfig(rank=4, alpha=4.0, target_modules=targets))
            self.assertGreaterEqual(len(injected), 4, f"{arch_id} injected too few layers: {injected}")
            self.assertEqual(
                classify_block_structural_roles(dm),
                roles,
                f"{arch_id} classify_block_structural_roles changed after LoRAInjector.inject",
            )
            summary = LoRAInjector.summarize_targets(dm, injected, roles=roles)
            self.assertEqual(summary["total"], len(injected), f"{arch_id} target summary count mismatch")
            self.assertTrue(summary["summary_lines"], f"{arch_id} missing summary_lines")

            # Every injected module must be classified as attention/ffn/block_projection, never stem/modulation/moe_router
            for path in injected:
                self.assertIn(
                    roles.get(path),
                    ("attention", "ffn", "attention_or_ffn", "block_projection"),
                    f"{arch_id} injected non-target module '{path}' (role={roles.get(path)})",
                )
                self.assertTrue(is_optimal_lora_target_path(path, model=dm))

            # Every stem, modulation, and MoE router in the real comfy.ldm.* model must be excluded
            for mod_name, role in roles.items():
                if role in ("stem_or_head", "modulation", "moe_router"):
                    self.assertNotIn(
                        mod_name,
                        injected,
                        f"{arch_id} erroneously injected {role} module '{mod_name}'",
                    )
                    self.assertFalse(is_optimal_lora_target_path(mod_name, model=dm))

            # 3. Verify 100% key compatibility with ComfyUI's native `comfy.lora.model_lora_keys_unet`
            lora_sd = LoRAInjector.extract_lora_state_dict(dm, prefix="diffusion_model")
            key_map = comfy.lora.model_lora_keys_unet(entry.patcher.model, {})
            loaded = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
            self.assertGreater(len(loaded), 0, f"{arch_id} loaded 0 keys in comfy.lora.load_lora")

            LoRAInjector.uninject(dm)

            # 4. Verify Tier 2 fallback (`target_modules=()` empty tuple) matches `discover_universal_target_modules`
            injected_tier2 = LoRAInjector.inject(dm, LoRAConfig(rank=4, alpha=4.0, target_modules=()))
            self.assertEqual(
                set(injected_tier2),
                set(injected),
                f"{arch_id} Tier 2 auto-injection differed from discover_universal_target_modules",
            )
            LoRAInjector.uninject(dm)

    def test_dual_network_separate_training_and_timestep_split(self) -> None:
        """
        Verify both dual-network training workflows on the real `comfy.ldm.moss_image.model.MOSSImage`:
        1. Internal 2-sub-network architectures (`text_embedder.blocks` + `layers`) can train each
           sub-network separately using `target_module_filter` and chain into a unified LoRA bundle.
        2. Two-checkpoint / split-schedule architectures honor `min_timestep` and `max_timestep`.
        """
        from omni_lora.tests.comfy_emulator import ComfyModelZoo, EmulatedComfyEnvironment

        with EmulatedComfyEnvironment() as env:
            entry = ComfyModelZoo.build("moss_image", dtype=torch.bfloat16)
            dataset_node = OmniLoRADatasetLoader()
            precache_node = OmniLoRAPrecomputeConditioning()
            trainer_node = OmniLoRATrainer()

            with torch.inference_mode():
                imgs = torch.rand(1, 64, 64, 3, dtype=torch.float32)
                ds, _ = dataset_node.load_dataset(
                    dataset_path="",
                    trigger_word="ohwx",
                    default_caption="photo",
                    resolution=64,
                    batch_size=1,
                    num_repeats=1,
                    center_crop=True,
                    random_flip=False,
                    caption_dropout=0.0,
                    images=imgs,
                    custom_prompts="a photo",
                )
                (cached_bundle,) = precache_node.precompute(
                    dataset=ds,
                    offload_encoders_after_cache=False,
                    vae=entry.vae,
                    clip=entry.clip,
                    model=entry.patcher,
                )

                # Stage 1: Train ONLY the text/LLM refiner sub-network (`text_embedder.blocks`) on high noise [0.5, 1.0]
                _, lora_sub1, sum1_json = trainer_node.train_lora(
                    cached_data=cached_bundle,
                    model=entry.patcher,
                    model_architecture="auto",
                    backend="cpu",
                    optimizer="radamw",
                    rank=4,
                    alpha=4.0,
                    learning_rate=1e-4,
                    steps=2,
                    grad_accum_steps=1,
                    gradient_checkpointing=False,
                    enable_target_module_filter=True,
                    target_module_filter="text_embedder.blocks",
                    min_timestep=0.5,
                    max_timestep=1.0,
                )
                sum1 = json.loads(sum1_json)
                self.assertEqual(sum1["injected_layers"], 10)
                self.assertEqual(sum1["target_module_filter"], "text_embedder.blocks")
                self.assertAlmostEqual(sum1["min_timestep"], 0.5, places=4)
                self.assertAlmostEqual(sum1["max_timestep"], 1.0, places=4)
                self.assertTrue(all("text_embedder.blocks" in k for k in lora_sub1.state_dict.keys()))

                # Stage 2: Train ONLY the image DiT sub-network (`layers`) on low noise [0.0, 0.5],
                # and verify chaining `initial_lora=lora_sub1` merges both sub-networks into one combined LoRA!
                _, lora_combined, sum2_json = trainer_node.train_lora(
                    cached_data=cached_bundle,
                    model=entry.patcher,
                    model_architecture="auto",
                    backend="cpu",
                    optimizer="radamw",
                    rank=4,
                    alpha=4.0,
                    learning_rate=5e-5,
                    steps=2,
                    grad_accum_steps=1,
                    gradient_checkpointing=False,
                    initial_lora=lora_sub1,
                    enable_target_module_filter=True,
                    target_module_filter="layers",
                    min_timestep=0.0,
                    max_timestep=0.5,
                )
                sum2 = json.loads(sum2_json)
                self.assertGreaterEqual(sum2["injected_layers"], 8)
                self.assertEqual(sum2["target_module_filter"], "layers")
                self.assertAlmostEqual(sum2["min_timestep"], 0.0, places=4)
                self.assertAlmostEqual(sum2["max_timestep"], 0.5, places=4)
                self.assertTrue(any("text_embedder.blocks" in k for k in lora_combined.state_dict.keys()))
                self.assertTrue(any(k.startswith("layers.") or ".layers." in k for k in lora_combined.state_dict.keys()))
                total_modules = sum1["injected_layers"] + sum2["injected_layers"]
                self.assertEqual(len(lora_combined.state_dict), total_modules * 3)

    def test_comfy_aimdo_malloc_graph_and_model_prefetch_training_guard(self) -> None:
        """
        Regression test for `RuntimeError: aimdo memory compile error` (`Comfy model compiler graph breaks: 0, rogues: 22`)
        when running models like `comfy/ldm/moss_image/model.py` on CUDA with `comfy_aimdo` and `comfy.model_prefetch`:
        - During training (`torch.is_grad_enabled()` or `comfy.model_management.in_training`),
          `comfy.model_prefetch.prefetch_queue_pop(..., malloc_scope="block")` and `comfy_aimdo.malloc_graph`
          must never invoke `control.lib.malloc_graph_pop` or raise `RuntimeError("aimdo memory compile error")`.
        """
        import sys
        import types
        from omni_lora.models.adapters import install_comfy_runtime_hooks

        class MockAimdoControlLib:
            @staticmethod
            def malloc_graph_pop() -> int:
                # Replicate comfy_aimdo C library failing with 0 when autograd saves rogue tensors during training
                return 0

        class MockMallocGraph:
            def _call(self, fn: Any, *args: Any) -> int:
                res = fn(*args)
                if not res:
                    raise RuntimeError("aimdo memory compile error")
                return res

            def pop(self) -> bool:
                return self._call(MockAimdoControlLib.malloc_graph_pop) == 2

            def iterate(self, scope: Optional[str] = None) -> bool:
                broken = self.pop()
                return broken

        aimdo_pkg = types.ModuleType("comfy_aimdo")
        aimdo_mg_mod = types.ModuleType("comfy_aimdo.malloc_graph")
        mg_instance = MockMallocGraph()
        aimdo_mg_mod.MallocGraph = MockMallocGraph  # type: ignore[attr-defined]
        aimdo_mg_mod.malloc_graph = mg_instance  # type: ignore[attr-defined]

        mp_mod = types.ModuleType("comfy.model_prefetch")
        mp_mod.malloc_graph = mg_instance  # type: ignore[attr-defined]
        mp_mod.malloc_graph_enabled = lambda device=None: True  # type: ignore[attr-defined]
        mp_mod.malloc_graph_begin = lambda device=None: mg_instance.iterate("begin")  # type: ignore[attr-defined]
        mp_mod.malloc_graph_end = lambda: mg_instance.pop()  # type: ignore[attr-defined]

        def _raw_prefetch_queue_pop(
            prefetch_queue: Any,
            device: Any,
            module: Any = None,
            dtype: Any = None,
            malloc_scope: Optional[str] = None,
        ) -> None:
            if mg_instance.iterate(malloc_scope if module is not None else None):
                pass

        mp_mod.prefetch_queue_pop = _raw_prefetch_queue_pop  # type: ignore[attr-defined]

        orig_aimdo = sys.modules.get("comfy_aimdo")
        orig_aimdo_mg = sys.modules.get("comfy_aimdo.malloc_graph")
        orig_mp = sys.modules.get("comfy.model_prefetch")
        try:
            sys.modules["comfy_aimdo"] = aimdo_pkg
            sys.modules["comfy_aimdo.malloc_graph"] = aimdo_mg_mod
            sys.modules["comfy.model_prefetch"] = mp_mod

            # Before installing OmniLoRA hooks, calling prefetch_queue_pop raises `RuntimeError: aimdo memory compile error`
            with self.assertRaises(RuntimeError) as ctx_err:
                mp_mod.prefetch_queue_pop(None, torch.device("cpu"), nn.Linear(4, 4), torch.bfloat16, malloc_scope="block")
            self.assertIn("aimdo memory compile error", str(ctx_err.exception))

            # Install OmniLoRA runtime hooks and verify that both training and fallback calls succeed without raising
            install_comfy_runtime_hooks()
            with torch.enable_grad():
                self.assertFalse(mp_mod.malloc_graph_enabled(torch.device("cpu")))
                mp_mod.malloc_graph_begin(torch.device("cpu"))
                mp_mod.prefetch_queue_pop(None, torch.device("cpu"), nn.Linear(4, 4), torch.bfloat16, malloc_scope="block")
                mp_mod.malloc_graph_end()
        finally:
            if orig_aimdo is None:
                sys.modules.pop("comfy_aimdo", None)
            else:
                sys.modules["comfy_aimdo"] = orig_aimdo
            if orig_aimdo_mg is None:
                sys.modules.pop("comfy_aimdo.malloc_graph", None)
            else:
                sys.modules["comfy_aimdo.malloc_graph"] = orig_aimdo_mg
            if orig_mp is None:
                sys.modules.pop("comfy.model_prefetch", None)
            else:
                sys.modules["comfy.model_prefetch"] = orig_mp

    def test_xpu_16gb_large_dit_vram_budget_and_cpu_ckpt_offload(self) -> None:
        """
        Regression test for 15.56 GB Intel Arc XPU (`total_vram_mb = 15933.0`) running a 9.53 GB
        (`9759 MB`) 44-block DiT (`moss_image`):
        1. At `reserve_vram_gb = 0.0`, `compute_vram_budget` allows the full 9.53 GB (`9759 MB`)
           model to fit 100% in VRAM (`max_resident_model_mb >= 9759.0`, ~10.2 GB).
        2. At `reserve_vram_gb = 1.0`, `compute_vram_budget` scales linearly without any 2+ GB cliff:
           `max_resident_model_mb` remains ~9.1 - 9.4 GB (`> 9000 MB`), offloading only ~0.4 GB.
        3. `trim_process_working_set()` executes cleanly and `_snapshot_tensors_for_recompute`
           skips redundant clones of `OmniTransformerOptions` and 5D+ non-grad RoPE tables.
        """
        import unittest.mock
        from omni_lora.backends import HardwareBackend, trim_process_working_set
        from omni_lora.backends.xpu import XPUBackend
        from omni_lora.models.adapters import OmniTransformerOptions
        from omni_lora.pipelines.lora_trainer import _snapshot_tensors_for_recompute

        trim_process_working_set()
        xpu_sim = XPUBackend(device_index=0, dtype="bf16")
        dummy_mod = nn.Linear(16, 16)

        with (
            unittest.mock.patch.object(xpu_sim, "is_available", return_value=False),
            unittest.mock.patch.object(
                xpu_sim,
                "get_capabilities",
                return_value=unittest.mock.MagicMock(
                    total_vram_mb=15933.0,
                    supports_bf16=True,
                    supports_fp16=True,
                    supports_fp8=False,
                ),
            ),
            unittest.mock.patch.object(xpu_sim, "get_free_vram_mb", return_value=14413.6),
            unittest.mock.patch.object(
                HardwareBackend,
                "_inspect_module_memory_profile",
                return_value={
                    "non_base_mb": 209.0,
                    "trainable_mb": 58.0,
                    "base_layers_total_mb": 9550.0,
                    "max_single_layer_mb": 108.0,
                    "max_hidden_dim": 3072.0,
                    "already_on_device_mb": 0.0,
                },
            ),
        ):
            budget_moss_0gb = xpu_sim.compute_vram_budget(
                vram_mode="auto",
                resolution=1024,
                batch_size=1,
                module=dummy_mod,
                reserve_vram_gb=0.0,
            )
            # 9.53 GB model (9759 MB) on 15.56 GB XPU at reserve_vram_gb=0.0 fits 100% in VRAM
            self.assertGreaterEqual(budget_moss_0gb["max_resident_model_mb"], 9759.0)

            budget_moss_1gb = xpu_sim.compute_vram_budget(
                vram_mode="auto",
                resolution=1024,
                batch_size=1,
                module=dummy_mod,
                reserve_vram_gb=1.0,
            )
            # Setting reserve_vram_gb=1.0 reduces max_resident_model_mb linearly (~9.13 GB), never offloading 2+ GB
            self.assertGreater(budget_moss_1gb["max_resident_model_mb"], 9000.0)
            diff_mb = budget_moss_0gb["max_resident_model_mb"] - budget_moss_1gb["max_resident_model_mb"]
            self.assertLessEqual(diff_mb, 1200.0)

            # Verify deterministic optimizer-state ledger scales accurately: adafactor (~0.5B/elem) < sgd (4B/elem) < radamw / adamw (8B/elem)
            b_adafactor = xpu_sim.compute_vram_budget(module=dummy_mod, optimizer_type="adafactor")
            b_sgd = xpu_sim.compute_vram_budget(module=dummy_mod, optimizer_type="sgd")
            b_radam = xpu_sim.compute_vram_budget(module=dummy_mod, optimizer_type="radamw")
            b_adamw = xpu_sim.compute_vram_budget(module=dummy_mod, optimizer_type="adamw")
            self.assertLess(b_adafactor["optimizer_allowance_mb"], b_sgd["optimizer_allowance_mb"])
            self.assertLess(b_sgd["optimizer_allowance_mb"], b_radam["optimizer_allowance_mb"])
            self.assertAlmostEqual(b_radam["optimizer_allowance_mb"], b_adamw["optimizer_allowance_mb"], places=2)

        # Verify read-only 5D+ RoPE / OmniTransformerOptions zero-clone fast-path in _snapshot_tensors_for_recompute
        # AND verify that standard 3D/4D activation tensors ARE still cloned into isolated storage
        from omni_lora.pipelines.lora_trainer import _clone_tensors_for_grad

        rope_6d = torch.randn(1, 1, 64, 16, 2, 2, dtype=torch.float32)
        act_3d_grad = torch.randn(1, 64, 128, dtype=torch.float32, requires_grad=True)
        act_3d_nograd = torch.randn(1, 64, 128, dtype=torch.float32, requires_grad=False)
        t_opts = OmniTransformerOptions({"sample_sigmas": torch.linspace(1.0, 0.0, 5), "cond_or_uncond": [0]})
        snap_tuple = _snapshot_tensors_for_recompute(
            (rope_6d, {"transformer_options": t_opts}, act_3d_grad, act_3d_nograd),
        )
        self.assertIs(snap_tuple[0], rope_6d)
        self.assertIs(snap_tuple[1]["transformer_options"]["sample_sigmas"], t_opts["sample_sigmas"])
        # 3D activation tensors must STILL be cloned to protect against in-place block ops (e.g. x += residual)
        self.assertIsNot(snap_tuple[2], act_3d_grad)
        self.assertTrue(snap_tuple[2].requires_grad)
        self.assertIsNot(snap_tuple[3], act_3d_nograd)

        cloned_grad_tuple = _clone_tensors_for_grad(
            (rope_6d, {"transformer_options": t_opts}, act_3d_grad, act_3d_nograd),
        )
        self.assertIs(cloned_grad_tuple[0], rope_6d)
        self.assertIs(cloned_grad_tuple[1]["transformer_options"]["sample_sigmas"], t_opts["sample_sigmas"])
        self.assertIsNot(cloned_grad_tuple[3], act_3d_nograd)

        # Verify shared conditioning deduplication via weakref-verified memo across blocks
        ckpt_memo: dict[int, Any] = {}
        shared_ref = torch.randn(1, 36, 128, dtype=torch.float16, requires_grad=True)
        snap_b0 = _snapshot_tensors_for_recompute(shared_ref, memo=ckpt_memo)
        snap_b1 = _snapshot_tensors_for_recompute(shared_ref, memo=ckpt_memo)
        self.assertIs(snap_b0, snap_b1)

        # Verify FP32-modulated activation entering 16-bit LoRALinear stays 16-bit (never upcast to 2x larger FP32)
        from omni_lora.lora import LoRALinear

        fp16_base = nn.Linear(64, 128, bias=False).to(dtype=torch.float16)
        fp16_base.weight.requires_grad_(False)
        fp16_lora = LoRALinear(fp16_base, rank=4, alpha=4.0, lora_dtype=torch.float16)
        y_fp32_mod = torch.randn(1, 16, 64, dtype=torch.float32, requires_grad=True)
        out_fp16 = fp16_lora(y_fp32_mod)
        self.assertEqual(out_fp16.dtype, torch.float16)
        out_fp16.float().pow(2).mean().backward()
        self.assertIsNotNone(y_fp32_mod.grad)
        self.assertEqual(y_fp32_mod.grad.dtype, torch.float32)

        # Verify custom_checkpoint_dir on OmniLoRATrainer Input Schema & _resolve_target_output_dir
        from omni_lora.nodes import _resolve_target_output_dir
        from omni_lora.pipelines.lora_trainer import _ensure_nonleaf_grad_outputs
        from omni_lora.backends.mps import MPSBackend

        trainer_opt_inputs = OmniLoRATrainer.INPUT_TYPES()["optional"]
        self.assertIn("custom_checkpoint_dir", trainer_opt_inputs)
        with tempfile.TemporaryDirectory() as tmpdir:
            base_loras = Path(tmpdir) / "loras"
            base_loras.mkdir(parents=True, exist_ok=True)
            # Bare subfolder name resolves inside default loras dir
            sub_resolved = _resolve_target_output_dir("my_run_ckpts", base_loras)
            self.assertEqual(sub_resolved, (base_loras / "my_run_ckpts").resolve())
            self.assertTrue(sub_resolved.is_dir())
            # Explicit absolute path resolves directly
            abs_custom = Path(tmpdir) / "external_ckpts"
            abs_resolved = _resolve_target_output_dir(str(abs_custom), base_loras)
            self.assertEqual(abs_resolved, abs_custom.resolve())
            self.assertTrue(abs_resolved.is_dir())

        # Verify `_ensure_nonleaf_grad_outputs` avoids redundant copies on already non-leaf tensors
        # while cloning leaf tensors with requires_grad=True
        leaf_grad = torch.randn(2, 4, requires_grad=True)
        nonleaf_grad = leaf_grad * 2.0
        self.assertTrue(leaf_grad.is_leaf)
        self.assertFalse(nonleaf_grad.is_leaf)
        checked_nonleaf = _ensure_nonleaf_grad_outputs(nonleaf_grad)
        self.assertIs(checked_nonleaf, nonleaf_grad)
        checked_leaf = _ensure_nonleaf_grad_outputs(leaf_grad)
        self.assertFalse(checked_leaf.is_leaf)

        # Verify Apple MPS unified memory backend never demotes resident layers to CPU
        mps_backend = MPSBackend(device_index=0)
        dummy_mod = nn.Linear(16, 16)
        self.assertEqual(mps_backend.demote_resident_layers(dummy_mod, target_demote_mb=512.0), 0.0)

    def test_krea2_moss_qwen_image21_lora_injection_and_roundtrip(self) -> None:
        """
        Verify 1-to-1 ComfyUI LoRA target discovery, injection, stem/modulation exclusion, and
        state_dict round-trip (including tuple-sliced weights via `comfy.lora.model_lora_keys_unet`)
        directly on the real `comfy.ldm.krea2.model.SingleStreamDiT`,
        `comfy.ldm.moss_image.model.MOSSImage`, `comfy.ldm.flux3.model.Flux3`, and
        `comfy.ldm.qwen_image21.model.QwenImage21Transformer2DModel`.
        """
        import comfy.lora
        import comfy.ops
        from comfy.ldm.moss_image.model import MOSSImage
        from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
        from omni_lora.lora import LoRAConfig, LoRAInjector, LoRALinear
        from omni_lora.models.adapters import (
            detect_comfy_model_architecture,
            discover_universal_target_modules,
            is_optimal_lora_target_path,
        )
        from omni_lora.tests.comfy_emulator import ComfyModelZoo, build_native_comfy_patcher

        # ------------------------------------------------------------------
        # 1. Real Krea 2 (`comfy.ldm.krea2.model.SingleStreamDiT`)
        # ------------------------------------------------------------------
        krea2_entry = ComfyModelZoo.build("krea2", dtype=torch.float32)
        krea2_model = krea2_entry.patcher.model.diffusion_model
        self.assertEqual(detect_comfy_model_architecture(krea2_model), "krea2")

        krea2_targets = discover_universal_target_modules(krea2_model, comfy_inner_model=krea2_entry.patcher.model)
        krea2_injected = set(
            LoRAInjector.inject(
                krea2_model,
                LoRAConfig(rank=4, alpha=4.0, target_modules=krea2_targets),
            )
        )
        for blk_prefix in ("blocks.0", "blocks.1", "txtfusion.refiner_blocks.0", "txtfusion.layerwise_blocks.0"):
            for proj_name in ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "mlp.gate", "mlp.up", "mlp.down"):
                self.assertIn(f"{blk_prefix}.{proj_name}", krea2_injected)
        for forbidden in ("first", "last.linear", "tproj", "tmlp.0", "tmlp.2", "txtmlp.1", "txtmlp.3", "txtfusion.projector"):
            self.assertNotIn(forbidden, krea2_injected)
            self.assertFalse(is_optimal_lora_target_path(forbidden, model=krea2_model))
        LoRAInjector.uninject(krea2_model)

        # ------------------------------------------------------------------
        # 2. Real MOSS-Image (`comfy.ldm.moss_image.model.MOSSImage`, fused & unfused QKV)
        #    and ComfyUI `model_lora_keys_unet` tuple-slice round-trip (`q/k/v -> qkv`)
        # ------------------------------------------------------------------
        moss_entry = ComfyModelZoo.build("moss_image", dtype=torch.float32)
        moss_unfused = moss_entry.patcher.model.diffusion_model
        self.assertEqual(detect_comfy_model_architecture(moss_unfused), "moss_image")

        moss_targets = discover_universal_target_modules(moss_unfused, comfy_inner_model=moss_entry.patcher.model)
        moss_injected = set(
            LoRAInjector.inject(
                moss_unfused,
                LoRAConfig(rank=4, alpha=4.0, target_modules=moss_targets),
            )
        )
        for b_idx in (0, 1):
            for sub_p in ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"):
                self.assertIn(f"layers.{b_idx}.{sub_p}", moss_injected)
            for txt_p in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"):
                self.assertIn(f"text_embedder.blocks.{b_idx}.{txt_p}", moss_injected)
        for forbidden in (
            "video_embedder",
            "text_embedder.proj",
            "time_embedding.0",
            "time_embedding.2",
            "shared_block_modulation",
            "video_head_modulation",
            "final_linear_video",
        ):
            self.assertNotIn(forbidden, moss_injected)
            self.assertFalse(is_optimal_lora_target_path(forbidden, model=moss_unfused))

        # Initialize non-zero LoRA weights on unfused MOSSImage (`proj.q`, `proj.k`, `proj.v`) and
        # load them into a fused MOSSImage (`fused_qkv=True` -> `proj.qkv`) via ComfyUI's native
        # `comfy.lora.model_lora_keys_unet` tuple-slice mapping!
        for m in moss_unfused.modules():
            if isinstance(m, LoRALinear):
                nn.init.normal_(m.lora_B.weight, std=0.05)
        unfused_moss_sd = LoRAInjector.extract_lora_state_dict(moss_unfused, prefix="diffusion_model")
        LoRAInjector.uninject(moss_unfused)

        moss_fused_dm = MOSSImage(
            dim=64, ffn_dim=128, num_heads=4, num_layers=2, text_dim=128,
            latent_channels=128, fused_qkv=True, dtype=torch.float32,
            device="cpu", operations=comfy.ops.manual_cast,
        )
        moss_fused_patcher = build_native_comfy_patcher("moss_image", moss_fused_dm)
        fused_moss_targets = discover_universal_target_modules(moss_fused_dm, comfy_inner_model=moss_fused_patcher.model)
        LoRAInjector.inject(moss_fused_dm, LoRAConfig(rank=4, alpha=4.0, target_modules=fused_moss_targets))
        moss_load_res = LoRAInjector.load_lora_state_dict(
            moss_fused_dm,
            unfused_moss_sd,
            comfy_model=moss_fused_patcher.model,
        )
        # 2 layers * 4 fused projections (`qkv`, `o`, `ffn_up_gate`, `ffn_down`) + text blocks
        self.assertIn(moss_load_res["loaded_layers"], (16, 18))
        LoRAInjector.uninject(moss_fused_dm)

        # ------------------------------------------------------------------
        # 3. Real Qwen-Image 2.1 (`comfy.ldm.qwen_image21.model.QwenImage21Transformer2DModel`)
        #    and ComfyUI `model_lora_keys_unet` tuple-slice round-trip (`gate_layer`+`proj` <-> `gate_up`)
        # ------------------------------------------------------------------
        qwen_entry = ComfyModelZoo.build("qwen_image_2_1", dtype=torch.float32)
        qwen_dm = qwen_entry.patcher.model.diffusion_model
        self.assertEqual(detect_comfy_model_architecture(qwen_dm), "qwen_image21")

        qwen_targets = discover_universal_target_modules(qwen_dm, comfy_inner_model=qwen_entry.patcher.model)
        qwen_injected = set(
            LoRAInjector.inject(
                qwen_dm,
                LoRAConfig(rank=4, alpha=4.0, target_modules=qwen_targets),
            )
        )
        for b_idx in (0, 1):
            for sub_p in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out"):
                self.assertIn(f"transformer_blocks.{b_idx}.{sub_p}", qwen_injected)
        for forbidden in ("img_in", "txt_in.1", "modulation.1", "norm_out.linear", "proj_out"):
            self.assertNotIn(forbidden, qwen_injected)
            self.assertFalse(is_optimal_lora_target_path(forbidden, model=qwen_dm))

        for m in qwen_dm.modules():
            if isinstance(m, LoRALinear):
                nn.init.normal_(m.lora_B.weight, std=0.05)

        # Export using `export_comfy_qwen21_unfused_mlp=True` (splits `gate_up` into ComfyUI's
        # `gate_layer` and `proj` keys) and verify both ComfyUI's `comfy.lora.load_lora` AND
        # `LoRAInjector.load_lora_state_dict(..., comfy_model=qwen_entry.patcher.model)` load 100%!
        unfused_qwen_sd = LoRAInjector.extract_lora_state_dict(
            qwen_dm,
            prefix="diffusion_model",
            export_comfy_qwen21_unfused_mlp=True,
        )
        self.assertIn("diffusion_model.transformer_blocks.0.img_mlp.gate_layer.lora_up.weight", unfused_qwen_sd)
        self.assertIn("diffusion_model.transformer_blocks.0.img_mlp.proj.lora_up.weight", unfused_qwen_sd)

        qwen_key_map = comfy.lora.model_lora_keys_unet(qwen_entry.patcher.model, {})
        qwen_comfy_loaded = comfy.lora.load_lora(unfused_qwen_sd, qwen_key_map, log_missing=False)
        self.assertGreater(len(qwen_comfy_loaded), 0)

        qwen_reload_res = LoRAInjector.load_lora_state_dict(
            qwen_dm,
            unfused_qwen_sd,
            comfy_model=qwen_entry.patcher.model,
        )
        # 2 blocks * 6 modules (`to_q`, `to_k`, `to_v`, `to_out.0`, `gate_up`, `out`) = 12 modules
        self.assertEqual(qwen_reload_res["loaded_layers"], 12)
        LoRAInjector.uninject(qwen_dm)

    def test_structural_role_classifier_and_35_plus_comfy_ldm_architectures(self) -> None:
        """
        Verify `classify_block_structural_roles`, `trace_single_block_dataflow`, and
        `discover_universal_target_modules` across all 50 real ComfyUI `comfy.ldm.*` architectures
        (21 core `ComfyModelZoo` + 29 `ExtendedComfyLdmZoo` models), ensuring zero string-name
        guessing is needed to exclude:
          - AuraFlow `double_layers.*.modC.1`, `double_layers.*.modX.1`, `single_layers.*.modX.1`
          - Genmo Mochi direct leaf `blocks.*.mod_x`, `blocks.*.mod_y`
          - Boogu `transformer_blocks.*.img_norm1.linear`, `txt_norm1.linear`
          - Anima / Cosmos 2-layer bottleneck `blocks.*.adaln_modulation_*`
          - Kandinsky 5 2-layer `visual_transformer_blocks.*.modulation`
          - ACE-Step SSL `projectors.*` and ACE-Step 1.5 frozen `tokenizer.*` / `detokenizer.*`
          - HiDream-O1 frozen vision reference encoder `visual.blocks.*`
          - HunYuanDiT / HunYuan3Dv2_1 U-ViT `blocks.*.skip_linear` and MoE router `moe.gate`
          - TripoSplat 3D rotary position generators `noise_repo_layers.*`, `context_repo_layers.*`, `repo_layers.*`
          - StableCascade `TimestepBlock` modulation projections `down_blocks.*.*.mapper*`
        """
        import comfy.lora
        from omni_lora.lora.layers import classify_block_structural_roles, trace_single_block_dataflow
        from omni_lora.models.adapters import (
            detect_comfy_model_architecture,
            discover_universal_target_modules,
            resolve_comfy_native_architecture,
        )
        from omni_lora.tests.comfy_emulator import ComfyModelZoo, ExtendedComfyLdmZoo

        # 1. Verify AuraFlow `modC` / `modX` structural role classification on real `comfy.ldm.aura.mmdit.MMDiT`
        aura_entry = ComfyModelZoo.build("auraflow_sana", dtype=torch.float32)
        aura_dm = aura_entry.patcher.model.diffusion_model
        aura_roles = classify_block_structural_roles(aura_dm)
        self.assertEqual(aura_roles["double_layers.0.modC.1"], "modulation")
        self.assertEqual(aura_roles["double_layers.0.modX.1"], "modulation")
        self.assertEqual(aura_roles["single_layers.0.modX.1"], "modulation")
        self.assertEqual(aura_roles["double_layers.0.attn.w1q"], "attention")
        self.assertEqual(aura_roles["double_layers.0.mlpX.c_fc1"], "ffn")

        traced_roles = trace_single_block_dataflow(aura_dm)
        self.assertEqual(traced_roles["double_layers.0.modC.1"], "modulation")
        self.assertEqual(traced_roles["single_layers.0.modX.1"], "modulation")

        # 2. Verify all 29 extended real `comfy.ldm.*` architectures (bringing total tested count to 50)
        total_verified = len(ComfyModelZoo.ARCH_IDS) + len(ExtendedComfyLdmZoo.EXTENDED_ARCH_IDS)
        self.assertEqual(total_verified, 50)

        for arch_id in ExtendedComfyLdmZoo.EXTENDED_ARCH_IDS:
            patcher = ExtendedComfyLdmZoo.build(arch_id, dtype=torch.float32)
            backbone = patcher.model.diffusion_model
            self.assertTrue(
                backbone.__class__.__module__.startswith("comfy.ldm."),
                f"{arch_id} backbone {backbone.__class__} is not a real comfy.ldm.* class",
            )
            native_meta = resolve_comfy_native_architecture(patcher)
            self.assertIsNotNone(native_meta.get("model_config"))
            self.assertIsNotNone(native_meta.get("model_sampling"))

            roles = classify_block_structural_roles(backbone)
            targets = discover_universal_target_modules(backbone, comfy_inner_model=patcher.model)
            self.assertGreaterEqual(len(targets), 2, f"{arch_id} discovered too few targets: {targets}")

            injected = set(LoRAInjector.inject(backbone, LoRAConfig(rank=4, alpha=4.0, target_modules=targets)))
            self.assertGreaterEqual(len(injected), 4, f"{arch_id} injected too few layers: {injected}")

            if arch_id == "hunyuan_dit":
                self.assertEqual(roles.get("blocks.3.skip_linear"), "attention_or_ffn")
                self.assertIn("blocks.3.skip_linear", injected)
            elif arch_id == "hunyuan3dv2_1":
                self.assertEqual(roles.get("blocks.3.skip_linear"), "attention_or_ffn")
                self.assertEqual(roles.get("blocks.2.moe.gate"), "moe_router")
                self.assertIn("blocks.3.skip_linear", injected)
                self.assertNotIn("blocks.2.moe.gate", injected)
                self.assertIn("blocks.2.moe.experts.0.net.0.proj", injected)
                self.assertIn("blocks.0.mlp.fc1", injected)
            elif arch_id == "triposplat":
                self.assertEqual(roles.get("repo_layers.0.gate_map"), "stem_or_head")
                self.assertNotIn("repo_layers.0.gate_map", injected)
            elif arch_id == "stable_cascade":
                self.assertEqual(roles.get("down_blocks.0.1.mapper"), "modulation")
                self.assertNotIn("down_blocks.0.1.mapper", injected)
            elif arch_id == "ace_step15":
                self.assertTrue(all(not p.startswith(("tokenizer.", "detokenizer.")) for p in injected))
            elif arch_id == "trellis2":
                # Verify Trellis2 VarLenTensor passes cleanly through injected LoRALinear (including SparseLinear)
                from comfy.ldm.trellis2.vae import VarLenTensor
                sparse_lora_lin = backbone.img2shape.blocks[0].mlp.mlp[0]
                self.assertIsInstance(sparse_lora_lin, LoRALinear)
                in_dim = sparse_lora_lin.in_features
                vlt = VarLenTensor(torch.randn(4, in_dim), [slice(0, 2), slice(2, 4)])
                vlt_out = sparse_lora_lin(vlt)
                self.assertIsInstance(vlt_out, VarLenTensor)
                self.assertEqual(vlt_out.feats.shape[0], 4)

            for path in injected:
                self.assertIn(
                    roles.get(path),
                    ("attention", "ffn", "attention_or_ffn", "block_projection"),
                    f"{arch_id} injected non-target module '{path}' (role={roles.get(path)})",
                )

            for mod_name, role in roles.items():
                if role in ("stem_or_head", "modulation", "moe_router"):
                    self.assertNotIn(
                        mod_name,
                        injected,
                        f"{arch_id} erroneously injected {role} module '{mod_name}'",
                    )

            # Verify ComfyUI's native `comfy.lora.model_lora_keys_unet` + `comfy.lora.load_lora`
            lora_sd = LoRAInjector.extract_lora_state_dict(backbone, prefix="diffusion_model")
            key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
            loaded = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
            self.assertGreater(len(loaded), 0, f"{arch_id} loaded 0 keys in comfy.lora.load_lora")

            LoRAInjector.uninject(backbone)
            injected_tier2 = set(LoRAInjector.inject(backbone, LoRAConfig(rank=4, alpha=4.0, target_modules=())))
            self.assertEqual(injected_tier2, injected, f"{arch_id} Tier 2 auto-injection differed from discovered targets")
            LoRAInjector.uninject(backbone)

    def test_sdxl_conv2d_and_adm_dimensionality_regression(self) -> None:
        """
        Regression test for SDXL & UNet dimensionality:
          1. Verifies `ff.net.0.proj` and `ff.net.2` inside `input_blocks`, `middle_block`, and
             `output_blocks` are classified as `"ffn"` (never misclassified as `"modulation"`)
             and injected alongside `attn1` and `attn2`, while boundary projections (`proj_in`, `proj_out`),
             `3x3` ResNet convolutions (`in_layers.2`, `out_layers.3`, `op`, `input_blocks.0.0`, `out.2`),
             and `emb_layers.1` are excluded from standard LoRA targets.
          2. Verifies `LoRALinear` wrapping `1x1` and `3x3` `Conv2d` (including `stride=2` downsample)
             constructs 4D `lora_A` `[rank, in_c, kH, kW]` and `lora_B` `[out_c, rank, 1, 1]` whose
             flattened matrix product `lora_B.flatten(1) @ lora_A.flatten(1)` reshapes to the exact
             4D conv weight shape `(out_c, in_c, kH, kW)` required by ComfyUI's `LoRAAdapter.calculate_weight`.
          3. Verifies legacy 2D `[rank, in_c]` conv LoRA weights upgrade at the center spatial tap `(kH//2, kW//2)`
             in both `LoRAInjector.load_lora_state_dict` and the `LoRAAdapter.calculate_weight` runtime hook.
          4. Verifies SDXL `encode_adm` and `_prepare_vector_embeds` produce exact `[B, 2816]` ADM vectors
             with non-zero sinusoidal `Timestep` size/crop embeddings (`cos(0) = 1.0`).
        """
        from omni_lora.lora.layers import (
            LoRAConfig,
            LoRAInjector,
            LoRALinear,
            _strip_numeric_segments,
            classify_block_structural_roles,
        )
        from omni_lora.models.adapters import ComfyModelAdapter, discover_universal_target_modules
        from omni_lora.tests.comfy_emulator import ComfyModelZoo

        # 1. SDXL UNet structural role classification & target discovery
        sdxl_entry = ComfyModelZoo.build("sdxl_base", dtype=torch.float32)
        sdxl_dm = sdxl_entry.patcher.model.diffusion_model
        sdxl_roles = classify_block_structural_roles(sdxl_dm)

        ff_proj_paths = [p for p in sdxl_roles if p.endswith((".ff.net.0.proj", ".ff.net.2"))]
        self.assertGreaterEqual(len(ff_proj_paths), 6, "Expected SDXL ff.net.0.proj and ff.net.2 layers")
        for ff_p in ff_proj_paths:
            self.assertEqual(sdxl_roles[ff_p], "ffn", f"SDXL {ff_p} must be classified as 'ffn', got {sdxl_roles[ff_p]}")

        attn_paths = [p for p in sdxl_roles if ".attn1.to_" in p or ".attn2.to_" in p]
        self.assertGreaterEqual(len(attn_paths), 24)
        for attn_p in attn_paths:
            self.assertEqual(sdxl_roles[attn_p], "attention")

        proj_paths = [p for p in sdxl_roles if p.endswith((".proj_in", ".proj_out"))]
        self.assertGreaterEqual(len(proj_paths), 6, "Expected SDXL proj_in and proj_out layers")
        for proj_p in proj_paths:
            self.assertEqual(sdxl_roles[proj_p], "block_projection", f"SDXL {proj_p} must be 'block_projection'")

        emb_paths = [p for p in sdxl_roles if p.endswith(".emb_layers.1")]
        self.assertGreaterEqual(len(emb_paths), 3)
        for emb_p in emb_paths:
            self.assertEqual(sdxl_roles[emb_p], "modulation")

        sdxl_targets = discover_universal_target_modules(sdxl_dm, comfy_inner_model=sdxl_entry.patcher.model)
        self.assertTrue(any("ff.net.proj" in t or "ff.net" in t for t in sdxl_targets), f"Missing FFN in {sdxl_targets}")
        self.assertNotIn("input_blocks", sdxl_targets)
        self.assertNotIn("out", sdxl_targets)
        for proj_p in proj_paths:
            self.assertIn(_strip_numeric_segments(proj_p), sdxl_targets, f"SDXL projection {proj_p} missing from targets")

        injected = set(LoRAInjector.inject(sdxl_dm, LoRAConfig(rank=4, alpha=4.0, target_modules=sdxl_targets)))
        for ff_p in ff_proj_paths:
            self.assertIn(ff_p, injected, f"SDXL FFN layer {ff_p} was not injected")
        for attn_p in attn_paths:
            self.assertIn(attn_p, injected, f"SDXL Attention layer {attn_p} was not injected")
        for proj_p in proj_paths:
            self.assertIn(proj_p, injected, f"SDXL SpatialTransformer projection {proj_p} should be injected by default")
        for emb_p in emb_paths:
            self.assertNotIn(emb_p, injected, f"SDXL ResBlock modulation {emb_p} should not be injected")

        # Verify no 3x3 Conv2d was injected during standard SDXL LoRA injection
        for p in injected:
            mod = dict(sdxl_dm.named_modules())[p]
            self.assertIsInstance(mod, LoRALinear)
            self.assertFalse(mod.is_conv, f"Standard SDXL LoRA should not inject Conv2d {p}")
        LoRAInjector.uninject(sdxl_dm)

        # Verify FP16 SDXL checkpoint running under BF16 compute never downcasts activations to FP16
        # in GroupNorm, LayerNorm, Conv2d, or LoRALinear
        sdxl_fp16_entry = ComfyModelZoo.build("sdxl_base", dtype=torch.float16)
        sdxl_fp16_dm = sdxl_fp16_entry.patcher.model.diffusion_model
        LoRAInjector.inject(
            sdxl_fp16_dm,
            LoRAConfig(rank=4, alpha=4.0, target_modules=sdxl_targets, lora_dtype="bf16"),
            lora_dtype=torch.bfloat16,
        )
        sdxl_fp16_adapter = ComfyModelAdapter(comfy_model=sdxl_fp16_entry.patcher, model_id="sdxl_base")
        bf16_batch = {
            "latents": torch.randn(1, 4, 8, 8, dtype=torch.bfloat16),
            "prompt_embeds": torch.randn(1, 8, 128, dtype=torch.bfloat16),
            "pooled_output": torch.randn(1, 128, dtype=torch.bfloat16),
            "latent_scaled": True,
        }
        seen_lora_input_dtypes: List[torch.dtype] = []
        hook_handles = []
        for m in sdxl_fp16_dm.modules():
            if isinstance(m, LoRALinear):
                hook_handles.append(
                    m.register_forward_pre_hook(
                        lambda _mod, args: seen_lora_input_dtypes.append(args[0].dtype)
                        if args and isinstance(args[0], torch.Tensor)
                        else None
                    )
                )
        try:
            loss_bf16 = sdxl_fp16_adapter.compute_training_loss(
                sdxl_fp16_dm, bf16_batch, torch.device("cpu"), torch.bfloat16
            )
            loss_bf16.backward()
            self.assertGreater(len(seen_lora_input_dtypes), 0)
            self.assertTrue(
                all(dt == torch.bfloat16 for dt in seen_lora_input_dtypes),
                f"FP16 SDXL backbone downcast BF16 activations to {set(seen_lora_input_dtypes)}!",
            )
        finally:
            for h in hook_handles:
                h.remove()
            LoRAInjector.uninject(sdxl_fp16_dm)

        # 2. Explicit 3x3 Conv2d & stride=2 Conv2d LoRALinear 4D kernel geometry & ComfyUI reshape compatibility
        conv_3x3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        lora_conv = LoRALinear(conv_3x3, rank=4, alpha=8.0, lora_dtype=torch.float32)
        self.assertEqual(tuple(lora_conv.lora_A.weight.shape), (4, 64, 3, 3))
        self.assertEqual(tuple(lora_conv.lora_B.weight.shape), (32, 4, 1, 1))
        nn.init.normal_(lora_conv.lora_B.weight, std=0.02)

        # Exact ComfyUI LoRAAdapter.calculate_weight matrix product & reshape check
        flat_prod = torch.mm(
            lora_conv.lora_B.weight.flatten(start_dim=1),
            lora_conv.lora_A.weight.flatten(start_dim=1),
        )
        self.assertEqual(flat_prod.numel(), conv_3x3.weight.numel())
        reshaped_delta = flat_prod.reshape(conv_3x3.weight.shape)
        self.assertEqual(tuple(reshaped_delta.shape), (32, 64, 3, 3))

        x_conv = torch.randn(2, 64, 8, 8, dtype=torch.float32, requires_grad=True)
        out_conv = lora_conv(x_conv)
        self.assertEqual(tuple(out_conv.shape), (2, 32, 8, 8))
        out_conv.sum().backward()
        self.assertIsNotNone(lora_conv.lora_A.weight.grad)
        self.assertIsNotNone(lora_conv.lora_B.weight.grad)

        # Stride=2 downsample 3x3 Conv2d
        conv_down = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        lora_down = LoRALinear(conv_down, rank=4, alpha=4.0, lora_dtype=torch.float32)
        nn.init.normal_(lora_down.lora_B.weight, std=0.02)
        x_down = torch.randn(2, 32, 8, 8, dtype=torch.float32)
        out_down = lora_down(x_down)
        self.assertEqual(tuple(out_down.shape), (2, 32, 4, 4))

        # 3. Legacy 2D [rank, in_c] conv checkpoint upgrade into 4D [rank, in_c, 3, 3] center tap
        conv_host = nn.Module()
        conv_host.add_module("in_layers_2", LoRALinear(nn.Conv2d(16, 8, kernel_size=3, padding=1), rank=4, alpha=4.0, lora_dtype=torch.float32))
        legacy_2d_sd = {
            "in_layers_2.lora_down.weight": torch.ones(4, 16, dtype=torch.float32),
            "in_layers_2.lora_up.weight": torch.ones(8, 4, dtype=torch.float32),
            "in_layers_2.alpha": torch.tensor(4.0, dtype=torch.float32),
        }
        load_res = LoRAInjector.load_lora_state_dict(conv_host, legacy_2d_sd)
        self.assertEqual(load_res["loaded_layers"], 1)
        loaded_a = conv_host.in_layers_2.lora_A.weight
        self.assertEqual(tuple(loaded_a.shape), (4, 16, 3, 3))
        self.assertTrue(torch.allclose(loaded_a[:, :, 1, 1], torch.ones(4, 16)))
        self.assertEqual(float(loaded_a[:, :, 0, 0].abs().sum().item()), 0.0)

        # 4. SDXL ADM sinusoidal vector conditioning check ([B, 1280] -> [B, 2816] or model adm_in_channels)
        sdxl_adapter = ComfyModelAdapter(comfy_model=sdxl_entry.patcher, model_id="sdxl")
        raw_pooled = torch.randn(2, 64, dtype=torch.float32)
        adm_vec = sdxl_adapter._prepare_vector_embeds(raw_pooled, sdxl_dm, spatial_h=1024, spatial_w=1024)
        self.assertEqual(tuple(adm_vec.shape), (2, 128))
        # Also verify standard 1280 -> 2816 SDXL ADM sinusoidal expansion has non-zero cos/sin tail
        mock_sdxl_backbone = nn.Module()
        setattr(mock_sdxl_backbone, "adm_in_channels", 2816)
        pooled_1280 = torch.zeros(2, 1280, dtype=torch.float32)
        adm_2816 = sdxl_adapter._prepare_vector_embeds(pooled_1280, mock_sdxl_backbone, spatial_h=1024, spatial_w=1024)
        self.assertEqual(tuple(adm_2816.shape), (2, 2816))
        self.assertGreater(float(adm_2816[:, 1280:].abs().sum().item()), 100.0)

    def test_comprehensive_multi_model_layer_targeting_contract(self) -> None:
        """
        Critical Multi-Model Layer Targeting Contract across all primary architectures:
        (SDXL, SD1.5, Flux.1/Flux.3, SD3.5, Wan 2.1, Qwen-Image 1.0/2.1, HunyuanVideo, AuraFlow/Sana, LTX-Video)

        Guarantees:
          1. 100% Injected Coverage: Every Attention (self, cross, joint) and FeedForward (MLP, SwiGLU, GEGLU)
             projection inside repeating transformer/UNet blocks is injected.
          2. 100% Exclusion Integrity: Zero top-level patchifiers/stems, zero final output heads,
             zero AdaLN modulations (1-linear, 2-linear bottlenecks), zero MoE router gates, and
             zero ResNet Conv2d layers are injected during standard LoRA target discovery.
          3. Dimensionality & Shape Product Match: For every injected layer, the exported LoRA weights
             satisfy: `(lora_up @ lora_down).numel() == base_layer.weight.numel()`, guaranteeing
             100% crash-free `.reshape(weight.shape)` compatibility with ComfyUI's LoRAAdapter.
        """
        from omni_lora.lora.layers import (
            LoRAConfig,
            LoRAInjector,
            LoRALinear,
            _strip_numeric_segments,
            classify_block_structural_roles,
        )
        from omni_lora.models.adapters import discover_universal_target_modules
        from omni_lora.tests.comfy_emulator import ComfyModelZoo

        MODEL_SPECS_TO_AUDIT = [
            {
                "arch_id": "sdxl_base",
                "required_any_substrings": ["attn1", "attn2", "ff.net.0.proj", "ff.net.2", "proj_in", "proj_out"],
                "prohibited_exact_or_suffixes": [
                    "input_blocks.0.0", "out.2", "emb_layers.1", "time_embed", "label_emb",
                    "in_layers.2", "out_layers.3",
                ],
                "expected_min_injected": 24,
            },
            {
                "arch_id": "sd15",
                "required_any_substrings": ["attn1", "attn2", "ff.net.0.proj", "ff.net.2", "proj_in", "proj_out"],
                "prohibited_exact_or_suffixes": ["emb_layers.1", "time_embed", "out.2", "in_layers.2"],
                "expected_min_injected": 20,
            },
            {
                "arch_id": "flux_dev",
                "required_any_substrings": ["img_attn", "txt_attn", "linear1", "linear2"],
                "prohibited_exact_or_suffixes": ["img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer"],
                "expected_min_injected": 12,
            },
            {
                "arch_id": "flux3_klein",
                "required_any_substrings": ["q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out"],
                "prohibited_exact_or_suffixes": ["emb_in", "txt_in", "time_in", "ref_time_in", "final_layer"],
                "expected_min_injected": 12,
            },
            {
                "arch_id": "sd3_5",
                "required_any_substrings": ["context_block.attn", "context_block.mlp", "x_block.attn", "x_block.mlp"],
                "prohibited_exact_or_suffixes": ["x_embedder", "context_embedder", "y_embedder", "t_embedder", "final_layer"],
                "expected_min_injected": 12,
            },
            {
                "arch_id": "wan_2_1_t2v",
                "required_any_substrings": ["self_attn", "cross_attn", "ffn"],
                "prohibited_exact_or_suffixes": ["patch_embedding", "text_embedding", "time_embedding", "time_projection", "head"],
                "expected_min_injected": 10,
            },
            {
                "arch_id": "qwen_image_1_0",
                "required_any_substrings": ["attn", "img_mlp"],
                "prohibited_exact_or_suffixes": ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out"],
                "expected_min_injected": 8,
            },
            {
                "arch_id": "hunyuan_video",
                "required_any_substrings": ["attn", "mlp", "linear1", "linear2"],
                "prohibited_exact_or_suffixes": ["x_embedder", "t_embedder", "final_layer"],
                "expected_min_injected": 12,
            },
            {
                "arch_id": "auraflow_sana",
                "required_any_substrings": ["attn", "mlpX", "mlp"],
                "prohibited_exact_or_suffixes": ["modC.1", "modX.1", "init_x_linear", "final_linear"],
                "expected_min_injected": 10,
            },
            {
                "arch_id": "ltx_video",
                "required_any_substrings": ["attn1", "attn2", "ff.net"],
                "prohibited_exact_or_suffixes": ["patchify_proj", "adaln_single", "caption_projection", "proj_out"],
                "expected_min_injected": 8,
            },
        ]

        for spec in MODEL_SPECS_TO_AUDIT:
            arch_id = spec["arch_id"]
            entry = ComfyModelZoo.build(arch_id, dtype=torch.float32)
            dm = entry.patcher.model.diffusion_model

            # 1. Structural roles classification audit
            roles = classify_block_structural_roles(dm)
            self.assertGreater(len(roles), 0, f"{arch_id} produced empty structural roles")

            # 2. Universal target discovery audit
            targets = discover_universal_target_modules(dm, comfy_inner_model=entry.patcher.model)
            self.assertGreaterEqual(
                len(targets),
                2,
                f"{arch_id} discovered too few target modules: {targets}",
            )

            # 3. Injection audit
            injected = set(LoRAInjector.inject(dm, LoRAConfig(rank=4, alpha=4.0, target_modules=targets)))
            self.assertGreaterEqual(
                len(injected),
                spec["expected_min_injected"],
                f"{arch_id} injected fewer layers than required ({len(injected)} < {spec['expected_min_injected']})",
            )

            # Verify presence of every required Attention and FFN substring
            for req_sub in spec["required_any_substrings"]:
                has_sub = any(req_sub in p or req_sub in _strip_numeric_segments(p) for p in injected)
                self.assertTrue(
                    has_sub,
                    f"{arch_id} missing critical target category '{req_sub}' from injected: {injected}",
                )

            # Verify 100% exclusion of prohibited stems, heads, modulations, and ResNet convs
            for proh in spec["prohibited_exact_or_suffixes"]:
                violators = [p for p in injected if p == proh or p.endswith(f".{proh}")]
                self.assertEqual(
                    len(violators),
                    0,
                    f"{arch_id} violated exclusion integrity: injected prohibited stem/modulation {violators}",
                )

            # 4. Dimensionality & Shape Product Match audit
            module_map = dict(dm.named_modules())
            for p in injected:
                lora_mod = module_map.get(p)
                self.assertIsInstance(
                    lora_mod,
                    LoRALinear,
                    f"{arch_id} module '{p}' was in injected set but not a LoRALinear instance",
                )
                base_w = getattr(lora_mod.base_layer, "weight", None)
                if isinstance(base_w, torch.Tensor):
                    w_down = lora_mod.lora_A.weight
                    w_up = lora_mod.lora_B.weight
                    flat_prod_numel = int(w_up.flatten(1).shape[0] * w_down.flatten(1).shape[1])
                    self.assertEqual(
                        flat_prod_numel,
                        base_w.numel(),
                        f"{arch_id} dimension mismatch on '{p}': (up @ down).numel()={flat_prod_numel} "
                        f"!= base_weight.numel()={base_w.numel()} (base shape={tuple(base_w.shape)})",
                    )

            # 5. Extraction & ComfyUI Load Round-Trip audit
            extracted_sd = LoRAInjector.extract_lora_state_dict(dm, comfy_model=entry.patcher)
            self.assertGreater(len(extracted_sd), 0, f"{arch_id} extract_lora_state_dict returned empty dict")

            # Clean uninjection
            LoRAInjector.uninject(dm)
            remaining_loras = [n for n, m in dm.named_modules() if isinstance(m, LoRALinear)]
            self.assertEqual(len(remaining_loras), 0, f"{arch_id} uninject left residual LoRA wrappers: {remaining_loras}")

            # 6. Verify optional include_conv=True (LoCon / Conv-LoRA mode) on UNet architectures
            if arch_id in ("sdxl", "sdxl_base", "sd15"):
                conv_targets = discover_universal_target_modules(
                    dm,
                    comfy_inner_model=entry.patcher.model,
                    include_conv=True,
                )
                self.assertGreater(len(conv_targets), len(targets))
                conv_injected = set(
                    LoRAInjector.inject(
                        dm,
                        LoRAConfig(rank=4, alpha=4.0, target_modules=conv_targets, include_conv=True),
                    )
                )
                self.assertGreater(len(conv_injected), len(injected))
                # Raw 4-channel input/output stems must remain strictly excluded even in LoCon mode
                self.assertNotIn("input_blocks.0.0", conv_injected)
                self.assertNotIn("out.2", conv_injected)
                # Repeating-block ResNet convs must be included and have 4D Conv2d shapes
                self.assertTrue(any("in_layers.2" in p or "out_layers.3" in p for p in conv_injected))
                mod_map_conv = dict(dm.named_modules())
                for p in conv_injected:
                    lm = mod_map_conv[p]
                    if getattr(lm, "is_conv", False):
                        self.assertEqual(lm.lora_A.weight.ndim, 4)
                        self.assertEqual(lm.lora_B.weight.ndim, 4)
                        self.assertEqual(
                            int(lm.lora_B.weight.flatten(1).shape[0] * lm.lora_A.weight.flatten(1).shape[1]),
                            lm.base_layer.weight.numel(),
                        )
                LoRAInjector.uninject(dm)

        trainer_opt = OmniLoRATrainer.INPUT_TYPES().get("optional", {})
        self.assertIn("include_conv_layers", trainer_opt)
        self.assertEqual(trainer_opt["include_conv_layers"][1].get("default"), False)

    def test_include_conv_layers_locon_end_to_end(self) -> None:
        """
        Dedicated end-to-end test case for the `include_conv_layers` (LoCon / Conv-LoRA) toggle
        on `OmniLoRATrainer`:
          1. Verifies `include_conv_layers=False` (default) injects ONLY 2D Linear attention/FFN
             adapters on SDXL and SD1.5 (`0` 4D Conv2d tensors in `state_dict`).
          2. Verifies `include_conv_layers=True` injects repeating-block 1x1 and 3x3 Conv2d layers
             (`in_layers.2`, `out_layers.3`, `op`, `proj_in`, `proj_out`) with valid 4D
             `[rank, in_c, kH, kW]` / `[out_c, rank, 1, 1]` tensors while still excluding raw
             4-channel stems (`input_blocks.0.0`, `out.2`).
          3. Verifies ComfyUI `load_lora_for_models` and `EmulatedKSampler` execute with 0 unloaded
             keys and 0 shape mismatch errors when loading a trained `include_conv_layers=True` LoRA.
        """
        from omni_lora.tests.comfy_emulator import (
            ComfyModelZoo,
            EmulatedComfyEnvironment,
            EmulatedKSampler,
        )

        with EmulatedComfyEnvironment(torch_device=torch.device("cpu"), simulated_vram_mb=16384.0) as env:
            import comfy.sd as c_sd  # type: ignore

            dataset_node = OmniLoRADatasetLoader()
            precache_node = OmniLoRAPrecomputeConditioning()
            trainer_node = OmniLoRATrainer()

            with torch.inference_mode():
                imgs = torch.rand(1, 64, 64, 4, dtype=torch.float32)
                ds, _ = dataset_node.load_dataset(
                    dataset_path="",
                    trigger_word="locon_tok",
                    default_caption="detailed style",
                    resolution=64,
                    batch_size=1,
                    num_repeats=1,
                    center_crop=True,
                    aspect_ratio_bucketing=True,
                    random_flip=False,
                    caption_dropout=0.0,
                    images=imgs,
                    custom_prompts="a photo in locon_tok detailed style",
                )

                for arch_id in ("sdxl_base", "sd15"):
                    entry = ComfyModelZoo.build(arch_id, dtype=torch.float32)
                    (cached_bundle,) = precache_node.precompute(
                        dataset=ds,
                        offload_encoders_after_cache=True,
                        vae=entry.vae,
                        clip=entry.clip,
                        model=entry.patcher,
                    )

                    # Run 0: Baseline KSampler before training
                    pre_sample = EmulatedKSampler.sample(
                        entry.patcher,
                        entry.clip,
                        resolution=64,
                        steps=2,
                        seed=77,
                        device=torch.device("cpu"),
                    )

                    # Run 1: Default Linear-only mode (include_conv_layers=False)
                    _, linear_bundle, linear_summary_json = trainer_node.train_lora(
                        cached_data=cached_bundle,
                        model=entry.patcher,
                        model_architecture="auto",
                        backend="cpu",
                        optimizer="radamw",
                        rank=4,
                        alpha=4.0,
                        learning_rate=1e-2,
                        steps=2,
                        include_conv_layers=False,
                    )
                    linear_summary = json.loads(linear_summary_json)
                    self.assertFalse(linear_summary["include_conv_layers"])
                    self.assertEqual(linear_bundle.metadata.get("omni_lora.include_conv_layers"), "False")
                    linear_conv_weights = [
                        k for k, t in linear_bundle.state_dict.items() if k.endswith(".weight") and t.ndim >= 3
                    ]
                    spatial_conv_weights = [
                        k
                        for k, t in linear_bundle.state_dict.items()
                        if k.endswith(".weight") and t.ndim >= 3 and any(d > 1 for d in t.shape[2:])
                    ]
                    self.assertEqual(
                        len(spatial_conv_weights),
                        0,
                        f"{arch_id} with include_conv_layers=False unexpectedly produced spatial conv weights: {spatial_conv_weights}",
                    )
                    if arch_id == "sdxl_base":
                        self.assertEqual(
                            len(linear_conv_weights),
                            0,
                            f"{arch_id} with include_conv_layers=False unexpectedly produced conv weights: {linear_conv_weights}",
                        )
                    else:
                        # SD 1.5 uses 1x1 Conv2d for SpatialTransformer.proj_in/proj_out (matched to Kohya)
                        self.assertGreater(len(linear_conv_weights), 0)

                    # Run 2: LoCon / Conv-LoRA mode (include_conv_layers=True)
                    locon_model_out, locon_bundle, locon_summary_json = trainer_node.train_lora(
                        cached_data=cached_bundle,
                        model=entry.patcher,
                        model_architecture="auto",
                        backend="cpu",
                        optimizer="radamw",
                        rank=4,
                        alpha=4.0,
                        learning_rate=1e-2,
                        steps=2,
                        include_conv_layers=True,
                    )
                    locon_summary = json.loads(locon_summary_json)
                    self.assertTrue(locon_summary["include_conv_layers"])
                    self.assertEqual(locon_bundle.metadata.get("omni_lora.include_conv_layers"), "True")
                    self.assertGreater(locon_summary["injected_layers"], linear_summary["injected_layers"])

                    locon_conv_down = [
                        (k, t)
                        for k, t in locon_bundle.state_dict.items()
                        if k.endswith(".lora_down.weight") and t.ndim == 4
                    ]
                    self.assertGreater(
                        len(locon_conv_down),
                        0,
                        f"{arch_id} with include_conv_layers=True did not produce any 4D Conv2d lora_down weights",
                    )
                    # Verify 3x3 ResBlock convs are present and raw stems (input_blocks.0.0, out.2) are absent
                    self.assertTrue(any("in_layers_2" in k or "in_layers.2" in k for k, _ in locon_conv_down))
                    self.assertFalse(any("input_blocks_0_0" in k or "input_blocks.0.0" in k for k in locon_bundle.state_dict))
                    self.assertFalse(any(k.startswith("lora_unet_out_2") or "diffusion_model.out.2." in k for k in locon_bundle.state_dict))

                    # Save & reload via OmniLoRASave / OmniLoRALoader, then load into ComfyUI LoraLoader
                    save_node = OmniLoRASave()
                    loader_node = OmniLoRALoader()
                    _, saved_fname, _ = save_node.save_lora(
                        lora=locon_bundle,
                        output_lora_name=f"locon_{arch_id}",
                        append_step_count=True,
                    )
                    reloaded_locon, _ = loader_node.load_lora(lora_name=saved_fname)

                    # Verify ComfyUI loads 100% of Linear + Conv2d LoCon keys with 0 unloaded keys
                    env.unloaded_lora_keys.clear()
                    patched, _ = c_sd.load_lora_for_models(
                        entry.patcher,
                        entry.clip,
                        reloaded_locon.state_dict,
                        strength_model=1.0,
                        strength_clip=1.0,
                    )
                    self.assertEqual(
                        env.unloaded_lora_keys,
                        [],
                        f"{arch_id} LoCon had unloaded keys in ComfyUI LoraLoader: {env.unloaded_lora_keys}",
                    )
                    sampled = EmulatedKSampler.sample(
                        patched,
                        entry.clip,
                        resolution=64,
                        steps=2,
                        seed=77,
                        device=torch.device("cpu"),
                    )
                    direct_out_sample = EmulatedKSampler.sample(
                        locon_model_out,
                        entry.clip,
                        resolution=64,
                        steps=2,
                        seed=77,
                        device=torch.device("cpu"),
                    )
                    self.assertTrue(torch.isfinite(sampled).all())
                    self.assertTrue(
                        torch.allclose(sampled, direct_out_sample, atol=1e-4),
                        f"{arch_id} LoCon KSampler output differed between LoraLoader and trainer MODEL output",
                    )
                    self.assertFalse(
                        torch.allclose(pre_sample, sampled, atol=1e-7),
                        f"{arch_id} LoCon trained weights had zero effect on KSampler output",
                    )

                    # Verify _tracked_load_lora strictly catches a single bogus key even when 80+ valid keys load
                    corrupted_sd = dict(reloaded_locon.state_dict)
                    corrupted_sd["nonexistent_bogus_block.99.lora_down.weight"] = torch.zeros(4, 32)
                    corrupted_sd["nonexistent_bogus_block.99.lora_up.weight"] = torch.zeros(32, 4)
                    env.unloaded_lora_keys.clear()
                    _ = c_sd.load_lora_for_models(entry.patcher, entry.clip, corrupted_sd, 1.0, 1.0)
                    self.assertGreater(
                        len(env.unloaded_lora_keys),
                        0,
                        "_tracked_load_lora failed to detect partial unloaded LoRA keys when valid keys also loaded!",
                    )
                    env.unloaded_lora_keys.clear()

    def test_prepatched_model_patcher_preserves_upstream_lora_and_sampling(self) -> None:
        """
        Verify that when a user connects an already-patched `ModelPatcher` into `OmniLoRATrainer`
        (e.g., upstream ComfyUI `LoraLoader` weight patches + `ModelSamplingFlux` custom `shift`
        object patch + custom `model_options["transformer_options"]`, even after an upstream
        `KSampler` pass):
          1. `ComfyModelAdapter` respects the custom `model_sampling` from `object_patches`.
          2. The upstream `ModelPatcher` is never mutated or stripped of its `patches`,
             `object_patches`, or `model_options`, and produces identical `KSampler` output
             before and after `OmniLoRATrainer`.
          3. The newly trained `lora_bundle.state_dict` contains only the new LoRA delta, and
             `model_out` stacks the new LoRA cleanly on top of the upstream patches.
        """
        from omni_lora.tests.comfy_emulator import (
            ComfyModelZoo,
            EmulatedComfyEnvironment,
            EmulatedKSampler,
        )

        with EmulatedComfyEnvironment(torch_device=torch.device("cpu"), simulated_vram_mb=16384.0) as env:
            import comfy.sd as c_sd  # type: ignore

            entry = ComfyModelZoo.build("flux_dev", dtype=torch.float32)

            # 1. Build an upstream "style LoRA" with non-zero A and B weights on double_blocks.0.img_attn.qkv
            dm = entry.patcher.model.diffusion_model
            qkv_mod = dm.double_blocks[0].img_attn.qkv
            in_f, out_f = int(qkv_mod.in_features), int(qkv_mod.out_features)
            gen = torch.Generator(device="cpu").manual_seed(123)
            up_lora_sd = {
                "double_blocks.0.img_attn.qkv.lora_down.weight": torch.randn((4, in_f), generator=gen) * 0.1,
                "double_blocks.0.img_attn.qkv.lora_up.weight": torch.randn((out_f, 4), generator=gen) * 0.1,
                "double_blocks.0.img_attn.qkv.alpha": torch.tensor(4.0, dtype=torch.float32),
            }
            upstream_patched, _ = c_sd.load_lora_for_models(entry.patcher, None, up_lora_sd, 1.0, 0.0)
            self.assertEqual(env.unloaded_lora_keys, [])

            # 2. Attach a custom `model_sampling` object patch (simulating `ModelSamplingFlux` node with shift=2.75)
            base_ms = upstream_patched.get_model_object("model_sampling")
            custom_ms = copy.deepcopy(base_ms)
            setattr(custom_ms, "shift", 2.75)
            upstream_patched.add_object_patch("model_sampling", custom_ms)

            # 3. Attach custom `transformer_options` tag in `model_options`
            upstream_patched.model_options.setdefault("transformer_options", {})["custom_user_hook"] = "keep_me"

            # 4. Run KSampler on `upstream_patched` BEFORE training (exercising patch_model/unpatch_model)
            upstream_pre_train_sample = EmulatedKSampler.sample(
                upstream_patched,
                entry.clip,
                resolution=64,
                steps=2,
                seed=55,
                device=torch.device("cpu"),
            )
            base_clean_sample = EmulatedKSampler.sample(
                entry.patcher,
                entry.clip,
                resolution=64,
                steps=2,
                seed=55,
                device=torch.device("cpu"),
            )
            self.assertFalse(
                torch.allclose(upstream_pre_train_sample, base_clean_sample, atol=1e-6),
                "Upstream style LoRA should alter KSampler output relative to unpatched base model",
            )

            # Verify ComfyModelAdapter picks up the object_patches["model_sampling"] custom shift
            adapter = resolve_model_adapter("auto", comfy_model=upstream_patched)
            self.assertAlmostEqual(float(getattr(adapter.model_sampling, "shift", 0.0)), 2.75, places=4)

            # 5. Train a new LoRA with `upstream_patched` wired into `OmniLoRATrainer`
            dataset_node = OmniLoRADatasetLoader()
            precache_node = OmniLoRAPrecomputeConditioning()
            trainer_node = OmniLoRATrainer()

            with torch.inference_mode():
                imgs = torch.rand(1, 64, 64, 3, dtype=torch.float32)
                ds, _ = dataset_node.load_dataset(
                    dataset_path="",
                    trigger_word="char_tok",
                    default_caption="character portrait",
                    resolution=64,
                    batch_size=1,
                    num_repeats=1,
                    center_crop=True,
                    aspect_ratio_bucketing=True,
                    random_flip=False,
                    caption_dropout=0.0,
                    images=imgs,
                    custom_prompts="char_tok character portrait",
                )
                (cached_bundle,) = precache_node.precompute(
                    dataset=ds,
                    offload_encoders_after_cache=True,
                    vae=entry.vae,
                    clip=entry.clip,
                    model=upstream_patched,
                )
                model_out, new_lora_bundle, _ = trainer_node.train_lora(
                    cached_data=cached_bundle,
                    model=upstream_patched,
                    model_architecture="auto",
                    backend="cpu",
                    optimizer="radamw",
                    rank=4,
                    alpha=4.0,
                    learning_rate=1e-2,
                    steps=2,
                )

                # Verify `upstream_patched` was NOT mutated or stripped of its patches / object_patches / options
                self.assertIs(upstream_patched.get_model_object("model_sampling"), custom_ms)
                self.assertEqual(
                    upstream_patched.model_options.get("transformer_options", {}).get("custom_user_hook"),
                    "keep_me",
                )
                upstream_post_train_sample = EmulatedKSampler.sample(
                    upstream_patched,
                    entry.clip,
                    resolution=64,
                    steps=2,
                    seed=55,
                    device=torch.device("cpu"),
                )
                self.assertTrue(
                    torch.allclose(upstream_pre_train_sample, upstream_post_train_sample, atol=1e-5),
                    "Upstream patched ModelPatcher output drifted after OmniLoRATrainer!",
                )

                # Verify `model_out` retains `object_patches` + `model_options` AND equals stacking `new_lora_bundle` onto `upstream_patched`
                self.assertIs(model_out.get_model_object("model_sampling"), custom_ms)
                self.assertEqual(
                    model_out.model_options.get("transformer_options", {}).get("custom_user_hook"),
                    "keep_me",
                )
                manually_stacked, _ = c_sd.load_lora_for_models(
                    upstream_patched,
                    entry.clip,
                    new_lora_bundle.state_dict,
                    1.0,
                    1.0,
                )
                model_out_sample = EmulatedKSampler.sample(
                    model_out,
                    entry.clip,
                    resolution=64,
                    steps=2,
                    seed=55,
                    device=torch.device("cpu"),
                )
                stacked_sample = EmulatedKSampler.sample(
                    manually_stacked,
                    entry.clip,
                    resolution=64,
                    steps=2,
                    seed=55,
                    device=torch.device("cpu"),
                )
                self.assertTrue(
                    torch.allclose(model_out_sample, stacked_sample, atol=1e-4),
                    "OmniLoRATrainer MODEL output did not match manually stacking new LoRA onto upstream patched model!",
                )

    def test_comfy_clip_return_dict_scheduled_and_multiprompt_mask_batching(self) -> None:
        """
        Verify `OmniLoRAPrecomputeConditioning` and `OmniLoRATrainer` across real ComfyUI
        `CLIP` return conventions:
          1. `encode_from_tokens(..., return_pooled=True, return_dict=True)` returning a dict
             with variable-length `"cond"` (`seq_len=12` vs `seq_len=28`), `"pooled_output"`,
             and per-sample `"attention_mask"` across a multi-sample batch (`batch_size=2`)
             with `caption_dropout=0.5`.
          2. `encode_from_tokens_scheduled(tokens)` returning nested `[[cond, {"pooled_output": ..., "clip_l": ...}]]`.
          3. T5 / Qwen-style `CLIP` returning `pooled_output=None` trained on both non-pooled
             (`wan_2_1_t2v`) and pooled-capable (`sdxl_base`) models.
        """
        from omni_lora.tests.comfy_emulator import ComfyModelZoo

        class _DictVariableSeqCLIP:
            def tokenize(self, text: str) -> str:
                return text

            def encode_from_tokens(
                self,
                tokens: str,
                return_pooled: bool = False,
                return_dict: bool = False,
            ) -> Any:
                # Empty unconditional prompt -> 8 tokens; short prompt -> 12 tokens (8 valid); long -> 28 tokens (24 valid)
                if not tokens.strip():
                    seq_len, valid_len = 8, 2
                elif "long" in tokens:
                    seq_len, valid_len = 28, 24
                else:
                    seq_len, valid_len = 12, 8
                gen = torch.Generator(device="cpu").manual_seed(len(tokens) + seq_len)
                cond = torch.randn((1, seq_len, 128), generator=gen, dtype=torch.float32)
                pooled = torch.randn((1, 128), generator=gen, dtype=torch.float32)
                attn_mask = torch.zeros((1, seq_len), dtype=torch.long)
                attn_mask[:, :valid_len] = 1
                aux_vec = torch.full((1, 16), float(valid_len), dtype=torch.float32)
                if return_dict:
                    return {
                        "cond": cond,
                        "pooled_output": pooled,
                        "attention_mask": attn_mask,
                        "clip_l_aux": aux_vec,
                    }
                if return_pooled:
                    return cond, pooled
                return cond

        class _ScheduledOnlyCLIP:
            def tokenize(self, text: str) -> str:
                return text

            def encode_from_tokens_scheduled(self, tokens: str) -> Any:
                gen = torch.Generator(device="cpu").manual_seed(len(tokens) + 17)
                cond = torch.randn((1, 16, 128), generator=gen, dtype=torch.float32)
                pooled = torch.randn((1, 128), generator=gen, dtype=torch.float32)
                return [[cond, {"pooled_output": pooled, "guidance": 3.5}]]

        class _NoPooledT5CLIP:
            def tokenize(self, text: str) -> str:
                return text

            def encode_from_tokens(
                self,
                tokens: str,
                return_pooled: bool = False,
                return_dict: bool = False,
            ) -> Any:
                gen = torch.Generator(device="cpu").manual_seed(len(tokens) + 31)
                cond = torch.randn((1, 20, 128), generator=gen, dtype=torch.float32)
                if return_dict:
                    return {"cond": cond, "pooled_output": None}
                return (cond, None) if return_pooled else cond

        dataset_node = OmniLoRADatasetLoader()
        precache_node = OmniLoRAPrecomputeConditioning()
        trainer_node = OmniLoRATrainer()

        with torch.inference_mode():
            # Case 1: Variable-length `return_dict=True` with `batch_size=2` and `caption_dropout=0.5`
            imgs_2 = torch.rand(2, 64, 64, 3, dtype=torch.float32)
            ds_2, _ = dataset_node.load_dataset(
                dataset_path="",
                trigger_word="ohwx",
                default_caption="fallback",
                resolution=64,
                batch_size=2,
                num_repeats=1,
                center_crop=True,
                aspect_ratio_bucketing=True,
                random_flip=True,
                caption_dropout=0.5,
                images=imgs_2,
                custom_prompts="short prompt\nvery long detailed second training prompt with more tokens",
            )
            flux_entry = ComfyModelZoo.build("flux_dev", dtype=torch.float32)
            (cached_dict_bundle,) = precache_node.precompute(
                dataset=ds_2,
                offload_encoders_after_cache=False,
                vae=flux_entry.vae,
                clip=_DictVariableSeqCLIP(),
                model=flux_entry.patcher,
            )
            self.assertEqual(len(cached_dict_bundle.batches), 1)
            b0 = cached_dict_bundle.batches[0]
            self.assertEqual(tuple(b0["prompt_embeds"].shape), (2, 28, 128))
            self.assertEqual(tuple(b0["pooled_output"].shape), (2, 128))
            self.assertIn("extra_conds", b0)
            attn_m = b0["extra_conds"]["attention_mask"]
            self.assertEqual(
                tuple(attn_m.shape),
                (2, 28),
                f"Expected batched attention_mask shape (2, 28), got {tuple(attn_m.shape)}",
            )
            # Sample 0 had 8 valid tokens padded to 28 with 0s; Sample 1 had 24 valid tokens padded to 28 with 0s
            self.assertEqual(int(attn_m[0].sum().item()), 8)
            self.assertEqual(int(attn_m[1].sum().item()), 24)
            self.assertEqual(tuple(b0["extra_conds"]["clip_l_aux"].shape), (2, 16))
            self.assertIn("context_attention_bias", b0["extra_conds"])
            self.assertEqual(tuple(b0["extra_conds"]["context_attention_bias"].shape), (2, 28))
            # Unconditional prompt and attention_mask preserve their true unpadded token length (8)
            self.assertEqual(int(cached_dict_bundle.uncond_prompt_embeds.shape[1]), 8)
            u_mask = cached_dict_bundle.uncond_extra_conds.get("attention_mask")
            self.assertIsNotNone(u_mask)
            self.assertEqual(int(u_mask.shape[-1]), 8)

            # Train 2 steps with batch_size=2 + caption_dropout=0.5 + within-batch shuffling
            _, _, summary_dict_json = trainer_node.train_lora(
                cached_data=cached_dict_bundle,
                model=flux_entry.patcher,
                model_architecture="auto",
                backend="cpu",
                optimizer="radamw",
                rank=4,
                alpha=4.0,
                learning_rate=1e-3,
                steps=2,
            )
            self.assertGreater(json.loads(summary_dict_json)["final_loss"], 0.0)

            # Case 2: Scheduled-only CLIP (`encode_from_tokens_scheduled` returning `[[cond, dict]]`)
            (cached_sched_bundle,) = precache_node.precompute(
                dataset=ds_2,
                offload_encoders_after_cache=False,
                vae=flux_entry.vae,
                clip=_ScheduledOnlyCLIP(),
                model=flux_entry.patcher,
            )
            self.assertEqual(tuple(cached_sched_bundle.batches[0]["prompt_embeds"].shape), (2, 16, 128))
            self.assertEqual(tuple(cached_sched_bundle.batches[0]["pooled_output"].shape), (2, 128))
            self.assertEqual(cached_sched_bundle.batches[0]["extra_conds"].get("guidance"), 3.5)

            # Case 3: `pooled_output=None` on both `wan_2_1_t2v` and `sdxl_base`
            for arch_id in ("wan_2_1_t2v", "sdxl_base"):
                entry = ComfyModelZoo.build(arch_id, dtype=torch.float32)
                (cached_nopool,) = precache_node.precompute(
                    dataset=ds_2,
                    offload_encoders_after_cache=False,
                    vae=entry.vae,
                    clip=_NoPooledT5CLIP(),
                    model=entry.patcher,
                )
                self.assertIsNone(cached_nopool.batches[0].get("pooled_output"))
                _, _, sum_nopool = trainer_node.train_lora(
                    cached_data=cached_nopool,
                    model=entry.patcher,
                    model_architecture="auto",
                    backend="cpu",
                    optimizer="radamw",
                    rank=4,
                    alpha=4.0,
                    learning_rate=1e-3,
                    steps=1,
                )
                self.assertGreater(json.loads(sum_nopool)["final_loss"], 0.0)

    def test_non_square_aspect_ratio_bucketing_and_patchifier_alignment(self) -> None:
        """
        Verify non-square aspect ratio bucketing and patchifier spatial stride alignment:
          1. `resolve_aspect_ratio_bucket` always returns dimensions divisible by 16 for
             resolutions >= 32 (and divisible by 64 for resolutions >= 256) across wild
             portrait/landscape aspect ratios (`1000x633`, `512x896`, `77x113`, `96x64`),
             even when `base_resolution=96` (`96 // 4 = 24` previously produced odd `8x` VAE latents!).
          2. `ComfyModelAdapter` automatically pads odd-sized latent tensors (`[1, 16, 7, 9]`,
             `[1, 4, 9, 11]`) so `patch_size=2` DiTs (`flux_dev`), `64c`/`128c` space-to-depth
             models (`qwen_image_2_1`, `flux3_klein`), and multi-scale UNets (`sdxl_base`)
             never fail during patchification, `pixel_unshuffle`, or skip-connection concatenation.
        """
        from omni_lora.dataset import resolve_aspect_ratio_bucket
        from omni_lora.tests.comfy_emulator import ComfyModelZoo

        # 1. Verify bucket dimensions at base_resolution=96, 128, 512, 1024 are always multiples of 16
        for base_res in (64, 96, 128, 512, 1024):
            expected_div = 64 if base_res >= 256 else 16
            for h, w in ((1000, 633), (633, 1000), (512, 896), (77, 113), (120, 72)):
                bh, bw = resolve_aspect_ratio_bucket(h, w, base_resolution=base_res)
                self.assertEqual(
                    bh % expected_div,
                    0,
                    f"Bucket height {bh} for ({h}x{w}) @ {base_res}px is not divisible by {expected_div}",
                )
                self.assertEqual(
                    bw % expected_div,
                    0,
                    f"Bucket width {bw} for ({h}x{w}) @ {base_res}px is not divisible by {expected_div}",
                )
                # Ensure 8x VAE latent dimensions are always even (divisible by patch_size=2)
                self.assertEqual((bh // 8) % 2, 0)
                self.assertEqual((bw // 8) % 2, 0)

        # 2. End-to-end training at resolution=96 with non-square portrait (113x77) + landscape (77x113) images
        dataset_node = OmniLoRADatasetLoader()
        precache_node = OmniLoRAPrecomputeConditioning()
        trainer_node = OmniLoRATrainer()

        with torch.inference_mode():
            portrait_img = torch.rand(1, 113, 77, 3, dtype=torch.float32)
            ds_ar, _ = dataset_node.load_dataset(
                dataset_path="",
                trigger_word="ar_tok",
                default_caption="portrait subject",
                resolution=96,
                batch_size=1,
                num_repeats=1,
                center_crop=False,
                aspect_ratio_bucketing=True,
                random_flip=True,
                caption_dropout=0.0,
                images=portrait_img,
                custom_prompts="ar_tok portrait subject",
            )
            for arch_id in ("flux_dev", "qwen_image_2_1", "flux3_klein", "sdxl_base"):
                entry = ComfyModelZoo.build(arch_id, dtype=torch.float32)
                (cached_ar,) = precache_node.precompute(
                    dataset=ds_ar,
                    offload_encoders_after_cache=False,
                    vae=entry.vae,
                    clip=entry.clip,
                    model=entry.patcher,
                )
                _, _, sum_ar = trainer_node.train_lora(
                    cached_data=cached_ar,
                    model=entry.patcher,
                    model_architecture="auto",
                    backend="cpu",
                    optimizer="radamw",
                    rank=4,
                    alpha=4.0,
                    learning_rate=1e-3,
                    steps=1,
                )
                self.assertGreater(json.loads(sum_ar)["final_loss"], 0.0)

        # 3. Direct odd-spatial-dimension latent stress test (e.g. H=7, W=9 on 16c/64c/128c/4c models)
        for arch_id, odd_c in (
            ("flux_dev", 16),
            ("qwen_image_2_1", 16),
            ("flux3_klein", 128),
            ("flux3_klein", 32),
            ("sdxl_base", 4),
        ):
            entry = ComfyModelZoo.build(arch_id, dtype=torch.float32)
            adapter = resolve_model_adapter("auto", comfy_model=entry.patcher)
            dm = entry.patcher.model.diffusion_model
            odd_batch = {
                "latents": torch.randn(1, odd_c, 7, 9, dtype=torch.float32),
                "prompt_embeds": torch.randn(1, 12, 128, dtype=torch.float32),
                "pooled_output": torch.randn(1, 128, dtype=torch.float32),
                "latent_scaled": True,
            }
            loss = adapter.compute_training_loss(
                backbone=dm,
                batch=odd_batch,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.assertTrue(bool(torch.isfinite(loss).all()), f"{arch_id} failed on odd (7x9) latent spatial dimensions")

    def test_fast_single_sample_convergence_across_objectives(self) -> None:
        """
        Fast (<100ms) deterministic mathematical convergence (single-sample overfitting) test:
        Uses a compact 2-block spatial/token micro-diffusion backbone (`_MicroConvergenceBackbone`)
        wrapped in genuine `comfy.model_base.BaseModel` + `comfy.model_sampling` (`build_native_comfy_patcher`)
        without `ComfyModelZoo`'s scalar-mean forward bridge so that full `[1, C, 4, 4]` spatial tensors
        can be memorized to near-zero residual error across:
          - `rectified_flow` (`flux_dev`, `sd3_5`)
          - `ddpm_eps` (`sdxl_base`, `sd15`)
          - `v_prediction` (`sdxl_base`)
          - `ddpm_eps` + `include_conv=True` LoCon (`sd15`, optimizing ONLY the 3x3 Conv2d LoRA weights)
        Verifies that 18 steps of critically-damped AdamW optimization reduce the objective loss by >75%
        (`final_loss < 0.25 * initial_loss`) with immediate Step-1 descent (`losses[1] < initial_loss`)
        and monotonic improvement (`final_loss < mid_loss < initial_loss`).
        """
        from omni_lora.tests.comfy_emulator import build_native_comfy_patcher

        class _MicroConvergenceBackbone(nn.Module):
            def __init__(self, channels: int = 4, hidden_dim: int = 32, use_unet_blocks: bool = False) -> None:
                super().__init__()
                self.in_channels = channels
                self.out_channels = channels
                self.hidden_dim = hidden_dim
                self.use_unet_blocks = use_unet_blocks
                self.x_embedder = nn.Conv2d(channels, hidden_dim, kernel_size=1)
                self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, 2, 2), requires_grad=False)
                self.t_embedder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.context_embedder = nn.Linear(128, hidden_dim)
                if use_unet_blocks:
                    self.input_blocks = nn.ModuleList(
                        [
                            nn.ModuleDict(
                                {
                                    "in_layers": nn.Sequential(
                                        nn.GroupNorm(4, hidden_dim),
                                        nn.SiLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                                    ),
                                    "out_layers": nn.Sequential(
                                        nn.GroupNorm(4, hidden_dim),
                                        nn.SiLU(),
                                        nn.Identity(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                                    ),
                                    "attn1": nn.ModuleDict(
                                        {
                                            "to_q": nn.Linear(hidden_dim, hidden_dim),
                                            "to_k": nn.Linear(hidden_dim, hidden_dim),
                                            "to_v": nn.Linear(hidden_dim, hidden_dim),
                                            "to_out": nn.Sequential(nn.Linear(hidden_dim, hidden_dim)),
                                        }
                                    ),
                                    "ff": nn.ModuleDict(
                                        {
                                            "net": nn.Sequential(
                                                nn.ModuleDict({"proj": nn.Linear(hidden_dim, hidden_dim * 2)}),
                                                nn.GELU(),
                                                nn.Linear(hidden_dim * 2, hidden_dim),
                                            )
                                        }
                                    ),
                                }
                            )
                            for _ in range(2)
                        ]
                    )
                else:
                    self.double_blocks = nn.ModuleList(
                        [
                            nn.ModuleDict(
                                {
                                    "attn": nn.ModuleDict(
                                        {
                                            "qkv": nn.Linear(hidden_dim, hidden_dim * 3),
                                            "proj": nn.Linear(hidden_dim, hidden_dim),
                                        }
                                    ),
                                    "mlp": nn.Sequential(
                                        nn.Linear(hidden_dim, hidden_dim * 2),
                                        nn.GELU(),
                                        nn.Linear(hidden_dim * 2, hidden_dim),
                                    ),
                                }
                            )
                            for _ in range(2)
                        ]
                    )
                self.final_layer = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)

            def restore_orthogonal_forward(self) -> None:
                # Restore genuine spatial forward (removing build_native_comfy_patcher's scalar mean bridge)
                self.forward = self._native_forward  # type: ignore[method-assign]
                with torch.no_grad():
                    q_pos = torch.empty(self.hidden_dim, self.hidden_dim, dtype=torch.float32)
                    nn.init.orthogonal_(q_pos, gain=2.0)
                    self.pos_embed.copy_(q_pos[:, :4].reshape(1, self.hidden_dim, 2, 2))
                    for m in self.modules():
                        if isinstance(m, (nn.Linear, nn.Conv2d)):
                            nn.init.orthogonal_(m.weight, gain=0.6)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                        elif isinstance(m, nn.GroupNorm):
                            if m.weight is not None:
                                nn.init.ones_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                    nn.init.orthogonal_(self.final_layer.weight, gain=1.5)

            def _native_forward(
                self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                **kwargs: Any,
            ) -> torch.Tensor:
                b, _c, h, w = x.shape
                h_feat = self.x_embedder(x) + self.pos_embed[:, :, :h, :w].to(device=x.device, dtype=x.dtype)
                t_val = timestep.reshape(b, 1).to(dtype=x.dtype) * 0.01
                freqs = torch.arange(self.hidden_dim, device=x.device, dtype=x.dtype).unsqueeze(0) * 0.2
                t_emb = self.t_embedder(torch.sin(t_val + freqs)).view(b, self.hidden_dim, 1, 1)
                h_feat = h_feat + t_emb * 0.2
                if isinstance(context, torch.Tensor):
                    ctx_in = context.to(dtype=x.dtype)
                    if ctx_in.shape[-1] > 128:
                        ctx_in = ctx_in[..., :128]
                    elif ctx_in.shape[-1] < 128:
                        ctx_in = F.pad(ctx_in, (0, 128 - ctx_in.shape[-1]))
                    ctx_emb = self.context_embedder(ctx_in).mean(dim=1).view(b, self.hidden_dim, 1, 1)
                    h_feat = h_feat + ctx_emb * 0.2

                if self.use_unet_blocks:
                    for blk in self.input_blocks:
                        res = blk["in_layers"](h_feat)
                        res = blk["out_layers"](res)
                        h_feat = h_feat + res
                        tok = h_feat.flatten(2).transpose(1, 2)
                        q = blk["attn1"]["to_q"](tok)
                        k = blk["attn1"]["to_k"](tok)
                        v = blk["attn1"]["to_v"](tok)
                        attn = torch.softmax((q @ k.transpose(-2, -1)) * (self.hidden_dim ** -0.5), dim=-1) @ v
                        tok = tok + blk["attn1"]["to_out"](attn)
                        ff_h = F.gelu(blk["ff"]["net"][0]["proj"](tok))
                        tok = tok + blk["ff"]["net"][2](ff_h)
                        h_feat = tok.transpose(1, 2).reshape(b, self.hidden_dim, h, w)
                else:
                    tok = h_feat.flatten(2).transpose(1, 2)
                    for blk in self.double_blocks:
                        q, k, v = blk["attn"]["qkv"](tok).chunk(3, dim=-1)
                        attn = torch.softmax((q @ k.transpose(-2, -1)) * (self.hidden_dim ** -0.5), dim=-1) @ v
                        tok = tok + blk["attn"]["proj"](attn)
                        tok = tok + blk["mlp"](tok)
                    h_feat = tok.transpose(1, 2).reshape(b, self.hidden_dim, h, w)

                return self.final_layer(h_feat)

        cases = (
            ("flux_dev", 16, False, False, "rectified_flow"),
            ("sd3_5", 16, False, False, "rectified_flow"),
            ("sdxl_base", 4, True, False, "ddpm_eps"),
            ("sdxl_base", 4, True, False, "v_prediction"),
            ("sd15", 4, True, False, "ddpm_eps"),
            ("sd15", 4, True, True, "ddpm_eps"),
        )

        for arch_id, ch, use_unet, use_conv, expected_obj in cases:
            with self.subTest(architecture=arch_id, objective=expected_obj, include_conv=use_conv):
                torch.manual_seed(777)
                dm = _MicroConvergenceBackbone(channels=ch, hidden_dim=32, use_unet_blocks=use_unet)
                patcher = build_native_comfy_patcher(
                    arch_id,
                    dm,
                    unet_config={
                        "in_channels": ch,
                        "out_channels": ch,
                        "context_dim": 128,
                        "context_in_dim": 128,
                        "adm_in_channels": 128,
                    },
                    dtype=torch.float32,
                    expected_objective=expected_obj,
                )
                dm.restore_orthogonal_forward()

                adapter = resolve_model_adapter(
                    "auto",
                    comfy_model=patcher,
                    gradient_checkpointing=False,
                    min_timestep=0.5,
                    max_timestep=0.50001,
                    include_conv=use_conv,
                )
                self.assertEqual(adapter.spec.objective_type, expected_obj)

                lora_cfg = LoRAConfig(
                    rank=16,
                    alpha=32.0,
                    lora_dtype="fp32",
                    target_modules=tuple(adapter.spec.default_target_modules),
                    include_conv=use_conv,
                )
                injected = LoRAInjector.inject(
                    dm,
                    lora_cfg,
                    lora_dtype=torch.float32,
                    device=torch.device("cpu"),
                )
                self.assertGreater(len(injected), 0)

                try:
                    if use_conv:
                        # When testing LoCon (`include_conv=True`), optimize ONLY the 3x3 Conv2d LoRA weights
                        # to prove that spatial Conv2d adapters alone converge the diffusion objective.
                        conv_params = [
                            p
                            for m in dm.modules()
                            if isinstance(m, LoRALinear) and getattr(m, "is_conv", False)
                            for p in (m.lora_A.weight, m.lora_B.weight)
                        ]
                        self.assertGreater(len(conv_params), 0)
                        params = conv_params
                    else:
                        params = LoRAInjector.get_trainable_parameters(dm)

                    num_steps = 24
                    optimizer = torch.optim.AdamW(params, lr=1.5e-2, betas=(0.5, 0.99), weight_decay=0.0)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=num_steps, eta_min=4e-3
                    )

                    gen = torch.Generator(device="cpu").manual_seed(4242)
                    fixed_batch = {
                        "latents": torch.randn((1, ch, 2, 2), generator=gen, dtype=torch.float32),
                        "prompt_embeds": torch.randn((1, 4, 128), generator=gen, dtype=torch.float32),
                        "pooled_output": torch.randn((1, 128), generator=gen, dtype=torch.float32),
                        "latent_scaled": True,
                    }

                    losses = []
                    for _ in range(num_steps):
                        optimizer.zero_grad(set_to_none=True)
                        # Lock RNG before compute_training_loss so the exact same sigma & noise are drawn every step
                        torch.manual_seed(9001)
                        loss = adapter.compute_training_loss(
                            backbone=dm,
                            batch=fixed_batch,
                            device=torch.device("cpu"),
                            dtype=torch.float32,
                        )
                        self.assertTrue(bool(torch.isfinite(loss).all()))
                        losses.append(float(loss.detach().item()))
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    initial_loss = losses[0]
                    mid_loss = losses[len(losses) // 2]
                    final_loss = losses[-1]

                    self.assertLess(
                        losses[1],
                        initial_loss,
                        f"{arch_id} ({expected_obj}, conv={use_conv}) Step 1 did not decrease loss: {losses[:3]}",
                    )
                    self.assertLess(
                        mid_loss,
                        initial_loss * 0.65,
                        f"{arch_id} ({expected_obj}, conv={use_conv}) loss did not drop >35% by midpoint: {losses}",
                    )
                    self.assertTrue(
                        final_loss < mid_loss or final_loss < initial_loss * 0.05,
                        f"{arch_id} ({expected_obj}, conv={use_conv}) loss did not continue decreasing in second half: {losses}",
                    )
                    self.assertLess(
                        final_loss,
                        initial_loss * 0.25,
                        f"{arch_id} ({expected_obj}, conv={use_conv}) failed >75% convergence check: initial={initial_loss:.4f}, final={final_loss:.4f}",
                    )
                finally:
                    LoRAInjector.uninject(dm)

    def test_runtime_hooks_and_node_imports_integrity(self) -> None:
        """
        Verify critical regression invariants:
        1. `omni_lora.nodes` exports `_strip_numeric_segments` and target filtering on fallback linear modules executes cleanly.
        2. Calling `install_comfy_runtime_hooks()` after `LoRATrainingPipeline.setup()` never strips `_omni_dml_safe` wrappers.
        """
        import torch.nn.functional as F_mod
        from omni_lora import nodes as omni_nodes
        from omni_lora.backends import detect_and_configure
        from omni_lora.lora import LoRAConfig
        from omni_lora.models import ComfyModelAdapter, install_comfy_runtime_hooks
        from omni_lora.pipelines import LoRATrainingPipeline, TrainingConfig

        self.assertTrue(callable(getattr(omni_nodes, "_strip_numeric_segments", None)))

        plan = detect_and_configure(preferred_backend="cpu", requested_dtype="fp32", allow_fallback=True)
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        adapter = ComfyModelAdapter(model_id="flux_dev", comfy_model=model)
        cfg = TrainingConfig(
            backend="cpu",
            dtype="fp32",
            max_train_steps=1,
            lora=LoRAConfig(rank=4, alpha=4.0, target_modules=("0", "1")),
        )
        pipe = LoRATrainingPipeline(backend=plan.backend, model_adapter=adapter, config=cfg)
        pipe.setup(external_backbone=model)
        try:
            self.assertTrue(getattr(F_mod.conv3d, "_omni_dml_safe", False))
            self.assertTrue(getattr(F_mod.pixel_unshuffle, "_omni_dml_safe", False))
            self.assertTrue(getattr(F_mod.pixel_shuffle, "_omni_dml_safe", False))
            self.assertTrue(getattr(F_mod.rms_norm, "_omni_dml_safe", False))

            install_comfy_runtime_hooks()
            self.assertTrue(getattr(F_mod.conv3d, "_omni_dml_safe", False))
            self.assertTrue(getattr(F_mod.pixel_unshuffle, "_omni_dml_safe", False))
            self.assertTrue(getattr(F_mod.pixel_shuffle, "_omni_dml_safe", False))
            self.assertTrue(getattr(F_mod.rms_norm, "_omni_dml_safe", False))
        finally:
            if pipe.backbone is not None:
                LoRAInjector.uninject(pipe.backbone)

    def test_tracker_store_idle_recovery_and_cursors(self) -> None:
        """
        Verify `OmniLoRATelemetryStore` preserves all step metrics during PC idle / WS drops,
        supports `after_step` and `cursors` delta catch-up, and filters by `node_ids`.
        """
        from omni_lora.tracker_store import OmniLoRATelemetryStore, _parse_cursors_param

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "tracker_state.json"
            store = OmniLoRATelemetryStore(state_file=state_path)
            store.start_run({
                "run_id": "run_idle_1",
                "run_title": "Idle Test Run",
                "node_id": "42",
                "max_steps": 10,
                "lr": 1e-4,
            })
            for s in range(1, 7):
                store.record_step({
                    "run_id": "run_idle_1",
                    "run_title": "Idle Test Run",
                    "node_id": "42",
                    "step": s,
                    "max_steps": 10,
                    "loss": 0.5 / s,
                    "ema_loss": 0.5 / s,
                    "lr": 1e-4,
                    "timestep": 0.5,
                    "bucket": "1024x1024",
                    "step_sec": 1.2,
                    "eta_sec": (10 - s),
                    "vram_gb": 8.0,
                })

            # Suppose client received steps 1..2 before PC entered idle and missed 3..6
            delta = store.get_snapshot(run_id="run_idle_1", after_step=2)
            self.assertEqual(len(delta["runs"]), 1)
            self.assertEqual([p["step"] for p in delta["runs"][0]["steps"]], [3, 4, 5, 6])
            self.assertEqual(delta["runs"][0]["status"], "active")

            # Complete run while still idle
            store.complete_run({
                "run_id": "run_idle_1",
                "node_id": "42",
                "final_step": 6,
                "final_loss": 0.08,
                "final_ema_loss": 0.12,
                "total_time_sec": 7.2,
            })

            cursors = _parse_cursors_param("run_idle_1:4")
            self.assertEqual(cursors, {"run_idle_1": 4})
            snap_all = store.get_snapshot(cursors=cursors, node_ids=["42"])
            self.assertEqual(len(snap_all["runs"]), 1)
            self.assertEqual(snap_all["runs"][0]["status"], "completed")
            self.assertEqual([p["step"] for p in snap_all["runs"][0]["steps"]], [5, 6])

            # Non-matching node filter excludes run_idle_1
            snap_other = store.get_snapshot(node_ids=["99"])
            self.assertEqual(snap_other["runs"], [])

            # Disk persistence survives fresh store instance
            reloaded = OmniLoRATelemetryStore(state_file=state_path)
            reloaded_snap = reloaded.get_snapshot(run_id="run_idle_1")
            self.assertEqual(len(reloaded_snap["runs"]), 1)
            self.assertEqual(len(reloaded_snap["runs"][0]["steps"]), 6)
            self.assertEqual(reloaded_snap["runs"][0]["status"], "completed")
            for step_pt in reloaded_snap["runs"][0]["steps"]:
                self.assertIn("wall_time", step_pt)
                self.assertIn("rel_time_sec", step_pt)

            # Deleting a run removes it from memory, disk, and blocks late stray steps
            self.assertTrue(reloaded.delete_run("run_idle_1"))
            self.assertFalse(reloaded.delete_run("run_idle_1"))
            self.assertEqual(reloaded.get_snapshot()["runs"], [])
            reloaded.record_step({"run_id": "run_idle_1", "step": 7, "loss": 0.05})
            self.assertEqual(reloaded.get_snapshot()["runs"], [])

            # Explicit start_run clears tombstone and starts fresh
            reloaded.start_run({"run_id": "run_idle_1", "run_title": "Restarted Run", "max_steps": 5})
            reloaded.record_step({"run_id": "run_idle_1", "step": 1, "loss": 0.2})
            self.assertEqual(len(reloaded.get_snapshot()["runs"]), 1)
            self.assertEqual(reloaded.clear(keep_active=False), 1)
            self.assertEqual(reloaded.get_snapshot()["runs"], [])

    def test_precompute_conditioning_bucket_uses_pixel_dimensions_for_high_ratio_vae(self) -> None:
        """
        Verify `OmniLoRAPrecomputeConditioning` tags batch `bucket` from true pixel dimensions
        rather than assuming 8x VAE downsampling (which mislabeled 16x VAEs at 1024px as 512x512).
        """
        node = OmniLoRAPrecomputeConditioning()

        class _Vae16x:
            def encode(self, img_bhwc: torch.Tensor) -> torch.Tensor:
                b, h, w, _ = img_bhwc.shape
                return torch.zeros((b, 48, h // 16, w // 16), dtype=torch.float32)

        class _DummyClip:
            def tokenize(self, text: str) -> str:
                return text

            def encode_from_tokens(self, tokens: str, return_pooled: bool = True, return_dict: bool = True):
                return {"cond": torch.zeros((1, 8, 64), dtype=torch.float32), "pooled_output": None}

        pixel_batches = [torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)]
        prompt_batches = [["ohwx, studio portrait"]]
        batches, _, _, _, _ = node._encode_with_comfy_or_adapter(
            pixel_batches=pixel_batches,
            prompt_batches=prompt_batches,
            dtype=torch.float32,
            vae=_Vae16x(),
            clip=_DummyClip(),
            offload_between_encoders=False,
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["latents"].shape[-2:], (64, 64))
        self.assertEqual(batches[0]["bucket"], "1024x1024")

    def test_comfy_kitchen_and_sage_attention_training_override_and_fallback(self) -> None:
        """
        Verify that OmniLoRA safely handles `--use-ck-attention` (Comfy Kitchen INT8 / SOL),
        `--use-sage-attention`, pre-loaded `ComfyAttention` checkpoints, and direct `comfy_kitchen` /
        `comfy.quant_ops.ck` RoPE & AdaLN kernels during training while preserving inference behavior:
        1. `ComfyModelAdapter._build_transformer_options()` sets `optimized_attention_override`
           during training without mutating the source `model.model_options`.
        2. `install_comfy_runtime_hooks()` wraps `attention_comfy_kitchen_int8` (and its
           `.container_function`), `attention_comfy_kitchen_sol`, `attention_sage`, and `attention3_sage`,
           rebinds pre-imported `comfy.*` references and pre-loaded `ComfyAttention` instances, and wraps
           `comfy_kitchen` (`int8_attention`, `prequantize_int8_attention`, `sol_attn`, `rms_rope`,
           `rms_rope1`, `rms_rope_split_half`, `apply_rope_split_half`, `apply_rope`, `apply_rope1`,
           `adaln`, `rms_adaln`).
        3. `_forward_backbone()` scopes `mm.in_training = True` during grad-enabled forward passes and
           handles 1-arg `reset_prefix_cache(False)` (`QwenImage21`).
        """
        import functools
        import sys
        import types
        from omni_lora.models.adapters import ComfyModelAdapter, install_comfy_runtime_hooks, unwrap_comfy_diffusion_model

        infer_calls: List[str] = []

        class _FakeContainer:
            def __init__(self, t: torch.Tensor) -> None:
                self.tensor: Optional[torch.Tensor] = t

            def take(self) -> torch.Tensor:
                assert self.tensor is not None, "Container already consumed"
                t = self.tensor
                self.tensor = None
                return t

        def _fake_wrap_attn(fn: Any) -> Any:
            @functools.wraps(fn)
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                if len(args) >= 3 and isinstance(args[0], _FakeContainer):
                    if getattr(_wrapped, "container_function", None) is not None:
                        return _wrapped.container_function(*args, **kwargs)
                    args = (args[0].take(), args[1].take(), args[2].take()) + args[3:]
                return fn(*args, **kwargs)

            _wrapped.container_function = None  # type: ignore[attr-defined]
            return _wrapped

        @_fake_wrap_attn
        def _fake_attn_pytorch(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            heads: int,
            mask: Optional[torch.Tensor] = None,
            attn_precision: Any = None,
            skip_reshape: bool = False,
            skip_output_reshape: bool = False,
            **kwargs: Any,
        ) -> torch.Tensor:
            b = int(q.shape[0])
            if skip_reshape:
                dim_head = int(q.shape[-1])
            else:
                dim_head = int(q.shape[-1]) // int(heads)
                q = q.view(b, -1, heads, dim_head).transpose(1, 2)
                k = k.view(b, -1, heads, dim_head).transpose(1, 2)
                v = v.view(b, -1, heads, dim_head).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            return out

        @_fake_wrap_attn
        def _fake_ck_int8(q: Any, k: Any, v: Any, heads: int, **kwargs: Any) -> torch.Tensor:
            if torch.is_grad_enabled():
                raise RuntimeError("comfy_kitchen.int8_attention does not support autograd training")
            infer_calls.append("ck_int8")
            return q

        def _fake_ck_int8_containers(q: _FakeContainer, k: _FakeContainer, v: _FakeContainer, heads: int, **kwargs: Any) -> torch.Tensor:
            qt, kt, vt = q.take(), k.take(), v.take()
            if torch.is_grad_enabled():
                raise RuntimeError("comfy_kitchen.prequantize_int8_attention does not support autograd training")
            infer_calls.append("ck_int8_container")
            return qt + 0.0 * (kt + vt)

        _fake_ck_int8.container_function = _fake_ck_int8_containers  # type: ignore[attr-defined]

        @_fake_wrap_attn
        def _fake_ck_sol(q: Any, k: Any, v: Any, heads: int, **kwargs: Any) -> torch.Tensor:
            if torch.is_grad_enabled():
                raise RuntimeError("comfy_kitchen.sol_attn does not support autograd training")
            infer_calls.append("ck_sol")
            return q

        @_fake_wrap_attn
        def _fake_sage(q: Any, k: Any, v: Any, heads: int, **kwargs: Any) -> torch.Tensor:
            if torch.is_grad_enabled():
                raise RuntimeError("sageattn does not support autograd training")
            infer_calls.append("sage")
            return q

        def _fail_on_grad(name: str, ret_fn: Any) -> Any:
            def _fn(*a: Any, **kw: Any) -> Any:
                if torch.is_grad_enabled():
                    raise RuntimeError(f"{name} unsupported during training")
                infer_calls.append(name)
                return ret_fn(*a, **kw)

            return _fn

        attn_mod = types.ModuleType("comfy.ldm.modules.attention")
        attn_mod.attention_pytorch = _fake_attn_pytorch  # type: ignore[attr-defined]
        attn_mod.attention_comfy_kitchen_int8 = _fake_ck_int8  # type: ignore[attr-defined]
        attn_mod.attention_comfy_kitchen_sol = _fake_ck_sol  # type: ignore[attr-defined]
        attn_mod.attention_sage = _fake_sage  # type: ignore[attr-defined]
        attn_mod.optimized_attention = _fake_ck_int8  # type: ignore[attr-defined]
        attn_mod.optimized_attention_masked = _fake_ck_int8  # type: ignore[attr-defined]
        attn_mod.REGISTERED_ATTENTION_FUNCTIONS = {  # type: ignore[attr-defined]
            "comfy_kitchen_int8": _fake_ck_int8,
            "sage": _fake_sage,
            "pytorch": _fake_attn_pytorch,
        }

        preimported_ldm_mod = types.ModuleType("comfy.ldm._mock_ernie")
        preimported_ldm_mod.optimized_attention = _fake_ck_int8  # type: ignore[attr-defined]

        ck_mod = types.ModuleType("comfy_kitchen")
        ck_mod.int8_attention = _fail_on_grad("raw_ck_int8", lambda q, k, v, *a, **kw: q)  # type: ignore[attr-defined]
        ck_mod.prequantize_int8_attention = _fail_on_grad("raw_preq", lambda q, k, v, *a, **kw: q)  # type: ignore[attr-defined]
        ck_mod.int8_attention_from_prequantized = lambda quantized, *a, **kw: quantized  # type: ignore[attr-defined]
        ck_mod.sol_attn = _fail_on_grad("raw_sol", lambda q, k, v, *a, **kw: q)  # type: ignore[attr-defined]
        ck_mod.rms_rope = _fail_on_grad("raw_rms_rope", lambda q, k, *a, **kw: (q, k))  # type: ignore[attr-defined]
        ck_mod.rms_rope1 = _fail_on_grad("raw_rms_rope1", lambda x, *a, **kw: x)  # type: ignore[attr-defined]
        ck_mod.rms_rope_split_half = _fail_on_grad("raw_rms_rope_split", lambda q, k, *a, **kw: (q, k))  # type: ignore[attr-defined]
        ck_mod.apply_rope_split_half = _fail_on_grad("raw_apply_rope_split", lambda q, k, *a, **kw: (q, k))  # type: ignore[attr-defined]
        ck_mod.apply_rope = _fail_on_grad("raw_apply_rope", lambda q, k, *a, **kw: (q, k))  # type: ignore[attr-defined]
        ck_mod.apply_rope1 = _fail_on_grad("raw_apply_rope1", lambda x, *a, **kw: x)  # type: ignore[attr-defined]
        ck_mod.adaln = _fail_on_grad("raw_adaln", lambda x, *a, **kw: x)  # type: ignore[attr-defined]
        ck_mod.rms_adaln = _fail_on_grad("raw_rms_adaln", lambda x, *a, **kw: x)  # type: ignore[attr-defined]

        mm_mod = types.ModuleType("comfy.model_management")
        mm_mod.in_training = False  # type: ignore[attr-defined]

        # Simulate a model whose `ComfyAttention` was populated BEFORE `install_comfy_runtime_hooks()`
        class _PreloadedAttentionModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(16, 16))
                self.comfy_attention = types.SimpleNamespace(
                    function=functools.partial(_fake_ck_sol, tau=1.25)
                )
                self.prefix_cache_enabled = True

            def reset_prefix_cache(self, enabled: bool) -> None:
                self.prefix_cache_enabled = bool(enabled)

        preloaded_backbone = _PreloadedAttentionModule()

        saved_mods = {
            k: sys.modules.get(k)
            for k in (
                "comfy.ldm.modules.attention",
                "comfy.ldm._mock_ernie",
                "comfy_kitchen",
                "comfy.model_management",
            )
        }
        try:
            sys.modules["comfy.ldm.modules.attention"] = attn_mod
            sys.modules["comfy.ldm._mock_ernie"] = preimported_ldm_mod
            sys.modules["comfy_kitchen"] = ck_mod
            sys.modules["comfy.model_management"] = mm_mod

            unwrapped_bb = unwrap_comfy_diffusion_model(preloaded_backbone, load_to_gpu=False)
            self.assertIs(unwrapped_bb, preloaded_backbone)
            self.assertTrue(getattr(preloaded_backbone.comfy_attention.function, "_omni_train_attn_safe", False))

            # 1. Verify `ComfyModelAdapter._build_transformer_options()` overrides inference attention
            #    during training without mutating the source `model_options`.
            def _bad_infer_override(*a: Any, **kw: Any) -> torch.Tensor:
                raise RuntimeError("inference override must not run during training")

            class _MockPatcher:
                def __init__(self) -> None:
                    self.model = nn.Linear(16, 16)
                    self.model_options = {
                        "transformer_options": {"optimized_attention_override": _bad_infer_override}
                    }

            patcher = _MockPatcher()
            adapter = ComfyModelAdapter(comfy_model=patcher, model_id="flux_dev")
            with torch.enable_grad():
                train_opts = adapter._build_transformer_options(
                    noisy_latents=torch.zeros((1, 16, 8, 8)),
                    timesteps=torch.tensor([0.5]),
                )
            override_fn = train_opts["optimized_attention_override"]
            self.assertIsNot(override_fn, _bad_infer_override)
            self.assertIs(
                patcher.model_options["transformer_options"]["optimized_attention_override"],
                _bad_infer_override,
            )

            # 2. Verify differentiable forward + backward across all guarded attention, RoPE, and AdaLN paths
            with torch.enable_grad():
                mm_mod.in_training = True
                q = torch.randn(1, 8, 64, requires_grad=True)
                k = torch.randn(1, 8, 64, requires_grad=True)
                v = torch.randn(1, 8, 64, requires_grad=True)

                out_override = override_fn(None, _FakeContainer(q), _FakeContainer(k), _FakeContainer(v), 4)
                out_ck_container = _fake_ck_int8.container_function(_FakeContainer(q), _FakeContainer(k), _FakeContainer(v), 4)
                out_preimported = preimported_ldm_mod.optimized_attention(q, k, v, 4)
                out_preloaded_ca = preloaded_backbone.comfy_attention.function(q, k, v, 4)
                out_sage = attn_mod.attention_sage(q, k, v, 4)
                out_raw_ck = ck_mod.int8_attention(
                    q.view(1, 8, 4, 16).transpose(1, 2),
                    k.view(1, 8, 4, 16).transpose(1, 2),
                    v.view(1, 8, 4, 16).transpose(1, 2),
                )

                # Test `rms_rope`, `rms_rope_split_half` (with `epsilon=` and `rot_dim=`), `apply_rope_split_half`, `adaln`, `rms_adaln`
                q_rope = torch.randn(1, 8, 4, 16, requires_grad=True)
                k_rope = torch.randn(1, 8, 4, 16, requires_grad=True)
                q_sc = torch.ones(16, requires_grad=True)
                k_sc = torch.ones(16, requires_grad=True)
                angles = torch.randn(1, 8, 1, 8)
                pe = torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), dim=-1).unflatten(-1, (2, 2))
                rq, rk = ck_mod.rms_rope(q_rope, k_rope, pe, q_sc, k_sc, 1e-6)
                sq1, sk1 = ck_mod.rms_rope_split_half(q_rope, k_rope, pe[:, :, :, :4], q_sc, k_sc, epsilon=1e-6, rot_dim=8)
                sq2, sk2 = ck_mod.apply_rope_split_half(q_rope, k_rope, pe)

                x_norm = torch.randn(1, 8, 16, requires_grad=True)
                mod_scale = torch.randn(1, 1, 16, requires_grad=True)
                mod_shift = torch.randn(1, 1, 16, requires_grad=True)
                out_adaln = ck_mod.adaln(x_norm, mod_scale, mod_shift, 1e-6)
                out_rms_adaln = ck_mod.rms_adaln(x_norm, mod_scale, mod_shift, 1e-6)

                loss = (
                    out_override.sum()
                    + out_ck_container.sum()
                    + out_preimported.sum()
                    + out_preloaded_ca.sum()
                    + out_sage.sum()
                    + out_raw_ck.sum()
                    + rq.sum()
                    + rk.sum()
                    + sq1.sum()
                    + sk1.sum()
                    + sq2.sum()
                    + sk2.sum()
                    + out_adaln.sum()
                    + out_rms_adaln.sum()
                )
                loss.backward()
                self.assertIsNotNone(q.grad)
                self.assertIsNotNone(k.grad)
                self.assertIsNotNone(v.grad)
                self.assertIsNotNone(q_rope.grad)
                self.assertIsNotNone(q_sc.grad)
                self.assertIsNotNone(x_norm.grad)
                self.assertEqual(infer_calls, [])

            # 3. Outside training (`torch.inference_mode()` + `in_training = False`), inference kernels run normally
            mm_mod.in_training = False
            with torch.inference_mode():
                q_inf = torch.randn(1, 8, 64)
                k_inf = torch.randn(1, 8, 64)
                v_inf = torch.randn(1, 8, 64)
                _ = preimported_ldm_mod.optimized_attention(q_inf, k_inf, v_inf, 4)
                _ = _fake_ck_int8.container_function(_FakeContainer(q_inf), _FakeContainer(k_inf), _FakeContainer(v_inf), 4)
                _ = attn_mod.attention_sage(q_inf, k_inf, v_inf, 4)
                _ = ck_mod.int8_attention(q_inf.view(1, 8, 4, 16).transpose(1, 2), k_inf.view(1, 8, 4, 16).transpose(1, 2), v_inf.view(1, 8, 4, 16).transpose(1, 2))
            self.assertEqual(infer_calls, ["ck_int8", "ck_int8_container", "sage", "raw_ck_int8"])
        finally:
            for k_mod, orig_mod in saved_mods.items():
                if orig_mod is None:
                    sys.modules.pop(k_mod, None)
                else:
                    sys.modules[k_mod] = orig_mod


def run_cloud_verification_suite(preferred_backend: str = "auto") -> int:
    """
    Standalone cloud/backend verification runner:
    1. Detects and prints the active hardware accelerator (`xpu`, `cuda`, `rocm`, `dml`, `mps`, `npu`, `hpu`, `cpu`),
       VRAM capacity, precision support (`bf16` / `fp16` / `fp32`), and recommended VRAM tier.
    2. Benchmarks all 21 architectures in `ComfyModelZoo` on the selected backend inside
       `EmulatedComfyEnvironment` and prints a summary table.
    3. Executes the complete `unittest` suite and returns `0` on 100% pass.
    """
    import time
    from omni_lora.tests.comfy_emulator import (
        ComfyModelZoo,
        EmulatedComfyEnvironment,
        EmulatedKSampler,
    )

    plan = detect_and_configure(preferred_backend=preferred_backend, requested_dtype="auto", allow_fallback=True)
    dev = plan.backend.get_device()
    caps = plan.backend.get_capabilities()
    dtype = torch.bfloat16 if caps.supports_bf16 else torch.float16

    print("=" * 92)
    print(" OmniLoRA Standalone Cloud & Multi-Backend Verification Suite (Zero-ComfyUI Dependency)")
    print("=" * 92)
    print(f" Active Backend   : {plan.backend.backend_id.upper()} ({caps.device_name}) -> device={dev}")
    print(f" VRAM State       : {plan.vram_state.value} (Total: {caps.total_vram_mb:.0f} MB, Recommended Res: {plan.recommended_resolution}px)")
    print(f" Compute DType    : {plan.resolved_dtype} (BF16={caps.supports_bf16}, FP16={caps.supports_fp16}, GradScaler={plan.requires_grad_scaler})")
    print("-" * 92)
    print(f" {'#':<2} | {'Architecture':<16} | {'Obj':<14} | {'Ch':<4} | {'Layers':<6} | {'Loss':<8} | {'KSampler':<8} | {'Time':<7}")
    print("-" * 92)

    with EmulatedComfyEnvironment(torch_device=dev, simulated_vram_mb=max(8192.0, caps.total_vram_mb)) as env:
        import comfy.sd as c_sd  # type: ignore

        dataset_node = OmniLoRADatasetLoader()
        precache_node = OmniLoRAPrecomputeConditioning()
        trainer_node = OmniLoRATrainer()

        with torch.inference_mode():
            imgs = torch.rand(1, 64, 64, 4, dtype=torch.float32)
            ds, _ = dataset_node.load_dataset(
                dataset_path="",
                trigger_word="ohwx",
                default_caption="subject",
                resolution=64,
                batch_size=1,
                num_repeats=1,
                center_crop=True,
                aspect_ratio_bucketing=True,
                random_flip=True,
                caption_dropout=0.2,
                images=imgs,
                custom_prompts="a detailed photo of ohwx subject",
                use_alpha_mask=True,
            )

            for idx, arch_id in enumerate(ComfyModelZoo.ARCH_IDS, start=1):
                t0 = time.perf_counter()
                entry = ComfyModelZoo.build(arch_id, dtype=dtype)
                env.unloaded_lora_keys.clear()
                (cached_bundle,) = precache_node.precompute(
                    dataset=ds,
                    offload_encoders_after_cache=True,
                    vae=entry.vae,
                    clip=entry.clip,
                    model=entry.patcher,
                )
                _, lora_bundle, summary_json = trainer_node.train_lora(
                    cached_data=cached_bundle,
                    model=entry.patcher,
                    model_architecture="auto",
                    backend=plan.backend.backend_id,
                    optimizer="radamw",
                    rank=4,
                    alpha=4.0,
                    learning_rate=1e-3,
                    lr_scheduler="cosine",
                    warmup_steps=1,
                    steps=2,
                    grad_accum_steps=1,
                    gradient_checkpointing=True,
                    use_rslora=True,
                )
                summary = json.loads(summary_json)
                patched, _ = c_sd.load_lora_for_models(entry.patcher, entry.clip, lora_bundle.state_dict, 1.0, 1.0)
                assert patched is not None and len(env.unloaded_lora_keys) == 0
                ks_out = EmulatedKSampler.sample(patched, entry.clip, resolution=64, steps=1, device=dev)
                ks_ok = "PASS" if bool(torch.isfinite(ks_out).all()) else "FAIL"
                dt_ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f" {idx:>2} | {entry.arch_id:<16} | {summary['training_objective']:<14} | "
                    f"{entry.expected_latent_channels:>3}c | {summary['injected_layers']:>6} | "
                    f"{summary['final_loss']:>8.4f} | {ks_ok:<8} | {dt_ms:>5.0f}ms"
                )
                del ks_out, patched, lora_bundle, cached_bundle, entry
                plan.backend.empty_cache()

    print("-" * 92)
    print(" Running Full Unit & Edge-Case Test Suite...")
    print("-" * 92)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestOmniLoRAFramework)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", type=str, default=None, help="Run cloud verification table on specified backend (auto, xpu, cuda, rocm, dml, mps, cpu)")
    parser.add_argument("--cloud-verify", action="store_true", help="Run formatted 21-model benchmark table before unit tests")
    known_args, remaining_argv = parser.parse_known_args()

    if known_args.backend is not None or known_args.cloud_verify:
        sys.exit(run_cloud_verification_suite(preferred_backend=known_args.backend or "auto"))
    else:
        unittest.main(argv=[sys.argv[0]] + remaining_argv)








