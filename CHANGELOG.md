# Changelog

All notable changes to **ComfyUI-OmniLoRA** are documented in this file.
Versioning follows [Semantic Versioning](https://semver.org/) and is tracked via Git tags (`v1.0.x`) on `master` for the Comfy Registry and ComfyUI Manager.

---

## [1.0.3] - 2026-04-12

### Added
- **MiniMax H3 Dual-Stream Audio-Video Training Support (`MiniMaxH3Model` / `MiniMaxH3AV`)**:
  - Prioritize `backbone.latents_dim` (`24` video channels on `Sol-H3-VAE` / `MiniMaxH3Video`) over `MiniMaxH3AV.latent_channels` (`32`, the max across video and audio streams) during latent channel inference and strict channel validation.
  - Automatically synthesize a companion silent audio latent stream `(B, audio_latents_dim, 2, audio_t)` using ComfyUI's `MiniMaxH3AV` frame-to-audio temporal scaling (`frames_per_token = (1, 4, 4, 4, 4)`, `audio_frame_rescale = 5.0 / 3.0`) when training `MiniMaxH3Model` on single-stream video or image datasets.
  - Extract the primary video velocity prediction `out[0]` from dual-stream `[-video_out, -audio_out]` outputs when `latent_shapes` is `None` instead of flattening multi-modal outputs via `comfy.utils.pack_latents`.
  - Add autograd-safe out-of-place overrides for `_mod_scale_shift` and `_mod_gate` in `comfy.ldm.minimax.model` while `torch.is_grad_enabled()` is active so `RMSNorm` inputs and outputs are never mutated in-place during backpropagation.
- **ROCm & Gradient Checkpointing Enhancements** (PR [#2](https://github.com/cyberterm/ComfyUI-OmniLoRA/pull/2) by [@0xDELUXA](https://github.com/0xDELUXA)):
  - Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` automatically at import time on AMD ROCm (`torch.version.hip`) to enable experimental AOTriton SDPA kernels on Windows and Linux ROCm builds.
  - Restore CPU-offloaded 5D (`[B, C, T, H, W]`) activation checkpoint inputs back to the active accelerator device during recomputation when `cpu_offload_activations=True`.

### Fixed
- **Flux.2 Klein (`flux-2-klein-base-9b-fp8`) / Flux.1 / Chroma / MageFlow 4D vs 5D `process_img` Disambiguation**:
  - Fixed `ValueError: too many values to unpack (expected 4)` in `Flux._forward` / `Flux.process_img` when `patch_size=1` and `in_channels == out_channels == 128` (Flux.2 128-channel latent format).
  - Updated `ComfyModelAdapter._infer_backbone_accepts_5d` to inspect `backbone.process_img` and `backbone._forward` bytecode (`co_varnames`) so 4D image models (`Flux`, `Flux2`, `Chroma`, `MageFlow`) stay 4D `[B, C, H, W]` while 5D backbones (`QwenImageTransformer2DModel`, `Krea2` `SingleStreamDiT`) continue receiving 5D `[B, C, T, H, W]` latents.
- **`ModelSamplingDiscreteFlow` Fallback**:
  - Guarded `sampling.noise_scaling` and `sampling.calculate_input` in `ComfyModelAdapter.compute_training_loss` with standard `CONST` / `EPS` analytical fallbacks when a `model_sampling` instance does not mix in `comfy.model_sampling.CONST`.

---

## [1.0.2] - 2026-04-11

### Added
- **Comfy Kitchen (`--use-ck-attention`) & SageAttention (`--use-sage-attention`) Training Guards**:
  - Automatically intercept inference-only `comfy_kitchen` / `sage` attention and custom `comfy.quant_ops.ck` kernels (`apply_rope`, `apply_rope1`, `rms_norm`, `adaln`, `rms_rope_split_half`) during training and route them through differentiable PyTorch operations, restoring the user's inference backend after training finishes.
- **Anima / Cosmos Predict2 `extra_conds` Support**:
  - Preserve autograd tracking when preprocessing model-specific `extra_conds` (`t5xxl_ids`, `t5xxl_weights`, `crossattn_mask`) and coerce integer token ID tensors vs floating-point weight tensors accurately.

### Fixed
- Ignore runtime `.loss_tracker_history.json` telemetry files in `.gitignore`.

---

## [1.0.1] - 2026-04-10

### Fixed
- Refactored subprocess and file-handling helpers to eliminate Comfy Registry YARA security scanner false positives.

---

## [1.0.0] - 2026-04-09

### Added
- Initial stable release of **ComfyUI-OmniLoRA**:
  - 6 native ComfyUI custom nodes (`OmniLoRADatasetLoader`, `OmniLoRAPrecomputeConditioning`, `OmniLoRATrainer`, `OmniLoRALoader`, `OmniLoRASaveLoRA`, `OmniLoRALossTracker`) + interactive canvas Web UI extension.
  - Universal runtime model introspection supporting 50 ComfyUI `comfy.ldm.*` diffusion, Rectified Flow, Flow-Map, and video/audio architectures.
  - Direct training on FP8, INT8 `QuantizedTensor`, `ComfyUI-GGUF` (`GGMLTensor`), and `bitsandbytes` backbones with deterministic VRAM budgeting and on-demand CPU layer streaming across CUDA, Intel Arc XPU, AMD ROCm, Apple MPS, Windows DirectML, Cloud TPU, NPU, HPU, and CPU.
