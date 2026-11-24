# Changelog

All notable changes to **ComfyUI-OmniLoRA** are documented in this file.
Versioning follows [Semantic Versioning](https://semver.org/) and is tracked via Git tags (`v1.x.y`) on `master` for the Comfy Registry and ComfyUI Manager.

---

## [1.2.0] - 2026-11-24

### Added
- **First-Class Google Cloud TPU (`torch.tpu` / `torch_xla`) Support & Automatic ComfyUI Hook**:
  - Upgraded `TPUBackend` (`backends/tpu.py`) to prefer Google's native `torch.tpu` (`torch-tpu` / `tpu:0` via `torch.accelerator`) with legacy `torch_xla` (`xla:0`) fallback, plus `TPUBackend.optimize_module()` `openxla` compilation support.
  - Added `install_comfy_tpu_hooks()` (auto-invoked on import in `__init__.py`, opt-out via `OMNI_LORA_DISABLE_COMFY_TPU_HOOK=1`), automatically upgrading vanilla ComfyUI launched with `--cpu --use-pytorch-cross-attention` to `tpu:0` in native `bfloat16` (no external boot wrapper required), switching attention to `attention_pytorch`, deduplicating `absl` root log handlers, and defaulting recurrent LLM text encoders to CPU (`OMNI_LORA_TPU_TEXT_ENCODER=1` to opt in).
  - Verified **44/44 unmocked ComfyUI architectures** and live **5.15B `moss_image_base_bf16`** training on Cloud TPU v5e-1 (`~4.4 s/step`, 100% HBM-resident).
- **Portable `.safetensors` Disk Pre-Cache & Lazy `VAE`/`CLIP` Skipping (`conditioning_cache.py`, `nodes.py`)**:
  - Added `disk_cache` (`"auto (load or save)"` default, `"off"`, `"rebuild (overwrite)"`) and optional `model` input to `OmniLoRAPrecomputeConditioning`, saving unique pre-encoded latents, text embeddings, masks, control latents, and `uncond_*` tensors into `<dataset_path>/.omni_lora_cache/` using portable content fingerprints (relative filenames + byte sizes + captions — transfers cleanly from Windows to Colab/Linux).
  - Marked `vae` and `clip` as ComfyUI `lazy: True` with `check_lazy_status()` so `VAELoader` and `CLIPLoader` are **completely skipped** on cache hits (zero disk read, zero RAM, zero encoding), and added `OUTPUT_NODE = True` + standalone [`example_workflows/omni_lora_precache_only_workflow.json`](./example_workflows/omni_lora_precache_only_workflow.json).
- **TPU & Cross-Backend Training Speedups**:
  - Tuned headless TPU HBM budget in `compute_vram_budget()` (3% HBM reserve, lean XLA workspace, and HBM-resident checkpointed block inputs when they fit) so 9.6 GB models stay 100% HBM-resident on 16 GB TPU v5e-1 instead of CPU-streaming.
  - Added static-shape `prompt_embeds` padding (multiple of 64) on TPU to eliminate XLA recompilations from variable caption lengths.
  - Switched `clip_grad_norm_fp32_` in `train_step()` to zero-`.item()` on-device clipping (`return_norm=False`) and removed redundant per-step TPU `synchronize()` stalls.
  - Added batched CPU text encoding (`batch_size = 8`) in `OmniLoRAPrecomputeConditioning` for single-encoder LLM/CLIP models running on CPU.

### Fixed
- **TPU Masked SDPA Backward `NaN`**: Routed masked TPU training SDPA in `_wrapped_sdpa` (`pipelines/lora_trainer.py`) through FP32-softmax math attention with finite `-1e9` masking (fixing `hunyuan_video`, `ltxav_2_0`, `ltxav_2_5` on `torch-tpu`).
- **Loss Tracker Active-Run Protection (`web/omni_lora_tracker.js`)**: `Clear` and `Delete` now preserve any `status === "active"` training run (sending `clear_runs=inactive` to `tracker_store.py`), and `↻ Sync` always un-dismisses active runs on the server.
- **Early Setup Exception Handling (`nodes.py`)**: Initialized `prompt_server = None` before `pipeline.setup()` in `train_lora()` so early setup failures never raise `UnboundLocalError`.
- **Synced All 7 Example Workflows**: Updated all `example_workflows/*.json` templates to match the latest `OmniLoRAPrecomputeConditioning` schema (`model` optional slot + `disk_cache = "auto (load or save)"`).

### Documentation
- Updated `README.md` Hardware Backends table (`tpu` $\rightarrow$ **Actively tested on hardware**, `rocm` $\rightarrow$ **Community-tested on hardware**), documented portable disk pre-caching & lazy VAE/CLIP skipping, added `omni_lora_precache_only_workflow.json` to the workflows table, and linked GitHub Discussions (Benchmarks) & Issues.

---

## [1.1.4] - 2026-11-23

### Added
- **Shared Diagnostics & Recoverable Exception Framework (`omni_lora/diagnostics.py`)**:
  - Added `get_logger()`, thread-safe memory-bounded `log_once()`, `clear_log_once_cache()`, `strict_errors_enabled()`, and `report_exception()` so recoverable fallbacks never swallow errors blindly or spam hot loops.
  - Added `OMNILORA_STRICT_ERRORS=1` environment switch to re-raise unexpected exceptions at critical cleanup and state-restoration boundaries during CI or debugging.
- **Repo-Wide AST Silent-Exception Regression Guard & Numerical Parity Tests**:
  - Added `test_diagnostics_and_repo_wide_silent_exception_guard` enforcing zero bare `except:` and zero silent `except Exception: pass` handlers across all production Python files.
  - Added `test_optimizer_reference_equivalence_and_quantized_backward_parity` verifying step-for-step numerical equivalence for `PureFP32Adafactor` (2D factored + 1D unfactored), FP32-master `RAdamW` / `AdamW` / `SGD`, and quantized `LoRALinear` (`int8_qlora` / `fp8_e4m3fn`) forward/backward gradients (`dx`, `lora_A`, `lora_B`) against dense dequantized references.

### Fixed
- **Genuine `RAdamW` Execution With `warmup_steps > 0`**:
  - Removed the silent `has_lr_warmup` fallback to `AdamW` in `LoRATrainingPipeline._build_fp32_safe_optimizer()` and `_fp32_master_step()` so selecting `optimizer="radamw"` always runs genuine `torch.optim.RAdam(..., betas=(0.9, 0.99), decoupled_weight_decay=True)` regardless of `warmup_steps`.
- **Repo-Wide Exception Hygiene (209 $\rightarrow$ 0 silent `except Exception: pass`)**:
  - Audited and hardened all exception handlers across `pipelines/lora_trainer.py`, `lora/layers.py`, `models/adapters.py`, `nodes.py`, `backends/*.py`, `dataset.py`, and `tracker_store.py`: upgraded state-restoration and weight-uninject failures to `WARNING` (+ `reraise_in_strict=True`), surfaced malformed dataset caption metadata / unreadable directory warnings, and narrowed hardware/ComfyUI probing fallbacks to specific exception types with deduplicated `DEBUG` logging.

### Removed
- **Legacy Windows DirectML (`dml` / `torch-directml`) Backend & PyTorch 2.4 Workarounds**:
  - Removed `backends/dml.py` (`DirectMLBackend`), the `"dml"` option from `BACKEND_REGISTRY`, `AUTO_PROBE_ORDER`, and `OmniLoRATrainer`'s `backend` dropdown, and ~1,100 lines of `privateuseone`-specific monkey-patches (`install_dml_fp64_scalar_guards()` global `torch.Tensor` operator hooks, 5D `pixel_unshuffle`/`pixel_shuffle`/`conv3d` fallbacks, `_pure_fp32_sdpa`, and `_unify_16bit_seq_tensors`).
  - `torch-directml` is frozen on PyTorch 2.4 and incompatible with modern ComfyUI / `comfy_kitchen` custom ops; Windows GPUs are now served by native PyTorch backends (`cuda` for NVIDIA, `xpu` for Intel Arc, and `rocm` for AMD).

### Documentation
- Refined VRAM budgeting terminology in `README.md` to **Analytical VRAM Ledger** with explicit dynamic OOM self-healing context, added a **Verification Scope** callout and hardware **Status** column distinguishing actively hardware-tested vs. community-tested backends, and documented `OMNILORA_STRICT_ERRORS=1`.

---

## [1.1.2] - 2026-11-22

### Fixed
- **Ideogram 4 / FP16 Autocast MRoPE & Additive Attention Mask Overflow (`NaN` at Step 1)**:
  - Preserved `float32` precision for pure non-differentiable `float32` `matmul` / `@` / `bmm` / `mm` / `addmm` / `baddbmm` / `einsum` operations inside `_scoped_training_op_wrappers()` when `torch.amp.autocast(..., dtype=torch.float16)` is active (preventing large positional coordinates such as Ideogram 4's `IMAGE_POSITION_OFFSET = 65536 > 65504.0` from overflowing to `+inf`/`NaN` on FP16-only GPUs like Tesla T4 / RTX 20-series), with full `torch.Tensor` teardown on scope exit.
  - Hardened `_wrapped_sdpa` FP32-to-FP16 floating-point `attn_mask` casting to clamp finite large negative sentinels (e.g. `-1e9`) into `[-1e4, 1e4]` while preserving explicit `-inf` entries.
- **CPU-Only CI & Headless ComfyUI Import Safety**:
  - Updated `ensure_comfy_importable()` (and test startup) to automatically set `comfy.cli_args.args.cpu = True` before `comfy.model_management` initializes when no PyTorch hardware accelerator is available, fixing `AssertionError: Torch not compiled with CUDA enabled` on CPU-only PyTorch / GitHub Actions CI runners.
- **`--cloud-verify` Combined Final Verdict Banner**:
  - Updated `run_cloud_verification_suite()` to flush `stderr`/`stdout` after `unittest` finishes and print an unambiguous combined `CLOUD VERIFICATION SUMMARY: PASS` / `FAILED` banner re-listing any active-backend real-forward failures so runs never end with a misleading trailing `OK`.

---

## [1.1.1] - 2026-11-22

### Added
- **Lifecycle Cleanup & CI Verification**:
  - Added `uninstall_comfy_runtime_hooks()` (exported from `omni_lora.models`) for clean teardown of global PyTorch/ComfyUI training wrappers, plus diagnostic logging across ComfyUI cache eviction/unloading and cleanup exception paths.
  - Added `.github/workflows/ci.yml` for automated CPU unit and real-model contract test runs on pushes and pull requests.
  - Added regression test suites covering scoped training flags, zero-match external backbones, signature-agnostic `comfy_kitchen` wrappers, DML-safe markers, grouped/1D/3D Conv LoRA adapters, gradient accumulation parity, LoRA dropout train/eval isolation, periodic/manual/interrupt checkpoints, tracker HTTP routes, target module filters, and workflow/version parity.

### Fixed
- **`LoRATrainingPipeline.setup()` & `train_step()` `in_training` Flag Leak**:
  - Scoped `comfy.model_management.in_training` with `_scoped_comfy_in_training` across `setup()` and `train_step()` so its prior value is always restored in `finally:`, even when calling `LoRATrainingPipeline` directly or when an exception occurs.
- **LoRA Dropout Base-Layer Mode Isolation**:
  - Preserved `.eval()` on each `LoRALinear.base_layer` immediately after enabling `.train()` on injected `LoRALinear` adapters in `LoRATrainingPipeline.setup()`, preventing frozen base layers from entering `.train()` mode.
- **Grouped `Conv2d` LoRA Input Channel Mismatch**:
  - Fixed `LoRALinear` grouped convolution fallback when `rank % groups != 0` (`eff_groups = 1`) to pass `self.in_features` instead of `weight.shape[1]` (`in_channels // raw_groups`) to `lora_A`.
- **`.pt` LoRA Checkpoint Metadata Preservation**:
  - Updated `_load_lora_safetensors_or_pt()` to read non-`.safetensors` `.pt` bundles via `torch.load(..., map_location="cpu", weights_only=True)` before `comfy.utils.load_torch_file()`, preserving top-level `"metadata"` (`omni_lora.steps`, `omni_lora.rank`, `omni_lora.alpha`, `omni_lora.trigger_word`) alongside `"state_dict"`.
- **Training Op Wrapper, Device Safety, Cleanup & Dataset Hardening**:
  - Restored pre-existing `mm.in_training` / `disable_comfy_compiler` states when `OmniLoRATrainer.train_lora()` errors during early setup; aborted on zero-target external backbones instead of falling back; made `comfy_kitchen` wrappers signature-agnostic; added `_omni_dml_safe` idempotency markers; raised explicitly on LoRA device mismatch instead of calling `.to()` and severing optimizer parameter references; ensured `cleanup()` unpatches module `__globals__` even with `gradient_checkpointing=False`; avoided writing interrupted checkpoints on step-1 errors; surfaced `load_lora_for_models` warnings; rejected mismatched `control_images` batch sizes and avoided stem collisions across nested dataset subfolders.
- **Telemetry Store Persistence & Multi-Node Isolation**:
  - Switched `OmniLoRATelemetryStore` checkpoint requests to an explicit `node_id` queue, skipped writing empty `.omni_lora_tracker_runs.json` files on import, and added automatic fallback to the system temp directory when the package folder is read-only.
- **Example Workflows Updated to Latest Node Schemas**:
  - Updated `omni_lora_basic_workflow.json`, `omni_lora_dual_network_workflow.json`, `omni_lora_masked_in_graph_workflow.json`, `omni_lora_resume_checkpoint_workflow.json`, and `omni_lora_staged_eval_workflow.json` with the latest `OmniLoRADatasetLoader` (`control_images`, `control_path`, `control_resize_mode`), `OmniLoRAPrecomputeConditioning` (`pass_control_to_clip`, `clip_vision_resolution`, `clip_prompt_template`), and `OmniLoRATrainer` (`control_mode`, `reference_latents_method`, `control_dropout`) inputs, widgets, and non-overlapping layouts.

---

## [1.1.0] - 2026-11-21

### Added
- **Universal Image Editing / Control Folder Support**:
  - Added paired control/source folder loading to `OmniLoRADatasetLoader` (`control_path`, `control_resize_mode` `"match_target"` / `"preserve_aspect"`, direct in-graph `control_images` tensor input, cross-extension and mirrored subfolder stem matching, mixed controlled/uncontrolled datasets, and fail-fast 0-match validation).
  - Upgraded `OmniLoRAPrecomputeConditioning` to cache raw unscaled VAE control latents (matching ComfyUI's native `ReferenceLatent` and `InstructPixToPixConditioning` nodes), auto-detect VLM text encoders accepting `clip.tokenize(..., images=[...])`, resize control images via `clip_vision_resolution`, support custom `clip_prompt_template` (`{prompt}` / `{}`), and synchronize `random_flip` across target latents, control latents, and VLM visual tokens.
  - Added `control_mode` (`"auto"` / `"reference_latents"` / `"concat_latent_image"`), `reference_latents_method` (`"auto"` / `"offset"` / `"index"` / `"index_timestep_zero"` / `"uxo"`), `control_dropout`, edit-aware VRAM ledger activation accounting (`has_control_latents`), and safe per-item microbatching for mixed controlled/uncontrolled batches in `OmniLoRATrainer` and `ComfyModelAdapter`.
  - Added [`example_workflows/omni_lora_image_edit_workflow.json`](./example_workflows/omni_lora_image_edit_workflow.json) and a 14-architecture unmocked edit contract test suite (`TestRealForwardEditModels`) covering Flux Kontext, Flux 2, Flux 3 Klein, Qwen-Image 1.0/2.1, OmniGen2, Boogu, MOSS-Image, FIBO 2.0, Krea 2, JoyImage, Mage-Flow (plus live dynamic discovery of future `ref_latents` models), HiDream-E1, and SD1.5 InstructPix2Pix — verified in both FP32 and BF16, at both square and mismatched (`preserve_aspect`) control resolutions, across `reference_latents_method` modes (`offset`, `index`, `index_timestep_zero`, `uxo`), and through native ComfyUI `LoraLoader` round-trips (including dedicated edit branches `ref_image_refiner.*` and `content_mode_blocks.image_cond.*`).
- **Lazy Streaming Dataset Loading & Pre-Caching**:
  - `OmniLoRADataset` now scans folder-based datasets via image headers only (dimensions, EXIF orientation, alpha/sidecar mask detection, and control-folder pairing) without eagerly decoding pixel buffers into CPU RAM.
  - `OmniLoRAPrecomputeConditioning` streams one image pair at a time during Stage 1 (VAE encoding + immediate latent-resolution mask downsampling) and decodes control images on-demand only on cache misses during Stage 2 VLM CLIP encoding — reducing peak host RAM on large paired datasets from `O(N)` to `O(1)` image pair while keeping `DatasetSample` properties 100% backward-compatible.
- **Scoped (Non-Permanent) PyTorch Op Wrappers**:
  - Replaced permanent global monkey-patching of `torch.*` (`cat`, `stack`, `add`, `lerp`, `matmul`, etc.) and `torch.nn.functional.*` (`scaled_dot_product_attention`, `linear`, `conv1d/2d/3d`, `layer_norm`, `rms_norm`, `group_norm`, `pixel_shuffle/unshuffle`) with a thread-safe, ref-counted `_scoped_training_op_wrappers()` lease restored cleanly in `LoRATrainingPipeline.cleanup()` and `finally:` blocks so post-training ComfyUI inference and `torch.compile` run on native unpatched PyTorch callables.
- **44-Architecture Unmocked Real-Forward Verification Suite (`RealForwardModelRegistry`)**:
  - Replaced mocked forward tests with real unmocked `comfy.ldm.*` + `comfy.model_base.*` execution across 44 Image & Video architectures, verifying 7 end-to-end training invariants on every model plus a fast `torch.bfloat16` real-forward + backward smoke contract across all 44 models on CPU.
  - Broadened `TestComfyUIUpstreamDriftGuard` to scan for *any* `.preprocess_*(` call or custom `process_latent_in` override across `comfy.model_base` subclasses, added live `comfy.text_encoders.*` VLM tokenizer detection verification, and upgraded `ComfyModelAdapter` to emit a clear rate-limited `logging.warning` whenever `inner_model.extra_conds(...)` raises instead of failing silently.
- **LTX-AV 2.0 & 2.5 (`LTXAVModel`) Full Support**:
  - Automatically excludes paired secondary `audio_*` transformer branches and audio connectors during video-only LoRA training while preserving full targeting on audio-only architectures.
  - Preserves all 3 legal pre- and post-connector text embedding widths (`cross_attention_dim`, concatenated `cross_attention_dim + audio_cross_attention_dim`, and `caption_proj_before_connector` / `first_linear` mode) and reconciles post-connector `attention_mask` lengths for variable-length multi-sample batches.
- **MiniMax-H3-Image (`MiniMaxH3ImageModel`) Support**:
  - Added full support for `MiniMaxH3ImageModel` single-stream image latents (`audio_latents_dim=None`), PDD curves, gate compression, and strict `batch_size == 1` microbatching alongside `MiniMaxH3Model` dual-stream video/audio.
  - Preserved `float32` precision on MiniMax H3 / H3-Image small numerical islands (`video_patch_proj`, `audio_patch_proj`, `time_embedder`, `final_layer` PDD heads, and curve-form `adaln_proj`) during VRAM placement so training matches inference numerically.
  - Guarded `OmniLoRAPrecomputeConditioning` against stale `inner_model.latent_shapes` left on multimodal models (`MiniMaxH3`, `LTXAV`) by prior ComfyUI `KSampler` runs.
- **ComfyUI `comfy_aimdo.malloc_graph` & `comfy.model_prefetch` Training Guard**:
  - Safely guards ComfyUI's inference-only `aimdo` memory compiler and prefetch hooks during training to prevent `RuntimeError: aimdo memory compile error` (`rogues: N`) when autograd saves intermediate activations.
- **Unbounded In-Place Autograd Fidelity Guards**:
  - Removed the previous 8 MB saved-tensor snapshot cap and hardened autograd safety for in-place slice mutations (`CopySlices`), `torch.cat(..., out=buf)` (Flux 3 Klein), and MiniMax H3 `_mod_scale_shift` / `_mod_gate` under both `gradient_checkpointing=True` and `False`, guaranteeing `comfy.model_management.in_training` is always restored on exceptions.

### Fixed
- **MiniMax-H3-Image BF16 `torch.lerp` Mixed-DType Crash**:
  - Aligned floating-point operand dtypes in the scoped `torch.lerp` wrapper to prevent `RuntimeError: expected dtype BFloat16 for weight but got dtype float` during BF16 training and probes.
- **Krea 2 `reference_latents_method="auto"` Routing**:
  - Automatically defaults to `reference_latents_method="index"` when `control_mode="reference_latents"` is active on a backbone whose `default_ref_method` is `None`.
- **Mismatched `preserve_aspect` Control Alignment on `concat_latent_image` Models**:
  - Bilinearly aligns control latents to `noisy_latents` `(H, W)` in FP32 (preserving 5D temporal dimensions) for channel-concatenated / `image_cond` models (`HiDream-E1`, `InstructPix2Pix`, etc.) while keeping `reference_latents` DiTs at their native aspect ratio.
- **Instance-Wrapped `backbone.forward` Signature Resolution**:
  - Inspects `type(backbone).forward` in `ComfyModelAdapter` so instance-level `(*args, **kwargs)` wrappers from ComfyUI hooks/extensions never hide required positional parameters (such as `LTXBaseModel.forward(..., attention_mask)`).
- **VLM CLIP Detection & Post-Injection LoRA Key-Map Hardening**:
  - Hardened `_clip_supports_images_kwarg` / `_clip_supports_llama_template_kwarg` against `None` `llama_template_images` class attributes on non-VLM tokenizers, and sanitized `.base_layer.` in `LoRAInjector.capture_comfy_lora_key_map` if invoked after LoRA injection.
- **PixArt (`PixArtMS`) Crash & Frozen Base-Model `.eval()` Discipline**:
  - Kept the frozen base backbone in `.eval()` mode during LoRA fine-tuning while enabling `.train()` exclusively on injected LoRA adapter layers (`LoRALinear` / `lora_dropout`).
  - Eliminates PixArt's `CaptionEmbedder.token_drop` crash (`AssertionError` on variable-length prompts plus a hardcoded `.cuda()` call that failed on XPU/CPU/MPS/DirectML) and prevents frozen base-model dropout, stochastic depth, and BatchNorm running-stat updates from perturbing LoRA training or checkpointing parity.
- **OmniGen2 & Boogu Multi-Batch (`B > 1`) `attention_mask` & `num_tokens` Crashes**:
  - Reshaped raw 2D `(B, L_k)` per-sample key padding masks to `(B, 1, 1, L_k)` in `_wrapped_sdpa` to prevent right-aligned broadcast collisions against `L_q` when `batch_size > 1`.
  - Clamped batched `num_tokens` to `context.shape[1]` and added OmniGen2 / Boogu to multi-row `attention_mask` single-item microbatching so each sample receives its exact per-prompt token count and RoPE positions.
- **Bria FIBO 2.0 Bundled Multi-Layer Context Width**:
  - Fixed `_infer_backbone_context_dim` and `_collect_valid_backbone_context_dims` to recognize Fibo2's bundled `(1 + len(injection_layers)) * cap_feat_dim` context width instead of misadapting or truncating conditioning tensors.
- **Anima `batch_size > 1` `extra_conds` `inference_mode` Escape**:
  - Exited `torch.inference_mode()` cleanly under `torch.no_grad()` when running `Anima.extra_conds` on multi-sample batches so conditioning tensors remain autograd-compatible.
- **Stable Cascade 1D Scalar Conditioning Broadcast**:
  - Fixed 1D scalar conditioning broadcast (`sca`, `crp`) across `batch_size > 1` and refined nested `nn.ModuleList` structural role classification to exclude pure modulation `TimestepBlock` layers.

---

## [1.0.4] - 2026-11-18

### Fixed
- **Flux / Flux.2 / Chroma `int64` `attention_mask` SDPA Crash**:
  - Fixed `RuntimeError: Expected attn_mask dtype to be bool or float or to match query dtype, but got attn_mask.dtype: __int64 and query.dtype: struct c10::BFloat16 instead.` during Flux / Flux.2 / Chroma training whenever text encoders or variable-length multi-sample batches produced a 2D `int64` `attention_mask` in `extra_conds`.
  - Updated `ComfyModelAdapter._forward_backbone` to treat a successful `BaseModel.extra_conds(...)` call as authoritative instead of re-injecting omitted conditioning keys via `**kwargs`, preventing raw text token masks (dropped by `Flux.extra_conds` when `attention_mask_img_shape` is `None`) from reaching `Flux._forward` joint attention. Standalone backbones without `extra_conds` also restrict mask/bias keys to explicitly declared parameters.
  - Added defensive SDPA attention mask normalization (`_normalize_sdpa_attn_mask` in training attention overrides/fallbacks, plus integer-to-`bool` conversion in `_wrapped_sdpa` and `_pure_fp32_sdpa`) so models that legitimately pass integer `0/1` or batched 2D `[B, L_k]` masks to `F.scaled_dot_product_attention` convert cleanly to `torch.bool` / `[B, 1, 1, L_k]`.

---

## [1.0.3] - 2026-11-18

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

## [1.0.2] - 2026-11-17

### Added
- **Comfy Kitchen (`--use-ck-attention`) & SageAttention (`--use-sage-attention`) Training Guards**:
  - Automatically intercept inference-only `comfy_kitchen` / `sage` attention and custom `comfy.quant_ops.ck` kernels (`apply_rope`, `apply_rope1`, `rms_norm`, `adaln`, `rms_rope_split_half`) during training and route them through differentiable PyTorch operations, restoring the user's inference backend after training finishes.
- **Anima / Cosmos Predict2 `extra_conds` Support**:
  - Preserve autograd tracking when preprocessing model-specific `extra_conds` (`t5xxl_ids`, `t5xxl_weights`, `crossattn_mask`) and coerce integer token ID tensors vs floating-point weight tensors accurately.

### Fixed
- Ignore runtime `.loss_tracker_history.json` telemetry files in `.gitignore`.

---

## [1.0.1] - 2026-11-17

### Fixed
- Refactored subprocess and file-handling helpers to eliminate Comfy Registry YARA security scanner false positives.

---

## [1.0.0] - 2026-11-17

### Added
- Initial stable release of **ComfyUI-OmniLoRA**:
  - 6 native ComfyUI custom nodes (`OmniLoRADatasetLoader`, `OmniLoRAPrecomputeConditioning`, `OmniLoRATrainer`, `OmniLoRALoader`, `OmniLoRASaveLoRA`, `OmniLoRALossTracker`) + interactive canvas Web UI extension.
  - Universal runtime model introspection supporting 50 ComfyUI `comfy.ldm.*` diffusion, Rectified Flow, Flow-Map, and video/audio architectures.
  - Direct training on FP8, INT8 `QuantizedTensor`, `ComfyUI-GGUF` (`GGMLTensor`), and `bitsandbytes` backbones with deterministic VRAM budgeting and on-demand CPU layer streaming across CUDA, Intel Arc XPU, AMD ROCm, Apple MPS, Windows DirectML, Cloud TPU, NPU, HPU, and CPU.
