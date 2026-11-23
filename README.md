# ComfyUI-OmniLoRA

**Universal in-graph LoRA training nodes for ComfyUI across CUDA, Intel Arc XPU, ROCm, TPU, MPS & CPU. Supports any ComfyUI model, quantized backbones (FP8/INT8/GGUF), analytical VRAM budgeting & automatic CPU layer streaming.**

Train LoRAs directly inside your ComfyUI workflow — for standard **text-to-image/video** or paired **image-editing / control-folder** datasets — using the models, VAEs, and text encoders you already have on canvas. Works out of the box with **Flux (1 / Kontext / 2 / 3 Klein), Qwen-Image (1.0 / 2.1), Wan (2.1 / 2.2 Video & Image), SDXL, SD 3.5, SD 1.5, Chroma, Bria FIBO 2.0, HiDream (I1 / E1), MOSS-Image, OmniGen2, HunyuanVideo, LTX-Video & LTX-AV (2.0 / 2.5), MiniMax H3 (Video & Image), Lumina 2 / Z-Image, AuraFlow / Sana**, and quantized **FP8 / INT8 / GGUF / bitsandbytes** checkpoints across **NVIDIA CUDA, Intel Arc XPU, AMD ROCm, Apple Silicon MPS, Cloud TPU, Ascend NPU, Gaudi HPU, and CPU**.

---

## Quick Start

1. **Install** *(zero external `pip` dependencies — uses ComfyUI's built-in `torch`, `safetensors`, and `Pillow`)*:
   - **ComfyUI Manager (Recommended)**: Open **ComfyUI Manager** $\rightarrow$ **Custom Nodes Manager**, search for **`OmniLoRA`**, click **Install**, and restart ComfyUI.
   - **Git Clone** inside your custom_nodes folder:
     ```bash
     git clone https://github.com/cyberterm/ComfyUI-OmniLoRA
     ```
     Restart ComfyUI after cloning.
2. **Load a Template**: [`example_workflows/omni_lora_basic_workflow.json`](./example_workflows/omni_lora_basic_workflow.json) can be found in the templates.
3. **Set 3 Things & Click Queue**:
   - In **`Load Diffusion Model` / `CLIPLoader` / `VAELoader`**: Pick your model, text encoder (set the matching `type`, e.g. `flux`, `qwen_image`, `wan`, `sd3`, `lumina2`), and VAE. You can also use a gguf loader or a checkpoint loader (SDXL).
   - In **`OmniLoRA Dataset & Prompt Loader`**: Paste the folder path to your training images into `dataset_path` and set your `trigger_word` (e.g. `ohwx`).
   - Click **Queue Prompt**. Your trained `.safetensors` LoRA will be saved to `ComfyUI/models/loras/` and the patched `MODEL` output can wire straight into a `KSampler` for instant testing.

```text
[Load Diffusion Model] ──────────────────────────────────────────────────────┐
[CLIPLoader] ───────────────────────┐                                        ▼
[VAELoader] ──────────────┐         ▼                               [OmniLoRA Universal Trainer] ──(model)──> [KSampler]
                          ▼         ▼                                        ▲        │        │
[Dataset Loader] ──> [VAE & CLIP Pre-Cache] ────────(cached_data)────────────┘     (lora)  (summary)
                                                                                      ▼        ▼
                                                                             [Save LoRA]  [Loss Tracker]
```

---

## Beginner's Guide: Training Your First LoRA

### Step 1: Prepare Your Images (Captions Are Optional!)
Create a folder anywhere on your drive (for example `C:/datasets/my_character` or `./datasets/train`) and drop in **10 to 25 images**:
- **Mix any image formats**: `.png`, `.jpg` / `.jpeg`, `.webp`, `.avif`, `.bmp`, `.tiff`, `.gif`, and `.heic` / `.heif` can all live in the same folder.
- **Any aspect ratio works**: Portrait, square, and landscape photos are automatically grouped into aspect-ratio-preserving 64px buckets (`aspect_ratio_bucketing = True`) without stretching or squishing faces.
- **How captions work**:
  - **No `.txt` files? No problem.** If an image doesn't have a caption file, OmniLoRA uses `default_caption` and automatically prefixes your `trigger_word` (e.g. `"ohwx, a high-resolution detailed photograph"`).
  - **Have captions?** Place matching `<image_name>.txt` or `<image_name>.caption` files next to your images (or use a single `metadata.jsonl` / `captions.json` file in the folder). If a caption doesn't already start with your `trigger_word`, OmniLoRA prepends it automatically.
  - **Transparent PNGs or Masks?** If your images have RGBA transparency (or matching `<image_name>.mask.png` files / a `masks/` subfolder), `use_alpha_mask = True` automatically weights training loss toward the subject so the model doesn't memorize the background.

### Step 2: Pick Settings for Your Goal (Good starting values)

| What You Are Training | Recommended Images | `steps` | `rank` / `alpha` | `learning_rate` | `optimizer` | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Character / Person / Object** | 10 – 25 images | `800 – 1500` | `32` / `32.0` | `0.0001` (`1e-4`) | `radamw` | Set a unique `trigger_word` (e.g. `ohwx`). Keep `min_snr_gamma = 0.0`. |
| **Artistic Style / Aesthetic** | 25 – 100 images | `1500 – 3000` | `32 – 64` / `32.0` | `0.0001` (`1e-4`) | `radamw` or `adafactor` | Enable `use_rslora = True` if `rank >= 64`. Optional `min_snr_gamma = 5.0`. |
| **Quick Concept Test** | 5 – 12 images | `300 – 500` | `16` / `16.0` | `0.00015` | `radamw` | Set `warmup_steps = 20`. Wire trainer `model` output directly to `KSampler`. |
| **Low VRAM (8–12 GB GPU)** | Any | `1000` | `16 – 32` | `0.0001` (`1e-4`) | `adafactor` | Cuts optimizer VRAM by ~80%; frozen layers exceeding VRAM stream from CPU RAM automatically. |

> **Using Your GPU for Your Monitor?**
> If your display is plugged into the same GPU you train on, setting **`reserve_vram_gb = 0.5` to `1.5`** on `OmniLoRA Universal Trainer` may increase stability.
>
> **Using Comfy Kitchen (`--use-ck-attention`) or SageAttention (`--use-sage-attention`)?**
> OmniLoRA automatically overrides inference-only attention and `comfy_kitchen` RoPE/AdaLN kernels with differentiable PyTorch SDPA (`attention_pytorch`) for the duration of training and restores your inference backend afterward — though launching ComfyUI with default PyTorch attention (`--use-pytorch-cross-attention`) remains the cleanest baseline for dedicated training sessions.

### Step 3: Monitor, Checkpoint & Test in One Click
- **Live Loss Graph**: Connect `training_summary` from the Trainer to **`OmniLoRA Loss Tracker`** to watch real-time loss, smoothed EMA trend, learning rate, speed (`s/step`), ETA, and VRAM usage directly on the ComfyUI canvas.
- **Save a Checkpoint Anytime**: Click the **`Save Ckpt`** button inside the `OmniLoRA Loss Tracker` header while training is running to immediately write a `.safetensors` snapshot at the end of the current step — or set `checkpoint_every_n_steps = 200` on the trainer node.
- **Safe Cancel / Interrupt**: Need to stop early? Clicking ComfyUI's **Cancel / Interrupt** button mid-run automatically saves an emergency `<prefix>_interrupted_step_<N>.safetensors` file to `ComfyUI/models/loras/` so your progress is never lost.

---

## Included Ready-to-Run Workflows

All 6 workflow templates are located in [`example_workflows/`](./example_workflows/) — drag any `.json` file into ComfyUI:

| Workflow File | Use Case |
| :--- | :--- |
| **[`omni_lora_basic_workflow.json`](./example_workflows/omni_lora_basic_workflow.json)** | **Standard Folder Training (Start Here)**: Loads a folder of images + captions $\rightarrow$ Pre-caches VAE & CLIP $\rightarrow$ Trains for `1000` steps $\rightarrow$ Saves `.safetensors` + live `OmniLoRA Loss Tracker`. |
| **[`omni_lora_image_edit_workflow.json`](./example_workflows/omni_lora_image_edit_workflow.json)** | **Image Editing / Control Folder Training**: Pairs target images (`dataset_path`) with source/control images (`control_path`) for any ComfyUI edit model (`reference_latents`, `concat_latent_image` / `image_cond`, and VLM vision-language text encoders). |
| **[`omni_lora_staged_eval_workflow.json`](./example_workflows/omni_lora_staged_eval_workflow.json)** | **Staged Training + Mid-Run KSampler Preview**: Runs `Trainer 1 (500 steps)` $\rightarrow$ renders a test image with `KSampler` $\rightarrow$ continues into `Trainer 2 (steps 501–1000)` $\rightarrow$ `Save LoRA`. |
| **[`omni_lora_masked_in_graph_workflow.json`](./example_workflows/omni_lora_masked_in_graph_workflow.json)** | **No-Folder Canvas Training (`IMAGE` + `MASK`)**: Connects ComfyUI `LoadImage` nodes (`IMAGE` and alpha `MASK`) directly into `OmniLoRA Dataset & Prompt Loader` for foreground-weighted training. |
| **[`omni_lora_resume_checkpoint_workflow.json`](./example_workflows/omni_lora_resume_checkpoint_workflow.json)** | **Resume an Existing LoRA + Periodic Checkpoints**: Loads a saved `.safetensors` file via `OmniLoRA Load LoRA Network`, continues training with `grad_accum_steps=2`, and saves checkpoints every 100 steps. |
| **[`omni_lora_dual_network_workflow.json`](./example_workflows/omni_lora_dual_network_workflow.json)** | **Dual-Network & Split-Timestep Expert Training**: Trains two sub-networks (`"text_embedder.blocks"` on `[0.5, 1.0]` and `"layers"` on `[0.0, 0.5]`) with separate learning rates and merges them into one LoRA. |

---

## Why Use OmniLoRA Over Core `TrainLoraNode` or External CLIs?

ComfyUI includes an experimental built-in `TrainLoraNode` (`comfy_extras/nodes_train.py`), and external trainers require separate Python environments and model-specific scripts. `ComfyUI-OmniLoRA` bridges both worlds:

1. **Two-Stage VAE & CLIP Pre-Caching + Full VRAM Eviction**: Encodes your dataset through the VAE first (one image at a time), unloads the VAE from VRAM, encodes deduplicated prompts through CLIP, and unloads CLIP to CPU RAM *before* loading the diffusion backbone. All 100% of free VRAM goes to the UNet/DiT.
2. **Train Directly on Quantized & GGUF Models**: Train LoRAs on top of **FP8** (`float8_e4m3fn` / `float8_e5m2`), ComfyUI **`QuantizedTensor`** (`int8_tensorwise`, `convrot`), **`ComfyUI-GGUF`** (`GGMLTensor` Q4_K_M / Q5_K / Q8_0 via on-the-fly GPU dequantization), and **`bitsandbytes`** (`NF4` / `INT8` on CUDA) without upcasting the frozen backbone to FP32.
3. **Analytical VRAM Ledger & Automatic CPU Layer Streaming**: Before Step 1, OmniLoRA inspects the live LoRA-injected model graph to compute a bottom-up byte budget for optimizer states (`opt`), peak single-block backward activations (`act`), and checkpointed block inputs (`ckpt_in`). Frozen base layers that fit stay in VRAM; remaining layers stream on-demand from pinned CPU RAM into a reusable 2-layer GPU staging buffer, backed by automatic mid-step OOM demotion if driver overhead or allocator fragmentation spikes.
4. **Universal Runtime Model Introspection (`model_architecture="auto"`)**: Queries ComfyUI's live `ModelPatcher`, `BaseModel`, `latent_format`, `model_sampling`, and `extra_conds` directly. Automatically handles 4D images, 5D video (`[B, C, T, H, W]`), $2\times2$ patch-packed latents (`16ch <-> 64ch`), `c_concat` Inpaint/I2V/Control channels, Rectified Flow / Flow-Map `sample_sigmas` / V-Prediction / EDM / EPS schedules, and multi-axis RoPE conditioning.
5. **Multi-Vendor Hardware Native (100% FP64-Free)**: Zero custom CUDA-only C++ extensions, zero `xformers` dependencies, and strict `float64` sanitization so GPUs without hardware FP64 or with 4 GiB single-allocation limits (**Intel Arc XPU**, **AMD ROCm**, **Apple MPS**) train at full native speed.

---

# Detailed Reference (Nodes, Architectures & Internals)

## 1. Custom Nodes Reference (Category: `OmniLoRA/Training`)

### Node 1: `OmniLoRA Dataset & Prompt Loader` (`OmniLoRADatasetLoader`)
Loads training images and captions from a folder on disk, direct ComfyUI `IMAGE` / `MASK` node connections, or both simultaneously.
- **Supported image formats**: `.png`, `.jpg`, `.jpeg`, `.jfif`, `.webp`, `.avif`, `.bmp`, `.tiff`, `.tif`, `.gif`, `.heic`, and `.heif` (with automatic multi-decoder fallback across Pillow, `imageio`, OpenCV `cv2`, and `ffmpeg`).
- **Flexible folder structures & caption formats**:
  - Scans `dataset_path` and any nested concept subfolders (e.g. Kohya-style `10_ohwx_subject/`).
  - Reads `<image_stem>.txt` or `<image_stem>.caption` sidecars, HuggingFace `metadata.jsonl` / `captions.jsonl`, or `captions.json` / `metadata.json`.
  - Falls back to `default_caption` when no caption file exists, and prepends `trigger_word` (without duplicating it if already present).
- **In-graph `images`, `custom_prompts`, and `masks` inputs**: Connect any ComfyUI `IMAGE` batch (`[B, H, W, C]`) and optional `MASK` batch directly from canvas nodes (`LoadImage`, segmentation nodes, or generated latents) alongside multiline `custom_prompts` (one line per image in the batch).
- **Aspect-ratio bucketing (`aspect_ratio_bucketing = True`)**: Assigns each image to the closest multiple-of-64 `(bucket_h, bucket_w)` resolution bucket bounded by `resolution × resolution`.
  - `center_crop = True`: Scales uniformly to cover the bucket and center-crops.
  - `center_crop = False`: Scales uniformly to fit inside the bucket, pads borders with neutral `0.5` gray, and automatically zeroes out loss on the padded border bars.
- **Alpha & mask loss weighting (`use_alpha_mask = True`, `mask_unmasked_weight = 0.1`)**: Extracts foreground masks from connected `MASK` inputs, `<stem>.mask.<ext>` / `<stem>_mask.<ext>` sidecar files, a `masks/<stem>.<ext>` subfolder, or RGBA transparency channels (compositing RGB over neutral `0.5` gray before VAE encoding so transparent pixels never bleed dirty RGB halo artifacts into latents).
- **Lazy header-only folder indexing**: Folder-based datasets read only lightweight image headers (dimensions, EXIF orientation, masks, and control pairing) at load time — zero full pixel buffers are held in CPU RAM until streamed one pair at a time during pre-caching.
- **Control / source image folder (`control_path`, `control_resize_mode`, `control_images`)**: Point `control_path` to a folder of paired source/before images (matched 1-to-1 with targets in `dataset_path` by mirrored subfolder path or filename stem across any image extension, e.g. `001.png` $\leftrightarrow$ `001.jpg`), or wire a ComfyUI `IMAGE` batch into `control_images`. Supports `"match_target"` (exact target bucket framing) and `"preserve_aspect"` (independent control aspect ratio), plus mixed controlled/uncontrolled datasets.

### Node 2: `OmniLoRA VAE & CLIP Pre-Cache` (`OmniLoRAPrecomputeConditioning`)
Encodes all dataset images with the connected `VAE` and all captions with the connected `CLIP` / text encoder.
- **Sequential low-VRAM & `O(1)` host-RAM streaming**: Streams one image pair at a time from disk through the `VAE` (downsampling spatial masks to latent resolution immediately), immediately unloads the `VAE` from VRAM, then encodes deduplicated text prompts (and on-demand VLM control thumbnails when applicable) through `CLIP`, and finally unloads `CLIP` to CPU RAM (`offload_encoders_after_cache = True`).
- **Dynamic per-step augmentations**: Pre-encodes horizontally flipped latents (`latents_flipped`, plus `control_latents_flipped` and flipped VLM prompt embeddings when control images are present) when `random_flip = True` and pre-encodes the unconditional empty prompt `""` once so `OmniLoRA Universal Trainer` can apply `random_flip` and CFG `caption_dropout` dynamically on every training step with zero encoder overhead.
- **Control & VLM vision conditioning (`pass_control_to_clip`, `clip_vision_resolution`, `clip_prompt_template`)**: Caches raw unscaled VAE control latents (matching ComfyUI's native `ReferenceLatent` / `InstructPixToPixConditioning` nodes) and automatically detects VLM text encoders whose `clip.tokenize` accepts `images=[...]` (resizing control images to `clip_vision_resolution`, default `384`). Leave `clip_prompt_template` empty to use captions and ComfyUI's native tokenizer template as-is, or supply a custom `{prompt}` / `{}` template.
- **Multi-modal & variable-length conditioning**: Preserves full ComfyUI conditioning dictionaries (`pooled_output`, `attention_mask`, `txt_mask`, `context_attention_bias`, `clip_l`, `guidance`, `ref_latents`, etc.), downsamples spatial `loss_mask` tensors to latent resolution, and expands `num_repeats` in CPU RAM without re-encoding.

### Node 3: `OmniLoRA Universal Trainer` (`OmniLoRATrainer`)
Injects trainable LoRA layers into the connected `MODEL`, executes the training loop outside `torch.inference_mode()` with scoped PyTorch op wrappers (restored cleanly upon exit so subsequent ComfyUI inference and `torch.compile` run with zero wrapper overhead), and returns:
- **`model` (`MODEL`)**: A cloned ComfyUI `ModelPatcher` with the trained LoRA patches attached, ready to connect directly to `KSampler` or to a downstream `OmniLoRA Universal Trainer` node.
- **`lora` (`OMNILORA_NETWORK`)**: Portable in-memory LoRA bundle containing the trained `diffusion_model.*` state dict and metadata (`rank`, `alpha`, `steps_trained`, `model_architecture`, `trigger_word`).
- **`training_summary` (`STRING`)**: JSON report of resolved architecture, VRAM ledger breakdown, step timing, checkpoint paths, and loss metrics.

#### Key Trainer Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `model_architecture` | `"auto"` | Introspects the connected ComfyUI `MODEL` automatically. |
| `backend` | `"auto"` | Auto-probes `xpu` $\rightarrow$ `rocm` $\rightarrow$ `cuda` $\rightarrow$ `tpu` $\rightarrow$ `npu` $\rightarrow$ `hpu` $\rightarrow$ `mps` $\rightarrow$ `cpu` (or force a specific backend). |
| `reserve_vram_gb` | `0.0` | Extra GPU VRAM (in GB) to leave free for your OS / monitor display driver (`0.5 – 1.5` recommended on display GPUs). |
| `optimizer` | `"radamw"` | `"radamw"` (Rectified AdamW), `"adamw"`, `"adafactor"` (~75–85% less optimizer VRAM), or `"sgd"` (Nesterov momentum). |
| `rank` / `alpha` | `32` / `32.0` | LoRA rank $r$ and scaling factor $\alpha$ (automatically inherited if `initial_lora` is connected). |
| `learning_rate` | `0.0001` | Peak base learning rate (`1e-4`). |
| `lr_scheduler` | `"cosine"` | `"cosine"`, `"rex"`, `"constant_with_warmup"`, `"constant"`, `"linear"`, or `"cosine_with_restarts"`. |
| `warmup_steps` | `50` | Linear LR warmup steps (automatically clamped to at most 50% of `steps` on short runs). |
| `steps` | `1000` | Number of optimizer steps to run in this training node. |
| `grad_accum_steps` | `1` | Micro-batches accumulated in FP32 per optimizer step. |
| `gradient_checkpointing` | `True` | In-place-safe non-reentrant activation checkpointing across repeating blocks. |
| `checkpoint_every_n_steps` | `0` | Save intermediate `.safetensors` checkpoints every $N$ steps (`0` = disabled). |
| `checkpoint_prefix` / `custom_checkpoint_dir` | `"omni_lora_ckpt"` / `""` | Prefix and optional directory for periodic & emergency interrupt checkpoints (defaults to `ComfyUI/models/loras/`). |
| `lora_dtype` | `"bf16"` | Precision of trainable LoRA weights (`"bf16"`, `"fp32"`, or `"fp16"`). |
| `lora_plus_lr_ratio` | `1.0` | LoRA+ multiplier for `lora_B` learning rate relative to `lora_A` (`1.0` = balanced; `2.0–4.0` = accelerated `lora_B`). |
| `min_lr_ratio` | `0.1` | Floor learning rate fraction at the end of `cosine`, `rex`, or `linear` decay (`0.1` = 10% of peak LR). |
| `weight_decay` | `0.001` | Decoupled weight decay regularization (`1e-3`). |
| `use_rslora` | `False` | Rank-Stabilized LoRA scaling ($\alpha / \sqrt{r}$ instead of $\alpha / r$). Recommended when `rank >= 64`. |
| `include_conv_layers` | `False` | **LoCon mode**: When `True`, also adapts repeating-block ResNet/Downsample/Upsample spatial convolutions (`1×1` / `3×3`) while excluding raw input/output stems. |
| `enable_target_module_filter` + `target_module_filter` | `False` / `""` | When enabled, restricts LoRA injection to comma-separated module path substrings (e.g. `"double_blocks"`, `"single_blocks"`, `"text_embedder.blocks"`, `"layers"`). |
| `min_timestep` / `max_timestep` | `0.0` / `1.0` | Active normalized noise interval ($t \in [0, 1]$) for training split-schedule or High-Noise / Low-Noise expert models. |
| `min_snr_gamma` | `0.0` | Min-SNR-$\gamma$ loss weighting (`0.0` = disabled; `5.0` optional for style training). |
| `control_mode` | `"auto"` | Routes paired control latents via live ComfyUI model introspection: `"reference_latents"` (token-concatenated edit DiTs) or `"concat_latent_image"` (channel-concatenated / `image_cond` edit models). |
| `reference_latents_method` | `"auto"` | Optional positional-encoding method for `reference_latents` models (`"auto"`, `"offset"`, `"index"`, `"index_timestep_zero"`, `"uxo"`). `"auto"` defers to ComfyUI's native model default (`"index_timestep_zero"` is for global-modulation DiTs such as Flux 2). |
| `control_dropout` | `0.0` | Per-sample probability (`0.0 – 1.0`) of dropping the control image for CFG / unconditional fallback without leaking the clean target image. |

### Node 4: `OmniLoRA Load LoRA Network` (`OmniLoRALoader`)
Loads an existing `.safetensors` or `.pt` LoRA file from `ComfyUI/models/loras/` (or `custom_lora_path`), inspects its tensors and metadata to detect `rank`, `alpha`, `use_rslora`, and `steps_trained`, and outputs an `OMNILORA_NETWORK` bundle for resuming training via `initial_lora`.

### Node 5: `OmniLoRA Save LoRA` (`OmniLoRASave`)
Writes any `OMNILORA_NETWORK` bundle to `ComfyUI/models/loras/` (or `custom_output_dir`) in ComfyUI-native `diffusion_model.<layer>.lora_down.weight` / `lora_up.weight` `.safetensors` format, with optional `append_step_count = True` (`<name>_step_01000.safetensors`) and complete training metadata.

### Node 6: `OmniLoRA Loss Tracker` (`OmniLoRALossTracker`)
Mounts a TensorBoard-grade interactive HTML5 canvas dashboard directly on the ComfyUI node (`web/omni_lora_tracker.js`):
- **Debiased EMA & Outlier-Resistant Auto-Scaling**: Computes TensorBoard debiased EMA (`S_t / (1 - w^(t+1))`) with a $\pm 1\sigma$ rolling volatility band and 3 Y-axis framing modes (`Auto Clip [P05–P95]`, `EMA Focus`, `Full Raw`) so single-step diffusion noise spikes never flatten the curve.
- **Interactive Filtering & Zoom**: Filter points in real time by **Noise Timestep Bin** (`All`, `High t`, `Mid t`, `Low t`) or **Aspect Ratio Bucket** (`1024x1024`, `832x1216`, etc.); switch X-axis between `Step (Auto-Fit)`, `Step (0..Max)`, and `Relative Time`; box-drag to zoom, scroll-wheel zoom, or open the **Fullscreen Pop-Out Modal**.
- **Multi-Run Comparison & Live Actions**: Overlay up to 16 runs with individual visibility toggles, trigger an immediate mid-training `.safetensors` save via **`Save Ckpt`**, and export metrics as **CSV** or **PNG**.
- **Sleep / Tab-Freeze / Cloudflare Tunnel Resilient**: Backed by `OmniLoRATelemetryStore` (`/omni_lora/metrics`), which buffers all step metrics on the server and automatically syncs any steps missed while your display was asleep or your browser tab was backgrounded.

---

## Advanced Workflow Patterns

### 1. Chaining & Mid-Training Evaluation (`Trainer 1 -> KSampler -> Trainer 2`)
You can train in stages and sample test images between stages in a single graph:
```text
[Load Diffusion Model] ──> [Trainer 1 (500 steps)] ──(model)──> [KSampler (500-step preview)]
                                     │
                                  (model)
                                     ▼
                           [Trainer 2 (500 steps)] ──(model)──> [KSampler (1000-step preview)]
```
When `Trainer 2` receives a `MODEL` output from `Trainer 1` (even without `initial_lora` wired), it automatically extracts the trained LoRA weights from `Trainer 1`, restores clean unpatched base weights so adapters are never double-applied, and continues training for the next stage.

### 2. Dual-Network & Split-Timestep Expert Training
- **Sub-networks inside a single checkpoint** (e.g., MOSS-Image `text_embedder.blocks` + `layers`, or Flux / FIBO / HiDream `double_blocks` + `single_blocks`): Enable `enable_target_module_filter = True` on two chained `OmniLoRA Universal Trainer` nodes to give each sub-network its own learning rate or timestep range, and wire `Trainer 1 (lora) -> Trainer 2 (initial_lora)` to merge both into a single `.safetensors` file.
- **Separate High-Noise & Low-Noise checkpoints** (e.g., Wan 2.2 high/low noise experts): Share one `OmniLoRA VAE & CLIP Pre-Cache` node across two `OmniLoRA Universal Trainer` nodes and set `min_timestep=0.5, max_timestep=1.0` on the High-Noise trainer and `min_timestep=0.0, max_timestep=0.5` on the Low-Noise trainer.

### 3. Image Editing / Control Folder Training
Train instruction-based or paired image-to-image editing LoRAs on **any** ComfyUI edit architecture — including token-concatenated `reference_latents` DiTs (**Flux.1 Kontext**, **Flux 2 / 3 Klein**, **Qwen-Image-Edit 1.0 / 2.1**, **OmniGen2**, **Boogu**, **MOSS-Image**, **FIBO 2.0**, **Krea 2**, **JoyImage**, **Mage-Flow**) and channel-concatenated / `image_cond` models (**HiDream-E1**, **SD1.5 / SDXL InstructPix2Pix**, **Flux Fill / Control**, **Wan I2V**) — without hardcoding model-specific prompts or checklists:
1. **Prepare two paired folders**: Put edited **target ("after")** images and `.txt` edit instructions in `dataset_path`, and put matching **control ("before")** images in `control_path` with matching filename stems (any extension, e.g. `001.png` $\leftrightarrow$ `001.jpg`, or mirrored subfolders `sub/002.png` $\leftrightarrow$ `sub/002.webp`). `control_resize_mode = "preserve_aspect"` preserves native control aspect ratios on `reference_latents` DiTs and automatically aligns spatial dimensions on `concat_latent_image` models.
2. **Pre-cache (`OmniLoRAPrecomputeConditioning`)**: Target and control images are encoded sequentially through the `VAE` (target scaled, control kept raw unscaled for ComfyUI's native `extra_conds`). If your connected `CLIP` is a VLM text encoder (such as Qwen-VL, Mistral/Flux 2, Boogu, or OmniGen2), `pass_control_to_clip = "auto"` automatically passes the resized control image (`clip_vision_resolution = 384`) into `clip.tokenize(..., images=[...])` — and synchronizes horizontal flips across target latent, control latent, and VLM tokens when `random_flip = True`.
3. **Train (`OmniLoRATrainer`)**: Leave `control_mode = "auto"` and `reference_latents_method = "auto"` to defer to ComfyUI's native model structure (or override explicitly at any time), and optionally enable `control_dropout` (e.g. `0.05–0.10`) for stronger CFG behavior. The analytical VRAM ledger automatically accounts for doubled visual sequence length on `reference_latents` DiTs.
4. **Template**: Start immediately with [`example_workflows/omni_lora_image_edit_workflow.json`](./example_workflows/omni_lora_image_edit_workflow.json).

---

## Supported Model Architectures

Because `ComfyModelAdapter` routes forward passes through ComfyUI's own `BaseModel._apply_model`, `model_sampling`, `latent_format`, and `extra_conds`, any checkpoint or GGUF/FP8 model supported by ComfyUI works with `model_architecture="auto"`.

> **Verification Scope**: The 44-architecture suite below verifies **unmocked real `comfy.ldm.*` + `comfy.model_base.*` contract execution** (architecture detection, layer targeting, zero dead layers, real forward/backward gradients, gradient-checkpointing parity, `batch_size > 1`, ComfyUI `ModelPatcher` / `LoraLoader` key parity, and clean base-weight restoration) directly on your active hardware backend. End-to-end multi-hundred-step real-checkpoint benchmarks across flagship models are planned for `v2.0.0`.

The unmocked real-forward verification suite (`python -m omni_lora.tests.test_framework --cloud-verify`) tests **44 real `comfy.ldm.*` + `comfy.model_base.*` Image & Video architectures** alongside a 7-invariant FP32 + BF16 contract suite and a **14-architecture FP32 + BF16 unmocked image-editing suite**:

| Family | Verified Architectures & Capabilities |
| :--- | :--- |
| **Flux & Rectified Flow MMDiTs** | **Flux.1 Dev/Schnell/Kontext** (`flux_dev`), **Flux 2** (`flux2` 128ch), **Flux 3 Klein** (`flux3_klein` with flow-map `sample_sigmas` lookahead), **Chroma** (`chroma` with `distilled_guidance_layer`), **Chroma Radiance**, **Bria FIBO 2.0** (`fibo2` 32ch Flux.2 VAE + Qwen3-VL Perceiver blocks), **SD 3 / SD 3.5** (`sd3_5` `MMDiT` `joint_blocks`). |
| **Multimodal, LLM-Conditioned & MoE DiTs** | **Qwen-Image 1.0 & 2.1** (`qwen_image_1_0` 16ch / `qwen_image_2_1` 64ch + `llm_cond_norm` + `ref_latents`), **MOSS-Image** (`moss_image` dual `text_embedder.blocks` + `layers`), **OmniGen2** (`omnigen2` `noise_refiner` + `context_refiner`), **HiDream-I1 / E1 / O1** (`hidream_i1` 17B Sparse-MoE MMDiT with router-gate exclusion), **MiniMax H3 Image**, **Ernie-Image**, **Krea 2**, **Ideogram v4**, **JoyImage**, **Mage-Flow**, **Boogu**, **Anima**, **PixelDiT**, **Lens**. |
| **Video & Spatiotemporal 5D DiTs** | **Wan 2.1 T2V** (`wan_2_1_t2v` 16ch 5D), **Wan 2.1 I2V** (`wan_2_1_i2v` 36ch `c_concat`), **Wan 2.2 TI2V** (`wan_2_2_ti2v` 48ch 3D VAE), **Wan VACE / Animate / Camera / WanDancer**, **HunyuanVideo** (`hunyuan_video`), **LTX-Video & LTX-AV 2.0 / 2.5** (`ltxv`, `ltxav_2_0`, `ltxav_2_5` 128ch audio-video DiT), **NVIDIA Cosmos Predict2** (`cosmos_predict2`), **Genmo Mochi**, **CogVideoX**, **MiniMax H3 Video**, **Kandinsky 5**, **SeedVR2**. |
| **NextDiT, DC-AE & Linear DiTs** | **Lumina-Image 2.0 / Z-Image** (`lumina2_z_image` NextDiT SwiGLU), **AuraFlow / Sana** (`auraflow_sana` 32ch DC-AE), **PixArt-$\alpha$/$\Sigma$**, **HunyuanDiT**. |
| **UNet & Audio / 3D Diffusion** | **SDXL** (`sdxl_base` with 2816-dim sinusoidal ADM synthesis), **SD 1.5 / SD 2.x** (`sd15` with chunked outer attention + optional LoCon convolutions), **Stable Cascade**, **ACE-Step 1.5 / Stable Audio / YuE2 / MiniMax Music**, **Hunyuan3D 2.1 / Trellis2 / TripoSplat**. |

---

## Supported Hardware Backends

| Backend ID | Target Hardware | Status | Key Optimizations in `omni_lora/backends/` |
| :--- | :--- | :--- | :--- |
| **`cuda`** | NVIDIA GeForce / RTX / Data Center GPUs | Actively tested on hardware | FlashAttention-2 / Cutlass Memory-Efficient SDPA, Tensor Core `bf16`/`tf32`, `expandable_segments:True` allocator, pinned-memory async PCIe layer streaming, plus `bitsandbytes` & GGUF support. |
| **`xpu`** | Intel Arc (A770 16GB, B580 Battlemage) & Data Center Max | Actively tested on hardware | First-class `torch.xpu` backend with OneAPI Level Zero flags, XMX `bf16` runtime verification, 100% FP64-free execution, and proactive 64 MB query-chunked SDPA to stay below Level Zero's 4.00 GiB single-allocation limit. |
| **`rocm`** | AMD Radeon RX 7000/9000 & Instinct MI200/MI300 (`HIP`) | Implemented & unit-tested *([community reports welcome](https://github.com/cyberterm/ComfyUI-OmniLoRA/discussions/categories/benchmarks))* | ROCm HIP detection (`torch.version.hip`), AOTriton / FlashAttention SDPA, Matrix Core `bf16`/`fp16`, and tuning flags. |
| **`mps`** | Apple Silicon (M1 / M2 / M3 / M4 Pro, Max, Ultra) | Implemented & unit-tested *([community reports welcome](https://github.com/cyberterm/ComfyUI-OmniLoRA/discussions/categories/benchmarks))* | Metal Performance Shaders (`torch.backends.mps`) unified memory budget detection, `bf16`/`fp16` autocast, and chunked attention. |
| **`tpu`** | Google Cloud TPU (`torch_xla`) | Implemented & unit-tested *([community reports welcome](https://github.com/cyberterm/ComfyUI-OmniLoRA/discussions/categories/benchmarks))* | XLA device resolution, native Matrix Multiply Unit `bf16`, and per-step `xm.mark_step()` graph execution. |
| **`npu`** | Huawei Ascend 910A / 910B (`torch_npu` / CANN) | Implemented & unit-tested *([community reports welcome](https://github.com/cyberterm/ComfyUI-OmniLoRA/discussions/categories/benchmarks))* | DaVinci Cube `bf16` execution and CANN memory management. |
| **`hpu`** | Intel Gaudi 2 / Gaudi 3 (`habana_frameworks.torch`) | Implemented & unit-tested *([community reports welcome](https://github.com/cyberterm/ComfyUI-OmniLoRA/discussions/categories/benchmarks))* | SynapseAI HPU eager/lazy mode `htcore.mark_step()` and `bf16` execution. |
| **`cpu`** | Host CPU (x86_64 AVX-512 / ARM64) | Actively tested (CI) | OneDNN `bf16`/`fp32` execution and offline workflow verification. |

> **Have a benchmark, hardware report, or training recipe to share?** Post your `training_summary`, GPU/model stats, or sample outputs in [GitHub Discussions → Benchmarks](https://github.com/cyberterm/ComfyUI-OmniLoRA/discussions/categories/benchmarks) (or ask questions in [Discussions](https://github.com/cyberterm/ComfyUI-OmniLoRA/discussions)).

---

## Under the Hood: Optimizers & Analytical VRAM Ledger

### FP64-Free Optimizers
All four optimizers execute with `foreach=False` (or pure-FP32 factored updates) so PyTorch never allocates internal `float64` step counters or multi-tensor lists on GPUs without FP64 hardware support:
- **`radamw` (Default)**: Rectified Adam with decoupled weight decay (`torch.optim.RAdam(..., decoupled_weight_decay=True)`). Built-in variance rectification prevents early adaptive step-size spikes on both fresh runs and resumed/chained LoRA runs.
- **`adafactor`**: Sublinear-memory Adafactor (Shazeer & Stern, 2018) with factored row/column second moments (`exp_avg_sq_row` + `exp_avg_sq_col`), zero first-moment buffer (`beta1=0`), on-device RMS clipping (`d=1.0`), and unbiased 16-bit stochastic rounding. Cuts optimizer VRAM by **~75–85%** (e.g. `681 MB -> ~170 MB` at Rank 32), freeing `0.5–2.0 GB` of VRAM for additional resident base layers or higher ranks (`64–128`).
- **`adamw`**: Standard `torch.optim.AdamW`.
- **`sgd`**: Nesterov momentum SGD (`momentum=0.9`, single FP32 momentum buffer — uses 50% less optimizer VRAM than `radamw`/`adamw`).

### How the Analytical VRAM Ledger Works (`[ledger: opt=...MB, act=...MB, ckpt_in=...MB]`)
Instead of relying on static percentage heuristics, `enforce_vram_budget` inspects the live LoRA-injected model graph before Step 1 and computes a bottom-up byte budget (paired with dynamic OOM recovery to handle allocator fragmentation, temporary dequantization buffers, and backend kernel workspace variance):
1. **Usable Hardware Ceiling (`usable_vram_mb`)**: Queries live OS/driver free memory (`mem_get_info()`) plus OmniLoRA's already-resident tensors, minus the hardware driver reserve and `reserve_vram_gb`.
2. **Non-Base Resident Parameters & Trainable LoRA Weights**: Exact storage bytes of non-streamable stems, norms, embeddings, and `lora_A` / `lora_B` matrices.
3. **Optimizer State + `.grad` Buffers (`opt`)**: Computed from trainable parameter count and chosen `optimizer` (`~0.5 B/elem` for `adafactor`, `4 B/elem` for `sgd`, `8 B/elem` for `radamw`/`adamw`, plus `.grad` storage).
4. **Checkpointed Block Input Snapshots (`ckpt_in`)**: $N_{\text{blocks}} \times B \times N_{\text{tokens}} \times D_{\text{model}} \times \text{sizeof}(\text{compute dtype})$ when resident in VRAM (or a 2-block prefetch window when CPU layer streaming offloads block boundary snapshots to host RAM).
5. **Peak Single-Block Backward Recomputation & GEMM Workspace (`act`)**: Estimated from the `in_features`, `out_features`, and `rank` of every `LoRALinear` inside the largest transformer block, plus cuBLAS/oneDNN/rocBLAS workspace headroom (`64–128 MB`).
6. **Greedy Resident Fill, CPU Streaming & OOM Self-Healing**: All remaining VRAM (`max_resident_model_mb`) is filled with frozen base layers; any remaining frozen layers stay in pinned CPU RAM and stream on-demand into a pre-allocated GPU staging buffer (`2 × max_single_layer_mb`) to minimize allocator fragmentation. Because peak runtime memory also depends on allocator fragmentation and backend-specific temporary buffers, any unexpected mid-step OOM automatically demotes a slice of resident frozen layers to CPU RAM and retries the step cleanly.

---

## Troubleshooting & FAQ

- **My loss jumps up and down every step — is it broken?**
  No! Diffusion and Rectified Flow models sample a random noise timestep $t \in [0, 1]$ on every step. Steps near $t \approx 0.5$ naturally have much higher instantaneous loss than steps near $t \approx 0$ or $t \approx 1$. Watch the **EMA smoothed curve** (`avg` in console logs, or the solid line in `OmniLoRA Loss Tracker`), or filter by a single noise bin (`High` / `Mid` / `Low`) in the Loss Tracker.
- **I got a dimension mismatch error in `OmniLoRA VAE & CLIP Pre-Cache` or `Trainer`:**
  Make sure the **`type`** dropdown on your ComfyUI `CLIPLoader` / `DualCLIPLoader` matches your diffusion model (e.g. `flux` for Flux, `sd3` for SD 3.5, `qwen_image` for Qwen-Image, `wan` for Wan 2.1/2.2, `lumina2` for Lumina 2 / Z-Image, `hidream` for HiDream) and that you connected the matching `VAELoader`.
- **My PC lagged or ran out of VRAM while my browser / monitor was active:**
  Increase **`reserve_vram_gb`** on `OmniLoRA Universal Trainer` to **`1.0` or `1.5`**, or switch `optimizer` from `radamw` to **`adafactor`**.
- **Can I train if I launch ComfyUI with `--use-ck-attention` (Comfy Kitchen) or `--use-sage-attention`?**
  Yes. Comfy Kitchen INT8/SOL attention (`comfy_kitchen.int8_attention` / `sol_attn`), SageAttention (`sageattn`), and `comfy_kitchen` fused RoPE/AdaLN kernels are inference-only C++/Triton kernels without PyTorch autograd backward formulas. OmniLoRA automatically intercepts `transformer_options["optimized_attention_override"]`, `comfy.ldm.modules.attention`, pre-loaded `ComfyAttention` checkpoints, and `comfy_kitchen` / `comfy.quant_ops.ck` calls during training to route through differentiable PyTorch SDPA (`F.scaled_dot_product_attention`), and restores your inference attention backend as soon as training finishes.
- **Can I use `.avif` or `.heic` images?**
  Yes. Pillow $\ge 11.2$, `imageio`, `opencv-python` (`cv2`), and `ffmpeg` are all checked automatically. If your Python environment has an older Pillow without AVIF support, run `pip install pillow-avif-plugin` or `pip install --upgrade Pillow>=11.2.0`.
- **Hit a bug, OOM, or unsupported checkpoint?**
  Please open a [GitHub Issue](https://github.com/cyberterm/ComfyUI-OmniLoRA/issues) — including your GPU/backend, model & weight format (BF16/FP8/GGUF) and console traceback. This makes issues much faster to reproduce and fix. For general questions or sharing runs, use [GitHub Discussions](https://github.com/cyberterm/ComfyUI-OmniLoRA/discussions).

---

## Running the Unmocked Real-Forward Test & Verification Suite

You can run the full unit, autograd-fidelity, AST silent-exception guard, and 44-architecture unmocked ComfyUI verification suite from your `ComfyUI/custom_nodes` directory (the parent folder containing `omni_lora/`) without starting the ComfyUI server:

```bash
cd ComfyUI/custom_nodes

# Run all unit tests + 7-invariant unmocked real-forward contract suites (44 architectures)
python -m omni_lora.tests.test_framework

# Run the live active-backend (CUDA / XPU / ROCm / MPS / CPU) 44-architecture sweep + full test suite
python -m omni_lora.tests.test_framework --cloud-verify

# Optional: fail fast on any unexpected state-restoration exception during CI or debugging
OMNILORA_STRICT_ERRORS=1 python -m omni_lora.tests.test_framework
```

> **Diagnostics & Exception Hygiene**: All production modules route recoverable fallbacks through `omni_lora/diagnostics.py` (`log_once` / `report_exception` under the `omni_lora` logger), enforced by a repo-wide AST regression test guaranteeing zero bare `except:` or silent `except Exception: pass` handlers. Setting `OMNILORA_STRICT_ERRORS=1` re-raises unexpected errors at critical cleanup and restoration boundaries instead of falling back.

---

## License

[Apache-2.0](./LICENSE)
