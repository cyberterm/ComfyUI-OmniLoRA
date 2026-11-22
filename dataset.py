"""
Dataset & Prompt Pipeline for ComfyUI-OmniLoRA (`omni_lora/dataset.py`).

Supports two ComfyUI-native input sources (which can be used individually or combined):
1. Local Dataset Folder (`dataset_path`):
   - Scans `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` images.
   - Reads matching `<image_stem>.txt` or `<image_stem>.caption` sidecar files.
   - Falls back to `default_caption` if a sidecar `.txt` file is missing.
2. Direct ComfyUI Node Inputs (`comfy_images` + `custom_prompts`):
   - Accepts ComfyUI `IMAGE` batches (`[B, H, W, C]` float32 in `[0, 1]`).
   - Accepts multiline `custom_prompts` (one line per image, or a single shared caption).

Features:
- Automatic `trigger_word` prefixing (e.g. `"ohwx_moss, a macro botanical photo..."`).
- Automatic Aspect Ratio Bucketing (`aspect_ratio_bucketing=True`): matches portrait, square,
  and landscape images to the closest `(bucket_h, bucket_w)` multiple-of-64 resolution bucket
  around the target `resolution * resolution` pixel area budget, scales without distortion,
  and groups batches by bucket shape so `batch_size > 1` never mixes incompatible dimensions.
- Optional `random_flip`, `caption_dropout` (for CFG training), and Kohya-style `num_repeats`.
- Strictly `torch.float32` / `torch.bfloat16` (zero `torch.float64` / `fp64` operations).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .diagnostics import report_exception

SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".jfif",
    ".webp",
    ".avif",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
    ".heic",
    ".heif",
)

_AVIF_PLUGINS_REGISTERED: bool = False


def _ensure_image_format_plugins_registered() -> None:
    """Opportunistically register AVIF / HEIF Pillow plugins when installed in the Python environment."""
    global _AVIF_PLUGINS_REGISTERED
    if _AVIF_PLUGINS_REGISTERED:
        return
    _AVIF_PLUGINS_REGISTERED = True
    try:
        import pillow_avif  # type: ignore  # noqa: F401
    except ImportError:
        pass
    except Exception as exc:
        report_exception("dataset.register_pillow_avif", "Failed to initialize pillow_avif plugin", exc, level=logging.DEBUG, once=True, logger_name="dataset")
    try:
        import pillow_heif  # type: ignore

        if hasattr(pillow_heif, "register_heif_opener"):
            pillow_heif.register_heif_opener()
        if hasattr(pillow_heif, "register_avif_opener"):
            pillow_heif.register_avif_opener()
    except ImportError:
        pass
    except Exception as exc:
        report_exception("dataset.register_pillow_heif", "Failed to initialize pillow_heif plugin", exc, level=logging.DEBUG, once=True, logger_name="dataset")


def _open_pil_image_any_format(img_path: Path):
    """
    Open an image file of any supported format (`.png`, `.jpg`, `.jpeg`, `.jfif`, `.webp`,
    `.avif`, `.bmp`, `.tiff`, `.tif`, `.gif`, `.heic`, `.heif`) and return a detached RGB/RGBA-ready
    `PIL.Image.Image` with EXIF orientation applied.

    Decoding cascade for `.avif` / `.heic` / `.webp`:
      1. Native `PIL.Image.open` (Pillow >= 11.2 supports AVIF natively; also uses `pillow-avif-plugin`
         or `pillow-heif` when installed).
      2. `imageio.v3.imread` / `imageio.imread` fallback (supports AVIF via pyav/imageio plugins).
      3. `cv2.imread(..., cv2.IMREAD_UNCHANGED)` (`opencv-python`, standard in ComfyUI venvs).
    """
    from PIL import Image, ImageOps

    _ensure_image_format_plugins_registered()

    last_err: Optional[Exception] = None
    try:
        with Image.open(img_path) as raw_img:
            try:
                raw_img.seek(0)
            except (EOFError, AttributeError, ValueError) as exc:
                report_exception("dataset.pil_seek_zero", "PIL seek(0) skipped on single-frame image", exc, level=logging.DEBUG, once=True, logger_name="dataset")
            transposed = ImageOps.exif_transpose(raw_img)
            return transposed.copy()
    except Exception as exc:
        last_err = exc

    # Fallback 1: imageio (v3 or v2)
    try:
        import imageio.v3 as iio  # type: ignore

        arr = iio.imread(str(img_path))
        if arr is not None:
            if arr.ndim == 4:
                arr = arr[0]
            return Image.fromarray(arr)
    except Exception as exc:
        report_exception("dataset.decode_imageio_v3", "imageio.v3 decode fallback failed", exc, level=logging.DEBUG, once=True, logger_name="dataset")
        try:
            import imageio  # type: ignore

            arr = imageio.imread(str(img_path))
            if arr is not None:
                if arr.ndim == 4:
                    arr = arr[0]
                return Image.fromarray(arr)
        except Exception as exc_v2:
            report_exception("dataset.decode_imageio_v2", "imageio v2 decode fallback failed", exc_v2, level=logging.DEBUG, once=True, logger_name="dataset")

    # Fallback 2: OpenCV (cv2)
    try:
        import cv2  # type: ignore

        mat = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if mat is not None and getattr(mat, "size", 0) > 0:
            if mat.ndim == 2:
                return Image.fromarray(mat, mode="L")
            if mat.ndim == 3 and mat.shape[2] == 3:
                rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_mat, mode="RGB")
            if mat.ndim == 3 and mat.shape[2] == 4:
                rgba_mat = cv2.cvtColor(mat, cv2.COLOR_BGRA2RGBA)
                return Image.fromarray(rgba_mat, mode="RGBA")
    except Exception as exc:
        report_exception("dataset.decode_cv2", "OpenCV cv2 decode fallback failed", exc, level=logging.DEBUG, once=True, logger_name="dataset")

    ext_lower = img_path.suffix.lower()
    if ext_lower in (".avif", ".heic", ".heif"):
        raise RuntimeError(
            f"[OmniLoRA] Could not decode `{img_path.name}` ({ext_lower}). "
            f"Install AVIF/HEIF support in your ComfyUI Python environment with: "
            f"`pip install pillow-avif-plugin` or `pip install --upgrade Pillow>=11.2.0` "
            f"(original error: {last_err})"
        ) from last_err
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"[OmniLoRA] Failed to load image file: {img_path}")


def _probe_image_header(img_path: Path) -> Tuple[int, int, bool]:
    """
    Read only the image header `(orig_h, orig_w, has_alpha)` from disk with EXIF
    orientation applied — without decoding full pixel buffers into CPU RAM.
    Falls back to `_open_pil_image_any_format` only for formats PIL cannot open
    natively (e.g. `.avif`/`.heic` needing `imageio`/`cv2`), closing immediately.
    """
    from PIL import Image

    _ensure_image_format_plugins_registered()
    try:
        with Image.open(img_path) as raw_img:
            w, h = raw_img.size
            try:
                exif = raw_img.getexif()
                orientation = int(exif.get(274, 1)) if exif is not None else 1
                if orientation in (5, 6, 7, 8):
                    h, w = w, h
            except Exception as exc:
                report_exception("dataset.probe_exif", "Could not read EXIF orientation during header probe", exc, level=logging.DEBUG, once=True, logger_name="dataset")
            has_alpha = raw_img.mode in ("RGBA", "LA", "PA") or (
                raw_img.mode == "P" and "transparency" in raw_img.info
            )
            return int(h), int(w), bool(has_alpha)
    except Exception:
        pil_img = _open_pil_image_any_format(img_path)
        try:
            w, h = pil_img.size
            has_alpha = pil_img.mode in ("RGBA", "LA", "PA") or (
                pil_img.mode == "P" and "transparency" in pil_img.info
            )
            return int(h), int(w), bool(has_alpha)
        finally:
            try:
                pil_img.close()
            except Exception as exc:
                report_exception("dataset.pil_close_probe", "Could not close PIL image handle", exc, level=logging.DEBUG, once=True, logger_name="dataset")


class DatasetSample:
    """
    A single normalized `[3, H, W]` image sample (`[-1, 1]`) paired with its formatted caption,
    bucket geometry, optional spatial loss mask, and optional control/source image.

    For folder-backed datasets, pixel tensors are lazily decoded on demand (or streamed one
    sample at a time inside `OmniLoRAPrecomputeConditioning`) so large paired datasets never
    hold gigabytes of uncompressed float32 pixels in CPU RAM simultaneously.
    """

    def __init__(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        caption: str = "",
        source_name: str = "",
        bucket_size: Tuple[int, int] = (1024, 1024),
        loss_mask: Optional[torch.Tensor] = None,
        control_pixel_values: Optional[torch.Tensor] = None,
        control_source_name: Optional[str] = None,
        image_path: Optional[Path] = None,
        control_file_path: Optional[Path] = None,
        orig_hw: Optional[Tuple[int, int]] = None,
        has_potential_mask: bool = False,
        _loader: Optional[object] = None,
    ) -> None:
        self._pixel_values: Optional[torch.Tensor] = pixel_values
        self._loss_mask: Optional[torch.Tensor] = loss_mask
        self._control_pixel_values: Optional[torch.Tensor] = control_pixel_values
        self._decoded_target: bool = pixel_values is not None
        self._decoded_control: bool = (control_pixel_values is not None) or (control_file_path is None)
        self._released: bool = False
        self.caption: str = caption
        self.source_name: str = source_name
        self.bucket_size: Tuple[int, int] = bucket_size
        self.control_source_name: Optional[str] = control_source_name
        self.image_path: Optional[Path] = image_path
        self.control_file_path: Optional[Path] = control_file_path
        self.orig_hw: Optional[Tuple[int, int]] = orig_hw
        self.has_potential_mask: bool = bool(has_potential_mask or (loss_mask is not None))
        self._loader: Optional[object] = _loader

    @property
    def has_control(self) -> bool:
        return (
            isinstance(self._control_pixel_values, torch.Tensor)
            and self._control_pixel_values.ndim >= 3
            and self._control_pixel_values.numel() > 0
        ) or (self.control_file_path is not None)

    def _lazy_load_target(self) -> None:
        if self._decoded_target or self._released:
            return
        if self._loader is not None and self.image_path is not None:
            px, mask_t = self._loader._decode_target_from_disk(self)  # type: ignore[attr-defined]
            self._pixel_values = px
            self._loss_mask = mask_t
            self._decoded_target = True

    def _lazy_load_control(self) -> None:
        if self._decoded_control or self._released:
            return
        if self._loader is not None and self.control_file_path is not None:
            self._control_pixel_values = self._loader._decode_control_from_disk(self)  # type: ignore[attr-defined]
            self._decoded_control = True

    @property
    def pixel_values(self) -> torch.Tensor:
        self._lazy_load_target()
        if self._pixel_values is None:
            return torch.empty((0,), dtype=torch.float32, device="cpu")
        return self._pixel_values

    @pixel_values.setter
    def pixel_values(self, value: Optional[torch.Tensor]) -> None:
        self._pixel_values = value
        if isinstance(value, torch.Tensor) and value.ndim >= 3 and value.numel() > 0:
            self._decoded_target = True
            self._released = False
        else:
            self._decoded_target = False
            self._released = True

    @property
    def loss_mask(self) -> Optional[torch.Tensor]:
        self._lazy_load_target()
        return self._loss_mask

    @loss_mask.setter
    def loss_mask(self, value: Optional[torch.Tensor]) -> None:
        self._loss_mask = value

    @property
    def control_pixel_values(self) -> Optional[torch.Tensor]:
        self._lazy_load_control()
        return self._control_pixel_values

    @control_pixel_values.setter
    def control_pixel_values(self, value: Optional[torch.Tensor]) -> None:
        self._control_pixel_values = value
        if isinstance(value, torch.Tensor) and value.ndim >= 3 and value.numel() > 0:
            self._decoded_control = True
        else:
            self._decoded_control = self.control_file_path is None

    def __repr__(self) -> str:
        return (
            f"DatasetSample(source_name={self.source_name!r}, bucket_size={self.bucket_size}, "
            f"lazy={self.image_path is not None and self._pixel_values is None}, "
            f"has_control={self.has_control})"
        )


@dataclass
class OmniLoRADatasetConfig:
    """Configuration for loading and augmenting image + prompt training pairs in ComfyUI."""

    dataset_path: str = ""
    trigger_word: str = "ohwx"
    default_caption: str = "a high-resolution detailed photograph"
    resolution: int = 1024
    batch_size: int = 1
    num_repeats: int = 1
    center_crop: bool = True
    aspect_ratio_bucketing: bool = True
    random_flip: bool = True
    caption_dropout: float = 0.0
    use_alpha_mask: bool = True
    mask_unmasked_weight: float = 0.1
    seed: int = 42
    control_path: str = ""
    control_resize_mode: str = "match_target"


def resolve_aspect_ratio_bucket(
    height: int,
    width: int,
    base_resolution: int,
    step: int = 64,
    downscale_only: bool = False,
) -> Tuple[int, int]:
    """
    Compute the closest `(bucket_h, bucket_w)` aspect ratio bucket for an image of size
    `(height, width)` given a target square `base_resolution` pixel budget.

    All returned bucket dimensions are exact multiples of `step` (64 for standard >=256px
    training, or 16 for tiny unit-test resolutions <256px) and never exceed the pixel area
    budget of `base_resolution * base_resolution` so VRAM usage and DiT token counts remain
    strictly bounded across all aspect ratios.
    """
    base_res = max(16, int(base_resolution))
    if height <= 0 or width <= 0:
        return (base_res, base_res)

    if base_res >= 256:
        div = max(16, (int(step) // 16) * 16)
    elif base_res >= 32:
        raw_div = min(int(step), max(16, base_res // 4))
        div = max(16, (raw_div // 16) * 16)
    else:
        div = 8
    if downscale_only and (height * width) < (base_res * base_res):
        clamped_side = max(div, int(math.sqrt(float(height * width)) // div) * div)
        base_res = min(base_res, clamped_side)
    else:
        base_res = max(div, (base_res // div) * div)

    if height == width:
        return (base_res, base_res)

    target_area = float(base_res * base_res)

    min_side = max(div, (base_res // 2 // div) * div)
    max_side = max(base_res, ((base_res * 2) // div) * div)

    candidates: set[Tuple[int, int]] = {(base_res, base_res)}
    for bh in range(min_side, max_side + div, div):
        raw_bw = int(round((target_area / float(bh)) / float(div))) * div
        bw = max(min_side, min(max_side, raw_bw))
        while bh * bw > target_area and bw > min_side:
            bw -= div
        if bh >= min_side and bw >= min_side and (bh * bw <= target_area or bh == min_side or bw == min_side):
            candidates.add((bh, bw))
            candidates.add((bw, bh))

    target_log_ar = math.log(float(width) / float(height))
    best_bucket = (base_res, base_res)
    best_score = float("inf")

    for bh, bw in sorted(candidates):
        bucket_log_ar = math.log(float(bw) / float(bh))
        ar_diff = abs(target_log_ar - bucket_log_ar)
        area_diff = abs(float(bh * bw) - target_area) / target_area
        score = ar_diff + 0.05 * area_diff
        if score < best_score:
            best_score = score
            best_bucket = (bh, bw)

    return best_bucket


def format_caption_with_trigger(
    raw_caption: str,
    trigger_word: str,
    default_caption: str,
    caption_dropout: float = 0.0,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Format a training caption with optional `trigger_word` prefix and CFG `caption_dropout`.
    """
    active_rng = rng or random
    if caption_dropout > 0.0 and active_rng.random() < caption_dropout:
        return ""

    base = raw_caption.strip() if raw_caption and raw_caption.strip() else default_caption.strip()
    trig = trigger_word.strip()

    if not trig:
        return base
    if not base:
        return trig
    # Avoid duplicating trigger word if already present at the start of the .txt file
    if base.lower().startswith(trig.lower()):
        return base
    return f"{trig}, {base}"


def preprocess_comfy_image_tensor(
    comfy_img: torch.Tensor,
    resolution: int,
    center_crop: bool = True,
    do_flip: bool = False,
    aspect_ratio_bucketing: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Convert a single ComfyUI image tensor `[H, W, C]` (or `[1, H, W, C]`) in `[0, 1]`
    into a normalized PyTorch `[3, target_h, target_w]` tensor in `[-1, 1]` (`torch.float32`).

    - When `target_size` is provided (`(target_h, target_w)`), resizes/crops/pads to that exact bucket
      (used e.g. for `control_resize_mode="match_target"` so control and target share identical framing).
    - When `aspect_ratio_bucketing=True`:
      * `center_crop=True`: scales uniformly to cover `(target_h, target_w)` and center-crops any remainder.
      * `center_crop=False`: scales uniformly to fit inside `(target_h, target_w)` and center-pads with neutral `0.5` gray.
    - When `aspect_ratio_bucketing=False`, uses a square `(resolution, resolution)` target.
    - If `comfy_img` has 4 channels (RGBA), composites RGB over neutral `0.5` gray using the alpha channel
      before resizing so transparent backgrounds never bleed uncleaned RGB artifacts into VAE latents.
    """
    with torch.inference_mode(False):
        img = comfy_img.detach().clone().to(dtype=torch.float32, device="cpu")
        if img.ndim == 4:
            img = img[0]
        # ComfyUI IMAGE is [H, W, C] -> convert to [C, H, W]
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            if img.shape[-1] == 4:
                rgb_hwc = img[..., :3].clamp(0.0, 1.0)
                alpha_hwc = img[..., 3:4].clamp(0.0, 1.0)
                if float(alpha_hwc.min().item()) < 0.999:
                    rgb_hwc = rgb_hwc * alpha_hwc + 0.5 * (1.0 - alpha_hwc)
                img = rgb_hwc.permute(2, 0, 1).contiguous()
            else:
                img = img[..., :3].permute(2, 0, 1).contiguous()
        elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
            if img.shape[0] == 4:
                rgb_chw = img[:3, :, :].clamp(0.0, 1.0)
                alpha_chw = img[3:4, :, :].clamp(0.0, 1.0)
                if float(alpha_chw.min().item()) < 0.999:
                    rgb_chw = rgb_chw * alpha_chw + 0.5 * (1.0 - alpha_chw)
                img = rgb_chw.contiguous()
            else:
                img = img[:3, :, :].contiguous()
        else:
            raise ValueError(f"Expected ComfyUI IMAGE tensor [H, W, C], got shape {tuple(comfy_img.shape)}")

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        _, h, w = img.shape
        if target_size is not None:
            target_h, target_w = max(1, int(target_size[0])), max(1, int(target_size[1]))
        elif aspect_ratio_bucketing:
            target_h, target_w = resolve_aspect_ratio_bucket(h, w, resolution)
        else:
            target_h, target_w = (resolution, resolution)

        if center_crop:
            # Uniformly scale image so both dimensions >= (target_h, target_w), then center-crop
            scale = max(float(target_h) / float(max(1, h)), float(target_w) / float(max(1, w)))
            scaled_h = max(target_h, int(round(float(h) * scale)))
            scaled_w = max(target_w, int(round(float(w) * scale)))
            if scaled_h != h or scaled_w != w:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(scaled_h, scaled_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            if img.shape[1] != target_h or img.shape[2] != target_w:
                top = max(0, (int(img.shape[1]) - target_h) // 2)
                left = max(0, (int(img.shape[2]) - target_w) // 2)
                img = img[:, top : top + target_h, left : left + target_w]
        elif aspect_ratio_bucketing:
            # Fit inside (target_h, target_w) without cropping and pad remainder with neutral 0.5 gray
            scale = min(float(target_h) / float(max(1, h)), float(target_w) / float(max(1, w)))
            scaled_h = max(1, min(target_h, int(round(float(h) * scale))))
            scaled_w = max(1, min(target_w, int(round(float(w) * scale))))
            if scaled_h != h or scaled_w != w:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(scaled_h, scaled_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            if scaled_h != target_h or scaled_w != target_w:
                pad_top = (target_h - scaled_h) // 2
                pad_bottom = target_h - scaled_h - pad_top
                pad_left = (target_w - scaled_w) // 2
                pad_right = target_w - scaled_w - pad_left
                img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.5)
        else:
            if img.shape[1] != target_h or img.shape[2] != target_w:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        if do_flip:
            img = torch.flip(img, dims=[2])

        # Normalize from [0, 1] to [-1, 1] in pure float32 (never float64)
        img = (img.clamp(0.0, 1.0) * 2.0) - 1.0
        return img.to(dtype=torch.float32)


def preprocess_comfy_mask_tensor(
    mask_tensor: torch.Tensor,
    resolution: int,
    orig_h: Optional[int] = None,
    orig_w: Optional[int] = None,
    center_crop: bool = True,
    do_flip: bool = False,
    aspect_ratio_bucketing: bool = False,
    unmasked_weight: float = 0.1,
) -> torch.Tensor:
    """
    Apply the exact same aspect-ratio bucket resize and crop/pad geometry as
    `preprocess_comfy_image_tensor` to a 2D/3D spatial mask tensor in `[0, 1]`,
    returning a `[1, target_h, target_w]` `torch.float32` loss weight tensor in
    `[unmasked_weight, 1.0]` (with letterbox-padded border regions set to `0.0`).
    """
    with torch.inference_mode(False):
        m = mask_tensor.detach().clone().to(dtype=torch.float32, device="cpu")
        while m.ndim > 3:
            m = m[0]
        if m.ndim == 3:
            if m.shape[0] == 1:
                pass
            elif m.shape[-1] == 1:
                m = m.permute(2, 0, 1).contiguous()
            else:
                m = m[:1, :, :].contiguous()
        elif m.ndim == 2:
            m = m.unsqueeze(0)
        else:
            raise ValueError(f"Expected 2D or 3D mask tensor, got shape {tuple(mask_tensor.shape)}")

        m = m.clamp(0.0, 1.0)
        _, mh, mw = m.shape
        ref_h = int(orig_h) if orig_h is not None and orig_h > 0 else mh
        ref_w = int(orig_w) if orig_w is not None and orig_w > 0 else mw

        if mh != ref_h or mw != ref_w:
            m = F.interpolate(
                m.unsqueeze(0),
                size=(ref_h, ref_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mh, mw = ref_h, ref_w

        if aspect_ratio_bucketing:
            target_h, target_w = resolve_aspect_ratio_bucket(mh, mw, resolution)
        else:
            target_h, target_w = (resolution, resolution)

        floor_w = min(1.0, max(0.0, float(unmasked_weight)))
        m = floor_w + (1.0 - floor_w) * m

        if center_crop:
            scale = max(float(target_h) / float(max(1, mh)), float(target_w) / float(max(1, mw)))
            scaled_h = max(target_h, int(round(float(mh) * scale)))
            scaled_w = max(target_w, int(round(float(mw) * scale)))
            if scaled_h != mh or scaled_w != mw:
                m = F.interpolate(
                    m.unsqueeze(0),
                    size=(scaled_h, scaled_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            if m.shape[1] != target_h or m.shape[2] != target_w:
                top = max(0, (int(m.shape[1]) - target_h) // 2)
                left = max(0, (int(m.shape[2]) - target_w) // 2)
                m = m[:, top : top + target_h, left : left + target_w]
        elif aspect_ratio_bucketing:
            scale = min(float(target_h) / float(max(1, mh)), float(target_w) / float(max(1, mw)))
            scaled_h = max(1, min(target_h, int(round(float(mh) * scale))))
            scaled_w = max(1, min(target_w, int(round(float(mw) * scale))))
            if scaled_h != mh or scaled_w != mw:
                m = F.interpolate(
                    m.unsqueeze(0),
                    size=(scaled_h, scaled_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            if scaled_h != target_h or scaled_w != target_w:
                pad_top = (target_h - scaled_h) // 2
                pad_bottom = target_h - scaled_h - pad_top
                pad_left = (target_w - scaled_w) // 2
                pad_right = target_w - scaled_w - pad_left
                m = F.pad(m, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
        else:
            if m.shape[1] != target_h or m.shape[2] != target_w:
                m = F.interpolate(
                    m.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        if do_flip:
            m = torch.flip(m, dims=[2])

        return m.clamp(0.0, 1.0).to(dtype=torch.float32)


def _is_sidecar_mask_filename(path: Path) -> bool:
    """Return True if `path` is a `<stem>.mask.<ext>` or `<stem>_mask.<ext>` sidecar mask file."""
    stem_lower = path.stem.lower()
    return stem_lower.endswith(".mask") or stem_lower.endswith("_mask")


def _find_sidecar_mask_file(img_path: Path) -> Optional[Path]:
    """Locate an optional sidecar mask image (`<stem>.mask.png`, `<stem>_mask.png`, or `masks/<stem>.png`)."""
    parent = img_path.parent
    stem = img_path.stem
    for suffix_pat in (f"{stem}.mask", f"{stem}_mask"):
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            cand = parent / f"{suffix_pat}{ext}"
            if cand.is_file():
                return cand
    masks_dir = parent / "masks"
    if masks_dir.is_dir():
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            cand = masks_dir / f"{stem}{ext}"
            if cand.is_file():
                return cand
    return None


def _load_mask_file_as_tensor(mask_path: Path) -> torch.Tensor:
    """Load a grayscale or alpha mask image file from disk into a `[H, W]` `torch.float32` tensor in `[0, 1]`."""
    pil_img = _open_pil_image_any_format(mask_path)
    try:
        gray = pil_img.convert("L")
        w, h = gray.size
        raw_bytes = gray.tobytes()
        byte_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
        return byte_tensor.reshape(h, w).to(dtype=torch.float32) / 255.0
    finally:
        try:
            pil_img.close()
        except Exception as exc:
            report_exception("dataset.pil_close", "Could not close PIL image handle", exc, level=logging.DEBUG, once=True, logger_name="dataset")


def _load_image_and_mask_from_file(
    img_path: Path,
    use_alpha_mask: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load an image file from disk into a `[H, W, 3]` `torch.float32` tensor in `[0, 1]`,
    and optionally extract a `[H, W]` spatial mask from either:
      1. A matching `<stem>.mask.png` / `<stem>_mask.png` / `masks/<stem>.png` sidecar file, or
      2. The image's embedded RGBA/LA/palette transparency alpha channel (compositing RGB
         over neutral `0.5` gray so transparent pixels do not introduce halo artifacts).
    """
    alpha_mask: Optional[torch.Tensor] = None
    pil_img = _open_pil_image_any_format(img_path)
    try:
        has_alpha = pil_img.mode in ("RGBA", "LA", "PA") or (
            pil_img.mode == "P" and "transparency" in pil_img.info
        )
        if has_alpha:
            rgba = pil_img.convert("RGBA")
            w, h = rgba.size
            raw_bytes = rgba.tobytes()
            byte_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
            rgba_tensor = byte_tensor.reshape(h, w, 4).to(dtype=torch.float32) / 255.0
            rgb_tensor = rgba_tensor[..., :3]
            alpha_tensor = rgba_tensor[..., 3]
            if float(alpha_tensor.min().item()) < 0.999:
                alpha_3d = alpha_tensor.unsqueeze(-1)
                rgb_tensor = (rgb_tensor * alpha_3d) + 0.5 * (1.0 - alpha_3d)
                if use_alpha_mask and float(alpha_tensor.max().item()) > 1e-5:
                    alpha_mask = alpha_tensor.contiguous()
            float_tensor = rgb_tensor.contiguous()
        else:
            rgb = pil_img.convert("RGB")
            w, h = rgb.size
            raw_bytes = rgb.tobytes()
            byte_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
            float_tensor = byte_tensor.reshape(h, w, 3).to(dtype=torch.float32) / 255.0
    finally:
        try:
            pil_img.close()
        except Exception as exc:
            report_exception("dataset.pil_close", "Could not close PIL image handle", exc, level=logging.DEBUG, once=True, logger_name="dataset")

    if use_alpha_mask:
        sidecar_path = _find_sidecar_mask_file(img_path)
        if sidecar_path is not None:
            sidecar_mask = _load_mask_file_as_tensor(sidecar_path)
            if float(sidecar_mask.max().item()) > 1e-5:
                alpha_mask = sidecar_mask

    return float_tensor, alpha_mask


def _load_image_file_as_comfy_tensor(img_path: Path) -> torch.Tensor:
    """
    Load an image file from disk into a `[H, W, 3]` `torch.float32` tensor in `[0, 1]`.
    Uses PIL (`Pillow`, standard in ComfyUI) with multi-decoder fallback for AVIF/HEIF/WebP.
    """
    rgb_tensor, _ = _load_image_and_mask_from_file(img_path, use_alpha_mask=False)
    return rgb_tensor


class OmniLoRADataset:
    """
    Container of normalized image-caption training samples and mini-batch iterator
    used by `OmniLoRADatasetLoader` and `OmniLoRAPrecomputeConditioning`.
    """

    def __init__(
        self,
        config: OmniLoRADatasetConfig,
        comfy_images: Optional[torch.Tensor] = None,
        custom_prompts: Optional[str] = None,
        comfy_masks: Optional[torch.Tensor] = None,
        comfy_control_images: Optional[torch.Tensor] = None,
    ) -> None:
        self.config = config
        self._comfy_images = comfy_images
        self._custom_prompts = custom_prompts
        self._comfy_masks = comfy_masks
        self._comfy_control_images = comfy_control_images
        self.rng = random.Random(config.seed)
        self.samples: List[DatasetSample] = []
        self.bucket_counts: Dict[str, int] = {}
        self._masked_sample_count: int = 0
        self._controlled_sample_count: int = 0
        self._load_samples(
            comfy_images=comfy_images,
            custom_prompts=custom_prompts,
            comfy_masks=comfy_masks,
            comfy_control_images=comfy_control_images,
        )

    def _ensure_pixel_values_loaded(self) -> None:
        """
        Reload pixel tensors on demand if ComfyUI re-executes `OmniLoRAPrecomputeConditioning`
        (e.g. after the user changes `Load Diffusion Model`, `CLIPLoader`, or `VAELoader`)
        using a cached `OmniLoRADataset` instance whose `pixel_values` were freed after the previous run.
        Disk-backed samples stream directly from `sample.image_path` without eager reloads.
        """
        needs_reload = not self.samples or any(
            s.image_path is None
            and (s._pixel_values is None or s._pixel_values.ndim < 3 or s._pixel_values.numel() == 0)
            for s in self.samples
        )
        if needs_reload:
            self.rng = random.Random(self.config.seed)
            self.samples.clear()
            self.bucket_counts.clear()
            self._masked_sample_count = 0
            self._controlled_sample_count = 0
            self._load_samples(
                comfy_images=self._comfy_images,
                custom_prompts=self._custom_prompts,
                comfy_masks=self._comfy_masks,
                comfy_control_images=self._comfy_control_images,
            )
        else:
            for s in self.samples:
                if s.image_path is not None and s._released:
                    s._released = False

    def _read_sidecar_caption(
        self,
        img_path: Path,
        manifest_captions: Optional[Dict[str, str]] = None,
        root_folder: Optional[Path] = None,
    ) -> str:
        for ext in (".txt", ".caption"):
            caption_file = img_path.with_suffix(ext)
            if caption_file.is_file():
                with open(caption_file, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip()
        if manifest_captions:
            cand_keys: List[str] = []
            if root_folder is not None and img_path.is_relative_to(root_folder):
                rel_p = img_path.relative_to(root_folder)
                cand_keys.append(rel_p.as_posix())
                cand_keys.append(rel_p.with_suffix("").as_posix())
            cand_keys.extend((img_path.name, img_path.stem))
            for cand_key in cand_keys:
                if cand_key and cand_key in manifest_captions:
                    return manifest_captions[cand_key]
        return ""

    @staticmethod
    def _load_manifest_captions(folder: Path) -> Dict[str, str]:
        """
        Load optional HuggingFace / Kohya `metadata.jsonl`, `captions.jsonl`, or `captions.json`
        files from `folder` (and immediate concept subfolders) so datasets without per-image `.txt`
        sidecars work out of the box.
        """
        import json as _json

        out: Dict[str, str] = {}
        short_key_owner: Dict[str, str] = {}
        ambiguous_short_keys: set[str] = set()

        def _register_entry(scan_dir: Path, raw_fn: str, cap: str) -> None:
            fn_norm = raw_fn.replace("\\", "/").strip()
            while fn_norm.startswith("./"):
                fn_norm = fn_norm[2:]
            if not fn_norm or not cap:
                return
            try:
                rel_dir = scan_dir.relative_to(folder).as_posix()
            except ValueError:
                rel_dir = ""
            rel_prefix = "" if rel_dir in ("", ".") else rel_dir
            full_key = f"{rel_prefix}/{fn_norm}" if rel_prefix else fn_norm
            out[fn_norm] = cap
            out[full_key] = cap
            full_stem_rel = str(Path(full_key).with_suffix("")).replace("\\", "/")
            if full_stem_rel:
                out[full_stem_rel] = cap

            for sk in (Path(fn_norm).name, Path(fn_norm).stem):
                if not sk or sk in ambiguous_short_keys:
                    continue
                prev_owner = short_key_owner.get(sk)
                if prev_owner is not None and prev_owner != full_key:
                    ambiguous_short_keys.add(sk)
                    out.pop(sk, None)
                else:
                    short_key_owner[sk] = full_key
                    out[sk] = cap

        candidate_dirs = [folder] + [
            d
            for d in sorted(folder.iterdir())
            if d.is_dir() and not d.name.startswith(".") and d.name.lower() not in ("masks", "__pycache__")
        ]
        for scan_dir in candidate_dirs:
            for jsonl_name in ("metadata.jsonl", "captions.jsonl"):
                jl_path = scan_dir / jsonl_name
                if jl_path.is_file():
                    try:
                        with open(jl_path, "r", encoding="utf-8", errors="ignore") as f:
                            jl_lines = f.read().splitlines()
                        for raw_line in jl_lines:
                            line = raw_line.strip()
                            if not line:
                                continue
                            rec = _json.loads(line)
                            if isinstance(rec, dict):
                                fn = str(
                                    rec.get("file_name")
                                    or rec.get("image")
                                    or rec.get("file")
                                    or rec.get("filename")
                                    or ""
                                ).strip()
                                cap = str(
                                    rec.get("text")
                                    or rec.get("caption")
                                    or rec.get("prompt")
                                    or ""
                                ).strip()
                                if fn and cap:
                                    _register_entry(scan_dir, fn, cap)
                    except Exception as exc:
                        report_exception(
                            f"dataset.parse_jsonl.{jl_path.name}",
                            f"Could not parse caption metadata file '{jl_path}'",
                            exc,
                            level=logging.WARNING,
                            once=True,
                            logger_name="dataset",
                        )
            for json_name in ("captions.json", "metadata.json"):
                j_path = scan_dir / json_name
                if j_path.is_file():
                    try:
                        with open(j_path, "r", encoding="utf-8", errors="ignore") as f:
                            parsed = _json.loads(f.read())
                        if isinstance(parsed, dict):
                            for k, v in parsed.items():
                                if isinstance(v, str) and v.strip():
                                    _register_entry(scan_dir, str(k), v.strip())
                                elif isinstance(v, dict):
                                    cap = str(v.get("text") or v.get("caption") or v.get("prompt") or "").strip()
                                    if cap:
                                        _register_entry(scan_dir, str(k), cap)
                    except Exception as exc:
                        report_exception(
                            f"dataset.parse_json.{j_path.name}",
                            f"Could not parse caption metadata file '{j_path}'",
                            exc,
                            level=logging.WARNING,
                            once=True,
                            logger_name="dataset",
                        )
        return out

    @staticmethod
    def _discover_dataset_image_files(folder: Path) -> List[Path]:
        """
        Discover supported image files in `folder` and any nested concept subfolders
        (e.g. Kohya `10_ohwx_subject/*.png`), excluding `masks/`, `.git/`, `__pycache__/`,
        and hidden directories starting with `.`.
        """
        discovered: List[Path] = []

        def _walk_dir(cur: Path) -> None:
            try:
                entries = sorted(cur.iterdir())
            except OSError as exc:
                report_exception(
                    "dataset.walk_dir",
                    f"Skipping unreadable dataset directory '{cur}'",
                    exc,
                    level=logging.WARNING,
                    once=True,
                    logger_name="dataset",
                )
                return
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_file():
                    if (
                        entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                        and not _is_sidecar_mask_filename(entry)
                    ):
                        discovered.append(entry)
                elif entry.is_dir() and entry.name.lower() not in ("masks", "mask", "__pycache__", ".git"):
                    _walk_dir(entry)

        _walk_dir(folder)
        return discovered

    @classmethod
    def _index_control_folder(cls, control_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
        """
        Build lookup maps for a control image folder:
          - `by_rel_stem`: normalized lowercase relative path without extension (e.g. `"sub/001"`) -> `Path`
          - `by_stem`: lowercase filename stem (e.g. `"001"`) -> `Path`
        Allows matching target & control images across different extensions (e.g. `.png` <-> `.jpg`)
        and mirrored subfolder layouts (`dataset/sub/001.png` <-> `control/sub/001.webp`) with flat fallback.
        """
        by_rel_stem: Dict[str, Path] = {}
        by_stem: Dict[str, Path] = {}
        ambiguous_stems: set[str] = set()
        for ctrl_path in cls._discover_dataset_image_files(control_dir):
            try:
                rel_no_ext = str(ctrl_path.relative_to(control_dir).with_suffix("")).replace("\\", "/").lower()
            except ValueError:
                rel_no_ext = ctrl_path.stem.lower()
            by_rel_stem.setdefault(rel_no_ext, ctrl_path)
            stem_low = ctrl_path.stem.lower()
            if stem_low in ambiguous_stems:
                continue
            if stem_low in by_stem:
                del by_stem[stem_low]
                ambiguous_stems.add(stem_low)
            else:
                by_stem[stem_low] = ctrl_path
        return by_rel_stem, by_stem

    @staticmethod
    def _match_control_file(
        img_path: Path,
        dataset_root: Path,
        by_rel_stem: Dict[str, Path],
        by_stem: Dict[str, Path],
    ) -> Optional[Path]:
        """Locate the matching control image for `img_path` by mirrored relative path, then by filename stem."""
        try:
            rel_no_ext = str(img_path.relative_to(dataset_root).with_suffix("")).replace("\\", "/").lower()
            if rel_no_ext in by_rel_stem:
                return by_rel_stem[rel_no_ext]
        except ValueError:
            pass
        return by_stem.get(img_path.stem.lower())

    def _preprocess_control_tensor(
        self,
        ctrl_img: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Preprocess a control image tensor according to `self.config.control_resize_mode`:
          - `"match_target"` (default): resize/crop/pad to the exact `(target_h, target_w)` bucket.
          - `"preserve_aspect"`: resize into the control image's own aspect-ratio bucket.
        """
        mode = str(self.config.control_resize_mode or "match_target").strip().lower()
        if mode == "preserve_aspect":
            return preprocess_comfy_image_tensor(
                ctrl_img,
                resolution=self.config.resolution,
                center_crop=self.config.center_crop,
                do_flip=False,
                aspect_ratio_bucketing=self.config.aspect_ratio_bucketing,
            )
        return preprocess_comfy_image_tensor(
            ctrl_img,
            resolution=self.config.resolution,
            center_crop=self.config.center_crop,
            do_flip=False,
            aspect_ratio_bucketing=self.config.aspect_ratio_bucketing,
            target_size=target_hw,
        )

    def _decode_target_from_disk(
        self,
        sample: DatasetSample,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode and preprocess a single disk-backed sample's target image and optional spatial loss mask."""
        assert sample.image_path is not None
        comfy_t, raw_mask = _load_image_and_mask_from_file(
            sample.image_path,
            use_alpha_mask=bool(self.config.use_alpha_mask),
        )
        orig_h, orig_w = int(comfy_t.shape[0]), int(comfy_t.shape[1])
        pixel_values = preprocess_comfy_image_tensor(
            comfy_t,
            resolution=self.config.resolution,
            center_crop=self.config.center_crop,
            do_flip=False,
            aspect_ratio_bucketing=self.config.aspect_ratio_bucketing,
            target_size=sample.bucket_size,
        )
        target_hw = (int(pixel_values.shape[1]), int(pixel_values.shape[2]))
        loss_mask: Optional[torch.Tensor] = None
        if raw_mask is not None and self.config.use_alpha_mask:
            loss_mask = preprocess_comfy_mask_tensor(
                raw_mask,
                resolution=self.config.resolution,
                orig_h=orig_h,
                orig_w=orig_w,
                center_crop=self.config.center_crop,
                do_flip=False,
                aspect_ratio_bucketing=self.config.aspect_ratio_bucketing,
                unmasked_weight=self.config.mask_unmasked_weight,
            )
        elif (
            not self.config.center_crop
            and self.config.aspect_ratio_bucketing
            and (orig_h != target_hw[0] or orig_w != target_hw[1])
        ):
            ones_mask = torch.ones((orig_h, orig_w), dtype=torch.float32)
            loss_mask = preprocess_comfy_mask_tensor(
                ones_mask,
                resolution=self.config.resolution,
                orig_h=orig_h,
                orig_w=orig_w,
                center_crop=False,
                do_flip=False,
                aspect_ratio_bucketing=True,
                unmasked_weight=1.0,
            )
            if float(loss_mask.min().item()) >= 0.999:
                loss_mask = None
        return pixel_values, loss_mask

    def _decode_control_from_disk(self, sample: DatasetSample) -> Optional[torch.Tensor]:
        """Decode and preprocess a single disk-backed sample's paired control image."""
        if sample.control_file_path is None:
            return None
        ctrl_raw = _load_image_file_as_comfy_tensor(sample.control_file_path)
        return self._preprocess_control_tensor(ctrl_raw, target_hw=sample.bucket_size)

    def decode_sample_tensors(
        self,
        sample: DatasetSample,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Stream-decode `(pixel_values, loss_mask, control_pixel_values)` for a single sample.
        For lazy disk-backed samples that have not been eagerly cached on `sample`, tensors are
        decoded fresh and returned WITHOUT pinning them onto `sample` so peak host RAM remains
        O(1 image pair) during `OmniLoRAPrecomputeConditioning`.
        """
        if sample.image_path is not None and sample._pixel_values is None:
            px, mask_t = self._decode_target_from_disk(sample)
        else:
            px = sample.pixel_values
            mask_t = sample.loss_mask

        if sample.control_file_path is not None and sample._control_pixel_values is None:
            ctrl_px = self._decode_control_from_disk(sample)
        else:
            ctrl_px = sample.control_pixel_values
        return px, mask_t, ctrl_px

    def decode_sample_control_tensor(self, sample: DatasetSample) -> Optional[torch.Tensor]:
        """Stream-decode only a single sample's control tensor without caching full pixels on `sample`."""
        if sample.control_file_path is not None and sample._control_pixel_values is None:
            return self._decode_control_from_disk(sample)
        return sample.control_pixel_values

    def _record_lazy_disk_sample(
        self,
        img_file: Path,
        orig_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
        caption: str,
        has_mask: bool = False,
        control_file_path: Optional[Path] = None,
        control_source_name: Optional[str] = None,
    ) -> None:
        bh, bw = int(target_hw[0]), int(target_hw[1])
        bucket_key = f"{bw}x{bh}"
        self.bucket_counts[bucket_key] = self.bucket_counts.get(bucket_key, 0) + 1
        if has_mask:
            self._masked_sample_count += 1
        if control_file_path is not None:
            self._controlled_sample_count += 1
        self.samples.append(
            DatasetSample(
                pixel_values=None,
                caption=caption,
                source_name=img_file.name,
                bucket_size=(bh, bw),
                loss_mask=None,
                control_pixel_values=None,
                control_source_name=control_source_name,
                image_path=img_file,
                control_file_path=control_file_path,
                orig_hw=orig_hw,
                has_potential_mask=has_mask,
                _loader=self,
            )
        )

    def _record_sample(
        self,
        pixel_values: torch.Tensor,
        caption: str,
        source_name: str,
        loss_mask: Optional[torch.Tensor] = None,
        control_pixel_values: Optional[torch.Tensor] = None,
        control_source_name: Optional[str] = None,
    ) -> None:
        bh = int(pixel_values.shape[1])
        bw = int(pixel_values.shape[2])
        bucket_key = f"{bw}x{bh}"
        self.bucket_counts[bucket_key] = self.bucket_counts.get(bucket_key, 0) + 1
        if loss_mask is not None:
            self._masked_sample_count += 1
        if control_pixel_values is not None:
            self._controlled_sample_count += 1
        self.samples.append(
            DatasetSample(
                pixel_values=pixel_values,
                caption=caption,
                source_name=source_name,
                bucket_size=(bh, bw),
                loss_mask=loss_mask,
                control_pixel_values=control_pixel_values,
                control_source_name=control_source_name,
            )
        )

    @staticmethod
    def _extract_comfy_mask_for_index(
        comfy_masks: Optional[torch.Tensor],
        idx: int,
    ) -> Optional[torch.Tensor]:
        """Extract sample `idx`'s 2D `[H, W]` mask from an optional ComfyUI `MASK` batch."""
        if not isinstance(comfy_masks, torch.Tensor) or comfy_masks.numel() == 0:
            return None
        m = comfy_masks.detach()
        if m.ndim == 2:
            single = m
        elif m.ndim == 3:
            single = m[idx] if idx < int(m.shape[0]) else m[-1]
        elif m.ndim == 4:
            sub = m[idx] if idx < int(m.shape[0]) else m[-1]
            single = sub[0] if sub.shape[0] == 1 else sub[..., 0]
        else:
            return None
        single_f = single.to(dtype=torch.float32, device="cpu").clamp(0.0, 1.0)
        # Ignore degenerate all-zero masks (e.g. ComfyUI LoadImage default MASK output on RGB images)
        if float(single_f.max().item()) <= 1e-5:
            return None
        return single_f

    def _load_samples(
        self,
        comfy_images: Optional[torch.Tensor],
        custom_prompts: Optional[str],
        comfy_masks: Optional[torch.Tensor] = None,
        comfy_control_images: Optional[torch.Tensor] = None,
    ) -> None:
        # 0. Optional control image folder index for paired image-editing datasets
        control_folder_str = (self.config.control_path or "").strip().strip("\"'")
        control_dir: Optional[Path] = None
        ctrl_by_rel: Dict[str, Path] = {}
        ctrl_by_stem: Dict[str, Path] = {}
        if control_folder_str:
            control_dir = Path(control_folder_str).expanduser()
            if not control_dir.is_dir():
                raise ValueError(
                    f"[OmniLoRA] control_path directory not found: '{control_dir}'. "
                    f"Leave `control_path` empty for standard text-to-image training, or provide a valid folder "
                    f"containing control/source images with matching filenames."
                )
            ctrl_by_rel, ctrl_by_stem = self._index_control_folder(control_dir)

        # 1. Index local dataset folder lazily via image headers (zero full pixel decode in RAM)
        folder_str = (self.config.dataset_path or "").strip().strip("\"'")
        folder_image_count = 0
        folder_control_matches = 0
        first_target_stem = ""
        if folder_str:
            folder = Path(folder_str).expanduser()
            if folder.is_dir():
                manifest_captions = self._load_manifest_captions(folder)
                image_files = self._discover_dataset_image_files(folder)
                for img_file in image_files:
                    folder_image_count += 1
                    if not first_target_stem:
                        first_target_stem = img_file.stem
                    raw_cap = self._read_sidecar_caption(
                        img_file,
                        manifest_captions=manifest_captions,
                        root_folder=folder,
                    )
                    orig_h, orig_w, has_alpha = _probe_image_header(img_file)
                    if self.config.aspect_ratio_bucketing:
                        target_hw = resolve_aspect_ratio_bucket(orig_h, orig_w, self.config.resolution)
                    else:
                        target_hw = (self.config.resolution, self.config.resolution)

                    has_sidecar_mask = bool(
                        self.config.use_alpha_mask and _find_sidecar_mask_file(img_file) is not None
                    )
                    has_letterbox_mask = False
                    if (
                        not self.config.center_crop
                        and self.config.aspect_ratio_bucketing
                        and (orig_h != target_hw[0] or orig_w != target_hw[1])
                    ):
                        scale = min(
                            float(target_hw[0]) / float(max(1, orig_h)),
                            float(target_hw[1]) / float(max(1, orig_w)),
                        )
                        scaled_h = max(1, min(target_hw[0], int(round(float(orig_h) * scale))))
                        scaled_w = max(1, min(target_hw[1], int(round(float(orig_w) * scale))))
                        has_letterbox_mask = (scaled_h != target_hw[0]) or (scaled_w != target_hw[1])

                    has_mask = bool(
                        has_sidecar_mask
                        or (self.config.use_alpha_mask and has_alpha)
                        or has_letterbox_mask
                    )

                    matched_ctrl: Optional[Path] = None
                    control_source_name: Optional[str] = None
                    if control_dir is not None:
                        matched_ctrl = self._match_control_file(
                            img_file,
                            dataset_root=folder,
                            by_rel_stem=ctrl_by_rel,
                            by_stem=ctrl_by_stem,
                        )
                        if matched_ctrl is not None:
                            try:
                                control_source_name = str(matched_ctrl.relative_to(control_dir)).replace("\\", "/")
                            except ValueError:
                                control_source_name = matched_ctrl.name
                            folder_control_matches += 1

                    caption = format_caption_with_trigger(
                        raw_caption=raw_cap,
                        trigger_word=self.config.trigger_word,
                        default_caption=self.config.default_caption,
                        caption_dropout=0.0,
                        rng=self.rng,
                    )
                    self._record_lazy_disk_sample(
                        img_file=img_file,
                        orig_hw=(orig_h, orig_w),
                        target_hw=target_hw,
                        caption=caption,
                        has_mask=has_mask,
                        control_file_path=matched_ctrl,
                        control_source_name=control_source_name,
                    )

        if control_dir is not None and folder_image_count > 0 and folder_control_matches == 0:
            sample_ctrl_stem = next(
                iter(ctrl_by_stem.keys()),
                next(iter(ctrl_by_rel.keys()), "(no images found)"),
            )
            raise ValueError(
                f"[OmniLoRA] control_path='{control_dir}' matched 0 of {folder_image_count} target images "
                f"in '{folder_str}'. Target and control images must share matching filename stems "
                f"(e.g. target '{first_target_stem}.png' <-> control '{first_target_stem}.jpg'; "
                f"found example control stem '{sample_ctrl_stem}')."
            )

        # 2. Load from direct ComfyUI IMAGE batch input if connected
        if comfy_images is not None:
            imgs = comfy_images.detach()
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(0)
            ctrl_imgs: Optional[torch.Tensor] = None
            if isinstance(comfy_control_images, torch.Tensor) and comfy_control_images.numel() > 0:
                ctrl_imgs = comfy_control_images.detach()
                if ctrl_imgs.ndim == 3:
                    ctrl_imgs = ctrl_imgs.unsqueeze(0)
                if int(ctrl_imgs.shape[0]) not in (1, int(imgs.shape[0])):
                    raise ValueError(
                        f"[OmniLoRA] control_images batch size ({int(ctrl_imgs.shape[0])}) must match "
                        f"images batch size ({int(imgs.shape[0])}) or be 1 for single-control broadcasting."
                    )

            prompt_lines = [
                line.strip()
                for line in (custom_prompts or "").splitlines()
                if line.strip()
            ]
            for idx in range(imgs.shape[0]):
                if idx < len(prompt_lines):
                    raw_cap = prompt_lines[idx]
                elif len(prompt_lines) == 1:
                    raw_cap = prompt_lines[0]
                else:
                    raw_cap = self.config.default_caption

                single_img = imgs[idx]
                orig_h = int(single_img.shape[0]) if single_img.ndim == 3 and single_img.shape[-1] in (1, 3, 4) else int(single_img.shape[-2])
                orig_w = int(single_img.shape[1]) if single_img.ndim == 3 and single_img.shape[-1] in (1, 3, 4) else int(single_img.shape[-1])

                raw_mask = self._extract_comfy_mask_for_index(comfy_masks, idx)
                if raw_mask is None and self.config.use_alpha_mask and single_img.ndim == 3:
                    if single_img.shape[-1] == 4:
                        alpha_ch = single_img[..., 3].to(dtype=torch.float32, device="cpu").clamp(0.0, 1.0)
                        if float(alpha_ch.min().item()) < 0.999 and float(alpha_ch.max().item()) > 1e-5:
                            raw_mask = alpha_ch
                    elif single_img.shape[0] == 4:
                        alpha_ch = single_img[3, :, :].to(dtype=torch.float32, device="cpu").clamp(0.0, 1.0)
                        if float(alpha_ch.min().item()) < 0.999 and float(alpha_ch.max().item()) > 1e-5:
                            raw_mask = alpha_ch

                pixel_values = preprocess_comfy_image_tensor(
                    single_img,
                    resolution=self.config.resolution,
                    center_crop=self.config.center_crop,
                    do_flip=False,
                    aspect_ratio_bucketing=self.config.aspect_ratio_bucketing,
                )
                target_hw = (int(pixel_values.shape[1]), int(pixel_values.shape[2]))
                loss_mask = None
                if raw_mask is not None and (comfy_masks is not None or self.config.use_alpha_mask):
                    loss_mask = preprocess_comfy_mask_tensor(
                        raw_mask,
                        resolution=self.config.resolution,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        center_crop=self.config.center_crop,
                        do_flip=False,
                        aspect_ratio_bucketing=self.config.aspect_ratio_bucketing,
                        unmasked_weight=self.config.mask_unmasked_weight,
                    )
                elif (
                    not self.config.center_crop
                    and self.config.aspect_ratio_bucketing
                    and (orig_h != target_hw[0] or orig_w != target_hw[1])
                ):
                    ones_mask = torch.ones((orig_h, orig_w), dtype=torch.float32)
                    loss_mask = preprocess_comfy_mask_tensor(
                        ones_mask,
                        resolution=self.config.resolution,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        center_crop=False,
                        do_flip=False,
                        aspect_ratio_bucketing=True,
                        unmasked_weight=1.0,
                    )
                    if float(loss_mask.min().item()) >= 0.999:
                        loss_mask = None

                control_pixel_values = None
                control_source_name = None
                if ctrl_imgs is not None:
                    ctrl_idx = idx if idx < int(ctrl_imgs.shape[0]) else (0 if int(ctrl_imgs.shape[0]) == 1 else -1)
                    if ctrl_idx >= 0:
                        control_pixel_values = self._preprocess_control_tensor(
                            ctrl_imgs[ctrl_idx],
                            target_hw=target_hw,
                        )
                        control_source_name = f"comfy_control_{ctrl_idx:03d}"

                caption = format_caption_with_trigger(
                    raw_caption=raw_cap,
                    trigger_word=self.config.trigger_word,
                    default_caption=self.config.default_caption,
                    caption_dropout=0.0,
                    rng=self.rng,
                )
                self._record_sample(
                    pixel_values=pixel_values,
                    caption=caption,
                    source_name=f"comfy_image_{idx:03d}",
                    loss_mask=loss_mask,
                    control_pixel_values=control_pixel_values,
                    control_source_name=control_source_name,
                )

        if not self.samples:
            raise ValueError(
                "No training images found! Provide either a valid `dataset_path` folder containing "
                "images (.png/.jpg/.webp) + optional .txt captions, or connect a ComfyUI `IMAGE` "
                "node into `images`."
            )

    @property
    def num_unique_images(self) -> int:
        return len(self.samples)

    @property
    def num_controlled_images(self) -> int:
        return self._controlled_sample_count

    @property
    def total_effective_samples(self) -> int:
        return len(self.samples) * max(1, self.config.num_repeats)

    def build_streaming_batch_plans(self) -> List[List[DatasetSample]]:
        """
        Group repeated dataset samples by target aspect ratio bucket `(bucket_h, bucket_w)` and
        slice into lightweight `List[DatasetSample]` mini-batch plans WITHOUT decoding any pixel
        buffers from disk. Used by `OmniLoRAPrecomputeConditioning` to stream-encode one image
        pair at a time in O(1) host RAM.
        """
        self._ensure_pixel_values_loaded()

        expanded: List[DatasetSample] = []
        for _ in range(max(1, self.config.num_repeats)):
            expanded.extend(self.samples)

        bsz = max(1, self.config.batch_size)
        bucket_groups: Dict[Tuple[int, int], List[DatasetSample]] = {}
        for sample in expanded:
            b_key = (int(sample.bucket_size[0]), int(sample.bucket_size[1]))
            bucket_groups.setdefault(b_key, []).append(sample)

        plans: List[List[DatasetSample]] = []
        for group_samples in bucket_groups.values():
            for i in range(0, len(group_samples), bsz):
                plans.append(group_samples[i : i + bsz])
        return plans

    def build_batches_full(
        self,
    ) -> Tuple[
        List[torch.Tensor],
        List[List[str]],
        List[Optional[torch.Tensor]],
        List[List[Optional[torch.Tensor]]],
        List[List[Optional[str]]],
    ]:
        """
        Group repeated dataset samples by target aspect ratio bucket `(bucket_h, bucket_w)` and
        slice into mini-batches of `(pixel_batch, prompt_batch, mask_batch, control_batch, control_sources)` where:
          - `pixel_batch` has shape `[B, 3, bucket_h, bucket_w]` (`torch.float32`, `[-1, 1]`)
          - `mask_batch` has shape `[B, 1, bucket_h, bucket_w]` (`torch.float32`, `[0, 1]`) or `None`
          - `control_batch` is a per-sample `List[Optional[torch.Tensor]]` (each `[3, H_c, W_c]` in `[-1, 1]` or `None`),
            allowing mixed controlled/uncontrolled samples and independent control aspect ratios (`preserve_aspect`).
          - `control_sources` is a per-sample `List[Optional[str]]` identifying each control image for CLIP cache keying.
        """
        plans = self.build_streaming_batch_plans()

        pixel_batches: List[torch.Tensor] = []
        prompt_batches: List[List[str]] = []
        mask_batches: List[Optional[torch.Tensor]] = []
        control_batches: List[List[Optional[torch.Tensor]]] = []
        control_source_batches: List[List[Optional[str]]] = []

        for chunk in plans:
            bh, bw = int(chunk[0].bucket_size[0]), int(chunk[0].bucket_size[1])
            decoded_chunk = [self.decode_sample_tensors(s) for s in chunk]
            pixels = torch.stack([t[0] for t in decoded_chunk], dim=0).to(dtype=torch.float32)
            prompts = [s.caption for s in chunk]
            if any(t[1] is not None for t in decoded_chunk):
                full_ones: Optional[torch.Tensor] = None
                chunk_masks: List[torch.Tensor] = []
                for _, m_t, _ in decoded_chunk:
                    if m_t is not None and m_t.shape[-2:] == (bh, bw):
                        chunk_masks.append(m_t)
                    else:
                        if full_ones is None:
                            full_ones = torch.ones((1, bh, bw), dtype=torch.float32, device="cpu")
                        chunk_masks.append(full_ones)
                mask_batch: Optional[torch.Tensor] = torch.stack(chunk_masks, dim=0).to(dtype=torch.float32)
            else:
                mask_batch = None

            ctrl_chunk = [t[2] for t in decoded_chunk]
            ctrl_src_chunk = [s.control_source_name for s in chunk]

            pixel_batches.append(pixels)
            prompt_batches.append(prompts)
            mask_batches.append(mask_batch)
            control_batches.append(ctrl_chunk)
            control_source_batches.append(ctrl_src_chunk)

        return pixel_batches, prompt_batches, mask_batches, control_batches, control_source_batches

    def build_batches_with_masks(
        self,
    ) -> Tuple[List[torch.Tensor], List[List[str]], List[Optional[torch.Tensor]]]:
        """
        Group repeated dataset samples by aspect ratio bucket `(bucket_h, bucket_w)` and
        slice into mini-batches of `(pixel_batch, prompt_batch, mask_batch)` where:
          - `pixel_batch` has shape `[B, 3, bucket_h, bucket_w]` (`torch.float32`, `[-1, 1]`)
          - `mask_batch` has shape `[B, 1, bucket_h, bucket_w]` (`torch.float32`, `[0, 1]`) or `None`.
        """
        pixel_batches, prompt_batches, mask_batches, _, _ = self.build_batches_full()
        return pixel_batches, prompt_batches, mask_batches

    def build_batches(self) -> Tuple[List[torch.Tensor], List[List[str]]]:
        """
        Group repeated dataset samples by aspect ratio bucket `(bucket_h, bucket_w)` and
        slice into mini-batches of `(pixel_batch, prompt_batch)` where `pixel_batch` has
        shape `[B, 3, bucket_h, bucket_w]` (`torch.float32`, `[-1, 1]`).
        """
        pixel_batches, prompt_batches, _, _, _ = self.build_batches_full()
        return pixel_batches, prompt_batches

    def release_pixel_memory(self) -> None:
        """
        Release any in-memory `[3, H, W]` pixel tensors, control image tensors, and `[1, H, W]` mask
        tensors after `OmniLoRAPrecomputeConditioning` has encoded all samples into latents,
        freeing host RAM. Disk-backed samples remain streamable for future re-runs.
        """
        placeholder = torch.empty((0,), dtype=torch.float32, device="cpu")
        for sample in self.samples:
            if sample.image_path is not None:
                sample._pixel_values = None
                sample._loss_mask = None
                sample._control_pixel_values = None
                sample._decoded_target = False
                sample._decoded_control = sample.control_file_path is None
                sample._released = False
            else:
                sample.pixel_values = placeholder
                sample.loss_mask = None
                sample.control_pixel_values = None

    def summary_dict(self) -> Dict[str, object]:
        return {
            "unique_images": self.num_unique_images,
            "num_repeats": self.config.num_repeats,
            "total_effective_samples": self.total_effective_samples,
            "resolution": self.config.resolution,
            "aspect_ratio_bucketing": self.config.aspect_ratio_bucketing,
            "aspect_ratio_buckets": dict(self.bucket_counts),
            "random_flip": self.config.random_flip,
            "caption_dropout": self.config.caption_dropout,
            "use_alpha_mask": self.config.use_alpha_mask,
            "mask_unmasked_weight": self.config.mask_unmasked_weight,
            "masked_images": self._masked_sample_count,
            "control_path": self.config.control_path,
            "control_resize_mode": self.config.control_resize_mode,
            "controlled_images": self._controlled_sample_count,
            "batch_size": self.config.batch_size,
            "trigger_word": self.config.trigger_word,
            "sample_captions": [s.caption for s in self.samples[:3]],
        }
