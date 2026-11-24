"""
Portable `.safetensors` Disk Cache for `OmniLoRAPrecomputeConditioning`.

Stores unique pre-encoded VAE latents, CLIP/LLM text embeddings, pooled outputs,
masks, control latents, and unconditional embeddings inside `<dataset_path>/.omni_lora_cache/`
so repeated local runs or Windows -> Cloud/TPU transfers skip VAE and text-encoder
execution completely.

Design guarantees:
- 100% pickle-free (`.safetensors` tensor storage + JSON header metadata).
- Portable fingerprinting: hashes relative filenames, byte sizes, captions, and
  dataset/encoder configuration (never absolute paths or file `mtime`s) so a cache
  built on a local Windows GPU matches seamlessly when zipped and uploaded to
  Colab / Kaggle / GCP Linux.
- Multi-model safe: encodes `(latent_channels, context_dim, vae_class, clip_class,
  model_arch)` in the header and filename so multiple model caches (e.g. MOSS-Image
  and Flux) coexist in the same dataset folder without collisions.
- Atomic writes (`.tmp` -> `os.replace`) so an interrupted save never leaves a corrupt file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import safe_open, save_file

from .diagnostics import report_exception

CACHE_DIR_NAME = ".omni_lora_cache"
CACHE_FORMAT_VERSION = "1"


def resolve_dataset_cache_dir(dataset: Any) -> Optional[str]:
    """Return `<dataset_path>/.omni_lora_cache` when `dataset` is backed by a folder on disk."""
    cfg = getattr(dataset, "config", None)
    ds_path = getattr(cfg, "dataset_path", "") if cfg is not None else ""
    if not isinstance(ds_path, str) or not ds_path.strip():
        return None
    norm_path = os.path.abspath(ds_path.strip())
    if not os.path.isdir(norm_path):
        return None
    return os.path.join(norm_path, CACHE_DIR_NAME)


def compute_dataset_fingerprint(
    dataset: Any,
    clip_prompt_template: str = "",
    pass_control_to_clip: str = "auto",
    clip_vision_resolution: int = 384,
) -> str:
    """
    Compute a portable SHA-256 fingerprint (16 hex chars) of the dataset's unique samples
    and conditioning-relevant settings. Uses relative filenames, byte sizes, and formatted
    captions (never absolute paths or timestamps) so caches transfer cleanly across OSes.
    """
    cfg = getattr(dataset, "config", None)
    ds_root = os.path.abspath(getattr(cfg, "dataset_path", "") or "") if cfg is not None else ""

    def _portable_name_and_size(path_val: Optional[str]) -> Tuple[str, int]:
        if not path_val or not isinstance(path_val, str):
            return ("", 0)
        p_norm = os.path.abspath(path_val)
        if ds_root and p_norm.startswith(ds_root):
            rel = os.path.relpath(p_norm, ds_root).replace("\\", "/")
        else:
            rel = os.path.basename(p_norm)
        size_b = 0
        try:
            if os.path.isfile(p_norm):
                size_b = int(os.path.getsize(p_norm))
        except OSError:
            size_b = 0
        return (rel, size_b)

    sample_entries: List[Dict[str, Any]] = []
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list) and samples:
        for s in samples:
            img_rel, img_size = _portable_name_and_size(getattr(s, "image_path", None))
            ctrl_rel, ctrl_size = _portable_name_and_size(getattr(s, "control_path", None))
            if not ctrl_rel:
                ctrl_rel = str(getattr(s, "control_source_name", "") or "")
            sample_entries.append(
                {
                    "img": img_rel or str(getattr(s, "source_name", "") or ""),
                    "size": img_size,
                    "cap": str(getattr(s, "caption", "") or ""),
                    "has_mask": bool(getattr(s, "has_mask", False) or getattr(s, "mask_path", None)),
                    "has_ctrl": bool(getattr(s, "has_control", False) or getattr(s, "control_path", None)),
                    "ctrl": ctrl_rel,
                    "ctrl_size": ctrl_size,
                }
            )
        sample_entries.sort(key=lambda e: (e["img"], e["cap"], e["ctrl"]))

    payload = {
        "v": CACHE_FORMAT_VERSION,
        "resolution": int(getattr(cfg, "resolution", 1024) or 1024),
        "batch_size": int(getattr(cfg, "batch_size", 1) or 1),
        "aspect_ratio_bucketing": bool(getattr(cfg, "aspect_ratio_bucketing", True)),
        "center_crop": bool(getattr(cfg, "center_crop", True)),
        "random_flip": bool(getattr(cfg, "random_flip", False)),
        "use_alpha_mask": bool(getattr(cfg, "use_alpha_mask", True)),
        "control_resize_mode": str(getattr(cfg, "control_resize_mode", "preserve_aspect") or "preserve_aspect"),
        "trigger_word": str(getattr(cfg, "trigger_word", "") or ""),
        "clip_prompt_template": str(clip_prompt_template or ""),
        "pass_control_to_clip": str(pass_control_to_clip or "auto"),
        "clip_vision_resolution": int(clip_vision_resolution or 0),
        "samples": sample_entries,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def resolve_encoder_signature(
    vae: Optional[Any] = None,
    clip: Optional[Any] = None,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract lightweight, portable class & dimension hints from connected VAE / CLIP / Model."""
    sig: Dict[str, Any] = {
        "vae_class": "",
        "clip_class": "",
        "model_arch": "",
        "latent_channels": 0,
        "context_dim": 0,
    }
    if vae is not None:
        inner_vae = getattr(vae, "first_stage_model", vae)
        sig["vae_class"] = type(inner_vae).__name__
    if clip is not None:
        inner_clip = getattr(clip, "cond_stage_model", clip)
        sig["clip_class"] = type(inner_clip).__name__
    if model is not None:
        base_m = getattr(model, "model", model)
        diff_m = getattr(base_m, "diffusion_model", base_m)
        sig["model_arch"] = type(diff_m).__name__
        lat_fmt = getattr(base_m, "latent_format", None)
        lat_ch = getattr(lat_fmt, "latent_channels", None) if lat_fmt is not None else None
        if isinstance(lat_ch, int) and lat_ch > 0:
            sig["latent_channels"] = lat_ch
        elif isinstance(getattr(diff_m, "in_channels", None), int) and int(diff_m.in_channels) > 0:
            sig["latent_channels"] = int(diff_m.in_channels)
        for attr in ("context_in_dim", "context_dim", "caption_channels", "txt_in_dim", "text_dim"):
            val = getattr(diff_m, attr, None)
            if isinstance(val, int) and val > 0:
                sig["context_dim"] = val
                break
    return sig


def _prep_save_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return a detached, contiguous CPU tensor with unshared storage safe for `safetensors.save_file`."""
    out = t.detach().to(device="cpu").contiguous()
    return out.clone()


def save_conditioning_cache(
    cache_dir: str,
    unique_batches: List[Dict[str, Any]],
    latent_scaled: bool,
    uncond_embed: Optional[torch.Tensor],
    uncond_pooled: Optional[torch.Tensor],
    uncond_extra: Dict[str, Any],
    clip_used_control_images: bool,
    num_controlled_images: int,
    resolution: int,
    dataset_fp: str,
    encoder_sig: Dict[str, Any],
) -> Optional[str]:
    """
    Serialize unique pre-encoded batches and unconditional embeddings into an atomic
    `.safetensors` file under `cache_dir`. Returns the saved file path or `None` on failure.
    """
    if not cache_dir or not unique_batches:
        return None
    os.makedirs(cache_dir, exist_ok=True)

    tensors: Dict[str, torch.Tensor] = {}
    batch_manifests: List[Dict[str, Any]] = []

    lat_ch = int(encoder_sig.get("latent_channels", 0) or 0)
    ctx_dim = int(encoder_sig.get("context_dim", 0) or 0)

    for bi, b in enumerate(unique_batches):
        manifest: Dict[str, Any] = {
            "non_tensor_extra": {},
            "non_tensor_extra_flipped": {},
            "ctrl_count": 0,
            "ctrl_flipped_count": 0,
        }
        for field_name in (
            "latents",
            "latents_flipped",
            "prompt_embeds",
            "prompt_embeds_flipped",
            "pooled_output",
            "pooled_output_flipped",
            "loss_mask",
            "loss_mask_flipped",
        ):
            val = b.get(field_name)
            if isinstance(val, torch.Tensor):
                tensors[f"b{bi}.{field_name}"] = _prep_save_tensor(val)
                if field_name == "latents" and val.ndim >= 2 and lat_ch <= 0:
                    lat_ch = int(val.shape[1])
                if field_name == "prompt_embeds" and val.ndim >= 2 and ctx_dim <= 0:
                    ctx_dim = int(val.shape[-1])

        for extra_key, prefix, nt_slot in (
            ("extra_conds", f"b{bi}.extra.", "non_tensor_extra"),
            ("extra_conds_flipped", f"b{bi}.extra_flipped.", "non_tensor_extra_flipped"),
        ):
            ec = b.get(extra_key)
            if isinstance(ec, dict):
                for ek, ev in ec.items():
                    if isinstance(ev, torch.Tensor):
                        tensors[f"{prefix}{ek}"] = _prep_save_tensor(ev)
                    elif isinstance(ev, (int, float, bool, str, list, dict)) or ev is None:
                        manifest[nt_slot][ek] = ev

        for ctrl_key, prefix, count_slot in (
            ("control_latents", f"b{bi}.ctrl.", "ctrl_count"),
            ("control_latents_flipped", f"b{bi}.ctrl_flipped.", "ctrl_flipped_count"),
        ):
            clist = b.get(ctrl_key)
            if isinstance(clist, list) and clist:
                manifest[count_slot] = len(clist)
                for cj, c_val in enumerate(clist):
                    if isinstance(c_val, torch.Tensor):
                        tensors[f"{prefix}{cj}"] = _prep_save_tensor(c_val)

        batch_manifests.append(manifest)

    if isinstance(uncond_embed, torch.Tensor):
        tensors["uncond.prompt_embeds"] = _prep_save_tensor(uncond_embed)
    if isinstance(uncond_pooled, torch.Tensor):
        tensors["uncond.pooled_output"] = _prep_save_tensor(uncond_pooled)
    uncond_non_tensor: Dict[str, Any] = {}
    if isinstance(uncond_extra, dict):
        for ek, ev in uncond_extra.items():
            if isinstance(ev, torch.Tensor):
                tensors[f"uncond.extra.{ek}"] = _prep_save_tensor(ev)
            elif isinstance(ev, (int, float, bool, str, list, dict)) or ev is None:
                uncond_non_tensor[ek] = ev

    if not tensors:
        return None

    metadata: Dict[str, str] = {
        "omni_lora_cache.format_version": CACHE_FORMAT_VERSION,
        "omni_lora_cache.dataset_fingerprint": dataset_fp,
        "omni_lora_cache.num_batches": str(len(unique_batches)),
        "omni_lora_cache.latent_scaled": "1" if latent_scaled else "0",
        "omni_lora_cache.clip_used_control_images": "1" if clip_used_control_images else "0",
        "omni_lora_cache.num_controlled_images": str(int(num_controlled_images)),
        "omni_lora_cache.resolution": str(int(resolution)),
        "omni_lora_cache.latent_channels": str(lat_ch),
        "omni_lora_cache.context_dim": str(ctx_dim),
        "omni_lora_cache.vae_class": str(encoder_sig.get("vae_class", "") or ""),
        "omni_lora_cache.clip_class": str(encoder_sig.get("clip_class", "") or ""),
        "omni_lora_cache.model_arch": str(encoder_sig.get("model_arch", "") or ""),
        "omni_lora_cache.batch_manifests": json.dumps(batch_manifests, ensure_ascii=True),
        "omni_lora_cache.uncond_non_tensor": json.dumps(uncond_non_tensor, ensure_ascii=True),
    }

    arch_tag = (encoder_sig.get("model_arch") or encoder_sig.get("clip_class") or "model")[:24]
    arch_tag = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in arch_tag).strip("_") or "model"
    fname = f"cache_{arch_tag}_c{lat_ch}_d{ctx_dim}_{int(resolution)}px_{dataset_fp[:10]}.safetensors"
    final_path = os.path.join(cache_dir, fname)
    tmp_path = final_path + ".tmp"

    try:
        save_file(tensors, tmp_path, metadata=metadata)
        os.replace(tmp_path, final_path)
        size_mb = os.path.getsize(final_path) / (1024.0 * 1024.0)
        logging.info(
            "[OmniLoRA] Saved pre-cache to %s (%.1f MB, %d unique batch(es), c=%d, d=%d).",
            final_path,
            size_mb,
            len(unique_batches),
            lat_ch,
            ctx_dim,
        )
        return final_path
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        report_exception(
            "conditioning_cache.save",
            f"Failed to save pre-cache to {final_path}",
            exc,
            level=logging.WARNING,
            once=True,
            logger_name="nodes",
        )
        return None


def _meta_matches_encoder_sig(meta: Dict[str, str], encoder_sig: Dict[str, Any]) -> Tuple[bool, int]:
    """
    Check if a cache file's safetensors header matches the connected VAE/CLIP/Model signature.
    Returns `(is_compatible, specificity_score)`.
    """
    score = 0
    for key in ("vae_class", "clip_class"):
        req = str(encoder_sig.get(key, "") or "")
        cached = str(meta.get(f"omni_lora_cache.{key}", "") or "")
        if req and cached:
            if req != cached:
                return False, 0
            score += 2

    req_arch = str(encoder_sig.get("model_arch", "") or "")
    cached_arch = str(meta.get("omni_lora_cache.model_arch", "") or "")
    if req_arch and cached_arch:
        if req_arch != cached_arch:
            return False, 0
        score += 3

    for key in ("latent_channels", "context_dim"):
        req_dim = int(encoder_sig.get(key, 0) or 0)
        try:
            cached_dim = int(meta.get(f"omni_lora_cache.{key}", "0") or 0)
        except ValueError:
            cached_dim = 0
        if req_dim > 0 and cached_dim > 0:
            if req_dim != cached_dim:
                return False, 0
            score += 2

    return True, score


def _find_best_cache_entry(
    cache_dir: Optional[str],
    dataset_fp: str,
    encoder_sig: Dict[str, Any],
    dataset_root: Optional[str] = None,
) -> Optional[Tuple[str, Dict[str, str]]]:
    """Header-only scan returning `(best_path, best_meta)` for a matching `.safetensors` cache file."""
    search_dirs: List[str] = []
    if cache_dir and os.path.isdir(cache_dir):
        search_dirs.append(cache_dir)
    if dataset_root and os.path.isdir(dataset_root) and dataset_root not in search_dirs:
        search_dirs.append(dataset_root)
    if not search_dirs:
        return None

    best_path: Optional[str] = None
    best_meta: Optional[Dict[str, str]] = None
    best_key: Tuple[int, float] = (-1, -1.0)

    for sdir in search_dirs:
        try:
            entries = sorted(os.listdir(sdir))
        except OSError:
            continue
        for fname in entries:
            if not fname.lower().endswith(".safetensors"):
                continue
            fpath = os.path.join(sdir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with safe_open(fpath, framework="pt", device="cpu") as sf:
                    meta = sf.metadata() or {}
            except Exception as exc:
                report_exception(
                    "conditioning_cache.header_probe",
                    f"Skipping unreadable safetensors file {fpath}",
                    exc,
                    level=logging.DEBUG,
                    once=True,
                    logger_name="nodes",
                )
                continue
            if meta.get("omni_lora_cache.format_version") != CACHE_FORMAT_VERSION:
                continue
            if meta.get("omni_lora_cache.dataset_fingerprint") != dataset_fp:
                continue
            ok, score = _meta_matches_encoder_sig(meta, encoder_sig)
            if not ok:
                continue
            try:
                mtime = float(os.path.getmtime(fpath))
            except OSError:
                mtime = 0.0
            cand_key = (score, mtime)
            if cand_key > best_key:
                best_key = cand_key
                best_path = fpath
                best_meta = meta

    if best_path is None or best_meta is None:
        return None
    return best_path, best_meta


def has_matching_conditioning_cache(
    cache_dir: Optional[str],
    dataset_fp: str,
    encoder_sig: Dict[str, Any],
    dataset_root: Optional[str] = None,
) -> bool:
    """Fast header-only check used by ComfyUI `check_lazy_status` to skip loading VAE & CLIP on cache hits."""
    return _find_best_cache_entry(cache_dir, dataset_fp, encoder_sig, dataset_root=dataset_root) is not None


def find_and_load_conditioning_cache(
    cache_dir: Optional[str],
    dataset_fp: str,
    encoder_sig: Dict[str, Any],
    target_dtype: torch.dtype,
    dataset_root: Optional[str] = None,
) -> Optional[Tuple[List[Dict[str, Any]], bool, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any], bool, str]]:
    """
    Search `<dataset_path>/.omni_lora_cache/*.safetensors` (and top-level `<dataset_path>/*.safetensors`)
    for a matching pre-cache file. If found, loads and returns:
        `(unique_batches, latent_scaled, uncond_embed, uncond_pooled, uncond_extra, clip_used_control_images, path)`
    Returns `None` if no matching cache file exists.
    """
    match = _find_best_cache_entry(cache_dir, dataset_fp, encoder_sig, dataset_root=dataset_root)
    if match is None:
        return None
    best_path, best_meta = match

    try:
        num_batches = int(best_meta.get("omni_lora_cache.num_batches", "0") or 0)
        if num_batches <= 0:
            return None
        batch_manifests = json.loads(best_meta.get("omni_lora_cache.batch_manifests", "[]") or "[]")
        uncond_non_tensor = json.loads(best_meta.get("omni_lora_cache.uncond_non_tensor", "{}") or "{}")
        latent_scaled = best_meta.get("omni_lora_cache.latent_scaled", "1") == "1"
        clip_used_ctrl = best_meta.get("omni_lora_cache.clip_used_control_images", "0") == "1"

        loaded_tensors: Dict[str, torch.Tensor] = {}
        with safe_open(best_path, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                loaded_tensors[k] = sf.get_tensor(k)

        def _cast_float_tensor(t: torch.Tensor, keep_fp32: bool = False) -> torch.Tensor:
            if keep_fp32:
                return t.to(device="cpu", dtype=torch.float32).clone()
            if t.is_floating_point():
                return t.to(device="cpu", dtype=target_dtype).clone()
            return t.to(device="cpu").clone()

        unique_batches: List[Dict[str, Any]] = []
        with torch.inference_mode(False):
            for bi in range(num_batches):
                manifest = batch_manifests[bi] if bi < len(batch_manifests) else {}
                item: Dict[str, Any] = {}
                for field_name in (
                    "latents",
                    "latents_flipped",
                    "prompt_embeds",
                    "prompt_embeds_flipped",
                    "pooled_output",
                    "pooled_output_flipped",
                    "loss_mask",
                    "loss_mask_flipped",
                ):
                    tk = f"b{bi}.{field_name}"
                    if tk in loaded_tensors:
                        keep_fp32 = field_name in ("loss_mask", "loss_mask_flipped")
                        item[field_name] = _cast_float_tensor(loaded_tensors[tk], keep_fp32=keep_fp32)

                for extra_key, prefix, nt_slot in (
                    ("extra_conds", f"b{bi}.extra.", "non_tensor_extra"),
                    ("extra_conds_flipped", f"b{bi}.extra_flipped.", "non_tensor_extra_flipped"),
                ):
                    ec_out: Dict[str, Any] = dict(manifest.get(nt_slot, {}) or {})
                    for tk, tv in loaded_tensors.items():
                        if tk.startswith(prefix):
                            ek = tk[len(prefix) :]
                            ec_out[ek] = _cast_float_tensor(tv, keep_fp32=False)
                    if ec_out or extra_key == "extra_conds":
                        item[extra_key] = ec_out

                for ctrl_key, prefix, count_slot in (
                    ("control_latents", f"b{bi}.ctrl.", "ctrl_count"),
                    ("control_latents_flipped", f"b{bi}.ctrl_flipped.", "ctrl_flipped_count"),
                ):
                    c_count = int(manifest.get(count_slot, 0) or 0)
                    if c_count > 0:
                        clist: List[Optional[torch.Tensor]] = []
                        for cj in range(c_count):
                            ck = f"{prefix}{cj}"
                            clist.append(
                                _cast_float_tensor(loaded_tensors[ck], keep_fp32=False)
                                if ck in loaded_tensors
                                else None
                            )
                        item[ctrl_key] = clist

                unique_batches.append(item)

            uncond_embed = (
                _cast_float_tensor(loaded_tensors["uncond.prompt_embeds"])
                if "uncond.prompt_embeds" in loaded_tensors
                else None
            )
            uncond_pooled = (
                _cast_float_tensor(loaded_tensors["uncond.pooled_output"])
                if "uncond.pooled_output" in loaded_tensors
                else None
            )
            uncond_extra: Dict[str, Any] = dict(uncond_non_tensor or {})
            for tk, tv in loaded_tensors.items():
                if tk.startswith("uncond.extra."):
                    ek = tk[len("uncond.extra.") :]
                    uncond_extra[ek] = _cast_float_tensor(tv, keep_fp32=False)

        logging.info(
            "[OmniLoRA] Loaded matching disk pre-cache from %s (%d unique batch(es) — skipped VAE & CLIP encoding!).",
            best_path,
            len(unique_batches),
        )
        return (
            unique_batches,
            latent_scaled,
            uncond_embed,
            uncond_pooled,
            uncond_extra,
            clip_used_ctrl,
            best_path,
        )
    except Exception as exc:
        report_exception(
            "conditioning_cache.load",
            f"Failed to load disk pre-cache from {best_path}; will re-encode",
            exc,
            level=logging.WARNING,
            once=True,
            logger_name="nodes",
        )
        return None
