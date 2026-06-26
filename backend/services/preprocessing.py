"""
Image preprocessing pipeline.

The pipeline is alpha-aware: an optional alpha mask is tracked through every
step so transparent input images do not lose their transparency on the way to
the vectorizer.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageEnhance, ImageFilter
from skimage.filters import median
from skimage.morphology import (
    binary_closing,
    disk,
    label,
    remove_small_objects,
)
from skimage.restoration import denoise_bilateral

from config import (
    ALLOWED_EXTENSIONS,
    MAX_DIMENSION_AFTER_UPSCALE,
    PROCESSED_DIR,
    UPLOAD_DIR,
)
from models.settings import VectorSettings
from services.background_removal import remove_background as run_rembg
from services.color_reduction import (
    reduce_image_colors_image,
    reduce_image_colors_lab_kmeans,
)


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_image_for_tracing(
    input_path: Path,
    output_path: Path,
    settings: VectorSettings,
) -> None:
    """Run the full cleanup pipeline and save the processed PNG."""

    image, alpha_mask = load_image_keep_alpha(input_path)

    if settings.remove_background:
        image, alpha_mask = run_rembg(image=image, model_name=settings.rembg_model)

    if alpha_mask is not None and settings.background_mode == "white":
        image = composite_onto_white(image, alpha_mask)
        alpha_mask = None

    image, alpha_mask = apply_upscale(image, alpha_mask, settings.upscale_factor)

    image = apply_contrast_boost(image, factor=1.2)

    palette_used = None

    if settings.limit_colors:
        if settings.color_reduction_method == "lab_kmeans":
            image, palette_used = reduce_image_colors_lab_kmeans(
                image=image,
                target_colors=settings.target_colors,
            )
        else:
            image, palette_used = reduce_image_colors_image(
                image=image,
                target_colors=settings.target_colors,
            )

    palette_path = output_path.with_name(output_path.stem + "_palette.json")

    if palette_used is not None:
        save_palette(palette_path, palette_used)
    else:
        if palette_path.exists():
            try:
                palette_path.unlink()
            except OSError:
                pass

    if settings.median_filter_enabled:
        image = apply_median_filter(image, strength=settings.median_filter_strength)

    if settings.morphological_close_enabled:
        image = apply_morphological_close(
            image,
            strength=settings.morphological_close_strength,
            alpha_mask=alpha_mask,
        )

    if settings.drop_tiny_blobs_enabled:
        image = apply_drop_tiny_blobs(
            image,
            min_size=settings.drop_tiny_blobs_min_size,
            alpha_mask=alpha_mask,
        )

    image = apply_smoothing(
        image,
        strength=settings.smoothing_strength,
        method=settings.smoothing_method,
    )
    image = apply_sharpen(image, strength=settings.sharpen_strength)

    save_processed_png(output_path, image, alpha_mask)


def preprocess_only_to_image(filename: str, settings: VectorSettings) -> dict:
    """Run only the cleanup pipeline. Used by the Cleanup page."""

    input_path = resolve_upload_path(filename)

    file_stem = input_path.stem
    processed_filename = f"{file_stem}_processed.png"
    processed_path = PROCESSED_DIR / processed_filename

    try:
        preprocess_image_for_tracing(
            input_path=input_path,
            output_path=processed_path,
            settings=settings,
        )
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to preprocess image: {ex}",
        )

    return {
        "processed_filename": processed_filename,
        "processed_download_url": f"/download/processed/{processed_filename}",
    }


def resolve_upload_path(filename: str) -> Path:
    candidate_path = (UPLOAD_DIR / filename).resolve()

    if not str(candidate_path).startswith(str(UPLOAD_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not candidate_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found.")

    file_extension = candidate_path.suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    return candidate_path


# ---------------------------------------------------------------------------
# Image loading (alpha-aware)
# ---------------------------------------------------------------------------
def load_image_keep_alpha(input_path: Path) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """Load image and return (RGB image, optional alpha ndarray uint8)."""

    image = Image.open(input_path)

    if image.mode in ("RGBA", "LA", "PA"):
        rgba = image.convert("RGBA")
        rgba_array = np.array(rgba)
        rgb_image = Image.fromarray(rgba_array[:, :, :3], mode="RGB")
        alpha_mask = rgba_array[:, :, 3].astype(np.uint8)
        return rgb_image, alpha_mask

    if image.mode == "P" and "transparency" in image.info:
        rgba = image.convert("RGBA")
        rgba_array = np.array(rgba)
        rgb_image = Image.fromarray(rgba_array[:, :, :3], mode="RGB")
        alpha_mask = rgba_array[:, :, 3].astype(np.uint8)
        return rgb_image, alpha_mask

    return image.convert("RGB"), None


def composite_onto_white(image: Image.Image, alpha_mask: np.ndarray) -> Image.Image:
    rgb_array = np.array(image.convert("RGB"), dtype=np.float32)
    alpha_normalised = alpha_mask.astype(np.float32) / 255.0
    alpha_3channel = np.stack([alpha_normalised] * 3, axis=2)

    white = np.full_like(rgb_array, 255.0)
    composited = rgb_array * alpha_3channel + white * (1.0 - alpha_3channel)

    return Image.fromarray(np.clip(composited, 0, 255).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------
def apply_upscale(
    image: Image.Image,
    alpha_mask: Optional[np.ndarray],
    upscale_factor: int,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    if upscale_factor <= 1:
        return image, alpha_mask

    new_width = image.width * upscale_factor
    new_height = image.height * upscale_factor

    if max(new_width, new_height) > MAX_DIMENSION_AFTER_UPSCALE:
        scale = MAX_DIMENSION_AFTER_UPSCALE / max(new_width, new_height)
        new_width = int(new_width * scale)
        new_height = int(new_height * scale)

    upscaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    upscaled_alpha = None
    if alpha_mask is not None:
        alpha_image = Image.fromarray(alpha_mask, mode="L").resize(
            (new_width, new_height),
            Image.Resampling.NEAREST,
        )
        upscaled_alpha = np.array(alpha_image, dtype=np.uint8)

    return upscaled_image, upscaled_alpha


def apply_contrast_boost(image: Image.Image, factor: float) -> Image.Image:
    if factor == 1.0:
        return image
    return ImageEnhance.Contrast(image).enhance(factor)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def apply_smoothing(image: Image.Image, strength: int, method: str) -> Image.Image:
    if strength <= 0:
        return image
    if method == "bilateral":
        return apply_bilateral_smoothing(image, strength)
    return apply_gaussian_smoothing(image, strength)


def apply_gaussian_smoothing(image: Image.Image, strength: int) -> Image.Image:
    radius = strength * 0.6
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_bilateral_smoothing(image: Image.Image, strength: int) -> Image.Image:
    rgb_array = np.array(image.convert("RGB"), dtype=np.float64) / 255.0

    sigma_color = 0.05 + (strength * 0.04)
    sigma_spatial = 1.0 + (strength * 1.2)

    smoothed = denoise_bilateral(
        rgb_array,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=-1,
    )

    smoothed_uint8 = np.clip(smoothed * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(smoothed_uint8, mode="RGB")


def apply_sharpen(image: Image.Image, strength: int) -> Image.Image:
    if strength <= 0:
        return image
    sharpen_factor = 1.0 + (strength * 0.35)
    return ImageEnhance.Sharpness(image).enhance(sharpen_factor)


# ---------------------------------------------------------------------------
# Phase 1F - Classical artifact cleanup (alpha-aware)
# ---------------------------------------------------------------------------
def apply_median_filter(image: Image.Image, strength: int) -> Image.Image:
    if strength <= 0:
        return image

    radius = max(1, int(strength))
    structuring_element = disk(radius)

    rgb_array = np.array(image.convert("RGB"))

    cleaned_channels = []
    for channel_index in range(3):
        channel = rgb_array[:, :, channel_index]
        cleaned = median(channel, footprint=structuring_element)
        cleaned_channels.append(cleaned)

    cleaned_array = np.stack(cleaned_channels, axis=2).astype(np.uint8)
    return Image.fromarray(cleaned_array, mode="RGB")


def apply_morphological_close(
    image: Image.Image,
    strength: int,
    alpha_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    if strength <= 0:
        return image

    radius = max(1, (strength * 2) - 1)
    structuring_element = disk(radius)

    rgb_array = np.array(image.convert("RGB"))

    luminance = (
        0.299 * rgb_array[:, :, 0]
        + 0.587 * rgb_array[:, :, 1]
        + 0.114 * rgb_array[:, :, 2]
    )
    ink_mask = luminance < 230

    if alpha_mask is not None:
        opaque_mask = alpha_mask > 128
        ink_mask = ink_mask & opaque_mask

    closed_mask = binary_closing(ink_mask, footprint=structuring_element)

    if alpha_mask is not None:
        closed_mask = closed_mask & (alpha_mask > 128)

    output = rgb_array.copy()
    grew_mask = closed_mask & (~ink_mask)
    if np.any(grew_mask):
        output[grew_mask] = [0, 0, 0]

    return Image.fromarray(output, mode="RGB")


def apply_drop_tiny_blobs(
    image: Image.Image,
    min_size: int,
    alpha_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    if min_size <= 0:
        return image

    rgb_array = np.array(image.convert("RGB"))

    luminance = (
        0.299 * rgb_array[:, :, 0]
        + 0.587 * rgb_array[:, :, 1]
        + 0.114 * rgb_array[:, :, 2]
    )
    ink_mask = luminance < 230

    if alpha_mask is not None:
        opaque_mask = alpha_mask > 128
        ink_mask = ink_mask & opaque_mask

    labelled_mask = label(ink_mask, connectivity=2)
    kept_mask = remove_small_objects(labelled_mask, min_size=min_size, connectivity=2) > 0

    removed_mask = ink_mask & (~kept_mask)
    if not np.any(removed_mask):
        return image

    output = rgb_array.copy()
    output[removed_mask] = [255, 255, 255]
    return Image.fromarray(output, mode="RGB")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_processed_png(
    output_path: Path,
    image: Image.Image,
    alpha_mask: Optional[np.ndarray],
) -> None:
    if alpha_mask is None:
        image.save(output_path, format="PNG")
        return

    rgb_array = np.array(image.convert("RGB"))
    rgba_array = np.dstack([rgb_array, alpha_mask.astype(np.uint8)])
    rgba_image = Image.fromarray(rgba_array, mode="RGBA")
    rgba_image.save(output_path, format="PNG")


# ---------------------------------------------------------------------------
# Palette persistence (Phase 1D)
# ---------------------------------------------------------------------------
def save_palette(palette_path: Path, palette: list) -> None:
    safe_palette = []
    for colour in palette:
        if hasattr(colour, "tolist"):
            colour = colour.tolist()
        safe_palette.append([
            int(colour[0]),
            int(colour[1]),
            int(colour[2]),
        ])

    palette_path.write_text(
        json.dumps(safe_palette, indent=2),
        encoding="utf-8",
    )


def load_palette(palette_path: Path) -> list:
    if not palette_path.exists():
        return []

    try:
        data = json.loads(palette_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [
            [int(colour[0]), int(colour[1]), int(colour[2])]
            for colour in data
            if isinstance(colour, (list, tuple)) and len(colour) >= 3
        ]
    except Exception:
        return []


def compute_auto_color_precision(palette: list, max_precision: int) -> int:
    if not palette:
        return max_precision

    capped_max = max(1, min(int(max_precision), 8))

    for precision in range(1, capped_max + 1):
        levels = 2 ** precision
        bucket_size = 256 / levels

        seen_buckets = set()
        collision = False

        for colour in palette:
            r_bucket = int(min(colour[0], 255) // bucket_size)
            g_bucket = int(min(colour[1], 255) // bucket_size)
            b_bucket = int(min(colour[2], 255) // bucket_size)
            key = (r_bucket, g_bucket, b_bucket)

            if key in seen_buckets:
                collision = True
                break
            seen_buckets.add(key)

        if not collision:
            return precision

    return capped_max