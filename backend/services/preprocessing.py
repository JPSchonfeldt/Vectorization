import json
from pathlib import Path

from fastapi import HTTPException
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.restoration import denoise_bilateral
from skimage.filters import median
from skimage.morphology import (
    disk,
    binary_closing,
    label,
    remove_small_objects,
)

from config import (
    ALLOWED_EXTENSIONS,
    MAX_DIMENSION_AFTER_UPSCALE,
    PROCESSED_DIR,
    UPLOAD_DIR,
)
from models.settings import VectorSettings
from services.color_reduction import (
    reduce_image_colors_image,
    reduce_image_colors_lab_kmeans,
)

# ---------------------------------------------------------------------------
# Public preprocessing entry point
# ---------------------------------------------------------------------------
def preprocess_image_for_tracing(input_path: Path, output_path: Path, settings: VectorSettings) -> None:
    image = load_image_as_rgb(input_path)

    image = apply_upscale(image, settings.upscale_factor)
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

    # Save palette so the vectorize phase can lock to it
    palette_path = output_path.with_name(output_path.stem + "_palette.json")

    if palette_used is not None:
        save_palette(palette_path, palette_used)
    else:
        if palette_path.exists():
            try:
                palette_path.unlink()
            except OSError:
                pass

    # --- Phase 1F: Classical artifact cleanup ---
    if settings.median_filter_enabled:
        image = apply_median_filter(image, strength=settings.median_filter_strength)

    if settings.morphological_close_enabled:
        image = apply_morphological_close(image, strength=settings.morphological_close_strength)

    if settings.drop_tiny_blobs_enabled:
        image = apply_drop_tiny_blobs(image, min_size=settings.drop_tiny_blobs_min_size)

    image = apply_smoothing(
        image,
        strength=settings.smoothing_strength,
        method=settings.smoothing_method,
    )
    image = apply_sharpen(image, strength=settings.sharpen_strength)

    image.save(output_path, format="PNG")


# ---------------------------------------------------------------------------
# Individual processing steps
# ---------------------------------------------------------------------------
def load_image_as_rgb(input_path: Path) -> Image.Image:
    """
    Load an image and flatten any transparency onto a white background.
    """

    image = Image.open(input_path)

    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        return background.convert("RGB")

    return image.convert("RGB")


def apply_upscale(image: Image.Image, upscale_factor: int) -> Image.Image:
    """
    Upscale the image by an integer factor, capped at MAX_DIMENSION_AFTER_UPSCALE
    so VTracer does not stall on huge images.
    """

    if upscale_factor <= 1:
        return image

    new_width = image.width * upscale_factor
    new_height = image.height * upscale_factor

    if max(new_width, new_height) > MAX_DIMENSION_AFTER_UPSCALE:
        scale = MAX_DIMENSION_AFTER_UPSCALE / max(new_width, new_height)
        new_width = int(new_width * scale)
        new_height = int(new_height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def apply_contrast_boost(image: Image.Image, factor: float) -> Image.Image:
    """
    Apply a mild contrast boost to make colour boundaries clearer for the tracer.
    """

    if factor == 1.0:
        return image

    return ImageEnhance.Contrast(image).enhance(factor)


def apply_smoothing(image: Image.Image, strength: int, method: str) -> Image.Image:
    """
    Apply smoothing using the chosen algorithm.
    Strength 0 = no smoothing.
    """

    if strength <= 0:
        return image

    if method == "bilateral":
        return apply_bilateral_smoothing(image, strength)

    return apply_gaussian_smoothing(image, strength)


def apply_gaussian_smoothing(image: Image.Image, strength: int) -> Image.Image:
    """
    Standard Gaussian blur. Smooths everything uniformly, including edges.
    """

    radius = strength * 0.6
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_bilateral_smoothing(image: Image.Image, strength: int) -> Image.Image:
    """
    Edge-preserving bilateral filter.

    Smooths within colour regions but does not blur across colour edges.
    Better for vector preparation because VTracer traces edges from this.
    """

    rgb_array = np.array(image.convert("RGB"), dtype=np.float64) / 255.0

    # Strength tuning:
    # sigma_color controls colour similarity (higher = more colours merged)
    # sigma_spatial controls geometric range (higher = wider smoothing area)
    sigma_color = 0.05 + (strength * 0.04)     # 0.05 .. 0.25
    sigma_spatial = 1.0 + (strength * 1.2)     # 1.0 .. 7.0

    smoothed = denoise_bilateral(
        rgb_array,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=-1,
    )

    smoothed_uint8 = np.clip(smoothed * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(smoothed_uint8, mode="RGB")


def apply_sharpen(image: Image.Image, strength: int) -> Image.Image:
    """
    Restore edge clarity after smoothing.
    Strength 0 = no sharpen.
    """

    if strength <= 0:
        return image

    sharpen_factor = 1.0 + (strength * 0.35)
    return ImageEnhance.Sharpness(image).enhance(sharpen_factor)

# ---------------------------------------------------------------------------
# Standalone preprocessing entry point for the cleanup phase
# ---------------------------------------------------------------------------
def preprocess_only_to_image(filename: str, settings: VectorSettings) -> dict:
    """
    Run only the Pillow preprocessing pipeline on an uploaded image.

    This is used by Phase 1 (cleanup page). It does not run VTracer.
    Returns the processed filename and download URL so the frontend
    can preview the cleanup result.
    """

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
    """
    Resolve and validate the uploaded file path.
    Guards against path traversal and unsupported file types.
    """

    candidate_path = (UPLOAD_DIR / filename).resolve()

    if not str(candidate_path).startswith(str(UPLOAD_DIR.resolve())):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    if not candidate_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded file not found.",
        )

    file_extension = candidate_path.suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
        )

    return candidate_path

# ---------------------------------------------------------------------------
# Palette persistence (Phase 1D)
# ---------------------------------------------------------------------------
def save_palette(palette_path: Path, palette: list) -> None:
    """
    Save the cleanup palette as a sidecar JSON file so the vectorize phase
    can lock the SVG output to exactly these colours.
    """

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
    """
    Load a previously saved cleanup palette.
    Returns an empty list if the file is missing or invalid.
    """

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

# ---------------------------------------------------------------------------
# Auto color precision (Phase 1D)
# ---------------------------------------------------------------------------
def compute_auto_color_precision(palette: list, max_precision: int) -> int:
    """
    Find the smallest VTracer color_precision (1..8) that keeps every palette
    colour in its own quantization bucket.

    VTracer color_precision is the number of bits per RGB channel.
    Bucket size per channel = 256 / 2^precision.

    If no precision in the allowed range can fully separate the palette,
    we return max_precision so the user's chosen ceiling still applies.
    """

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

# ---------------------------------------------------------------------------
# Phase 1F — Classical artifact cleanup
# ---------------------------------------------------------------------------
def apply_median_filter(image: Image.Image, strength: int) -> Image.Image:
    """
    Reduce JPEG/compression noise and salt & pepper artifacts.

    Strength 0 = off. Higher = larger neighbourhood = more aggressive
    noise removal. Too high can blur fine details.

    Runs per channel because scikit-image median expects 2D arrays.
    """

    if strength <= 0:
        return image

    radius = strength  # disk radius in pixels (1..5)
    structuring_element = disk(radius)

    rgb_array = np.array(image.convert("RGB"))

    cleaned_channels = []
    for channel_index in range(3):
        channel = rgb_array[:, :, channel_index]
        cleaned = median(channel, footprint=structuring_element)
        cleaned_channels.append(cleaned)

    cleaned_array = np.stack(cleaned_channels, axis=2).astype(np.uint8)

    return Image.fromarray(cleaned_array, mode="RGB")


def apply_morphological_close(image: Image.Image, strength: int) -> Image.Image:
    """
    Close tiny gaps in shape outlines.

    Useful when a logo or text shape has thin breaks caused by JPEG
    compression or quantization. The closing operation grows shapes
    by N pixels, then shrinks them back by N pixels — gaps disappear
    but shapes keep their original size.

    Strength 0 = off. 1 = 1px radius. 2 = 3px radius. 3 = 5px radius.
    """

    if strength <= 0:
        return image

    radius = (strength * 2) - 1  # 1, 3, 5
    structuring_element = disk(radius)

    rgb_array = np.array(image.convert("RGB"))

    # Build a binary mask of non-near-white pixels (everything that is "ink")
    # so we close ink-side gaps without disturbing the background.
    luminance = (
        0.299 * rgb_array[:, :, 0]
        + 0.587 * rgb_array[:, :, 1]
        + 0.114 * rgb_array[:, :, 2]
    )
    ink_mask = luminance < 230  # tune-able threshold

    closed_mask = binary_closing(ink_mask, footprint=structuring_element)

    # Pixels that became ink after closing inherit the nearest existing
    # ink colour by simply darkening the original by a small bias. This
    # keeps the closing visible without inventing a colour from nothing.
    output = rgb_array.copy()
    grew_mask = closed_mask & (~ink_mask)
    if np.any(grew_mask):
        # Use a black "shape extension" so the closure is meaningful
        # for the tracer.
        output[grew_mask] = [0, 0, 0]

    return Image.fromarray(output, mode="RGB")


def apply_drop_tiny_blobs(image: Image.Image, min_size: int) -> Image.Image:
    """
    Remove tiny disconnected dark pixel blobs.

    These are usually JPEG specks or noise that VTracer would otherwise
    turn into nuisance SVG paths.

    min_size = minimum connected pixel count to keep.
    """

    if min_size <= 0:
        return image

    rgb_array = np.array(image.convert("RGB"))

    luminance = (
        0.299 * rgb_array[:, :, 0]
        + 0.587 * rgb_array[:, :, 1]
        + 0.114 * rgb_array[:, :, 2]
    )
    ink_mask = luminance < 230

    labelled_mask, _ = label(ink_mask, connectivity=2, return_num=True)
    kept_mask = remove_small_objects(labelled_mask, min_size=min_size, connectivity=2) > 0

    removed_mask = ink_mask & (~kept_mask)
    if not np.any(removed_mask):
        return image

    # Replace removed pixels with white background
    output = rgb_array.copy()
    output[removed_mask] = [255, 255, 255]

    return Image.fromarray(output, mode="RGB")