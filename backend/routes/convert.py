from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from models.settings import validate_vector_settings
from services.preprocessing import preprocess_only_to_image
from services.vectorizer import vectorize_existing_processed_image

router = APIRouter(tags=["convert"])


# ---------------------------------------------------------------------------
# New JSON-based flow for the editor page:
#   The image is already uploaded; we only re-run conversion with new settings.
# ---------------------------------------------------------------------------
@router.post("/api/convert/{file_id}")
async def convert_existing_image(file_id: str, payload: dict = Body(...)):
    settings = validate_vector_settings(
        limit_colors=bool(payload.get("limit_colors", False)),
        target_colors=int(payload.get("target_colors", 16)),
        upscale_factor=int(payload.get("upscale_factor", 1)),
        smoothing_strength=int(payload.get("smoothing_strength", 0)),
        sharpen_strength=int(payload.get("sharpen_strength", 0)),
        color_precision=int(payload.get("color_precision", 6)),
        filter_speckle=int(payload.get("filter_speckle", 4)),
        layer_difference=int(payload.get("layer_difference", 16)),
        corner_threshold=int(payload.get("corner_threshold", 60)),
        length_threshold=float(payload.get("length_threshold", 4.0)),
        path_precision=int(payload.get("path_precision", 3)),
        max_iterations=int(payload.get("max_iterations", 20)),
        splice_threshold=int(payload.get("splice_threshold", 45)),
    )

    result = convert_file_to_vector_outputs(
        filename=file_id,
        settings=settings,
    )

    return JSONResponse(
        content={
            "message": "Image converted successfully.",
            "file_id": file_id,
            "svg_filename": result["svg_filename"],
            "pdf_filename": result["pdf_filename"],
            "processed_filename": result["processed_filename"],
            "svg_download_url": result["svg_download_url"],
            "pdf_download_url": result["pdf_download_url"],
            "processed_download_url": result["processed_download_url"],
            "settings": settings.to_dict(),
        }
    )

# ---------------------------------------------------------------------------
# Phase 1 (cleanup only) — used by the cleanup page
# ---------------------------------------------------------------------------
@router.post("/api/preprocess/{file_id}")
async def preprocess_existing_image(file_id: str, payload: dict = Body(...)):
    settings = validate_vector_settings(
        limit_colors=bool(payload.get("limit_colors", False)),
        color_reduction_method=str(payload.get("color_reduction_method", "smart_hue")),
        target_colors=int(payload.get("target_colors", 16)),
        upscale_factor=int(payload.get("upscale_factor", 1)),
        smoothing_strength=int(payload.get("smoothing_strength", 0)),
        smoothing_method=str(payload.get("smoothing_method", "gaussian")),
        sharpen_strength=int(payload.get("sharpen_strength", 0)),
        median_filter_enabled=bool(payload.get("median_filter_enabled", False)),
        median_filter_strength=int(payload.get("median_filter_strength", 1)),
        morphological_close_enabled=bool(payload.get("morphological_close_enabled", False)),
        morphological_close_strength=int(payload.get("morphological_close_strength", 1)),
        drop_tiny_blobs_enabled=bool(payload.get("drop_tiny_blobs_enabled", False)),
        drop_tiny_blobs_min_size=int(payload.get("drop_tiny_blobs_min_size", 8)),
        color_precision=int(payload.get("color_precision", 6)),
        filter_speckle=int(payload.get("filter_speckle", 4)),
        layer_difference=int(payload.get("layer_difference", 16)),
        corner_threshold=int(payload.get("corner_threshold", 60)),
        length_threshold=float(payload.get("length_threshold", 4.0)),
        path_precision=int(payload.get("path_precision", 3)),
        max_iterations=int(payload.get("max_iterations", 20)),
        splice_threshold=int(payload.get("splice_threshold", 45)),
    )

    result = preprocess_only_to_image(
        filename=file_id,
        settings=settings,
    )

    return JSONResponse(
        content={
            "message": "Image preprocessed successfully.",
            "file_id": file_id,
            "processed_filename": result["processed_filename"],
            "processed_download_url": result["processed_download_url"],
            "settings": settings.to_dict(),
        }
    )

# ---------------------------------------------------------------------------
# Phase 2 (vectorize only) — used by the vectorize page
# ---------------------------------------------------------------------------
@router.post("/api/vectorize/{file_id}")
async def vectorize_existing_image(file_id: str, payload: dict = Body(...)):
    settings = validate_vector_settings(
        # Cleanup settings (passed through for completeness, not used here)
        limit_colors=bool(payload.get("limit_colors", False)),
        color_reduction_method=str(payload.get("color_reduction_method", "smart_hue")),
        target_colors=int(payload.get("target_colors", 16)),
        upscale_factor=int(payload.get("upscale_factor", 1)),
        smoothing_strength=int(payload.get("smoothing_strength", 0)),
        smoothing_method=str(payload.get("smoothing_method", "gaussian")),
        sharpen_strength=int(payload.get("sharpen_strength", 0)),
        median_filter_enabled=bool(payload.get("median_filter_enabled", False)),
        median_filter_strength=int(payload.get("median_filter_strength", 1)),
        morphological_close_enabled=bool(payload.get("morphological_close_enabled", False)),
        morphological_close_strength=int(payload.get("morphological_close_strength", 1)),
        drop_tiny_blobs_enabled=bool(payload.get("drop_tiny_blobs_enabled", False)),
        drop_tiny_blobs_min_size=int(payload.get("drop_tiny_blobs_min_size", 8)),
        color_precision=int(payload.get("color_precision", 6)),
        filter_speckle=int(payload.get("filter_speckle", 4)),
        layer_difference=int(payload.get("layer_difference", 16)),
        corner_threshold=int(payload.get("corner_threshold", 60)),
        length_threshold=float(payload.get("length_threshold", 4.0)),
        path_precision=int(payload.get("path_precision", 3)),
        max_iterations=int(payload.get("max_iterations", 20)),
        splice_threshold=int(payload.get("splice_threshold", 45)),
        svg_round_coordinates=bool(payload.get("svg_round_coordinates", True)),
        svg_coordinate_decimals=int(payload.get("svg_coordinate_decimals", 2)),
        svg_remove_tiny_paths=bool(payload.get("svg_remove_tiny_paths", True)),
        svg_min_path_size=int(payload.get("svg_min_path_size", 4)),
        svg_merge_same_color_paths=bool(payload.get("svg_merge_same_color_paths", True)),
        svg_smooth_curves=bool(payload.get("svg_smooth_curves", True)),
        svg_smoothing_passes=int(payload.get("svg_smoothing_passes", 2)),
        svg_curve_sample_step=float(payload.get("svg_curve_sample_step", 2.0)),
    )

    result = vectorize_existing_processed_image(
        filename=file_id,
        settings=settings,
    )

    return JSONResponse(
        content={
            "message": "Image vectorized successfully.",
            "file_id": file_id,
            "svg_filename": result["svg_filename"],
            "pdf_filename": result["pdf_filename"],
            "processed_filename": result["processed_filename"],
            "svg_download_url": result["svg_download_url"],
            "pdf_download_url": result["pdf_download_url"],
            "processed_download_url": result["processed_download_url"],
            "palette_size": result.get("palette_size", 0),
            "effective_color_precision": result.get("effective_color_precision", settings.color_precision),
            "settings": settings.to_dict(),
        }
    )