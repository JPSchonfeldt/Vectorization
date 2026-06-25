from fastapi import APIRouter, Body, File, Form, UploadFile
from fastapi.responses import JSONResponse

from models.settings import validate_vector_settings
from routes.upload import save_uploaded_file
from services.preprocessing import preprocess_only_to_image
from services.vectorizer import (
    convert_file_to_vector_outputs,
    vectorize_existing_processed_image,
)

router = APIRouter(tags=["convert"])

# ---------------------------------------------------------------------------
# Legacy single-step flow:
#   Upload + convert in one request, used by the current home page.
# ---------------------------------------------------------------------------
@router.post("/convert-upload")
async def upload_and_convert_image(
    file: UploadFile = File(...),
    limit_colors: bool = Form(False),
    color_reduction_method: str = Form("smart_hue"),
    target_colors: int = Form(16),
    upscale_factor: int = Form(1),
    smoothing_strength: int = Form(0),
    smoothing_method: str = Form("gaussian"),
    sharpen_strength: int = Form(0),
    color_precision: int = Form(6),
    filter_speckle: int = Form(4),
    layer_difference: int = Form(16),
    corner_threshold: int = Form(60),
    length_threshold: float = Form(4.0),
    path_precision: int = Form(3),
    max_iterations: int = Form(20),
    splice_threshold: int = Form(45),
    svg_round_coordinates: bool = Form(True),
    svg_coordinate_decimals: int = Form(2),
    svg_remove_tiny_paths: bool = Form(True),
    svg_min_path_size: int = Form(4),
    svg_merge_same_color_paths: bool = Form(True),
    svg_smooth_curves: bool = Form(True),
    svg_smoothing_passes: int = Form(2),
    svg_curve_sample_step: float = Form(2.0),
):
    settings = validate_vector_settings(
        limit_colors=limit_colors,
        color_reduction_method=color_reduction_method,
        target_colors=target_colors,
        upscale_factor=upscale_factor,
        smoothing_strength=smoothing_strength,
        smoothing_method=smoothing_method,
        sharpen_strength=sharpen_strength,
        color_precision=color_precision,
        filter_speckle=filter_speckle,
        layer_difference=layer_difference,
        corner_threshold=corner_threshold,
        length_threshold=length_threshold,
        path_precision=path_precision,
        max_iterations=max_iterations,
        splice_threshold=splice_threshold,
        svg_round_coordinates=svg_round_coordinates,
        svg_coordinate_decimals=svg_coordinate_decimals,
        svg_remove_tiny_paths=svg_remove_tiny_paths,
        svg_min_path_size=svg_min_path_size,
        svg_merge_same_color_paths=svg_merge_same_color_paths,
        svg_smooth_curves=svg_smooth_curves,
        svg_smoothing_passes=svg_smoothing_passes,
        svg_curve_sample_step=svg_curve_sample_ste
    )

    saved = await save_uploaded_file(file)

    result = convert_file_to_vector_outputs(
        filename=saved["saved_filename"],
        settings=settings,
    )

    return JSONResponse(
        content={
            "message": "Image uploaded and converted successfully.",
            "original_filename": saved["original_filename"],
            "saved_filename": saved["saved_filename"],
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

        # The cleanup phase does not really use these, but the same
        # validation function expects them. We use safe defaults.
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

        # Vector settings (this is what this phase actually uses)
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