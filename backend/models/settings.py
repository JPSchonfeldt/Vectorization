from dataclasses import dataclass, asdict
from fastapi import HTTPException


@dataclass
class VectorSettings:
    """All user-controllable vectorization settings."""

    # Image cleanup
    limit_colors: bool = False
    color_reduction_method: str = "smart_hue"
    target_colors: int = 16
    upscale_factor: int = 1

    # Stage C - Background removal
    remove_background: bool = False
    background_mode: str = "transparent"
    rembg_model: str = "u2netp"

    # Phase 1F - Classical artifact cleanup
    median_filter_enabled: bool = False
    median_filter_strength: int = 1
    morphological_close_enabled: bool = False
    morphological_close_strength: int = 1
    drop_tiny_blobs_enabled: bool = False
    drop_tiny_blobs_min_size: int = 8

    # Smoothing / sharpening
    smoothing_strength: int = 0
    smoothing_method: str = "gaussian"
    sharpen_strength: int = 0

    # VTracer settings
    color_precision: int = 6
    filter_speckle: int = 4
    layer_difference: int = 16
    corner_threshold: int = 60
    length_threshold: float = 4.0
    path_precision: int = 3
    max_iterations: int = 20
    splice_threshold: int = 45

    # SVG curve smoothing (Phase 1E)
    svg_smooth_curves: bool = True
    svg_smoothing_passes: int = 2
    svg_curve_sample_step: float = 2.0

    # SVG post-processing (Phase 1C)
    svg_round_coordinates: bool = True
    svg_coordinate_decimals: int = 2
    svg_remove_tiny_paths: bool = True
    svg_min_path_size: int = 4
    svg_merge_same_color_paths: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def validate_vector_settings(
    limit_colors: bool,
    color_reduction_method: str,
    target_colors: int,
    upscale_factor: int,
    remove_background: bool,
    background_mode: str,
    rembg_model: str,
    smoothing_strength: int,
    smoothing_method: str,
    sharpen_strength: int,
    median_filter_enabled: bool,
    median_filter_strength: int,
    morphological_close_enabled: bool,
    morphological_close_strength: int,
    drop_tiny_blobs_enabled: bool,
    drop_tiny_blobs_min_size: int,
    color_precision: int,
    filter_speckle: int,
    layer_difference: int,
    corner_threshold: int,
    length_threshold: float,
    path_precision: int,
    max_iterations: int,
    splice_threshold: int,
    svg_round_coordinates: bool = True,
    svg_coordinate_decimals: int = 2,
    svg_remove_tiny_paths: bool = True,
    svg_min_path_size: int = 4,
    svg_merge_same_color_paths: bool = True,
    svg_smooth_curves: bool = True,
    svg_smoothing_passes: int = 2,
    svg_curve_sample_step: float = 2.0,
) -> VectorSettings:
    """Validate the incoming settings and return a VectorSettings object."""

    if target_colors < 2 or target_colors > 128:
        raise HTTPException(status_code=400, detail="Target colours must be between 2 and 128.")

    valid_methods = {"smart_hue", "lab_kmeans"}
    if color_reduction_method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid color reduction method. Must be one of {sorted(valid_methods)}.",
        )

    if upscale_factor < 1 or upscale_factor > 4:
        raise HTTPException(status_code=400, detail="Upscale factor must be between 1 and 4.")

    valid_bg_modes = {"transparent", "white"}
    if background_mode not in valid_bg_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid background mode. Must be one of {sorted(valid_bg_modes)}.",
        )

    valid_rembg_models = {"u2netp", "u2net", "isnet-general-use", "silueta"}
    if rembg_model not in valid_rembg_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rembg model. Must be one of {sorted(valid_rembg_models)}.",
        )

    if smoothing_strength < 0 or smoothing_strength > 5:
        raise HTTPException(status_code=400, detail="Smoothing strength must be between 0 and 5.")

    valid_smoothing_methods = {"gaussian", "bilateral"}
    if smoothing_method not in valid_smoothing_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid smoothing method. Must be one of {sorted(valid_smoothing_methods)}.",
        )

    if sharpen_strength < 0 or sharpen_strength > 5:
        raise HTTPException(status_code=400, detail="Sharpen strength must be between 0 and 5.")

    if median_filter_strength < 0 or median_filter_strength > 5:
        raise HTTPException(status_code=400, detail="Median filter strength must be between 0 and 5.")

    if morphological_close_strength < 0 or morphological_close_strength > 3:
        raise HTTPException(status_code=400, detail="Morphological closing strength must be between 0 and 3.")

    if drop_tiny_blobs_min_size < 0 or drop_tiny_blobs_min_size > 50:
        raise HTTPException(status_code=400, detail="Drop tiny blobs min size must be between 0 and 50.")

    if color_precision < 1 or color_precision > 8:
        raise HTTPException(status_code=400, detail="Color precision must be between 1 and 8.")

    if filter_speckle < 0 or filter_speckle > 30:
        raise HTTPException(status_code=400, detail="Speckle filter must be between 0 and 30.")

    if layer_difference < 1 or layer_difference > 64:
        raise HTTPException(status_code=400, detail="Layer difference must be between 1 and 64.")

    if corner_threshold < 0 or corner_threshold > 180:
        raise HTTPException(status_code=400, detail="Corner threshold must be between 0 and 180.")

    if length_threshold < 3.5 or length_threshold > 10:
        raise HTTPException(status_code=400, detail="Length threshold must be between 3.5 and 10.")

    if path_precision < 1 or path_precision > 8:
        raise HTTPException(status_code=400, detail="Path precision must be between 1 and 8.")

    if max_iterations < 1 or max_iterations > 50:
        raise HTTPException(status_code=400, detail="Max iterations must be between 1 and 50.")

    if splice_threshold < 0 or splice_threshold > 180:
        raise HTTPException(status_code=400, detail="Splice threshold must be between 0 and 180.")

    if svg_coordinate_decimals < 1 or svg_coordinate_decimals > 5:
        raise HTTPException(status_code=400, detail="SVG coordinate decimals must be between 1 and 5.")

    if svg_min_path_size < 0 or svg_min_path_size > 50:
        raise HTTPException(status_code=400, detail="SVG min path size must be between 0 and 50.")

    if svg_smoothing_passes < 0 or svg_smoothing_passes > 10:
        raise HTTPException(status_code=400, detail="SVG smoothing passes must be between 0 and 10.")

    if svg_curve_sample_step <= 0 or svg_curve_sample_step > 20:
        raise HTTPException(status_code=400, detail="SVG curve sample step must be between 0 and 20.")

    return VectorSettings(
        limit_colors=limit_colors,
        color_reduction_method=color_reduction_method,
        target_colors=target_colors,
        upscale_factor=upscale_factor,
        remove_background=remove_background,
        background_mode=background_mode,
        rembg_model=rembg_model,
        median_filter_enabled=median_filter_enabled,
        median_filter_strength=median_filter_strength,
        morphological_close_enabled=morphological_close_enabled,
        morphological_close_strength=morphological_close_strength,
        drop_tiny_blobs_enabled=drop_tiny_blobs_enabled,
        drop_tiny_blobs_min_size=drop_tiny_blobs_min_size,
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
        svg_curve_sample_step=svg_curve_sample_step,
    )