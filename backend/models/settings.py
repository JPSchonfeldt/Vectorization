from dataclasses import dataclass, asdict
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------
@dataclass
class VectorSettings:
    """
    All user-controllable vectorization settings.

    These come from form fields in the upload page or JSON in the editor page.
    """

    # Image cleanup
    limit_colors: bool = False
    color_reduction_method: str = "smart_hue"
    target_colors: int = 16
    upscale_factor: int = 1
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

    # SVG post-processing
    svg_round_coordinates: bool = True
    svg_coordinate_decimals: int = 2
    svg_remove_tiny_paths: bool = True
    svg_min_path_size: int = 4
    svg_merge_same_color_paths: bool = True

    # SVG curve smoothing (Phase 1E)
    svg_smooth_curves: bool = True
    svg_smoothing_passes: int = 2
    svg_curve_sample_step: float = 2.0

    def to_dict(self) -> dict:
        """
        Convert the settings into a plain dictionary
        for serialisation in API responses.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_vector_settings(
    limit_colors: bool,
    color_reduction_method: str,
    target_colors: int,
    upscale_factor: int,
    smoothing_strength: int,
    smoothing_method: str,
    sharpen_strength: int,
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
    """
    Validate the incoming settings and return a VectorSettings object.

    Raises HTTPException with a clear message if any value is out of range.
    """

    if target_colors < 2 or target_colors > 128:
        raise HTTPException(
            status_code=400,
            detail="Target colours must be between 2 and 128."
        )

    if upscale_factor < 1 or upscale_factor > 4:
        raise HTTPException(
            status_code=400,
            detail="Upscale factor must be between 1 and 4."
        )

    if smoothing_strength < 0 or smoothing_strength > 5:
        raise HTTPException(
            status_code=400,
            detail="Smoothing strength must be between 0 and 5."
        )


    valid_smoothing_methods = {"gaussian", "bilateral"}
    if smoothing_method not in valid_smoothing_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid smoothing method. Must be one of {sorted(valid_smoothing_methods)}.",
        )

    if sharpen_strength < 0 or sharpen_strength > 5:
        raise HTTPException(
            status_code=400,
            detail="Sharpen strength must be between 0 and 5."
        )

    if color_precision < 1 or color_precision > 8:
        raise HTTPException(
            status_code=400,
            detail="Color precision must be between 1 and 8."
        )

    if filter_speckle < 0 or filter_speckle > 30:
        raise HTTPException(
            status_code=400,
            detail="Speckle filter must be between 0 and 30."
        )

    if layer_difference < 1 or layer_difference > 64:
        raise HTTPException(
            status_code=400,
            detail="Layer difference must be between 1 and 64."
        )

    if corner_threshold < 0 or corner_threshold > 180:
        raise HTTPException(
            status_code=400,
            detail="Corner threshold must be between 0 and 180."
        )

    if length_threshold < 3.5 or length_threshold > 10:
        raise HTTPException(
            status_code=400,
            detail="Length threshold must be between 3.5 and 10."
        )

    if path_precision < 1 or path_precision > 8:
        raise HTTPException(
            status_code=400,
            detail="Path precision must be between 1 and 8."
        )

    if max_iterations < 1 or max_iterations > 50:
        raise HTTPException(
            status_code=400,
            detail="Max iterations must be between 1 and 50."
        )

    if splice_threshold < 0 or splice_threshold > 180:
        raise HTTPException(
            status_code=400,
            detail="Splice threshold must be between 0 and 180."
        )

    if svg_coordinate_decimals < 1 or svg_coordinate_decimals > 5:
        raise HTTPException(
            status_code=400,
            detail="SVG coordinate decimals must be between 1 and 5.",
        )

    if svg_min_path_size < 0 or svg_min_path_size > 50:
        raise HTTPException(
            status_code=400,
            detail="SVG min path size must be between 0 and 50.",
        )

    if svg_smoothing_passes < 0 or svg_smoothing_passes > 10:
        raise HTTPException(
            status_code=400,
            detail="SVG smoothing passes must be between 0 and 10.",
        )

    if svg_curve_sample_step <= 0 or svg_curve_sample_step > 20:
        raise HTTPException(
            status_code=400,
            detail="SVG curve sample step must be between 0 and 20.",
        )

    return VectorSettings(
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
        svg_curve_sample_step=svg_curve_sample_step,
    )