from pathlib import Path

import vtracer
from fastapi import HTTPException
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

from config import (
    ALLOWED_EXTENSIONS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    UPLOAD_DIR,
)
from models.settings import VectorSettings
from services.preprocessing import compute_auto_color_precision, load_palette
from services.svg_cleanup import cleanup_svg_file
from services.svg_smoothing import smooth_svg_curves


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def resolve_input_path(filename: str) -> Path:
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


def run_vtracer(
    input_path: Path,
    output_path: Path,
    settings: VectorSettings,
    override_color_precision: int | None = None,
) -> None:
    """
    Run VTracer with the supplied settings.

    If override_color_precision is provided (Phase 1D auto precision),
    use it instead of settings.color_precision. The user's slider acts
    as an upper cap.
    """

    color_precision = (
        override_color_precision
        if override_color_precision is not None
        else settings.color_precision
    )

    vtracer.convert_image_to_svg_py(
        str(input_path),
        str(output_path),
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=settings.filter_speckle,
        color_precision=color_precision,
        layer_difference=settings.layer_difference,
        corner_threshold=settings.corner_threshold,
        length_threshold=settings.length_threshold,
        max_iterations=settings.max_iterations,
        splice_threshold=settings.splice_threshold,
        path_precision=settings.path_precision,
    )

def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    """
    Convert an SVG file into a PDF using svglib + reportlab.
    """

    drawing = svg2rlg(str(svg_path))

    if drawing is None:
        raise Exception("SVG could not be converted to PDF.")

    renderPDF.drawToFile(drawing, str(pdf_path))

# ---------------------------------------------------------------------------
# Standalone vectorization entry point for the vectorize phase
# ---------------------------------------------------------------------------
def vectorize_existing_processed_image(filename: str, settings: VectorSettings) -> dict:
    """
    Run only VTracer + PDF generation on an already-processed image.

    This is used by Phase 2 (vectorize page). It assumes the cleanup phase
    has already produced storage/processed/{file_stem}_processed.png.

    If a palette sidecar file exists (Phase 1D), VTracer's color_precision
    is automatically chosen to fit the palette, and the slider becomes a cap.
    """

    input_path = resolve_input_path(filename)
    file_stem = input_path.stem

    processed_filename = f"{file_stem}_processed.png"
    processed_path = PROCESSED_DIR / processed_filename

    if not processed_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Processed image not found. "
                "Run the cleanup phase before vectorizing."
            ),
        )

    svg_filename = f"{file_stem}.svg"
    pdf_filename = f"{file_stem}.pdf"

    svg_path = OUTPUT_DIR / svg_filename
    pdf_path = OUTPUT_DIR / pdf_filename

    # Phase 1D — load palette and compute effective precision
    palette_path = processed_path.with_name(processed_path.stem + "_palette.json")
    palette = load_palette(palette_path)

    effective_precision = settings.color_precision
    if palette:
        effective_precision = compute_auto_color_precision(
            palette=palette,
            max_precision=settings.color_precision,
        )

    try:
        run_vtracer(
            input_path=processed_path,
            output_path=svg_path,
            settings=settings,
            override_color_precision=effective_precision,
        )

        smooth_svg_curves(
            svg_path=svg_path,
            enabled=settings.svg_smooth_curves,
            smoothing_passes=settings.svg_smoothing_passes,
            sample_step=settings.svg_curve_sample_step,
        )

        cleanup_svg_file(
            svg_path=svg_path,
            round_coordinates=settings.svg_round_coordinates,
            coordinate_decimals=settings.svg_coordinate_decimals,
            remove_tiny_paths=settings.svg_remove_tiny_paths,
            min_path_size=settings.svg_min_path_size,
            merge_same_color_paths=settings.svg_merge_same_color_paths,
        )

        convert_svg_to_pdf(
            svg_path=svg_path,
            pdf_path=pdf_path,
        )

    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to vectorize image: {ex}",
        )

    return {
        "svg_filename": svg_filename,
        "pdf_filename": pdf_filename,
        "processed_filename": processed_filename,
        "svg_download_url": f"/download/svg/{svg_filename}",
        "pdf_download_url": f"/download/pdf/{pdf_filename}",
        "processed_download_url": f"/download/processed/{processed_filename}",
        "palette_size": len(palette),
        "effective_color_precision": effective_precision,
    }