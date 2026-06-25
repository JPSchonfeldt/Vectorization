from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import OUTPUT_DIR, PROCESSED_DIR, UPLOAD_DIR

router = APIRouter(tags=["download"])


@router.get("/download/svg/{filename}")
def download_svg(filename: str):
    file_path = safe_resolve_file(OUTPUT_DIR, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="SVG file not found.",
        )

    return FileResponse(
        path=file_path,
        media_type="image/svg+xml",
        filename=filename,
    )


@router.get("/download/pdf/{filename}")
def download_pdf(filename: str):
    file_path = safe_resolve_file(OUTPUT_DIR, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF file not found.",
        )

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename,
    )


@router.get("/download/processed/{filename}")
def download_processed_image(filename: str):
    file_path = safe_resolve_file(PROCESSED_DIR, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Processed image file not found.",
        )

    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=filename,
    )

@router.get("/download/upload/{filename}")
def download_upload_preview(filename: str):
    file_path = safe_resolve_file(UPLOAD_DIR, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded file not found.",
        )

    # Browser will infer the type from the extension.
    return FileResponse(
        path=file_path,
        filename=filename,
    )

def safe_resolve_file(base_dir: Path, filename: str) -> Path:
    """
    Resolve a file path inside a known base directory.
    Protects against path traversal attempts like '../../etc/passwd'.
    """

    resolved = (base_dir / filename).resolve()

    if not str(resolved).startswith(str(base_dir.resolve())):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    return resolved
