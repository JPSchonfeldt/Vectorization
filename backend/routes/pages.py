from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    UPLOAD_DIR,
)


router = APIRouter(tags=["pages"])

templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# New home page — upload step
# ---------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse(request, "upload.html")


# ---------------------------------------------------------------------------
# Phase 1 — Cleanup page (placeholder for now, real page in Stage B7)
# ---------------------------------------------------------------------------
@router.get("/cleanup/{file_id}", response_class=HTMLResponse)
def cleanup_page(request: Request, file_id: str):
    upload_path = UPLOAD_DIR / file_id

    if not upload_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded file not found.",
        )

    return templates.TemplateResponse(
        request,
        "cleanup.html",
        {"file_id": file_id},
    )

# ---------------------------------------------------------------------------
# Phase 2 — Vectorize page
# ---------------------------------------------------------------------------
@router.get("/vectorize/{file_id}", response_class=HTMLResponse)
def vectorize_page(request: Request, file_id: str):
    upload_path = UPLOAD_DIR / file_id

    if not upload_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded file not found.",
        )

    # We also need the processed image to exist before vectorization can run.
    file_stem = upload_path.stem
    processed_path = PROCESSED_DIR / f"{file_stem}_processed.png"

    if not processed_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "Processed image not found. "
                "Please complete the cleanup phase first."
            ),
        )

    return templates.TemplateResponse(
        request,
        "vectorize.html",
        {"file_id": file_id, "processed_filename": processed_path.name},
    )

# ---------------------------------------------------------------------------
# Diagnostic endpoints
# ---------------------------------------------------------------------------
@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/files/uploads")
def list_uploaded_files():
    return {
        "uploads": [
            file.name
            for file in UPLOAD_DIR.iterdir()
            if file.is_file()
        ]
    }


@router.get("/files/outputs")
def list_output_files():
    return {
        "outputs": [
            file.name
            for file in OUTPUT_DIR.iterdir()
            if file.is_file()
        ]
    }


@router.get("/files/processed")
def list_processed_files():
    return {
        "processed": [
            file.name
            for file in PROCESSED_DIR.iterdir()
            if file.is_file()
        ]
    }