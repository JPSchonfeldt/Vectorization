import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    UPLOAD_DIR,
)


router = APIRouter(tags=["upload"])


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Save an uploaded image and return a file_id that the editor page can use.
    """

    saved = await save_uploaded_file(file)

    return JSONResponse(
        content={
            "message": "File uploaded successfully.",
            "file_id": saved["saved_filename"],
            "original_filename": saved["original_filename"],
            "cleanup_url": f"/cleanup/{saved['saved_filename']}",
        }
    )


async def save_uploaded_file(file: UploadFile) -> dict:
    """
    Validate, size-check and save an uploaded file into the upload directory.
    Returns metadata about the saved file.
    """

    original_filename = file.filename or ""

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="No file name supplied.",
        )

    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG, JPEG and WEBP files are allowed.",
        )

    content = await file.read()
    await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large. Maximum allowed size is "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            ),
        )

    safe_file_id = str(uuid.uuid4())
    saved_filename = f"{safe_file_id}{file_extension}"
    saved_path = UPLOAD_DIR / saved_filename

    try:
        saved_path.write_bytes(content)
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {ex}",
        )

    return {
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "saved_path": str(saved_path),
    }