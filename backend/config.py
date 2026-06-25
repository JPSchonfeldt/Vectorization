from pathlib import Path


# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
# Paths are relative to the working directory of the running container.
# Inside Docker, the working directory is /app, and storage is mounted there.
UPLOAD_DIR = Path("storage/uploads")
OUTPUT_DIR = Path("storage/outputs")
PROCESSED_DIR = Path("storage/processed")


# ---------------------------------------------------------------------------
# Upload restrictions
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Maximum allowed upload size in bytes (20 MB).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


# ---------------------------------------------------------------------------
# Processing limits
# ---------------------------------------------------------------------------
# After upscale, prevent images from becoming too large for VTracer to handle.
MAX_DIMENSION_AFTER_UPSCALE = 4000


# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_TITLE = "Image Vector POC API"
APP_DESCRIPTION = "Proof of concept API for converting uploaded images to vector files."
APP_VERSION = "0.2.0"


def ensure_storage_directories() -> None:
    """
    Ensure all storage directories exist.
    Called at application startup.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)