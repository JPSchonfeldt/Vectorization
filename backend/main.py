from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    ensure_storage_directories,
)
from routes.convert import router as convert_router
from routes.download import router as download_router
from routes.pages import router as pages_router
from routes.upload import router as upload_router


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ensure_storage_directories()

    yield

    # Shutdown
    # Nothing to clean up for now.


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(pages_router)
app.include_router(upload_router)
app.include_router(convert_router)
app.include_router(download_router)