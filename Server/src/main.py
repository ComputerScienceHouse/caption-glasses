import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

import logging

from core import patches
from modules import classification, diarization, transcription
from api import endpoints

from config import BASE_DIR

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

logger.info("Starting FastAPI application")
app: FastAPI = FastAPI(docs_url="/swag")

if os.path.exists(os.path.join(BASE_DIR, "docs")):
    logger.info("Documentation directory found, setting up documentation endpoint!")

    app.mount(
        "/docs", StaticFiles(directory=os.path.join(BASE_DIR, "docs")), name="docs"
    )

    @app.get("/docs", include_in_schema=False)
    async def docs_redirect():
        # Mkdocs links dynamically and not being on the direct index.html causes issues
        return RedirectResponse(url="/docs/index.html")
else:
    logger.warning("Documentation directory not found, skipping documentation setup!")

app.include_router(endpoints.router)


@app.get("/", response_class=RedirectResponse)
async def read_index():
    return RedirectResponse(url="/docs/index.html")
