"""
app.py
-------
SatQuery AI — FastAPI backend and static front-end host.

One process serves both the JSON API under `/api` and the single-page console
in `web/`. That keeps deployment to a single command and avoids CORS entirely,
which matters when the whole thing has to run in a Colab cell or on a lab
machine with no reverse proxy.

Run:
    python app.py
    # or, with auto-reload during development:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import base64
import io
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from api.hf_config import load_config
from api.hf_model import get_runner
from controller.agentic_controller import SatQueryController
from utils.spectral_indices import BAND_PRESETS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "satquery_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load .env before anything constructs a model runner.
load_config(os.path.join(BASE_DIR, ".env"))

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Begin loading weights in the background so the first request is faster.

    This never blocks startup: the runner loads on its own thread and the UI
    polls /api/model for progress. If the model cannot load at all, the console
    still works on local specialists.
    """
    runner = get_runner()
    if runner.config.configured:
        runner.ensure_loading()
    yield


app = FastAPI(
    title="SatQuery AI",
    description="Vision-language analysis of satellite imagery through text queries.",
    version="2.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

controller = SatQueryController(reports_dir=REPORTS_DIR)

ALLOWED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MB, comfortably above a large GeoTIFF tile

MODE_TO_PAIR_TYPE = {
    "single": None,
    "change": "bi_temporal",
    "fusion": "cross_modal",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save_upload(upload: UploadFile, role: str) -> str:
    """Persist an upload to disk and return its path.

    The pipeline works from file paths because the loader needs the extension
    to pick a reader (rasterio for GeoTIFF, PIL otherwise) and the filename to
    guess modality. The role becomes part of the name so a file called
    `sar_scene.png` is recognised as SAR.
    """
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix or 'unknown'}'. "
                f"Upload GeoTIFF, TIFF, PNG, or JPEG."
            ),
        )

    payload = upload.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    path = os.path.join(UPLOAD_DIR, f"{role}_{uuid.uuid4().hex[:10]}{suffix}")
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def _png_data_uri(array: Optional[np.ndarray]) -> Optional[str]:
    """Encode an evidence board as a data URI the browser can render directly."""
    if array is None:
        return None
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _cleanup(paths: List[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Liveness probe. Answers even while the model is still loading."""
    return {"status": "ok", "service": "satquery-ai", "version": app.version}


@app.get("/api/model")
def model_status() -> Dict[str, Any]:
    """Model state, load progress, and configuration (token never included)."""
    return controller.model_status()


@app.post("/api/model/load")
def model_load() -> Dict[str, Any]:
    """Ask the runner to start loading now, if it has not already."""
    runner = get_runner()
    runner.ensure_loading()
    return runner.status()


@app.get("/api/options")
def options() -> Dict[str, Any]:
    """Everything the front end needs to build its forms."""
    return {
        "band_presets": ["(none — RGB proxy only)"] + list(BAND_PRESETS.keys()),
        "modes": [
            {
                "id": "single",
                "name": "Single image",
                "images": 1,
                "labels": ["Satellite image"],
                "blurb": "Ask what a scene contains. The model reads it; "
                         "SatQuery measures the land-cover breakdown.",
                "examples": [
                    "What is present in this image?",
                    "Identify the major land-cover types.",
                    "Describe this satellite image.",
                    "Is there a significant built-up area in this image?",
                    "Highlight the water body referred to in the query.",
                ],
            },
            {
                "id": "change",
                "name": "Change detection",
                "images": 2,
                "labels": ["Image T1 — earlier date", "Image T2 — later date"],
                "blurb": "Compare the same area on two dates. Change figures are "
                         "measured from pixels, never written by a model.",
                "examples": [
                    "What changed between these two images?",
                    "Calculate the percentage of changed area.",
                    "Has the built-up area increased, decreased, or remained unchanged?",
                    "What changed between these dates, and where?",
                ],
            },
            {
                "id": "fusion",
                "name": "Optical + SAR",
                "images": 2,
                "labels": ["Optical image", "SAR image"],
                "blurb": "Combine two sensors. Agreement between optical and radar "
                         "is what raises confidence.",
                "examples": [
                    "Compare these optical and SAR images and explain what they show.",
                    "Use both images together to identify built-up regions.",
                    "Which areas are confirmed as water by both sensors?",
                ],
            },
        ],
    }


@app.post("/api/analyze")
async def analyze(
    mode: str = Form("single"),
    query: str = Form(""),
    band_preset: str = Form(""),
    image_a: UploadFile = File(...),
    image_b: Optional[UploadFile] = File(None),
) -> JSONResponse:
    """Run one analysis and return the answer, evidence, and execution trace."""
    if mode not in MODE_TO_PAIR_TYPE:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{mode}'. Expected one of: "
                   f"{', '.join(MODE_TO_PAIR_TYPE)}.",
        )

    needs_pair = mode in ("change", "fusion")
    if needs_pair and image_b is None:
        detail = (
            "This analysis requires two images. Please provide T1 and T2."
            if mode == "change"
            else "Optical + SAR analysis requires both modalities."
        )
        raise HTTPException(status_code=400, detail=detail)

    # Role-tagged filenames so the loader's modality heuristic has a hint.
    roles = {
        "single": ("scene", None),
        "change": ("scene_t1", "scene_t2"),
        "fusion": ("optical_scene", "sar_scene"),
    }[mode]

    paths: List[str] = [_save_upload(image_a, roles[0])]
    if needs_pair and image_b is not None:
        paths.append(_save_upload(image_b, roles[1]))

    if not query.strip():
        query = {
            "single": "What is present in this image?",
            "change": "What changed between these two images?",
            "fusion": "Compare these optical and SAR images and explain what they show.",
        }[mode]

    preset = band_preset if band_preset in BAND_PRESETS else None
    started = time.time()
    try:
        result = controller.run(
            paths, query,
            declared_pair_type=MODE_TO_PAIR_TYPE[mode],
            band_preset=preset,
        )
    finally:
        _cleanup(paths)

    return JSONResponse({
        "success": result.success,
        "task": result.task,
        "scenario": result.scenario,
        "answer": result.answer,
        "confidence": result.confidence,
        "confidence_kind": result.confidence_kind,
        "measurements": result.measurements,
        "model_text": result.model_text,
        "model_labels": result.model_labels,
        "model_source": result.model_source,
        "model_ready": result.model_ready,
        "trace": result.trace_steps,
        "trace_text": result.trace_text,
        "evidence": _png_data_uri(result.evidence_image),
        "elapsed_seconds": round(time.time() - started, 2),
        "error": result.error,
    })


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Return errors in the same envelope as a successful analysis.

    The front end then renders a rejection the same way it renders a result,
    rather than needing a second code path for failures.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "answer": exc.detail, "error": exc.detail,
                 "measurements": [], "trace": [], "evidence": None},
    )


# --------------------------------------------------------------------------- #
# Front end
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/favicon.svg")
def favicon() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "static", "img", "favicon.svg"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("SATQUERY_HOST", "127.0.0.1"),
        port=int(os.environ.get("SATQUERY_PORT", "8000")),
        log_level="info",
    )
