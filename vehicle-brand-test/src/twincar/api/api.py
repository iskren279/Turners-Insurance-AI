"""
src/twincar/api/api.py — TwinCar FastAPI inference API

Endpoints:
    GET  /health       — liveness check
    GET  /model-info   — loaded model metadata
    POST /predict      — upload an image, get make/model/year/confidence

Launch (local):
    uvicorn twincar.api:app --reload
    uvicorn twincar.api:app --reload --port 8000

With a checkpoint:
    TWINCAR_CHECKPOINT=outputs/checkpoints/best.pt uvicorn twincar.api:app --reload

On HF Spaces (set env vars in Space settings):
    HF_MODEL_REPO=twincar-group2/twincar-classifier

Install API dependencies:
    pip install -e ".[api]"
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from twincar.inference import ModelBundle, load_bundle, predict_single

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

# ── Checkpoint resolution (mirrors gradio_app.py) ─────────────────────────────

DEFAULT_HF_REPO = "twincar-group2/twincar-classifier"


def _download_from_hub(repo: str) -> Path | None:
    """
    Download the current candidate checkpoint from HF Hub.

    Reads checkpoint_manifest.json first to get the correct filename —
    avoids hardcoding 'best.pt' so deployments pick up model promotions
    automatically when the manifest is updated.
    """
    try:
        import json

        from huggingface_hub import hf_hub_download

        manifest_path = hf_hub_download(repo_id=repo, filename="checkpoint_manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        filename = manifest["current_candidate"]["file"]
        log.info("Manifest → checkpoint: '%s'", filename)
        return Path(hf_hub_download(repo_id=repo, filename=filename))

    except Exception as exc:
        log.warning("HF Hub checkpoint download failed (%s) — stub mode.", exc)
        return None


def _resolve_checkpoint() -> Path | None:
    """
    Resolve checkpoint from environment variables (priority order):
      1. TWINCAR_CHECKPOINT — local path
      2. HF_MODEL_REPO     — reads manifest, downloads current candidate

    Returns None if neither is set (stub mode).

    Note: HF Hub download only runs when HF_MODEL_REPO is explicitly set —
    not as a silent fallback — so tests without env vars stay in stub mode.
    """
    local = os.environ.get("TWINCAR_CHECKPOINT")
    if local:
        return Path(local)

    repo = os.environ.get("HF_MODEL_REPO")
    if not repo:
        return None

    return _download_from_hub(repo)


# ── App state ─────────────────────────────────────────────────────────────────

_bundle: ModelBundle | None = None


def get_bundle() -> ModelBundle:
    if _bundle is None:
        raise RuntimeError("Bundle not initialised — lifespan did not run.")
    return _bundle


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release nothing (model stays in memory)."""
    global _bundle
    device = os.environ.get("TWINCAR_DEVICE", "auto")
    checkpoint = _resolve_checkpoint()
    _bundle = load_bundle(checkpoint=checkpoint, device=device)
    mode = "stub" if _bundle.stub else f"{len(_bundle.class_names)} classes on {_bundle.device}"
    log.info("API ready — %s", mode)
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="TwinCar API",
    description="Vehicle make, model, and year identification from a single image.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Response schemas ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    stub_mode: bool


class ModelInfoResponse(BaseModel):
    stub: bool
    arch: str
    num_classes: int
    device: str
    image_size: int
    label_mappings_available: bool


class TopKItem(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    fine_class: str
    make: str
    model: str
    year: str
    confidence: float
    top_k: list[TopKItem]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
def health() -> HealthResponse:
    """Returns 200 when the API is running. ``model_loaded`` is False in stub mode."""
    bundle = get_bundle()
    return HealthResponse(
        status="ok",
        model_loaded=not bundle.stub,
        stub_mode=bundle.stub,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Loaded model metadata",
)
def model_info() -> ModelInfoResponse:
    """Returns architecture, class count, device, and image size of the loaded model."""
    bundle = get_bundle()

    # Derive arch name from the underlying module class when not in stub mode.
    if bundle.stub:
        arch = "none"
    else:
        arch = type(bundle.model).__name__

    return ModelInfoResponse(
        stub=bundle.stub,
        arch=arch,
        num_classes=len(bundle.class_names),
        device=bundle.device,
        image_size=bundle.image_size,
        label_mappings_available=bool(bundle.label_mappings),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict vehicle make, model, and year from an image",
    responses={
        400: {"description": "Uploaded file is not a valid image"},
        503: {"description": "Model not loaded (stub mode)"},
    },
)
async def predict(
    file: Annotated[UploadFile, File(description="Vehicle image (JPEG or PNG)")],
    top_k: int = 3,
) -> PredictResponse:
    """
    Upload a vehicle image and receive the predicted make, model, year,
    full fine-class label, confidence score, and top-K alternatives.
    """
    bundle = get_bundle()

    if bundle.stub:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded — API is running in stub mode. "
                "Set TWINCAR_CHECKPOINT or HF_MODEL_REPO and restart."
            ),
        )

    # ── Read and validate uploaded image ─────────────────────────────────────
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image: {exc}",
        ) from exc

    # ── Run inference ─────────────────────────────────────────────────────────
    result = predict_single(
        image,
        bundle,
        top_k=max(1, min(top_k, 10)),   # clamp 1–10
        source_path=file.filename or "",
    )

    if result.prediction == "ERROR":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {result.error}",
        )

    return PredictResponse(
        fine_class=result.prediction,
        make=result.make,
        model=result.model,
        year=result.year,
        confidence=round(result.confidence, 4),
        top_k=[TopKItem(label=e.label, confidence=round(e.confidence, 4)) for e in result.top_k],
    )


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("twincar.api:app", host="0.0.0.0", port=8000, reload=True)
