from __future__ import annotations
import os
import time
import uuid
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from src.ingestion.loader import load_media
from src.inference.engine import InferenceEngine
from src.inference.roi_extractor import ROIExtractor
from src.inference.brain_animator import BrainAnimator
from src.scoring.scorer import compute

STATIC_DIR = Path("api/static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
GIF_TTL_SEC = 3600

_engine: InferenceEngine | None = None
_roi_extractor: ROIExtractor | None = None
_animator: BrainAnimator | None = None
_gif_registry: dict[str, float] = {}  # gif_path → created_at timestamp


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _roi_extractor, _animator
    mock = os.getenv("MOCK_MODE", "1") == "1"
    _engine = InferenceEngine(mock=mock)
    _roi_extractor = ROIExtractor()
    _animator = BrainAnimator()
    yield


app = FastAPI(title="Neuro-Learn API", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _cleanup_stale_gifs() -> None:
    now = time.time()
    stale = [p for p, t in list(_gif_registry.items()) if now - t > GIF_TTL_SEC]
    for p in stale:
        Path(p).unlink(missing_ok=True)
        _gif_registry.pop(p, None)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "mock_mode": _engine.mock if _engine else None,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    mock: bool = Form(False),
    window_sec: int = Form(5),
):
    _cleanup_stale_gifs()

    suffix = Path(file.filename or "upload.wav").suffix.lower()
    tmp_upload = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    gif_id: str | None = None
    wav_path: str | None = None

    try:
        tmp_upload.write(await file.read())
        tmp_upload.close()

        try:
            wav_path, duration_sec = load_media(tmp_upload.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Per-request mock override: if request sends mock=True, use mock engine
        engine = InferenceEngine(mock=True) if mock else _engine

        try:
            preds = engine.predict(wav_path, duration_sec)
            roi_signals = _roi_extractor.extract(preds)
            result = compute(roi_signals, window_sec=window_sec)

            gif_id = uuid.uuid4().hex
            gif_path = str(STATIC_DIR / f"brain_anim_{gif_id}.gif")
            _animator.animate(preds, gif_path)
            _gif_registry[gif_path] = time.time()

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            if wav_path is not None:
                Path(wav_path).unlink(missing_ok=True)

    finally:
        Path(tmp_upload.name).unlink(missing_ok=True)

    return {
        "overall_score": result.overall_score,
        "duration_sec": result.duration_sec,
        "timeseries": [
            {
                "t_start": s.t_start,
                "t_end": s.t_end,
                "score": s.score,
                "flagged": s.flagged,
            }
            for s in result.timeseries
        ],
        "flagged_segments": result.flagged_segments,
        "brain_animation_path": f"/static/brain_anim_{gif_id}.gif",
    }
