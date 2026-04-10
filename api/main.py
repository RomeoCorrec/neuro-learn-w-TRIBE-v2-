from __future__ import annotations
import os
import threading
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
_progress: dict = {"stage": "idle", "pct": 0}
_progress_lock = threading.Lock()  # single-worker guard; use per-request IDs for multi-user


def _set_progress(stage: str, pct: int) -> None:
    with _progress_lock:
        _progress["stage"] = stage
        _progress["pct"] = pct


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


@app.get("/progress")
def progress():
    return _progress


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

    _set_progress("uploading", 5)
    try:
        tmp_upload.write(await file.read())
        tmp_upload.close()

        _set_progress("loading", 10)
        try:
            wav_path, duration_sec = load_media(tmp_upload.name)
        except ValueError as exc:
            _set_progress("idle", 0)
            raise HTTPException(status_code=400, detail=str(exc))

        # Per-request mock override: if request sends mock=True, use mock engine
        engine = InferenceEngine(mock=True) if mock else _engine

        try:
            _set_progress("inference", 15)
            preds = engine.predict(wav_path, duration_sec)
            _set_progress("roi_extraction", 70)
            roi_signals = _roi_extractor.extract(preds)
            _set_progress("scoring", 75)
            result = compute(roi_signals, window_sec=window_sec)

            _set_progress("animation", 80)
            gif_id = uuid.uuid4().hex
            gif_path = str(STATIC_DIR / f"brain_anim_{gif_id}.gif")
            _animator.animate(preds, gif_path)
            _gif_registry[gif_path] = time.time()
            _set_progress("done", 100)

        except HTTPException:
            _set_progress("idle", 0)
            raise
        except Exception as exc:
            _set_progress("idle", 0)
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            if wav_path is not None:
                Path(wav_path).unlink(missing_ok=True)

    finally:
        try:
            Path(tmp_upload.name).unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows: file still held by pydub/ffprobe subprocess on error

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
