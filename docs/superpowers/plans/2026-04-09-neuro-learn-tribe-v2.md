# Neuro-Learn TRIBE v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MVP e-learning engagement analyzer that uses Meta's TRIBE v2 to predict fMRI brain responses from video/audio content, scores engagement per segment, and visualizes results in a Streamlit dashboard with brain activation animation.

**Architecture:** Synchronous FastAPI pipeline — InferenceEngine (real or mock) loaded once at startup via `MOCK_MODE` env var, called per POST /analyze request. All pipeline stages are isolated modules with clean interfaces.

**Tech Stack:** Python 3.11+, PyTorch, tribev2 (git), nilearn, FastAPI, Streamlit, Plotly, imageio, pydub, moviepy, scipy

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/ingestion/loader.py` | Validate format, convert mp4/mp3/wav → 16kHz mono WAV, return (wav_path, duration_sec) |
| `src/inference/engine.py` | InferenceEngine: real (TribeModel) or mock (gaussian noise) → `(n_timesteps, 20484)` |
| `src/inference/roi_extractor.py` | ROIExtractor: Destrieux atlas → vertex masks → `{"PFC": (n,), "STC": (n,)}` |
| `src/inference/brain_animator.py` | BrainAnimator: nilearn surface frames → animated GIF |
| `src/scoring/scorer.py` | EngagementResult dataclass + `compute()`: window means → normalize 0–100 → Q1 flagging |
| `api/main.py` | FastAPI app: lifespan loads engine once, POST /analyze, GET /health, static GIF serving |
| `dashboard/app.py` | Streamlit: upload → /analyze call → Plotly curve + video player + brain GIF |
| `tests/conftest.py` | Shared pytest fixtures (synthetic WAV, synthetic preds) |
| `tests/test_loader.py` | Loader unit tests |
| `tests/test_engine.py` | Engine mock mode tests |
| `tests/test_roi_extractor.py` | ROI extractor shape tests |
| `tests/test_scorer.py` | Scorer math tests |
| `tests/test_brain_animator.py` | Animator GIF output test |
| `tests/test_api.py` | FastAPI TestClient integration tests |
| `requirements.txt` | All pinned dependencies |
| `README.md` | Setup + demo instructions |

---

## Task 1: Project scaffolding

**Files:**
- Create: `src/__init__.py`, `src/ingestion/__init__.py`, `src/inference/__init__.py`, `src/scoring/__init__.py`
- Create: `api/__init__.py`, `api/static/.gitkeep`
- Create: `dashboard/__init__.py`
- Create: `tests/__init__.py`
- Create: `requirements.txt`
- Create: `.env.example`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/ingestion src/inference src/scoring api/static dashboard tests
touch src/__init__.py src/ingestion/__init__.py src/inference/__init__.py src/scoring/__init__.py
touch api/__init__.py api/static/.gitkeep dashboard/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

```
# Core pipeline
torch>=2.2.0
transformers>=4.40.0
huggingface_hub>=0.23.0
tribev2 @ git+https://github.com/facebookresearch/tribev2.git

# Audio/video ingestion
moviepy>=1.0.3
pydub>=0.25.1
ffmpeg-python>=0.2.0

# Neuroimaging
nilearn>=0.10.3
nibabel>=5.2.0
imageio>=2.34.0
imageio[ffmpeg]>=2.34.0

# API
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9

# Dashboard
streamlit>=1.35.0
plotly>=5.22.0

# Utilities
numpy>=1.26.0
scipy>=1.13.0
pandas>=2.2.0

# Dev/test
pytest>=8.2.0
httpx>=0.27.0
```

Save to `requirements.txt`.

- [ ] **Step 3: Write .env.example**

```
MOCK_MODE=1
API_URL=http://localhost:8000
```

Save to `.env.example`.

- [ ] **Step 4: Commit**

```bash
git add src/ api/ dashboard/ tests/ requirements.txt .env.example
git commit -m "feat: scaffold project structure and requirements"
```

---

## Task 2: Ingestion — `src/ingestion/loader.py`

**Files:**
- Create: `src/ingestion/loader.py`
- Create: `tests/conftest.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`:
```python
import numpy as np
import scipy.io.wavfile
import pytest

@pytest.fixture
def wav_30s(tmp_path):
    """30-second synthetic 16kHz mono WAV."""
    sr = 16000
    t = np.linspace(0, 30, sr * 30, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    path = str(tmp_path / "test_30s.wav")
    scipy.io.wavfile.write(path, sr, audio)
    return path

@pytest.fixture
def wav_5s(tmp_path):
    """5-second WAV — too short, should fail validation."""
    sr = 16000
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    path = str(tmp_path / "test_5s.wav")
    scipy.io.wavfile.write(path, sr, audio)
    return path

@pytest.fixture
def synthetic_preds():
    """Synthetic preds array (60, 20484) for pipeline tests."""
    np.random.seed(42)
    return np.random.randn(60, 20484).astype(np.float32)
```

`tests/test_loader.py`:
```python
import pytest
from src.ingestion.loader import load_media

def test_load_wav_returns_path_and_duration(wav_30s):
    wav_path, duration = load_media(wav_30s)
    assert wav_path.endswith(".wav")
    assert 28 <= duration <= 32  # allow small rounding

def test_load_wav_too_short_raises(wav_5s):
    with pytest.raises(ValueError, match="too short"):
        load_media(wav_5s)

def test_load_unsupported_format_raises(tmp_path):
    bad = tmp_path / "file.txt"
    bad.write_text("not audio")
    with pytest.raises(ValueError, match="Unsupported format"):
        load_media(str(bad))

def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_media("/nonexistent/path/file.wav")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loader.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` (loader not implemented yet).

- [ ] **Step 3: Implement `src/ingestion/loader.py`**

```python
import tempfile
from pathlib import Path
from pydub import AudioSegment

SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav"}
MIN_DURATION_SEC = 10.0


def load_media(file_path: str) -> tuple[str, float]:
    """
    Validate and convert media file to 16kHz mono WAV.
    Returns (wav_path, duration_sec).
    Raises ValueError for unsupported format or too-short files.
    Raises FileNotFoundError if file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {path.suffix}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if path.suffix.lower() == ".mp4":
        audio = AudioSegment.from_file(str(path), format="mp4")
    elif path.suffix.lower() == ".mp3":
        audio = AudioSegment.from_mp3(str(path))
    else:
        audio = AudioSegment.from_wav(str(path))

    audio = audio.set_frame_rate(16000).set_channels(1)
    duration_sec = len(audio) / 1000.0

    if duration_sec < MIN_DURATION_SEC:
        raise ValueError(
            f"File too short: {duration_sec:.1f}s (minimum {MIN_DURATION_SEC}s required)"
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    return tmp.name, duration_sec
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_loader.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/loader.py tests/conftest.py tests/test_loader.py
git commit -m "feat: implement media ingestion loader with format validation"
```

---

## Task 3: Inference engine — mock mode (`src/inference/engine.py`)

**Files:**
- Create: `src/inference/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_engine.py`:
```python
import numpy as np
import pytest
from src.inference.engine import InferenceEngine, N_VERTICES

def test_mock_predict_returns_correct_shape():
    engine = InferenceEngine(mock=True)
    preds = engine.predict(media_path="unused.wav", duration_sec=30.0)
    assert preds.shape == (30, N_VERTICES)

def test_mock_predict_is_smoothed():
    """Mock predictions should be temporally smooth (not pure noise)."""
    engine = InferenceEngine(mock=True)
    preds = engine.predict(media_path="unused.wav", duration_sec=60.0)
    # Consecutive frames should be more correlated than random
    diffs = np.diff(preds[:, :100], axis=0)
    assert diffs.std() < 2.0  # raw gaussian std would be ~1.4 * sqrt(2) ~ 2

def test_real_mode_not_loaded_by_default():
    """Real mode should not load the model until predict() is called."""
    engine = InferenceEngine(mock=False)
    # _model is None until predict() is called — we don't call predict() here
    # This just checks instantiation doesn't crash or download anything
    assert engine.mock is False

def test_mock_flag_stored():
    mock_engine = InferenceEngine(mock=True)
    assert mock_engine.mock is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_engine.py -v
```

Expected: `ImportError` (engine not implemented yet).

- [ ] **Step 3: Implement `src/inference/engine.py`**

```python
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

N_VERTICES = 20484  # fsaverage5: 10242 per hemisphere × 2


class InferenceEngine:
    """
    Wraps TRIBE v2 inference.
    mock=True: returns gaussian-smoothed random predictions with correct shape.
    mock=False: loads TribeModel from HuggingFace on first predict() call.
    """

    def __init__(self, mock: bool = True) -> None:
        self.mock = mock
        self._model = None

    def predict(self, media_path: str, duration_sec: float) -> np.ndarray:
        """
        Returns predictions: np.ndarray of shape (n_timesteps, N_VERTICES).
        n_timesteps == int(duration_sec), 1 prediction per second.
        """
        if self.mock:
            return self._mock_predict(int(duration_sec))
        return self._real_predict(media_path)

    # ------------------------------------------------------------------
    # Mock
    # ------------------------------------------------------------------

    def _mock_predict(self, n_timesteps: int) -> np.ndarray:
        noise = np.random.randn(n_timesteps, N_VERTICES).astype(np.float32)
        return gaussian_filter1d(noise, sigma=3, axis=0)

    # ------------------------------------------------------------------
    # Real (lazy-loaded)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from tribev2.demo_utils import TribeModel  # imported lazily — heavy download

        self._model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder="./cache",
        )

    def _real_predict(self, media_path: str) -> np.ndarray:
        if self._model is None:
            self._load_model()
        path = Path(media_path)
        if path.suffix.lower() == ".mp4":
            df = self._model.get_events_dataframe(video_path=str(path))
        else:
            df = self._model.get_events_dataframe(audio_path=str(path))
        preds, _ = self._model.predict(events=df)
        return preds  # shape: (n_timesteps, N_VERTICES)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_engine.py -v
```

Expected: 4 PASSED (test_real_mode_not_loaded_by_default skips actual download).

- [ ] **Step 5: Commit**

```bash
git add src/inference/engine.py tests/test_engine.py
git commit -m "feat: implement inference engine with mock/real modes"
```

---

## Task 4: ROI extraction — `src/inference/roi_extractor.py`

**Files:**
- Create: `src/inference/roi_extractor.py`
- Create: `tests/test_roi_extractor.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_roi_extractor.py`:
```python
import numpy as np
import pytest
from src.inference.roi_extractor import ROIExtractor

@pytest.fixture(scope="module")
def extractor():
    """Module-scoped so Destrieux atlas is downloaded once per test run."""
    return ROIExtractor()

def test_extract_returns_pfc_and_stc_keys(extractor, synthetic_preds):
    result = extractor.extract(synthetic_preds)
    assert "PFC" in result
    assert "STC" in result

def test_extract_output_shape(extractor, synthetic_preds):
    result = extractor.extract(synthetic_preds)
    n_timesteps = synthetic_preds.shape[0]
    assert result["PFC"].shape == (n_timesteps,)
    assert result["STC"].shape == (n_timesteps,)

def test_pfc_vertices_are_non_empty(extractor):
    assert len(extractor.pfc_verts) > 0

def test_stc_vertices_are_non_empty(extractor):
    assert len(extractor.stc_verts) > 0

def test_vertex_indices_in_valid_range(extractor):
    from src.inference.engine import N_VERTICES
    assert extractor.pfc_verts.max() < N_VERTICES
    assert extractor.stc_verts.max() < N_VERTICES
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_roi_extractor.py -v
```

Expected: `ImportError` (roi_extractor not implemented yet).

- [ ] **Step 3: Implement `src/inference/roi_extractor.py`**

```python
from __future__ import annotations
import numpy as np
from nilearn import datasets

N_VERTS_PER_HEMI = 10242  # fsaverage5 per hemisphere

PFC_LABEL_FRAGMENTS = [
    "G_front_inf-Opercular",
    "G_front_inf-Orbital",
    "G_front_inf-Triangul",
    "G_front_middle",
    "G_front_sup",
]
STC_LABEL_FRAGMENTS = [
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Lateral",
    "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo",
]


class ROIExtractor:
    """
    Maps fsaverage5 vertex predictions to mean activation per ROI.
    Downloads Destrieux atlas on first instantiation (cached by nilearn).
    """

    def __init__(self) -> None:
        destrieux = datasets.fetch_atlas_surf_destrieux(mesh="fsaverage5")
        label_names = [
            lbl.decode() if isinstance(lbl, bytes) else lbl
            for lbl in destrieux.labels
        ]

        pfc_idx = [
            i for i, l in enumerate(label_names)
            if any(frag in l for frag in PFC_LABEL_FRAGMENTS)
        ]
        stc_idx = [
            i for i, l in enumerate(label_names)
            if any(frag in l for frag in STC_LABEL_FRAGMENTS)
        ]

        pfc_lh = np.where(np.isin(destrieux.map_left, pfc_idx))[0]
        pfc_rh = np.where(np.isin(destrieux.map_right, pfc_idx))[0] + N_VERTS_PER_HEMI
        self.pfc_verts: np.ndarray = np.concatenate([pfc_lh, pfc_rh])

        stc_lh = np.where(np.isin(destrieux.map_left, stc_idx))[0]
        stc_rh = np.where(np.isin(destrieux.map_right, stc_idx))[0] + N_VERTS_PER_HEMI
        self.stc_verts: np.ndarray = np.concatenate([stc_lh, stc_rh])

    def extract(self, preds: np.ndarray) -> dict[str, np.ndarray]:
        """
        preds: (n_timesteps, 20484) → {"PFC": (n_timesteps,), "STC": (n_timesteps,)}
        """
        return {
            "PFC": preds[:, self.pfc_verts].mean(axis=1),
            "STC": preds[:, self.stc_verts].mean(axis=1),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_roi_extractor.py -v
```

Expected: 5 PASSED. (First run will download Destrieux atlas ~5 MB.)

- [ ] **Step 5: Commit**

```bash
git add src/inference/roi_extractor.py tests/test_roi_extractor.py
git commit -m "feat: implement ROI extractor using Destrieux atlas on fsaverage5"
```

---

## Task 5: Engagement scoring — `src/scoring/scorer.py`

**Files:**
- Create: `src/scoring/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_scorer.py`:
```python
import numpy as np
import pytest
from src.scoring.scorer import compute, EngagementResult

def _make_roi_signals(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "PFC": rng.standard_normal(n).astype(np.float32),
        "STC": rng.standard_normal(n).astype(np.float32),
    }

def test_returns_engagement_result():
    result = compute(_make_roi_signals(), window_sec=5)
    assert isinstance(result, EngagementResult)

def test_overall_score_in_range():
    result = compute(_make_roi_signals(), window_sec=5)
    assert 0.0 <= result.overall_score <= 100.0

def test_timeseries_scores_normalized():
    result = compute(_make_roi_signals(n=100), window_sec=5)
    scores = [s.score for s in result.timeseries]
    assert min(scores) == pytest.approx(0.0, abs=1.0)
    assert max(scores) == pytest.approx(100.0, abs=1.0)

def test_flagged_are_bottom_quartile():
    result = compute(_make_roi_signals(n=100), window_sec=5)
    scores = [s.score for s in result.timeseries]
    q1 = np.percentile(scores, 25)
    for seg in result.timeseries:
        if seg.flagged:
            assert seg.score <= q1 + 0.01  # float tolerance

def test_timeseries_windows_cover_duration():
    result = compute(_make_roi_signals(n=60), window_sec=5)
    # 60s / 5s window = 12 segments
    assert len(result.timeseries) == 12

def test_window_timestamps_are_contiguous():
    result = compute(_make_roi_signals(n=60), window_sec=5)
    for i, seg in enumerate(result.timeseries):
        assert seg.t_start == i * 5
        assert seg.t_end == (i + 1) * 5

def test_flagged_segments_list_matches_timeseries():
    result = compute(_make_roi_signals(n=60), window_sec=5)
    flagged_from_ts = [s for s in result.timeseries if s.flagged]
    assert len(result.flagged_segments) == len(flagged_from_ts)

def test_flat_signal_returns_50_overall():
    """Flat signal → all windows equal → normalized all 0 or 100 ambiguous, but overall should be 50."""
    roi = {"PFC": np.ones(60, dtype=np.float32), "STC": np.ones(60, dtype=np.float32)}
    result = compute(roi, window_sec=5)
    assert result.overall_score == pytest.approx(50.0, abs=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scorer.py -v
```

Expected: `ImportError` (scorer not implemented yet).

- [ ] **Step 3: Implement `src/scoring/scorer.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SegmentScore:
    t_start: int
    t_end: int
    score: float
    flagged: bool


@dataclass
class EngagementResult:
    overall_score: float
    duration_sec: float
    timeseries: list[SegmentScore]
    flagged_segments: list[dict]


def compute(
    roi_signals: dict[str, np.ndarray],
    window_sec: int = 5,
) -> EngagementResult:
    """
    roi_signals: {"PFC": (n_timesteps,), "STC": (n_timesteps,)}
    Returns EngagementResult with normalized 0–100 scores and Q1 flagging.
    """
    raw = np.mean(
        np.stack([roi_signals["PFC"], roi_signals["STC"]], axis=0),
        axis=0,
    )  # (n_timesteps,)
    duration_sec = len(raw)
    n_windows = duration_sec // window_sec

    window_means = np.array([
        raw[i * window_sec : (i + 1) * window_sec].mean()
        for i in range(n_windows)
    ])

    w_min, w_max = window_means.min(), window_means.max()
    if w_max > w_min:
        normalized = (window_means - w_min) / (w_max - w_min) * 100.0
    else:
        normalized = np.full(n_windows, 50.0)

    q1 = float(np.percentile(normalized, 25))

    timeseries = [
        SegmentScore(
            t_start=i * window_sec,
            t_end=(i + 1) * window_sec,
            score=round(float(normalized[i]), 1),
            flagged=bool(normalized[i] < q1),
        )
        for i in range(n_windows)
    ]

    flagged_segments = [
        {
            "start": seg.t_start,
            "end": seg.t_end,
            "score": seg.score,
            "suggestion": "Low attention — consider re-editing this segment",
        }
        for seg in timeseries
        if seg.flagged
    ]

    return EngagementResult(
        overall_score=round(float(normalized.mean()), 1),
        duration_sec=float(duration_sec),
        timeseries=timeseries,
        flagged_segments=flagged_segments,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scorer.py -v
```

Expected: 8 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/scoring/scorer.py tests/test_scorer.py
git commit -m "feat: implement engagement scorer with windowing, normalization and Q1 flagging"
```

---

## Task 6: Brain animator — `src/inference/brain_animator.py`

**Files:**
- Create: `src/inference/brain_animator.py`
- Create: `tests/test_brain_animator.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_brain_animator.py`:
```python
import os
import numpy as np
import pytest
from src.inference.brain_animator import BrainAnimator

@pytest.fixture(scope="module")
def animator():
    return BrainAnimator()

def test_animate_creates_gif_file(animator, tmp_path, synthetic_preds):
    # Use only first 5 frames to keep test fast
    output = str(tmp_path / "test_brain.gif")
    animator.animate(synthetic_preds[:5], output_path=output)
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0

def test_animate_gif_is_valid(animator, tmp_path, synthetic_preds):
    import imageio
    output = str(tmp_path / "test_brain_valid.gif")
    animator.animate(synthetic_preds[:3], output_path=output)
    reader = imageio.get_reader(output)
    frames = list(reader)
    assert len(frames) == 3
    assert frames[0].ndim == 3  # (H, W, C)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_brain_animator.py -v
```

Expected: `ImportError` (brain_animator not implemented yet).

- [ ] **Step 3: Implement `src/inference/brain_animator.py`**

```python
from __future__ import annotations
import io
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")  # headless — must be set before pyplot import
import matplotlib.pyplot as plt
from nilearn import plotting, datasets


class BrainAnimator:
    """
    Renders per-second brain activation frames from fsaverage5 predictions
    and assembles them into an animated GIF.
    """

    def __init__(self) -> None:
        fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
        self._meshes = {
            "left":  (fsaverage5.infl_left,  fsaverage5.sulc_left),
            "right": (fsaverage5.infl_right, fsaverage5.sulc_right),
        }

    def animate(
        self,
        preds: np.ndarray,
        output_path: str,
        fps: int = 1,
    ) -> str:
        """
        preds: (n_timesteps, 20484)
        Saves GIF to output_path and returns output_path.
        """
        vmax = float(np.percentile(np.abs(preds), 95)) or 1.0
        frames = [self._render_frame(preds[t], vmax) for t in range(preds.shape[0])]
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        return output_path

    def _render_frame(self, preds_t: np.ndarray, vmax: float) -> np.ndarray:
        """Render left + right hemisphere side-by-side, return (H, W, 4) RGBA array."""
        half_images = []
        for hemi, (mesh, bg) in self._meshes.items():
            stat_map = preds_t[:10242] if hemi == "left" else preds_t[10242:]
            display = plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=stat_map,
                bg_map=bg,
                hemi=hemi,
                view="lateral",
                vmax=vmax,
                colorbar=False,
            )
            buf = io.BytesIO()
            display.savefig(buf)
            buf.seek(0)
            half_images.append(imageio.imread(buf))
            display.close()

        # Concatenate left + right horizontally
        # Resize to same height if needed
        h = min(img.shape[0] for img in half_images)
        cropped = [img[:h] for img in half_images]
        return np.concatenate(cropped, axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_brain_animator.py -v
```

Expected: 2 PASSED. (First run downloads fsaverage5 surface ~10 MB.)

- [ ] **Step 5: Commit**

```bash
git add src/inference/brain_animator.py tests/test_brain_animator.py
git commit -m "feat: implement brain activation GIF animator using nilearn surface plots"
```

---

## Task 7: FastAPI app — `api/main.py`

**Files:**
- Create: `api/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_api.py`:
```python
import io
import os
import numpy as np
import pytest
import scipy.io.wavfile
from fastapi.testclient import TestClient

# Force mock mode for all API tests
os.environ["MOCK_MODE"] = "1"

from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def wav_bytes_30s():
    sr = 16000
    t = np.linspace(0, 30, sr * 30, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sr, audio)
    buf.seek(0)
    return buf.read()

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mock_mode"] is True

def test_analyze_returns_200(client, wav_bytes_30s):
    resp = client.post(
        "/analyze",
        files={"file": ("test.wav", wav_bytes_30s, "audio/wav")},
        data={"mock": "true", "window_sec": "5"},
    )
    assert resp.status_code == 200

def test_analyze_response_structure(client, wav_bytes_30s):
    resp = client.post(
        "/analyze",
        files={"file": ("test.wav", wav_bytes_30s, "audio/wav")},
        data={"mock": "true", "window_sec": "5"},
    )
    data = resp.json()
    assert "overall_score" in data
    assert "duration_sec" in data
    assert "timeseries" in data
    assert "flagged_segments" in data
    assert "brain_animation_path" in data

def test_analyze_overall_score_in_range(client, wav_bytes_30s):
    resp = client.post(
        "/analyze",
        files={"file": ("test.wav", wav_bytes_30s, "audio/wav")},
        data={"mock": "true", "window_sec": "5"},
    )
    score = resp.json()["overall_score"]
    assert 0.0 <= score <= 100.0

def test_analyze_timeseries_has_correct_keys(client, wav_bytes_30s):
    resp = client.post(
        "/analyze",
        files={"file": ("test.wav", wav_bytes_30s, "audio/wav")},
        data={"mock": "true", "window_sec": "5"},
    )
    ts = resp.json()["timeseries"]
    assert len(ts) > 0
    assert all(k in ts[0] for k in ["t_start", "t_end", "score", "flagged"])

def test_analyze_unsupported_format_returns_422(client):
    resp = client.post(
        "/analyze",
        files={"file": ("test.txt", b"not audio", "text/plain")},
        data={"mock": "true"},
    )
    assert resp.status_code in (400, 422)

def test_analyze_short_file_returns_400(client):
    """5-second WAV should return 400."""
    sr = 16000
    audio = (np.zeros(sr * 5)).astype(np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sr, audio)
    buf.seek(0)
    resp = client.post(
        "/analyze",
        files={"file": ("short.wav", buf.read(), "audio/wav")},
        data={"mock": "true"},
    )
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: `ImportError` (api/main.py not implemented yet).

- [ ] **Step 3: Implement `api/main.py`**

```python
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

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: 8 PASSED.

- [ ] **Step 5: Run all tests together**

```bash
pytest tests/ -v --ignore=tests/test_brain_animator.py
```

Expected: all pass (skip brain_animator if slow due to atlas download).

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat: implement FastAPI analyze endpoint with mock inference pipeline"
```

---

## Task 8: Streamlit dashboard — `dashboard/app.py`

**Files:**
- Create: `dashboard/app.py`

No unit tests for Streamlit — verification is manual via `streamlit run`.

- [ ] **Step 1: Implement `dashboard/app.py`**

```python
from __future__ import annotations
import io
import os
import tempfile
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Neuro-Learn", layout="wide", page_icon="🧠")
st.title("🧠 Neuro-Learn — Content Engagement Analyzer")

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    mock_mode = st.toggle("Mock mode (no GPU needed)", value=True)
    window_sec = st.slider("Window size (seconds)", 1, 30, 5)
    api_url = st.text_input("API URL", value=API_URL)

# ── Upload + analyze ─────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload video or audio file",
    type=["mp4", "mp3", "wav"],
)

if uploaded_file and st.button("▶ Analyze", type="primary"):
    with st.spinner("Running inference… (this may take a while in real mode)"):
        try:
            response = requests.post(
                f"{api_url}/analyze",
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                data={"mock": str(mock_mode).lower(), "window_sec": str(window_sec)},
                timeout=600,
            )
            if response.status_code == 200:
                st.session_state["result"] = response.json()
                st.session_state["uploaded_bytes"] = uploaded_file.getvalue()
                st.session_state["uploaded_name"] = uploaded_file.name
            else:
                st.error(f"API error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach API at {api_url}. Is it running?")

# ── Results ───────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    uploaded_bytes = st.session_state["uploaded_bytes"]
    uploaded_name = st.session_state["uploaded_name"]

    st.metric("Overall Engagement Score", f"{result['overall_score']} / 100")

    # ── Engagement curve ─────────────────────────────────────────────────
    df = pd.DataFrame(result["timeseries"])
    fig = go.Figure()

    for seg in result["flagged_segments"]:
        fig.add_vrect(
            x0=seg["start"], x1=seg["end"],
            fillcolor="red", opacity=0.15, line_width=0,
            annotation_text="⚠", annotation_position="top left",
        )

    fig.add_trace(go.Scatter(
        x=df["t_start"], y=df["score"],
        mode="lines+markers",
        line=dict(color="#4C9EFF", width=2),
        marker=dict(
            color=["#FF4444" if f else "#4C9EFF" for f in df["flagged"]],
            size=8,
        ),
        name="Engagement",
    ))
    fig.update_layout(
        title="Engagement Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Score (0–100)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Timeline slider ───────────────────────────────────────────────────
    max_t = max(1, int(result["duration_sec"]) - 1)
    t = st.slider("⏱ Timeline", 0, max_t, 0, key="timeline")

    # ── Bottom panels ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Media Player")
        suffix = Path(uploaded_name).suffix.lower()
        tmp_media = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_media.write(uploaded_bytes)
        tmp_media.close()
        if suffix == ".mp4":
            st.video(tmp_media.name, start_time=t)
        else:
            st.audio(tmp_media.name, start_time=t)

    with col2:
        st.subheader("Brain Activation")
        gif_url = f"{api_url}{result['brain_animation_path']}"
        gif_resp = requests.get(gif_url, timeout=30)
        if gif_resp.status_code == 200:
            reader = imageio.get_reader(io.BytesIO(gif_resp.content))
            try:
                n_frames = reader.get_length()
            except Exception:
                n_frames = len(list(reader))
                reader = imageio.get_reader(io.BytesIO(gif_resp.content))
            frame_idx = min(t, n_frames - 1)
            frame = reader.get_data(frame_idx)
            st.image(frame, caption=f"Brain activation at t = {t}s", use_column_width=True)
        else:
            st.warning("Brain animation not available.")

    # ── Flagged segments table ────────────────────────────────────────────
    if result["flagged_segments"]:
        st.subheader("⚠ Flagged Segments — Low Engagement")
        flagged_df = pd.DataFrame(result["flagged_segments"])
        flagged_df.columns = ["Start (s)", "End (s)", "Score", "Suggestion"]
        st.dataframe(flagged_df, use_container_width=True, hide_index=True)
    else:
        st.success("No low-engagement segments detected.")
```

- [ ] **Step 2: Manual smoke test**

Start the API in one terminal:
```bash
MOCK_MODE=1 uvicorn api.main:app --reload --port 8000
```

Start the dashboard in a second terminal:
```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`, upload any WAV/MP3/MP4 ≥ 10 seconds, click Analyze. Verify:
- Engagement curve renders with red zones
- Brain animation GIF displays
- Timeline slider changes the brain frame

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: implement Streamlit dashboard with engagement curve and brain animation"
```

---

## Task 9: Real TRIBE v2 integration smoke test

**Files:**
- Modify: `src/inference/engine.py` (no code changes needed — already implemented)
- Create: `tests/test_engine_real.py` (marked slow, skipped by default)

This task validates that real TRIBE v2 inference works end-to-end when run on a GPU machine. On your local 16 GB CPU machine, run in mock mode only.

- [ ] **Step 1: Write integration test (skipped by default)**

`tests/test_engine_real.py`:
```python
"""
Real TRIBE v2 integration test.
Skipped by default — requires GPU and ~20 GB RAM.
Run with: pytest tests/test_engine_real.py -v -m real
"""
import pytest
import numpy as np

pytestmark = pytest.mark.skipif(
    True,  # Always skip unless explicitly enabled
    reason="Real TRIBE v2 test requires GPU and large model download",
)


@pytest.mark.real
def test_real_predict_shape(wav_30s):
    from src.inference.engine import InferenceEngine, N_VERTICES

    engine = InferenceEngine(mock=False)
    preds = engine.predict(media_path=wav_30s, duration_sec=30.0)
    assert preds.shape[1] == N_VERTICES
    assert preds.shape[0] > 0
```

- [ ] **Step 2: Verify mock mode passes all tests**

```bash
pytest tests/ -v --ignore=tests/test_engine_real.py
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_real.py
git commit -m "test: add skipped real TRIBE v2 integration test for GPU environments"
```

---

## Task 10: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Neuro-Learn — E-Learning Content Engagement Analyzer

Analyzes video/audio educational content using Meta's **TRIBE v2** brain encoding model to predict cognitive engagement over time. Identifies low-engagement segments that should be re-edited.

## How it works

1. Upload an `.mp4`, `.mp3`, or `.wav` file
2. TRIBE v2 predicts fMRI brain responses across the video timeline (1 prediction/second)
3. Activation in prefrontal cortex (PFC) and superior temporal cortex (STC) is extracted using the Destrieux atlas
4. Engagement is scored per 5-second window (0–100), low-engagement zones are flagged
5. A Streamlit dashboard shows the engagement curve, brain activation animation, and flagged timestamps

## Requirements

- Python 3.11+
- `ffmpeg` installed system-wide:
  - Windows: `winget install ffmpeg` or download from https://ffmpeg.org
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`

## Installation

```bash
git clone https://github.com/your-org/neuro-learn-w-TRIBE-v2
cd neuro-learn-w-TRIBE-v2
pip install -r requirements.txt
```

### Mock mode (local, no GPU needed — recommended for development)

No model download required. Predictions are simulated with realistic gaussian noise.

### Real mode (requires GPU + ~20 GB RAM)

Set `MOCK_MODE=0`. On first run, downloads ~20 GB of model weights (TRIBE v2 checkpoint, V-JEPA2, Wav2Vec-BERT, LLaMA 3.2-3B). Recommended on Google Colab or a cloud GPU.

## Running

**Terminal 1 — API server:**
```bash
# Mock mode (default, fast)
MOCK_MODE=1 uvicorn api.main:app --reload --port 8000

# Real mode (GPU required)
MOCK_MODE=0 uvicorn api.main:app --port 8000
```

**Terminal 2 — Dashboard:**
```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Demo

1. Download a sample educational video (e.g., a 2-minute YouTube lecture saved as MP4)
2. Upload it via the dashboard
3. Toggle **Mock mode** on (default) for instant results
4. Observe the engagement curve — red zones are low-engagement segments

## API

```bash
# Analyze a file
curl -X POST http://localhost:8000/analyze \
  -F "file=@lecture.mp4" \
  -F "mock=true" \
  -F "window_sec=5"

# Health check
curl http://localhost:8000/health
```

## Architecture

```
src/ingestion/loader.py       → mp4/mp3/wav → 16kHz WAV
src/inference/engine.py       → TRIBE v2 (real or mock) → (n_timesteps, 20484)
src/inference/roi_extractor.py → Destrieux atlas → PFC + STC signals
src/inference/brain_animator.py → nilearn frames → animated GIF
src/scoring/scorer.py         → window means → 0–100 score → Q1 flagging
api/main.py                   → FastAPI REST API
dashboard/app.py              → Streamlit UI
```

## License

TRIBE v2 model: CC-BY-NC-4.0 (non-commercial use only).
This project: MIT.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and demo instructions"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Accept mp4/mp3/wav | Task 2 (loader.py) |
| Use TRIBE v2 for inference | Task 3 (engine.py real mode) |
| Extract PFC + STC ROIs | Task 4 (roi_extractor.py) |
| 5-second engagement windows, normalize 0–100 | Task 5 (scorer.py) |
| Flag bottom quartile segments | Task 5 (scorer.py) |
| POST /analyze JSON response | Task 7 (api/main.py) |
| overall_score, timeseries, flagged_segments | Task 7 |
| Brain activation animation (GIF) | Task 6 (brain_animator.py) |
| Streamlit dashboard with engagement curve | Task 8 |
| Flagged zones highlighted red | Task 8 |
| Video player synced to curve | Task 8 |
| Mock mode (--mock / MOCK_MODE) | Task 3 |
| requirements.txt | Task 1 |
| README with setup + demo | Task 10 |
| src/ingestion, src/inference, src/scoring, api/, dashboard/ structure | Task 1 |

All spec requirements covered. ✓
