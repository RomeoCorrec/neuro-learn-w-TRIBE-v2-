# Design Spec — Neuro-Learn E-Learning Content Optimization Tool

**Date:** 2026-04-09  
**Status:** Approved  
**Model:** Meta TRIBE v2 (`facebook/tribev2`)  
**Architecture:** Option A — Synchronous pipeline with isolated worker

---

## 1. Goal

Analyze educational video/audio content using TRIBE v2 to predict cognitive engagement over time. Identify low-engagement segments (bottom quartile) and surface them to content creators for re-editing. Expose a REST API and a Streamlit dashboard with brain activation visualization.

---

## 2. Constraints

- **Hardware:** No GPU. 16 GB RAM. TRIBE v2 full inference (V-JEPA2 + LLaMA 3.2-3B + Wav2Vec-BERT + TRIBE checkpoint) may exceed available RAM.
- **Mock mode:** The env var `MOCK_MODE=1` at startup replaces TRIBE v2 with numpy-simulated predictions of shape `(n_timesteps, 20484)` — same interface, no model download. The `POST /analyze` body also accepts a per-request `mock` bool that overrides the server default.
- **Real mode:** Activates full TRIBE v2 pipeline. Recommended on GPU (Colab/Kaggle/cloud) or with patience on CPU.
- **License:** TRIBE v2 is CC-BY-NC-4.0.

---

## 3. Project Structure

```
neuro-learn-w-TRIBE-v2/
├── src/
│   ├── ingestion/
│   │   └── loader.py          # Validate + normalize mp4/mp3/wav → 16kHz WAV
│   ├── inference/
│   │   ├── engine.py          # InferenceEngine: loads TRIBE v2 (real or mock)
│   │   ├── roi_extractor.py   # Maps fsaverage5 vertices → PFC + STC via Destrieux atlas
│   │   └── brain_animator.py  # Renders per-second brain frames → GIF via nilearn + imageio
│   └── scoring/
│       └── scorer.py          # Window aggregation, normalization, flagging
├── api/
│   └── main.py                # FastAPI: POST /analyze, GET /health, GET /static/{file}
├── dashboard/
│   └── app.py                 # Streamlit: engagement curve + brain animation + video player
├── requirements.txt
└── README.md
```

---

## 4. Data Flow

```
Input file (mp4 / mp3 / wav)
    │
    ▼ src/ingestion/loader.py
    Validated, extracted to 16kHz mono WAV
    events DataFrame: [type, start, duration, filepath, text, context]
    │
    ▼ src/inference/engine.py
    Real:  TribeModel.from_pretrained("facebook/tribev2").predict(events)
    Mock:  numpy random with HRF-shaped smoothing
    → preds: np.ndarray (n_timesteps, ~20484 vertices) at 1 Hz
    │
    ▼ src/inference/roi_extractor.py
    nilearn fetch_atlas_surf_destrieux on fsaverage5
    PFC vertices: frontal labels (G_front_inf-Opercular, G_front_inf-Orbital,
                                  G_front_inf-Triangul, G_front_middle, G_front_sup,
                                  G_precentral — bilateral)
    STC vertices: temporal labels (G_temp_sup-G_T_transv, G_temp_sup-Lateral,
                                   G_temp_sup-Plan_polar, G_temp_sup-Plan_tempo — bilateral)
    → roi_signals: {"PFC": (n_timesteps,), "STC": (n_timesteps,)}
    │
    ▼ src/scoring/scorer.py
    mean(PFC, STC) → raw signal (n_timesteps,) at 1 Hz
    window_sec (default 5s) → window means
    min-max normalize → 0–100
    flag: score < Q1 → flagged=True
    → EngagementResult dataclass
    │
    ▼ src/inference/brain_animator.py
    nilearn.plotting.plot_surf_stat_map (headless matplotlib)
    one frame per second → GIF via imageio
    → brain_anim_<uuid>.gif saved to api/static/
    │
    ▼ api/main.py → JSON response
    │
    ▼ dashboard/app.py → Streamlit UI
```

---

## 5. TRIBE v2 Inference API

```python
# Real mode
from tribev2.demo_utils import TribeModel
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
df = model.get_events_dataframe(video_path="path/to/video.mp4")
# or: audio_path="path/to/audio.wav"
preds, segments = model.predict(events=df)
# preds.shape == (n_timesteps, ~20484)  — 1 prediction per second
```

```python
# Mock mode — same interface
import numpy as np
from scipy.ndimage import gaussian_filter1d
n_timesteps = int(duration_sec)
preds = gaussian_filter1d(np.random.randn(n_timesteps, 20484), sigma=3, axis=0)
```

---

## 6. ROI Extraction

```python
from nilearn import datasets
destrieux = datasets.fetch_atlas_surf_destrieux(mesh='fsaverage5')
# destrieux.map_left, destrieux.map_right: (10242,) vertex label arrays
# destrieux.labels: list of label names (bytes)

PFC_LABELS = [b'G_front_inf-Opercular', b'G_front_inf-Orbital',
              b'G_front_inf-Triangul', b'G_front_middle', b'G_front_sup']
STC_LABELS = [b'G_temp_sup-G_T_transv', b'G_temp_sup-Lateral',
              b'G_temp_sup-Plan_polar', b'G_temp_sup-Plan_tempo']

# Extract vertex indices per ROI (both hemispheres)
# Average preds over those vertices per timestep
```

TRIBE v2 outputs bilateral fsaverage5 (left + right hemisphere concatenated → 20484 vertices total). Left: indices 0–10241, Right: 10242–20483.

---

## 7. API Specification

### `POST /analyze`

**Request:** `multipart/form-data`
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | File | required | mp4 / mp3 / wav |
| `mock` | bool | false | Use mock inference |
| `window_sec` | int | 5 | Aggregation window size |

**Response 200:**
```json
{
  "overall_score": 72.4,
  "duration_sec": 120,
  "timeseries": [
    {"t_start": 0, "t_end": 5, "score": 68.1, "flagged": false},
    {"t_start": 5, "t_end": 10, "score": 31.2, "flagged": true}
  ],
  "flagged_segments": [
    {
      "start": 5, "end": 10, "score": 31.2,
      "suggestion": "Low attention — consider re-editing this segment"
    }
  ],
  "brain_animation_path": "/static/brain_anim_<uuid>.gif"
}
```

**Errors:**
| Code | Condition |
|------|-----------|
| 400 | File too short (< 10s) |
| 422 | Unsupported format |
| 500 | Inference crash (structured message) |

### `GET /health`
```json
{"status": "ok", "model_loaded": true, "mock_mode": false}
```

### `GET /static/{filename}`
Serves generated GIF files. TTL: 1 hour (in-memory registry, cleaned on next request).

---

## 8. Streamlit Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [Upload fichier]   [▶ Analyser]   [Mock mode toggle]       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Engagement curve (Plotly)                                 │
│   ████████░░░░░░████████  ← flagged zones highlighted red  │
│   ──────────────────────── ← timeline slider               │
│                                                             │
├──────────────────┬──────────────────────────────────────────┤
│  Video/audio     │  Brain activation animation              │
│  player          │  (GIF, lateral L/R views)               │
│  (st.video /     │  synced to timeline slider              │
│   st.audio)      │                                         │
└──────────────────┴──────────────────────────────────────────┘
```

Synchronization: `st.slider` controls both a JS seek on the video element (via `st.components.v1.html`) and the currently-displayed GIF frame index (rendered as `st.image`).

---

## 9. Dependencies (requirements.txt sketch)

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
```

---

## 10. README structure

1. Overview + demo screenshot
2. Requirements (Python 3.11+, ffmpeg system dependency)
3. Installation (real mode vs mock mode)
4. Running the API: `MOCK_MODE=1 uvicorn api.main:app --reload` (or `MOCK_MODE=0` for real TRIBE v2)
5. Running the dashboard: `streamlit run dashboard/app.py`
6. Demo instructions with sample file
7. Architecture diagram

---

## 11. Out of scope (MVP)

- Multi-subject averaging
- Per-subject fine-tuning
- Real-time streaming inference
- Authentication / rate limiting
- Persistent storage of results
- Training or fine-tuning TRIBE v2
