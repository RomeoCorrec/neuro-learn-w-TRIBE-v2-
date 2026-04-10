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
src/ingestion/loader.py        → mp4/mp3/wav → 16kHz WAV
src/inference/engine.py        → TRIBE v2 (real or mock) → (n_timesteps, 20484)
src/inference/roi_extractor.py → Destrieux atlas → PFC + STC signals
src/inference/brain_animator.py → nilearn frames → animated GIF
src/scoring/scorer.py          → window means → 0–100 score → Q1 flagging
api/main.py                    → FastAPI REST API
dashboard/app.py               → Streamlit UI
```

## License

TRIBE v2 model: CC-BY-NC-4.0 (non-commercial use only).
This project: MIT.
