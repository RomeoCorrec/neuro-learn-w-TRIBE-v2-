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
