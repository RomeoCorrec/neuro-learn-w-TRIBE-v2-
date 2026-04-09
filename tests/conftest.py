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
