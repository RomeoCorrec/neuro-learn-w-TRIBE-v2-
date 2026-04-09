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
