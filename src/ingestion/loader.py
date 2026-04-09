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
