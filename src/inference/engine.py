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
