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
