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
