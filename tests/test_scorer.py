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
