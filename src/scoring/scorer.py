from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SegmentScore:
    t_start: int
    t_end: int
    score: float
    flagged: bool


@dataclass
class EngagementResult:
    overall_score: float
    duration_sec: float
    timeseries: list[SegmentScore]
    flagged_segments: list[dict]


def compute(
    roi_signals: dict[str, np.ndarray],
    window_sec: int = 5,
) -> EngagementResult:
    """
    roi_signals: {"PFC": (n_timesteps,), "STC": (n_timesteps,)}
    Returns EngagementResult with normalized 0–100 scores and Q1 flagging.
    """
    raw = np.mean(
        np.stack([roi_signals["PFC"], roi_signals["STC"]], axis=0),
        axis=0,
    )  # (n_timesteps,)
    duration_sec = len(raw)
    n_windows = duration_sec // window_sec

    window_means = np.array([
        raw[i * window_sec : (i + 1) * window_sec].mean()
        for i in range(n_windows)
    ])

    w_min, w_max = window_means.min(), window_means.max()
    if w_max > w_min:
        normalized = (window_means - w_min) / (w_max - w_min) * 100.0
    else:
        normalized = np.full(n_windows, 50.0)

    q1 = float(np.percentile(normalized, 25))

    timeseries = [
        SegmentScore(
            t_start=i * window_sec,
            t_end=(i + 1) * window_sec,
            score=round(float(normalized[i]), 1),
            flagged=bool(normalized[i] < q1),
        )
        for i in range(n_windows)
    ]

    flagged_segments = [
        {
            "start": seg.t_start,
            "end": seg.t_end,
            "score": seg.score,
            "suggestion": "Low attention — consider re-editing this segment",
        }
        for seg in timeseries
        if seg.flagged
    ]

    return EngagementResult(
        overall_score=round(float(normalized.mean()), 1),
        duration_sec=float(duration_sec),
        timeseries=timeseries,
        flagged_segments=flagged_segments,
    )
