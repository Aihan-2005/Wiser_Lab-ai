from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def select_candidate_indices(
    starts: np.ndarray,
    ends: np.ndarray,
    rough_start: float,
    rough_end: float,
    radius: float,
    min_candidates: int,
    max_candidates: int,
) -> np.ndarray:


    
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)

    if len(starts) != len(ends) or len(starts) == 0:
        raise ValueError("Invalid segment timestamp arrays.")

    mids = (starts + ends) / 2.0
    cue_mid = (rough_start + rough_end) / 2.0

    lo_time = rough_start - radius
    hi_time = rough_end + radius

    lo = int(np.searchsorted(mids, lo_time, side="left"))
    hi = int(np.searchsorted(mids, hi_time, side="right"))

    lo = max(0, min(lo, len(mids) - 1))
    hi = max(lo + 1, min(hi, len(mids)))



    if hi - lo < min_candidates:
        center = int(np.argmin(np.abs(mids - cue_mid)))
        half = min_candidates // 2
        lo = max(0, center - half)
        hi = min(len(mids), lo + min_candidates)
        lo = max(0, hi - min_candidates)


    if hi - lo > max_candidates:
        local = mids[lo:hi]
        center_local = int(np.argmin(np.abs(local - cue_mid)))
        center = lo + center_local
        half = max_candidates // 2
        lo = max(0, center - half)
        hi = min(len(mids), lo + max_candidates)
        lo = max(0, hi - max_candidates)

    return np.arange(lo, hi, dtype=np.int64)


@dataclass(frozen=True)
class CandidateSpan:
    start_index: int
    end_index: int  
    
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2.0


def enumerate_candidate_spans(
    starts: np.ndarray,
    ends: np.ndarray,
    max_span_segments: int,
    max_span_seconds: float,
) -> list[CandidateSpan]:
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)

    spans: list[CandidateSpan] = []

    for a in range(len(starts)):
        max_b = min(len(starts), a + max_span_segments)

        for b in range(a + 1, max_b + 1):
            start = float(starts[a])
            end = float(ends[b - 1])

            if end <= start:
                continue

            if end - start > max_span_seconds:
                break

            spans.append(
                CandidateSpan(
                    start_index=a,
                    end_index=b,
                    start_time=start,
                    end_time=end,
                )
            )

    return spans


def interval_iou(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    if inter <= 0.0:
        return 0.0
    union = max(end_a, end_b) - min(start_a, start_b)
    return inter / union if union > 0 else 0.0


def temporal_prior(
    spans: list[CandidateSpan],
    rough_start: float,
    rough_end: float,
    sigma_seconds: float,
    duration_weight: float,
) -> np.ndarray:
    rough_mid = (rough_start + rough_end) / 2.0
    rough_duration = max(1e-3, rough_end - rough_start)
    sigma = max(1e-3, sigma_seconds)

    values = []

    for span in spans:
        midpoint_term = -0.5 * (
            (span.midpoint - rough_mid) / sigma
        ) ** 2

        duration_term = -duration_weight * abs(
            np.log(
                max(span.duration, 1e-3)
                / rough_duration
            )
        )

        values.append(midpoint_term + duration_term)

    return np.asarray(values, dtype=np.float32)
