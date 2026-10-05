from dataclasses import dataclass
import numpy as np

from persian_sea.alignment.costs import alignment_cost
from persian_sea.types import Alignment, Cue, SignSegment


@dataclass
class DPConfig:
    duration_penalty_weight: float = 1.0
    gap_penalty_weight: float = 5.0
    max_gap_seconds: float = 10.0
    candidate_window_size: int = 50
    similarity_weight: float = 0.0


def _candidate_bounds(
    cues: list[Cue],
    signs: list[SignSegment],
    window_size: int,
) -> list[tuple[int, int]]:
    mids = np.asarray([s.mid for s in signs], dtype=np.float64)
    bounds: list[tuple[int, int]] = []

    for cue in cues:
        k = min(window_size, len(signs))
        nearest = np.argsort(np.abs(mids - cue.mid))[:k]
        bounds.append((int(nearest.min()), int(nearest.max()) + 1))
    return bounds


def _best_contiguous_subgroup(
    cue: Cue,
    group: list[SignSegment],
    cfg: DPConfig,
) -> tuple[int, int]:
    # Split an assigned group at very large gaps and keep the sub-group
    # with the lowest SEA-style timing/duration/gap cost.
    if not group:
        return 0, 0

    chunks: list[tuple[int, int]] = []
    chunk_start = 0

    for i in range(1, len(group)):
        if group[i].start - group[i - 1].end > cfg.max_gap_seconds:
            chunks.append((chunk_start, i))
            chunk_start = i

    chunks.append((chunk_start, len(group)))

    return min(
        chunks,
        key=lambda ab: alignment_cost(
            cue,
            group[ab[0]:ab[1]],
            cfg.duration_penalty_weight,
            cfg.gap_penalty_weight,
        ),
    )


def align_monotonic_dp(
    cues: list[Cue],
    signs: list[SignSegment],
    cfg: DPConfig,
    sim_matrix: np.ndarray | None = None,
) -> list[Alignment]:
    # State:
    # dp[i, j] = minimum cost after assigning first i cues
    #            to first j sign segments.
    if not cues or not signs:
        return []

    m, n = len(cues), len(signs)

    if n < m:
        raise ValueError(
            f"Need at least as many sign segments as subtitle cues: "
            f"signs={n}, cues={m}"
        )

    if sim_matrix is not None and sim_matrix.shape != (m, n):
        raise ValueError(
            f"sim_matrix shape must be {(m, n)}, got {sim_matrix.shape}"
        )

    candidate_bounds = _candidate_bounds(
        cues,
        signs,
        cfg.candidate_window_size,
    )

    dp = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    prev = np.full((m + 1, n + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    sim_prefix = None
    if sim_matrix is not None:
        sim_prefix = np.zeros((m, n + 1), dtype=np.float64)
        sim_prefix[:, 1:] = np.cumsum(sim_matrix, axis=1)

    for i in range(1, m + 1):
        cand_lo, cand_hi = candidate_bounds[i - 1]
        j_lo = max(i, cand_lo + 1)
        j_hi = min(n, max(j_lo, cand_hi))

        for j in range(j_lo, j_hi + 1):
            for k in range(i - 1, j):
                if not np.isfinite(dp[i - 1, k]):
                    continue

                group = signs[k:j]

                similarity_total = 0.0
                if sim_prefix is not None:
                    similarity_total = (
                        sim_prefix[i - 1, j] - sim_prefix[i - 1, k]
                    )

                cost = alignment_cost(
                    cues[i - 1],
                    group,
                    cfg.duration_penalty_weight,
                    cfg.gap_penalty_weight,
                    similarity_total=similarity_total,
                    similarity_weight=cfg.similarity_weight,
                )

                total = dp[i - 1, k] + cost
                if total < dp[i, j]:
                    dp[i, j] = total
                    prev[i, j] = k

    best_j = int(np.argmin(dp[m]))
    if not np.isfinite(dp[m, best_j]):
        raise RuntimeError(
            "DP found no valid path. Increase candidate_window_size "
            "or inspect segmentation quality."
        )

    boundaries = [0] * (m + 1)
    boundaries[m] = best_j
    cur = best_j

    for i in range(m, 0, -1):
        k = int(prev[i, cur])
        if k < 0:
            raise RuntimeError("Broken DP backtracking path.")
        boundaries[i - 1] = k
        cur = k

    results: list[Alignment] = []

    for i, cue in enumerate(cues):
        g0, g1 = boundaries[i], boundaries[i + 1]
        group = signs[g0:g1]
        if not group:
            continue

        local0, local1 = _best_contiguous_subgroup(cue, group, cfg)
        selected = group[local0:local1]

        results.append(
            Alignment(
                cue_index=i,
                start=selected[0].start,
                end=selected[-1].end,
                first_sign_index=g0 + local0,
                last_sign_index=g0 + local1 - 1,
            )
        )

    return results
