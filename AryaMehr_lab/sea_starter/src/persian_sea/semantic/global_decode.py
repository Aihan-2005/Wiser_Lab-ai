from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentCandidate:
    start: float
    end: float
    score: float
    local_rank: int


def monotonic_global_decode(
    candidates_per_cue: list[list[AlignmentCandidate]],
    rough_times: list[tuple[float, float]],
    overlap_penalty: float,
    gap_change_penalty: float,
) -> list[AlignmentCandidate]:
    """
    Viterbi-style decoding over subtitle order.

    It rewards strong local semantic scores while discouraging:
      - large overlap between consecutive subtitle spans;
      - a drastic change in inter-subtitle gap compared with rough timing.

    Candidates that move backward in start time are disallowed.
    """
    if not candidates_per_cue:
        return []

    dp: list[list[float]] = []
    back: list[list[int]] = []

    dp.append([c.score for c in candidates_per_cue[0]])
    back.append([-1] * len(candidates_per_cue[0]))

    for i in range(1, len(candidates_per_cue)):
        current_scores = [-1e30] * len(candidates_per_cue[i])
        current_back = [-1] * len(candidates_per_cue[i])

        rough_prev = rough_times[i - 1]
        rough_cur = rough_times[i]
        rough_gap = rough_cur[0] - rough_prev[1]

        for j, cur in enumerate(candidates_per_cue[i]):
            for k, prev in enumerate(candidates_per_cue[i - 1]):
                if cur.start < prev.start:
                    continue

                overlap = max(0.0, prev.end - cur.start)
                actual_gap = cur.start - prev.end

                transition = (
                    - overlap_penalty * overlap
                    - gap_change_penalty * abs(actual_gap - rough_gap)
                )

                score = dp[i - 1][k] + cur.score + transition

                if score > current_scores[j]:
                    current_scores[j] = score
                    current_back[j] = k

        # Defensive fallback if all candidates violated monotonic start order.
        if max(current_scores) <= -1e29:
            best_prev = max(range(len(dp[i - 1])), key=lambda x: dp[i - 1][x])
            for j, cur in enumerate(candidates_per_cue[i]):
                current_scores[j] = dp[i - 1][best_prev] + cur.score
                current_back[j] = best_prev

        dp.append(current_scores)
        back.append(current_back)

    last = max(range(len(dp[-1])), key=lambda j: dp[-1][j])
    path = [last]

    for i in range(len(candidates_per_cue) - 1, 0, -1):
        last = back[i][last]
        path.append(last)

    path.reverse()

    return [
        candidates_per_cue[i][path[i]]
        for i in range(len(path))
    ]
