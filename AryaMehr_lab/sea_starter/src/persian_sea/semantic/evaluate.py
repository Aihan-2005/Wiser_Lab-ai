from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from persian_sea.config import AppConfig, read_video_ids
from persian_sea.io.subtitles import (
    find_subtitle,
    read_subtitles,
)
from persian_sea.semantic.candidates import interval_iou


def evaluate_semantic_outputs(
    cfg: AppConfig,
    video_ids_path: Path,
) -> dict:
    rows = []

    for video_id in read_video_ids(video_ids_path):
        pred_path = (
            cfg.paths.alignment_output
            / "semantic"
            / f"{video_id}.vtt"
        )
        gt_path = find_subtitle(
            cfg.paths.subtitles_gt,
            video_id,
        )

        if not pred_path.exists() or gt_path is None:
            continue

        pred = read_subtitles(pred_path)
        gt = read_subtitles(gt_path)

        n = min(len(pred), len(gt))

        for i in range(n):
            rows.append({
                "video_id": video_id,
                "cue_index": i,
                "iou": interval_iou(
                    pred[i].start,
                    pred[i].end,
                    gt[i].start,
                    gt[i].end,
                ),
                "start_error": abs(pred[i].start - gt[i].start),
                "end_error": abs(pred[i].end - gt[i].end),
            })

    if not rows:
        return {
            "num_cues": 0,
            "message": "No prediction/GT pairs found.",
        }

    ious = np.asarray([r["iou"] for r in rows], dtype=np.float64)

    result = {
        "num_cues": len(rows),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "f1_proxy_iou_0.10": float((ious >= 0.10).mean()),
        "f1_proxy_iou_0.25": float((ious >= 0.25).mean()),
        "f1_proxy_iou_0.50": float((ious >= 0.50).mean()),
        "start_mae": float(
            np.mean([r["start_error"] for r in rows])
        ),
        "end_mae": float(
            np.mean([r["end_error"] for r in rows])
        ),
    }

    return result


def save_semantic_evaluation(
    result: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
