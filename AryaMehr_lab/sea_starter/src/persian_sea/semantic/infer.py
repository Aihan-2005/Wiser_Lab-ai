from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from persian_sea.config import AppConfig
from persian_sea.io.subtitles import (
    SubtitleCue,
    find_subtitle,
    read_subtitles,
    write_vtt,
)
from persian_sea.semantic.dataset import (
    SubtitleAlignmentDataset,
    collate_alignment_batch,
)
from persian_sea.semantic.global_decode import (
    AlignmentCandidate,
    monotonic_global_decode,
)
from persian_sea.semantic.model import SemanticAligner
from persian_sea.semantic.train import move_batch, resolve_device


def load_model(cfg: AppConfig, checkpoint_path: Path):
    device = resolve_device(cfg.training.device)
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = SemanticAligner(
        text_input_dim=int(payload["text_dim"]),
        rgb_input_dim=int(payload["rgb_dim"]),
        num_views=int(payload["num_views"]),
        embedding_dim=int(payload["embedding_dim"]),
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    return model, device


@torch.no_grad()
def align_video(
    video_id: str,
    cfg: AppConfig,
    checkpoint_path: Path,
) -> Path:
    # Make a temporary one-video list so we can reuse the exact training
    # candidate construction.
    tmp_ids = cfg.paths.reports / f".tmp_{video_id}_ids.txt"
    tmp_ids.parent.mkdir(parents=True, exist_ok=True)
    tmp_ids.write_text(video_id + "\n", encoding="utf-8")

    dataset = SubtitleAlignmentDataset(
        cfg,
        tmp_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_alignment_batch,
    )

    model, device = load_model(
        cfg,
        checkpoint_path,
    )

    all_candidates: list[list[AlignmentCandidate]] = []

    for batch in loader:
        batch = move_batch(batch, device)

        output = model(
            batch,
            max_span_segments=cfg.semantic.max_span_segments,
            max_span_seconds=cfg.semantic.max_span_seconds,
            align_temperature=cfg.semantic.align_temperature,
            temporal_prior_weight=cfg.semantic.temporal_prior_weight,
            temporal_sigma_seconds=cfg.semantic.temporal_sigma_seconds,
            duration_prior_weight=cfg.semantic.duration_prior_weight,
        )

        sample = output["samples"][0]
        k = min(
            cfg.semantic.global_top_k,
            len(sample.spans),
        )

        values, indices = torch.topk(
            sample.final_scores,
            k=k,
        )

        cue_candidates = []
        for rank, (score, idx) in enumerate(
            zip(values.tolist(), indices.tolist())
        ):
            span = sample.spans[idx]
            cue_candidates.append(
                AlignmentCandidate(
                    start=span.start_time,
                    end=span.end_time,
                    score=float(score),
                    local_rank=rank,
                )
            )

        all_candidates.append(cue_candidates)

    raw_path = find_subtitle(
        cfg.paths.subtitles_raw,
        video_id,
    )
    if raw_path is None:
        raise FileNotFoundError(video_id)

    rough = read_subtitles(raw_path)
    rough_times = [(c.start, c.end) for c in rough]

    selected = monotonic_global_decode(
        all_candidates,
        rough_times,
        overlap_penalty=cfg.semantic.overlap_penalty,
        gap_change_penalty=cfg.semantic.gap_change_penalty,
    )

    aligned = [
        SubtitleCue(
            start=candidate.start,
            end=candidate.end,
            text=rough[i].text,
        )
        for i, candidate in enumerate(selected)
    ]

    out_path = (
        cfg.paths.alignment_output
        / "semantic"
        / f"{video_id}.vtt"
    )
    write_vtt(aligned, out_path)

    debug_path = out_path.with_suffix(".json")
    debug_path.write_text(
        json.dumps(
            [
                {
                    "cue_index": i,
                    "text": rough[i].text,
                    "rough_start": rough[i].start,
                    "rough_end": rough[i].end,
                    "pred_start": selected[i].start,
                    "pred_end": selected[i].end,
                    "score": selected[i].score,
                    "local_rank": selected[i].local_rank,
                }
                for i in range(len(selected))
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        tmp_ids.unlink()
    except OSError:
        pass

    print(f"[semantic-align] {video_id} -> {out_path}")
    return out_path
