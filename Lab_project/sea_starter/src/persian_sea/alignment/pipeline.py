from pathlib import Path

from persian_sea.alignment.dp import DPConfig, align_monotonic_dp
from persian_sea.config import AppConfig, segmentation_subdir
from persian_sea.io.elan import (
    read_sign_segments,
    write_aligned_subtitle_tier,
)
from persian_sea.io.subtitles import (
    find_subtitle,
    read_subtitles,
    shifted,
    write_vtt,
)
from persian_sea.types import Cue


def read_video_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def align_one(video_id: str, cfg: AppConfig) -> Path:
    seg_eaf = segmentation_subdir(cfg) / f"{video_id}.eaf"
    if not seg_eaf.exists():
        raise FileNotFoundError(f"Segmentation EAF not found: {seg_eaf}")

    sub_path = find_subtitle(cfg.paths.subtitles_raw, video_id)
    if sub_path is None:
        raise FileNotFoundError(f"Subtitle not found for {video_id}")

    out_vtt = cfg.paths.alignment_output / f"{video_id}.vtt"
    if out_vtt.exists() and not cfg.alignment.overwrite:
        return out_vtt

    signs, tier_name = read_sign_segments(
        seg_eaf,
        cfg.elan.sign_tiers,
    )

    cues = read_subtitles(sub_path)
    cues = shifted(
        cues,
        cfg.alignment.pre_shift_start_seconds,
        cfg.alignment.pre_shift_end_seconds,
    )

    if not signs:
        raise ValueError(f"No valid sign segments found in {seg_eaf}")

    dp_cfg = DPConfig(
        duration_penalty_weight=cfg.alignment.duration_penalty_weight,
        gap_penalty_weight=cfg.alignment.gap_penalty_weight,
        max_gap_seconds=cfg.alignment.max_gap_seconds,
        candidate_window_size=cfg.alignment.candidate_window_size,
        similarity_weight=0.0,
    )

    alignments = align_monotonic_dp(cues, signs, dp_cfg)

    aligned_cues = [
        Cue(c.start, c.end, c.text)
        for c in cues
    ]

    for a in alignments:
        aligned_cues[a.cue_index].start = a.start
        aligned_cues[a.cue_index].end = a.end

    aligned_cues = shifted(
        aligned_cues,
        cfg.alignment.post_shift_start_seconds,
        cfg.alignment.post_shift_end_seconds,
    )

    write_vtt(aligned_cues, out_vtt)

    if cfg.alignment.write_updated_eaf:
        updated_eaf = seg_eaf.with_name(
            seg_eaf.stem + "_updated.eaf"
        )
        write_aligned_subtitle_tier(
            seg_eaf,
            aligned_cues,
            updated_eaf,
            tier_id=cfg.elan.aligned_subtitle_tier,
        )

    print(
        f"[align] {video_id}: "
        f"{len(cues)} cues, {len(signs)} signs, tier={tier_name}"
    )

    return out_vtt


def align_all(cfg: AppConfig) -> None:
    cfg.paths.alignment_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for video_id in read_video_ids(cfg.paths.video_ids):
        align_one(video_id, cfg)
