from pathlib import Path
import subprocess

from persian_sea.config import AppConfig, segmentation_subdir


def read_video_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_segment_command(video_id: str, cfg: AppConfig) -> list[str]:
    s = cfg.segmentation
    eaf_dir = segmentation_subdir(cfg)
    eaf_path = eaf_dir / f"{video_id}.eaf"
    pose_path = cfg.paths.poses / f"{video_id}.pose"
    video_path = cfg.paths.videos / f"{video_id}.mp4"

    if not pose_path.exists():
        raise FileNotFoundError(f"Pose not found: {pose_path}")

    cmd = [
        "pose_to_segments",
        "--no-pose-link",
        f"--model={s.model}",
        f"--pose={pose_path}",
        f"--elan={eaf_path}",
        "--sign-b-threshold",
        str(s.sign_b_threshold),
        "--sign-o-threshold",
        str(s.sign_o_threshold),
    ]

    if video_path.exists():
        cmd.append(f"--video={video_path.resolve()}")

    raw_vtt = cfg.paths.subtitles_raw / f"{video_id}.vtt"
    raw_srt = cfg.paths.subtitles_raw / f"{video_id}.srt"
    if raw_vtt.exists():
        cmd.append(f"--subtitles={raw_vtt.resolve()}")
    elif raw_srt.exists():
        cmd.append(f"--subtitles={raw_srt.resolve()}")

    gt_vtt = cfg.paths.subtitles_gt / f"{video_id}.vtt"
    gt_srt = cfg.paths.subtitles_gt / f"{video_id}.srt"
    if gt_vtt.exists():
        cmd.append(f"--subtitles-corrected={gt_vtt.resolve()}")
    elif gt_srt.exists():
        cmd.append(f"--subtitles-corrected={gt_srt.resolve()}")

    return cmd


def segment_one(video_id: str, cfg: AppConfig) -> Path:
    out_dir = segmentation_subdir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_eaf = out_dir / f"{video_id}.eaf"

    if out_eaf.exists() and not cfg.segmentation.overwrite:
        return out_eaf

    subprocess.run(build_segment_command(video_id, cfg), check=True)

    if not out_eaf.exists():
        raise RuntimeError(
            f"Segmentation command finished but EAF is missing: {out_eaf}"
        )
    return out_eaf


def segment_all(cfg: AppConfig) -> None:
    ids = read_video_ids(cfg.paths.video_ids)
    for video_id in ids:
        print(f"[segment] {video_id}")
        segment_one(video_id, cfg)
