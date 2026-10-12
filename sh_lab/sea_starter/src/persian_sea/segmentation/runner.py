from pathlib import Path
import subprocess

from persian_sea.config import AppConfig, read_video_ids, segmentation_subdir


def build_segment_command(video_id: str, cfg: AppConfig) -> list[str]:
    pose = cfg.paths.poses / f"{video_id}.pose"
    if not pose.exists():
        raise FileNotFoundError(pose)

    output_dir = segmentation_subdir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    eaf = output_dir / f"{video_id}.eaf"

    s = cfg.segmentation
    cmd = [
        "pose_to_segments",
        "--no-pose-link",
        f"--model={s.model}",
        f"--pose={pose}",
        f"--elan={eaf}",
        "--sign-b-threshold",
        str(s.sign_b_threshold),
        "--sign-o-threshold",
        str(s.sign_o_threshold),
    ]

    video = cfg.paths.videos / f"{video_id}.mp4"
    if video.exists():
        cmd.append(f"--video={video.resolve()}")

    for ext in (".vtt", ".srt"):
        subtitle = cfg.paths.subtitles_raw / f"{video_id}{ext}"
        if subtitle.exists():
            cmd.append(f"--subtitles={subtitle.resolve()}")
            break

    for ext in (".vtt", ".srt"):
        subtitle = cfg.paths.subtitles_gt / f"{video_id}{ext}"
        if subtitle.exists():
            cmd.append(f"--subtitles-corrected={subtitle.resolve()}")
            break

    return cmd


def segment_one(video_id: str, cfg: AppConfig) -> Path:
    output_dir = segmentation_subdir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    eaf = output_dir / f"{video_id}.eaf"

    if eaf.exists() and not cfg.segmentation.overwrite:
        print(f"[segment] skip existing: {eaf}")
        return eaf

    cmd = build_segment_command(video_id, cfg)
    print("[segment]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not eaf.exists():
        raise RuntimeError(f"Expected EAF was not created: {eaf}")
    return eaf


def segment_all(cfg: AppConfig) -> None:
    for video_id in read_video_ids(cfg.paths.video_ids):
        segment_one(video_id, cfg)
