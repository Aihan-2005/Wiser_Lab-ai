from dataclasses import dataclass
from pathlib import Path
from typing import List
import yaml


@dataclass
class PathsConfig:
    videos: Path
    poses: Path
    subtitles_raw: Path
    subtitles_gt: Path
    video_ids: Path
    segmentation_output: Path
    alignment_output: Path


@dataclass
class ElanConfig:
    sign_tiers: List[str]
    aligned_subtitle_tier: str = "SUBTITLE_SHIFTED"


@dataclass
class SegmentationConfig:
    model: str
    sign_b_threshold: int = 30
    sign_o_threshold: int = 50
    num_workers: int = 1
    overwrite: bool = False


@dataclass
class AlignmentConfig:
    duration_penalty_weight: float = 1.0
    gap_penalty_weight: float = 5.0
    max_gap_seconds: float = 10.0
    candidate_window_size: int = 50
    pre_shift_start_seconds: float = 0.0
    pre_shift_end_seconds: float = 0.0
    post_shift_start_seconds: float = 0.0
    post_shift_end_seconds: float = 0.0
    similarity_weight: float = 0.0
    write_updated_eaf: bool = True
    overwrite: bool = False


@dataclass
class AppConfig:
    paths: PathsConfig
    elan: ElanConfig
    segmentation: SegmentationConfig
    alignment: AlignmentConfig


def _pathify(d: dict) -> PathsConfig:
    return PathsConfig(**{k: Path(v) for k, v in d.items()})


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        paths=_pathify(raw["paths"]),
        elan=ElanConfig(**raw.get("elan", {})),
        segmentation=SegmentationConfig(**raw["segmentation"]),
        alignment=AlignmentConfig(**raw["alignment"]),
    )


def model_tag(model: str) -> str:
    stem = Path(model).stem
    if stem.startswith("model_"):
        stem = stem[len("model_"):]
    return stem


def segmentation_subdir(cfg: AppConfig) -> Path:
    s = cfg.segmentation
    return cfg.paths.segmentation_output / (
        f"{model_tag(s.model)}_{s.sign_b_threshold}_{s.sign_o_threshold}"
    )
