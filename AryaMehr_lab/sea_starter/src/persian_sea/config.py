from dataclasses import dataclass
from pathlib import Path
from typing import List
import yaml


@dataclass(frozen=True)
class PathsConfig:
    videos: Path
    poses: Path
    subtitles_raw: Path
    subtitles_gt: Path
    segmentation_gt: Path
    video_ids: Path
    segmentation_output: Path
    reports: Path


@dataclass(frozen=True)
class PoseConfig:
    format: str = "mediapipe"
    model_complexity: int = 2
    smooth_landmarks: bool = False
    refine_face_landmarks: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ElanConfig:
    sign_tiers: List[str]
    gt_sign_tiers: List[str]


@dataclass(frozen=True)
class SegmentationConfig:
    model: str = "model_E4s-1.pth"
    sign_b_threshold: int = 30
    sign_o_threshold: int = 50
    overwrite: bool = False


@dataclass(frozen=True)
class SegmentationEvalConfig:
    iou_thresholds: List[float]
    boundary_tolerance_ms: List[int]
    tune_b_values: List[int]
    tune_o_values: List[int]
    primary_iou_threshold: float = 0.50


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    pose: PoseConfig
    elan: ElanConfig
    segmentation: SegmentationConfig
    segmentation_eval: SegmentationEvalConfig


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        paths=PathsConfig(**{k: Path(v) for k, v in raw["paths"].items()}),
        pose=PoseConfig(**raw.get("pose", {})),
        elan=ElanConfig(**raw["elan"]),
        segmentation=SegmentationConfig(**raw["segmentation"]),
        segmentation_eval=SegmentationEvalConfig(**raw["segmentation_eval"]),
    )


def read_video_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def model_tag(model: str) -> str:
    stem = Path(model).stem
    return stem[len("model_"):] if stem.startswith("model_") else stem


def segmentation_run_name(model: str, b: int, o: int) -> str:
    return f"{model_tag(model)}_{b}_{o}"


def segmentation_subdir(cfg: AppConfig) -> Path:
    s = cfg.segmentation
    return cfg.paths.segmentation_output / segmentation_run_name(
        s.model,
        s.sign_b_threshold,
        s.sign_o_threshold,
    )
