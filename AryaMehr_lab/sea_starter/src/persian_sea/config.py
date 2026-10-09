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
    alignment_output: Path
    reports: Path
    feature_cache: Path


@dataclass(frozen=True)
class PoseConfig:
    format: str = "mediapipe"
    num_workers: int = 1
    model_complexity: int = 2
    smooth_landmarks: bool = False
    refine_face_landmarks: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ElanConfig:
    sign_tiers: List[str]
    gt_sign_tiers: List[str]
    aligned_subtitle_tier: str = "SUBTITLE_SHIFTED"


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
class ScannerConfig:
    rgb_model: str = "facebook/dinov2-base"
    device: str = "auto"
    amp: bool = True
    rgb_batch_size: int = 32
    samples_per_segment: int = 8
    views: List[str] | None = None
    hand_padding: float = 2.0
    face_padding: float = 1.55
    signer_padding: float = 1.20
    local_fallback_to_signer: bool = True
    pose_include_confidence: bool = True
    pose_shoulder_epsilon: float = 0.02
    cache_dtype: str = "float16"
    overwrite: bool = False

    def resolved_views(self) -> list[str]:
        return self.views or ["signer", "left_hand", "right_hand", "face"]


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    pose: PoseConfig
    elan: ElanConfig
    segmentation: SegmentationConfig
    segmentation_eval: SegmentationEvalConfig
    scanner: ScannerConfig
    alignment: AlignmentConfig


def _pathify(raw: dict) -> PathsConfig:
    return PathsConfig(**{k: Path(v) for k, v in raw.items()})


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        paths=_pathify(raw["paths"]),
        pose=PoseConfig(**raw.get("pose", {})),
        elan=ElanConfig(**raw["elan"]),
        segmentation=SegmentationConfig(**raw["segmentation"]),
        segmentation_eval=SegmentationEvalConfig(**raw["segmentation_eval"]),
        scanner=ScannerConfig(**raw.get("scanner", {})),
        alignment=AlignmentConfig(**raw.get("alignment", {})),
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
