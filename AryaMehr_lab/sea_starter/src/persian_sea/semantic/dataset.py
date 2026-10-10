from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from persian_sea.config import AppConfig, read_video_ids
from persian_sea.features.cache import FeatureCache
from persian_sea.io.subtitles import (
    SubtitleCue,
    find_subtitle,
    pair_corrected_cues,
    read_subtitles,
)
from persian_sea.semantic.candidates import select_candidate_indices
from persian_sea.text.cache import TextFeatureCache


@dataclass(frozen=True)
class SampleRef:
    video_id: str
    cue_index: int
    rough: SubtitleCue
    corrected: SubtitleCue | None


class _VideoCacheLRU:
    def __init__(self, max_items: int = 2):
        self.max_items = max_items
        self._visual: OrderedDict[str, FeatureCache] = OrderedDict()
        self._text: OrderedDict[str, TextFeatureCache] = OrderedDict()

    def _touch(self, cache: OrderedDict, key: str, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.max_items:
            cache.popitem(last=False)
        return value

    def visual(self, path: Path, video_id: str) -> FeatureCache:
        if video_id in self._visual:
            value = self._visual[video_id]
            self._visual.move_to_end(video_id)
            return value
        return self._touch(
            self._visual,
            video_id,
            FeatureCache(path),
        )

    def text(self, path: Path, video_id: str) -> TextFeatureCache:
        if video_id in self._text:
            value = self._text[video_id]
            self._text.move_to_end(video_id)
            return value
        return self._touch(
            self._text,
            video_id,
            TextFeatureCache(path),
        )


class SubtitleAlignmentDataset(Dataset):
    def __init__(
        self,
        cfg: AppConfig,
        video_ids_path: Path,
        memory_cache_videos: int = 2,
    ):
        self.cfg = cfg
        self.refs: list[SampleRef] = []
        self.cache = _VideoCacheLR(memory_cache_videos)

        for video_id in read_video_ids(video_ids_path):
            raw_path = find_subtitle(cfg.paths.subtitles_raw, video_id)
            if raw_path is None:
                print(f"[dataset] skip {video_id}: raw subtitles missing")
                continue

            visual_path = cfg.paths.feature_cache / f"{video_id}.npz"
            text_path = cfg.paths.text_feature_cache / f"{video_id}.npz"

            if not visual_path.exists() or not text_path.exists():
                print(f"[dataset] skip {video_id}: feature cache missing")
                continue

            rough = read_subtitles(raw_path)

            gt_path = find_subtitle(cfg.paths.subtitles_gt, video_id)
            corrected = (
                read_subtitles(gt_path)
                if gt_path is not None
                else None
            )
            paired_gt = pair_corrected_cues(rough, corrected)

            text_cache = TextFeatureCache(text_path)
            if len(text_cache.embeddings) != len(rough):
                raise ValueError(
                    f"Text cache/subtitle count mismatch for {video_id}: "
                    f"{len(text_cache.embeddings)} != {len(rough)}"
                )

            for cue_index, cue in enumerate(rough):
                self.refs.append(
                    SampleRef(
                        video_id=video_id,
                        cue_index=cue_index,
                        rough=cue,
                        corrected=paired_gt[cue_index],
                    )
                )

        if not self.refs:
            raise RuntimeError("No usable subtitle-alignment samples.")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict:
        ref = self.refs[index]
        cfg = self.cfg

        visual = self.cache.visual(
            cfg.paths.feature_cache / f"{ref.video_id}.npz",
            ref.video_id,
        )
        text = self.cache.text(
            cfg.paths.text_feature_cache / f"{ref.video_id}.npz",
            ref.video_id,
        )

        candidate_idx = select_candidate_indices(
            visual.starts,
            visual.ends,
            rough_start=ref.rough.start,
            rough_end=ref.rough.end,
            radius=cfg.semantic.search_radius_seconds,
            min_candidates=cfg.semantic.min_candidate_segments,
            max_candidates=cfg.semantic.max_candidate_segments,
        )

        corrected = ref.corrected

        return {
            "video_id": ref.video_id,
            "cue_index": ref.cue_index,
            "text": ref.rough.text,

            "rough_start": np.float32(ref.rough.start),
            "rough_end": np.float32(ref.rough.end),

            "gt_start": np.float32(
                corrected.start if corrected is not None else np.nan
            ),
            "gt_end": np.float32(
                corrected.end if corrected is not None else np.nan
            ),

            "text_embedding": torch.from_numpy(
                text.embeddings[ref.cue_index].astype(np.float32)
            ),

            "pose": torch.from_numpy(
                visual.pose[candidate_idx].astype(np.float32)
            ),
            "rgb": torch.from_numpy(
                visual.rgb[candidate_idx].astype(np.float32)
            ),

            "segment_starts": torch.from_numpy(
                visual.starts[candidate_idx].astype(np.float32)
            ),
            "segment_ends": torch.from_numpy(
                visual.ends[candidate_idx].astype(np.float32)
            ),

            "global_segment_indices": torch.from_numpy(
                candidate_idx.astype(np.int64)
            ),
        }


def collate_alignment_batch(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("Empty batch.")

    B = len(batch)
    max_m = max(item["pose"].shape[0] for item in batch)

    pose_shape = batch[0]["pose"].shape[1:]
    rgb_shape = batch[0]["rgb"].shape[1:]

    pose = torch.zeros((B, max_m, *pose_shape), dtype=torch.float32)
    rgb = torch.zeros((B, max_m, *rgb_shape), dtype=torch.float32)

    starts = torch.zeros((B, max_m), dtype=torch.float32)
    ends = torch.zeros((B, max_m), dtype=torch.float32)
    global_idx = torch.full((B, max_m), -1, dtype=torch.long)

    mask = torch.zeros((B, max_m), dtype=torch.bool)

    for i, item in enumerate(batch):
        m = item["pose"].shape[0]
        pose[i, :m] = item["pose"]
        rgb[i, :m] = item["rgb"]
        starts[i, :m] = item["segment_starts"]
        ends[i, :m] = item["segment_ends"]
        global_idx[i, :m] = item["global_segment_indices"]
        mask[i, :m] = True

    return {
        "video_id": [item["video_id"] for item in batch],
        "cue_index": torch.tensor(
            [item["cue_index"] for item in batch],
            dtype=torch.long,
        ),
        "text": [item["text"] for item in batch],

        "rough_start": torch.tensor(
            [item["rough_start"] for item in batch],
            dtype=torch.float32,
        ),
        "rough_end": torch.tensor(
            [item["rough_end"] for item in batch],
            dtype=torch.float32,
        ),

        "gt_start": torch.tensor(
            [item["gt_start"] for item in batch],
            dtype=torch.float32,
        ),
        "gt_end": torch.tensor(
            [item["gt_end"] for item in batch],
            dtype=torch.float32,
        ),

        "text_embedding": torch.stack(
            [item["text_embedding"] for item in batch],
            dim=0,
        ),

        "pose": pose,
        "rgb": rgb,
        "candidate_mask": mask,
        "segment_starts": starts,
        "segment_ends": ends,
        "global_segment_indices": global_idx,
    }
