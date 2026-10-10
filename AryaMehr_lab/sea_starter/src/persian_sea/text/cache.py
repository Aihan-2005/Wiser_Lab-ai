from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from persian_sea.config import AppConfig, read_video_ids
from persian_sea.io.subtitles import find_subtitle, read_subtitles
from persian_sea.text.bge_m3 import BGEM3DenseExtractor


class TextFeatureCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=False)

    @property
    def embeddings(self) -> np.ndarray:
        return self.data["embeddings"]

    @property
    def starts(self) -> np.ndarray:
        return self.data["starts"]

    @property
    def ends(self) -> np.ndarray:
        return self.data["ends"]

    @property
    def texts(self) -> list[str]:
        return [str(x) for x in self.data["texts"].tolist()]


def scan_text_video(
    video_id: str,
    cfg: AppConfig,
    extractor: BGEM3DenseExtractor,
) -> Path:
    out_dir = cfg.paths.text_feature_cache
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{video_id}.npz"
    out_json = out_dir / f"{video_id}.json"

    if out_npz.exists() and not cfg.semantic.overwrite_text_cache:
        print(f"[text] skip existing: {out_npz}")
        return out_npz

    subtitle_path = find_subtitle(cfg.paths.subtitles_raw, video_id)
    if subtitle_path is None:
        raise FileNotFoundError(
            f"No subtitle for {video_id} in {cfg.paths.subtitles_raw}"
        )

    cues = read_subtitles(subtitle_path)
    texts = [c.text for c in cues]

    embeddings = extractor.encode(
        texts,
        batch_size=cfg.semantic.text_batch_size,
    )

    dtype = (
        np.float16
        if cfg.semantic.text_cache_dtype == "float16"
        else np.float32
    )

    np.savez_compressed(
        out_npz,
        embeddings=embeddings.astype(dtype),
        starts=np.asarray([c.start for c in cues], dtype=np.float32),
        ends=np.asarray([c.end for c in cues], dtype=np.float32),
        texts=np.asarray(texts),
        hidden=np.asarray([extractor.hidden_size], dtype=np.int32),
    )

    out_json.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "subtitle": str(subtitle_path),
                "text_model": cfg.semantic.text_model,
                "num_cues": len(cues),
                "hidden_size": extractor.hidden_size,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[text] {video_id}: {len(cues)} cues, "
        f"embeddings={embeddings.shape}"
    )
    return out_npz


def scan_text_all(cfg: AppConfig) -> None:
    extractor = BGEM3DenseExtractor(
        model_name=cfg.semantic.text_model,
        device=cfg.semantic.text_device,
        amp=cfg.semantic.text_amp,
        max_length=cfg.semantic.text_max_length,
    )

    for video_id in read_video_ids(cfg.paths.video_ids):
        scan_text_video(video_id, cfg, extractor)
