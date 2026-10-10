from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata



@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return max(1e-6, self.end - self.start)


_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
})


def normalize_persian_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ARABIC_TO_PERSIAN)


    text = text.replace("\u200c", " ")


    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _vtt_time_to_seconds(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        h = 0
        m, s = parts
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError(f"Invalid timestamp: {value}")
    return int(h) * 3600 + int(m) * 60 + float(s)


def find_subtitle(directory: Path, video_id: str) -> Path | None:
    for ext in (".vtt", ".srt"):
        candidate = directory / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_subtitles(path: str | Path) -> list[SubtitleCue]:
    path = Path(path)
    cues: list[SubtitleCue] = []

    if path.suffix.lower() == ".vtt":
        try:
            import webvtt
        except ImportError as exc:
            raise RuntimeError("Install webvtt-py to read .vtt files.") from exc
        for item in webvtt.read(str(path)):
            cues.append(
                SubtitleCue(
                    start=_vtt_time_to_seconds(item.start),
                    end=_vtt_time_to_seconds(item.end),
                    text=normalize_persian_text(item.text),
                )
            )
    elif path.suffix.lower() == ".srt":
        try:
            import pysrt
        except ImportError as exc:
            raise RuntimeError("Install pysrt to read .srt files.") from exc
        for item in pysrt.open(str(path), encoding="utf-8"):
            cues.append(
                SubtitleCue(
                    start=item.start.ordinal / 1000.0,
                    end=item.end.ordinal / 1000.0,
                    text=normalize_persian_text(
                        " ".join(item.text.splitlines())
                    ),
                )
            )
    else:
        raise ValueError(f"Unsupported subtitle file: {path}")

    cues.sort(key=lambda c: (c.start, c.end))
    return cues


def pair_corrected_cues(
    rough: list[SubtitleCue],
    corrected: list[SubtitleCue] | None,
) -> list[SubtitleCue | None]:



    if not corrected:
        return [None] * len(rough)

    if len(rough) == len(corrected):
        return list(corrected)

    by_text: dict[str, list[SubtitleCue]] = {}
    for cue in corrected:
        by_text.setdefault(cue.text, []).append(cue)

    result: list[SubtitleCue | None] = []
    for cue in rough:
        matches = by_text.get(cue.text, [])
        result.append(matches[0] if len(matches) == 1 else None)
    return result


def seconds_to_vtt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_vtt(cues: list[SubtitleCue], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.append(
            f"{seconds_to_vtt(cue.start)} --> "
            f"{seconds_to_vtt(cue.end)}"
        )
        lines.append(cue.text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
