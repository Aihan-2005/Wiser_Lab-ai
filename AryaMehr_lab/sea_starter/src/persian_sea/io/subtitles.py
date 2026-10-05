from pathlib import Path
from typing import Iterable
import pysrt
import webvtt

from persian_sea.types import Cue


def _ts_to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_timestamp(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def read_subtitles(path: str | Path) -> list[Cue]:
    path = Path(path)
    ext = path.suffix.lower()
    cues: list[Cue] = []

    if ext == ".srt":
        subs = pysrt.open(str(path), encoding="utf-8")
        for sub in subs:
            cue = Cue(
                start=sub.start.ordinal / 1000.0,
                end=sub.end.ordinal / 1000.0,
                text=" ".join(sub.text.splitlines()).strip(),
            )
            cue.validate()
            cues.append(cue)
    elif ext == ".vtt":
        for item in webvtt.read(str(path)):
            cue = Cue(
                start=_ts_to_seconds(item.start),
                end=_ts_to_seconds(item.end),
                text=item.text.strip(),
            )
            cue.validate()
            cues.append(cue)
    else:
        raise ValueError(f"Unsupported subtitle extension: {ext}")

    validate_cue_sequence(cues)
    return cues


def validate_cue_sequence(cues: Iterable[Cue]) -> None:
    last_start = -1.0
    for i, cue in enumerate(cues):
        cue.validate()
        if cue.start < last_start:
            raise ValueError(
                f"Subtitle cues are not sorted at index {i}: "
                f"{cue.start} < {last_start}"
            )
        last_start = cue.start


def write_vtt(cues: Iterable[Cue], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for cue in cues:
        cue.validate()
        lines.append(
            f"{seconds_to_timestamp(cue.start)} --> "
            f"{seconds_to_timestamp(cue.end)}"
        )
        lines.append(cue.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def find_subtitle(directory: str | Path, video_id: str) -> Path | None:
    directory = Path(directory)
    for ext in (".vtt", ".srt"):
        p = directory / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def shifted(cues: list[Cue], delta_start: float, delta_end: float) -> list[Cue]:
    out: list[Cue] = []
    for cue in cues:
        start = max(0.0, cue.start + delta_start)
        end = max(start, cue.end + delta_end)
        out.append(Cue(start=start, end=end, text=cue.text))
    return out
