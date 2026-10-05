from dataclasses import dataclass
from typing import Optional


@dataclass
class Cue:
    start: float
    end: float
    text: str = ""

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def validate(self) -> None:
        if self.start < 0:
            raise ValueError(f"Negative cue start: {self.start}")
        if self.end < self.start:
            raise ValueError(f"Cue end < start: {self.start} -> {self.end}")


@dataclass
class SignSegment:
    start: float
    end: float
    text: str = ""

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def validate(self) -> None:
        if self.start < 0:
            raise ValueError(f"Negative sign start: {self.start}")
        if self.end <= self.start:
            raise ValueError(f"Invalid sign segment: {self.start} -> {self.end}")


@dataclass
class Alignment:
    cue_index: int
    start: float
    end: float
    first_sign_index: Optional[int] = None
    last_sign_index: Optional[int] = None
