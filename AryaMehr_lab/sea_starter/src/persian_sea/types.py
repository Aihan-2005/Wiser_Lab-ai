from dataclasses import dataclass


@dataclass(frozen=True)
class SignSegment:

    start: float

    end: float

    text: str = ""


    @property
    def mid(self) -> float:

        return (
            self.start
            +
            self.end
        ) / 2.0


    @property
    def duration(self) -> float:

        return (
            self.end
            -
            self.start
        )


    def validate(self) -> None:

        if self.start < 0:

            raise ValueError(
                f"Negative start: "
                f"{self.start}"
            )

        if self.end <= self.start:

            raise ValueError(
                f"Invalid segment: "
                f"{self.start} -> {self.end}"
            )