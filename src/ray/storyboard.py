"""Core dataclasses representing the storyboard output."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass(slots=True)
class Shot:
    """A single storyboard shot."""

    index: int
    title: str
    shot_type: str
    summary: str
    visuals: str
    dialogue: Optional[str] = None
    characters: List[str] = field(default_factory=list)
    image_prompt: str = ""
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    duration: Optional[str] = None
    notes: Optional[str] = None
    image_path: Optional[Path] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.image_path is not None:
            data["image_path"] = str(self.image_path)
        return data


@dataclass(slots=True)
class Storyboard:
    """Storyboard composed of multiple shots."""

    source_text: str
    shots: List[Shot]

    def to_dict(self) -> dict:
        return {
            "source_text": self.source_text,
            "shots": [shot.to_dict() for shot in self.shots],
        }


__all__ = ["Shot", "Storyboard"]
