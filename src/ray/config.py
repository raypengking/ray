"""Configuration models for the storyboard pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


def _ensure_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


@dataclass(slots=True)
class LoRAConfig:
    """Configuration for a single LoRA adapter."""

    path: Path
    scale: float = 0.8

    @classmethod
    def from_input(cls, value: str | Path, scale: float | None = None) -> "LoRAConfig":
        return cls(path=_ensure_path(value), scale=0.8 if scale is None else float(scale))


@dataclass(slots=True)
class EmbeddingConfig:
    """Configuration for an embedding file."""

    path: Path

    @classmethod
    def from_input(cls, value: str | Path) -> "EmbeddingConfig":
        return cls(path=_ensure_path(value))


@dataclass(slots=True)
class LLMConfig:
    """Configuration for the LLM that produces storyboard descriptions."""

    model: str = "qwen2.5"
    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    api_key: Optional[str] = None
    temperature: float = 0.35
    max_tokens: int = 1800
    top_p: float = 0.9


@dataclass(slots=True)
class PromptConfig:
    """Prompt level configuration."""

    language: str = "zh"
    style_keywords: tuple[str, ...] = ("电影质感", "写实光影")
    keep_original_dialogue: bool = True
    shots_per_chunk: int = 2


@dataclass(slots=True)
class NanoBananaConfig:
    """Configuration for Nano Banana image generation."""

    endpoint: str = "http://127.0.0.1:3921/api/generate"
    negative_prompt: str = ""
    guidance_scale: float = 7.5
    steps: int = 28
    width: int = 832
    height: int = 1216
    seed: int = 20240613
    sampler: str = "dpmpp_2m"
    output_format: str = "png"
    timeout: int = 300
    loras: List[LoRAConfig] = field(default_factory=list)
    embeddings: List[EmbeddingConfig] = field(default_factory=list)
    enabled: bool = True

    def with_loras(self, items: Iterable[LoRAConfig]) -> "NanoBananaConfig":
        self.loras.extend(items)
        return self

    def with_embeddings(self, items: Iterable[EmbeddingConfig]) -> "NanoBananaConfig":
        self.embeddings.extend(items)
        return self


@dataclass(slots=True)
class PipelineConfig:
    """General pipeline configuration."""

    chunk_size: int = 260
    chunk_overlap: int = 20
    base_seed: int = 20240613
    output_dir: Path = Path("outputs")
    html_file_name: str = "index.html"
    json_file_name: str = "storyboard.json"
    save_json: bool = True


__all__ = [
    "LLMConfig",
    "PromptConfig",
    "NanoBananaConfig",
    "PipelineConfig",
    "LoRAConfig",
    "EmbeddingConfig",
]
