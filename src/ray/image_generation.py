"""Nano Banana integration for storyboard image rendering."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

try:  # pragma: no cover - optional dependency
    import requests
except ImportError:  # pragma: no cover - handled at runtime
    requests = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - handled at runtime
    Image = ImageDraw = ImageFont = None  # type: ignore

from .config import NanoBananaConfig, PipelineConfig
from .storyboard import Shot

logger = logging.getLogger(__name__)

_BLANK_PNG_BYTES = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO1hBzgAAAAASUVORK5CYII="
)


class NanoBananaError(RuntimeError):
    """Raised when the Nano Banana service fails."""


class NanoBananaClient:
    """Wrapper around the Nano Banana HTTP API."""

    def __init__(
        self,
        config: NanoBananaConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self.config = config or NanoBananaConfig()
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.dry_run = dry_run or not self.config.enabled

    def _build_payload(self, shot: Shot, *, seed: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": shot.image_prompt,
            "negative_prompt": shot.negative_prompt or self.config.negative_prompt,
            "seed": seed,
            "steps": self.config.steps,
            "cfg_scale": self.config.guidance_scale,
            "sampler": self.config.sampler,
            "width": self.config.width,
            "height": self.config.height,
            "format": self.config.output_format,
        }
        if self.config.loras:
            payload["loras"] = [
                {"path": str(item.path), "scale": item.scale} for item in self.config.loras
            ]
        if self.config.embeddings:
            payload["embeddings"] = [str(item.path) for item in self.config.embeddings]
        return payload

    def _call_service(self, payload: Dict[str, Any]) -> bytes:
        if requests is None:
            raise NanoBananaError("requests package is required for Nano Banana integration")

        try:
            response = requests.post(self.config.endpoint, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - requires real service
            raise NanoBananaError(f"Nano Banana request failed: {exc}") from exc

        data: Dict[str, Any] = response.json()
        images = data.get("images") or data.get("data")
        if not images:
            raise NanoBananaError(f"Unexpected Nano Banana response: {data}")
        first = images[0]
        if isinstance(first, str):
            return base64.b64decode(first)
        if isinstance(first, dict) and "data" in first:
            return base64.b64decode(first["data"])
        raise NanoBananaError(f"Unsupported Nano Banana image payload: {first}")

    def _placeholder_image(self, path: Path, shot: Shot, *, seed: int) -> None:
        logger.info("Generating placeholder image for shot %s", shot.index)
        if Image is None:
            path.write_bytes(_BLANK_PNG_BYTES)
            return

        width, height = self.config.width, self.config.height
        image = Image.new("RGB", (width, height), color=(32, 32, 40))
        draw = ImageDraw.Draw(image)
        title = f"Shot {shot.index}: {shot.title}"
        prompt_lines = [shot.image_prompt[i : i + 32] for i in range(0, len(shot.image_prompt), 32)]
        text = "\n".join([title, "Seed: " + str(seed), ""] + prompt_lines[:6])
        font = None
        if ImageFont is not None:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 28)
            except OSError:  # pragma: no cover - fonts differ per env
                font = ImageFont.load_default()
        draw.multiline_text((40, 40), text, fill=(220, 220, 230), font=font, spacing=10)
        image.save(path)

    def generate_image(self, shot: Shot, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = shot.seed if shot.seed is not None else self.pipeline_config.base_seed + shot.index
        file_name = f"shot_{shot.index:03d}.{self.config.output_format}"
        output_path = output_dir / file_name

        if self.dry_run or not shot.image_prompt:
            self._placeholder_image(output_path, shot, seed=seed)
            shot.seed = seed
            shot.image_path = output_path
            return output_path

        payload = self._build_payload(shot, seed=seed)
        logger.debug("Nano Banana payload for shot %s: %s", shot.index, payload)

        try:
            image_bytes = self._call_service(payload)
            output_path.write_bytes(image_bytes)
        except NanoBananaError as exc:
            logger.warning("%s，生成占位图。", exc)
            self._placeholder_image(output_path, shot, seed=seed)

        shot.seed = seed
        shot.image_path = output_path
        return output_path

    def generate_images(self, storyboard: Iterable[Shot], output_dir: Path) -> None:
        for shot in storyboard:
            self.generate_image(shot, output_dir)


__all__ = ["NanoBananaClient", "NanoBananaError"]
