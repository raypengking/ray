"""LLM client abstractions for generating storyboard scripts."""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency
    import requests
except ImportError:  # pragma: no cover - handled at runtime
    requests = None  # type: ignore

from .config import LLMConfig

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


class LLMError(RuntimeError):
    """Raised when an LLM call fails."""


class LLMClient(ABC):
    """Abstract LLM client."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Run a completion request with the prompt."""


class QwenLLMClient(LLMClient):
    """Client targeting an OpenAI-compatible Qwen2.5 endpoint."""

    def __init__(self, config: Optional[LLMConfig] = None, *, api_key: Optional[str] = None):
        self.config = config or LLMConfig()
        self.api_key = api_key or self.config.api_key or os.getenv("QWEN_API_KEY")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "你是一名资深分镜师，擅长把小说转换为影视分镜脚本。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

        if requests is None:
            raise LLMError("requests package is required to call the Qwen endpoint")

        logger.debug("Sending prompt to Qwen endpoint %s", self.config.endpoint)
        try:
            response = requests.post(
                self.config.endpoint,
                json=payload,
                headers=self._headers(),
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network errors
            raise LLMError(f"Failed to call LLM endpoint: {exc}") from exc

        data: Dict[str, Any] = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - unexpected schema
            raise LLMError(f"Unexpected LLM response: {data}") from exc

        logger.debug("Received LLM response: %s", content)
        return content


class MockLLMClient(LLMClient):
    """Lightweight mock used for unit tests or dry runs."""

    def __init__(self, *, shots_per_chunk: int = 2):
        self.shots_per_chunk = shots_per_chunk

    def complete(self, prompt: str) -> str:  # pragma: no cover - deterministic output
        items = []
        for idx in range(self.shots_per_chunk):
            shot_index = idx + 1
            items.append(
                {
                    "title": f"示例镜头 {shot_index}",
                    "shot_type": "中景",
                    "summary": "主角与配角在雨中的街道对峙。",
                    "visuals": "湿漉漉的路面反射霓虹灯，远处有行人匆匆走过。",
                    "dialogue": "主角：你为什么背叛我们？",
                    "characters": ["主角", "配角"],
                    "image_prompt": "cinematic rain street confrontation, neon reflections, realistic lighting",
                    "notes": "保持人物服装一致，注意雨滴细节。",
                }
            )
        return json.dumps({"shots": items}, ensure_ascii=False)


def extract_first_json_block(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON block from an LLM response."""

    match = _JSON_BLOCK.search(text)
    if not match:
        raise LLMError("No JSON payload found in LLM response")
    block = match.group(0)
    try:
        return json.loads(block)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on LLM output
        raise LLMError(f"Failed to parse JSON from LLM response: {exc}\n{block}") from exc


__all__ = [
    "LLMClient",
    "QwenLLMClient",
    "MockLLMClient",
    "LLMError",
    "extract_first_json_block",
]
