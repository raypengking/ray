"""Logic that coordinates prompt construction and LLM parsing."""

from __future__ import annotations

import json
import logging
from typing import List

from .config import PipelineConfig, PromptConfig
from .llm import LLMClient, LLMError, extract_first_json_block
from .storyboard import Shot, Storyboard
from .text_processing import split_text

logger = logging.getLogger(__name__)


class StoryboardGenerator:
    """Generate storyboards from text using an LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_config: PromptConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_config = prompt_config or PromptConfig()
        self.pipeline_config = pipeline_config or PipelineConfig()

    def _build_prompt(self, chunk: str, chunk_index: int) -> str:
        keywords = "、".join(self.prompt_config.style_keywords)
        instructions = [
            "请根据以下小说片段生成分镜脚本。",
            "对每个镜头输出 JSON 数组 shots，字段包含：title、shot_type、summary、visuals、dialogue、characters、image_prompt、notes、seed（可选）、negative_prompt（可选）、duration（可选）。",
            "镜头描述需具体到光线、场景、构图，并保留重要的角色信息。",
            f"若文本包含对话，dialogue 字段应保留原文（{ '保留' if self.prompt_config.keep_original_dialogue else '可改写' }）。",
            "image_prompt 字段使用英文描述画面，方便传入 Nano Banana。",
            f"结合关键词：{keywords}。",
            f"每个片段建议生成 {self.prompt_config.shots_per_chunk} 个镜头，可根据内容增减。",
        ]
        instruction_text = "\n".join(f"- {line}" for line in instructions)
        return (
            f"你正在处理第 {chunk_index} 段文本。\n"
            f"{instruction_text}\n\n"
            f"小说内容：\n{chunk}\n"
            "请仅输出 JSON。"
        )

    def _parse_shots(self, payload: dict, *, next_index: int) -> List[Shot]:
        shots_data = payload.get("shots")
        if not isinstance(shots_data, list):
            raise LLMError("LLM response缺少 shots 数组")
        shots: List[Shot] = []
        for item in shots_data:
            if not isinstance(item, dict):
                logger.warning("Invalid shot entry: %s", item)
                continue
            shot_index = next_index + len(shots)
            characters_value = item.get("characters", [])
            if isinstance(characters_value, (list, tuple)):
                characters = [str(char).strip() for char in characters_value if str(char).strip()]
            elif isinstance(characters_value, str):
                characters = [
                    token.strip()
                    for token in characters_value.replace("、", ",").split(",")
                    if token.strip()
                ]
            else:
                characters = []

            seed_value = item.get("seed")
            try:
                seed = int(seed_value) if seed_value is not None else None
            except (TypeError, ValueError):
                seed = None

            shot = Shot(
                index=shot_index,
                title=str(item.get("title", f"镜头 {shot_index}")),
                shot_type=str(item.get("shot_type", "中景")),
                summary=str(item.get("summary", "")),
                visuals=str(item.get("visuals", "")),
                dialogue=item.get("dialogue"),
                characters=characters,
                image_prompt=str(item.get("image_prompt", "")),
                negative_prompt=item.get("negative_prompt"),
                seed=seed,
                duration=item.get("duration"),
                notes=item.get("notes"),
            )
            shots.append(shot)
        return shots

    def generate(self, text: str) -> Storyboard:
        chunks = split_text(text, self.pipeline_config.chunk_size, self.pipeline_config.chunk_overlap)
        logger.info("Split input into %s chunks", len(chunks))
        shots: List[Shot] = []
        for idx, chunk in enumerate(chunks, start=1):
            prompt = self._build_prompt(chunk, idx)
            logger.debug("Prompt for chunk %s: %s", idx, prompt)
            response = self.llm_client.complete(prompt)
            logger.debug("Raw LLM response for chunk %s: %s", idx, response)
            payload = extract_first_json_block(response)
            new_shots = self._parse_shots(payload, next_index=len(shots) + 1)
            shots.extend(new_shots)
            logger.info("Chunk %s produced %s shots", idx, len(new_shots))
        return Storyboard(source_text=text, shots=shots)

    def to_json(self, storyboard: Storyboard) -> str:
        return json.dumps(storyboard.to_dict(), ensure_ascii=False, indent=2)


__all__ = ["StoryboardGenerator"]
