"""Command line interface for the storyboard pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer

from .config import (
    EmbeddingConfig,
    LLMConfig,
    LoRAConfig,
    NanoBananaConfig,
    PipelineConfig,
    PromptConfig,
)
from .html_exporter import StoryboardHTMLExporter
from .image_generation import NanoBananaClient
from .llm import LLMClient, MockLLMClient, QwenLLMClient
from .storyboard_generator import StoryboardGenerator

app = typer.Typer(help="将中文文章转换为分镜脚本与 Nano Banana 图像。")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def _parse_lora_options(values: List[str]) -> List[LoRAConfig]:
    result: List[LoRAConfig] = []
    for value in values:
        if ":" in value:
            path, scale = value.split(":", 1)
            result.append(LoRAConfig.from_input(path, float(scale)))
        else:
            result.append(LoRAConfig.from_input(value))
    return result


def _parse_embedding_options(values: List[str]) -> List[EmbeddingConfig]:
    return [EmbeddingConfig.from_input(value) for value in values]


@app.command()
def run(
    input_path: Path = typer.Argument(..., exists=True, help="输入的中文文本文件"),
    output_dir: Path = typer.Option(Path("outputs"), "--output", "-o", help="输出目录"),
    chunk_size: int = typer.Option(260, help="文本分段最大长度"),
    chunk_overlap: int = typer.Option(20, help="分段重叠字数"),
    shots_per_chunk: int = typer.Option(2, help="每段生成的镜头数量参考"),
    keep_dialogue: bool = typer.Option(True, help="是否保留原文对白"),
    style_keyword: Optional[List[str]] = typer.Option(
        None,
        "--style",
        help="分镜整体风格关键词，可多次传入",
    ),
    llm_endpoint: str = typer.Option(
        "http://127.0.0.1:8000/v1/chat/completions",
        help="Qwen2.5 接口地址 (OpenAI 兼容)",
    ),
    llm_model: str = typer.Option("qwen2.5", help="使用的模型名称"),
    llm_temperature: float = typer.Option(0.35, help="LLM 温度"),
    llm_max_tokens: int = typer.Option(1800, help="LLM 最大输出 token"),
    llm_top_p: float = typer.Option(0.9, help="LLM top_p"),
    llm_api_key: Optional[str] = typer.Option(None, envvar="QWEN_API_KEY", help="Qwen API Key"),
    nano_endpoint: str = typer.Option("http://127.0.0.1:3921/api/generate", help="Nano Banana 接口地址"),
    nano_negative: str = typer.Option("", help="全局负面提示词"),
    nano_guidance: float = typer.Option(7.5, help="CFG Scale"),
    nano_steps: int = typer.Option(28, help="采样步数"),
    nano_sampler: str = typer.Option("dpmpp_2m", help="采样器"),
    nano_width: int = typer.Option(832, help="图像宽度"),
    nano_height: int = typer.Option(1216, help="图像高度"),
    base_seed: int = typer.Option(20240613, help="基础随机种子"),
    lora: List[str] = typer.Option([], help="LoRA 配置，格式为 路径[:权重]"),
    embedding: List[str] = typer.Option([], help="Embedding 路径，可多次传入"),
    dry_run: bool = typer.Option(False, help="仅生成脚本和占位图，不调用外部服务"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="输出调试日志"),
) -> None:
    """将输入小说转换为分镜脚本、图像和 HTML。"""

    configure_logging(verbose)

    text = input_path.read_text(encoding="utf-8")
    typer.echo(f"读取文本 {input_path}，长度 {len(text)} 字")

    prompt_config = PromptConfig(
        keep_original_dialogue=keep_dialogue,
        shots_per_chunk=shots_per_chunk,
        style_keywords=tuple(style_keyword) if style_keyword else PromptConfig().style_keywords,
    )
    pipeline_config = PipelineConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        base_seed=base_seed,
        output_dir=output_dir,
    )

    llm_config = LLMConfig(
        model=llm_model,
        endpoint=llm_endpoint,
        api_key=llm_api_key,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        top_p=llm_top_p,
    )

    llm_client: LLMClient
    if dry_run:
        llm_client = MockLLMClient(shots_per_chunk=shots_per_chunk)
    else:
        llm_client = QwenLLMClient(llm_config, api_key=llm_api_key)

    generator = StoryboardGenerator(llm_client, prompt_config, pipeline_config)
    storyboard = generator.generate(text)
    typer.echo(f"已生成 {len(storyboard.shots)} 个镜头")

    nano_config = NanoBananaConfig(
        endpoint=nano_endpoint,
        negative_prompt=nano_negative,
        guidance_scale=nano_guidance,
        steps=nano_steps,
        width=nano_width,
        height=nano_height,
        seed=base_seed,
        sampler=nano_sampler,
        enabled=not dry_run,
    )
    if lora:
        nano_config.with_loras(_parse_lora_options(lora))
    if embedding:
        nano_config.with_embeddings(_parse_embedding_options(embedding))

    image_client = NanoBananaClient(nano_config, pipeline_config, dry_run=dry_run)
    image_dir = output_dir / "images"
    for shot in storyboard.shots:
        actual_path = image_client.generate_image(shot, image_dir)
        # 在 HTML 中使用相对路径
        shot.image_path = Path("images") / actual_path.name

    exporter = StoryboardHTMLExporter()
    html_path = exporter.export(storyboard, output_dir, pipeline_config.html_file_name)
    typer.echo(f"预览文件已输出：{html_path}")
    typer.echo(f"分镜数据 JSON：{output_dir / 'storyboard.json'}")


if __name__ == "__main__":  # pragma: no cover
    app()
