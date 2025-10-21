import pytest

from ray.config import NanoBananaConfig, PipelineConfig, PromptConfig
from ray.image_generation import NanoBananaClient
from ray.llm import MockLLMClient
from ray.html_exporter import StoryboardHTMLExporter
from ray.storyboard_generator import StoryboardGenerator


TEXT = "在雨夜的街道上，主角推开昏黄的路灯向前走去。" * 2


def test_generator_with_mock_llm(tmp_path):
    generator = StoryboardGenerator(MockLLMClient(), PromptConfig(), PipelineConfig())
    storyboard = generator.generate(TEXT)
    assert storyboard.shots
    assert storyboard.shots[0].title.startswith("示例镜头")

    client = NanoBananaClient(NanoBananaConfig(enabled=False), PipelineConfig(), dry_run=True)
    image_dir = tmp_path / "images"
    for shot in storyboard.shots:
        path = client.generate_image(shot, image_dir)
        assert path.exists()
        assert path.suffix == ".png"

    exporter_output = tmp_path / "export"
    exporter_output.mkdir()
    try:
        exporter = StoryboardHTMLExporter()
    except RuntimeError:
        pytest.skip("jinja2 is not installed")
    else:
        html_path = exporter.export(storyboard, exporter_output)
        assert html_path.exists()
        json_path = exporter_output / "storyboard.json"
        assert json_path.exists()
        data = json_path.read_text(encoding="utf-8")
        assert "\"shots\"" in data
