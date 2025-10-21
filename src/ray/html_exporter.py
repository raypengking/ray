"""Render storyboard outputs into an HTML preview."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:  # pragma: no cover - handled at runtime
    Environment = FileSystemLoader = select_autoescape = None  # type: ignore

from .storyboard import Storyboard


class StoryboardHTMLExporter:
    """Export storyboard results to HTML using Jinja2 templates."""

    def __init__(self, template_dir: Optional[Path] = None):
        if Environment is None or FileSystemLoader is None or select_autoescape is None:
            raise RuntimeError("jinja2 is required to export HTML. Please install it via `pip install jinja2`.")

        if template_dir is not None:
            loader = FileSystemLoader(str(template_dir))
        else:
            default_dir = Path(__file__).resolve().parent / "templates"
            loader = FileSystemLoader(str(default_dir))
        self.env = Environment(loader=loader, autoescape=select_autoescape(["html", "xml"]))

    def export(
        self,
        storyboard: Storyboard,
        output_dir: Path,
        file_name: str = "index.html",
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        template = self.env.get_template("storyboard.html.j2")
        generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        html = template.render(storyboard=storyboard, generated_at=generated_at)
        html_path = output_dir / file_name
        html_path.write_text(html, encoding="utf-8")

        json_path = output_dir / "storyboard.json"
        json_path.write_text(json.dumps(storyboard.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return html_path


__all__ = ["StoryboardHTMLExporter"]
