"""Utilities for preparing the input text before sending to the LLM."""

from __future__ import annotations

import re
from typing import Iterable, List

_SENTENCE_END = re.compile(r"(?<=[。！？；!?])")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace and remove redundant blank lines."""

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    normalized = "\n".join(lines)
    return _WHITESPACE.sub(" ", normalized)


def _split_sentences(paragraph: str) -> List[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    tokens: List[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        end = match.end()
        segment = paragraph[start:end].strip()
        if segment:
            tokens.append(segment)
        start = end
    tail = paragraph[start:].strip()
    if tail:
        tokens.append(tail)
    return tokens or [paragraph]


def split_text(text: str, max_length: int = 260, overlap: int = 20) -> List[str]:
    """Split long Chinese text into overlapping chunks suitable for prompting."""

    text = normalize_text(text)
    if not text:
        return []

    paragraphs: Iterable[str] = [p for p in text.split("\n") if p]
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_length = 0

    for paragraph in paragraphs:
        sentences = _split_sentences(paragraph)
        for sentence in sentences:
            sentence_len = len(sentence)
            if buffer and buffer_length + sentence_len > max_length:
                chunk = "".join(buffer).strip()
                if chunk:
                    chunks.append(chunk)
                if overlap > 0 and buffer:
                    overlap_text = "".join(buffer)[-overlap:]
                else:
                    overlap_text = ""
                buffer = [overlap_text, sentence] if overlap_text else [sentence]
                buffer_length = sum(len(item) for item in buffer)
            else:
                buffer.append(sentence)
                buffer_length += sentence_len

    if buffer:
        chunk = "".join(buffer).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


__all__ = ["split_text", "normalize_text"]
