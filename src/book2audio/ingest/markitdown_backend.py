"""MarkItDown wrapper: pull Markdown out of documents that have a real text layer."""

from pathlib import Path

from markitdown import MarkItDown

SUPPORTED_SUFFIXES = {".pdf", ".epub", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".html"}


def run_markitdown(path: Path) -> str:
    result = MarkItDown().convert(path)
    return result.text_content or ""
