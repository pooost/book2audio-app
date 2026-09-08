"""Get Markdown text out of a book: MarkItDown first, OpenOCR as a fallback
for pages with no usable text layer (scanned PDFs, raw images)."""

from pathlib import Path
from typing import Literal

from book2audio.ingest.markitdown_backend import SUPPORTED_SUFFIXES as MARKITDOWN_SUFFIXES
from book2audio.ingest.markitdown_backend import run_markitdown
from book2audio.ocr.backend import IMAGE_SUFFIXES, run_openocr
from book2audio.ocr.quality import looks_scanned

OcrMode = Literal["auto", "force", "never"]


class ExtractionError(Exception):
    pass


def extract_markdown(
    input_path: Path,
    ocr_output_dir: Path | None = None,
    ocr_mode: OcrMode = "auto",
    allow_download: bool = True,
) -> str:
    suffix = input_path.suffix.lower()
    is_image_input = suffix in IMAGE_SUFFIXES or input_path.is_dir()

    if is_image_input:
        return _ocr(input_path, ocr_output_dir, allow_download)

    if suffix not in MARKITDOWN_SUFFIXES:
        raise ExtractionError(f"Unsupported input: {input_path} (suffix {suffix!r})")

    if suffix == ".pdf" and ocr_mode == "force":
        return _ocr(input_path, ocr_output_dir, allow_download)

    text = run_markitdown(input_path)

    if suffix == ".pdf" and ocr_mode == "auto" and looks_scanned(input_path, text):
        return _ocr(input_path, ocr_output_dir, allow_download)

    return text


def _ocr(input_path: Path, ocr_output_dir: Path | None, allow_download: bool) -> str:
    try:
        return run_openocr(input_path, ocr_output_dir, allow_download=allow_download)
    except Exception as e:
        raise ExtractionError(str(e)) from e
