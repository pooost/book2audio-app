"""Get Markdown text out of a book: MarkItDown first, OpenOCR as a fallback
for pages with no usable text layer (scanned PDFs, raw images)."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf
from markitdown import MarkItDown

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MARKITDOWN_SUFFIXES = {".pdf", ".epub", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".html"}

# Below this many extracted characters per PDF page, assume the page has no
# real text layer (i.e. it's a scan) and fall back to OCR.
MIN_CHARS_PER_PAGE = 40


class ExtractionError(Exception):
    pass


def extract_markdown(input_path: Path, ocr_output_dir: Path | None = None) -> str:
    suffix = input_path.suffix.lower()

    if suffix in IMAGE_SUFFIXES or input_path.is_dir():
        return _run_openocr(input_path, ocr_output_dir)

    if suffix in MARKITDOWN_SUFFIXES:
        text = _run_markitdown(input_path)
        if suffix == ".pdf" and _looks_scanned(input_path, text):
            return _run_openocr(input_path, ocr_output_dir)
        return text

    raise ExtractionError(f"Unsupported input: {input_path} (suffix {suffix!r})")


def _run_markitdown(path: Path) -> str:
    result = MarkItDown().convert(path)
    return result.text_content or ""


def _looks_scanned(pdf_path: Path, extracted_text: str) -> bool:
    doc = pymupdf.open(pdf_path)
    page_count = max(doc.page_count, 1)
    doc.close()
    avg_chars_per_page = len(extracted_text.strip()) / page_count
    return avg_chars_per_page < MIN_CHARS_PER_PAGE


def _run_openocr(input_path: Path, output_dir: Path | None) -> str:
    if shutil.which("openocr") is None:
        raise ExtractionError("openocr is not on PATH; install it with the rest of the stack.")

    own_tempdir = None
    if output_dir is None:
        own_tempdir = tempfile.TemporaryDirectory()
        output_dir = Path(own_tempdir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "openocr",
                "--task", "doc",
                "--input_path", str(input_path),
                "--output_path", str(output_dir),
                "--use_layout_detection",
                "--save_markdown",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise ExtractionError(f"openocr failed on {input_path}:\n{e.stderr}") from e

    md_files = sorted(output_dir.rglob("*.md"))
    if not md_files:
        raise ExtractionError(f"openocr produced no markdown output for {input_path} in {output_dir}")

    text = "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in md_files)

    if own_tempdir is not None:
        own_tempdir.cleanup()

    return text
