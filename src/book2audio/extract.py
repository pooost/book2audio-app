"""Get Markdown text out of a book: MarkItDown first, OpenOCR as a fallback
for pages with no usable text layer (scanned PDFs, raw images)."""

import shutil
import subprocess
import sys
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


def _find_openocr() -> str:
    # Console scripts land next to the interpreter that installed them.
    # Resolving via sys.executable means this works whether or not the venv
    # is actually activated (running `.venv/bin/book2audio` directly, e.g.,
    # doesn't put `.venv/bin` on PATH) -- shutil.which alone would silently
    # fail to find an openocr that is, in fact, right there and installed.
    venv_local = Path(sys.executable).parent / "openocr"
    if venv_local.exists():
        return str(venv_local)
    found = shutil.which("openocr")
    if found:
        return found
    raise ExtractionError("openocr is not on PATH; install it with the rest of the stack.")


def _run_openocr(input_path: Path, output_dir: Path | None) -> str:
    openocr_bin = _find_openocr()

    if output_dir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            return _run_openocr_into(input_path, Path(tmpdir), openocr_bin)
    return _run_openocr_into(input_path, output_dir, openocr_bin)


def _run_openocr_into(input_path: Path, output_dir: Path, openocr_bin: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        openocr_bin,
        "--task", "doc",
        "--input_path", str(input_path),
        "--output_path", str(output_dir),
        "--use_layout_detection",
        "--save_markdown",
    ]

    # Offline-first: only let openocr hit the network if its local model
    # cache is actually missing something.
    try:
        subprocess.run(base_cmd + ["--no_auto_download"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        try:
            subprocess.run(base_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise ExtractionError(f"openocr failed on {input_path}:\n{e.stderr}") from e

    md_files = sorted(output_dir.rglob("*.md"))
    if not md_files:
        raise ExtractionError(f"openocr produced no markdown output for {input_path} in {output_dir}")

    return "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in md_files)
