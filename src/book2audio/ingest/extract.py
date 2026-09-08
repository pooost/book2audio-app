"""Get Markdown text out of a book: MarkItDown first, OpenOCR as a fallback
for pages with no usable text layer (scanned PDFs, raw images).

Pipeline: extract -> deterministic cleanup -> optional Qwen narration
cleanup (text-only; see processing/narration_cleanup.py). Narration
cleanup applies uniformly regardless of extraction path -- it's no longer
tied to OCR/page images the way the earlier vision-review design was, so
there's no "skip reason" concept anymore.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from book2audio.ingest.markitdown_backend import SUPPORTED_SUFFIXES as MARKITDOWN_SUFFIXES
from book2audio.ingest.markitdown_backend import run_markitdown
from book2audio.ocr.backend import IMAGE_SUFFIXES, run_openocr
from book2audio.ocr.quality import looks_scanned

OcrMode = Literal["auto", "force", "never"]


class ExtractionError(Exception):
    pass


@dataclass
class ExtractResult:
    raw_text: str
    cleaned_text: str
    narration_text: str | None = None  # None unless narration cleanup actually ran
    narration_cleanup_applied: bool = False
    cleanup_audit: list = field(default_factory=list)  # CleanupAuditEntry list, only when debug=True

    @property
    def final_text(self) -> str:
        """What chapter-detection/chunking/TTS should use: the narration-
        cleaned text when cleanup ran, otherwise the deterministic-cleaned
        text."""
        return self.narration_text if self.narration_text is not None else self.cleaned_text


def _ocr(input_path: Path, ocr_output_dir: Path | None, allow_download: bool) -> str:
    try:
        return run_openocr(input_path, ocr_output_dir, allow_download=allow_download)
    except Exception as e:
        raise ExtractionError(str(e)) from e


def extract_and_clean(
    input_path: Path,
    ocr_output_dir: Path,
    ocr_mode: OcrMode = "auto",
    allow_download: bool = True,
    narration_cleanup: bool = False,
    narration_cleanup_model: str = "qwen3-vl:4b-instruct",
    on_cleanup_progress: Callable[[int, int], None] | None = None,
    debug_cleanup: bool = False,
) -> ExtractResult:
    """Extract raw text (MarkItDown or OpenOCR), run deterministic cleanup,
    then optionally run Qwen narration cleanup (text-only, chunked)."""
    from book2audio.processing.clean import clean_text

    suffix = input_path.suffix.lower()
    is_image_input = suffix in IMAGE_SUFFIXES or input_path.is_dir()

    if not is_image_input and suffix not in MARKITDOWN_SUFFIXES:
        raise ExtractionError(f"Unsupported input: {input_path} (suffix {suffix!r})")

    if is_image_input:
        raw = _ocr(input_path, ocr_output_dir, allow_download)
    elif suffix == ".pdf" and ocr_mode == "force":
        raw = _ocr(input_path, ocr_output_dir, allow_download)
    elif suffix == ".pdf" and ocr_mode != "never":
        raw = run_markitdown(input_path)
        if looks_scanned(input_path, raw):
            raw = _ocr(input_path, ocr_output_dir, allow_download)
    else:
        raw = run_markitdown(input_path)

    cleaned = clean_text(raw)

    if not narration_cleanup:
        return ExtractResult(raw_text=raw, cleaned_text=cleaned)

    from book2audio.processing.narration_cleanup import cleanup_document

    result = cleanup_document(
        cleaned,
        model=narration_cleanup_model,
        cache_dir=ocr_output_dir.parent / "narration_cleanup",
        on_progress=on_cleanup_progress,
        debug=debug_cleanup,
    )

    return ExtractResult(
        raw_text=raw,
        cleaned_text=cleaned,
        narration_text=result.text,
        narration_cleanup_applied=True,
        cleanup_audit=result.audit,
    )
