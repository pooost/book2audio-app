"""Get Markdown text out of a book: MarkItDown first, OpenOCR as a fallback
for pages with no usable text layer (scanned PDFs, raw images).

Pipeline order: extract -> deterministic cleanup -> optional Qwen review.
Cleanup runs BEFORE review (not after) so Qwen compares a page's image
against text that's already had mechanical OCR noise (page numbers,
hyphen-linebreaks, control characters) stripped, not raw noise plus real
content mixed together.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from book2audio.ingest.markitdown_backend import SUPPORTED_SUFFIXES as MARKITDOWN_SUFFIXES
from book2audio.ingest.markitdown_backend import run_markitdown
from book2audio.ocr.backend import IMAGE_SUFFIXES, run_openocr, run_openocr_batch
from book2audio.ocr.quality import looks_scanned

OcrMode = Literal["auto", "force", "never"]


class ExtractionError(Exception):
    pass


@dataclass
class ExtractResult:
    raw_text: str
    cleaned_text: str
    reviewed_text: str | None = None  # None unless ai_review actually ran
    review_flags: list[str] = field(default_factory=list)
    ai_review_applied: bool = False
    ai_review_skip_reason: str | None = None

    @property
    def final_text(self) -> str:
        """What chapter-detection/chunking/TTS should use: the reviewed
        text when review ran, otherwise the cleaned text."""
        return self.reviewed_text if self.reviewed_text is not None else self.cleaned_text


def _ocr(input_path: Path, ocr_output_dir: Path | None, allow_download: bool) -> str:
    try:
        return run_openocr(input_path, ocr_output_dir, allow_download=allow_download)
    except Exception as e:
        raise ExtractionError(str(e)) from e


def extract_with_review(
    input_path: Path,
    ocr_output_dir: Path,
    ocr_mode: OcrMode = "auto",
    allow_download: bool = True,
    ai_review: bool = False,
    ai_review_model: str = "qwen3-vl:4b-instruct",
    on_page_progress: Callable[[int, int], None] | None = None,
) -> ExtractResult:
    """Page-aware extraction: when OCR applies and ai_review is on, each
    page's (cleaned) OCR text is checked against its own source image
    before being concatenated. AI review only makes sense against an
    actual OCR transcription -- content whose real text layer is used
    directly (no OCR involved) skips review, with
    ExtractResult.ai_review_skip_reason explaining why."""
    from book2audio.processing.clean import clean_text

    suffix = input_path.suffix.lower()
    is_image_input = suffix in IMAGE_SUFFIXES or input_path.is_dir()

    if not is_image_input and suffix not in MARKITDOWN_SUFFIXES:
        raise ExtractionError(f"Unsupported input: {input_path} (suffix {suffix!r})")

    uses_ocr = True
    markitdown_text: str | None = None

    if not is_image_input:
        if suffix != ".pdf" or ocr_mode == "never":
            uses_ocr = False
        elif ocr_mode == "auto":
            markitdown_text = run_markitdown(input_path)
            uses_ocr = looks_scanned(input_path, markitdown_text)
        # ocr_mode == "force" -> uses_ocr stays True

    if not uses_ocr:
        raw = markitdown_text if markitdown_text is not None else run_markitdown(input_path)
        skip_reason = (
            "This document's text layer was used directly -- no OCR transcription to review."
            if ai_review else None
        )
        return ExtractResult(raw_text=raw, cleaned_text=clean_text(raw), ai_review_skip_reason=skip_reason)

    if not ai_review:
        raw = _ocr(input_path, ocr_output_dir, allow_download)
        return ExtractResult(raw_text=raw, cleaned_text=clean_text(raw))

    return _ocr_with_review(input_path, is_image_input, ocr_output_dir, allow_download, ai_review_model, on_page_progress)


def _ocr_with_review(
    input_path: Path,
    is_image_input: bool,
    ocr_output_dir: Path,
    allow_download: bool,
    ai_review_model: str,
    on_page_progress: Callable[[int, int], None] | None,
) -> ExtractResult:
    from book2audio.processing.clean import clean_text

    try:
        if is_image_input and input_path.is_dir():
            page_images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if not page_images:
                raise ExtractionError(f"No supported images found in {input_path}")
        elif is_image_input:
            page_images = [input_path]
        else:
            from book2audio.ingest.page_render import render_pdf_pages

            page_images = render_pdf_pages(input_path, ocr_output_dir / "pages")

        page_texts = run_openocr_batch(page_images, ocr_output_dir / "ocr", allow_download=allow_download)
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(str(e)) from e

    raw_text = "\n\n".join(page_texts[p] for p in page_images)
    cleaned_pages = {p: clean_text(page_texts[p]) for p in page_images}
    cleaned_text = "\n\n".join(cleaned_pages[p] for p in page_images)

    from book2audio.processing.ai_review import review_page, unload_model

    reviewed_texts = []
    flags = []
    total = len(page_images)
    try:
        for i, img_path in enumerate(page_images, start=1):
            if on_page_progress:
                on_page_progress(i, total)
            # Review the CLEANED text, not the raw OCR -- mechanical noise
            # (page numbers, hyphen-linebreaks) is already gone by now, so
            # Qwen is comparing real content against the image, not noise.
            result = review_page(img_path, cleaned_pages[img_path], model=ai_review_model)
            reviewed_texts.append(result.text)
            flags.extend(f"page {i}: {f}" for f in result.flags)
    finally:
        # Free the VL model's GPU memory before the pipeline moves on to
        # load a TTS model -- see unload_model()'s docstring for why this
        # matters on a single shared GPU. `finally` so a failed/cancelled
        # review pass doesn't leave the model resident either.
        unload_model(ai_review_model)

    return ExtractResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        reviewed_text="\n\n".join(reviewed_texts),
        review_flags=flags,
        ai_review_applied=True,
    )
