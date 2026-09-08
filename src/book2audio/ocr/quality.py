"""Decide whether a PDF's MarkItDown extraction is real text or noise from a scan."""

from pathlib import Path

import pymupdf

# Below this many extracted characters per PDF page, assume the page has no
# real text layer (i.e. it's a scan) and fall back to OCR.
MIN_CHARS_PER_PAGE = 40


def looks_scanned(pdf_path: Path, extracted_text: str) -> bool:
    doc = pymupdf.open(pdf_path)
    page_count = max(doc.page_count, 1)
    doc.close()
    avg_chars_per_page = len(extracted_text.strip()) / page_count
    return avg_chars_per_page < MIN_CHARS_PER_PAGE
