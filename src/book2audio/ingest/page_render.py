"""Rasterize PDF pages to individual images.

Needed to feed OpenOCR one page at a time (rather than handing it a raw
multi-page PDF) and, more importantly, to pair each page's OCR text with
its own source image for AI review -- a vision model can't compare OCR
output against "the page" unless it actually has that page as an image.
"""

from pathlib import Path


def render_pdf_pages(pdf_path: Path, out_dir: Path, page_indices: list[int] | None = None, dpi: int = 200) -> list[Path]:
    """Render the given 0-indexed pages (or all pages) to PNGs in out_dir,
    named so their sort order matches page order. Returns the image paths
    in page order."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        indices = page_indices if page_indices is not None else list(range(doc.page_count))
        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)

        paths = []
        for i in indices:
            pix = doc[i].get_pixmap(matrix=matrix)
            out_path = out_dir / f"page_{i + 1:04d}.png"
            pix.save(out_path)
            paths.append(out_path)
        return paths
    finally:
        doc.close()
