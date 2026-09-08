"""Select a subset of a PDF's pages before extraction, so a user can convert
just a chapter or range instead of an entire book."""

from pathlib import Path


class PageRangeError(Exception):
    pass


def parse_page_range(spec: str, page_count: int) -> list[int]:
    """Parse a 1-indexed, inclusive spec like "1-10,15,20-25" into a sorted,
    deduplicated list of 0-indexed page numbers. An empty/blank spec means
    "all pages"."""
    spec = spec.strip()
    if not spec:
        return list(range(page_count))

    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_str, _, end_str = part.partition("-")
            try:
                start, end = int(start_str), int(end_str)
            except ValueError:
                raise PageRangeError(f"Invalid page range segment: {part!r}") from None
        else:
            try:
                start = end = int(part)
            except ValueError:
                raise PageRangeError(f"Invalid page number: {part!r}") from None

        if start < 1 or end < 1 or start > end:
            raise PageRangeError(f"Invalid page range: {part!r}")
        if end > page_count:
            raise PageRangeError(f"Page {end} is out of range -- this document has {page_count} page(s).")

        pages.update(range(start - 1, end))

    if not pages:
        raise PageRangeError("Page range resolved to no pages.")
    return sorted(pages)


def get_page_count(pdf_path: Path) -> int:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def extract_page_subset(pdf_path: Path, page_range: str, out_path: Path) -> Path:
    """Write a new PDF at out_path containing only the pages in page_range."""
    import pymupdf

    src = pymupdf.open(pdf_path)
    try:
        page_indices = parse_page_range(page_range, src.page_count)
        dst = pymupdf.open()
        try:
            for i in page_indices:
                dst.insert_pdf(src, from_page=i, to_page=i)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            dst.save(out_path)
        finally:
            dst.close()
    finally:
        src.close()
    return out_path
