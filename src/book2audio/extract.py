"""Extract plain text from supported input file formats."""

from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib
from pypdf import PdfReader


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _extract_txt(path)
    if suffix == ".epub":
        return _extract_epub(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_epub(path: Path) -> str:
    book = epub.read_epub(str(path))
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(parts)
