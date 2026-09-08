"""Turn raw extracted Markdown into clean prose, split into chapters."""

import re
from dataclasses import dataclass

_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_LINEBREAK = re.compile(r"(?<![.!?:\n])\n(?!\n)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(\[?\d{1,4}\]?|[ivxlcdm]{1,6}|-\s*\d{1,4}\s*-)\s*$", re.IGNORECASE)
_FOOTNOTE_MARKER = re.compile(r"\[\^?\d+\]|\(\d+\)$")
# Superscript footnote refs often flatten to a bare 1-2 digit number glued
# straight onto the preceding word (e.g. "Antiquity3 aside") with no space.
_GLUED_FOOTNOTE_REF = re.compile(r"(?<=[a-z])\d{1,2}\b")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SMART_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
})

_CHAPTER_HEADING = re.compile(
    r"^\s*(?:#{1,3}\s*)?(chapter|part|book)\s+([ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b.*$",
    re.IGNORECASE,
)
_MD_HEADING = re.compile(r"^#{1,3}\s+(.*)$")


@dataclass
class Chapter:
    title: str
    text: str


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(raw: str) -> str:
    text = _CONTROL_CHARS.sub(" ", raw)
    text = text.translate(_SMART_QUOTES)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    lines = [
        line for line in text.split("\n")
        if not _PAGE_NUMBER_LINE.match(line.strip())
    ]
    text = "\n".join(lines)

    text = _FOOTNOTE_MARKER.sub("", text)
    text = _GLUED_FOOTNOTE_REF.sub("", text)
    text = _SOFT_LINEBREAK.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)

    return text.strip()


def split_chapters(cleaned: str) -> list[Chapter]:
    lines = cleaned.split("\n")
    chapters: list[Chapter] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            chapters.append(Chapter(title=current_title or f"Chapter {len(chapters) + 1}", text=body))

    for line in lines:
        stripped = line.strip()
        heading_match = _CHAPTER_HEADING.match(stripped) or _MD_HEADING.match(stripped)
        if heading_match and len(stripped) < 100:
            flush()
            current_title = stripped.lstrip("#").strip() or f"Chapter {len(chapters) + 1}"
            current_lines = []
        else:
            current_lines.append(line)

    flush()

    if not chapters:
        chapters = [Chapter(title="Book", text=cleaned.strip())]

    return chapters
