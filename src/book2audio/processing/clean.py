"""Turn raw extracted Markdown into clean prose, split into chapters."""

import re
from collections import Counter
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

_TRAILING_PAGE_NUMBER = re.compile(r"\s*\d+\s*$")
_LEADING_PAGE_NUMBER = re.compile(r"^\s*\d+\s*")
_SENTENCE_ENDING = re.compile(r'[.!?"\')\]]$')
# Unambiguous page-furniture signals: a print-export filename (Foo.indd,
# Foo.docx), a date (06/12/2011), or a time (14:26). A line matching one
# of these is never legitimate running prose, so every occurrence is
# removed. Text with none of these signals (e.g. a bare title-case phrase)
# is genuinely ambiguous -- see the docstring below for why those are
# handled differently.
_STRONG_FURNITURE_SIGNAL = re.compile(
    r"\.\w{2,4}\b"  # filename extension: PRINT.indd, file.docx
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"  # date: 06/12/2011, 06-12-2011
    r"|\b\d{1,2}:\d{2}\b"  # time: 14:26
)
REPEATED_LINE_MIN_OCCURRENCES = 3
REPEATED_LINE_MAX_LENGTH = 80


def _strip_repeated_lines(text: str) -> str:
    """Strip running headers/footers: a line that recurs many times across
    the whole document (e.g. a print-export footer like "Book PRINT.indd
    9", stamped fresh on every page with only the page number changing,
    or a repeated production timestamp) is page layout furniture, not
    authored content. This can only work at the whole-document level --
    Qwen's narration cleanup only ever sees one local chunk at a time, so
    it has no way to notice a line is repeated dozens of pages apart.

    Guarded against two distinct false-positive risks:

    1. Legitimately repeated short prose (e.g. a one-word line of dialogue
       like "Yes." appearing several times): real running headers/footers
       are essentially never complete sentences, so a line ending in
       terminal punctuation is never treated as furniture regardless of
       how often it recurs.

    2. A chapter/section title that HAPPENS to be reused as the running
       header on every subsequent page of that chapter (a common print
       layout: "Technical Mentality" printed once as the real heading,
       then again as a running head on pages 3, 5, 7, 9...). Blindly
       stripping every occurrence of a repeated line deletes the real
       title along with the furniture copies -- reproduced and confirmed
       on a real book, not hypothetical. Fix: only strip every occurrence
       for lines matching a strong, unambiguous furniture signal (a
       filename extension, a date, or a time -- content prose essentially
       never looks like these). A repeated line with none of those
       signals keeps its first occurrence (plausibly the real heading)
       and only strips the 2nd+ (plausibly the running-header copies).
    """
    lines = text.split("\n")

    def normalize(line: str) -> str:
        # Strip a leading OR trailing page number so "Foo.indd 9" /
        # "Foo.indd 214" (trailing -- common footer convention) and
        # "6 Running Header" / "8 Running Header" (leading -- common
        # two-sided-book convention, page number on the outer margin
        # alternating sides) both collapse to the same recurring line.
        stripped = line.strip()
        return _LEADING_PAGE_NUMBER.sub("", _TRAILING_PAGE_NUMBER.sub("", stripped)).strip()

    counts = Counter(normalize(line) for line in lines if normalize(line))
    kept_first_occurrence: set[str] = set()

    def should_drop(line: str) -> bool:
        norm = normalize(line)
        if not norm or len(norm) >= REPEATED_LINE_MAX_LENGTH:
            return False
        if _SENTENCE_ENDING.search(norm):
            return False
        if counts[norm] < REPEATED_LINE_MIN_OCCURRENCES:
            return False

        # Check the furniture signal against the ORIGINAL line, not the
        # normalized one: normalize()'s leading-page-number strip mangles
        # a date like "06/12/2011" into "/12/2011" (eating the day as if
        # it were a page number), which then fails to match the date
        # pattern below -- reproduced live against a real 30-times-
        # repeated timestamp footer, where exactly one of the thirty
        # occurrences fell through to the "keep first" branch instead of
        # being dropped like the other twenty-nine, purely because of
        # this ordering bug.
        if _STRONG_FURNITURE_SIGNAL.search(line.strip()):
            return True  # unambiguous furniture -- drop every occurrence

        # Ambiguous (could be a genuine, once-only heading): keep the
        # first occurrence, drop the rest.
        if norm in kept_first_occurrence:
            return True
        kept_first_occurrence.add(norm)
        return False

    return "\n".join(line for line in lines if not should_drop(line))


def clean_text(raw: str) -> str:
    text = _CONTROL_CHARS.sub(" ", raw)
    text = text.translate(_SMART_QUOTES)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    lines = [
        line for line in text.split("\n")
        if not _PAGE_NUMBER_LINE.match(line.strip())
    ]
    text = "\n".join(lines)
    text = _strip_repeated_lines(text)

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
