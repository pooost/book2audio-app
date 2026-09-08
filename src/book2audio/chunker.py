"""Split chapter text into TTS-sized chunks without cutting sentences.

Chatterbox generates autoregressively up to a fixed step budget, so very
long inputs get truncated or drift in prosody. Keep chunks short.
"""

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

DEFAULT_MAX_CHARS = 300


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(_hard_split(sentence, max_chars))
                continue
            if current and len(current) + len(sentence) + 1 > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    words = sentence.split(" ")
    parts: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + len(word) + 1 > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts
