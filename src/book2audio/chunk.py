"""Split long text into TTS-friendly chunks without cutting sentences."""

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    sentences = _SENTENCE_SPLIT.split(text)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current.strip())

    return chunks
