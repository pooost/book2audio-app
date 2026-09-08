"""Text-only narration cleanup: a conservative pass that repairs obvious OCR
corruption (broken words, garbage digit/symbol strings, character
substitutions, malformed hyphenation) before narration -- WITHOUT
rewriting, summarizing, paraphrasing, or verifying against a page image.

Off by default (ConversionRequest.narration_cleanup). Talks only to a
local Ollama server (http://localhost:11434), text only -- never an
image, never a cloud API, so book content never leaves the machine even
with this feature on.

This replaced an earlier design (processing/ai_review.py, now this file)
that sent each page's image to a vision model alongside its OCR text.
That comparison is gone entirely: Qwen now only ever sees text.
"""

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "qwen3-vl:4b-instruct"
OLLAMA_URL = "http://localhost:11434"
TEMPERATURE = 0.1

# Bump this whenever CLEANUP_PROMPT changes in a way that would meaningfully
# change output -- it's part of the cache key, so a bump invalidates old
# cached cleanup results instead of silently reusing them under a
# different prompt.
PROMPT_VERSION = "v1"

CLEANUP_PROMPT = """You are the text-cleanup stage of a local audiobook conversion application.

Your input is OCR-derived text.

Your job is to repair obvious OCR corruption so the text can be read naturally by a text-to-speech system.

Correct broken words, accidental spaces, OCR character substitutions, malformed punctuation, duplicated fragments, broken line-break hyphenation, and obvious recognition errors.

Remove meaningless OCR garbage such as random digit/symbol sequences when context clearly shows they are not part of the text.

Use linguistic context and general knowledge to reconstruct obvious corrupted words.

Preserve legitimate numbers, dates, quotations, names, terminology, grammar, sentence order, paragraph order, and authorial style.

Do not summarize.
Do not simplify.
Do not paraphrase.
Do not modernize.
Do not improve the prose stylistically.
Do not add information.
Do not invent missing passages.

When uncertain whether something is legitimate text or OCR corruption, prefer preserving it.

Return only the cleaned text."""

_PARAGRAPH_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class NarrationCleanupError(Exception):
    pass


@dataclass
class CleanupAuditEntry:
    chunk_index: int
    input_text: str
    output_text: str
    model: str
    status: str  # "ok" | "cached" | "error"


@dataclass
class CleanupResult:
    text: str
    audit: list[CleanupAuditEntry] = field(default_factory=list)


def is_ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def available_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def chunk_for_cleanup(text: str, target_chars: int = 2000, max_chars: int = 3000) -> list[str]:
    """Group paragraphs into ~1000-3000 char chunks (spec range; default
    target 2000) for Qwen, preserving paragraph breaks ("\\n\\n") both
    within a chunk and across chunk boundaries -- unlike
    processing/chunker.py's TTS-oriented chunk_text(), which intentionally
    flattens paragraph structure (fine for TTS, which runs after chapter
    detection; wrong here, since chapter detection runs AFTER this and
    needs headings to stay on their own line). Never cuts a word: a single
    paragraph longer than max_chars falls back to sentence-level splits.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    units: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
        else:
            piece = ""
            for sentence in _PARAGRAPH_SENTENCE_SPLIT.split(para):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if piece and len(piece) + 1 + len(sentence) > max_chars:
                    units.append(piece)
                    piece = sentence
                else:
                    piece = f"{piece} {sentence}".strip()
            if piece:
                units.append(piece)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        added = len(unit) + (2 if current else 0)
        if current and current_len + added > target_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(unit)
        current_len += len(unit) + (2 if len(current) > 1 else 0)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _cache_key(text: str, model: str) -> str:
    h = hashlib.sha256()
    for part in (text, model, PROMPT_VERSION, str(TEMPERATURE)):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def cleanup_chunk(text: str, model: str = DEFAULT_MODEL, timeout: float = 180.0) -> str:
    """Send one text-only chunk to a local Ollama model for conservative
    OCR cleanup. Raises NarrationCleanupError on any failure -- never
    silently returns the uncleaned text, never falls back to a cloud API.
    """
    payload = json.dumps({
        "model": model,
        "system": CLEANUP_PROMPT,
        "prompt": text,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": 8192},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise NarrationCleanupError(f"Ollama rejected the request (HTTP {e.code}): {body}") from e
    except urllib.error.URLError as e:
        raise NarrationCleanupError(
            f"Could not reach local Ollama server at {OLLAMA_URL} ({e}). "
            f"Install Ollama and run `ollama pull {model}`, or turn narration cleanup off."
        ) from e
    except TimeoutError as e:
        raise NarrationCleanupError(f"Narration cleanup timed out after {timeout}s.") from e

    if "error" in data:
        raise NarrationCleanupError(f"Ollama error: {data['error']}")

    cleaned = data.get("response", "").strip()
    if not cleaned:
        raise NarrationCleanupError("Ollama returned an empty response.")
    return cleaned


def unload_model(model: str = DEFAULT_MODEL) -> None:
    """Ollama keeps a model resident on the GPU for a few minutes after use
    by default. On a single shared GPU that collides with the TTS model's
    own VRAM need right after cleanup finishes (this exact scenario -- an
    Ollama model left loaded causing the next stage's CUDA allocation to
    fail -- was reproduced and fixed earlier in this project for the
    vision-review predecessor of this module; same fix applies here). Call
    once after the whole cleanup pass, not per-chunk -- unloading per-chunk
    would force a costly reload before every single chunk instead of once
    at the end."""
    payload = json.dumps({"model": model, "prompt": "", "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=30).close()
    except Exception:
        pass  # best-effort -- don't fail the conversion over a GPU-memory-cleanup step


def cleanup_chunk_cached(text: str, model: str, cache_dir: Path) -> tuple[str, bool]:
    """Returns (cleaned_text, was_cache_hit). Cache identity = chunk text +
    model + prompt version + temperature (_cache_key), so an interrupted
    run resumes without re-cleaning finished chunks, and a prompt change
    (PROMPT_VERSION bump) never silently reuses stale results."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(text, model)
    cache_path = cache_dir / f"{key}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), True

    cleaned = cleanup_chunk(text, model=model)

    # Atomic write -- same reasoning as the TTS providers' chunk caching:
    # a killed/crashed run must never leave a partial cache entry that a
    # resumed run mistakes for a finished one.
    tmp_path = cache_path.with_suffix(".txt.tmp")
    tmp_path.write_text(cleaned, encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return cleaned, False


def cleanup_document(
    text: str,
    model: str = DEFAULT_MODEL,
    cache_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    debug: bool = False,
) -> CleanupResult:
    """Chunk `text`, clean each chunk (cached, resumable), and rejoin in
    original order with "\\n\\n" -- chunks never overlap, so recombination
    is a plain join, not a merge. debug=True additionally returns full
    per-chunk audit entries (input/output/status) for diagnosing cleanup
    mistakes; off by default so normal runs don't retain that."""
    chunks = chunk_for_cleanup(text)
    if not chunks:
        return CleanupResult(text="")

    effective_cache_dir = cache_dir or Path(".narration_cleanup_cache")
    cleaned_chunks = []
    audit: list[CleanupAuditEntry] = []

    try:
        for i, chunk in enumerate(chunks, start=1):
            if on_progress:
                on_progress(i, len(chunks))
            try:
                cleaned, was_cached = cleanup_chunk_cached(chunk, model, effective_cache_dir)
                status = "cached" if was_cached else "ok"
            except NarrationCleanupError:
                if debug:
                    audit.append(CleanupAuditEntry(i, chunk, "", model, "error"))
                raise
            cleaned_chunks.append(cleaned)
            if debug:
                audit.append(CleanupAuditEntry(i, chunk, cleaned, model, status))
    finally:
        # Free Ollama's GPU memory before the pipeline moves on to load a
        # TTS model -- see unload_model()'s docstring. `finally` so a
        # failed/cancelled cleanup pass doesn't leave the model resident.
        unload_model(model)

    return CleanupResult(text="\n\n".join(cleaned_chunks), audit=audit)
