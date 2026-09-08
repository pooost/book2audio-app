"""Optional local-LLM pass to fix OCR/extraction errors before narration.

Off by default (ConversionRequest.ai_review). Talks only to a local Ollama
server (http://localhost:11434) -- deliberately not wired to any cloud API,
so book text never leaves the machine even with this feature on.
"""

import json
import urllib.error
import urllib.request

DEFAULT_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434"

REVIEW_PROMPT = """You are cleaning up text extracted from a document via OCR or PDF text extraction, before it is narrated aloud by a text-to-speech system.

Fix ONLY clear extraction errors: misrecognized characters, garbled words, obviously wrong OCR substitutions, and broken word-joins. Do not rephrase, summarize, add, remove, or otherwise change meaning or style. If a passage already looks correct, leave it completely unchanged.

Return ONLY the corrected text -- no preamble, no explanation, no markdown formatting.

TEXT:
{text}"""


class AiReviewError(Exception):
    pass


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


def review_text(text: str, model: str = DEFAULT_MODEL, timeout: float = 180.0) -> str:
    """Send text to a local Ollama model for OCR-error cleanup.

    Raises AiReviewError if Ollama isn't reachable, the model isn't
    available, or the response is empty -- never silently falls back to
    returning the unreviewed text, and never falls back to a cloud API.
    """
    payload = json.dumps({
        "model": model,
        "prompt": REVIEW_PROMPT.format(text=text),
        "stream": False,
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
    except urllib.error.URLError as e:
        raise AiReviewError(
            f"Could not reach local Ollama server at {OLLAMA_URL} ({e}). "
            "Install Ollama and run `ollama pull llama3.2`, or turn AI review off."
        ) from e
    except TimeoutError as e:
        raise AiReviewError(f"Local model review timed out after {timeout}s.") from e

    if "error" in data:
        raise AiReviewError(f"Ollama error: {data['error']}")

    reviewed = data.get("response", "").strip()
    if not reviewed:
        raise AiReviewError("Ollama returned an empty response.")
    return reviewed
