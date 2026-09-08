"""Optional local vision-language review of OCR text against its source page
image, before narration.

Off by default (ConversionRequest.ai_review). Talks only to a local Ollama
server (http://localhost:11434) -- deliberately not wired to any cloud API,
so book content never leaves the machine even with this feature on.

Only meaningful for content that actually went through OCR: this compares
an OCR transcription against the page image it came from, so it needs both.
MarkItDown-extracted text (a real text layer, no OCR involved) has nothing
for this to check against and is left alone -- see
pipeline/convert.py's handling of ConversionRequest.ai_review.
"""

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "qwen3-vl:4b-instruct"
OLLAMA_URL = "http://localhost:11434"

# The model is instructed to wrap anything it isn't sure about in this
# marker, keeping the OCR text unchanged inside it (per the "if uncertain,
# preserve the OCR text and flag the passage" requirement). We strip the
# marker back out before the text reaches the chunker/TTS -- flags are for
# the user's attention, not something that should be narrated aloud -- and
# surface the flagged passages separately.
_FLAG_PATTERN = re.compile(r"\{\{FLAG:(.*?)\}\}", re.DOTALL)

REVIEW_PROMPT = """Compare the OCR transcription below against the supplied page image. Correct only errors directly supported by the visible source. Never paraphrase, summarize, improve, modernize, or rewrite the author's prose. Preserve wording and structure exactly whenever possible. If uncertain, preserve the OCR text and flag the passage by wrapping exactly that passage in {{{{FLAG: ...}}}}.

Return ONLY the corrected text -- no preamble, no explanation, no markdown formatting.

OCR TRANSCRIPTION:
{text}"""


class AiReviewError(Exception):
    pass


@dataclass
class ReviewResult:
    text: str  # flag markers stripped, ready for cleaning/chunking/TTS
    flags: list[str] = field(default_factory=list)  # flagged passages, for the user


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


def review_page(image_path: Path, ocr_text: str, model: str = DEFAULT_MODEL, timeout: float = 180.0) -> ReviewResult:
    """Send a page image + its OCR text to a local vision-language Ollama
    model, asking it to correct only clear OCR errors. Raises AiReviewError
    if Ollama isn't reachable, the model isn't available, or the response is
    empty -- never silently returns the unreviewed text, and never falls
    back to a cloud API."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    payload = json.dumps({
        "model": model,
        "prompt": REVIEW_PROMPT.format(text=ocr_text),
        "images": [image_b64],
        "stream": False,
        # Ollama defaults to a 4096-token context window regardless of the
        # model's actual max -- a real page's image tokens + OCR text +
        # prompt overhead routinely exceeds that ("request (4121 tokens)
        # exceeds the available context size (4096 tokens)" on a genuinely
        # ordinary page). 16384 gives real headroom for a dense page.
        "options": {"num_ctx": 16384},
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
        # HTTPError is a URLError subclass but means the server WAS reached
        # and rejected the request -- handle before the URLError branch
        # below, or this gets mislabeled as "could not reach the server".
        body = e.read().decode("utf-8", errors="replace")
        raise AiReviewError(f"Ollama rejected the request (HTTP {e.code}): {body}") from e
    except urllib.error.URLError as e:
        raise AiReviewError(
            f"Could not reach local Ollama server at {OLLAMA_URL} ({e}). "
            f"Install Ollama and run `ollama pull {DEFAULT_MODEL}`, or turn AI review off."
        ) from e
    except TimeoutError as e:
        raise AiReviewError(f"Local model review timed out after {timeout}s.") from e

    if "error" in data:
        raise AiReviewError(f"Ollama error: {data['error']}")

    reviewed = data.get("response", "").strip()
    if not reviewed:
        raise AiReviewError("Ollama returned an empty response.")

    flags = [m.group(1).strip() for m in _FLAG_PATTERN.finditer(reviewed)]
    cleaned = _FLAG_PATTERN.sub(lambda m: m.group(1), reviewed).strip()
    return ReviewResult(text=cleaned, flags=flags)


def unload_model(model: str = DEFAULT_MODEL) -> None:
    """Ollama keeps a model resident on the GPU for a few minutes after use
    by default. On an 8GB card that collides with Chatterbox's own VRAM
    need right after review finishes (reproduced: Ollama holding 5.7GB
    caused Chatterbox to fail loading with a CUDA OOM). Call this once
    after the whole review pass, not per-page -- unloading per-page would
    force a costly reload before every single page instead of once at the
    end."""
    payload = json.dumps({"model": model, "prompt": "", "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=30).close()
    except Exception:
        pass  # best-effort -- don't fail the conversion over a GPU-memory-cleanup step
