"""Shared interface for local TTS backends.

The chunker/pipeline calls a TTSProvider and gets back a wav path; it
never needs to know whether Chatterbox or Kokoro produced it. Every
provider shares the same cache-key scheme (compute_cache_key) so cache
identity always includes provider + model/voice + language + text --
switching backends can never accidentally reuse incompatible audio.
"""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path


class ModelMissingError(Exception):
    """Required model files aren't present locally, and downloading them
    automatically is not allowed (offline mode, or no --allow-download).
    Every provider raises this instead of silently downloading anything
    during a normal conversion."""


def compute_cache_key(engine_id: str, model_id: str, voice_id: str, language: str, text: str) -> str:
    h = hashlib.sha256()
    for part in (engine_id, model_id, voice_id or "", language, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


class TTSProvider(ABC):
    engine_id: str  # stable short id used in cache keys and CLI --tts values
    engine_name: str  # human-readable
    device: str

    @abstractmethod
    def synth_chunk(self, text: str, language: str, cache_dir: Path) -> tuple[Path, bool]:
        """Return (wav_path, was_cache_hit). Must be safe to call repeatedly
        with the same arguments -- cache hits should be near-instant. The
        was_cache_hit flag lets callers exclude cache hits from real-time-
        factor timing (a cache hit's near-zero wall time would otherwise
        make RTF look artificially fast on a resumed run)."""

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    def unload(self) -> None:
        """Release GPU/CPU resources held by this provider. Default is a
        no-op; providers that share a GPU with other local models (Qwen,
        or the other TTS backend) should override this to actually free
        VRAM -- see kokoro_backend.py's unload() for why this matters in
        practice, not just in theory."""
