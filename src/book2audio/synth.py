"""Chatterbox Multilingual V3 wrapper with on-disk per-chunk caching.

A book is thousands of chunks; caching by content hash means a crashed or
interrupted run resumes for free, and re-running with the same voice/text
doesn't re-synthesize anything.
"""

import hashlib
from pathlib import Path

import torch
import torchaudio

DEFAULT_VOICE_LANGUAGE = "en"


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Narrator:
    def __init__(self, device: str = "auto", audio_prompt_path: Path | None = None):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.device = pick_device(device)
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device, t3_model="v3")
        self.audio_prompt_path = str(audio_prompt_path) if audio_prompt_path else None

    @property
    def sample_rate(self) -> int:
        return self.model.sr

    def synth_chunk(self, text: str, language: str, cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(text, language)
        out_path = cache_dir / f"{key}.wav"

        if out_path.exists():
            return out_path

        kwargs = {"language_id": language}
        if self.audio_prompt_path:
            kwargs["audio_prompt_path"] = self.audio_prompt_path

        audio = self.model.generate(text, **kwargs)
        torchaudio.save(str(out_path), audio, self.sample_rate)
        return out_path

    def _cache_key(self, text: str, language: str) -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(language.encode("utf-8"))
        h.update((self.audio_prompt_path or "").encode("utf-8"))
        h.update(b"v3")
        return h.hexdigest()[:24]
