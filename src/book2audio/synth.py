"""Chatterbox Multilingual V3 wrapper with on-disk per-chunk caching.

A book is thousands of chunks; caching by content hash means a crashed or
interrupted run resumes for free, and re-running with the same voice/text
doesn't re-synthesize anything.
"""

import hashlib
import os
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
        self.model = self._load_model(ChatterboxMultilingualTTS)
        self.audio_prompt_path = str(audio_prompt_path) if audio_prompt_path else None

    def _load_model(self, model_cls):
        """Try the local cache first; only touch the network if a file is
        genuinely missing (offline mode freezes HF_HUB_OFFLINE as a module
        constant on huggingface_hub import, so env vars alone can't flip it
        mid-process -- patch the constant directly instead)."""
        from huggingface_hub import constants as hf_constants
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return model_cls.from_pretrained(device=self.device, t3_model="v3")
        except LocalEntryNotFoundError:
            print("Model cache incomplete -- downloading missing files (one-time)...")
            hf_constants.HF_HUB_OFFLINE = False
            try:
                return model_cls.from_pretrained(device=self.device, t3_model="v3")
            finally:
                hf_constants.HF_HUB_OFFLINE = True

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
        # torchaudio.save isn't atomic -- write to a sibling temp file and
        # rename, so a killed/crashed run never leaves a partial .wav that a
        # resumed run would mistake for a finished, cached chunk.
        tmp_path = out_path.with_suffix(".wav.tmp")
        torchaudio.save(str(tmp_path), audio, self.sample_rate)
        os.replace(tmp_path, out_path)
        return out_path

    def _cache_key(self, text: str, language: str) -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(language.encode("utf-8"))
        h.update((self.audio_prompt_path or "").encode("utf-8"))
        h.update(b"v3")
        return h.hexdigest()[:24]
