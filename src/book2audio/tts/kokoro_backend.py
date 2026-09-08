"""Kokoro: the fast, lightweight local TTS backend (82M params, ~327MB core
model vs. Chatterbox's ~3GB). Currently English-only in book2audio -- see
supported_languages().

Kokoro's English g2p (via misaki) needs a spaCy model (en_core_web_sm) it
will silently pip-install on first use if missing (misaki/en.py: `if not
spacy.util.is_package(name): spacy.cli.download(name)`) -- confirmed by
reproducing it live. That's a second, separate download path from the HF
model weights, and it would violate "no silent downloads during normal
conversion" if left unchecked, so is_model_cached() verifies both, and
download_model() fetches both explicitly.
"""

import os
from pathlib import Path

from book2audio.core.device import pick_device
from book2audio.tts.provider import ModelMissingError, TTSProvider, compute_cache_key

ENGINE_ID = "kokoro"
ENGINE_NAME = "Kokoro"
REPO_ID = "hexgrad/Kokoro-82M"
SPACY_MODEL = "en_core_web_sm"
MODEL_VERSION = "v1_0"

DEFAULT_VOICE = "af_heart"
# A curated subset of the repo's 54 voices -- small footprint (~500KB
# each), diverse enough to be useful, not so many the GUI selector becomes
# unwieldy. More can be added later; nothing about the architecture limits
# it to this list.
VOICES = {
    "af_heart": "Heart (American, female)",
    "af_bella": "Bella (American, female)",
    "af_nova": "Nova (American, female)",
    "am_michael": "Michael (American, male)",
    "am_adam": "Adam (American, male)",
    "bf_emma": "Emma (British, female)",
}


def supported_languages() -> dict[str, str]:
    return {"en": "English"}


def _lang_code_for_voice(voice: str) -> str:
    return "b" if voice.startswith("b") else "a"


def is_spacy_model_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec(SPACY_MODEL) is not None


def is_model_cached(voice: str = DEFAULT_VOICE) -> bool:
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    for filename in ("kokoro-v1_0.pth", "config.json", f"voices/{voice}.pt"):
        result = try_to_load_from_cache(repo_id=REPO_ID, filename=filename)
        if result is None or result is _CACHED_NO_EXIST:
            return False
    return is_spacy_model_installed()


def download_model(voices: list[str] | None = None) -> None:
    """Explicit, user-requested download -- bypasses offline-first behavior.
    Fetches the core model, the given voices (default: the curated set
    above), and the spaCy English model misaki needs."""
    from huggingface_hub import constants as hf_constants

    previous = hf_constants.HF_HUB_OFFLINE
    hf_constants.HF_HUB_OFFLINE = False
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=REPO_ID, filename="kokoro-v1_0.pth")
        hf_hub_download(repo_id=REPO_ID, filename="config.json")
        for v in (voices if voices is not None else list(VOICES)):
            hf_hub_download(repo_id=REPO_ID, filename=f"voices/{v}.pt")
    finally:
        hf_constants.HF_HUB_OFFLINE = previous

    if not is_spacy_model_installed():
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL], check=True)


class KokoroProvider(TTSProvider):
    engine_id = ENGINE_ID
    engine_name = ENGINE_NAME

    def __init__(self, device: str = "auto", voice: str = DEFAULT_VOICE):
        self.device = pick_device(device)
        self.voice = voice

        if not is_model_cached(voice):
            raise ModelMissingError(
                f"Kokoro isn't fully set up (voice {voice!r} or the core model or the "
                f"spaCy English model is missing). Run `book2audio setup-models --tts kokoro` "
                "to download everything explicitly -- normal conversion never downloads "
                "anything on its own."
            )

        self._pipeline = self._load_pipeline(_lang_code_for_voice(voice))

    def _load_pipeline(self, lang_code: str):
        from huggingface_hub import constants as hf_constants
        from huggingface_hub.errors import LocalEntryNotFoundError
        from kokoro import KPipeline

        try:
            return KPipeline(lang_code=lang_code, repo_id=REPO_ID, device=self.device)
        except LocalEntryNotFoundError as e:
            if not hf_constants.HF_HUB_OFFLINE:
                raise
            raise ModelMissingError(
                "Kokoro model files aren't fully cached locally, and offline mode is on. "
                "Run `book2audio setup-models --tts kokoro` to download them explicitly."
            ) from e

    @property
    def sample_rate(self) -> int:
        return 24000

    def synth_chunk(self, text: str, language: str, cache_dir: Path) -> tuple[Path, bool]:
        if language != "en":
            raise ValueError(
                f"Kokoro support in book2audio is currently English-only (got language={language!r})."
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        key = compute_cache_key(self.engine_id, MODEL_VERSION, self.voice, language, text)
        out_path = cache_dir / f"{key}.wav"
        if out_path.exists():
            return out_path, True

        import numpy as np
        import soundfile as sf

        pieces = []
        for result in self._pipeline(text, voice=self.voice):
            audio = result.audio
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            pieces.append(audio)
        audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

        # Same atomic-write requirement as Chatterbox's provider: the temp
        # file's own extension must be .wav (soundfile derives format from
        # the path, not a format= kwarg).
        tmp_path = out_path.with_name(out_path.stem + ".tmp.wav")
        sf.write(str(tmp_path), audio, self.sample_rate)
        os.replace(tmp_path, out_path)
        return out_path, False

    def unload(self) -> None:
        self._pipeline = None
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()
