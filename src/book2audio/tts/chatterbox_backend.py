"""Chatterbox Multilingual V3 wrapper with on-disk per-chunk caching.

A book is thousands of chunks; caching by content hash means a crashed or
interrupted run resumes for free, and re-running with the same voice/text
doesn't re-synthesize anything.
"""

import hashlib
import os
from pathlib import Path

from book2audio.core.device import pick_device

ENGINE_NAME = "Chatterbox Multilingual V3"
REPO_ID = "ResembleAI/chatterbox"
REQUIRED_FILES = ["ve.pt", "t3_mtl23ls_v3.safetensors", "s3gen.pt", "conds.pt"]


def is_model_cached() -> bool:
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    for filename in REQUIRED_FILES:
        result = try_to_load_from_cache(repo_id=REPO_ID, filename=filename)
        if result is None or result is _CACHED_NO_EXIST:
            return False
    return True


def download_model() -> None:
    """Explicit, user-requested download -- bypasses offline-first behavior."""
    from huggingface_hub import constants as hf_constants

    previous = hf_constants.HF_HUB_OFFLINE
    hf_constants.HF_HUB_OFFLINE = False
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        ChatterboxMultilingualTTS.from_pretrained(device="cpu", t3_model="v3")
    finally:
        hf_constants.HF_HUB_OFFLINE = previous


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
        except LocalEntryNotFoundError as e:
            if not hf_constants.HF_HUB_OFFLINE:
                raise
            raise ModelMissingError(
                "Chatterbox model files aren't fully cached locally, and offline mode "
                "is on. Run `book2audio setup-models` to download them explicitly."
            ) from e

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
        import torchaudio

        # The temp file's own extension must be .wav: torchaudio's soundfile
        # backend derives format purely from splitting the path on "." and
        # taking the last part (`format=` is only honored for file-like
        # objects, not string paths) -- a ".tmp" suffix fails with
        # "Unsupported format: tmp" no matter what `format=` is passed.
        tmp_path = out_path.with_name(out_path.stem + ".tmp.wav")
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


class ModelMissingError(Exception):
    pass
