"""Chatterbox Multilingual V3: the higher-quality, voice-cloning-capable
local TTS backend. See tts/provider.py for the shared interface.
"""

import os
from pathlib import Path

from book2audio.core.device import pick_device
from book2audio.tts.provider import ModelMissingError, TTSProvider, compute_cache_key

ENGINE_ID = "chatterbox"
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


def supported_languages() -> dict[str, str]:
    from chatterbox import SUPPORTED_LANGUAGES

    return dict(SUPPORTED_LANGUAGES)


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


class ChatterboxProvider(TTSProvider):
    engine_id = ENGINE_ID
    engine_name = ENGINE_NAME

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

    def synth_chunk(self, text: str, language: str, cache_dir: Path) -> tuple[Path, bool]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = compute_cache_key(self.engine_id, "v3", self.audio_prompt_path or "default", language, text)
        out_path = cache_dir / f"{key}.wav"

        if out_path.exists():
            return out_path, True

        kwargs = {"language_id": language}
        if self.audio_prompt_path:
            kwargs["audio_prompt_path"] = self.audio_prompt_path

        audio = self.model.generate(text, **kwargs)
        # torchaudio.save isn't atomic -- write to a sibling temp file and
        # rename, so a killed/crashed run never leaves a partial .wav that a
        # resumed run would mistake for a finished, cached chunk. The temp
        # file's own extension must be .wav: torchaudio's soundfile backend
        # derives format purely from splitting the path on "." and taking
        # the last part (format= is only honored for file-like objects).
        import torchaudio

        tmp_path = out_path.with_name(out_path.stem + ".tmp.wav")
        torchaudio.save(str(tmp_path), audio, self.sample_rate)
        os.replace(tmp_path, out_path)
        return out_path, False

    def unload(self) -> None:
        del self.model
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()
