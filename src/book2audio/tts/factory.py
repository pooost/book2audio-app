"""Picks the right TTSProvider for a request. The rest of the pipeline
never imports chatterbox_backend/kokoro_backend directly."""

from pathlib import Path

from book2audio.tts.provider import TTSProvider

TTS_BACKENDS = {"chatterbox": "Chatterbox Multilingual V3", "kokoro": "Kokoro"}


def create_provider(tts_backend: str, device: str, voice_reference: Path | None, kokoro_voice: str) -> TTSProvider:
    if tts_backend == "chatterbox":
        from book2audio.tts.chatterbox_backend import ChatterboxProvider

        return ChatterboxProvider(device=device, audio_prompt_path=voice_reference)
    if tts_backend == "kokoro":
        from book2audio.tts.kokoro_backend import KokoroProvider

        return KokoroProvider(device=device, voice=kokoro_voice)
    raise ValueError(f"Unknown TTS backend {tts_backend!r} -- must be one of {list(TTS_BACKENDS)}")
