"""Text-to-speech synthesis backed by edge-tts."""

from pathlib import Path

import edge_tts

DEFAULT_VOICE = "en-US-AriaNeural"


async def synthesize_chunk(text: str, out_path: Path, voice: str = DEFAULT_VOICE, rate: str = "+0%") -> None:
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(str(out_path))


async def list_voices() -> list[dict]:
    return await edge_tts.list_voices()
