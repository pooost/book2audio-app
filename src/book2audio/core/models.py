"""Explicit, user-requested model downloads -- never triggered implicitly."""

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

TTS_CHOICES = ("chatterbox", "kokoro", "all")


def setup_models(progress: Callable[[str], None] | None = None, tts: str = "chatterbox") -> None:
    """Download TTS and OpenOCR model weights. Bypasses offline mode for
    exactly this operation. Meant to be run explicitly (`book2audio
    setup-models`), never as a side effect of a normal conversion.

    tts: "chatterbox" (default -- unchanged from before Kokoro existed),
    "kokoro", or "all"."""
    if tts not in TTS_CHOICES:
        raise ValueError(f"tts must be one of {TTS_CHOICES}, got {tts!r}")

    report = progress or (lambda _msg: None)

    if tts in ("chatterbox", "all"):
        report("Downloading Chatterbox Multilingual V3 weights...")
        from book2audio.tts.chatterbox_backend import download_model as download_chatterbox

        download_chatterbox()
        report("Chatterbox weights ready.")

    if tts in ("kokoro", "all"):
        report("Downloading Kokoro weights, voices, and the spaCy English model...")
        from book2audio.tts.kokoro_backend import download_model as download_kokoro

        download_kokoro()
        report("Kokoro ready.")

    report("Downloading OpenOCR model weights...")
    _download_openocr_models()
    report("OpenOCR weights ready.")


def _download_openocr_models() -> None:
    from PIL import Image

    from book2audio.ocr.backend import run_openocr

    with TemporaryDirectory() as tmpdir:
        probe = Path(tmpdir) / "probe.png"
        Image.new("RGB", (64, 64), "white").save(probe)
        run_openocr(probe, Path(tmpdir) / "out", allow_download=True)
