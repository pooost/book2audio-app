"""Explicit, user-requested model downloads -- never triggered implicitly."""

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory


def setup_models(progress: Callable[[str], None] | None = None) -> None:
    """Download Chatterbox and OpenOCR's model weights. Bypasses offline mode
    for exactly this operation. Meant to be run explicitly (`book2audio
    setup-models`), never as a side effect of a normal conversion."""
    report = progress or (lambda _msg: None)

    report("Downloading Chatterbox Multilingual V3 weights...")
    from book2audio.tts.chatterbox_backend import download_model

    download_model()
    report("Chatterbox weights ready.")

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
