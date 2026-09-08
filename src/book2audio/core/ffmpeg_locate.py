"""Locate ffmpeg/ffprobe.

A packaged Windows/Mac build ships its own ffmpeg/ffprobe binaries next to
the app (in a `ffmpeg-bin/` folder alongside the executable) so a friend who
just unzipped the app doesn't need to separately install ffmpeg. A source/dev
install has no such folder and falls back to whatever's on PATH.
"""

import shutil
import sys
from pathlib import Path


def _bundled_dir() -> Path | None:
    # PyInstaller sets sys.frozen=True; for a onedir build, the executable's
    # own directory is what the build step ships ffmpeg-bin/ alongside.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "ffmpeg-bin"
    return None


def _exe_name(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def find_ffmpeg() -> str | None:
    bundled = _bundled_dir()
    if bundled is not None:
        candidate = bundled / _exe_name("ffmpeg")
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffmpeg")


def find_ffprobe() -> str | None:
    bundled = _bundled_dir()
    if bundled is not None:
        candidate = bundled / _exe_name("ffprobe")
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffprobe")
