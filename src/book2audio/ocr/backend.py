"""OpenOCR subprocess wrapper: resolve the binary, run it, read back its Markdown."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class OcrError(Exception):
    pass


def find_openocr() -> str:
    # Console scripts land next to the interpreter that installed them.
    # Resolving via sys.executable means this works whether or not the venv
    # is actually activated (running `.venv/bin/book2audio` directly, e.g.,
    # doesn't put `.venv/bin` on PATH) -- shutil.which alone would silently
    # fail to find an openocr that is, in fact, right there and installed.
    venv_local = Path(sys.executable).parent / "openocr"
    if venv_local.exists():
        return str(venv_local)
    found = shutil.which("openocr")
    if found:
        return found
    raise OcrError("openocr is not on PATH; install it with the rest of the stack.")


def is_openocr_available() -> bool:
    try:
        find_openocr()
        return True
    except OcrError:
        return False


def openocr_cache_dir() -> Path:
    return Path.home() / ".cache" / "openocr"


def is_model_cached() -> bool:
    cache = openocr_cache_dir()
    return cache.exists() and any(cache.rglob("*.onnx"))


def _run_openocr_cmd(input_path: Path, output_dir: Path, allow_download: bool) -> None:
    openocr_bin = find_openocr()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        openocr_bin,
        "--task", "doc",
        "--input_path", str(input_path),
        "--output_path", str(output_dir),
        "--use_layout_detection",
        "--save_markdown",
    ]

    # Offline-first: only let openocr hit the network if its local model
    # cache is actually missing something (and only if the caller allows it
    # -- a "fully offline" toggle should fail loudly instead of downloading).
    try:
        subprocess.run(base_cmd + ["--no_auto_download"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as first_error:
        if not allow_download:
            raise OcrError(
                f"openocr failed on {input_path} and offline mode forbids downloading "
                f"missing models:\n{first_error.stderr}"
            ) from first_error
        try:
            subprocess.run(base_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise OcrError(f"openocr failed on {input_path}:\n{e.stderr}") from e


def run_openocr(input_path: Path, output_dir: Path | None = None, allow_download: bool = True) -> str:
    if output_dir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            return _run_merged(input_path, Path(tmpdir), allow_download)
    return _run_merged(input_path, output_dir, allow_download)


def _run_merged(input_path: Path, output_dir: Path, allow_download: bool) -> str:
    _run_openocr_cmd(input_path, output_dir, allow_download)
    md_files = sorted(output_dir.rglob("*.md"))
    if not md_files:
        raise OcrError(f"openocr produced no markdown output for {input_path} in {output_dir}")
    return "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in md_files)
