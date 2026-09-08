"""System diagnostics: what's installed, what's cached, what's missing.

Read-only -- never downloads anything. Shared by `book2audio doctor` and the
GUI's System/Diagnostics screen so the two can't drift.
"""

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    optional: bool = False  # excluded from all_ok -- e.g. a feature that's off by default


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks if not c.optional)


def run_doctor() -> DoctorReport:
    checks: list[CheckResult] = []

    checks.append(CheckResult("Operating system", True, f"{platform.system()} {platform.release()} ({platform.machine()})"))
    checks.append(CheckResult("Python", True, sys.version.split()[0]))

    ffmpeg_path = shutil.which("ffmpeg")
    checks.append(_bin_check("FFmpeg", ffmpeg_path))
    ffprobe_path = shutil.which("ffprobe")
    checks.append(_bin_check("FFprobe", ffprobe_path))

    checks.append(_markitdown_check())
    checks.append(_device_check())
    checks += _chatterbox_checks()
    checks += _openocr_checks()
    checks.append(_ollama_check())

    offline = os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False")
    checks.append(CheckResult("Offline mode", offline, "on (default)" if offline else "off"))

    return DoctorReport(checks=checks)


def _bin_check(name: str, path: str | None) -> CheckResult:
    if not path:
        return CheckResult(name, False, "not found on PATH")
    try:
        out = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5)
        first_line = out.stdout.splitlines()[0] if out.stdout else path
        return CheckResult(name, True, first_line)
    except Exception:
        return CheckResult(name, True, path)


def _markitdown_check() -> CheckResult:
    try:
        import markitdown  # noqa: F401

        return CheckResult("MarkItDown", True, "importable")
    except ImportError as e:
        return CheckResult("MarkItDown", False, str(e))


def _device_check() -> CheckResult:
    from book2audio.core.device import probe_devices

    info = probe_devices()
    if info.cuda_available:
        detail = f"CUDA: {info.cuda_device_name} ({info.cuda_vram_gb} GB VRAM)"
    elif info.mps_available:
        detail = "Apple MPS available"
    else:
        detail = "CPU only (no CUDA/MPS detected)"
    return CheckResult("Compute device", info.cuda_available or info.mps_available, detail)


def _chatterbox_checks() -> list[CheckResult]:
    from book2audio.tts.chatterbox_backend import REPO_ID, is_model_cached

    cached = is_model_cached()
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    detail = f"cached at {hf_cache}" if cached else f"not fully cached (repo: {REPO_ID}) -- run `book2audio setup-models`"
    return [CheckResult("Chatterbox Multilingual V3", cached, detail)]


def _openocr_checks() -> list[CheckResult]:
    from book2audio.ocr.backend import is_model_cached, is_openocr_available, openocr_cache_dir

    available = is_openocr_available()
    checks = [CheckResult("OpenOCR binary", available, "installed" if available else "not found")]

    cached = is_model_cached()
    detail = f"cached at {openocr_cache_dir()}" if cached else "not cached -- run `book2audio setup-models`"
    checks.append(CheckResult("OpenOCR models", cached, detail))
    return checks


def _ollama_check() -> CheckResult:
    from book2audio.processing.ai_review import is_ollama_available, available_models

    available = is_ollama_available()
    if not available:
        return CheckResult("Ollama (optional, for --ai-review)", False, "not running -- only needed if you use --ai-review", optional=True)
    models = available_models()
    detail = f"running, models: {', '.join(models)}" if models else "running, but no models pulled yet (`ollama pull llama3.2`)"
    return CheckResult("Ollama (optional, for --ai-review)", bool(models), detail, optional=True)
