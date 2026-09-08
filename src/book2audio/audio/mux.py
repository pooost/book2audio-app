"""Concatenate chunk/chapter WAVs and mux into a chaptered .m4b via ffmpeg."""

import subprocess
import tempfile
from pathlib import Path

from book2audio.core.ffmpeg_locate import find_ffmpeg, find_ffprobe


class MuxError(Exception):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise MuxError(f"{cmd[0]} failed:\n{e.stderr}") from e
    except FileNotFoundError as e:
        raise MuxError(
            f"{cmd[0]} not found. It should be bundled with this app; if you're running "
            "from source, install ffmpeg and make sure it's on PATH."
        ) from e


def concat_wavs(wav_paths: list[Path], out_path: Path) -> None:
    if not wav_paths:
        raise ValueError("no wav files to concatenate")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in wav_paths:
            # ffmpeg's concat-demuxer quoting: a literal single quote inside
            # the path must become '\'' (close quote, escaped quote, reopen).
            escaped = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        list_path = Path(f.name)

    try:
        ffmpeg = find_ffmpeg() or "ffmpeg"
        _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)])
    finally:
        list_path.unlink(missing_ok=True)


def get_duration_seconds(path: Path) -> float:
    ffprobe = find_ffprobe() or "ffprobe"
    result = _run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(result.stdout.strip())


def build_m4b(
    chapter_wavs: list[tuple[str, Path]],
    out_path: Path,
    title: str | None = None,
    author: str | None = None,
    bitrate: str = "64k",
) -> None:
    if not chapter_wavs:
        raise ValueError("no chapters to mux")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        master_wav = tmp / "master.wav"
        concat_wavs([wav for _title, wav in chapter_wavs], master_wav)

        metadata_path = tmp / "chapters.txt"
        lines = [";FFMETADATA1"]
        if title:
            lines.append(f"title={_escape(title)}")
        if author:
            lines.append(f"artist={_escape(author)}")
        lines.append("")

        cursor_ms = 0
        for chapter_title, wav in chapter_wavs:
            duration_ms = round(get_duration_seconds(wav) * 1000)
            lines += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={cursor_ms}",
                f"END={cursor_ms + duration_ms}",
                f"title={_escape(chapter_title)}",
                "",
            ]
            cursor_ms += duration_ms

        metadata_path.write_text("\n".join(lines), encoding="utf-8")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = find_ffmpeg() or "ffmpeg"
        _run([
            ffmpeg, "-y",
            "-i", str(master_wav),
            "-i", str(metadata_path),
            "-map_metadata", "1",
            "-map", "0:a",
            "-c:a", "aac",
            "-b:a", bitrate,
            str(out_path),
        ])


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ")
