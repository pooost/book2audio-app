"""Concatenate chunk/chapter WAVs and mux into a chaptered .m4b via ffmpeg."""

import subprocess
import tempfile
from pathlib import Path


def concat_wavs(wav_paths: list[Path], out_path: Path) -> None:
    if not wav_paths:
        raise ValueError("no wav files to concatenate")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in wav_paths:
            f.write(f"file '{p.resolve()}'\n")
        list_path = Path(f.name)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def get_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
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
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(master_wav),
                "-i", str(metadata_path),
                "-map_metadata", "1",
                "-map", "0:a",
                "-c:a", "aac",
                "-b:a", bitrate,
                str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ")
