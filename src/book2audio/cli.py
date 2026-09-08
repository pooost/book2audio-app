"""book2audio command-line interface."""

import asyncio
import tempfile
from pathlib import Path

import click
from pydub import AudioSegment

from book2audio.chunk import chunk_text
from book2audio.extract import extract_text
from book2audio.tts import DEFAULT_VOICE, list_voices, synthesize_chunk


@click.group()
def main():
    """Convert books and text files to audio."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), required=True, help="Output audio file (.mp3)")
@click.option("--voice", default=DEFAULT_VOICE, show_default=True, help="edge-tts voice name")
@click.option("--rate", default="+0%", show_default=True, help="Speech rate adjustment, e.g. +10%, -15%")
@click.option("--max-chars", default=3000, show_default=True, help="Max characters per TTS chunk")
def convert(input_path: Path, output_path: Path, voice: str, rate: str, max_chars: int):
    """Convert INPUT_PATH (.txt, .epub, .pdf) to an audio file."""
    click.echo(f"Extracting text from {input_path}...")
    text = extract_text(input_path)
    if not text.strip():
        raise click.ClickException("No text could be extracted from the input file.")

    chunks = chunk_text(text, max_chars=max_chars)
    click.echo(f"Split into {len(chunks)} chunk(s). Synthesizing with voice '{voice}'...")

    asyncio.run(_convert_chunks(chunks, output_path, voice, rate))
    click.echo(f"Done. Wrote {output_path}")


async def _convert_chunks(chunks: list[str], output_path: Path, voice: str, rate: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths = [Path(tmpdir) / f"chunk_{i:04d}.mp3" for i in range(len(chunks))]

        with click.progressbar(range(len(chunks)), label="Synthesizing") as bar:
            for i in bar:
                await synthesize_chunk(chunks[i], tmp_paths[i], voice=voice, rate=rate)

        click.echo("Merging audio...")
        combined = AudioSegment.empty()
        for path in tmp_paths:
            combined += AudioSegment.from_mp3(path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(output_path, format="mp3")


@main.command("list-voices")
@click.option("--lang", default=None, help="Filter voices by language prefix, e.g. en, es, zh")
def list_voices_cmd(lang: str | None):
    """List available edge-tts voices."""
    voices = asyncio.run(list_voices())
    if lang:
        voices = [v for v in voices if v["ShortName"].lower().startswith(lang.lower())]
    for v in voices:
        click.echo(f"{v['ShortName']:<25} {v['Gender']:<8} {v['Locale']}")


if __name__ == "__main__":
    main()
