"""book2audio: turn a book into a narrated .m4b audiobook.

    MarkItDown -> OpenOCR fallback -> Chatterbox Multilingual V3 -> FFmpeg -> .m4b
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from book2audio.chunker import DEFAULT_MAX_CHARS, chunk_text
from book2audio.clean import clean_text, split_chapters
from book2audio.extract import extract_markdown
from book2audio.mux import build_m4b

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def convert(
    input_path: Path = typer.Argument(..., exists=True, help="Book: .pdf, .epub, an image, or a directory of scanned pages."),
    output: Path = typer.Option(..., "-o", "--output", help="Output .m4b path."),
    voice: Optional[Path] = typer.Option(None, "--voice", help="Reference WAV/MP3 to clone as the narrator voice."),
    language: str = typer.Option("en", "--language", help="Language code for narration."),
    device: str = typer.Option("auto", "--device", help="cuda, mps, cpu, or auto."),
    max_chars: int = typer.Option(DEFAULT_MAX_CHARS, "--max-chars", help="Max characters per TTS chunk."),
    title: Optional[str] = typer.Option(None, "--title", help="Audiobook title (defaults to input filename)."),
    author: Optional[str] = typer.Option(None, "--author", help="Audiobook author metadata."),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Where synthesized chunk WAVs are cached (defaults next to output)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Extract, clean, and chunk only -- report counts, don't synthesize."),
):
    """Convert INPUT_PATH into a chaptered .m4b audiobook."""
    cache_dir = cache_dir or output.with_suffix("") .with_name(output.stem + "_cache")
    book_title = title or input_path.stem

    console.print(f"[bold]Extracting[/bold] {input_path} ...")
    raw = extract_markdown(input_path, ocr_output_dir=cache_dir / "ocr")
    if not raw.strip():
        raise typer.BadParameter("No text could be extracted from the input.")

    cleaned = clean_text(raw)
    chapters = split_chapters(cleaned)

    chapter_chunks = [(ch.title, chunk_text(ch.text, max_chars=max_chars)) for ch in chapters]
    total_chunks = sum(len(chunks) for _title, chunks in chapter_chunks)
    total_chars = sum(len(c) for _title, chunks in chapter_chunks for c in chunks)

    console.print(f"[bold]{len(chapters)}[/bold] chapter(s), [bold]{total_chunks}[/bold] chunk(s), [bold]{total_chars:,}[/bold] characters.")
    for ch_title, chunks in chapter_chunks:
        console.print(f"  - {ch_title}: {len(chunks)} chunks")

    if dry_run:
        console.print("[yellow]Dry run -- stopping before synthesis.[/yellow]")
        return

    from book2audio.synth import Narrator  # deferred: heavy torch/chatterbox import

    narrator = Narrator(device=device, audio_prompt_path=voice)
    console.print(f"[bold]Narrator ready[/bold] on device={narrator.device}")

    chunk_cache_dir = cache_dir / "chunks"
    chapter_wav_dir = cache_dir / "chapters"
    chapter_wav_dir.mkdir(parents=True, exist_ok=True)

    chapter_wavs: list[tuple[str, Path]] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesizing", total=total_chunks)

        for ch_index, (ch_title, chunks) in enumerate(chapter_chunks, start=1):
            chunk_wavs = []
            for chunk in chunks:
                wav_path = narrator.synth_chunk(chunk, language=language, cache_dir=chunk_cache_dir)
                chunk_wavs.append(wav_path)
                progress.advance(task)

            if not chunk_wavs:
                continue

            from book2audio.mux import concat_wavs

            chapter_wav = chapter_wav_dir / f"{ch_index:03d}.wav"
            concat_wavs(chunk_wavs, chapter_wav)
            chapter_wavs.append((ch_title, chapter_wav))

    console.print("[bold]Muxing[/bold] chapters into .m4b ...")
    build_m4b(chapter_wavs, output, title=book_title, author=author)
    console.print(f"[green]Done.[/green] Wrote {output}")


def main():
    app()


if __name__ == "__main__":
    main()
