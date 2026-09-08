"""book2audio: turn a book into a narrated .m4b audiobook.

    MarkItDown -> OpenOCR fallback -> Chatterbox Multilingual V3 -> FFmpeg -> .m4b

This is a thin frontend over book2audio.pipeline.convert -- the GUI
(book2audio.gui) calls the exact same functions. No conversion logic lives
here beyond translating flags into a ConversionRequest and rendering
progress events.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from book2audio.ingest.extract import ExtractionError
from book2audio.ingest.page_select import PageRangeError
from book2audio.processing.ai_review import AiReviewError
from book2audio.pipeline.convert import (
    ChunkSynthesisError,
    ConversionCancelled,
    ConversionRequest,
    ProgressEvent,
    plan_conversion,
    run_conversion,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def convert(
    input_path: Path = typer.Argument(..., exists=True, help="Book: .pdf, .epub, an image, or a directory of scanned pages."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output .m4b path. Defaults to INPUT_PATH with a .m4b extension."),
    voice: Optional[Path] = typer.Option(None, "--voice", exists=True, dir_okay=False, help="Reference WAV/MP3 to clone as the narrator voice."),
    language: str = typer.Option("en", "--language", help="Language code for narration."),
    device: str = typer.Option("auto", "--device", help="cuda, mps, cpu, or auto."),
    max_chars: int = typer.Option(300, "--max-chars", help="Max characters per TTS chunk."),
    title: Optional[str] = typer.Option(None, "--title", help="Audiobook title (defaults to input filename)."),
    author: Optional[str] = typer.Option(None, "--author", help="Audiobook author metadata."),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Where synthesized chunk WAVs are cached (defaults next to output)."),
    ocr_mode: str = typer.Option("auto", "--ocr-mode", help="auto (fall back to OCR only if the text layer looks bad), force (always OCR), or never."),
    preserve_chapters: bool = typer.Option(True, "--preserve-chapters/--no-preserve-chapters", help="Detect chapter headings, or treat the whole book as one chapter."),
    allow_download: bool = typer.Option(False, "--allow-download/--no-allow-download", help="Allow downloading a missing model during conversion (default: off -- run `book2audio setup-models` instead)."),
    bitrate: str = typer.Option("64k", "--bitrate", help="AAC bitrate for the output .m4b."),
    pages: Optional[str] = typer.Option(None, "--pages", help='PDF only. e.g. "1-10,15,20-25" (1-indexed, inclusive). Omit for the whole document.'),
    ai_review: bool = typer.Option(False, "--ai-review/--no-ai-review", help="Run extracted text through a local Ollama model to fix OCR errors before narration. Off by default; requires Ollama running locally."),
    ai_review_model: str = typer.Option("llama3.2", "--ai-review-model", help="Ollama model name to use for --ai-review."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Extract, clean, and chunk only -- report counts, don't synthesize."),
):
    """Convert INPUT_PATH into a chaptered .m4b audiobook."""
    if ocr_mode not in ("auto", "force", "never"):
        raise typer.BadParameter("--ocr-mode must be one of: auto, force, never")

    request = ConversionRequest(
        input_path=input_path,
        output_path=output or input_path.with_suffix(".m4b"),
        voice=voice,
        language=language,
        device=device,
        max_chars=max_chars,
        title=title,
        author=author,
        cache_dir=cache_dir,
        ocr_mode=ocr_mode,
        preserve_chapters=preserve_chapters,
        allow_download=allow_download,
        bitrate=bitrate,
        page_range=pages,
        ai_review=ai_review,
        ai_review_model=ai_review_model,
    )

    console.print(f"[bold]Extracting[/bold] {input_path} ...")

    def on_plan_progress(event: ProgressEvent) -> None:
        if event.stage == "ai_review":
            console.print(f"[bold]AI review[/bold] chapter {event.chapter_index}/{event.chapter_total}: {event.message}")

    try:
        plan, chapter_chunks = plan_conversion(request, on_progress=on_plan_progress)
    except (ValueError, ExtractionError, PageRangeError, AiReviewError) as e:
        raise typer.BadParameter(str(e)) from e

    console.print(f"[bold]{len(plan.chapter_titles)}[/bold] chapter(s), [bold]{plan.total_chunks}[/bold] chunk(s), [bold]{plan.total_chars:,}[/bold] characters.")
    for ch_title, count in zip(plan.chapter_titles, plan.chunks_per_chapter):
        console.print(f"  - {ch_title}: {count} chunks")

    if dry_run:
        console.print("[yellow]Dry run -- stopping before synthesis.[/yellow]")
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesizing", total=plan.total_chunks)
        narrator_announced = False

        def on_progress(event: ProgressEvent) -> None:
            nonlocal narrator_announced
            if event.stage == "tts" and not narrator_announced and event.device:
                console.print(f"[bold]Narrator ready[/bold] on device={event.device}")
                narrator_announced = True
            if event.stage == "tts" and event.chunk_index:
                progress.update(task, completed=event.chunk_index)
            if event.stage == "assembling":
                console.print("[bold]Muxing[/bold] chapters into .m4b ...")

        try:
            run_conversion(request, chapter_chunks=chapter_chunks, on_progress=on_progress)
        except ChunkSynthesisError as e:
            console.print(f"[red]Failed:[/red] {e}")
            console.print(f"[yellow]Chapter {e.failure.chapter_index} ({e.failure.chapter_title!r}), "
                           f"chunk {e.failure.chunk_index} failed. Earlier chunks are cached -- "
                           f"fix the issue and re-run the same command to resume.[/yellow]")
            raise typer.Exit(code=1) from e
        except ConversionCancelled as e:
            console.print(f"[yellow]{e}[/yellow]")
            raise typer.Exit(code=130) from e

    console.print(f"[green]Done.[/green] Wrote {request.output_path}")


@app.command()
def doctor():
    """Check the local environment: GPU, ffmpeg, models, offline state. Read-only."""
    from book2audio.core.doctor import run_doctor

    report = run_doctor()
    for check in report.checks:
        mark = "[green]OK[/green]  " if check.ok else "[red]MISSING[/red]"
        console.print(f"{mark} {check.name}: {check.detail}")

    if not report.all_ok:
        console.print("\n[yellow]Some checks failed. If models are missing, run `book2audio setup-models`.[/yellow]")
        raise typer.Exit(code=1)


@app.command("setup-models")
def setup_models_cmd():
    """Explicitly download Chatterbox + OpenOCR model weights. Never run implicitly."""
    from book2audio.core.models import setup_models

    def report(msg: str) -> None:
        console.print(f"[bold]{msg}[/bold]")

    setup_models(progress=report)
    console.print("[green]Done.[/green] Models are cached for offline use.")


def main():
    # Make `convert` the implicit default subcommand so `book2audio book.pdf`
    # works without spelling out `convert`, even though `doctor` and
    # `setup-models` also exist as named subcommands.
    import sys

    args = sys.argv[1:]
    subcommands = {"convert", "doctor", "setup-models"}
    passthrough = {"--help", "--install-completion", "--show-completion"}
    if args and not (subcommands & set(args)) and args[0] not in passthrough:
        sys.argv.insert(1, "convert")
    app()


if __name__ == "__main__":
    main()
