"""book2audio: turn a book into a narrated .m4b audiobook.

    MarkItDown -> OpenOCR fallback -> cleanup -> Qwen review -> chunk ->
    Chatterbox/Kokoro -> FFmpeg -> .m4b

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
from book2audio.pipeline.convert import (
    ChunkSynthesisError,
    ConversionCancelled,
    ConversionRequest,
    ProgressEvent,
    default_output_stem,
    plan_conversion,
    run_conversion,
)
from book2audio.processing.ai_review import AiReviewError
from book2audio.tts.factory import TTS_BACKENDS
from book2audio.tts.provider import ModelMissingError

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def convert(
    input_path: Path = typer.Argument(..., exists=True, help="Book: .pdf, .epub, an image, or a directory of scanned pages."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output .m4b path. Defaults to INPUT_PATH with a .m4b extension (page-range-qualified if --pages is set)."),
    tts: str = typer.Option("chatterbox", "--tts", help=f"Local TTS backend: {' or '.join(TTS_BACKENDS)}."),
    voice: Optional[Path] = typer.Option(None, "--voice", exists=True, dir_okay=False, help="Chatterbox only: reference WAV/MP3 to clone as the narrator voice."),
    kokoro_voice: str = typer.Option("af_heart", "--kokoro-voice", help="Kokoro only: named voice (see `book2audio doctor` for what's set up)."),
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
    ai_review: bool = typer.Option(False, "--ai-review/--no-ai-review", help="Run extracted text through a local Ollama vision model to fix OCR errors before narration. Off by default; requires Ollama running locally."),
    ai_review_model: str = typer.Option("qwen3-vl:4b-instruct", "--ai-review-model", help="Vision-capable Ollama model to use for --ai-review (compares OCR text against the page image)."),
    save_text_outputs: bool = typer.Option(True, "--save-text-outputs/--no-save-text-outputs", help="Write <output>.raw.md / .cleaned.md / .reviewed.md alongside the .m4b."),
    extract_only: bool = typer.Option(False, "--extract-only", help="Extract, clean, (optionally) review, and save readable Markdown -- skip TTS/assembly entirely."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Pure preview: report chapter/chunk counts, don't synthesize, don't save text files. For the latter, use --extract-only instead."),
):
    """Convert INPUT_PATH into a chaptered .m4b audiobook."""
    if ocr_mode not in ("auto", "force", "never"):
        raise typer.BadParameter("--ocr-mode must be one of: auto, force, never")
    if tts not in TTS_BACKENDS:
        raise typer.BadParameter(f"--tts must be one of: {', '.join(TTS_BACKENDS)}")

    output_path = output or input_path.with_name(default_output_stem(input_path, pages) + ".m4b")

    request = ConversionRequest(
        input_path=input_path,
        output_path=output_path,
        voice=voice,
        kokoro_voice=kokoro_voice,
        tts_backend=tts,
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
        save_text_outputs=save_text_outputs,
        extract_only=extract_only,
    )

    # --dry-run is a pure preview: no filesystem side effects beyond
    # cache/bookkeeping. --extract-only is the actual "give me the text"
    # operation and does write .md files. Without this, both looked nearly
    # identical (both extracted and saved text, differing only in the
    # printed message) -- not what a distinct --extract-only flag implies.
    if dry_run and not extract_only:
        request.save_text_outputs = False

    console.print(f"[bold]Extracting[/bold] {input_path} ...")

    def on_plan_progress(event: ProgressEvent) -> None:
        if event.stage == "ai_review" and event.page_total:
            console.print(f"[bold]AI review[/bold] (vision model vs. page image) page {event.page_index}/{event.page_total}")
        if event.stage == "extracting" and event.message:
            console.print(event.message)

    try:
        plan, chapter_chunks = plan_conversion(request, on_progress=on_plan_progress)
    except (ValueError, ExtractionError, PageRangeError, AiReviewError) as e:
        raise typer.BadParameter(str(e)) from e

    if plan.extraction_reused_cache:
        console.print("[cyan]Reused cached extraction[/cyan] (input and settings unchanged since last run).")

    if ai_review:
        if plan.ai_review_applied:
            console.print(f"[bold]AI review[/bold] complete, {len(plan.review_flags)} passage(s) flagged as uncertain.")
            for flag in plan.review_flags:
                console.print(f"  [yellow]! {flag}[/yellow]")
        elif plan.ai_review_skip_reason:
            console.print(f"[yellow]AI review skipped:[/yellow] {plan.ai_review_skip_reason}")

    if plan.text_outputs_saved:
        console.print(f"[bold]Saved readable text:[/bold] {', '.join(p.name for p in plan.text_outputs_saved)}")

    console.print(f"[bold]{len(plan.chapter_titles)}[/bold] chapter(s), [bold]{plan.total_chunks}[/bold] chunk(s), [bold]{plan.total_chars:,}[/bold] characters.")
    for ch_title, count in zip(plan.chapter_titles, plan.chunks_per_chapter):
        console.print(f"  - {ch_title}: {count} chunks")

    if extract_only:
        console.print("[green]Extraction complete.[/green] Skipping TTS (--extract-only).")
        return

    if dry_run:
        console.print("[yellow]Dry run -- stopping before synthesis.[/yellow]")
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[rtf]}"),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesizing", total=plan.total_chunks, rtf="")
        narrator_announced = False

        def on_progress(event: ProgressEvent) -> None:
            nonlocal narrator_announced
            if event.stage == "tts" and not narrator_announced and event.device:
                console.print(f"[bold]Narrator ready[/bold] backend={event.tts_backend} device={event.device}")
                narrator_announced = True
            if event.stage == "tts" and event.chunk_index:
                rtf_str = f"RTF {event.rtf:.2f}" if event.rtf is not None else ""
                progress.update(task, completed=event.chunk_index, rtf=rtf_str)
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
        except ModelMissingError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from e

    console.print(f"[green]Done.[/green] Wrote {request.output_path}")


@app.command()
def doctor():
    """Check the local environment: GPU, ffmpeg, models, offline state. Read-only."""
    from book2audio.core.doctor import run_doctor

    report = run_doctor()
    for category, label in (("system", "SYSTEM"), ("text", "TEXT"), ("tts", "TTS")):
        console.print(f"\n[bold underline]{label}[/bold underline]")
        for check in report.by_category(category):
            mark = "[green]OK[/green]  " if check.ok else ("[yellow]OPT[/yellow]  " if check.optional else "[red]MISSING[/red]")
            console.print(f"{mark} {check.name}: {check.detail}")

    if not report.all_ok:
        console.print("\n[yellow]Some checks failed. If models are missing, run `book2audio setup-models`.[/yellow]")
        raise typer.Exit(code=1)


@app.command("setup-models")
def setup_models_cmd(
    tts: str = typer.Option("chatterbox", "--tts", help=f"Which TTS backend to set up: {', '.join(TTS_BACKENDS)}, or 'all'."),
):
    """Explicitly download TTS + OpenOCR model weights. Never run implicitly."""
    from book2audio.core.models import TTS_CHOICES, setup_models

    if tts not in TTS_CHOICES:
        raise typer.BadParameter(f"--tts must be one of: {', '.join(TTS_CHOICES)}")

    def report(msg: str) -> None:
        console.print(f"[bold]{msg}[/bold]")

    setup_models(progress=report, tts=tts)
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
