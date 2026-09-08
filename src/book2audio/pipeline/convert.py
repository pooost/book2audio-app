"""The single conversion pipeline: extract -> clean -> chunk -> narrate -> mux.

Both the CLI (`book2audio convert`) and the GUI's worker thread call
`run_conversion` directly -- this is the one place the actual logic lives.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

OcrMode = Literal["auto", "force", "never"]


@dataclass
class ConversionRequest:
    input_path: Path
    output_path: Path
    voice: Path | None = None
    language: str = "en"
    device: str = "auto"
    max_chars: int = 300
    title: str | None = None
    author: str | None = None
    cache_dir: Path | None = None
    ocr_mode: OcrMode = "auto"
    preserve_chapters: bool = True
    allow_download: bool = False
    bitrate: str = "64k"
    page_range: str | None = None  # e.g. "1-10,15,20-25" (1-indexed, inclusive); PDF only
    ai_review: bool = False  # off by default -- see processing/ai_review.py
    ai_review_model: str = "llama3.2"

    def resolved_cache_dir(self) -> Path:
        return self.cache_dir or self.output_path.with_name(self.output_path.stem + "_cache")

    def resolved_title(self) -> str:
        return self.title or self.input_path.stem


@dataclass
class ProgressEvent:
    stage: str  # extracting | ocr | cleaning | chapter_detection | ai_review | tts | assembling | finished
    message: str = ""
    chapter_index: int = 0
    chapter_total: int = 0
    chunk_index: int = 0
    chunk_total: int = 0
    device: str | None = None


@dataclass
class ChunkFailure:
    chapter_index: int
    chapter_title: str
    chunk_index: int
    chunk_text: str
    error: Exception


class ConversionCancelled(Exception):
    pass


class ChunkSynthesisError(Exception):
    def __init__(self, failure: ChunkFailure):
        self.failure = failure
        super().__init__(
            f"Failed synthesizing chapter {failure.chapter_index} "
            f"({failure.chapter_title!r}), chunk {failure.chunk_index}: {failure.error}"
        )


@dataclass
class ConversionPlan:
    """What --dry-run / the GUI's preview shows before committing GPU time."""
    chapter_titles: list[str] = field(default_factory=list)
    chunks_per_chapter: list[int] = field(default_factory=list)
    total_chunks: int = 0
    total_chars: int = 0


def _summarize(chapter_chunks: list[tuple[str, list[str]]]) -> ConversionPlan:
    return ConversionPlan(
        chapter_titles=[t for t, _c in chapter_chunks],
        chunks_per_chapter=[len(c) for _t, c in chapter_chunks],
        total_chunks=sum(len(c) for _t, c in chapter_chunks),
        total_chars=sum(len(c) for _t, chunks in chapter_chunks for c in chunks),
    )


def plan_conversion(
    request: ConversionRequest,
    on_progress: Callable[[ProgressEvent], None] | None = None,
) -> tuple[ConversionPlan, list[tuple[str, list[str]]]]:
    """Run page-selection/extraction/cleaning/AI-review/chunking without
    touching the GPU. Returns the plan summary plus the actual (title,
    chunks) pairs so a caller that wants to proceed doesn't have to redo
    this work (see run_conversion's chapter_chunks parameter)."""
    from book2audio.ingest.extract import extract_markdown
    from book2audio.processing.chunker import chunk_text
    from book2audio.processing.clean import Chapter, clean_text, split_chapters

    report = on_progress or (lambda _e: None)

    effective_input = request.input_path
    if request.page_range and request.input_path.suffix.lower() == ".pdf":
        from book2audio.ingest.page_select import extract_page_subset

        report(ProgressEvent(stage="extracting", message=f"Selecting pages {request.page_range}..."))
        subset_path = request.resolved_cache_dir() / "pages_subset.pdf"
        extract_page_subset(request.input_path, request.page_range, subset_path)
        effective_input = subset_path

    raw = extract_markdown(
        effective_input,
        ocr_output_dir=request.resolved_cache_dir() / "ocr",
        ocr_mode=request.ocr_mode,
        allow_download=request.allow_download,
    )
    if not raw.strip():
        raise ValueError("No text could be extracted from the input.")

    cleaned = clean_text(raw)

    if request.preserve_chapters:
        chapters = split_chapters(cleaned)
    else:
        chapters = [Chapter(title=request.resolved_title(), text=cleaned)]

    if request.ai_review:
        from book2audio.processing.ai_review import review_text

        reviewed = []
        for i, ch in enumerate(chapters, start=1):
            report(ProgressEvent(
                stage="ai_review", chapter_index=i, chapter_total=len(chapters), message=ch.title,
            ))
            reviewed.append(Chapter(title=ch.title, text=review_text(ch.text, model=request.ai_review_model)))
        chapters = reviewed

    chapter_chunks = [(ch.title, chunk_text(ch.text, max_chars=request.max_chars)) for ch in chapters]
    return _summarize(chapter_chunks), chapter_chunks


def run_conversion(
    request: ConversionRequest,
    chapter_chunks: list[tuple[str, list[str]]] | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """If chapter_chunks is omitted, this runs the full plan (extraction
    included) itself. Pass the result of a prior plan_conversion() call to
    avoid re-running extraction/OCR/AI-review a second time -- both are
    real costs, not free work to redo."""
    from book2audio.audio.mux import build_m4b, concat_wavs
    from book2audio.tts.chatterbox_backend import Narrator

    report = on_progress or (lambda _e: None)
    cancelled = should_cancel or (lambda: False)

    if chapter_chunks is None:
        report(ProgressEvent(stage="extracting", message=str(request.input_path)))
        plan, chapter_chunks = plan_conversion(request, on_progress=on_progress)
    else:
        plan = _summarize(chapter_chunks)

    report(ProgressEvent(
        stage="chapter_detection",
        message=f"{len(plan.chapter_titles)} chapter(s), {plan.total_chunks} chunk(s)",
        chapter_total=len(plan.chapter_titles),
        chunk_total=plan.total_chunks,
    ))

    if plan.total_chunks == 0:
        raise ValueError("Nothing to synthesize after cleaning -- input may be empty or unreadable.")

    narrator = Narrator(device=request.device, audio_prompt_path=request.voice)
    report(ProgressEvent(stage="tts", message="Narrator ready", device=narrator.device,
                          chapter_total=len(plan.chapter_titles), chunk_total=plan.total_chunks))

    cache_dir = request.resolved_cache_dir()
    chunk_cache_dir = cache_dir / "chunks"
    chapter_wav_dir = cache_dir / "chapters"
    chapter_wav_dir.mkdir(parents=True, exist_ok=True)

    chapter_wavs: list[tuple[str, Path]] = []
    done_chunks = 0

    for ch_index, (ch_title, chunks) in enumerate(chapter_chunks, start=1):
        chunk_wavs = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            if cancelled():
                raise ConversionCancelled(f"Cancelled at chapter {ch_index}, chunk {chunk_index}")

            try:
                wav_path = narrator.synth_chunk(chunk, language=request.language, cache_dir=chunk_cache_dir)
            except Exception as e:
                failure = ChunkFailure(
                    chapter_index=ch_index, chapter_title=ch_title,
                    chunk_index=chunk_index, chunk_text=chunk, error=e,
                )
                raise ChunkSynthesisError(failure) from e

            chunk_wavs.append(wav_path)
            done_chunks += 1
            report(ProgressEvent(
                stage="tts", chapter_index=ch_index, chapter_total=len(chapter_chunks),
                chunk_index=done_chunks, chunk_total=plan.total_chunks, device=narrator.device,
                message=ch_title,
            ))

        if not chunk_wavs:
            continue

        chapter_wav = chapter_wav_dir / f"{ch_index:03d}.wav"
        concat_wavs(chunk_wavs, chapter_wav)
        chapter_wavs.append((ch_title, chapter_wav))

    report(ProgressEvent(stage="assembling", message=str(request.output_path)))
    build_m4b(chapter_wavs, request.output_path, title=request.resolved_title(),
              author=request.author, bitrate=request.bitrate)

    report(ProgressEvent(stage="finished", message=str(request.output_path)))
    return request.output_path
