"""The single conversion pipeline: extract -> clean -> [Qwen review] ->
save text -> chapter detect -> chunk -> narrate (Chatterbox or Kokoro) ->
mux.

Both the CLI (`book2audio convert`) and the GUI's worker thread call
`run_conversion` directly -- this is the one place the actual logic lives.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

OcrMode = Literal["auto", "force", "never"]


@dataclass
class ConversionRequest:
    input_path: Path
    output_path: Path
    voice: Path | None = None  # Chatterbox: reference clip for voice cloning
    kokoro_voice: str = "af_heart"  # Kokoro: named voice
    tts_backend: str = "chatterbox"  # "chatterbox" | "kokoro"
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
    narration_cleanup: bool = False  # off by default -- see processing/narration_cleanup.py
    narration_cleanup_model: str = "qwen3-vl:4b-instruct"  # text-only; despite the model name, no image is ever sent
    debug_narration_cleanup: bool = False  # retain full per-chunk input/output audit (not written unless requested)
    save_text_outputs: bool = True  # write <stem>.raw/cleaned/narration.md next to the .m4b
    extract_only: bool = False  # extract + save text, skip TTS entirely

    def resolved_cache_dir(self) -> Path:
        return self.cache_dir or self.output_path.with_name(self.output_path.stem + "_cache")

    def resolved_title(self) -> str:
        return self.title or self.input_path.stem

    def text_output_path(self, suffix: str) -> Path:
        """suffix e.g. 'raw.md', 'cleaned.md', 'narration.md', 'extract_meta.json'."""
        return self.output_path.with_name(f"{self.output_path.stem}.{suffix}")


def default_output_stem(input_path: Path, page_range: str | None) -> str:
    """The stem `-o`/the GUI's filename field defaults to when the user
    hasn't typed one -- page-range-qualified so converting different
    ranges of the same book doesn't collide (e.g. "book_pages_120-150").
    Only applied to *default* naming; an explicit -o/filename is never
    touched."""
    stem = input_path.stem
    if page_range:
        stem = f"{stem}_pages_{_sanitize_page_range(page_range)}"
    return stem


def _sanitize_page_range(page_range: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", page_range).strip("-")


@dataclass
class ProgressEvent:
    stage: str  # extracting | cleaning | chapter_detection | narration_cleanup | tts | assembling | finished
    message: str = ""
    chapter_index: int = 0
    chapter_total: int = 0
    chunk_index: int = 0
    chunk_total: int = 0
    page_index: int = 0
    page_total: int = 0
    device: str | None = None
    tts_backend: str | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    rtf: float | None = None  # generation_seconds / generated_audio_seconds, excludes cache hits


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
    """What --dry-run / --extract-only / the GUI's preview shows before
    committing GPU time to TTS."""
    chapter_titles: list[str] = field(default_factory=list)
    chunks_per_chapter: list[int] = field(default_factory=list)
    total_chunks: int = 0
    total_chars: int = 0
    narration_cleanup_applied: bool = False
    text_outputs_saved: list[Path] = field(default_factory=list)
    extraction_reused_cache: bool = False


def _summarize(chapter_chunks: list[tuple[str, list[str]]]) -> ConversionPlan:
    return ConversionPlan(
        chapter_titles=[t for t, _c in chapter_chunks],
        chunks_per_chapter=[len(c) for _t, c in chapter_chunks],
        total_chunks=sum(len(c) for _t, c in chapter_chunks),
        total_chars=sum(len(c) for _t, chunks in chapter_chunks for c in chunks),
    )


def _extraction_recipe(request: ConversionRequest) -> dict:
    st = request.input_path.stat()
    return {
        "input_mtime": st.st_mtime,
        "input_size": st.st_size,
        "ocr_mode": request.ocr_mode,
        "narration_cleanup": request.narration_cleanup,
        "narration_cleanup_model": request.narration_cleanup_model if request.narration_cleanup else None,
        "page_range": request.page_range,
    }


def _load_cached_extraction(request: ConversionRequest):
    """Reuse a prior extraction (raw/cleaned/narration .md + its recipe
    fingerprint) if the request matches exactly and the files are still
    there -- extraction (especially OCR + narration cleanup) is expensive
    and a re-run (e.g. --extract-only twice, or extract-only then a full
    run) shouldn't redo it."""
    from book2audio.ingest.extract import ExtractResult

    meta_path = request.resolved_cache_dir() / "extraction_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("recipe") != _extraction_recipe(request):
        return None

    raw_path = request.text_output_path("raw.md")
    cleaned_path = request.text_output_path("cleaned.md")
    narration_path = request.text_output_path("narration.md")
    if not raw_path.exists() or not cleaned_path.exists():
        return None
    if request.narration_cleanup and not narration_path.exists():
        return None

    return ExtractResult(
        raw_text=raw_path.read_text(encoding="utf-8"),
        cleaned_text=cleaned_path.read_text(encoding="utf-8"),
        narration_text=narration_path.read_text(encoding="utf-8") if narration_path.exists() else None,
        narration_cleanup_applied=meta.get("narration_cleanup_applied", False),
    )


def _save_extraction(request: ConversionRequest, extracted) -> list[Path]:
    saved = []

    def write(suffix: str, text: str) -> None:
        path = request.text_output_path(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        saved.append(path)

    write("raw.md", extracted.raw_text)
    write("cleaned.md", extracted.cleaned_text)
    if extracted.narration_text is not None:
        write("narration.md", extracted.narration_text)

    cache_dir = request.resolved_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "recipe": _extraction_recipe(request),
        "narration_cleanup_applied": extracted.narration_cleanup_applied,
    }
    (cache_dir / "extraction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Only written when --debug-narration-cleanup was passed (extracted.
    # cleanup_audit is empty otherwise) -- keeps the output directory free
    # of per-chunk debug clutter unless debug mode is actually on. Lives
    # alongside the other .md outputs (not just the cache dir) so it's
    # actually discoverable via "Open Output Folder" in the GUI.
    if extracted.cleanup_audit:
        audit_data = [
            {"chunk_index": e.chunk_index, "input": e.input_text, "output": e.output_text,
             "model": e.model, "status": e.status}
            for e in extracted.cleanup_audit
        ]
        write("narration_debug.json", json.dumps(audit_data, indent=2))

    return saved


def plan_conversion(
    request: ConversionRequest,
    on_progress: Callable[[ProgressEvent], None] | None = None,
) -> tuple[ConversionPlan, list[tuple[str, list[str]]]]:
    """Run page-selection/extraction/cleaning/AI-review/chunking without
    touching the GPU. Returns the plan summary plus the actual (title,
    chunks) pairs so a caller that wants to proceed doesn't have to redo
    this work (see run_conversion's chapter_chunks parameter)."""
    from book2audio.processing.chunker import chunk_text
    from book2audio.processing.clean import Chapter, split_chapters

    report = on_progress or (lambda _e: None)

    extracted = _load_cached_extraction(request)
    reused_cache = extracted is not None

    if extracted is None:
        from book2audio.ingest.extract import extract_and_clean

        effective_input = request.input_path
        if request.page_range and request.input_path.suffix.lower() == ".pdf":
            from book2audio.ingest.page_select import extract_page_subset

            report(ProgressEvent(stage="extracting", message=f"Selecting pages {request.page_range}..."))
            subset_path = request.resolved_cache_dir() / "pages_subset.pdf"
            extract_page_subset(request.input_path, request.page_range, subset_path)
            effective_input = subset_path

        def on_cleanup_progress(i: int, total: int) -> None:
            report(ProgressEvent(stage="narration_cleanup", page_index=i, page_total=total))

        extracted = extract_and_clean(
            effective_input,
            ocr_output_dir=request.resolved_cache_dir() / "ocr",
            ocr_mode=request.ocr_mode,
            allow_download=request.allow_download,
            narration_cleanup=request.narration_cleanup,
            narration_cleanup_model=request.narration_cleanup_model,
            on_cleanup_progress=on_cleanup_progress,
            debug_cleanup=request.debug_narration_cleanup,
        )
    else:
        report(ProgressEvent(stage="extracting", message="Reusing cached extraction (recipe unchanged)."))

    if not extracted.final_text.strip():
        raise ValueError("No text could be extracted from the input.")

    saved_paths: list[Path] = []
    if request.save_text_outputs:
        saved_paths = _save_extraction(request, extracted)

    if request.preserve_chapters:
        chapters = split_chapters(extracted.final_text)
    else:
        chapters = [Chapter(title=request.resolved_title(), text=extracted.final_text)]

    chapter_chunks = [(ch.title, chunk_text(ch.text, max_chars=request.max_chars)) for ch in chapters]
    plan = _summarize(chapter_chunks)
    plan.narration_cleanup_applied = extracted.narration_cleanup_applied
    plan.text_outputs_saved = saved_paths
    plan.extraction_reused_cache = reused_cache
    return plan, chapter_chunks


def run_conversion(
    request: ConversionRequest,
    chapter_chunks: list[tuple[str, list[str]]] | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """If chapter_chunks is omitted, this runs the full plan (extraction
    included) itself. Pass the result of a prior plan_conversion() call to
    avoid re-running extraction/OCR/AI-review a second time -- all three
    are real costs, not free work to redo."""
    from book2audio.audio.mux import build_m4b, concat_wavs
    from book2audio.tts.factory import create_provider

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

    # Load the selected TTS model once for the whole job, not per chunk --
    # and unload it (freeing GPU memory) once the job is done, matching
    # the same load-once/unload-after-job discipline already used for the
    # Qwen review pass.
    provider = create_provider(request.tts_backend, request.device, request.voice, request.kokoro_voice)
    report(ProgressEvent(stage="tts", message="Narrator ready", device=provider.device,
                          tts_backend=provider.engine_id,
                          chapter_total=len(plan.chapter_titles), chunk_total=plan.total_chunks))

    cache_dir = request.resolved_cache_dir()
    chunk_cache_dir = cache_dir / "chunks"
    chapter_wav_dir = cache_dir / "chapters"
    chapter_wav_dir.mkdir(parents=True, exist_ok=True)

    chapter_wavs: list[tuple[str, Path]] = []
    done_chunks = 0
    total_gen_seconds = 0.0
    total_audio_seconds = 0.0
    job_start = time.monotonic()

    try:
        for ch_index, (ch_title, chunks) in enumerate(chapter_chunks, start=1):
            chunk_wavs = []
            for chunk_index, chunk in enumerate(chunks, start=1):
                if cancelled():
                    raise ConversionCancelled(f"Cancelled at chapter {ch_index}, chunk {chunk_index}")

                t0 = time.monotonic()
                try:
                    wav_path, was_cached = provider.synth_chunk(chunk, language=request.language, cache_dir=chunk_cache_dir)
                except Exception as e:
                    failure = ChunkFailure(
                        chapter_index=ch_index, chapter_title=ch_title,
                        chunk_index=chunk_index, chunk_text=chunk, error=e,
                    )
                    raise ChunkSynthesisError(failure) from e
                gen_time = time.monotonic() - t0

                if not was_cached:
                    import soundfile as sf

                    total_gen_seconds += gen_time
                    total_audio_seconds += sf.info(str(wav_path)).duration

                chunk_wavs.append(wav_path)
                done_chunks += 1

                rtf = (total_gen_seconds / total_audio_seconds) if total_audio_seconds > 0 else None
                elapsed = time.monotonic() - job_start
                remaining = plan.total_chunks - done_chunks
                eta = (elapsed / done_chunks) * remaining if done_chunks else None

                report(ProgressEvent(
                    stage="tts", chapter_index=ch_index, chapter_total=len(chapter_chunks),
                    chunk_index=done_chunks, chunk_total=plan.total_chunks, device=provider.device,
                    tts_backend=provider.engine_id, message=ch_title,
                    elapsed_seconds=elapsed, eta_seconds=eta, rtf=rtf,
                ))

            if not chunk_wavs:
                continue

            chapter_wav = chapter_wav_dir / f"{ch_index:03d}.wav"
            concat_wavs(chunk_wavs, chapter_wav)
            chapter_wavs.append((ch_title, chapter_wav))
    finally:
        provider.unload()

    report(ProgressEvent(stage="assembling", message=str(request.output_path)))
    build_m4b(chapter_wavs, request.output_path, title=request.resolved_title(),
              author=request.author, bitrate=request.bitrate)

    report(ProgressEvent(stage="finished", message=str(request.output_path)))
    return request.output_path
