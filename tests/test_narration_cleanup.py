"""Tests for the text-only Qwen narration cleanup stage.

Chunking/recombination tests are pure Python (no Ollama needed) and always
run. Content-correctness tests need a real local Qwen via Ollama -- they're
skipped (not failed) when Ollama/the model isn't available, since "Qwen
returned a response" isn't what these check: they check the response is
actually correct, which requires calling the real model.
"""

from pathlib import Path

import pytest

from book2audio.processing.narration_cleanup import (
    DEFAULT_MODEL,
    NarrationCleanupError,
    chunk_for_cleanup,
    cleanup_chunk,
    cleanup_chunk_cached,
    cleanup_document,
    is_ollama_available,
)


def _model_ready() -> bool:
    if not is_ollama_available():
        return False
    from book2audio.processing.narration_cleanup import available_models

    return DEFAULT_MODEL in available_models()


requires_qwen = pytest.mark.skipif(
    not _model_ready(), reason=f"Ollama/{DEFAULT_MODEL} not available on this machine"
)


# ---------------------------------------------------------------------------
# Chunking / recombination -- no model needed, always run.
# ---------------------------------------------------------------------------

class TestChunkForCleanup:
    def test_identity_recombination_preserves_paragraph_order(self):
        paragraphs = [f"Paragraph {i} content here, nothing special." for i in range(1, 6)]
        original = "\n\n".join(paragraphs)

        chunks = chunk_for_cleanup(original, target_chars=100, max_chars=150)
        rejoined = "\n\n".join(chunks)

        assert rejoined == original
        assert len(chunks) > 1, "test is only meaningful if it actually produced multiple chunks"

    def test_no_paragraph_split_across_chunks(self):
        paragraphs = [f"Paragraph {i}." for i in range(1, 8)]
        original = "\n\n".join(paragraphs)
        chunks = chunk_for_cleanup(original, target_chars=20, max_chars=30)

        reconstructed_paragraphs = "\n\n".join(chunks).split("\n\n")
        assert reconstructed_paragraphs == paragraphs

    def test_oversized_single_paragraph_splits_on_sentences_not_words(self):
        long_para = " ".join(f"Sentence number {i}." for i in range(1, 60))
        chunks = chunk_for_cleanup(long_para, target_chars=100, max_chars=150)

        assert len(chunks) > 1
        for c in chunks:
            assert not c.startswith(" ") and not c.endswith(" ")
        # No word dropped or duplicated across the split.
        assert " ".join(chunks).split() == long_para.split()

    def test_empty_input(self):
        assert chunk_for_cleanup("") == []
        assert chunk_for_cleanup("   \n\n   ") == []

    def test_chunks_stay_within_spec_range_for_dense_text(self):
        # A long document made of short paragraphs should produce chunks
        # roughly in the 1000-3000 char range the spec asks for (using
        # defaults), not one-paragraph-per-chunk.
        paragraphs = [f"This is paragraph number {i} with some representative prose." for i in range(1, 100)]
        original = "\n\n".join(paragraphs)
        chunks = chunk_for_cleanup(original)  # defaults: target=2000, max=3000

        assert len(chunks) > 1
        for c in chunks[:-1]:  # last chunk may be shorter
            assert len(c) <= 3000


# ---------------------------------------------------------------------------
# Cache / resume -- mocked model call, no Ollama needed (tests the caching
# mechanism itself, independent of content correctness).
# ---------------------------------------------------------------------------

class TestCacheResume:
    def test_second_call_is_cache_hit_and_skips_model_call(self, tmp_path, monkeypatch):
        calls = []

        def fake_cleanup_chunk(text, model=DEFAULT_MODEL, timeout=180.0):
            calls.append(text)
            return text.upper()

        monkeypatch.setattr("book2audio.processing.narration_cleanup.cleanup_chunk", fake_cleanup_chunk)

        text = "some chunk text"
        result1, cached1 = cleanup_chunk_cached(text, DEFAULT_MODEL, tmp_path)
        result2, cached2 = cleanup_chunk_cached(text, DEFAULT_MODEL, tmp_path)

        assert cached1 is False
        assert cached2 is True
        assert result1 == result2 == "SOME CHUNK TEXT"
        assert calls == [text], "the model should only be called once -- second call must be a cache hit"

    def test_different_model_produces_different_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "book2audio.processing.narration_cleanup.cleanup_chunk",
            lambda text, model=DEFAULT_MODEL, timeout=180.0: f"cleaned-by-{model}",
        )
        text = "identical input text"
        _, cached_a = cleanup_chunk_cached(text, "model-a", tmp_path)
        _, cached_b = cleanup_chunk_cached(text, "model-b", tmp_path)
        assert cached_a is False
        assert cached_b is False, "switching models must not reuse the other model's cached result"

    def test_prompt_version_bump_invalidates_old_cache(self, tmp_path, monkeypatch):
        import book2audio.processing.narration_cleanup as nc

        monkeypatch.setattr(nc, "cleanup_chunk", lambda text, model=DEFAULT_MODEL, timeout=180.0: "v1-result")
        text = "some text"
        _, cached_before = cleanup_chunk_cached(text, DEFAULT_MODEL, tmp_path)
        assert cached_before is False

        # Simulate a prompt change by bumping PROMPT_VERSION -- the old
        # cache entry must not be silently reused under the new version.
        monkeypatch.setattr(nc, "PROMPT_VERSION", "v2-simulated")
        monkeypatch.setattr(nc, "cleanup_chunk", lambda text, model=DEFAULT_MODEL, timeout=180.0: "v2-result")
        result_after, cached_after = cleanup_chunk_cached(text, DEFAULT_MODEL, tmp_path)

        assert cached_after is False, "a prompt-version bump must invalidate the old cache entry"
        assert result_after == "v2-result"

    def test_interrupted_document_cleanup_resumes_without_recleaning(self, tmp_path, monkeypatch):
        """Simulates a crash partway through a multi-chunk document: the
        first chunk succeeded (cached), the run then "restarts" -- the
        already-completed chunk must not be sent to the model again."""
        import book2audio.processing.narration_cleanup as nc

        calls = []

        def fake_cleanup_chunk(text, model=DEFAULT_MODEL, timeout=180.0):
            calls.append(text)
            return text + " [cleaned]"

        monkeypatch.setattr(nc, "cleanup_chunk", fake_cleanup_chunk)
        monkeypatch.setattr(nc, "unload_model", lambda model=DEFAULT_MODEL: None)

        doc = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        # First (successful) run.
        result1 = cleanup_document(doc, model=DEFAULT_MODEL, cache_dir=tmp_path)
        assert len(calls) >= 1
        calls_after_first_run = list(calls)

        # "Resume": run again with the same cache dir -- nothing should be
        # sent to the model a second time.
        calls.clear()
        result2 = cleanup_document(doc, model=DEFAULT_MODEL, cache_dir=tmp_path)
        assert calls == [], "a resumed run must not re-clean already-cached chunks"
        assert result1.text == result2.text


# ---------------------------------------------------------------------------
# Content correctness -- the actual examples from the spec. Requires a real
# local Qwen; skipped (not failed) if unavailable.
# ---------------------------------------------------------------------------

@requires_qwen
class TestCleanupExamples:
    def test_garbage_digit_sequence_removed(self):
        result = cleanup_chunk("The political system 254678 367489 578944 was undergoing rapid change.")
        assert "254678" not in result
        assert "political system" in result
        assert "rapid change" in result

    def test_legitimate_year_preserved(self):
        result = cleanup_chunk("The novel was published in 1984.")
        assert "1984" in result

    def test_rn_confusion_repaired(self):
        result = cleanup_chunk("The rnaterial conditions shaped the outcome.")
        assert "material" in result.lower()
        assert "rnaterial" not in result

    def test_section_number_preserved(self):
        result = cleanup_chunk("Section 4.2 describes the second stage.")
        assert "4.2" in result

    def test_hyphenated_linebreak_reconstructed(self):
        result = cleanup_chunk("post-\nstructuralism")
        assert "post-structuralism" in result.replace("\n", "")

    def test_proper_nouns_preserved(self):
        result = cleanup_chunk("Deleuze and Guattari developed the concept.")
        assert "Deleuze" in result
        assert "Guattari" in result

    def test_decimal_value_preserved(self):
        result = cleanup_chunk("The value was 3.14159.")
        assert "3.14159" in result

    def test_ocr_duplicated_word_removed(self):
        result = cleanup_chunk("The argument argument was repeated due to OCR.")
        # The duplicate should be gone; the sentence should still make sense.
        assert result.lower().count("argument") == 1

    def test_unusual_philosophical_terminology_not_rewritten(self):
        original = (
            "The dispositif operates through an apparatus of heterogeneous "
            "discursive and non-discursive elements."
        )
        result = cleanup_chunk(original)
        assert result.strip() == original.strip(), (
            "correct, unusual-but-valid prose must be returned unchanged, not simplified or rewritten"
        )


@requires_qwen
class TestNarrationIsActualTTSSource:
    """The critical end-to-end check: when narration cleanup is enabled,
    the corrected text -- not raw or deterministically-cleaned text --
    is what actually reaches chapter detection / chunking (i.e. what
    would be passed to a TTS provider)."""

    def test_corrected_text_reaches_final_chunks(self, tmp_path):
        from book2audio.pipeline.convert import ConversionRequest, plan_conversion

        input_path = tmp_path / "book.txt"
        input_path.write_text(
            "Chapter One\n\n"
            "The rnaterial conditions 384729 384729 changed the outcome, "
            "and Deleuze remained relevant in 1984.\n",
            encoding="utf-8",
        )

        request = ConversionRequest(
            input_path=input_path,
            output_path=tmp_path / "book.m4b",
            narration_cleanup=True,
        )
        plan, chapter_chunks = plan_conversion(request)

        all_text = " ".join(c for _title, chunks in chapter_chunks for c in chunks)
        assert "rnaterial" not in all_text
        assert "384729" not in all_text
        assert "material" in all_text
        assert "1984" in all_text
        assert "Deleuze" in all_text

        narration_path = request.text_output_path("narration.md")
        cleaned_path = request.text_output_path("cleaned.md")
        assert narration_path.exists()
        assert "rnaterial" in cleaned_path.read_text(), "cleaned.md should retain the original error"
        assert "rnaterial" not in narration_path.read_text(), "narration.md should have it corrected"
        assert narration_path.read_text().strip() in all_text or all_text in narration_path.read_text()

    def test_cleaned_md_is_source_when_cleanup_disabled(self, tmp_path):
        from book2audio.pipeline.convert import ConversionRequest, plan_conversion

        input_path = tmp_path / "book.txt"
        input_path.write_text("Chapter One\n\nThe rnaterial conditions changed.\n", encoding="utf-8")

        request = ConversionRequest(
            input_path=input_path,
            output_path=tmp_path / "book.m4b",
            narration_cleanup=False,
        )
        plan, chapter_chunks = plan_conversion(request)

        all_text = " ".join(c for _title, chunks in chapter_chunks for c in chunks)
        assert "rnaterial" in all_text, "with cleanup off, the uncorrected cleaned text must be used"
        assert not request.text_output_path("narration.md").exists()
