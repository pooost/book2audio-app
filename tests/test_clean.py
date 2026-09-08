"""Tests for deterministic cleanup (processing/clean.py), in particular the
whole-document repeated-line (running header/footer) stripper."""

from book2audio.processing.clean import clean_text


class TestStripRepeatedLines:
    def test_indd_export_footer_removed(self):
        """Reproduces a real reported case: InDesign print-export footers
        ("<file> PRINT.indd <page>" + a repeated timestamp) stamped on
        every page, which is accurately-transcribed text (not OCR
        corruption) but not authored content -- would otherwise get read
        aloud mid-book as a jarring interruption."""
        raw = "\n\n".join([
            "This chapter examines the political stakes of the argument.",
            "De BOEVER PRINT.indd 9",
            "06/12/2011 14:26 06/12/2011 14:26",
            "The discussion continues with technical objects.",
            "De BOEVER PRINT.indd 10",
            "06/12/2011 14:26 06/12/2011 14:26",
            "A third section elaborates on the genesis of individuation.",
            "De BOEVER PRINT.indd 11",
            "06/12/2011 14:26 06/12/2011 14:26",
            "The chapter concludes with remarks on structure.",
        ])

        cleaned = clean_text(raw)

        assert "PRINT.indd" not in cleaned
        assert "06/12/2011" not in cleaned
        assert "political stakes" in cleaned
        assert "technical objects" in cleaned
        assert "individuation" in cleaned
        assert "concludes with remarks" in cleaned

    def test_date_stamp_as_its_own_repeated_line_fully_removed(self):
        """Regression test for a real bug found via the Simondon PDF: a
        date is unambiguous furniture (matches _STRONG_FURNITURE_SIGNAL),
        so every occurrence should be dropped -- but should_drop() used to
        check the signal against the NORMALIZED line, and normalize()'s
        leading-page-number strip mangles "06/12/2011" into "/12/2011"
        (eating the "06" as if it were a page number). The mangled form
        no longer matches the date pattern, so the line fell through to
        the "keep first occurrence" branch instead of "drop every
        occurrence" -- one stray timestamp survived out of thirty real
        occurrences in the actual book. Checking against the ORIGINAL
        line fixes it; this test would have caught the bug directly
        (unlike test_indd_export_footer_removed above, whose "06/12/2011
        not in cleaned" assertion doesn't notice a mangled "/12/2011"
        survivor -- a different string)."""
        raw = "\n\n".join(
            [f"Paragraph number {i} of the chapter, with some real content."
             if i % 2 == 0 else "06/12/2011   14:26"
             for i in range(10)]
        )
        cleaned = clean_text(raw)
        assert "2011" not in cleaned
        assert "/12/" not in cleaned
        assert "Paragraph number 0" in cleaned
        assert "Paragraph number 8" in cleaned

    def test_running_header_without_page_number_collapsed_to_one(self):
        """A repeated header with no filename/date/time signal is
        ambiguous -- it might be reused as a legitimate title/byline the
        first time (see test_title_reused_as_running_header_keeps_title
        below for the real case this matters for). Only the first
        occurrence survives; the running-header repeats are removed."""
        raw = "\n\n".join([
            "GILBERT SIMONDON",
            "The first paragraph of the chapter begins here.",
            "GILBERT SIMONDON",
            "A second paragraph continues the exposition.",
            "GILBERT SIMONDON",
            "A third paragraph wraps up the section.",
        ])
        cleaned = clean_text(raw)
        assert cleaned.count("GILBERT SIMONDON") == 1
        assert "begins here" in cleaned
        assert "wraps up" in cleaned

    def test_title_reused_as_running_header_keeps_title(self):
        """Reproduces a real bug found via user testing: a chapter title
        ("Technical Mentality") that's ALSO reused as the running header
        on every subsequent page. The original fix blindly stripped every
        occurrence of a repeated line, which deleted the real title along
        with the furniture copies -- worse than the original problem,
        since it silently destroyed real content. Only the first
        occurrence (the real title) must survive."""
        raw = "\n\n".join([
            "Chapter 1",
            "Technical Mentality",
            "This is the actual chapter content beginning here with real prose.",
            "Technical Mentality   3",
            "More real content continues in this paragraph of the book.",
            "Technical Mentality   5",
            "Even more real content appears in this later paragraph.",
            "Technical Mentality   7",
            "The chapter continues with additional discussion of the topic.",
        ])
        cleaned = clean_text(raw)
        assert cleaned.count("Technical Mentality") == 1
        assert "actual chapter content" in cleaned
        assert "additional discussion" in cleaned

    def test_leading_page_number_header_normalized_and_collapsed(self):
        """Reproduces a second real report: some print layouts put the
        page number BEFORE the running header text (common two-sided-book
        convention, page number on the outer margin), not just after --
        "6 Running Header" / "8 Running Header" must be recognized as the
        same recurring line, not two different ones that never hit the
        repetition threshold."""
        raw = "\n\n".join([
            "6 Gilbert Simondon: Being and Technology",
            "Some real prose on this page discussing the argument in depth.",
            "8 Gilbert Simondon: Being and Technology",
            "Further real prose continuing the discussion on this page.",
            "10 Gilbert Simondon: Being and Technology",
            "Additional real prose wrapping up this section of the chapter.",
        ])
        cleaned = clean_text(raw)
        assert cleaned.count("Gilbert Simondon: Being and Technology") == 1
        assert "real prose on this page" in cleaned
        assert "wrapping up this section" in cleaned

    def test_legitimately_repeated_dialogue_preserved(self):
        """A short line repeated several times is NOT automatically
        furniture -- real dialogue can legitimately repeat. The guard is
        that headers/footers essentially never end in terminal sentence
        punctuation; dialogue does."""
        raw = "\n\n".join([
            'He looked up and asked the question again.',
            '"Yes."',
            "She nodded slowly, considering the implications.",
            '"Yes."',
            "He pressed further, unconvinced by her answer.",
            '"Yes."',
            "The conversation continued in this manner for some time.",
        ])
        cleaned = clean_text(raw)
        assert cleaned.count('"Yes."') == 3

    def test_two_occurrences_not_enough_to_strip(self):
        """Below REPEATED_LINE_MIN_OCCURRENCES (3): could plausibly be a
        real coincidence rather than page furniture, so left alone."""
        raw = "\n\n".join([
            "Some Running Header",
            "First paragraph of real content.",
            "Some Running Header",
            "Second paragraph of real content.",
        ])
        cleaned = clean_text(raw)
        assert "Some Running Header" in cleaned

    def test_long_repeated_line_not_stripped(self):
        """A line >= REPEATED_LINE_MAX_LENGTH is treated as real content
        even if it recurs -- headers/footers are short by nature; a long
        recurring line is far more likely to be a genuine refrain."""
        long_line = "This is a deliberately long sentence-like line that exceeds the furniture length cutoff"
        raw = "\n\n".join([long_line, "Para one.", long_line, "Para two.", long_line, "Para three."])
        cleaned = clean_text(raw)
        assert cleaned.count(long_line) == 3
