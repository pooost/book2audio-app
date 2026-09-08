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

    def test_running_header_without_page_number_removed(self):
        raw = "\n\n".join([
            "GILBERT SIMONDON",
            "The first paragraph of the chapter begins here.",
            "GILBERT SIMONDON",
            "A second paragraph continues the exposition.",
            "GILBERT SIMONDON",
            "A third paragraph wraps up the section.",
        ])
        cleaned = clean_text(raw)
        assert "GILBERT SIMONDON" not in cleaned
        assert "begins here" in cleaned
        assert "wraps up" in cleaned

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
