# book2audio

Convert books (PDF, EPUB, or scanned images) into narrated, chaptered `.m4b` audiobooks — entirely local, GPU-accelerated.

```text
MarkItDown → OpenOCR fallback → Chatterbox Multilingual V3 → FFmpeg → .m4b
```

- **MarkItDown** extracts text from PDFs/EPUBs that have a real text layer.
- **OpenOCR** is the fallback for scanned pages with no usable text layer (triggered automatically — see [How extraction decides](#how-extraction-decides)).
- **Chatterbox Multilingual V3** (500M params, 23+ languages, voice cloning) narrates. CUDA on this machine (RTX 4060), MPS/CPU elsewhere.
- **FFmpeg** stitches chapters into one chaptered `.m4b` with title/author metadata.

Everything runs locally — see [Offline mode](#offline-mode).

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd ~/Projects/book2audio
uv python install 3.11
uv venv --python 3.11
uv pip install -e .
```

System packages needed: `ffmpeg libsndfile1 libgl1 libglib2.0-0 build-essential`.

One dependency note baked into `pyproject.toml`: `chatterbox-tts` installs from GitHub `main`, not PyPI — the published `0.1.7` release doesn't expose `t3_model="v3"` yet.

First run downloads model weights: ~3GB (Chatterbox) into `~/.cache/huggingface`, ~1.8GB (OpenOCR) into `~/.cache/openocr`. After that, nothing is re-downloaded — see [Offline mode](#offline-mode).

## Usage

```bash
book2audio convert INPUT -o OUTPUT.m4b [OPTIONS]
```

(Either `source .venv/bin/activate` first, or run `.venv/bin/book2audio` directly — both work.)

```bash
# Basic
book2audio convert book.pdf -o book.m4b

# Preview chapters/chunk counts before spending GPU time
book2audio convert book.pdf -o book.m4b --dry-run

# Clone a voice from a reference clip
book2audio convert book.pdf -o book.m4b --voice narrator.wav

# Set metadata explicitly (defaults: title = input filename, author = unset)
book2audio convert book.pdf -o book.m4b --title "Technical Mentality" --author "Gilbert Simondon"

# A directory of scanned page images also works
book2audio convert ./scans/ -o book.m4b
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `-o, --output` | *(required)* | Output `.m4b` path. |
| `--voice PATH` | none | Reference WAV/MP3 to clone as the narrator voice. Must exist. |
| `--language` | `en` | Language code passed to Chatterbox. |
| `--device` | `auto` | `cuda`, `mps`, `cpu`, or `auto` (picks the best available). |
| `--max-chars` | `300` | Max characters per TTS chunk — see [Why chunks are small](#why-chunks-are-small). |
| `--title` | input filename | Audiobook title tag. |
| `--author` | none | Audiobook author tag. |
| `--cache-dir` | `<output-stem>_cache/` next to the output | Where synthesized chunk WAVs are cached. |
| `--dry-run` | off | Extract/clean/chunk only — reports chapter and chunk counts, does no synthesis. |

Input formats: `.pdf`, `.epub`, `.docx`, `.pptx`, `.xlsx`, `.txt`, `.md`, `.html` (via MarkItDown), plus `.png`/`.jpg`/`.jpeg`/`.tif`/`.tiff`/`.bmp` and directories of those (straight to OCR).

## How extraction decides

For PDFs: MarkItDown extracts first. If the average characters-per-page comes out under 40, the PDF is assumed to be a scan with no real text layer, and the whole thing gets re-run through OpenOCR instead. EPUB/text formats always go through MarkItDown (always have real text). Images and directories of images always go straight to OpenOCR.

## Resuming an interrupted run

Every synthesized chunk is cached under `<cache-dir>/chunks/<hash>.wav`, keyed by the exact text + language + voice. Re-running the same `convert` command skips any chunk already on disk — a killed or crashed run picks back up for free. Chunk writes are atomic (write-then-rename), so a kill mid-write never leaves a corrupt chunk that gets treated as valid.

## Offline mode

`HF_HUB_OFFLINE=1` is the default (set in `book2audio/__init__.py`, so it applies no matter which module runs first). Model loading uses the local cache directly with **no network calls** once weights are downloaded. If a needed file is genuinely missing from the cache, it automatically falls back to downloading it, then returns to offline for the rest of the run. Same pattern for OpenOCR (`--no_auto_download` tried first, falls back to allowing download only on failure).

To force it explicitly or override: `HF_HUB_OFFLINE=0 book2audio convert ...`.

## Why chunks are small

Chatterbox generates autoregressively with a fixed step budget, so very long single inputs risk truncation or prosody drift — unlike some TTS engines that handle arbitrary-length text. 300 characters (roughly one to a few sentences) keeps quality consistent across a whole book. Text is split on sentence boundaries, never mid-sentence, except when a single sentence itself exceeds `--max-chars`.

## Verify the stack manually

```bash
# GPU / CUDA
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Digital PDF -> Markdown (no OCR needed)
.venv/bin/markitdown book.pdf -o book.md

# Scanned PDF/images -> Markdown via OCR
.venv/bin/openocr --task doc --input_path book.pdf --output_path ./ocr-output --use_layout_detection --save_markdown

# TTS smoke test
.venv/bin/python test_tts.py && ffplay test.wav
```

## Troubleshooting

- **`openocr is not on PATH`**: shouldn't happen — the CLI resolves the binary relative to the running interpreter, not `$PATH`, so it works whether or not the venv is activated. If it does happen, `uv pip install -e .` didn't complete; re-run it.
- **A chapter title looks wrong** (e.g. picks up a stray heading mid-chapter): chapter detection is a heuristic (`chapter N` / `part N` / markdown `#` headings under 100 chars) — it can over- or under-split on unusual formatting. Not currently configurable; if it matters, edit `src/book2audio/clean.py`'s `_CHAPTER_HEADING`/`_MD_HEADING` patterns.
- **Narration mispronounces a run-together number** (e.g. a footnote reference glued to a word like `Antiquity3`): the cleaner strips these heuristically; a legitimate number glued to a word in the same way (rare) would also get stripped. See `_GLUED_FOOTNOTE_REF` in `clean.py`.

## Project layout

```
src/book2audio/
  extract.py   MarkItDown + OpenOCR fallback
  clean.py     dehyphenation, page-number/footnote stripping, chapter detection
  chunker.py   sentence-aware chunking
  synth.py     Chatterbox wrapper, offline-first loading, per-chunk cache
  mux.py       ffmpeg concat + chaptered .m4b muxing
  cli.py       book2audio convert (Typer)
```
