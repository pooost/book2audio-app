# book2audio

Convert books (PDF, EPUB, or scanned images) into narrated `.m4b` audiobooks, run locally on GPU.

```text
MarkItDown → OpenOCR fallback → Chatterbox Multilingual V3 → FFmpeg → .m4b
```

- **MarkItDown** extracts text from PDFs/EPUBs with a real text layer.
- **OpenOCR** is the fallback for scanned pages with no usable text layer.
- **Chatterbox Multilingual V3** (500M, 23+ languages, voice cloning) does narration; CUDA on this machine (RTX 4060), MPS/CPU elsewhere.
- **FFmpeg** stitches chapter audio into a single chaptered `.m4b`.

The CLI itself (`book2audio.cli`) isn't built yet — this stage is environment setup and per-layer verification.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.11
uv venv --python 3.11
uv pip install -e .
```

System packages needed: `ffmpeg libsndfile1 libgl1 libglib2.0-0 build-essential` (already present on this machine; install via `apt` if missing elsewhere).

Notes baked into `pyproject.toml`:
- `chatterbox-tts` is pinned to GitHub `main`, not PyPI — the published `0.1.7` release doesn't yet expose `t3_model="v3"`.
- `setuptools<81` is pinned — Chatterbox's watermarker (`perth`) still imports `pkg_resources`, which newer setuptools dropped.

## Verify the stack

```bash
# GPU / CUDA
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Digital PDF -> Markdown (no OCR needed)
.venv/bin/markitdown book.pdf -o book.md

# Scanned PDF/images -> Markdown via OCR
.venv/bin/openocr --task doc --input_path book.pdf --output_path ./ocr-output --use_layout_detection --save_markdown --save_json

# TTS smoke test
.venv/bin/python test_tts.py && ffplay test.wav
```

## Voice cloning

Pass a reference clip to `model.generate(text, language_id="en", audio_prompt_path="narrator.wav")` to clone a voice you have permission to use.
