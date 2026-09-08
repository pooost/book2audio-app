# book2audio

Convert `.txt`, `.epub`, and `.pdf` files to audio (mp3) using free Microsoft Edge neural TTS (via [edge-tts](https://github.com/rany2/edge-tts)).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires `ffmpeg` on PATH (used by `pydub` to merge audio chunks).

## Usage

```bash
book2audio convert mybook.epub -o mybook.mp3
book2audio convert notes.txt -o notes.mp3 --voice en-US-GuyNeural --rate +10%
book2audio list-voices --lang en
```
