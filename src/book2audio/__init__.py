import os

# Runs before any submodule (huggingface_hub included) can be imported --
# huggingface_hub freezes HF_HUB_OFFLINE into a module constant at import
# time, so this only works if it lands here, not in cli.py: someone using
# book2audio.tts directly, without going through the CLI, still gets the
# offline-by-default behavior. Conversion never silently downloads a missing
# model -- tts/chatterbox_backend.py and ocr/backend.py raise a clear error
# instead; `book2audio setup-models` (core/models.py) is the only thing that
# explicitly flips this off, and only for the duration of that download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

__version__ = "0.1.0"
