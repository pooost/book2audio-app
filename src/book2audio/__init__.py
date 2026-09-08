import os

# Runs before any submodule (huggingface_hub included) can be imported --
# huggingface_hub freezes HF_HUB_OFFLINE into a module constant at import
# time, so this only works if it lands here, not in cli.py: someone using
# book2audio.synth directly, without going through the CLI, still gets the
# offline-by-default behavior. synth.py/extract.py fall back to allowing a
# real download only if a needed file is actually missing from the cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

__version__ = "0.1.0"
