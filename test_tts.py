import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxMultilingualTTS.from_pretrained(
    device=device,
    t3_model="v3",
)

text = """
It was a strange, quiet morning.

Nothing moved beyond the window. For several minutes, he simply
stood there, watching the light move across the floor.

Then the telephone rang.
"""

audio = model.generate(
    text,
    language_id="en",
)

torchaudio.save(
    "test.wav",
    audio,
    model.sr,
)

print("Created test.wav")
