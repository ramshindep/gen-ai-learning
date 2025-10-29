from transformers import pipeline
import IPython.display as ipd
from scipy.io.wavfile import write
import torch
import numpy as np

# Create pipeline
pipe = pipeline(
    "text-to-audio",
    model="facebook/musicgen-small",
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16
)

# Generate audio
text = "Hi, this is my first audio."
data = pipe(text)

# Convert tensor to numpy (and move to CPU if needed)
audio = data["audio"][0]
if isinstance(audio, torch.Tensor):
    audio = audio.detach().cpu().numpy()

# Normalize to int16
audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)

# Save file
write("output.wav", data["sampling_rate"], audio_int16)

print("✅ Audio generated and saved as output.wav")