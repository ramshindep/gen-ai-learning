from transformers import pipeline
import IPython.display as ipd
from scipy.io.wavfile import write
import torch
import numpy as np

pipe=pipeline(
    "text-to-audio", model="facebook/musicgen-small", device="cuda",torch_dtype=torch.float16
)

data = pipe(
    "Hi, this is my first audio"
) 



audio_int16 = np.int16(data["audio"][0] * 32767)

write("output.wav", data["sampling_rate"], audio_int16)


del pipe
torch.cuda.empty_cache()