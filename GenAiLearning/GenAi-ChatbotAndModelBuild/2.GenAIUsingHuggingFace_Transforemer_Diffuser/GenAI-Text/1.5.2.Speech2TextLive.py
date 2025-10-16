import sounddevice as sound_device
import numpy as np
import torch
from transformers import pipeline,logging

logging.set_verbosity_error()

device ="cuda" if torch.cuda.is_available() else "mps"
model_pipeline=pipeline(model="openai/whisper-medium",device=device)

sample_rate=16000
duration=5

print("Speak now(Ctrl+C to stop) ...")  

try:
    while True:
        print("\nListening...")
        audio=sound_device.rec(int(sample_rate*duration),samplerate=sample_rate,channels=1,dtype='float32')
        sound_device.wait()
        audio_data=np.squeeze(audio)
        
        print("Transcribing...")
        result=model_pipeline({"array":audio_data,"sampling_rate":sample_rate})
        
        print("You said:,result['text']")
except KeyboardInterrupt:
    print("\nStopped...")