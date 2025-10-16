import torch
from diffusers import StableDiffusionPipeline

# Set device

device ="cuda" if torch.cuda.is_available()else "mps"
torch.manual_seed(0)

pipe=StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
    
).to(device)

prompt="Astronaut on the horse"

image=pipe(prompt).images[0]

image.show()
