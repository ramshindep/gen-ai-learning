import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe =StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

pipe=pipe.to("cuda")

init_image=Image.open("cat.png").convert("RGB")
init_image=init_image.resize((512,512))

prompt="A cat wearing a beret, high quality, professional photograph"

images=pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,
    guidance_scale=7.5,
    num_inference_steps=50
).images

images[0].save("cat_beret.png")
