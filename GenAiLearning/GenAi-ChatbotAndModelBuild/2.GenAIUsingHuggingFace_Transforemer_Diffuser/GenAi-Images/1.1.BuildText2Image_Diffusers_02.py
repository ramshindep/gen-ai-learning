from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import os

client =OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt="A fantasy landscape with castles and dragons, vibrant colors, highly detailed"

response=client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    n=1,
    size="1024x1024",
    quality="standard"
)

image_url=response.data[0].url

response=requests.get(image_url)
image=Image.open(BytesIO(response.content))
image.save("fantasy_landscape.png")
image.show()
                 