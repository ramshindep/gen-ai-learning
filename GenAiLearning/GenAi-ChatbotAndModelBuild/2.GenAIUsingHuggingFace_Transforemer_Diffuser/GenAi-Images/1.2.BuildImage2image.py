import os
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("input_image.png", "rb")as image_file:
    response=client.images.create_variation(
        image=image_file,
        n=1,
        size="1024x1024"
    )
    
image_url=response.data[0].url

response=requests.get(image_url)
output_image=Image.open(BytesIO(response.content))
output_image.save("output_variation.png")
print("Variation image saved as output_variation.png")
output_image.show()