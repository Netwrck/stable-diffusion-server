import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
# model_id_or_path = "models/stable-diffusion-xl-base-0.9"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
pipe = pipe.to(device)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = Image.open("/mnt/c/Users/leepenkman/Pictures/aiknight-neon-punk-fantasy-art-good-looking-trending-fantastic-1.webp").convert("RGB")
# init_image = init_image.resize((768, 512))
init_image = init_image.resize((1920, 1080))

prompt = "knight neon punk fantasy art good looking trending fantastic"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("fantasy_landscape.png")
