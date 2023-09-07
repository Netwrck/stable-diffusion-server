import requests
import torch
from PIL import Image
from io import BytesIO

# from diffusers import StableDiffusionImg2ImgPipeline

# device = "cuda"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# # model_id_or_path = "models/stable-diffusion-xl-base-0.9"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
# pipe = pipe.to(device)

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0",
# "models/stable-diffusion-xl-base-0.9",
    torch_dtype = torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe = pipe.to("cuda")

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image = init_image.resize((768, 512))
init_image = init_image.resize((1080, 1920))

prompt = "A fantasy landscape, trending on artstation, beautiful amazing unreal surreal gorgeous impressionism"

# images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
# images[0].save("fantasy_landscape.png")
#
# # url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
#
# init_image = load_image(url).convert("RGB")
# prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, image=init_image).images[0]
image.save("fantasy_landscape.png")
