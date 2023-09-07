from pathlib import Path

import numpy as np
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

from stable_diffusion_server.utils import log_time

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0",
# "models/stable-diffusion-xl-base-0.9",
    torch_dtype = torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe = pipe.to("cuda") #  # "LayerNormKernelImpl" not implemented for 'Half' error if its on cpu it cant do fp16
# idea composite: and re prompt img-img to support different sizes

# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
#
# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image = init_image.resize((768, 512))
# successfully inpaints a deleted area strength=0.75
# init_image = Image.open("/mnt/c/Users/leepenkman/Pictures/aiart/ainostalgic-colorful-relaxing-chill-realistic-cartoon-Charcoal-illustration-fantasy-fauvist-abstract-impressionist-watercolor-painting-Background-location-scenery-amazing-wonderful-Dog-Shelter-Worker-Dog.webp").convert("RGB")
# redo something? strength 1
# init_image = Image.open("/home/lee/code/sdif/mask.png").convert("RGB")
init_image = Image.open("/mnt/c/Users/leepenkman/Pictures/dogstretch.png").convert("RGB")
# init_image = Image.open("/mnt/c/Users/leepenkman/Pictures/dogcenter.png").convert("RGB")

# init_image = init_image.resize((1080, 1920))
init_image = init_image.resize((1920, 1080))
# init_image = init_image.resize((1024, 1024))

prompt = "A fantasy landscape, trending on artstation, beautiful amazing unreal surreal gorgeous impressionism"
prompt = "mouth open nostalgic colorful relaxing chill realistic cartoon Charcoal illustration fantasy fauvist abstract impressionist watercolor painting Background location scenery amazing wonderful Dog Shelter Worker Dog"

# images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
# images[0].save("fantasy_landscape.png")
#
# # url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
#
# init_image = load_image(url).convert("RGB")
# prompt = "a photo of an astronaut riding a horse on mars"
study_dir = "images/study2"
Path(study_dir).mkdir(parents=True, exist_ok=True)

with log_time("img2img"):
    with torch.inference_mode():
        # for strength in range(.1, 1, .1):
        for strength in np.linspace(.1, 1, 10):
            image = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=7.6).images[0]
            image.save(
                study_dir + "/fantasy_dogimgimgdogstretchopening" + str(strength) + "guidance_scale" + str(7.6) + ".png")
        #     # for guidance_scale in range(1, 10, .5):
        #     for guidance_scale in np.linspace(1, 100, 10):
        #         image = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale).images[0]
        #         image.save("images/study/fantasy_dogimgimgdogstretch" + str(strength) + "guidance_scale" + str(guidance_scale) + ".png")
        # image = pipe(prompt, image=init_image, strength=0.2, guidance_scale=7.5).images[0]
        # image.save("images/fantasy_dogimgimgdogstretch.png")
        # image.save("images/fantasy_dogimgimgdogcenter.png")
