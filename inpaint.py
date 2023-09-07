import torch

from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

from stable_diffusion_server.utils import log_time

import numpy as np
import PIL.Image

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
# inpaint_and_upload_image?prompt=majestic tiger sitting on a bench&image_url=https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png&mask_url=https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png&save_path=tests/inpaint.webp
# inpainting can be used to upscale to 1080p


init_image = load_image(img_url).convert("RGB")
# mask_image = load_image(mask_url).convert("RGB")
# mask image all ones same shape as init_image

# here's a failed experiment: inpainting cannot be used as style transfer/it doesnt recreate ain image doing a full mask in this way
image_size = init_image.size
ones_of_size = np.ones(image_size, np.uint8) * 255
mask_image = PIL.Image.fromarray(ones_of_size.astype(np.uint8))
# mask_image = torch.ones_like(init_image) * 255
prompt = "A majestic tiger sitting on a bench, castle backdrop elegent anime"
num_inference_steps = 75
high_noise_frac = 0.7
with log_time("inpaint"):
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
        ).images[0]

image.save("inpaintfull.png")
