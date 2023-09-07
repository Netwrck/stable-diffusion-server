import PIL.Image

from diffusers import DiffusionPipeline
import torch

import numpy as np

from stable_diffusion_server.utils import log_time

pipe = DiffusionPipeline.from_pretrained(
    "models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
use_refiner = True
with log_time('diffuse'):
    with torch.inference_mode():
        image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil").images[0]
        # experiment try deleting a whole bunch of pixels and see if the refiner can recreate them
        # delete top 30% of pixels
        # image = image[0:0.7]
        #pixels to delete
        # pixels_to_delete = int(0.3 * 1024)
        # delete top 30% of pixels
        # image.save("latent.png")
        # image_data = PIL.Image.fromarray(image)
        # image_data.save("latent.png")

        # image = np.array(image)
        pixels_to_delete = int(0.3 * image.shape[0])
        idx_to_delete = np.ones(image.shape[0], dtype=bool, device="cuda")
        idx_to_delete[:pixels_to_delete] = False
        image[idx_to_delete] = [0,0,0]

        # image_data = PIL.Image.fromarray(image)
        # image_data.save("latentcleared.png")


        image = refiner(prompt=prompt, image=image[None, :]).images[0]



