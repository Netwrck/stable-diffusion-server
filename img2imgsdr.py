import PIL.Image

from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
import torch

import numpy as np

from stable_diffusion_server.utils import log_time

# pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

# pipe = DiffusionPipeline.from_pretrained(
#     "models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# pipe.watermark = None
#
# pipe.to("cuda")



unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-ssd-1b", torch_dtype=torch.float16, variant="fp16")
pipe = DiffusionPipeline.from_pretrained("segmind/SSD-1B", unet=unet, torch_dtype=torch.float16, variant="fp16")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.watermark = None

pipe.to("cuda")


refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.watermark = None

refiner.to("cuda")


# saving mem
# text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
# img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
# inpaint = StableDiffusionInpaintPipeline(**text2img.components)
def make_image(prompt):
    use_refiner = True
    with log_time('diffuse'):
        with torch.inference_mode():
            image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil", num_inference_steps=4).images[0]
            # experiment try deleting a whole bunch of pixels and see if the refiner can recreate them
            # delete top 30% of pixels
            # image = image[0:0.7]
            # pixels to delete
            # pixels_to_delete = int(0.3 * 1024)
            # delete top 30% of pixels
            # image.save("latent.png")
            # image_data = PIL.Image.fromarray(image)
            # image_data.save("latent.png")

            # image = np.array(image)
            # pixels_to_delete = int(0.3 * image.shape[0])
            # idx_to_delete = np.ones(image.shape[0], dtype=bool, device="cuda")
            # idx_to_delete[:pixels_to_delete] = False
            # image[idx_to_delete] = [0, 0, 0]

            # image_data = PIL.Image.fromarray(image)
            # image_data.save("latentcleared.png")
            if use_refiner:
                image = refiner(prompt=prompt, image=image[None, :]).images[0]
            return image


if __name__ == "__main__":

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    prompt = "attractive princess medieval lurid insanely detailed candlelit ultrarealistic 3d render dramatic light rain full body skin clad p"
    use_refiner = True
    with log_time('difuse'):
        with torch.inference_mode():
            # image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil").images[0]
            # only need 4 steps
            image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil", num_inference_steps=4).images[0]
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
            # pixels_to_delete = int(0.3 * image.shape[0])
            # idx_to_delete = np.ones(image.shape[0], dtype=bool, device="cuda")
            # idx_to_delete[:pixels_to_delete] = False
            # image[idx_to_delete] = [0,0,0]

            # image_data = PIL.Image.fromarray(image)
            # image_data.save("latentcleared.png")

            if use_refiner:
                for i in range(10, 50, 2):
                    imageresult = refiner(prompt=prompt, image=image[None, :],
                                    num_inference_steps=i).images[0]
                    imageresult.save("princess" + str(i) + ".png")
                # imageresult = refiner(prompt=prompt, image=image[None, :]).images[0]
    # image.save("woutrefiner4s.png")



