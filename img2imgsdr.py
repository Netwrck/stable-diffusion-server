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
pipe = DiffusionPipeline.from_pretrained("/mnt/d/models/lcm-ssd-1b", unet=unet, torch_dtype=torch.float16, variant="fp16"
# pipe = DiffusionPipeline.from_pretrained("segmind/SSD-1B", unet=unet, torch_dtype=torch.float16, variant="fp16"
                                         )

# use another main safetensors file
# pipe.main = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/diffusion_scripts/safetensors.py"
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
    # prompt = "attractive princess medieval lurid insanely detailed candlelit ultrarealistic 3d render dramatic light rain full body"
    prompt = "a handsome character from a fantasy book, a young wizard in modern clothes he has a broken expression, sad, he holds the wounded unconscious blonde in his arms, A hand-drawn mysterious, dark, handsome young adult Italian rebel, Seducive smile, Black eyes, tanned, black medium shoulder length hair, tattooed, Strong sexy jaw . High quality and fidelity vector. 2D white background, Shadows., Watercolor, trending on artstation, sharp focus, studio photo, intricate details, highly detailed, by greg rutkowski"
    prompt = "a handsome character from a fantasy book, seductive smile, abs, a young wizard. A hand-drawn mysterious, dark, handsome young adult Italian rebel, Seducive smile, Black eyes, tanned, black medium shoulder length hair, tattooed, Strong sexy jaw . High quality and fidelity vector. 2D white background, Shadows., Watercolor, trending on artstation, sharp focus, studio photo, intricate details, highly detailed"
    negative = "3 or 4 ears, never BUT ONE EAR, blurry, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, mangled teeth, weird teeth, poorly drawn eyes, blurry eyes, tan skin, oversaturated, teeth, poorly drawn, ugly, closed eyes, 3D, weird neck, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, extra limbs, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, text, logo, wordmark, writing, signature, blurry, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, Removed From Image Removed From Image flowers, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, long body, ((((mutated hands and fingers)))), cartoon, 3d ((disfigured)), ((bad art)), ((deformed)), ((extra limbs)), ((dose up)), ((b&w)), Wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), (poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), (extra limbs)), cloned face, (((disfigured))), out of frame ugly, extra limbs (bad anatomy), gross proportions (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, videogame, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured deformed cross-eye, ((body out of )), blurry, bad art, bad anatomy, 3d render, two faces, duplicate, coppy, multi, two, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly, disgusting, poorly drawn, childish, mutilated, mangled, old ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draf, blurry, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers"
    negative = "bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs"

    use_refiner = True
    with log_time('difuse'):
        with torch.inference_mode():
            # image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil").images[0]
            # only need 4 steps
            image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil", negative_prompt=negative, num_inference_steps=4).images[0]
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
                # for i in range(10, 50, 2):
                #     imageresult = refiner(prompt=prompt, image=image[None, :],
                #                     num_inference_steps=i).images[0]
                #     imageresult.save("princess" + str(i) + ".png")
                # imageresult = refiner(prompt=prompt, image=image[None, :]).images[0]
                imageresult = refiner(prompt=prompt, image=image[None, :], negative_prompt=negative).images[0]
    # image.save("princess.png")
    imageresult.save("wiz2-nonega5pldblnegsegmoe.png")


