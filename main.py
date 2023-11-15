import gc
import math
import multiprocessing
import os
import traceback
from datetime import datetime
from io import BytesIO
from itertools import permutations
from multiprocessing.pool import Pool
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import nltk
import torch
from PIL.Image import Image
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
    LCMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.utils import load_image
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.responses import JSONResponse

from env import BUCKET_PATH, BUCKET_NAME
from stable_diffusion_server.bucket_api import check_if_blob_exists, upload_to_bucket

unet = UNet2DConditionModel.from_pretrained("models/lcm-ssd-1b", torch_dtype=torch.float16, variant="fp16")
pipe = DiffusionPipeline.from_pretrained("models/SSD-1B", unet=unet, torch_dtype=torch.float16, variant="fp16")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# pipe = DiffusionPipeline.from_pretrained(
#     "models/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.bfloat16,
#     use_safetensors=True,
#     variant="fp16",
#     # safety_checker=None,
# )  # todo try torch_dtype=bfloat16
pipe.watermark = None

pipe.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16, # safer to use bfloat?
    use_safetensors=True,
    variant="fp16", #remember not to download the big model
)
refiner.watermark = None
refiner.to("cuda")

# {'scheduler', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'unet', 'vae'} can be passed in from existing model
# img2img = StableDiffusionImg2ImgPipeline(**pipe.components)
# inpaintpipe = StableDiffusionInpaintPipeline(**pipe.components)
inpaintpipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "models/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True,
    scheduler=pipe.scheduler,
    text_encoder=pipe.text_encoder,
    text_encoder_2=pipe.text_encoder_2,
    tokenizer=pipe.tokenizer,
    tokenizer_2=pipe.tokenizer_2,
    unet=pipe.unet,
    vae=pipe.vae,
    # load_connected_pipeline=
)
# # switch out to save gpu mem
# del inpaintpipe.vae
# del inpaintpipe.text_encoder_2
# del inpaintpipe.text_encoder
# del inpaintpipe.scheduler
# del inpaintpipe.tokenizer
# del inpaintpipe.tokenizer_2
# del inpaintpipe.unet
# inpaintpipe.vae = pipe.vae
# inpaintpipe.text_encoder_2 = pipe.text_encoder_2
# inpaintpipe.text_encoder = pipe.text_encoder
# inpaintpipe.scheduler = pipe.scheduler
# inpaintpipe.tokenizer = pipe.tokenizer
# inpaintpipe.tokenizer_2 = pipe.tokenizer_2
# inpaintpipe.unet = pipe.unet
# todo this should work
# inpaintpipe = StableDiffusionXLInpaintPipeline( # construct an inpainter using the existing model
#     vae=pipe.vae,
#     text_encoder_2=pipe.text_encoder_2,
#     text_encoder=pipe.text_encoder,
#     unet=pipe.unet,
#     scheduler=pipe.scheduler,
#     tokenizer=pipe.tokenizer,
#     tokenizer_2=pipe.tokenizer_2,
#     requires_aesthetics_score=False,
# )
inpaintpipe.to("cuda")
inpaintpipe.watermark = None
# inpaintpipe.register_to_config(requires_aesthetics_score=False)

# todo do we need this?
inpaint_refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=inpaintpipe.text_encoder_2,
    vae=inpaintpipe.vae,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="fp16",

    tokenizer_2=refiner.tokenizer_2,
    tokenizer=refiner.tokenizer,
    scheduler=refiner.scheduler,
    text_encoder=refiner.text_encoder,
    unet=refiner.unet,
)
# del inpaint_refiner.vae
# del inpaint_refiner.text_encoder_2
# del inpaint_refiner.text_encoder
# del inpaint_refiner.scheduler
# del inpaint_refiner.tokenizer
# del inpaint_refiner.tokenizer_2
# del inpaint_refiner.unet
# inpaint_refiner.vae = inpaintpipe.vae
# inpaint_refiner.text_encoder_2 = inpaintpipe.text_encoder_2
#
# inpaint_refiner.text_encoder = refiner.text_encoder
# inpaint_refiner.scheduler = refiner.scheduler
# inpaint_refiner.tokenizer = refiner.tokenizer
# inpaint_refiner.tokenizer_2 = refiner.tokenizer_2
# inpaint_refiner.unet = refiner.unet

# inpaint_refiner = StableDiffusionXLInpaintPipeline(
#     text_encoder_2=inpaintpipe.text_encoder_2,
#     vae=inpaintpipe.vae,
#     # the rest from the existing refiner
#     tokenizer_2=refiner.tokenizer_2,
#     tokenizer=refiner.tokenizer,
#     scheduler=refiner.scheduler,
#     text_encoder=refiner.text_encoder,
#     unet=refiner.unet,
#     requires_aesthetics_score=False,
# )
inpaint_refiner.to("cuda")
inpaint_refiner.watermark = None
# inpaint_refiner.register_to_config(requires_aesthetics_score=False)

n_steps = 40
high_noise_frac = 0.8

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()


# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# this can cause errors on some inputs so consider disabling it
# pipe.unet = torch.compile(pipe.unet)
# refiner.unet = torch.compile(refiner.unet)#, mode="reduce-overhead", fullgraph=True)
# compile the inpainters - todo reuse the other unets? swap out the models for others/del them so they share models and can be swapped efficiently
inpaintpipe.unet = pipe.unet
inpaint_refiner.unet = refiner.unet
# inpaintpipe.unet = torch.compile(inpaintpipe.unet)
# inpaint_refiner.unet = torch.compile(inpaint_refiner.unet)

app = FastAPI(
    openapi_url="/static/openapi.json",
    docs_url="/swagger-docs",
    redoc_url="/redoc",
    title="Generate Images Netwrck API",
    description="Character Chat API",
    # root_path="https://api.text-generator.io",
    version="1",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stopwords = nltk.corpus.stopwords.words("english")


@app.get("/make_image")
def make_image(prompt: str, save_path: str = ""):
    if Path(save_path).exists():
        return FileResponse(save_path, media_type="image/png")
    image = pipe(prompt=prompt, num_inference_steps=4).images[0]
    if not save_path:
        save_path = f"images/{prompt}.png"
    image.save(save_path)
    return FileResponse(save_path, media_type="image/png")


@app.get("/create_and_upload_image")
def create_and_upload_image(prompt: str, width: int=1024, height:int=1024, save_path: str = ""):
    path_components = save_path.split("/")[0:-1]
    final_name = save_path.split("/")[-1]
    if not path_components:
        path_components = []
    save_path = '/'.join(path_components) + quote_plus(final_name)
    path = get_image_or_create_upload_to_cloud_storage(prompt, width, height, save_path)
    return JSONResponse({"path": path})

@app.get("/inpaint_and_upload_image")
def inpaint_and_upload_image(prompt: str, image_url:str, mask_url:str, save_path: str = ""):
    path_components = save_path.split("/")[0:-1]
    final_name = save_path.split("/")[-1]
    if not path_components:
        path_components = []
    save_path = '/'.join(path_components) + quote_plus(final_name)
    path = get_image_or_inpaint_upload_to_cloud_storage(prompt, image_url, mask_url, save_path)
    return JSONResponse({"path": path})


def get_image_or_create_upload_to_cloud_storage(prompt:str,width:int, height:int, save_path:str):
    prompt = shorten_too_long_text(prompt)
    save_path = shorten_too_long_text(save_path)
    # check exists - todo cache this
    if check_if_blob_exists(save_path):
        return f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"
    bio = create_image_from_prompt(prompt, width, height)
    if bio is None:
        return None # error thrown in pool
    link = upload_to_bucket(save_path, bio, is_bytesio=True)
    return link
def get_image_or_inpaint_upload_to_cloud_storage(prompt:str, image_url:str, mask_url:str, save_path:str):
    prompt = shorten_too_long_text(prompt)
    save_path = shorten_too_long_text(save_path)
    # check exists - todo cache this
    if check_if_blob_exists(save_path):
        return f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"
    bio = inpaint_image_from_prompt(prompt, image_url, mask_url)
    if bio is None:
        return None # error thrown in pool
    link = upload_to_bucket(save_path, bio, is_bytesio=True)
    return link

# multiprocessing.set_start_method('spawn', True)
# processes_pool = Pool(1) # cant do too much at once or OOM errors happen
# def create_image_from_prompt_sync(prompt):
#     """have to call this sync to avoid OOM errors"""
#     return processes_pool.apply_async(create_image_from_prompt, args=(prompt,), ).wait()

def create_image_from_prompt(prompt, width, height):
    # round width and height down to multiple of 64
    block_width = width - (width % 64)
    block_height = height - (height % 64)
    prompt = shorten_too_long_text(prompt)
    use_refiner = True
    # image = pipe(prompt=prompt).images[0]
    try:
        image = pipe(prompt=prompt,
                     width=block_width,
                     height=block_height,
                     # denoising_end=high_noise_frac,
                     output_type='latent' if use_refiner else "pil",
                     # height=512,
                     # width=512,
                     num_inference_steps=4).images[0]  # normally uses 50 steps
    except Exception as e:
        # try rm stopwords + half the prompt
        # todo try prompt permutations
        logger.info(f"trying to shorten prompt of length {len(prompt)}")

        prompt = ' '.join((word for word in prompt if word not in stopwords))
        prompts = prompt.split()

        prompt = ' '.join(prompts[:len(prompts) // 2])
        logger.info(f"shortened prompt to: {len(prompt)}")
        image = None
        if prompt:
            try:
                image = pipe(prompt=prompt,
                             width=block_width,
                             height=block_height,
                             # denoising_end=high_noise_frac,
                             output_type='latent' if use_refiner else "pil",
                             # height=512,
                             # width=512,
                             num_inference_steps=4).images[0]  # normally uses 50 steps
            except Exception as e:
                # logger.info("trying to permute prompt")
                # # try two swaps of the prompt/permutations
                # prompt = prompt.split()
                # prompt = ' '.join(permutations(prompt, 2).__next__())
                logger.info(f"trying to shorten prompt of length {len(prompt)}")

                prompt = ' '.join((word for word in prompt if word not in stopwords))
                prompts = prompt.split()

                prompt = ' '.join(prompts[:len(prompts) // 2])
                logger.info(f"shortened prompt to: {len(prompt)}")

                try:
                    image = pipe(prompt=prompt,
                                 width=block_width,
                                 height=block_height,
                                 # denoising_end=high_noise_frac,
                                 output_type='latent' if use_refiner else "pil", # dont need latent yet - we refine the image at full res
                                 # height=512,
                                 # width=512,
                                 num_inference_steps=4).images[0]  # normally uses 50 steps
                except Exception as e:
                    # just error out
                    traceback.print_exc()
                    raise e
                    # logger.info("restarting server to fix cuda issues (device side asserts)")
                    # todo fix device side asserts instead of restart to fix
                    # todo only restart the correct gunicorn
                    # this could be really annoying if your running other gunicorns on your machine which also get restarted
                    # os.system("/usr/bin/bash kill -SIGHUP `pgrep gunicorn`")
                    # os.system("kill -1 `pgrep gunicorn`")
    # todo refine
    if image != None and use_refiner:
        image = refiner(
            prompt=prompt,
            # width=block_width,
            # height=block_height,
            # num_inference_steps=n_steps, # default
            # denoising_start=high_noise_frac,
            image=image,
        ).images[0]
    if width != block_width or height != block_height:
        # resize to original size width/height
        # find aspect ratio to scale up to that covers the original img input width/height
        scale_up_ratio = max(width / block_width, height / block_height)
        image = image.resize((math.ceil(block_width * scale_up_ratio), math.ceil(height * scale_up_ratio)))
        # crop image to original size
        image = image.crop((0, 0, width, height))
    # try:
    #     # gc.collect()
    #     torch.cuda.empty_cache()
    # except Exception as e:
    #     traceback.print_exc()
    #     logger.info("restarting server to fix cuda issues (device side asserts)")
    #     # todo fix device side asserts instead of restart to fix
    #     # todo only restart the correct gunicorn
    #     # this could be really annoying if your running other gunicorns on your machine which also get restarted
    #     os.system("/usr/bin/bash kill -SIGHUP `pgrep gunicorn`")
    #     os.system("kill -1 `pgrep gunicorn`")
    # save as bytesio
    bs = BytesIO()

    bright_count = np.sum(np.array(image) > 0)
    if bright_count == 0:
        # we have a black image, this is an error likely we need a restart
        logger.info("restarting server to fix cuda issues (device side asserts)")
        #     # todo fix device side asserts instead of restart to fix
        #     # todo only restart the correct gunicorn
        #     # this could be really annoying if your running other gunicorns on your machine which also get restarted
        os.system("/usr/bin/bash kill -SIGHUP `pgrep gunicorn`")
        os.system("kill -1 `pgrep gunicorn`")
        os.system("/usr/bin/bash kill -SIGHUP `pgrep uvicorn`")
        os.system("kill -1 `pgrep uvicorn`")

        return None
    image.save(bs, quality=85, optimize=True, format="webp")
    bio = bs.getvalue()
    # touch progress.txt file - if we dont do this we get restarted by supervisor/other processes for reliability
    with open("progress.txt", "w") as f:
        current_time = datetime.now().strftime("%H:%M:%S")
        f.write(f"{current_time}")
    return bio

def inpaint_image_from_prompt(prompt, image_url: str, mask_url: str):
    prompt = shorten_too_long_text(prompt)
    # image = pipe(prompt=prompt).images[0]

    init_image = load_image(image_url).convert("RGB")
    mask_image = load_image(mask_url).convert("RGB") # why rgb for a 1 channel mask?
    num_inference_steps = 75
    high_noise_frac = 0.7

    try:
        image = inpaintpipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            output_type="latent",
        ).images[0]  # normally uses 50 steps
    except Exception as e:
        # try rm stopwords + half the prompt
        # todo try prompt permutations
        logger.info(f"trying to shorten prompt of length {len(prompt)}")

        prompt = ' '.join((word for word in prompt if word not in stopwords))
        prompts = prompt.split()

        prompt = ' '.join(prompts[:len(prompts) // 2])
        logger.info(f"shortened prompt to: {len(prompt)}")
        image = None
        if prompt:
            try:
                image = pipe(
                    prompt=prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=num_inference_steps,
                    denoising_start=high_noise_frac,
                    output_type="latent",
                ).images[0]  # normally uses 50 steps
            except Exception as e:
                # logger.info("trying to permute prompt")
                # # try two swaps of the prompt/permutations
                # prompt = prompt.split()
                # prompt = ' '.join(permutations(prompt, 2).__next__())
                logger.info(f"trying to shorten prompt of length {len(prompt)}")

                prompt = ' '.join((word for word in prompt if word not in stopwords))
                prompts = prompt.split()

                prompt = ' '.join(prompts[:len(prompts) // 2])
                logger.info(f"shortened prompt to: {len(prompt)}")

                try:
                    image = inpaintpipe(
                            prompt=prompt,
                            image=init_image,
                            mask_image=mask_image,
                            num_inference_steps=num_inference_steps,
                            denoising_start=high_noise_frac,
                            output_type="latent",
                        ).images[0]  # normally uses 50 steps
                except Exception as e:
                    # just error out
                    traceback.print_exc()
                    raise e
                    # logger.info("restarting server to fix cuda issues (device side asserts)")
                    # todo fix device side asserts instead of restart to fix
                    # todo only restart the correct gunicorn
                    # this could be really annoying if your running other gunicorns on your machine which also get restarted
                    # os.system("/usr/bin/bash kill -SIGHUP `pgrep gunicorn`")
                    # os.system("kill -1 `pgrep gunicorn`")
    if image != None:
        image = inpaint_refiner(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,

        ).images[0]
    # try:
    #     # gc.collect()
    #     torch.cuda.empty_cache()
    # except Exception as e:
    #     traceback.print_exc()
    #     logger.info("restarting server to fix cuda issues (device side asserts)")
    #     # todo fix device side asserts instead of restart to fix
    #     # todo only restart the correct gunicorn
    #     # this could be really annoying if your running other gunicorns on your machine which also get restarted
    #     os.system("/usr/bin/bash kill -SIGHUP `pgrep gunicorn`")
    #     os.system("kill -1 `pgrep gunicorn`")
    # save as bytesio
    bs = BytesIO()

    bright_count = np.sum(np.array(image) > 0)
    if bright_count == 0:
        # we have a black image, this is an error likely we need a restart
        logger.info("restarting server to fix cuda issues (device side asserts)")
        #     # todo fix device side asserts instead of restart to fix
        #     # todo only restart the correct gunicorn
        #     # this could be really annoying if your running other gunicorns on your machine which also get restarted
        os.system("/usr/bin/bash kill -SIGHUP `pgrep gunicorn`")
        os.system("kill -1 `pgrep gunicorn`")
        os.system("/usr/bin/bash kill -SIGHUP `pgrep uvicorn`")
        os.system("kill -1 `pgrep uvicorn`")

        return None
    image.save(bs, quality=85, optimize=True, format="webp")
    bio = bs.getvalue()
    # touch progress.txt file - if we dont do this we get restarted by supervisor/other processes for reliability
    with open("progress.txt", "w") as f:
        current_time = datetime.now().strftime("%H:%M:%S")
        f.write(f"{current_time}")
    return bio



def shorten_too_long_text(prompt):
    if len(prompt) > 200:
        # remove stopwords
        prompt = prompt.split() # todo also split hyphens
        prompt = ' '.join((word for word in prompt if word not in stopwords))
        if len(prompt) > 200:
            prompt = prompt[:200]
    return prompt

# image = pipe(prompt=prompt).images[0]
#
# image.save("test.png")
# save all images
# for i, image in enumerate(images):
#     image.save(f"{i}.png")
