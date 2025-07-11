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
import uuid

# import tomesd

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
    LCMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    KDPM2AncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoPipelineForImage2Image,
    FluxPipeline,
    FluxControlNetPipeline,
)
from diffusers.utils import load_image
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.responses import JSONResponse
from transformers import set_seed

from env import BUCKET_PATH, BUCKET_NAME
from stable_diffusion_server.bucket_api import check_if_blob_exists, upload_to_bucket
from stable_diffusion_server.bumpy_detection import detect_too_bumpy
from stable_diffusion_server.image_processing import process_image_for_stable_diffusion
from stable_diffusion_server.utils import log_time
from stable_diffusion_server.prompt_utils import (
    shorten_too_long_text,
    shorten_prompt_for_retry,
    remove_stopwords,
)

try:
    import pillow_avif

    assert pillow_avif  # required to use avif
except Exception as e:
    logger.error(f"Error importing pillow_avif: {e}")

# model_name = "models/SSD-1B"
model_name = "models/ProteusV0.2"
# model_name = "dataautogpt3/ProteusV0.2"
# try:
#     unet = UNet2DConditionModel.from_pretrained(
#         "models/lcm-ssd-1b", torch_dtype=torch.float16, variant="fp16"
#     )
# except OSError as e:
#     unet = UNet2DConditionModel.from_pretrained(
#         "latent-consistency/lcm-ssd-1b", torch_dtype=torch.float16, variant="fp16"
#     )

try:
    # pipe = DiffusionPipeline.from_pretrained(
    #     "models/SSD-1B", unet=unet, torch_dtype=torch.float16, variant="fp16"
    # )
    pipe = DiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16, variant="fp16"
    )
except OSError as e:
    # pipe = DiffusionPipeline.from_pretrained(
    #     "segmind/SSD-1B", unet=unet, torch_dtype=torch.float16, variant="fp16"
    # )
    pipe = DiffusionPipeline.from_pretrained(
        "dataautogpt3/ProteusV0.2", torch_dtype=torch.float16, variant="fp16"
    )

old_scheduler = pipe.scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

if os.getenv("LOAD_LCM_LORA", "0") == "1":
    if os.path.exists("models/lcm-lora-sdxl"):
        pipe.load_lora_weights("models/lcm-lora-sdxl", adapter_name="lcm")
    else:
        pipe.load_lora_weights(
            "latent-consistency/lcm-lora-sdxl", adapter_name="lcm"
        )
    pipe.set_adapters(["lcm"], adapter_weights=[1.0])

# Load Flux Schnell pipeline for efficient text-to-image
flux_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
)
flux_pipe.enable_model_cpu_offload()
try:
    from dfloat11 import DFloat11Model
    dfloat_path = os.getenv("DF11_MODEL_PATH", "DFloat11/FLUX.1-schnell-DF11")
    DFloat11Model.from_pretrained(
        dfloat_path,
        device="cpu",
        bfloat16_model=flux_pipe.transformer,
    )
except Exception as e:
    logger.error(f"Failed to load DFloat11 weights: {e}")

try:
    flux_controlnet = ControlNetModel.from_pretrained(
        "black-forest-labs/flux-controlnet-canny", torch_dtype=torch.bfloat16
    )
    flux_controlnetpipe = FluxControlNetPipeline(
        controlnet=flux_controlnet, **flux_pipe.components
    )
    flux_controlnetpipe.enable_model_cpu_offload()
    try:
        lora_path = os.getenv(
            "CONTROLNET_LORA", "black-forest-labs/flux-controlnet-line-lora"
        )
        flux_controlnetpipe.load_lora_weights(lora_path, adapter_name="line")
        flux_controlnetpipe.set_adapters(["line"], adapter_weights=[1.0])
    except Exception as e:
        logger.error(f"Failed to load ControlNet LoRA: {e}")
except Exception as e:
    logger.error(f"Failed to load Flux ControlNet: {e}")
    flux_controlnetpipe = None


# quantizing
from optimum.quanto import freeze, qfloat8, quantize

# print(pipe.components)
# # # Quantize and freeze the text_encoder
# text_encoder = pipe.text_encoder
# quantize(text_encoder, weights=qfloat8)
# freeze(text_encoder)
# pipe.text_encoder = text_encoder
#
# # Quantize and freeze the text_encoder_2
# text_encoder_2 = pipe.text_encoder_2
# quantize(text_encoder_2, weights=qfloat8)
# freeze(text_encoder_2)
# pipe.text_encoder_2 = text_encoder_2


# Quantize and freeze the text_encoder_2
# text_encoder_3 = pipe.text_encoder_3
# quantize(text_encoder_3, weights=qfloat8)
# freeze(text_encoder_3)
# pipe.text_encoder_3 = text_encoder_3


# move unet too

# unet = pipe.unet
# quantize(unet, weights=qfloat8)
# freeze(unet)
# pipe.unet = unet

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

# mem efficient
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# pipe.to("cuda")

all_components = pipe.components
# all_components.pop("scheduler")
# all_components.pop("text_encoder")
# all_components.pop("text_encoder_2")
# all_components.pop("tokenizer")
# all_components.pop("tokenizer_2")

img2img = AutoPipelineForImage2Image.from_pipe(pipe)
img2img.watermark = None


# mem efficient
img2img.enable_attention_slicing()
img2img.enable_vae_slicing()
# img2img.to("cuda")
# img2img.enable_xformers_memory_efficient_attention()
img2img.enable_model_cpu_offload()
img2img.enable_sequential_cpu_offload()

# # Quantize and freeze the text_encoder
# text_encoder = img2img.text_encoder
# quantize(text_encoder, weights=qfloat8)
# freeze(text_encoder)
# img2img.text_encoder = text_encoder

# # Quantize and freeze the text_encoder_2
# text_encoder_2 = img2img.text_encoder_2
# quantize(text_encoder_2, weights=qfloat8)
# freeze(text_encoder_2)
# img2img.text_encoder_2 = text_encoder_2

# pipe = DiffusionPipeline.from_pretrained(
#     "models/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
#     # safety_checker=None,
# )  # todo try torch_dtype=float16
pipe.watermark = None


# deepcache
# from DeepCache import DeepCacheSDHelper

# helper = DeepCacheSDHelper(pipe=pipe)
# helper.set_params(
#     cache_interval=3,
#     cache_branch_id=0,
# )
# helper.enable()
# token merging
# tomesd.apply_patch(pipe, ratio=0.2)  # light speedup


refiner = DiffusionPipeline.from_pretrained(
    # "stabilityai/stable-diffusion-xl-refiner-1.0",
    # "dataautogpt3/OpenDalle",
    model_name,
    # "models/SSD-1B",
    unet=pipe.unet,
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,  # safer to use bfloat?
    use_safetensors=True,
    variant="fp16",  # remember not to download the big model
)

# refiner = pipe  # same model in this case
# refiner.scheduler = old_scheduler
# tomesd.apply_patch(refiner, ratio=0.2)  # light speedup

# refiner.schedu

refiner.watermark = None
# refiner.to("cuda")
refiner.enable_model_cpu_offload()
refiner.enable_sequential_cpu_offload()

# {'scheduler', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'unet', 'vae'} can be passed in from existing model
# inpaintpipe = StableDiffusionInpaintPipeline(**pipe.components)
inpaintpipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    # "models/stable-diffusion-xl-base-1.0",
    model_name,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    scheduler=pipe.scheduler,
    text_encoder=pipe.text_encoder,
    text_encoder_2=pipe.text_encoder_2,
    tokenizer=pipe.tokenizer,
    tokenizer_2=pipe.tokenizer_2,
    unet=pipe.unet,
    vae=pipe.vae,
    # load_connected_pipeline=
)
inpaintpipe.watermark = None
# inpaintpipe.enable_model_cpu_offload()

controlnet_conditioning_scale = 0.5  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
)
# controlnet.to("cuda")

controlnetpipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    # "stabilityai/stable-diffusion-xl-base-1.0",
    model_name,
    controlnet=controlnet,
    **pipe.components,
)
# controlnetpipe.to("cuda")
controlnetpipe.watermark = None

# efficiency
controlnetpipe.enable_model_cpu_offload()
controlnetpipe.enable_sequential_cpu_offload()

controlnetpipe.enable_attention_slicing()
controlnetpipe.enable_vae_slicing()

# # Quantize and freeze the text_encoder
# text_encoderz = controlnetpipe.text_encoder
# quantize(text_encoderz, weights=qfloat8)
# freeze(text_encoderz)
# controlnetpipe.text_encoder = text_encoderz

# # Quantize and freeze the text_encoder_2
# text_encoder_2z = controlnetpipe.text_encoder_2
# quantize(text_encoder_2z, weights=qfloat8)
# freeze(text_encoder_2z)
# controlnetpipe.text_encoder_2 = text_encoder_2z

# unet = controlnetpipe.unet
# quantize(unet, weights=qfloat8)
# freeze(unet)
# controlnetpipe.unet = unet


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
# inpaintpipe.to("cuda")
inpaintpipe.watermark = None
# inpaintpipe.register_to_config(requires_aesthetics_score=False)

# todo do we need this?
inpaint_refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    # "stabilityai/stable-diffusion-xl-refiner-1.0",
    model_name,
    text_encoder_2=inpaintpipe.text_encoder_2,
    vae=inpaintpipe.vae,
    torch_dtype=torch.float16,
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
# inpaint_refiner.to("cuda")
inpaint_refiner.watermark = None
# inpaint_refiner.register_to_config(requires_aesthetics_score=False)

n_steps = 5
n_refiner_steps = 10
high_noise_frac = 0.8
use_refiner = False


# efficiency

# inpaintpipe.enable_model_cpu_offload()
inpaint_refiner.enable_model_cpu_offload()
inpaint_refiner.enable_sequential_cpu_offload()
# pipe.enable_model_cpu_offload()
# refiner.enable_model_cpu_offload()
# img2img.enable_model_cpu_offload()


# pipe.enable_xformers_memory_efficient_attention()

# attn
# inpaintpipe.enable_xformers_memory_efficient_attention()
# inpaint_refiner.enable_xformers_memory_efficient_attention()
# pipe.enable_xformers_memory_efficient_attention()
# refiner.enable_xformers_memory_efficient_attention()
# img2img.enable_xformers_memory_efficient_attention()


# CFG Scale: Use a CFG scale of 8 to 7
# pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# errors a bit
# refiner.scheduler = KDPM2AncestralDiscreteScheduler.from_config(
#     refiner.scheduler.config
# )

# Sampler: DPM++ 2M SDE
# pipe.sa
# Scheduler: Karras
# img2img = StableDiffusionImg2ImgPipeline(**pipe.components)


# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()


# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True, backend="eager")
# this can cause errors on some inputs so consider disabling it
# pipe.unet = torch.compile(pipe.unet)
# refiner.unet = torch.compile(refiner.unet)#, mode="reduce-overhead", fullgraph=True)
# compile the inpainters - todo reuse the other unets? swap out the models for others/del them so they share models and can be swapped efficiently
inpaintpipe.unet = pipe.unet
inpaint_refiner.unet = refiner.unet
# inpaintpipe.unet = torch.compile(inpaintpipe.unet)
# inpaint_refiner.unet = torch.compile(inpaint_refiner.unet)

# img2img.unet = pipe.unet
# img2img.unet = torch.compile(img2img.unet, mode="reduce-overhead", fullgraph=True, backend="eager")

app = FastAPI(
    # openapi_url="/static/openapi.json",
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

# Simple health check endpoint
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

negative = "3 or 4 ears, never BUT ONE EAR, blurry, unclear, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, mangled teeth, weird teeth, poorly drawn eyes, blurry eyes, tan skin, oversaturated, teeth, poorly drawn, ugly, closed eyes, 3D, weird neck, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, extra limbs, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, text, logo, wordmark, writing, signature, blurry, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, Removed From Image Removed From Image flowers, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, long body, ((((mutated hands and fingers)))), cartoon, 3d ((disfigured)), ((bad art)), ((deformed)), ((extra limbs)), ((dose up)), ((b&w)), Wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), (poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), (extra limbs)), cloned face, (((disfigured))), out of frame ugly, extra limbs (bad anatomy), gross proportions (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, videogame, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured deformed cross-eye, ((body out of )), blurry, bad art, bad anatomy, 3d render, two faces, duplicate, coppy, multi, two, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly, disgusting, poorly drawn, childish, mutilated, mangled, old ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draf, blurry, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers"
negative2 = "ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers"
extra_pipe_args = {
    "guidance_scale": 1,
    "negative_prompt": negative,
    "negative_prompt2": negative2,
}
extra_refiner_pipe_args = {
    "guidance_scale": 7,
    "negative_prompt": negative,
    "negative_prompt2": negative2,
}


@app.get("/make_image")
def make_image(prompt: str, save_path: str = ""):
    if Path(save_path).exists():
        return FileResponse(save_path, media_type="image/png")
    with torch.inference_mode():
        image = pipe(
            prompt=prompt, num_inference_steps=n_steps, **extra_pipe_args
        ).images[0]
    if not save_path:
        save_path = f"images/{prompt}.png"
    image.save(save_path)
    return FileResponse(save_path, media_type="image/png")


@app.get("/create_and_upload_image")
async def create_and_upload_image(
    prompt: str, width: int = 1024, height: int = 1024, save_path: str = ""
):
    path_components = save_path.split("/")[0:-1]
    final_name = save_path.split("/")[-1]
    if not path_components:
        path_components = []
    save_path = "/".join(path_components) + quote_plus(final_name)
    path = get_image_or_create_upload_to_cloud_storage(prompt, width, height, save_path)
    return JSONResponse({"path": path})


@app.get("/inpaint_and_upload_image")
async def inpaint_and_upload_image(
    prompt: str, image_url: str, mask_url: str, save_path: str = ""
):
    path_components = save_path.split("/")[0:-1]
    final_name = save_path.split("/")[-1]
    if not path_components:
        path_components = []
    save_path = "/".join(path_components) + quote_plus(final_name)
    path = get_image_or_inpaint_upload_to_cloud_storage(
        prompt, image_url, mask_url, save_path
    )
    return JSONResponse({"path": path})


@app.get("/style_transfer_and_upload_image")
async def style_transfer_and_upload_image(
    prompt: str,
    image_url: str,
    save_path: str = "",
    strength: float = 0.6,
    canny: bool = False,
):
    canny=True # tmp only canny is working
    # todo also accept image bytes directly?
    path_components = save_path.split("/")[0:-1]
    final_name = save_path.split("/")[-1]
    if not path_components:
        path_components = []
    save_path = "/".join(path_components) + quote_plus(final_name)
    path = get_image_or_style_transfer_upload_to_cloud_storage(
        prompt, image_url, save_path, strength, canny
    )
    return JSONResponse({"path": path})


from fastapi import File, UploadFile


@app.post("/style_transfer_bytes_and_upload_image")
async def style_transfer_bytes_and_upload_image(
    prompt: str,
    image_url: str = None,
    save_path: str = "",
    strength: float = 0.6,
    canny: str = "true",
    image_file: UploadFile = File(None),
):

    uuid_str = str(uuid.uuid4())[:7]
    path_components = save_path.split("/")[0:-1]
    final_name = save_path.split("/")[-1]
    if canny == "true":
        canny_bool = True
    else:
        canny_bool = False
    canny_bool=True # tmp only canny is working

    if not path_components:
        path_components = []
    # Add UUID before the file extension
    if "." in final_name:
        name_parts = final_name.rsplit(".", 1)
        final_name = f"{name_parts[0]}_{uuid_str}.{name_parts[1]}"
    else:
        final_name = f"{final_name}_{uuid_str}"

    save_path = "/".join(path_components) + quote_plus(final_name)
    image_bytes = None
    if image_file:
        image_bytes = await image_file.read()
    elif image_url:
        path = get_image_or_style_transfer_upload_to_cloud_storage(
            prompt, image_url, save_path, strength, canny_bool
        )
    else:
        return JSONResponse(
            {"error": "Either image_url or image_file must be provided"},
            status_code=400,
        )

    path = get_image_or_style_transfer_upload_to_cloud_storage(
        prompt, image_url, save_path, strength, canny_bool, image_bytes
    )
    return JSONResponse({"path": path})


def get_image_or_style_transfer_upload_to_cloud_storage(
    prompt: str,
    image_url: str,
    save_path: str,
    strength=0.6,
    canny=False,
    image_bytes=None,
):
    prompt = shorten_too_long_text(prompt)
    save_path = shorten_too_long_text(save_path)
    # check exists - todo cache this
    if check_if_blob_exists(save_path):
        return f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"
    with torch.inference_mode():
        if image_bytes:
            input_image = Image.open(BytesIO(image_bytes))
            bio = style_transfer_image_from_prompt(
                prompt, image_url, strength, canny, input_pil=input_image
            )
        else:
            bio = style_transfer_image_from_prompt(prompt, image_url, strength, canny)
    if bio is None:
        return None  # error thrown in pool
    link = upload_to_bucket(save_path, bio, is_bytesio=True)
    return link


def get_image_or_create_upload_to_cloud_storage(
    prompt: str, width: int, height: int, save_path: str
):
    prompt = shorten_too_long_text(prompt)
    save_path = shorten_too_long_text(save_path)
    # check exists - todo cache this
    if check_if_blob_exists(save_path):
        return f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"
    with torch.inference_mode():
        bio = create_image_from_prompt(prompt, width, height)
    if bio is None:
        return None  # error thrown in pool
    link = upload_to_bucket(save_path, bio, is_bytesio=True)
    return link


def get_image_or_inpaint_upload_to_cloud_storage(
    prompt: str, image_url: str, mask_url: str, save_path: str
):
    prompt = shorten_too_long_text(prompt)
    save_path = shorten_too_long_text(save_path)
    # check exists - todo cache this
    if check_if_blob_exists(save_path):
        return f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"
    with torch.inference_mode():
        bio = inpaint_image_from_prompt(prompt, image_url, mask_url)
    if bio is None:
        return None  # error thrown in pool
    link = upload_to_bucket(save_path, bio, is_bytesio=True)
    return link


def is_defined(thing):
    # if isinstance(thing, pd.DataFrame):
    #     return not thing.empty
    if isinstance(thing, Image.Image):
        return True
    else:
        return thing is not None


def style_transfer_image_from_prompt(
    prompt,
    image_url: str | Image.Image,
    strength=0.6,
    canny=False,
    input_pil=None,
    retries=3,
):
    prompt = shorten_too_long_text(prompt)

    if not is_defined(input_pil):
        input_pil = load_image(image_url).convert("RGB")
    input_pil = process_image_for_stable_diffusion(input_pil)
    canny_image = None
    if canny:
        with log_time("canny"):
            in_image = np.array(input_pil)
            in_image = cv2.Canny(in_image, 100, 200)
            in_image = in_image[:, :, None]
            in_image = np.concatenate([in_image, in_image, in_image], axis=2)
            canny_image = Image.fromarray(in_image)
            set_seed(42)

    generator = torch.Generator("cpu").manual_seed(0)
    for attempt in range(retries + 1):
        try:
            if canny and flux_controlnetpipe:
                image = flux_controlnetpipe(
                    prompt=prompt,
                    image=canny_image,
                    num_inference_steps=n_steps,
                    guidance_scale=0.0,
                    generator=generator,
                    max_sequence_length=256,
                ).images[0]
            else:
                image = flux_pipe(
                    prompt=prompt,
                    width=input_pil.width,
                    height=input_pil.height,
                    guidance_scale=0.0,
                    num_inference_steps=n_steps,
                    generator=generator,
                    max_sequence_length=256,
                ).images[0]
            break
        except Exception as err:
            if attempt >= retries:
                raise
            logger.warning(
                f"Flux style transfer failed on attempt {attempt + 1}/{retries}: {err}"
            )
            prompt = remove_stopwords(prompt) if attempt == 0 else shorten_prompt_for_retry(prompt)
            if not prompt:
                raise err
    # todo refine
    # if image != None and use_refiner:
    #     image = refiner(
    #         prompt=prompt,
    #         # width=block_width,
    #         # height=block_height,
    #         # num_inference_steps=n_steps, # default
    #         # denoising_start=high_noise_frac,
    #         image=image,
    #     ).images[0]
    # if width != block_width or height != block_height:
    #     # resize to original size width/height
    #     # find aspect ratio to scale up to that covers the original img input width/height
    #     scale_up_ratio = max(width / block_width, height / block_height)
    #     image = image.resize((math.ceil(block_width * scale_up_ratio), math.ceil(height * scale_up_ratio)))
    #     # crop image to original size
    #     image = image.crop((0, 0, width, height))
    # try:
    #     # gc.collect()

    # add a refinement pass because the image is not always perfect/depending on the model if its not well tuned for LCM it might need more passes
    if use_refiner:
        lcm_scheduler = img2img.scheduler
        img2img.scheduler = old_scheduler

        image = img2img(
            prompt=prompt,
            image=image,
            num_inference_steps=n_refiner_steps,
            strength=strength,
            **extra_refiner_pipe_args,
        ).images[0]
        # revert scheduler
        img2img.scheduler = lcm_scheduler
    if detect_too_bumpy(image):
        if retries <= 0:
            raise Exception(
                "image too bumpy, retrying failed"
            )  # todo fix and just accept it?
        logger.info("image too bumpy, retrying once w different prompt detailed")
        return style_transfer_image_from_prompt(
            prompt + " detail",
            image_url,
            strength - 0.01,
            canny,
            input_pil,
            retries - 1,
        )

    return image_to_bytes(image)


def create_image_from_prompt(
    prompt, width, height, n_steps=5, extra_args=None, retries=3
):
    """Generate an image using the Flux Schnell pipeline with retries."""
    if extra_args is None:
        extra_args = {}

    block_width = width - (width % 64)
    block_height = height - (height % 64)
    prompt = shorten_too_long_text(prompt)
    generator = torch.Generator("cpu").manual_seed(extra_args.get("seed", 0))

    for attempt in range(retries + 1):
        try:
            image = flux_pipe(
                prompt=prompt,
                width=block_width,
                height=block_height,
                guidance_scale=0.0,
                num_inference_steps=n_steps,
                generator=generator,
                max_sequence_length=256,
            ).images[0]
            break
        except Exception as err:  # pragma: no cover - hardware/oom errors
            if attempt >= retries:
                raise
            logger.warning(
                f"Flux generation failed on attempt {attempt + 1}/{retries}: {err}"
            )
            if attempt == 0:
                prompt = remove_stopwords(prompt)
            else:
                prompt = shorten_prompt_for_retry(prompt)
            if not prompt:
                raise err

    if width != block_width or height != block_height:
        scale_up_ratio = max(width / block_width, height / block_height)
        image = image.resize(
            (math.ceil(block_width * scale_up_ratio), math.ceil(height * scale_up_ratio))
        )
        image = image.crop((0, 0, width, height))

    if detect_too_bumpy(image):
        if retries <= 0:
            raise Exception("image too bumpy, retrying failed")
        logger.info("image too bumpy, retrying once w different prompt detailed")
        return create_image_from_prompt(
            prompt + " detail", width, height, n_steps + 1, extra_args, retries - 1
        )

    return image_to_bytes(image)

# multiprocessing.set_start_method('spawn', True)
# processes_pool = Pool(1) # cant do too much at once or OOM errors happen
# def create_image_from_prompt_sync(prompt):
#     """have to call this sync to avoid OOM errors"""
#     return processes_pool.apply_async(create_image_from_prompt, args=(prompt,), ).wait()




def image_to_bytes(image):
    bs = BytesIO()

    bright_count = np.sum(np.array(image) > 0)
    if bright_count == 0:
        # we have a black image, this is an error likely we need a restart
        logger.info("restarting server to fix cuda issues (device side asserts)")
        logger.info("all black image")
        #     # todo fix device side asserts instead of restart to fix
        #     # todo only restart the correct gunicorn
        # this could be really annoying if your running other gunicorns on your machine which also get restarted
        os.system("/usr/bin/bash kill -SIGHUP `pgrep gunicorn`")
        os.system("kill -1 `pgrep gunicorn`")
        os.system("/usr/bin/bash kill -SIGHUP `pgrep uvicorn`")
        os.system("kill -1 `pgrep uvicorn`")

        return None
    image.save(bs, quality=85, optimize=True, format="webp")
    bio = bs.getvalue()
    return bio


def inpaint_image_from_prompt(prompt, image_url: str, mask_url: str, retries=3):
    prompt = shorten_too_long_text(prompt)
    # image = pipe(guidance_scale=7,prompt=prompt).images[0]

    init_image = load_image(image_url).convert("RGB")
    mask_image = load_image(mask_url).convert("RGB")  # why rgb for a 1 channel mask?
    # num_inference_steps = 75 # causes weird error ValueError: The combination of `original_steps x strength`: 50 x 1.0 is smaller than `num_inference_steps`: 75. Make sure to either reduce `num_inference_steps` to a value smaller than 50 or increase `strength` to a value higher than 1.5.
    num_inference_steps = 40
    high_noise_frac = 0.7

    generator = torch.Generator("cpu").manual_seed(0)
    for attempt in range(retries + 1):
        try:
            image = inpaintpipe(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
                denoising_start=high_noise_frac,
                output_type="latent",
            ).images[0]
            break
        except Exception as e:
            if attempt >= retries:
                traceback.print_exc()
                raise
            logger.warning(
                f"Inpainting failed on attempt {attempt + 1}/{retries}: {e}"
            )
            prompt = remove_stopwords(prompt) if attempt == 0 else shorten_prompt_for_retry(prompt)
            if not prompt:
                raise e
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

    # touch progress.txt file - if we dont do this we get restarted by supervisor/other processes for reliability
    with open("progress.txt", "w") as f:
        current_time = datetime.now().strftime("%H:%M:%S")
        f.write(f"{current_time}")
    return image_to_bytes(image)


