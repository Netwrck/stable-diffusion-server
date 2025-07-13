import math
import os
import traceback
from datetime import datetime
from io import BytesIO
from urllib.parse import quote_plus
import uuid
from pathlib import Path

# import tomesd

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLInpaintPipeline,
    LCMScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoPipelineForImage2Image,
    FluxPipeline,
    FluxControlNetPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
)
from diffusers.utils import load_image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from transformers import set_seed

from env import BUCKET_PATH, BUCKET_NAME
from stable_diffusion_server.bucket_api import check_if_blob_exists, upload_to_bucket
from stable_diffusion_server.bumpy_detection import detect_too_bumpy
from stable_diffusion_server.image_processing import (
    process_image_for_stable_diffusion,
)
from stable_diffusion_server.utils import log_time
from stable_diffusion_server.prompt_utils import (
    shorten_too_long_text,
    shorten_prompt_for_retry,
    remove_stopwords,
)

from stable_diffusion_server.custom_pipeline import CustomPipeline

try:
    import pillow_avif

    assert pillow_avif  # required to use avif
except Exception as e:
    logger.error(f"Error importing pillow_avif: {e}")

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set cache directory for model downloads
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "./models")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./models")

# Global variables for lazy loading
flux_pipe = None
img2img = None
inpaintpipe = None
pipe = None

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_flux_pipe():
    """Lazy load Flux pipeline to reduce memory usage"""
    global flux_pipe
    if flux_pipe is None:
        logger.info("Loading Flux Schnell pipeline...")
        clear_gpu_memory()
        
        try:
            flux_pipe = FluxPipeline.from_pretrained(
                "models/FLUX.1-schnell", 
                torch_dtype=torch.bfloat16,
                cache_dir="./models"
            )
            logger.info("Loaded Flux pipeline from local cache")
        except OSError:
            logger.info("Local model not found, downloading from hub...")
            flux_pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell", 
                torch_dtype=torch.bfloat16,
                cache_dir="./models"
            )
            logger.info("Downloaded Flux pipeline from hub")
        
        # Enable aggressive memory optimizations
        flux_pipe.enable_model_cpu_offload()
        flux_pipe.enable_sequential_cpu_offload()
        flux_pipe.enable_attention_slicing()
        if hasattr(flux_pipe, 'enable_vae_slicing'):
            flux_pipe.enable_vae_slicing()
        
        # Set device map for better memory management
        if hasattr(flux_pipe, 'device_map'):
            flux_pipe.device_map = 'auto'
            
        logger.info("Flux pipeline loaded successfully with memory optimizations")
    return flux_pipe
# Disable DFloat11 for now - causing CUDA memory issues
# try:
#     from dfloat11 import DFloat11Model
#     dfloat_path = os.getenv("DF11_MODEL_PATH", "models/DFloat11__FLUX.1-schnell-DF11")
#     DFloat11Model.from_pretrained(dfloat_path, device="cpu", bfloat16_model=flux_pipe.transformer)
# except Exception as e:
#     logger.error(f"Failed to load DFloat11 weights: {e}")

try:
    # Simplified ControlNet loading - just check if file exists for now
    if os.path.exists("models/controlnet.safetensors"):
        # We'll implement the actual loading later
        flux_controlnetpipe = "models/controlnet.safetensors"  # Store path for now
        logger.info("Found ControlNet at models/controlnet.safetensors (not loaded yet)")
    else:
        flux_controlnetpipe = None
        logger.warning("No ControlNet model found at models/controlnet.safetensors")
        
except Exception as e:
    logger.error(f"Failed to check for Flux ControlNet: {e}")
    flux_controlnetpipe = None


try:
    custom_pipeline = CustomPipeline(name="flux-schnell")
    # Only load custom controlnet if the specific safetensors file exists
    if os.path.exists("models/controlnet.safetensors"):
        custom_pipeline.load_controlnet("models/controlnet.safetensors")
except Exception as e:
    logger.error(f"Failed to load custom pipeline: {e}")
    custom_pipeline = None


# quantizing
# from optimum.quanto import freeze, qfloat8, quantize

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

# Replace old SDXL pipelines with Flux equivalents

def get_img2img_pipe():
    """Lazy load Flux img2img pipeline to reduce memory usage"""
    global img2img
    if img2img is None:
        logger.info("Loading Flux img2img pipeline...")
        try:
            img2img = FluxImg2ImgPipeline.from_pretrained(
                "models/FLUX.1-schnell", 
                torch_dtype=torch.bfloat16,
                cache_dir="./models"
            )
        except OSError:
            logger.info("Local model not found, downloading img2img from hub...")
            img2img = FluxImg2ImgPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell", 
                torch_dtype=torch.bfloat16,
                cache_dir="./models"
            )
        
        # Share components from main pipeline if available
        flux_pipeline = get_flux_pipe()
        if flux_pipeline is not None:
            img2img.text_encoder = flux_pipeline.text_encoder
            img2img.text_encoder_2 = flux_pipeline.text_encoder_2
            img2img.transformer = flux_pipeline.transformer
        
        img2img.enable_model_cpu_offload()
        img2img.enable_sequential_cpu_offload()
        img2img.enable_attention_slicing()
        if hasattr(img2img, 'enable_vae_slicing'):
            img2img.enable_vae_slicing()
            
        logger.info("Flux img2img pipeline loaded successfully")
    return img2img

def get_inpaint_pipe():
    """Lazy load Flux inpaint pipeline to reduce memory usage"""
    global inpaintpipe
    if inpaintpipe is None:
        logger.info("Loading Flux inpaint pipeline...")
        try:
            inpaintpipe = FluxInpaintPipeline.from_pretrained(
                "models/FLUX.1-schnell", 
                torch_dtype=torch.bfloat16,
                cache_dir="./models"
            )
        except OSError:
            logger.info("Local model not found, downloading inpaint from hub...")
            inpaintpipe = FluxInpaintPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell", 
                torch_dtype=torch.bfloat16,
                cache_dir="./models"
            )
        
        # Share components from main pipeline if available
        flux_pipeline = get_flux_pipe()
        if flux_pipeline is not None:
            inpaintpipe.text_encoder = flux_pipeline.text_encoder
            inpaintpipe.text_encoder_2 = flux_pipeline.text_encoder_2
            inpaintpipe.transformer = flux_pipeline.transformer
        
        inpaintpipe.enable_model_cpu_offload()
        inpaintpipe.enable_sequential_cpu_offload()
        inpaintpipe.enable_attention_slicing()
        if hasattr(inpaintpipe, 'enable_vae_slicing'):
            inpaintpipe.enable_vae_slicing()
            
        logger.info("Flux inpaint pipeline loaded successfully")
    return inpaintpipe

# Use the same pipeline for refiner
def get_inpaint_refiner():
    """Get inpaint refiner pipeline"""
    return get_inpaint_pipe()

# Set main pipe to flux_pipe for backwards compatibility
def get_pipe():
    """Get main pipeline for backwards compatibility"""
    return get_flux_pipe()


def generate_controlnet_image_bytes(prompt: str, image: Image.Image, retries=3):
    """Generate image from prompt and image path"""
    if not custom_pipeline:
        raise Exception("Pipeline not initialized")
    image_bytes = custom_pipeline.generate(prompt=prompt, image=image)
    return image_bytes


@app.get("/controlnet_image")
def controlnet_image(prompt: str, image_path: str, save_path: str = "", retries=3):
    """Generate image from prompt and image path"""
    if not custom_pipeline:
        return Response(status_code=500, content="Pipeline not initialized")
    input_image = load_image(image_path)
    image_bytes = generate_controlnet_image_bytes(
        prompt=prompt, image=input_image, retries=retries
    )
    if not image_bytes:
        return Response(status_code=500, content="Failed to generate image")

    if save_path:
        path_components = save_path.split("/")[0:-1]
        final_name = save_path.split("/")[-1]
        save_path = "/".join(path_components) + quote_plus(final_name)
        if check_if_blob_exists(save_path):
            return JSONResponse(
                {"path": f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"}
            )
        upload_to_bucket(save_path, image_bytes, is_bytesio=False)
        return JSONResponse(
            {"path": f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"}
        )
    return StreamingResponse(content=iter([image_bytes]), media_type="image/webp")


@app.get("/text_to_image")
def text_to_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    save_path: str = "",
    n_steps: int = 8,
    extra_pipe_args: dict = None,
):
    if extra_pipe_args is None:
        extra_pipe_args = {}
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
    canny = True  # tmp only canny is working
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
    canny_bool = True  # tmp only canny is working

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
    use_refiner=False,
    n_refiner_steps=20,
    extra_refiner_pipe_args=None,
):
    if extra_refiner_pipe_args is None:
        extra_refiner_pipe_args = {}
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
                    num_inference_steps=4,
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
                    num_inference_steps=4,
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
            prompt = (
                remove_stopwords(prompt)
                if attempt == 0
                else shorten_prompt_for_retry(prompt)
            )
            if not prompt:
                raise err
    # todo refine
    # if image is not None and use_refiner:
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
    # Disabled for now - needs proper scheduler handling
    # if use_refiner:
    #     lcm_scheduler = img2img.scheduler
    #     img2img.scheduler = old_scheduler

    #     image = img2img(
    #         prompt=prompt,
    #         image=image,
    #         num_inference_steps=n_refiner_steps,
    #         strength=strength,
    #         **extra_refiner_pipe_args,
    #     ).images[0]
    #     # revert scheduler
    #     img2img.scheduler = lcm_scheduler
    # if detect_too_bumpy(image):
    #     if retries <= 0:
    #         raise Exception(
    #             "image too bumpy, retrying failed"
    #         )  # todo fix and just accept it?
    #     logger.info("image too bumpy, retrying once w different prompt detailed")
    #     return style_transfer_image_from_prompt(
    #         prompt + " detail",
    #         image_url,
    #         strength - 0.01,
    #         canny,
    #         input_pil,
    #         retries - 1,
    #     )

    return image_to_bytes(image)


def create_image_from_prompt(
    prompt, width, height, n_steps=5, extra_args=None, retries=3
):
    """Generate an image using the Flux Schnell pipeline with retries."""
    if extra_args is None:
        extra_args = {}

    # For testing, use fewer steps to speed up inference
    if os.getenv("TESTING", "false").lower() == "true":
        n_steps = min(n_steps, 4)  # Limit to 4 steps during testing
        logger.info(f"Testing mode: reducing steps to {n_steps}")

    block_width = width - (width % 64)
    block_height = height - (height % 64)
    prompt = shorten_too_long_text(prompt)
    generator = torch.Generator("cpu").manual_seed(extra_args.get("seed", 0))
    
    # Get the pipeline lazily
    flux_pipeline = get_flux_pipe()

    # Clear GPU memory before inference
    clear_gpu_memory()

    for attempt in range(retries + 1):
        try:
            with torch.inference_mode():
                # Update progress for long-running tasks
                if os.path.exists("progress.txt"):
                    with open("progress.txt", "w") as f:
                        f.write(datetime.now().strftime("%H:%M:%S"))
                
                image = flux_pipeline(
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
            clear_gpu_memory()  # Clear memory on failure
            if attempt == 0:
                prompt = remove_stopwords(prompt)
            else:
                prompt = shorten_prompt_for_retry(prompt)
            if not prompt:
                raise err

    if width != block_width or height != block_height:
        scale_up_ratio = max(width / block_width, height / block_height)
        image = image.resize(
            (
                math.ceil(block_width * scale_up_ratio),
                math.ceil(height * scale_up_ratio),
            )
        )
        image = image.crop((0, 0, width, height))

    # Skip bumpy detection in testing mode for speed
    if os.getenv("TESTING", "false").lower() != "true":
        if detect_too_bumpy(image):
            if retries <= 2:
                logger.info("image too bumpy, retrying once w different prompt detailed")
                return create_image_from_prompt(
                    prompt + " detail", width, height, n_steps + 1, extra_args, retries - 1
                )
            else:
                logger.warning("image too bumpy after 2 retries, returning anyway")
                # Return the image anyway after 2 retries to prevent infinite recursion

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
    num_inference_steps = 4
    high_noise_frac = 0.7

    for attempt in range(retries + 1):
        try:
            image = inpaintpipe(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
                strength=1.0 - high_noise_frac,  # Convert denoising_start to strength
                output_type="pil",  # Flux doesn't support latent output
            ).images[0]
            break
        except Exception as e:
            if attempt >= retries:
                traceback.print_exc()
                raise
            logger.warning(
                f"Inpainting failed on attempt {attempt + 1}/{retries}: {e}"
            )
            prompt = (
                remove_stopwords(prompt)
                if attempt == 0
                else shorten_prompt_for_retry(prompt)
            )
            if not prompt:
                raise e
    if image is not None:
        refiner_pipe = get_inpaint_refiner()
        image = refiner_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            strength=1.0 - high_noise_frac,  # Convert denoising_start to strength
        ).images[0]
    # try:
    #     # gc.collect()
    #     torch.cuda.empty_cache()
    # except Exception:
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


