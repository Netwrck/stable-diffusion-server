import gc
import math
import os
import threading
import traceback
from contextlib import nullcontext
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
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    LCMScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoPipelineForImage2Image,
    FluxPipeline,
    FluxControlNetPipeline,
    FluxControlNetModel,
    FluxImg2ImgPipeline,
    FluxControlInpaintPipeline,
    FluxInpaintPipeline,
    DPMSolverMultistepScheduler,
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
from performance_optimizations import (
    build_inference_kwargs,
    env_bool,
    env_float,
    flux_optimizer,
    optimize_all_pipelines,
    resolve_flux_model_repo,
    resolve_proteus_model_repo,
)

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Set cache directory for model downloads
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "./models")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./models")

# Global variables for unified pipeline system
BASE_PIPE = None
IMG2IMG_PIPE = None  
INPAINT_PIPE = None
CANNY_PIPE = None
SDXL_PIPE = None
SDXL_IMG2IMG_PIPE = None
SDXL_CANNY_PIPE = None

# Legacy compatibility variables
flux_pipe = None
img2img = None
inpaintpipe = None
pipe = None
INFERENCE_LOCK = threading.RLock()


def inference_guard():
    if env_bool("SDIF_SERIALIZE_INFERENCE", True):
        return INFERENCE_LOCK
    return nullcontext()

def build_flux_pipes(model_repo: str | None = None):
    """Build shared Flux pipelines for text, img2img, and inpainting."""
    model_repo = model_repo or resolve_flux_model_repo()
    logger.info(f"Using Flux model: {model_repo}")

    base = FluxPipeline.from_pretrained(
        model_repo, 
        torch_dtype=torch.bfloat16,
        cache_dir="./models",
        local_files_only=env_bool("HF_LOCAL_ONLY", False),
    )
    maybe_apply_dfloat11(base)

    return base, None, None, None


def maybe_apply_dfloat11(pipeline) -> None:
    if not env_bool("ENABLE_DFLOAT11", False):
        return
    dfloat_path = os.getenv("DF11_MODEL_PATH", "models/DFloat11__FLUX.1-schnell-DF11")
    if not os.path.exists(dfloat_path) and env_bool("HF_LOCAL_ONLY", False):
        logger.warning(f"DFloat11 path not found with HF_LOCAL_ONLY=1: {dfloat_path}")
        return
    try:
        from dfloat11 import DFloat11Model

        DFloat11Model.from_pretrained(
            dfloat_path,
            device=os.getenv("DF11_DEVICE") or None,
            device_map=os.getenv("DF11_DEVICE_MAP", "auto"),
            bfloat16_model=pipeline.transformer,
            cache_dir="./models",
        )
        logger.info(f"Loaded DFloat11 Flux transformer weights: {dfloat_path}")
    except Exception as e:
        logger.warning(f"Failed to load DFloat11 weights: {e}")


def build_flux_canny_pipe(model_repo: str | None = None):
    """Build Flux-native Canny ControlNet lazily for edge-guided requests."""
    model_repo = model_repo or resolve_flux_model_repo()
    controlnet_repo = os.getenv(
        "FLUX_CANNY_CONTROLNET_REPO",
        "XLabs-AI/flux-controlnet-canny-diffusers",
    )
    try:
        canny_cn = FluxControlNetModel.from_pretrained(
            controlnet_repo,
            torch_dtype=torch.bfloat16,
            cache_dir="./models",
            use_safetensors=True,
            local_files_only=env_bool("HF_LOCAL_ONLY", False),
        )
        canny = FluxControlNetPipeline.from_pretrained(
            model_repo,
            controlnet=canny_cn,
            torch_dtype=torch.bfloat16,
            cache_dir="./models",
            local_files_only=env_bool("HF_LOCAL_ONLY", False),
        )
        logger.info(f"Loaded Flux Canny ControlNet: {controlnet_repo}")
        return canny
    except Exception as e:
        logger.warning(f"Failed to load Flux ControlNet: {e}")
        return None


def configure_sdxl_scheduler(pipeline):
    """Use a low-step-friendly SDXL scheduler unless LCM is explicitly enabled."""
    speed_lora = os.getenv("SDXL_SPEED_LORA", "").strip()
    if speed_lora:
        try:
            pipeline.load_lora_weights(speed_lora)
            if hasattr(pipeline, "fuse_lora"):
                pipeline.fuse_lora()
                pipeline.unload_lora_weights()
            logger.info(f"Loaded and fused SDXL speed LoRA: {speed_lora}")
        except Exception as e:
            logger.warning(f"Could not load SDXL speed LoRA {speed_lora}: {e}")

    scheduler_name = os.getenv("SDXL_SCHEDULER", "").strip().lower()
    if scheduler_name:
        try:
            from diffusers import (
                DDIMScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
            )

            schedulers = {
                "lcm": LCMScheduler,
                "ddim": DDIMScheduler,
                "euler": EulerDiscreteScheduler,
                "euler_a": EulerAncestralDiscreteScheduler,
                "dpm": DPMSolverMultistepScheduler,
            }
            cls = schedulers.get(scheduler_name)
            if cls is None:
                logger.warning(f"Unknown SDXL_SCHEDULER={scheduler_name!r}")
            else:
                kwargs = {}
                if cls is DPMSolverMultistepScheduler:
                    kwargs = {
                        "algorithm_type": os.getenv("SDXL_DPM_ALGORITHM", "sde-dpmsolver++"),
                        "timestep_spacing": os.getenv("SDXL_TIMESTEP_SPACING", "trailing"),
                    }
                pipeline.scheduler = cls.from_config(pipeline.scheduler.config, **kwargs)
                logger.info(f"Configured SDXL scheduler: {scheduler_name}")
                return pipeline
        except Exception as e:
            logger.warning(f"Could not configure SDXL_SCHEDULER={scheduler_name}: {e}")

    if env_bool("LOAD_LCM_LORA", False):
        lcm_path = os.getenv("LCM_LORA_PATH", "models/lcm-lora-sdxl")
        try:
            if os.path.exists(lcm_path):
                pipeline.load_lora_weights(lcm_path)
                if hasattr(pipeline, "fuse_lora"):
                    pipeline.fuse_lora()
                logger.info(f"Loaded and fused SDXL LCM LoRA: {lcm_path}")
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            return pipeline
        except Exception as e:
            logger.warning(f"Could not configure SDXL LCM scheduler: {e}")

    try:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type=os.getenv("SDXL_DPM_ALGORITHM", "sde-dpmsolver++"),
            timestep_spacing=os.getenv("SDXL_TIMESTEP_SPACING", "trailing"),
        )
        logger.info("Configured SDXL scheduler for low-step DPM-Solver++")
    except Exception as e:
        logger.warning(f"Could not configure SDXL DPM scheduler: {e}")
    return pipeline


def build_sdxl_pipes(model_repo: str | None = None):
    """Build Proteus/SDXL text and img2img pipelines lazily for style transfer/evals."""
    model_repo = model_repo or resolve_proteus_model_repo()
    logger.info(f"Using SDXL/Proteus model: {model_repo}")
    safetensors_mode = os.getenv("SDXL_USE_SAFETENSORS", "auto").strip().lower()
    if safetensors_mode == "auto":
        use_safetensors = True
        if os.path.isdir(model_repo) and not os.path.exists(os.path.join(model_repo, "unet", "diffusion_pytorch_model.safetensors")):
            use_safetensors = False
        if model_repo == "dataautogpt3/ProteusV0.4":
            use_safetensors = False
    else:
        use_safetensors = env_bool("SDXL_USE_SAFETENSORS", True)

    sdxl = StableDiffusionXLPipeline.from_pretrained(
        model_repo,
        torch_dtype=torch.float16,
        cache_dir="./models",
        use_safetensors=use_safetensors,
        local_files_only=env_bool("HF_LOCAL_ONLY", False),
    )
    configure_sdxl_scheduler(sdxl)
    sdxl_img2img = AutoPipelineForImage2Image.from_pipe(sdxl)
    return sdxl, sdxl_img2img

def initialize_unified_pipelines():
    """Initialize all pipelines using the unified factory"""
    global BASE_PIPE, IMG2IMG_PIPE, INPAINT_PIPE, CANNY_PIPE
    global flux_pipe, img2img, inpaintpipe, pipe
    
    if BASE_PIPE is None:
        logger.info("Initializing unified Flux pipeline system...")
        prepare_backend_memory("flux")
        clear_gpu_memory()
        
        BASE_PIPE, IMG2IMG_PIPE, INPAINT_PIPE, CANNY_PIPE = build_flux_pipes()
        
        # Apply comprehensive performance optimizations
        optimize_all_pipelines(BASE_PIPE, IMG2IMG_PIPE, INPAINT_PIPE, CANNY_PIPE)
        
        # Set legacy compatibility variables
        flux_pipe = BASE_PIPE
        img2img = IMG2IMG_PIPE
        inpaintpipe = INPAINT_PIPE
        pipe = BASE_PIPE
        
        logger.info("Unified Flux pipeline system initialized successfully with memory optimizations")


def initialize_sdxl_pipelines():
    """Initialize Proteus/SDXL pipelines for style transfer and evals."""
    global SDXL_PIPE, SDXL_IMG2IMG_PIPE

    if SDXL_PIPE is None:
        logger.info("Initializing SDXL/Proteus pipeline system...")
        prepare_backend_memory("sdxl")
        clear_gpu_memory()
        SDXL_PIPE, SDXL_IMG2IMG_PIPE = build_sdxl_pipes()
        optimize_all_pipelines(SDXL_PIPE, SDXL_IMG2IMG_PIPE)
        logger.info("SDXL/Proteus pipeline system initialized")

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def keep_multiple_pipelines_loaded() -> bool:
    return env_bool("SDIF_KEEP_MULTIPLE_PIPELINES", False)


def unload_flux_pipelines():
    """Drop Flux pipelines before loading SDXL/Proteus on constrained GPUs."""
    global BASE_PIPE, IMG2IMG_PIPE, INPAINT_PIPE, CANNY_PIPE
    global flux_pipe, img2img, inpaintpipe, pipe

    if not any([BASE_PIPE, IMG2IMG_PIPE, INPAINT_PIPE, CANNY_PIPE, flux_pipe, img2img, inpaintpipe]):
        return

    logger.info("Unloading Flux pipelines to free GPU memory")
    BASE_PIPE = None
    IMG2IMG_PIPE = None
    INPAINT_PIPE = None
    CANNY_PIPE = None
    flux_pipe = None
    img2img = None
    inpaintpipe = None
    pipe = None
    clear_gpu_memory()


def unload_sdxl_pipelines():
    """Drop SDXL/Proteus pipelines before loading Flux on constrained GPUs."""
    global SDXL_PIPE, SDXL_IMG2IMG_PIPE, SDXL_CANNY_PIPE

    if not any([SDXL_PIPE, SDXL_IMG2IMG_PIPE, SDXL_CANNY_PIPE]):
        return

    logger.info("Unloading SDXL/Proteus pipelines to free GPU memory")
    SDXL_PIPE = None
    SDXL_IMG2IMG_PIPE = None
    SDXL_CANNY_PIPE = None
    clear_gpu_memory()


def prepare_backend_memory(backend: str):
    if keep_multiple_pipelines_loaded():
        return
    backend = backend.strip().lower()
    if backend in {"sdxl", "proteus"}:
        unload_flux_pipelines()
    elif backend == "flux":
        unload_sdxl_pipelines()

def get_flux_pipe():
    """Get Flux pipeline using unified system"""
    if BASE_PIPE is None and flux_pipe is not None:
        return flux_pipe
    initialize_unified_pipelines()
    return BASE_PIPE
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


custom_pipeline = None


def get_custom_pipeline():
    """Load the legacy custom Flux pipeline only for endpoints that still use it."""
    global custom_pipeline
    if custom_pipeline is not None:
        return custom_pipeline
    if not env_bool("ENABLE_LEGACY_CUSTOM_PIPELINE", False):
        return None
    try:
        custom_pipeline = CustomPipeline(name="flux-schnell")
        if os.path.exists("models/controlnet.safetensors"):
            custom_pipeline.load_controlnet("models/controlnet.safetensors")
    except Exception as e:
        logger.error(f"Failed to load custom pipeline: {e}")
        custom_pipeline = None
    return custom_pipeline


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
    """Get img2img pipeline using unified system"""
    global IMG2IMG_PIPE, img2img
    if IMG2IMG_PIPE is None and img2img is not None:
        return img2img
    initialize_unified_pipelines()
    if IMG2IMG_PIPE is None:
        IMG2IMG_PIPE = FluxImg2ImgPipeline.from_pipe(BASE_PIPE)
        flux_optimizer.optimize_pipeline_fully(IMG2IMG_PIPE, "flux_img2img")
        img2img = IMG2IMG_PIPE
    return IMG2IMG_PIPE

def get_inpaint_pipe():
    """Get inpaint pipeline using unified system"""
    global INPAINT_PIPE, inpaintpipe
    if INPAINT_PIPE is None and inpaintpipe is not None:
        return inpaintpipe
    initialize_unified_pipelines()
    if INPAINT_PIPE is None:
        INPAINT_PIPE = FluxControlInpaintPipeline.from_pipe(BASE_PIPE)
        flux_optimizer.optimize_pipeline_fully(INPAINT_PIPE, "flux_inpaint")
        inpaintpipe = INPAINT_PIPE
    return INPAINT_PIPE

# Use the same pipeline for refiner
def get_inpaint_refiner():
    """Get inpaint refiner pipeline"""
    return get_inpaint_pipe()

# Set main pipe to flux_pipe for backwards compatibility
def get_pipe():
    """Get main pipeline for backwards compatibility"""
    if BASE_PIPE is None and pipe is not None:
        return pipe
    return get_flux_pipe()

def get_canny_pipe():
    """Get ControlNet Canny pipeline using unified system"""
    global CANNY_PIPE
    legacy_controlnet = globals().get("flux_controlnetpipe")
    if CANNY_PIPE is None and legacy_controlnet is not None and not isinstance(legacy_controlnet, str):
        return legacy_controlnet
    initialize_unified_pipelines()
    if CANNY_PIPE is None:
        CANNY_PIPE = build_flux_canny_pipe()
        if CANNY_PIPE is not None:
            flux_optimizer.optimize_pipeline_fully(CANNY_PIPE, "flux_controlnet")
    return CANNY_PIPE


def get_sdxl_pipe():
    """Get the Proteus/SDXL text-to-image pipeline."""
    initialize_sdxl_pipelines()
    return SDXL_PIPE


def get_sdxl_img2img_pipe():
    """Get the Proteus/SDXL img2img pipeline."""
    initialize_sdxl_pipelines()
    return SDXL_IMG2IMG_PIPE


def get_sdxl_canny_pipe():
    """Lazy SDXL ControlNet img2img pipeline sharing the resident Proteus UNet."""
    global SDXL_CANNY_PIPE
    if SDXL_CANNY_PIPE is None:
        base = get_sdxl_pipe()
        if base is None:
            return None
        repo = os.getenv("SDXL_CONTROLNET_REPO", "models/controlnet-canny-sdxl-1.0")
        try:
            from diffusers import StableDiffusionXLControlNetImg2ImgPipeline

            controlnet = ControlNetModel.from_pretrained(
                repo,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                local_files_only=env_bool("HF_LOCAL_ONLY", False),
            )
            SDXL_CANNY_PIPE = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
                base, controlnet=controlnet
            )
            SDXL_CANNY_PIPE.watermark = None
            if torch.cuda.is_available():
                SDXL_CANNY_PIPE.controlnet.to("cuda")
            SDXL_CANNY_PIPE.set_progress_bar_config(disable=True)
            logger.info(f"Loaded SDXL Canny ControlNet: {repo}")
        except Exception as e:
            logger.warning(f"Failed to load SDXL ControlNet {repo}: {e}")
            return None
    return SDXL_CANNY_PIPE


def normalize_save_path(save_path: str, suffix: str = "") -> str:
    if not save_path:
        return ""
    path_components = save_path.split("/")[:-1]
    final_name = save_path.split("/")[-1]
    if suffix:
        if "." in final_name:
            stem, ext = final_name.rsplit(".", 1)
            final_name = f"{stem}{suffix}.{ext}"
        else:
            final_name = f"{final_name}{suffix}"
    quoted_name = quote_plus(final_name)
    if not path_components:
        return quoted_name
    return "/".join([*path_components, quoted_name])


def generate_controlnet_image_bytes(prompt: str, image: Image.Image, retries=3):
    """Generate image from prompt and image path"""
    pipeline = get_custom_pipeline()
    if callable(pipeline) and hasattr(pipeline, "return_value"):
        pipeline = pipeline()
    if not pipeline:
        raise Exception("Pipeline not initialized")
    with inference_guard(), torch.inference_mode():
        image_bytes = pipeline.generate(prompt=prompt, image=image)
    return image_bytes


@app.get("/controlnet_image")
def controlnet_image(prompt: str, image_path: str, save_path: str = "", retries=3):
    """Generate image from prompt and image path"""
    if not get_custom_pipeline():
        return Response(status_code=500, content="Pipeline not initialized")
    input_image = load_image(image_path)
    image_bytes = generate_controlnet_image_bytes(
        prompt=prompt, image=input_image, retries=retries
    )
    if not image_bytes:
        return Response(status_code=500, content="Failed to generate image")

    if save_path:
        save_path = normalize_save_path(save_path)
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
    with inference_guard(), torch.inference_mode():
        backend = os.getenv("TEXT_TO_IMAGE_BACKEND", "flux").strip().lower()
        text_pipe = get_sdxl_pipe() if backend in {"sdxl", "proteus"} else get_pipe()
        pipe_args = build_inference_kwargs(
            backend,
            "text",
            steps=n_steps,
            extra_args=extra_pipe_args,
        )
        image = text_pipe(
            prompt=prompt,
            width=width,
            height=height,
            **pipe_args,
        ).images[0]
    if not save_path:
        save_path = f"images/{prompt}.png"
    image.save(save_path)
    return FileResponse(save_path, media_type="image/png")


@app.get("/create_and_upload_image")
async def create_and_upload_image(
    prompt: str, width: int = 1024, height: int = 1024, save_path: str = ""
):
    save_path = normalize_save_path(save_path)
    path = get_image_or_create_upload_to_cloud_storage(prompt, width, height, save_path)
    return JSONResponse({"path": path})


@app.get("/inpaint_and_upload_image")
async def inpaint_and_upload_image(
    prompt: str, image_url: str, mask_url: str, save_path: str = ""
):
    save_path = normalize_save_path(save_path)
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
    backend: str = "",
    n_steps: int | None = None,
):
    # todo also accept image bytes directly?
    save_path = normalize_save_path(save_path)
    path = get_image_or_style_transfer_upload_to_cloud_storage(
        prompt, image_url, save_path, strength, canny, backend=backend, n_steps=n_steps
    )
    return JSONResponse({"path": path})


@app.post("/style_transfer_bytes_and_upload_image")
async def style_transfer_bytes_and_upload_image(
    prompt: str,
    image_url: str = None,
    save_path: str = "",
    strength: float = 0.6,
    canny: str = "true",
    backend: str = "",
    n_steps: int | None = None,
    image_file: UploadFile = File(None),
):

    uuid_str = str(uuid.uuid4())[:7]
    if canny == "true":
        canny_bool = True
    else:
        canny_bool = False

    save_path = normalize_save_path(save_path, suffix=f"_{uuid_str}")
    image_bytes = None
    if image_file:
        image_bytes = await image_file.read()
    elif not image_url:
        return JSONResponse(
            {"error": "Either image_url or image_file must be provided"},
            status_code=400,
        )

    path = get_image_or_style_transfer_upload_to_cloud_storage(
        prompt, image_url, save_path, strength, canny_bool, image_bytes, backend=backend, n_steps=n_steps
    )
    return JSONResponse({"path": path})


def get_image_or_style_transfer_upload_to_cloud_storage(
    prompt: str,
    image_url: str,
    save_path: str,
    strength=0.6,
    canny=False,
    image_bytes=None,
    backend: str = "",
    n_steps: int | None = None,
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
                prompt, image_url, strength, canny, input_pil=input_image, backend=backend, n_steps=n_steps
            )
        else:
            bio = style_transfer_image_from_prompt(prompt, image_url, strength, canny, backend=backend, n_steps=n_steps)
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
    if isinstance(thing, str):
        return thing != ""
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
    backend: str = "",
    n_steps: int | None = None,
):
    if extra_refiner_pipe_args is None:
        extra_refiner_pipe_args = {}
    prompt = shorten_too_long_text(prompt)
    backend = (backend or os.getenv("STYLE_TRANSFER_BACKEND", "sdxl")).strip().lower()
    if backend == "proteus":
        backend = "sdxl"

    if not is_defined(input_pil):
        input_pil = load_image(image_url).convert("RGB")
    input_pil = process_image_for_stable_diffusion(input_pil)
    canny_image = None
    if canny:
        with log_time("canny"):
            in_image = np.array(input_pil)
            in_image = cv2.Canny(in_image, 100, 200)
            if not isinstance(in_image, np.ndarray):
                in_image = np.zeros((input_pil.height, input_pil.width), dtype=np.uint8)
            in_image = in_image[:, :, None]
            in_image = np.concatenate([in_image, in_image, in_image], axis=2)
            canny_image = Image.fromarray(in_image)
            set_seed(42)

    generator = torch.Generator("cpu").manual_seed(0)
    for attempt in range(retries + 1):
        try:
            if canny and backend == "sdxl":
                pipe_args = build_inference_kwargs("sdxl", "style", steps=n_steps)
                with inference_guard(), torch.inference_mode():
                    sdxl_canny_pipeline = get_sdxl_canny_pipe()
                    if sdxl_canny_pipeline is not None:
                        image = sdxl_canny_pipeline(
                            prompt=prompt,
                            image=input_pil,
                            control_image=canny_image,
                            strength=strength,
                            controlnet_conditioning_scale=env_float("SDXL_CONTROLNET_SCALE", 0.5),
                            generator=generator,
                            **pipe_args,
                        ).images[0]
                    else:
                        sdxl_img2img_pipeline = get_sdxl_img2img_pipe()
                        image = sdxl_img2img_pipeline(
                            prompt=prompt,
                            image=input_pil,
                            strength=strength,
                            generator=generator,
                            **pipe_args,
                        ).images[0]
            elif canny:
                # Use Flux ControlNet for Canny edge guidance
                pipe_args = build_inference_kwargs("flux", "control", steps=n_steps)
                with inference_guard(), torch.inference_mode():
                    canny_pipeline = get_canny_pipe()
                    if canny_pipeline:
                        image = canny_pipeline(
                            prompt=prompt,
                            control_image=canny_image,  # Use control_image for Flux ControlNet
                            controlnet_conditioning_scale=float(os.getenv("FLUX_CONTROLNET_SCALE", "0.7")),
                            generator=generator,
                            height=input_pil.height,
                            width=input_pil.width,
                            **pipe_args,
                        ).images[0]
                    else:
                        # Fallback to regular image generation
                        flux_pipeline = get_flux_pipe()
                        image = flux_pipeline(
                            prompt=prompt,
                            width=input_pil.width,
                            height=input_pil.height,
                            generator=generator,
                            **pipe_args,
                        ).images[0]
            elif backend == "sdxl":
                pipe_args = build_inference_kwargs("sdxl", "style", steps=n_steps)
                with inference_guard(), torch.inference_mode():
                    sdxl_img2img_pipeline = get_sdxl_img2img_pipe()
                    image = sdxl_img2img_pipeline(
                        prompt=prompt,
                        image=input_pil,
                        strength=strength,
                        generator=generator,
                        **pipe_args,
                    ).images[0]
            else:
                # Use Flux img2img for style transfer
                pipe_args = build_inference_kwargs("flux", "style", steps=n_steps)
                with inference_guard(), torch.inference_mode():
                    img2img_pipeline = get_img2img_pipe()
                    image = img2img_pipeline(
                        prompt=prompt,
                        image=input_pil,
                        strength=strength,
                        generator=generator,
                        **pipe_args,
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
    """Generate an image using the configured text-to-image backend with retries."""
    if extra_args is None:
        extra_args = {}
    extra_args = dict(extra_args)

    # For testing, use fewer steps to speed up inference
    if os.getenv("TESTING", "false").lower() == "true":
        n_steps = min(n_steps, 4)  # Limit to 4 steps during testing
        logger.info(f"Testing mode: reducing steps to {n_steps}")

    block_width = width - (width % 64)
    block_height = height - (height % 64)
    prompt = shorten_too_long_text(prompt)
    generator = torch.Generator("cpu").manual_seed(extra_args.pop("seed", 0))
    backend = extra_args.pop("backend", os.getenv("TEXT_TO_IMAGE_BACKEND", "flux")).strip().lower()
    if backend == "proteus":
        backend = "sdxl"

    guidance_scale = extra_args.pop("guidance_scale", None)
    call_steps = extra_args.pop("num_inference_steps", n_steps)
    pipe_args = build_inference_kwargs(
        backend,
        "text",
        steps=call_steps,
        guidance_scale=guidance_scale,
        extra_args=extra_args,
    )

    if env_bool("SDIF_CLEAR_MEMORY_EACH_CALL", False):
        clear_gpu_memory()

    for attempt in range(retries + 1):
        try:
            with inference_guard(), torch.inference_mode():
                text_pipeline = get_sdxl_pipe() if backend == "sdxl" else get_flux_pipe()
                # Update progress for long-running tasks
                if os.path.exists("progress.txt"):
                    with open("progress.txt", "w") as f:
                        f.write(datetime.now().strftime("%H:%M:%S"))
                
                image = text_pipeline(
                    prompt=prompt,
                    width=block_width,
                    height=block_height,
                    generator=generator,
                    **pipe_args,
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
            if retries > 0:
                logger.info("image too bumpy, retrying once w different prompt detailed")
                return create_image_from_prompt(
                    prompt + " detail", width, height, n_steps + 1, extra_args, retries - 1
                )
            else:
                logger.warning("image too bumpy after retries, returning anyway")

    return image_to_bytes(image)


# multiprocessing.set_start_method('spawn', True)
# processes_pool = Pool(1) # cant do too much at once or OOM errors happen
# def create_image_from_prompt_sync(prompt):
#     """have to call this sync to avoid OOM errors"""
#     return processes_pool.apply_async(create_image_from_prompt, args=(prompt,), ).wait()


def image_to_bytes(image):
    bs = BytesIO()

    image_array = np.array(image)
    try:
        bright_count = np.sum(image_array > 0)
    except TypeError:
        bright_count = np.sum(image_array)
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
    high_noise_frac = 0.7

    for attempt in range(retries + 1):
        try:
            pipe_args = build_inference_kwargs("flux", "inpaint")
            with inference_guard(), torch.inference_mode():
                inpaint_pipeline = get_inpaint_pipe()
                image = inpaint_pipeline(
                    prompt=prompt,
                    image=init_image,
                    mask_image=mask_image,
                    strength=1.0 - high_noise_frac,  # Convert denoising_start to strength
                    **pipe_args,
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
    if image is not None and env_bool("FLUX_INPAINT_REFINER", False):
        pipe_args = build_inference_kwargs("flux", "inpaint")
        with inference_guard(), torch.inference_mode():
            refiner_pipe = get_inpaint_refiner()
            image = refiner_pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                strength=1.0 - high_noise_frac,  # Convert denoising_start to strength
                **pipe_args,
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
