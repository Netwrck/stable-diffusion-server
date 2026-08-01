"""Low-system-RAM Proteus image service for the Windows RTX 3090 host."""

import asyncio
import base64
import os
import uuid
from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    LCMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.utils import load_image
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.responses import FileResponse, JSONResponse

try:
    from stable_diffusion_server.bucket_api import upload_to_bucket
except ModuleNotFoundError:
    upload_to_bucket = None


MODEL_FILE = os.getenv(
    "PROTEUS_MODEL",
    "models/ProteusV0.2/ProteusV0.2.safetensors",
)
CONFIG_DIR = os.getenv("PROTEUS_CONFIG", "models/stable-diffusion-xl-base-1.0")
OUTPUT_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", "images"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("Proteus low-RAM mode requires CUDA")

# Accelerate initializes every weight directly on GPU 0, avoiding a second
# full fp32/fp16 model copy in the 8 GiB system RAM.  "balanced" is not used:
# on a low-RAM Windows host it may silently place the whole pipeline on CPU.
pipe = StableDiffusionXLPipeline.from_single_file(
    MODEL_FILE,
    config=CONFIG_DIR,
    torch_dtype=torch.float16,
    device_map={"": 0},
    local_files_only=True,
    use_safetensors=True,
)
# from_single_file currently ignores a one-device map in Diffusers 0.30 on
# Windows.  Its mmap loader still keeps host working-set low; explicitly move
# the completed fp16 pipeline so inference never spills through system RAM.
pipe.to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
if Path("models/lcm-lora-sdxl").is_dir():
    pipe.load_lora_weights("models/lcm-lora-sdxl", adapter_name="lcm")
    pipe.set_adapters(["lcm"], adapter_weights=[1.0])
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.watermark = None

app = FastAPI(title="Proteus low-RAM image server")
gpu_lock = asyncio.Lock()
_img2img = None
_inpaint = None


class ImageRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"


def _img2img_pipe():
    global _img2img
    if _img2img is None:
        _img2img = AutoPipelineForImage2Image.from_pipe(pipe)
        _img2img.watermark = None
    return _img2img


def _inpaint_pipe():
    global _inpaint
    if _inpaint is None:
        _inpaint = AutoPipelineForInpainting.from_pipe(pipe)
        _inpaint.watermark = None
    return _inpaint


def _dimensions(size: str) -> tuple[int, int]:
    try:
        width, height = (int(value) for value in size.lower().split("x", 1))
    except (TypeError, ValueError):
        raise HTTPException(400, "size must look like 1024x1024")
    if width * height > 1024 * 1024 or min(width, height) < 256:
        raise HTTPException(400, "size must be between 256px and one megapixel")
    return width, height


def _generate(prompt: str, width: int, height: int):
    with torch.inference_mode():
        return pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=5,
            guidance_scale=1.0,
        ).images[0]


def _style(prompt: str, image_url, strength: float):
    source = load_image(image_url).convert("RGB")
    source.thumbnail((1024, 1024))
    with torch.inference_mode():
        return _img2img_pipe()(
            prompt=prompt,
            image=source,
            strength=max(0.05, min(strength, 1.0)),
            num_inference_steps=5,
            guidance_scale=1.0,
        ).images[0]


def _inpaint_image(prompt: str, image_url, mask_url):
    source = load_image(image_url).convert("RGB")
    mask = load_image(mask_url).convert("L").resize(source.size)
    with torch.inference_mode():
        return _inpaint_pipe()(
            prompt=prompt,
            image=source,
            mask_image=mask,
            num_inference_steps=5,
            guidance_scale=1.0,
        ).images[0]


async def _upload(image, save_path: str = "") -> str:
    if upload_to_bucket is None:
        raise HTTPException(503, "R2 upload support requires boto3")
    name = quote_plus(Path(save_path).name) if save_path else f"{uuid.uuid4()}.webp"
    payload = BytesIO()
    image.save(payload, format="WEBP", quality=92)
    return await asyncio.to_thread(upload_to_bucket, name, payload.getvalue(), True)


@app.get("/health")
def health():
    free, total = torch.cuda.mem_get_info()
    return {
        "status": "ok",
        "model": "ProteusV0.2",
        "device": torch.cuda.get_device_name(0),
        "unet_device": str(next(pipe.unet.parameters()).device),
        "vram_free_mib": round(free / 1024**2),
        "vram_total_mib": round(total / 1024**2),
        "system_ram_mode": "mmap-load-gpu-resident",
    }


@app.get("/make_image")
async def make_image(
    prompt: str, save_path: str = "", width: int = 1024, height: int = 1024
):
    if width * height > 1024 * 1024 or min(width, height) < 256:
        raise HTTPException(400, "size must be between 256px and one megapixel")
    target = Path(save_path) if save_path else OUTPUT_DIR / f"{uuid.uuid4()}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    async with gpu_lock:
        image = await asyncio.to_thread(_generate, prompt, width, height)
        image.save(target)
    return FileResponse(target, media_type="image/png")


@app.get("/create_and_upload_image")
async def create_and_upload_image(
    prompt: str, width: int = 1024, height: int = 1024, save_path: str = ""
):
    if width * height > 1024 * 1024:
        raise HTTPException(400, "maximum output is one megapixel")
    name = quote_plus(Path(save_path).name) if save_path else f"{uuid.uuid4()}.webp"
    async with gpu_lock:
        image = await asyncio.to_thread(_generate, prompt, width, height)
        path = await _upload(image, name)
    return JSONResponse({"path": path})


@app.get("/style_transfer_and_upload_image")
async def style_transfer_and_upload_image(
    prompt: str, image_url: str, save_path: str = "", strength: float = 0.6,
    canny: bool = False,
):
    if canny:
        raise HTTPException(501, "ControlNet canny is disabled in low-RAM mode")
    async with gpu_lock:
        image = await asyncio.to_thread(_style, prompt, image_url, strength)
        path = await _upload(image, save_path)
    return {"path": path}


@app.post("/style_transfer_bytes_and_upload_image")
async def style_transfer_bytes_and_upload_image(
    prompt: str, file: UploadFile = File(...), save_path: str = "",
    strength: float = 0.6,
):
    raw = await file.read()
    async with gpu_lock:
        image = await asyncio.to_thread(_style, prompt, BytesIO(raw), strength)
        path = await _upload(image, save_path)
    return {"path": path}


@app.get("/inpaint_and_upload_image")
async def inpaint_and_upload_image(
    prompt: str, image_url: str, mask_url: str, save_path: str = "",
):
    async with gpu_lock:
        image = await asyncio.to_thread(_inpaint_image, prompt, image_url, mask_url)
        path = await _upload(image, save_path)
    return {"path": path}


@app.post("/v1/images/generations")
async def openai_images(request: ImageRequest):
    if request.n != 1:
        raise HTTPException(400, "low-RAM mode supports n=1")
    width, height = _dimensions(request.size)
    name = f"{uuid.uuid4()}.webp"
    async with gpu_lock:
        image = await asyncio.to_thread(_generate, request.prompt, width, height)
        if request.response_format == "b64_json":
            payload = BytesIO()
            image.save(payload, format="PNG")
            item = {"b64_json": base64.b64encode(payload.getvalue()).decode("ascii")}
        else:
            item = {"url": await _upload(image, name)}
    return {"created": int(__import__("time").time()), "data": [item]}
