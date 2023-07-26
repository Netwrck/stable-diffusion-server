from io import BytesIO
from pathlib import Path

import nltk
from fastapi import FastAPI, Form
from fastapi.middleware.gzip import GZipMiddleware

from diffusers import DiffusionPipeline
import torch
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.responses import JSONResponse

from env import BUCKET_PATH, BUCKET_NAME
from stable_diffusion_server.bucket_api import check_if_blob_exists, upload_to_bucket

pipe = DiffusionPipeline.from_pretrained(
    "models/stable-diffusion-xl-base-0.9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    safety_checker=None,
)  # todo try torch_dtype=bfloat16
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()


pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


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
    image = pipe(prompt=prompt).images[0]
    if not save_path:
        save_path = f"images/{prompt}.png"
    image.save(save_path)
    return FileResponse(save_path, media_type="image/png")


@app.get("/create_and_upload_image")
def create_and_upload_image(prompt: str, save_path: str = ""):
    path = get_image_or_create_upload_to_cloud_storage(prompt, save_path)
    return JSONResponse({"path": path})


def get_image_or_create_upload_to_cloud_storage(prompt:str, save_path:str):
    # check exists - todo cache this
    if check_if_blob_exists(save_path):
        return f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"
    if len(prompt) > 200:
        # remove stopwords
        prompt = prompt.split()
        prompt = ' '.join((word for word in prompt if word not in stopwords))
        if len(prompt) > 200:
            prompt = prompt[:200]
    # image = pipe(prompt=prompt).images[0]
    image = pipe(prompt=prompt,
                 # height=512,
                 # width=512,
                 num_inference_steps=50).images[0] # normally uses 50 steps
    # save as bytesio
    bs = BytesIO()
    image.save(bs, format="webp")
    bio = bs.getvalue()
    link = upload_to_bucket(save_path, bio, is_bytesio=True)
    return link



# image = pipe(prompt=prompt).images[0]
#
# image.save("test.png")
# save all images
# for i, image in enumerate(images):
#     image.save(f"{i}.png")
