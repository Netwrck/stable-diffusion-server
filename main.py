import gc
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
from diffusers import DiffusionPipeline
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.responses import JSONResponse

from env import BUCKET_PATH, BUCKET_NAME
from stable_diffusion_server.bucket_api import check_if_blob_exists, upload_to_bucket

pipe = DiffusionPipeline.from_pretrained(
    "models/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="fp16",
    # safety_checker=None,
)  # todo try torch_dtype=bfloat16
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()


# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# this can cause errors on some inputs so consider disabling it
pipe.unet = torch.compile(pipe.unet)


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
    path_components = save_path.split("/")[0:-1]
    final_name = save_path.split("/")[-1]
    if not path_components:
        path_components = []
    save_path = '/'.join(path_components) + quote_plus(final_name)
    path = get_image_or_create_upload_to_cloud_storage(prompt, save_path)
    return JSONResponse({"path": path})


def get_image_or_create_upload_to_cloud_storage(prompt:str, save_path:str):
    # check exists - todo cache this
    if check_if_blob_exists(save_path):
        return f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"
    bio = create_image_from_prompt(prompt)
    if bio is None:
        return None # error thrown in pool
    link = upload_to_bucket(save_path, bio, is_bytesio=True)
    return link

# multiprocessing.set_start_method('spawn', True)
# processes_pool = Pool(1) # cant do too much at once or OOM errors happen
# def create_image_from_prompt_sync(prompt):
#     """have to call this sync to avoid OOM errors"""
#     return processes_pool.apply_async(create_image_from_prompt, args=(prompt,), ).wait()

def create_image_from_prompt(prompt):
    if len(prompt) > 200:
        # remove stopwords
        prompt = prompt.split()
        prompt = ' '.join((word for word in prompt if word not in stopwords))
        if len(prompt) > 200:
            prompt = prompt[:200]
    # image = pipe(prompt=prompt).images[0]
    try:
        image = pipe(prompt=prompt,
                     # height=512,
                     # width=512,
                     num_inference_steps=50).images[0]  # normally uses 50 steps
    except Exception as e:
        # try rm stopwords + half the prompt
        # todo try prompt permutations
        logger.info(f"trying to shorten prompt of length {len(prompt)}")

        prompt = ' '.join((word for word in prompt if word not in stopwords))
        prompts = prompt.split()

        prompt = ' '.join(prompts[:len(prompts) // 2])
        logger.info(f"shortened prompt to: {len(prompt)}")

        if prompt:
            try:
                image = pipe(prompt=prompt,
                             # height=512,
                             # width=512,
                             num_inference_steps=50).images[0]  # normally uses 50 steps
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
                                 # height=512,
                                 # width=512,
                                 num_inference_steps=50).images[0]  # normally uses 50 steps
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

# image = pipe(prompt=prompt).images[0]
#
# image.save("test.png")
# save all images
# for i, image in enumerate(images):
#     image.save(f"{i}.png")
