from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.middleware.gzip import GZipMiddleware

from diffusers import DiffusionPipeline
import torch
from starlette.responses import FileResponse

pipe = DiffusionPipeline.from_pretrained("models/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", safety_checker=None) # todo try torch_dtype=bfloat16
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


app = FastAPI(
    openapi_url="/static/openapi.json",
    docs_url="/swagger-docs",
    redoc_url="/redoc",
    title="Generate Text or Speech Netwrck API",
    description="Character Chat API",
    # root_path="https://api.text-generator.io",
    version="1",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/make_image")
def make_image(prompt:str, save_path:str=""):
    if Path(save_path).exists():
        return FileResponse(save_path, media_type="image/png")
    image = pipe(prompt=prompt).images[0]
    if not save_path:
        save_path = f"images/{prompt}.png"
    image.save(save_path)
    return FileResponse(image, media_type="image/png")

# image = pipe(prompt=prompt).images[0]
#
# image.save("test.png")
# save all imaegs
# for i, image in enumerate(images):
#     image.save(f"{i}.png")
