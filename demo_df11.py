import torch
from diffusers import FluxPipeline
from dfloat11 import DFloat11Model
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir="models")
pipe.enable_model_cpu_offload()

DFloat11Model.from_pretrained('DFloat11/FLUX.1-dev-DF11', device='cpu', bfloat16_model=pipe.transformer, cache_dir="models")

prompt = "A futuristic cityscape at sunset, with flying cars, neon lights, and reflective water canals"
image = pipe(
    prompt,
    width=1920,
    height=1440,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator(device="cuda").manual_seed(0)
).images[0]

image.save("image.png")
