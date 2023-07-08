# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("models/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
images = pipe(prompt=prompt).images[0]

# save all imaegs
for i, image in enumerate(images):
    image.save(f"{i}.png")
