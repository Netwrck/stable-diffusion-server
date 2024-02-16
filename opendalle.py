# from segmoe import SegMoEPipeline
# pip install -U accelerate segmoe torch transformers diffusers
# pipeline = SegMoEPipeline("models/SegMoE-2x1-v0", device="cuda")
from diffusers import AutoPipelineForText2Image
import torch
from diffusers import LCMScheduler

pipeline = AutoPipelineForText2Image.from_pretrained('dataautogpt3/OpenDalle', torch_dtype=torch.float16).to('cuda')        


prompt = "cosmic canvas, orange city background, painting of a chubby cat"
prompt = "a male character from a fantasy book, seductive smile, abs, a young wizard. A hand-drawn mysterious, dark, handsome young adult Italian rebel, Seducive smile, Black eyes, tanned, black medium shoulder length hair, tattooed, Strong sexy jaw . High quality and fidelity vector. 2D white background, Shadows., Watercolor, trending on artstation, sharp focus, studio photo, intricate details, highly detailed"
# image = pipeline(prompt).images[0]
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

negative_prompt = "bad quality, worse quality"
img = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=7.5,
).images[0]
img.save("testautogpt3.png")