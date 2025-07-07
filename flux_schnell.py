import torch
from diffusers import FluxPipeline, FluxControlNetPipeline, ControlNetModel


def load_pipeline(model="black-forest-labs/FLUX.1-schnell"):
    """Load the Flux pipeline with optional controlnet."""
    pipe = FluxPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def load_controlnet(pipe, model="black-forest-labs/flux-controlnet-canny"):
    controlnet = ControlNetModel.from_pretrained(model, torch_dtype=torch.bfloat16)
    cpipe = FluxControlNetPipeline(controlnet=controlnet, **pipe.components)
    cpipe.enable_model_cpu_offload()
    return cpipe


def generate_image(pipe, prompt, seed=0, steps=4):
    generator = torch.Generator("cpu").manual_seed(seed)
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=steps,
        max_sequence_length=256,
        generator=generator,
    ).images[0]
    return image


if __name__ == "__main__":
    prompt = "A cat holding a sign that says hello world"
    pipe = load_pipeline()
    image = generate_image(pipe, prompt)
    image.save("flux-schnell.png")
