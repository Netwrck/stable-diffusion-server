#!/usr/bin/env python
"""Test script using Stable Diffusion XL model."""

import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os

def generate_test_image():
    print("Loading Stable Diffusion XL pipeline...")
    
    # Load SDXL base model (public, no auth needed)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Move to GPU
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        pipe.enable_model_cpu_offload()
    
    # Generate image
    prompt = "a beautiful fantasy landscape with mountains and a crystal clear lake at sunset, highly detailed, masterpiece, 4k"
    negative_prompt = "ugly, blurry, low quality, distorted"
    
    print(f"\nGenerating image with prompt: {prompt}")
    print("This may take a minute...")
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=768,
            width=768
        ).images[0]
    
    # Save the image
    output_path = "test_output.png"
    image.save(output_path)
    print(f"\nImage saved to: {output_path}")
    
    # Also save as webp for smaller size
    output_webp = "test_output.webp"
    image.save(output_webp, "WEBP", quality=90)
    print(f"WebP version saved to: {output_webp}")
    
    return output_path

if __name__ == "__main__":
    try:
        output_file = generate_test_image()
        print(f"\nSuccess! Generated image at: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()