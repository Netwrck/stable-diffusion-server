#!/usr/bin/env python
"""Simple test script to generate an image using available pipelines."""

import torch
from diffusers import FluxPipeline
from PIL import Image
import os

def generate_test_image():
    print("Loading Flux pipeline...")
    
    # Load Flux Schnell pipeline (fast variant)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", 
        torch_dtype=torch.bfloat16
    )
    
    # Enable CPU offloading to manage memory
    pipe.enable_model_cpu_offload()
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Generate image
    prompt = "a beautiful fantasy landscape with mountains and a crystal clear lake at sunset, highly detailed, 4k"
    print(f"\nGenerating image with prompt: {prompt}")
    print("This may take a few minutes on first run...")
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=4,  # Schnell is optimized for 4 steps
            guidance_scale=0.0,  # Schnell doesn't use guidance
            height=512,
            width=512
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
        print(f"\n✅ Success! Generated image at: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()