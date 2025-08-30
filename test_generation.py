#!/usr/bin/env python
"""Simple test script to generate an image using the HDM pipeline."""

import torch
from hdmx import HDMXUTPipeline
from PIL import Image
import os

def generate_test_image():
    print("Loading HDM pipeline...")
    
    # Load the pipeline
    pipe = HDMXUTPipeline.from_pretrained(
        "hdmx/hdmx_composite3",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Generate image
    prompt = "a beautiful fantasy landscape with mountains and a crystal clear lake at sunset, highly detailed, 4k"
    print(f"\nGenerating image with prompt: {prompt}")
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
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