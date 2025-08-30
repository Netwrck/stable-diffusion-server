#!/usr/bin/env python
"""Fast generation test with optimizations for RTX 3070."""

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import time
from PIL import Image
import os

def generate_fast_image():
    print("Loading optimized pipeline for speed...")
    
    # Use SDXL Turbo or Lightning for faster generation
    model_id = "stabilityai/sdxl-turbo"  # Much faster variant
    
    pipe = DiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Enable optimizations
    pipe.enable_xformers_memory_efficient_attention()  # Memory efficient attention
    pipe.enable_vae_slicing()  # VAE slicing for memory
    pipe.enable_vae_tiling()  # VAE tiling for large images
    
    # Compile with torch.compile for speed (PyTorch 2.0+)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Generate with turbo settings (1-4 steps only!)
    prompt = "a majestic dragon flying over a medieval castle, fantasy art, highly detailed"
    
    print(f"\nGenerating image with TURBO settings...")
    print(f"Prompt: {prompt}")
    
    # Warm up the model
    print("Warming up GPU...")
    with torch.no_grad():
        _ = pipe(prompt="test", num_inference_steps=1, guidance_scale=0.0, height=512, width=512).images[0]
    torch.cuda.synchronize()
    
    # Time the actual generation
    start_time = time.time()
    
    with torch.no_grad():
        # SDXL Turbo uses 1-4 steps with no CFG
        image = pipe(
            prompt=prompt,
            num_inference_steps=1,  # Turbo mode: 1-4 steps only
            guidance_scale=0.0,  # No CFG for turbo
            height=512,
            width=512
        ).images[0]
    
    torch.cuda.synchronize()
    generation_time = time.time() - start_time
    
    # Save the image
    output_path = "test_fast.png"
    image.save(output_path)
    print(f"\n✅ Image saved to: {output_path}")
    print(f"⚡ Generation time: {generation_time:.2f} seconds")
    
    # Also test with slightly more steps for quality
    print("\nGenerating higher quality version (4 steps)...")
    start_time = time.time()
    
    with torch.no_grad():
        image_hq = pipe(
            prompt=prompt,
            num_inference_steps=4,  # Still very fast
            guidance_scale=0.0,
            height=768,
            width=768
        ).images[0]
    
    torch.cuda.synchronize()
    generation_time_hq = time.time() - start_time
    
    output_hq = "test_fast_hq.png"
    image_hq.save(output_hq)
    print(f"✅ HQ Image saved to: {output_hq}")
    print(f"⚡ HQ Generation time: {generation_time_hq:.2f} seconds")
    
    return output_path

if __name__ == "__main__":
    try:
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        output_file = generate_fast_image()
        print(f"\n🎯 Success! Check the generated images")
        
        # Print memory usage
        print(f"\nGPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()