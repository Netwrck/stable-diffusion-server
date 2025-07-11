import io
import base64
import os
import traceback
import torch
from PIL import Image
import runpod
from loguru import logger

# Import the main pipeline components - lazy loading for RunPod
flux_pipe = None
try:
    from main import (
        flux_pipe,
        create_image_and_upload,
        inpaint_image_and_upload,
        style_transfer_image_and_upload,
    )
except ImportError as e:
    logger.warning(f"Failed to import from main: {e}")
    # We'll handle model loading on-demand below

def handler(event):
    """
    RunPod serverless handler for stable diffusion operations.
    
    Expected JSON input formats:
    
    Text-to-image:
    {
      "input": {
        "operation": "txt2img",
        "prompt": "astronaut riding a horse",
        "negative_prompt": "blurry, low quality",
        "steps": 4,
        "seed": 42,
        "width": 1024,
        "height": 1024
      }
    }
    
    Style transfer:
    {
      "input": {
        "operation": "style_transfer", 
        "prompt": "anime style",
        "image_base64": "base64_encoded_image_data",
        "steps": 8,
        "seed": 42
      }
    }
    
    Inpainting:
    {
      "input": {
        "operation": "inpaint",
        "prompt": "beautiful flowers",
        "image_base64": "base64_encoded_image_data",
        "mask_base64": "base64_encoded_mask_data",
        "steps": 8,
        "seed": 42
      }
    }
    """
    try:
        inp = event.get("input", {})
        operation = inp.get("operation", "txt2img")
        
        logger.info(f"Processing {operation} request")
        
        if operation == "txt2img":
            return handle_txt2img(inp)
        elif operation == "style_transfer":
            return handle_style_transfer(inp)
        elif operation == "inpaint":
            return handle_inpaint(inp)
        else:
            return {"error": f"Unknown operation: {operation}"}
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def ensure_flux_pipeline():
    """Ensure Flux pipeline is loaded, load on-demand if not available"""
    global flux_pipe
    if flux_pipe is None:
        logger.info("Loading Flux pipeline on-demand...")
        try:
            from diffusers import FluxPipeline
            # Try local first, fallback to HuggingFace
            try:
                flux_pipe = FluxPipeline.from_pretrained(
                    "models/FLUX.1-schnell", 
                    torch_dtype=torch.bfloat16
                )
                logger.info("Loaded Flux pipeline from local models")
            except OSError:
                # Fallback to downloading from hub
                flux_pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell", 
                    torch_dtype=torch.bfloat16,
                    cache_dir="./models"
                )
                logger.info("Downloaded and loaded Flux pipeline from HuggingFace")
            
            # Enable CPU offloading for memory efficiency
            flux_pipe.enable_model_cpu_offload()
            flux_pipe.enable_sequential_cpu_offload()
            
        except Exception as e:
            logger.error(f"Failed to load Flux pipeline: {e}")
            raise e
    
    return flux_pipe

def handle_txt2img(inp):
    """Handle text-to-image generation"""
    prompt = inp.get("prompt", "")
    negative_prompt = inp.get("negative_prompt", "")
    steps = inp.get("steps", 4)
    seed = inp.get("seed")
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    
    if not prompt:
        return {"error": "Prompt is required"}
    
    # Ensure pipeline is loaded
    pipe = ensure_flux_pipeline()
    
    # Generate image using Flux pipeline
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
        width=width,
        height=height
    ).images[0]
    
    # Convert to base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_base64 = base64.b64encode(buf.getvalue()).decode()
    
    return {
        "image_base64": image_base64,
        "prompt": prompt,
        "steps": steps,
        "seed": seed,
        "width": width,
        "height": height
    }

def handle_style_transfer(inp):
    """Handle style transfer operation"""
    prompt = inp.get("prompt", "")
    image_base64 = inp.get("image_base64")
    steps = inp.get("steps", 8)
    seed = inp.get("seed")
    
    if not prompt or not image_base64:
        return {"error": "Prompt and image_base64 are required"}
    
    # Decode input image
    try:
        image_data = base64.b64decode(image_base64)
        input_image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        return {"error": f"Invalid image data: {e}"}
    
    # Use the existing style transfer function
    # We'll need to modify it to work without file uploads
    try:
        # For now, use a simplified approach with the flux pipeline
        # In production, you'd use the full style transfer pipeline
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
            
        # Ensure pipeline is loaded
        pipe = ensure_flux_pipeline()
        
        # This is a simplified implementation - you may want to use the full
        # style_transfer_image_and_upload function logic
        result_image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            generator=generator,
            width=input_image.width,
            height=input_image.height
        ).images[0]
        
        # Convert to base64
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        image_base64 = base64.b64encode(buf.getvalue()).decode()
        
        return {
            "image_base64": image_base64,
            "prompt": prompt,
            "steps": steps,
            "seed": seed
        }
        
    except Exception as e:
        return {"error": f"Style transfer failed: {e}"}

def handle_inpaint(inp):
    """Handle inpainting operation"""
    prompt = inp.get("prompt", "")
    image_base64 = inp.get("image_base64")
    mask_base64 = inp.get("mask_base64")
    steps = inp.get("steps", 8)
    seed = inp.get("seed")
    
    if not prompt or not image_base64 or not mask_base64:
        return {"error": "Prompt, image_base64, and mask_base64 are required"}
    
    try:
        # Decode input image and mask
        image_data = base64.b64decode(image_base64)
        mask_data = base64.b64decode(mask_base64)
        input_image = Image.open(io.BytesIO(image_data))
        mask_image = Image.open(io.BytesIO(mask_data))
        
        # Use the existing inpainting function
        # This would need to be adapted to work with PIL images instead of files
        # For now, return a placeholder
        return {"error": "Inpainting not yet implemented in RunPod handler"}
        
    except Exception as e:
        return {"error": f"Inpainting failed: {e}"}

# Start the RunPod serverless worker
if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})