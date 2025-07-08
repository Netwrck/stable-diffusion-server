# GitHub Copilot Instructions for Stable Diffusion Server

## Project Context
This is a production-ready Stable Diffusion server supporting multiple AI models (SDXL, Flux Schnell) with FastAPI backend and Gradio UI. The server handles text-to-image generation, inpainting, style transfer, and cloud storage integration.

## Code Style & Patterns

### Python Standards
- Use type hints for all function parameters and return values
- Follow existing error handling patterns with try/except and retry logic
- Use `torch.inference_mode()` context for all model inference operations
- Implement proper memory management with CPU offloading and cache clearing

### Model Pipeline Patterns
```python
# Always use this pattern for inference
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
```

### Image Processing Standards
- Convert all input images to RGB: `image.convert("RGB")`
- Use 64-pixel alignment for dimensions: `width - (width % 64)`
- Save images as WebP with quality=85 and optimize=True
- Always process images through `process_image_for_stable_diffusion()` before inference

### Error Handling Requirements
- Implement retry logic with prompt modification on failures
- Use `shorten_prompt_for_retry()` and `remove_stopwords()` for retries
- Log warnings with attempt counts: `logger.warning(f"Failed on attempt {attempt + 1}/{retries}: {err}")`
- Detect and handle "too bumpy" images with regeneration

### API Endpoint Patterns
```python
@app.get("/endpoint_name")
async def endpoint_function(
    prompt: str, 
    save_path: str = "",
    # other params with defaults
):
    # URL encode save_path components
    path_components = save_path.split("/")[0:-1]
    final_name = save_path.split("/")[-1] 
    save_path = "/".join(path_components) + quote_plus(final_name)
    
    # Check cache first
    if check_if_blob_exists(save_path):
        return JSONResponse({"path": f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"})
```

## Model Architecture Guidelines

### Pipeline Initialization
- Enable CPU offloading: `pipe.enable_model_cpu_offload()`
- Enable memory optimizations: `pipe.enable_attention_slicing()`, `pipe.enable_vae_slicing()`
- Share components between pipelines to reduce memory usage
- Set `pipe.watermark = None` to disable watermarking

### Memory Management
- Use sequential CPU offloading for production: `pipe.enable_sequential_cpu_offload()`
- Implement component sharing: `img2img.unet = pipe.unet`
- Consider Optimum Quanto quantization for memory-constrained environments
- Use torch.Generator with fixed seeds for reproducible results

## Cloud Storage Integration

### Upload Pattern
```python
# Always check existence before generation
if check_if_blob_exists(save_path):
    return f"https://{BUCKET_NAME}/{BUCKET_PATH}/{save_path}"

# Generate and upload
bio = create_image_from_prompt(prompt, width, height)
link = upload_to_bucket(save_path, bio, is_bytesio=True)
return link
```

### Environment Variables
- Use `STORAGE_PROVIDER` to switch between 'r2' and 'gcs'
- Support both R2_ENDPOINT_URL and GOOGLE_APPLICATION_CREDENTIALS
- Respect BUCKET_NAME and BUCKET_PATH configuration

## Gradio UI Patterns

### Interface Structure
```python
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # Input controls
        with gr.Column():
            # Output displays
    
    # Event handlers
    button.click(function, inputs=[...], outputs=[...])
```

### Image Handling
- Use `gr.Image(tool="editor", type="pil")` for inpainting interfaces
- Save intermediate results to outputs/ directory with descriptive names
- Yield intermediate results for progressive display: `yield images`

## Security & Performance

### Input Validation
- Always call `shorten_too_long_text()` on prompts and save_paths
- Validate image dimensions and adjust to model requirements
- Use UUID prefixes for file naming to avoid conflicts

### Production Considerations
- Implement progress tracking with progress.txt updates
- Handle CUDA memory issues with automatic server restart logic
- Use proper timeout settings: `--timeout-keep-alive 600`
- Limit concurrency: `--limit-concurrency 4` for memory management

## Testing & Development

### Local Development
- Use `python gradio_ui.py` for quick UI testing
- Test individual models with `python flux_schnell.py`
- Run server locally: `uvicorn main:app --port 8000 --reload`

### Production Testing
- Test with limited concurrency settings
- Verify cloud storage uploads work correctly
- Monitor memory usage and restart behavior
- Test error handling with malformed inputs

## Common Pitfalls to Avoid
- Never use models without CPU offloading in production
- Don't forget to convert masks to RGB for inpainting
- Always check for None returns from image generation functions
- Don't skip the "too bumpy" detection - it prevents bad outputs
- Remember to update progress.txt to prevent supervisor timeouts