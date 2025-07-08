# AGENTS.md - AI Assistant Guidelines

This file provides comprehensive guidance for AI assistants (Claude, ChatGPT, etc.) working with the Stable Diffusion Server codebase.

## Project Overview
A production-ready AI image generation server supporting multiple diffusion models with cloud storage integration, Gradio UI, and FastAPI backend.

## Core Capabilities
- **Text-to-Image**: Flux Schnell and SDXL model support
- **Style Transfer**: ControlNet-guided image transformation  
- **Inpainting**: Mask-based image editing with refinement
- **Cloud Storage**: R2/GCS integration with automatic caching
- **UI Components**: Gradio interfaces for local development

## Quick Start Commands

### Development Setup
```bash
# Environment setup
pip install uv && uv venv && source .venv/bin/activate
uv pip install -r requirements.txt -r dev-requirements.txt
python -c "import nltk; nltk.download('stopwords')"

# Local testing
python flux_schnell.py                    # Test Flux model
python gradio_ui.py                       # Launch UI
uvicorn main:app --port 8000             # Run API server
```

### Production Deployment
```bash
# With environment variables
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json \
PYTHONPATH=. uvicorn --port 8000 --timeout-keep-alive 600 --workers 1 --limit-concurrency 4 main:app
```

## Key Architecture Components

### Model Pipelines (main.py:74-444)
1. **Primary SDXL Pipeline** (`pipe`) - ProteusV0.2 with LCM scheduler
2. **Flux Schnell Pipeline** (`flux_pipe`) - Fast text-to-image generation  
3. **Image2Image Pipeline** (`img2img`) - Style transfer operations
4. **Inpainting Pipelines** (`inpaintpipe`, `inpaint_refiner`) - Mask-based editing
5. **ControlNet Pipelines** - Canny edge and line-guided generation

### Memory Management Strategy
- CPU offloading for all pipelines to manage GPU memory
- Component sharing between pipelines (shared UNet, VAE, encoders)
- Attention slicing and VAE slicing for efficiency
- Optional Optimum Quanto quantization support

### API Endpoints
- `/create_and_upload_image` - Text-to-image with cloud upload
- `/inpaint_and_upload_image` - Inpainting with cloud upload
- `/style_transfer_and_upload_image` - Style transfer with cloud upload  
- `/style_transfer_bytes_and_upload_image` - File upload support

## Development Guidelines

### When Adding New Features
1. **Follow existing patterns**: Use the same error handling, retry logic, and memory management
2. **Maintain compatibility**: Ensure new features work with existing pipeline architecture
3. **Test thoroughly**: Use both Gradio UI and API endpoints for validation
4. **Document changes**: Update relevant sections in CLAUDE.md and this file

### Code Quality Standards
```python
# Always use type hints
def generate_image(prompt: str, width: int = 1024) -> Image.Image:

# Use inference mode for all model operations  
with torch.inference_mode():
    image = pipe(prompt=prompt).images[0]

# Implement proper error handling with retries
for attempt in range(retries + 1):
    try:
        # Generation logic
        break
    except Exception as err:
        if attempt >= retries:
            raise
        logger.warning(f"Failed attempt {attempt + 1}/{retries}: {err}")
```

### Common Tasks and Solutions

#### Adding a New Model
1. Load model in main.py initialization section
2. Enable CPU offloading and memory optimizations
3. Share components with existing pipelines where possible
4. Add corresponding API endpoint following existing patterns
5. Test with Gradio UI integration

#### Modifying Image Processing
1. Update `stable_diffusion_server/image_processing.py`
2. Ensure compatibility with existing dimension requirements (64-pixel alignment)
3. Test with various input formats and sizes
4. Update error handling for edge cases

#### Cloud Storage Integration
1. Check existing `stable_diffusion_server/bucket_api.py` implementation
2. Follow the check-exists-before-generate pattern
3. Handle both R2 and GCS storage backends
4. Test upload/download functionality thoroughly

## Troubleshooting Common Issues

### Memory Problems
- **Black images**: Usually indicates CUDA memory issues, server auto-restarts
- **OOM errors**: Reduce concurrency, enable more aggressive CPU offloading
- **Slow inference**: Check if models are properly using CPU offloading

### Image Quality Issues  
- **"Too bumpy" images**: Automatic detection triggers regeneration with modified prompts
- **Poor style transfer**: Ensure canny edge detection is working correctly
- **Blurry outputs**: Check if proper refinement passes are enabled

### API/Server Issues
- **Timeouts**: Update progress.txt file during long operations
- **Upload failures**: Verify cloud storage credentials and bucket permissions
- **Rate limiting**: Adjust `--limit-concurrency` and `--backlog` settings

## Environment Configuration

### Required Environment Variables
```bash
# Storage (choose one)
STORAGE_PROVIDER=r2|gcs
BUCKET_NAME=your-bucket-name
BUCKET_PATH=static/uploads
R2_ENDPOINT_URL=https://account.r2.cloudflarestorage.com
PUBLIC_BASE_URL=your-domain.com
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Model paths (optional)
DF11_MODEL_PATH=DFloat11/FLUX.1-schnell-DF11
CONTROLNET_LORA=black-forest-labs/flux-controlnet-line-lora
LOAD_LCM_LORA=1
```

### Model Directory Structure
```
models/
├── ProteusV0.2/           # Primary SDXL model
├── stable-diffusion-xl-base-1.0/  # Base SDXL model
├── lcm-lora-sdxl/         # LCM LoRA weights
└── diffusers/
    └── controlnet-canny-sdxl-1.0/  # ControlNet model
```

## Testing and Validation

### Before Submitting Changes
1. **Run existing tests**: `pytest -q`
2. **Check code style**: `flake8`  
3. **Test UI functionality**: Launch `python gradio_ui.py` and verify all features
4. **Test API endpoints**: Send requests to key endpoints and verify responses
5. **Memory usage**: Monitor GPU/CPU usage during generation

### Integration Testing
```bash
# Test image generation
curl "http://localhost:8000/create_and_upload_image?prompt=test&save_path=test.webp"

# Test style transfer  
curl -X POST "http://localhost:8000/style_transfer_bytes_and_upload_image" \
  -F "prompt=anime style" -F "image_file=@test.jpg" -F "save_path=output.webp"
```

## Performance Optimization

### Memory Optimization
- Use `enable_sequential_cpu_offload()` for lowest memory usage
- Share model components between pipelines
- Consider quantization for memory-constrained environments
- Monitor and tune batch sizes for optimal throughput

### Speed Optimization  
- Use Flux Schnell for fastest generation (4-8 steps)
- Enable LCM LoRA for SDXL speed improvements
- Implement proper caching with `check_if_blob_exists()`
- Use appropriate guidance scales (0.0 for Flux, 7+ for SDXL)

## Security Considerations

### Input Validation
- Always validate and sanitize prompts using `shorten_too_long_text()`
- Validate image dimensions and file formats
- Use UUID prefixes for generated filenames to prevent conflicts

### Production Security
- Never expose cloud storage credentials in code
- Use proper environment variable management
- Implement rate limiting and request validation
- Monitor for suspicious usage patterns

## Contributing Guidelines

### Pull Request Checklist
- [ ] Code follows existing patterns and style
- [ ] New features include appropriate error handling
- [ ] Memory management is properly implemented
- [ ] Tests pass and new functionality is tested
- [ ] Documentation is updated (CLAUDE.md, this file, docstrings)
- [ ] No sensitive information is committed

### Code Review Focus Areas
1. **Memory safety**: Proper pipeline management and GPU memory usage
2. **Error handling**: Robust retry logic and graceful degradation
3. **API consistency**: Following established endpoint patterns
4. **Performance impact**: Changes don't negatively affect generation speed
5. **Security**: Input validation and credential management