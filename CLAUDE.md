## Overview
This is a Simple Stable Diffusion Server for AI-powered image generation and manipulation. It supports multiple diffusion models including SDXL, Flux Schnell, and includes specialized pipelines for text-to-image, image-to-image, inpainting, and style transfer operations. The server can run locally with Gradio UI or as a production FastAPI service with cloud storage integration.

## Key Commands

### Setup and Installation
```bash
# Install dependencies using uv (recommended)
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt -r dev-requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Set up Google Cloud credentials for production
export GOOGLE_APPLICATION_CREDENTIALS="secrets/google-credentials.json"
```

### Development Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Run Gradio UI for local development
python gradio_ui.py

# Run inpainting UI
python gradio_inpaint_ui.py

# Test Flux Schnell model
python flux_schnell.py

# Run the FastAPI server locally
uvicorn main:app --port 8000

# Run tests
pytest

# Lint code
flake8
```

### Production Server Commands
```bash
# Production server with gunicorn
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json \
    gunicorn -k uvicorn.workers.UvicornWorker -b :8000 main:app --timeout 600 -w 1

# Production server with uvicorn (rate limited)
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json \
    PYTHONPATH=. uvicorn --port 8000 --timeout-keep-alive 600 --workers 1 --backlog 1 --limit-concurrency 4 main:app
```

### Docker Commands
```bash
# Build RunPod image
make docker-runpod

# Build Cloud Run image
make docker-cloudrun
```

## Architecture

### Core Components

**Main Pipeline System (main.py:74-444)**
- Primary SDXL pipeline (`pipe`) using ProteusV0.2 model with LCM scheduler
- Flux Schnell pipeline (`flux_pipe`) for efficient text-to-image generation
- Image-to-image pipeline (`img2img`) for style transfer
- Inpainting pipeline (`inpaintpipe`) and refiner for image editing
- ControlNet pipeline (`controlnetpipe`) for canny edge-guided generation
- Flux ControlNet pipeline (`flux_controlnetpipe`) for line-based style transfer

**Memory Optimization**
- CPU offloading enabled for all pipelines to manage GPU memory
- Attention slicing and VAE slicing for memory efficiency
- Optimum Quanto quantization support (commented out but available)
- Component sharing between pipelines to reduce memory usage

**FastAPI Endpoints (main.py:488-599)**
- `/create_and_upload_image` - Text-to-image with cloud upload
- `/inpaint_and_upload_image` - Inpainting with cloud upload  
- `/style_transfer_and_upload_image` - Style transfer with cloud upload
- `/style_transfer_bytes_and_upload_image` - Style transfer with file upload

### Image Generation Functions

**Text-to-Image (main.py:777-829)**
- Uses Flux Schnell pipeline by default
- Automatic prompt shortening and retry logic
- "Bumpy" image detection and regeneration
- Supports custom dimensions with 64-pixel alignment

**Style Transfer (main.py:670-774)**
- Canny edge detection for ControlNet guidance
- Fallback to standard Flux generation
- Retry mechanism with prompt modification
- Refiner pass option for quality improvement

**Inpainting (main.py:862-918)**
- Two-stage process with base and refiner pipelines
- Automatic mask and image processing
- Progress tracking for monitoring

### Cloud Storage Integration
- R2 (Cloudflare) and Google Cloud Storage support
- Automatic duplicate detection to avoid regeneration
- Environment-based configuration for storage backends

### UI Components
- **gradio_ui.py** - Basic text-to-image and style transfer interface
- **gradio_inpaint_ui.py** - Unified generation and inpainting interface with image editor
- **flux_schnell.py** - Standalone Flux pipeline example

## Environment Variables

### Model Configuration
```bash
LOAD_LCM_LORA=1  # Enable LCM LoRA for SDXL
DF11_MODEL_PATH=DFloat11/FLUX.1-schnell-DF11  # DFloat11 quantized model path
CONTROLNET_LORA=black-forest-labs/flux-controlnet-line-lora  # ControlNet LoRA path
```

## Model Structure
The server expects models in the `models/` directory:
- `models/ProteusV0.2/` - Primary SDXL model
- `models/stable-diffusion-xl-base-1.0/` - Base SDXL model  
- `models/lcm-lora-sdxl/` - LCM LoRA weights
- `models/diffusers/controlnet-canny-sdxl-1.0/` - ControlNet model

## Common Issues
- Black image generation triggers automatic server restart (main.py:844-854)
- CUDA memory issues handled with CPU offloading
- "Too bumpy" detection causes automatic regeneration with modified prompts
- Progress tracking prevents supervisor timeouts during long operations

## API Usage Examples
```bash
# Generate image
curl "http://localhost:8000/create_and_upload_image?prompt=good%20looking%20elf%20fantasy%20character&save_path=created/elf.webp"

# Style transfer
curl -X POST "http://localhost:8000/style_transfer_bytes_and_upload_image" \
  -F "prompt=anime style" \
  -F "image_file=@input.jpg" \
  -F "save_path=outputs/styled.webp"
```