# Docker & RunPod Deployment Guide

This guide covers building and deploying the Stable Diffusion server using Docker for local development and RunPod serverless deployment.

## Quick Start

### 1. Test Docker Build
```bash
./scripts/test_build.sh
```

### 2. Build and Push to GHCR
```bash
# GitHub token should be set in environment
./scripts/build_and_push.sh [tag]
```

## Local Development with Docker

### Build Local Image
```bash
docker build -t sdif-local .
```

### Run with Volume Mount (for accessing local models)
```bash
docker run -it --rm \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -e MODE_TO_RUN=api \
  -e MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0 \
  sdif-local
```

### Run API Server
```bash
docker run -d --name sdif-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODE_TO_RUN=api \
  sdif-local
```

### Run with Docker Compose
```bash
docker-compose up -d
```

## RunPod Serverless Deployment

### 1. Build and Push Image
```bash
# Ensure GitHub token is set
export GITHUB_TOKEN="your_github_token"
./scripts/build_and_push.sh
```

### 2. RunPod Configuration

**Container Settings:**
- Image: `ghcr.io/your-github-username/stable-diffusion-server:latest`
- Container Disk: 20GB minimum
- GPU: A100 40GB or better recommended

**Environment Variables:**
```
MODE_TO_RUN=serverless
MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0
HF_TOKEN=your_huggingface_token
STORAGE_PROVIDER=r2
R2_ENDPOINT_URL=your_r2_endpoint
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
BUCKET_NAME=your_bucket_name
BUCKET_PATH=sdif-outputs
```

**Network Volumes (Optional):**
- Mount point: `/app/models`
- Size: 100GB+
- Use for persistent model storage across invocations

### 3. Test Deployment
Send a test request to your RunPod endpoint:
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a beautiful sunset over mountains",
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 20
    }
  }'
```

## Model Management

### Local Development
- Models are stored in `./models/` (symlinked to `/mnt/fast/models`)
- Models are downloaded automatically on first use
- Use volume mounts to share models between host and container

### Production/RunPod
- Models are downloaded fresh on each cold start
- Use RunPod Network Volumes for persistent model storage
- Models cache in `/tmp/huggingface` and `/tmp/transformers_cache`

## Environment Variables

### Required
- `MODE_TO_RUN`: `api` for local, `serverless` for RunPod
- `MODEL_NAME`: Model to load (e.g., `stabilityai/stable-diffusion-xl-base-1.0`)

### Optional
- `HF_TOKEN`: Hugging Face token for private models
- `STORAGE_PROVIDER`: `r2` or `gcs` for cloud storage
- `BUCKET_NAME`: Cloud storage bucket name
- `R2_ENDPOINT_URL`: R2 endpoint URL
- `R2_ACCESS_KEY_ID`: R2 access key
- `R2_SECRET_ACCESS_KEY`: R2 secret key

## Image Size Optimization

The Docker image excludes:
- Local model files (`models/` directory)
- Development scripts (`gradio_*.py`, `flux_schnell.py`)
- Test files and backdrops
- Git history and documentation
- Output directories and cached files

Final image size should be ~8-12GB with all dependencies.

## Troubleshooting

### Build Issues
```bash
# Clean Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -t sdif-local .
```

### Memory Issues
- Ensure GPU has at least 16GB VRAM
- Use `enable_sequential_cpu_offload()` for lower memory usage
- Reduce batch size or image dimensions

### Authentication Issues
```bash
# Configure GCP authentication
gcloud auth configure-docker
gcloud auth application-default login
```

### Model Download Issues
- Verify HF_TOKEN is valid
- Check internet connectivity in container
- Use RunPod Network Volumes for model persistence

## Performance Tips

1. **Use Network Volumes**: Store models on RunPod Network Volumes to avoid download times
2. **Pre-warm Models**: Include model loading in container startup
3. **Optimize Memory**: Use CPU offloading and attention slicing
4. **Cache Efficiently**: Use proper HuggingFace cache directories
