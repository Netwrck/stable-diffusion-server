# Deploying Stable Diffusion Server

This document describes how to build Docker images using GitHub Actions and deploy them to RunPod and Google Cloud Run with GPU support.

## GitHub Actions workflow

The repository includes a workflow located at `.github/workflows/docker-build.yml`. This workflow builds two Docker images whenever changes are pushed to `main` or a pull request is opened. It installs dependencies with [uv](https://github.com/astral-sh/uv) and caches the build layers for faster rebuilds:

- `runpod:latest` – image intended for [RunPod](https://www.runpod.io/)
- `cloudrun:latest` – image intended for Google Cloud Run

Both images are published to the GitHub Container Registry under `ghcr.io/<OWNER>/<REPO>`.

The provided `Dockerfile` installs packages via `uv pip` and uses BuildKit cache mounts to speed up dependency installation.

## Running on RunPod

1. Ensure you have a RunPod account and have created a container workspace with GPU access.
2. In RunPod, configure your deployment to pull the `runpod:latest` image from GHCR.
3. Expose port **8000**. The server will start automatically with the default command from the Dockerfile.
4. Optionally mount any model or data volumes required by the application.

## Running on Google Cloud Run with L4 GPUs

1. Enable the Cloud Run API and create a new service with GPU support. L4 GPUs are available in `us-central1` or other supported regions.
2. Grant Cloud Run permission to access Artifact Registry or GHCR where the image is stored.
3. Deploy using the `cloudrun:latest` image:
   ```bash
   gcloud run deploy sd-server \
       --image=ghcr.io/<OWNER>/<REPO>/cloudrun:latest \
       --region=us-central1 \
       --gpu=1 \
       --gpu-type=nvidia-l4 \
       --memory=8Gi \
       --min-instances=0 \
       --max-instances=1
   ```
4. Make sure to allocate sufficient memory and enable GPUs for the service.

## Local testing

To test the Docker image locally with GPU support, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and run:

```bash
docker run --gpus all -p 8000:8000 ghcr.io/<OWNER>/<REPO>/runpod:latest
```

The API will be available at `http://localhost:8000`.
