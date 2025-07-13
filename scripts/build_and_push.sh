#!/bin/bash
set -euo pipefail

# Configuration
GITHUB_USERNAME="${GITHUB_USERNAME:-$(git remote get-url origin | sed 's/.*github\.com[:/]\([^/]*\)\/.*/\1/' | tr '[:upper:]' '[:lower:]')}"  # Auto-detect from git remote
IMAGE_NAME="stable-diffusion-server"
TAG="${1:-latest}"
REGISTRY="ghcr.io"
FULL_IMAGE_NAME="${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"
echo "GitHub username: ${GITHUB_USERNAME}"

# Validate GitHub username (no spaces, lowercase)
if [[ "$GITHUB_USERNAME" =~ [[:space:]] ]]; then
    echo "Warning: GitHub username contains spaces. Please set GITHUB_USERNAME environment variable."
    echo "Example: export GITHUB_USERNAME=your-github-username"
    exit 1
fi

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Verify no models directory exists in build context
if [ -d "models" ] && [ "$(ls -A models)" ]; then
    echo "Warning: models directory is not empty. This will increase image size significantly!"
    echo "Contents of models directory:"
    ls -la models/
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting build. Please remove or empty the models directory."
        exit 1
    fi
fi

# Build the Docker image
echo "Building Docker image..."

# Ensure DOCKER_HOST doesn't interfere and enable BuildKit
unset DOCKER_HOST
export DOCKER_BUILDKIT=1

DOCKER_BUILDKIT=1 docker build \
    --platform linux/amd64 \
    --tag "${FULL_IMAGE_NAME}" \
    --progress=plain \
    .

# Check image size
echo "Docker image built successfully!"
echo "Image size:"
docker images "${FULL_IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Login to GitHub Container Registry
echo "Logging in to GitHub Container Registry..."
echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GITHUB_USERNAME}" --password-stdin

# Push to GitHub Container Registry
echo "Pushing to GitHub Container Registry..."
docker push "${FULL_IMAGE_NAME}"

echo "Successfully pushed ${FULL_IMAGE_NAME}"
echo ""
echo "To use this image in RunPod:"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Environment Variables:"
echo "  MODE_TO_RUN=serverless"
echo "  MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0"
echo "  HF_TOKEN=your_huggingface_token"
