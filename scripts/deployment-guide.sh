#!/bin/bash
set -euo pipefail

echo "🐳 Quick Docker Deployment Guide"
echo "================================"
echo ""

echo "1️⃣  Test the build (no models included):"
echo "   ./scripts/test_build.sh"
echo ""

echo "2️⃣  Ensure GitHub token is set:"
echo "   export GITHUB_TOKEN=your_github_token (✅ Already set)"
echo ""

echo "3️⃣  Build and push to GitHub Container Registry:"
echo "   ./scripts/build_and_push.sh"
echo ""

echo "4️⃣  Deploy to RunPod:"
echo "   Image: ghcr.io/YOUR_GITHUB_USERNAME/stable-diffusion-server:latest"
echo "   Environment: MODE_TO_RUN=serverless"
echo "   GPU: A100 40GB+ recommended"
echo ""

echo "📚 For detailed instructions, see:"
echo "   - docs/docker-deployment.md"
echo "   - DOCKER_SETUP.md"
echo ""

echo "💡 Local development with models:"
echo "   docker run --gpus all -v \$(pwd)/models:/app/models -p 8000:8000 IMAGE_NAME"
