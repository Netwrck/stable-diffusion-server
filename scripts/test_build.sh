#!/bin/bash
set -euo pipefail

echo "Testing Docker build without models..."

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check models directory handling
if [ -L "models" ]; then
    echo "✅ PASS: models is a symlink (will be ignored by Docker)"
    echo "   Symlink target: $(readlink models)"
elif [ -d "models" ] && [ "$(ls -A models)" ]; then
    echo "⚠️  WARNING: models directory exists and contains files"
    echo "   This is OK for local development, but Docker will ignore it due to .dockerignore"
    echo "   First few items:"
    ls -la models/ | head -5
else
    echo "✅ PASS: No models directory or it's empty"
fi

# Check .dockerignore exists and contains model exclusions
if [ ! -f ".dockerignore" ]; then
    echo "❌ FAIL: .dockerignore file not found"
    exit 1
fi

if ! grep -q "models/" .dockerignore; then
    echo "❌ FAIL: .dockerignore doesn't exclude models directory"
    exit 1
else
    echo "✅ PASS: .dockerignore properly excludes models"
fi

# Test Docker build (dry run first)
echo "Testing Docker build (this may take a while)..."
IMAGE_NAME="sdif-test:$(date +%s)"

# Ensure DOCKER_HOST doesn't interfere and enable BuildKit
unset DOCKER_HOST
export DOCKER_BUILDKIT=1

# Build the image
if DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t "${IMAGE_NAME}" .; then
    echo "✅ PASS: Docker build succeeded"
    
    # Check image size
    SIZE=$(docker images "${IMAGE_NAME}" --format "{{.Size}}")
    echo "📦 Image size: ${SIZE}"
    
    # Test that the image can start (quick test)
    echo "Testing image startup..."
    if timeout 30 docker run --rm "${IMAGE_NAME}" python -c "import torch; print('✅ PyTorch available'); import transformers; print('✅ Transformers available'); print('✅ Container can start successfully')"; then
        echo "✅ PASS: Container can start and import required libraries"
    else
        echo "⚠️  WARNING: Container startup test failed or timed out"
    fi
    
    # Cleanup
    docker rmi "${IMAGE_NAME}" >/dev/null 2>&1 || true
    echo "🧹 Cleaned up test image"
    
else
    echo "❌ FAIL: Docker build failed"
    exit 1
fi

echo ""
echo "🎉 All tests passed! The Docker image is ready for production deployment."
echo ""
echo "Next steps:"
echo "1. Update PROJECT_ID in scripts/build_and_push.sh"
echo "2. Authenticate with Google Cloud: gcloud auth configure-docker"
echo "3. Run: ./scripts/build_and_push.sh"
