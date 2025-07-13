# syntax=docker/dockerfile:1.6
ARG CUDA_VER=12.0.1
FROM --platform=linux/amd64 nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=api \
    MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/transformers_cache \
    TORCH_HOME=/tmp/torch

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        curl \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /tmp/huggingface /tmp/transformers_cache /tmp/torch /app/models

# Copy requirements first for better caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "runpod>=1.8.0"

# Copy project code
COPY . /app
COPY rp_handler.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Expose FastAPI port for local mode
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
