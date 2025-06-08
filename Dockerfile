# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    pip3 install uv && uv pip install -r requirements.txt

COPY . /app

# Expose port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600", "--workers", "1"]
