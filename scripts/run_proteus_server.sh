#!/usr/bin/env bash
set -euo pipefail

cd /media/lee/pcd/code/sdif

set +u
if [ -f "$HOME/.secretbashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.secretbashrc" >/dev/null 2>&1 || true
fi
set -u

export STORAGE_PROVIDER="${STORAGE_PROVIDER:-r2}"
if [ -n "${CLOUDFLARE_R2_ACCESS_KEY_ID:-}" ]; then
  export AWS_ACCESS_KEY_ID="$CLOUDFLARE_R2_ACCESS_KEY_ID"
fi
if [ -n "${CLOUDFLARE_R2_SECRET_ACCESS_KEY:-}" ]; then
  export AWS_SECRET_ACCESS_KEY="$CLOUDFLARE_R2_SECRET_ACCESS_KEY"
fi
if [ -n "${CLOUDFLARE_R2_ACCESS_KEY_ID:-}" ] || [ -n "${CLOUDFLARE_R2_SECRET_ACCESS_KEY:-}" ]; then
  unset AWS_SESSION_TOKEN
fi

export BUCKET_NAME="${BUCKET_NAME:-${CLOUDFLARE_BUCKET:-netwrckstatic.netwrck.com}}"
if [ -z "${R2_ENDPOINT_URL:-}" ] && [ -n "${R2_ACCOUNT_ID:-}" ]; then
  export R2_ENDPOINT_URL="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
fi
export PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-netwrckstatic.netwrck.com}"

export PYTHONPATH="${PYTHONPATH:-.}"
export ENABLE_LEGACY_CUSTOM_PIPELINE="${ENABLE_LEGACY_CUSTOM_PIPELINE:-false}"
export HF_LOCAL_ONLY="${HF_LOCAL_ONLY:-true}"
export TEXT_TO_IMAGE_BACKEND="${TEXT_TO_IMAGE_BACKEND:-proteus}"
export STYLE_TRANSFER_BACKEND="${STYLE_TRANSFER_BACKEND:-proteus}"
export SDIF_OPTIMIZATION_PROFILE="${SDIF_OPTIMIZATION_PROFILE:-fast}"
export SDIF_CACHE_MODE="${SDIF_CACHE_MODE:-none}"
export SDIF_KEEP_MULTIPLE_PIPELINES="${SDIF_KEEP_MULTIPLE_PIPELINES:-false}"
export SDIF_SERIALIZE_INFERENCE="${SDIF_SERIALIZE_INFERENCE:-true}"
export PROTEUS_MODEL_REPO="${PROTEUS_MODEL_REPO:-models/ProteusV0.2}"

# DMD2 4-step distillation: near-40-step quality at ~1s/768px on the 3090
export SDXL_SPEED_LORA="${SDXL_SPEED_LORA:-models/dmd2/dmd2_sdxl_4step_lora_fp16.safetensors}"
export SDXL_SCHEDULER="${SDXL_SCHEDULER:-lcm}"
export SDXL_TIMESTEPS="${SDXL_TIMESTEPS:-999,749,499,249}"
export SDXL_GUIDANCE_SCALE="${SDXL_GUIDANCE_SCALE:-0}"
export SDXL_CONTROLNET_SCALE="${SDXL_CONTROLNET_SCALE:-0.5}"
export SDIF_VAE="${SDIF_VAE:-fp16fix}"

exec .venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600
