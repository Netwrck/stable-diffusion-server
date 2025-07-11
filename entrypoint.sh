#!/usr/bin/env bash
set -euo pipefail

case "${MODE_TO_RUN}" in
  api)
    # Classic FastAPI server for dev
    exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 600
    ;;
  serverless)
    # Hand off to RunPod's launcher; rp_handler.py already calls runpod.serverless.start
    exec python rp_handler.py
    ;;
  *)
    echo "Unknown MODE_TO_RUN='${MODE_TO_RUN}'" >&2
    exit 1
    ;;
esac