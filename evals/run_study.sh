#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=.
export HF_LOCAL_ONLY="${HF_LOCAL_ONLY:-true}"
export TEXT_TO_IMAGE_BACKEND=proteus
export STYLE_TRANSFER_BACKEND=proteus
export PROTEUS_MODEL_REPO="${PROTEUS_MODEL_REPO:-models/ProteusV0.2}"
export SDIF_OPTIMIZATION_PROFILE=fast
export SDIF_SERIALIZE_INFERENCE=true

PY=.venv/bin/python
SIZE="${SIZE:-768}"
TASK="${TASK:-text}"
NUM_PROMPTS="${NUM_PROMPTS:-12}"
DMD2_LORA=models/dmd2/dmd2_sdxl_4step_lora_fp16.safetensors
HYPER_LORA=models/hyper-sd/Hyper-SDXL-8steps-CFG-lora.safetensors
DMD2_ENV="SDXL_SPEED_LORA=$DMD2_LORA SDXL_SCHEDULER=lcm SDXL_TIMESTEPS=999,749,499,249"

run() {
  local name="$1"; shift
  local envs=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do envs+=("$1"); shift; done
  echo "=== $name"
  env SDIF_CACHE_MODE=none "${envs[@]}" "$PY" evals/gen_images.py \
    --name "$name" --task "$TASK" --width "$SIZE" --height "$SIZE" --num-prompts "$NUM_PROMPTS" "$@"
}

CONFIGS="${CONFIGS:-ref40 dpm8 dmd2_4 hyper8 dmd2_4_taesd deepcache20 dmd2_4_compile dmd2_4_compile_taesd}"

for cfg in $CONFIGS; do
  case "$cfg" in
    ref40) run ref40 --steps 40 --guidance 5.0 ;;
    dpm8) run dpm8 --steps 8 --guidance 5.0 ;;
    dmd2_4) run dmd2_4 $DMD2_ENV --guidance 0 ;;
    hyper8) run hyper8 SDXL_SPEED_LORA=$HYPER_LORA --steps 8 --guidance 5.0 ;;
    dmd2_4_taesd) run dmd2_4_taesd $DMD2_ENV SDIF_VAE=taesdxl --guidance 0 ;;
    deepcache20) run deepcache20 SDIF_CACHE_MODE=deepcache SDIF_DEEPCACHE_INTERVAL=3 --steps 20 --guidance 5.0 ;;
    dmd2_4_compile) run dmd2_4_compile $DMD2_ENV SDIF_TORCH_COMPILE=1 --guidance 0 --warmup 4 ;;
    dmd2_4_compile_taesd) run dmd2_4_compile_taesd $DMD2_ENV SDIF_TORCH_COMPILE=1 SDIF_VAE=taesdxl --guidance 0 --warmup 4 ;;
    *) echo "unknown config $cfg"; exit 1 ;;
  esac
done

DIRS=""
for cfg in $CONFIGS; do DIRS="$DIRS evals/out/$cfg"; done
HF_LOCAL_ONLY=false "$PY" evals/score.py --dirs $DIRS --ref evals/out/ref40 \
  --output "evals/results/summary_${TASK}_${SIZE}.json" --sheet "evals/results/sheet_${TASK}_${SIZE}.png"
