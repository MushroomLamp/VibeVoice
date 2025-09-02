#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden by env or compose)
: "${MODEL_PATH:=microsoft/VibeVoice-1.5B}"
: "${DEVICE:=cuda}"
: "${PORT:=7860}"
: "${SHARE:=false}"
: "${INFERENCE_STEPS:=10}"

echo "Using device: ${DEVICE}"
echo "Model path: ${MODEL_PATH}"

# Install project in editable mode (mounted volume)
pip install -e .

# Optional flash-attn if present
if [[ "${ENABLE_FLASH_ATTN:-}" == "1" ]]; then
  pip install --no-build-isolation flash-attn || echo "flash-attn install skipped"
fi

ARGS=("--model_path" "${MODEL_PATH}" "--device" "${DEVICE}" "--inference_steps" "${INFERENCE_STEPS}" "--port" "${PORT}")

if [[ "${SHARE}" == "true" ]]; then
  ARGS+=("--share")
fi

exec python demo/gradio_demo.py "${ARGS[@]}"


