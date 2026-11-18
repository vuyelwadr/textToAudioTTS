#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bootstrap_spark_tts() {
    local spark_root="$MAHTI_EXTERNAL_ROOT/Spark-TTS"
    local env_path="$MAHTI_ENV_ROOT/spark-tts"
    local bootstrap_marker="$env_path/.bootstrap-complete"

    ensure_repo "https://github.com/SparkAudio/Spark-TTS.git" "$spark_root"
    if [[ -d "$env_path" && ! -f "$bootstrap_marker" ]]; then
        rm -rf "$env_path"
    fi
    ensure_python_venv "spark-tts"
    activate_python_venv "spark-tts"

    python -m pip install \
        "transformers==4.46.2" \
        "omegaconf==2.3.0" \
        "einops==0.8.1" \
        "einx==0.3.0" \
        "numpy==2.2.3" \
        "packaging==24.2" \
        "safetensors==0.5.2" \
        "soundfile==0.12.1" \
        "soxr==0.5.0.post1"

    python_hf_download \
        "SparkAudio/Spark-TTS-0.5B" \
        "$MAHTI_MODEL_ROOT/spark-tts/SparkAudio__Spark-TTS-0.5B"
}

run_bootstrap_once "spark-tts" bootstrap_spark_tts
activate_python_venv "spark-tts"

log "Spark-TTS bootstrap completed"
