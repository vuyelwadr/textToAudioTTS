#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

MIMO_REPO_URL="https://github.com/XiaomiMiMo/MiMo-Audio.git"
MIMO_REPO_DIR="$MAHTI_EXTERNAL_ROOT/mimo-audio"
FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

bootstrap_mimo_audio() {
    load_pytorch_module 2.6
    ensure_repo "$MIMO_REPO_URL" "$MIMO_REPO_DIR"
    ensure_python_venv "mimo-audio"

    python -m pip install \
        accelerate==1.9.0 \
        huggingface_hub \
        librosa==0.11.0 \
        scipy==1.16.1 \
        sentencepiece \
        soundfile \
        transformers==4.49.0 \
        zhon==2.1.1

    if ! python -c 'import flash_attn' >/dev/null 2>&1; then
        python -m pip install "$FLASH_ATTN_WHEEL_URL"
    fi

    python_hf_download \
        "XiaomiMiMo/MiMo-Audio-Tokenizer" \
        "$MAHTI_MODEL_ROOT/mimo-audio/XiaomiMiMo__MiMo-Audio-Tokenizer"
    python_hf_download \
        "XiaomiMiMo/MiMo-Audio-7B-Instruct" \
        "$MAHTI_MODEL_ROOT/mimo-audio/XiaomiMiMo__MiMo-Audio-7B-Instruct"
}

run_bootstrap_once "mimo-audio" bootstrap_mimo_audio

log "MiMo-Audio bootstrap completed"
