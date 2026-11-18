#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bootstrap_parler_tts_large() {
    ensure_python_venv "parler-tts-large"
    activate_python_venv "parler-tts-large"

    python -m pip install \
        "transformers==4.46.1" \
        sentencepiece \
        soundfile \
        "protobuf>=4.0.0" \
        descript-audio-codec \
        "descript-audiotools @ git+https://github.com/descriptinc/audiotools"
    python -m pip install --no-deps "git+https://github.com/huggingface/parler-tts.git"

    python_hf_download \
        "parler-tts/parler-tts-large-v1" \
        "$MAHTI_MODEL_ROOT/parler-tts/parler-tts__parler-tts-large-v1"
}

run_bootstrap_once "parler-tts-large" bootstrap_parler_tts_large
activate_python_venv "parler-tts-large"

log "Parler-TTS Large bootstrap completed"
