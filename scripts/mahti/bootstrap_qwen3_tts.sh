#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bootstrap_qwen3_tts() {
    ensure_python_venv "qwen3-tts"
    activate_python_venv "qwen3-tts"

    python -m pip install qwen-tts faster-whisper soundfile
    python_hf_download "Qwen/Qwen3-TTS-12Hz-1.7B-Base" "$MAHTI_MODEL_ROOT/qwen3-tts/Qwen__Qwen3-TTS-12Hz-1.7B-Base"
    python_hf_download "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" "$MAHTI_MODEL_ROOT/qwen3-tts/Qwen__Qwen3-TTS-12Hz-1.7B-VoiceDesign"
}

run_bootstrap_once "qwen3-tts" bootstrap_qwen3_tts
activate_python_venv "qwen3-tts"

log "Qwen3-TTS bootstrap completed"
