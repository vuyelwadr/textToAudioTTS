#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bootstrap_voxtral() {
    ensure_python_venv "voxtral"
    activate_python_venv "voxtral"

    python -m pip install "vllm==0.18.0" "vllm-omni==0.18.0" soundfile

    python_hf_download "mistralai/Voxtral-4B-TTS-2603" \
        "$MAHTI_MODEL_ROOT/voxtral/mistralai__Voxtral-4B-TTS-2603"
}

run_bootstrap_once "voxtral" bootstrap_voxtral
activate_python_venv "voxtral"

log "Voxtral bootstrap completed"
