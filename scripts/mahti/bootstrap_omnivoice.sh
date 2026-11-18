#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bootstrap_omnivoice() {
    ensure_python_venv "omnivoice"
    activate_python_venv "omnivoice"

    python -m pip uninstall -y torchcodec || true
    python -m pip install omnivoice faster-whisper soundfile
    python_hf_download "k2-fsa/OmniVoice" "$MAHTI_MODEL_ROOT/omnivoice/k2-fsa__OmniVoice"
}

run_bootstrap_once "omnivoice" bootstrap_omnivoice
activate_python_venv "omnivoice"

log "OmniVoice bootstrap completed"
