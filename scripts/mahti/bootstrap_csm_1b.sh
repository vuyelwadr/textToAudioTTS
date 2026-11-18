#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bootstrap_csm_1b() {
    ensure_python_venv "csm-1b"
    activate_python_venv "csm-1b"

    python -m pip install \
        "transformers>=4.52.1" \
        "datasets[audio]>=3.0.0" \
        accelerate \
        sentencepiece \
        soundfile

    python_hf_download "sesame/csm-1b" "$MAHTI_MODEL_ROOT/csm/sesame__csm-1b"
}

run_bootstrap_once "csm-1b" bootstrap_csm_1b
activate_python_venv "csm-1b"

log "CSM-1B bootstrap completed"
