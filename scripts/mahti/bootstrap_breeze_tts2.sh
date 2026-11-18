#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# Keep the large repo/venv on scratch; this also avoids the project's inode pressure.
export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
export MAHTI_LOCK_ROOT="$MAHTI_WORK_ROOT/.locks"
mkdir -p "$MAHTI_ENV_ROOT" "$MAHTI_EXTERNAL_ROOT" "$MAHTI_LOCK_ROOT"

load_pytorch_module 2.9

bootstrap_breeze_tts2() {
    local repo="$MAHTI_EXTERNAL_ROOT/breeze-tts"
    ensure_repo "https://github.com/breezeblue-ai/breeze-tts.git" "$repo"
    ensure_python_venv "breeze-tts2"
    activate_python_venv "breeze-tts2"
    python -m pip install -r "$repo/requirements.txt"
    python_hf_download "BreezeBlue/Breeze-TTS-2" "$MAHTI_MODEL_ROOT/breeze-tts2/BreezeBlue__Breeze-TTS-2"
}

run_bootstrap_once "breeze-tts2" bootstrap_breeze_tts2
activate_python_venv "breeze-tts2"
warn "Breeze TTS 2 open weights/self-hosted outputs are licensed for research/non-commercial use only; check the model license before distribution/monetization."
log "Breeze TTS 2 bootstrap completed"
