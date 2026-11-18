#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9
export COQUI_TOS_AGREED="${COQUI_TOS_AGREED:-1}"

bootstrap_xtts() {
    local env_name="xtts-v2"
    local env_path="$MAHTI_ENV_ROOT/$env_name"
    local bootstrap_marker="$env_path/.bootstrap-complete"
    local python_cmd="${PYTHON_FOR_XTTS:-python3.11}"
    local expected_version="3.11"

    command -v "$python_cmd" >/dev/null 2>&1 || die "required Python interpreter not found for XTTS bootstrap: $python_cmd"

    if [[ -d "$env_path" && ! -f "$bootstrap_marker" ]]; then
        warn "removing incomplete XTTS environment at $env_path before rebuild"
        rm -rf "$env_path"
    fi

    if [[ -f "$env_path/pyvenv.cfg" ]]; then
        if ! grep -q '^version = 3\.11' "$env_path/pyvenv.cfg" || grep -q '^include-system-site-packages = true$' "$env_path/pyvenv.cfg"; then
            warn "recreating XTTS environment at $env_path for isolated Python $expected_version"
            rm -rf "$env_path"
        fi
    fi

    if [[ ! -f "$env_path/bin/activate" ]]; then
        log "creating isolated XTTS virtual environment $env_path with $python_cmd"
        "$python_cmd" -m venv "$env_path"
    fi

    # shellcheck disable=SC1090
    source "$env_path/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install \
        "torch==2.3.1" \
        "torchaudio==2.3.1" \
        "TTS==0.22.0" \
        "transformers==4.46.3"
    python - <<'PY'
from TTS.api import TTS

TTS("tts_models/multilingual/multi-dataset/xtts_v2")
PY
}

run_bootstrap_once "xtts-v2" bootstrap_xtts
activate_python_venv "xtts-v2"

log "XTTS v2 bootstrap completed"
