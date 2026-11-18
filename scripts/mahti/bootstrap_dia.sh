#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bootstrap_dia() {
    ensure_python_venv "dia"
    activate_python_venv "dia"

    python -m pip install "transformers>=4.57.0" soundfile

    if ! python - <<'PY' >/dev/null 2>&1; then
from transformers import AutoProcessor, DiaForConditionalGeneration
PY
        python -m pip install "git+https://github.com/huggingface/transformers.git"
    fi

    python_hf_download "nari-labs/Dia-1.6B-0626" "$MAHTI_MODEL_ROOT/dia/nari-labs__Dia-1.6B-0626"
}

run_bootstrap_once "dia" bootstrap_dia
activate_python_venv "dia"

log "Dia bootstrap completed"
