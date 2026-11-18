#!/bin/bash

set -euo pipefail

if [[ -f "$PWD/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="$PWD/scripts/mahti"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mahti"
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.7
bash "$SCRIPT_DIR/bootstrap_inworld_tts.sh"

# shellcheck disable=SC1091
source "$MAHTI_EXTERNAL_ROOT/inworld-tts/.venv/bin/activate"

python "$SCRIPT_DIR/run_inworld_tts_poc.py" "$@"
