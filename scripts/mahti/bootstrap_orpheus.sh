#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9
ensure_python_venv "orpheus-tts"

ORPHEUS_DIR="$MAHTI_EXTERNAL_ROOT/Orpheus-TTS"
ORPHEUS_MODEL_ID="${ORPHEUS_MODEL_ID:-canopylabs/orpheus-3b-0.1-ft}"
ensure_repo "https://github.com/canopyai/Orpheus-TTS.git" "$ORPHEUS_DIR"
activate_python_venv "orpheus-tts"

# The CSC PyTorch module stack already exposes a compatible vllm build via
# system site packages. Keep the virtualenv clear of an older wheel that may be
# ABI-incompatible with the loaded torch module.
python -m pip uninstall -y vllm >/dev/null 2>&1 || true
python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("vllm") is None:
	raise SystemExit("vllm is not available from the active CSC module stack")
PY
python -m pip install --no-deps snac
python -m pip install --no-deps -e "$ORPHEUS_DIR/orpheus_tts_pypi"
python_hf_download "$ORPHEUS_MODEL_ID" "$MAHTI_MODEL_ROOT/${ORPHEUS_MODEL_ID##*/}"

log "Orpheus bootstrap completed"