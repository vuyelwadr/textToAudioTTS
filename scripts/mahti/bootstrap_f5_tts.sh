#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9
ensure_python_venv "f5-tts"

F5_DIR="$MAHTI_EXTERNAL_ROOT/F5-TTS"
ensure_repo "https://github.com/SWivid/F5-TTS.git" "$F5_DIR"
activate_python_venv "f5-tts"

python -m pip install \
	"numpy<2" \
	cached-path \
	faster-whisper \
	huggingface_hub \
	hydra-core \
	matplotlib \
	omegaconf \
	pydub \
	pypinyin \
	rjieba \
	safetensors \
	soundfile \
	tqdm \
	torchdiffeq \
	transformers \
	vocos \
	x_transformers
python -m pip install --no-deps -e "$F5_DIR"
python_hf_download "SWivid/F5-TTS" "$MAHTI_MODEL_ROOT/F5-TTS"

log "F5-TTS bootstrap completed"