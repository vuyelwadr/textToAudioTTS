#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9
ensure_python_venv "fish-speech"

FISH_DIR="$MAHTI_EXTERNAL_ROOT/fish-speech"
ensure_repo "https://github.com/fishaudio/fish-speech.git" "$FISH_DIR"
activate_python_venv "fish-speech"

python -m pip install \
	"numpy<2" \
	click \
	"protobuf>=3.20,<6" \
	einops \
	librosa \
	loguru \
	loralib \
	safetensors \
	sentencepiece \
	soundfile \
	tqdm \
	transformers
python -m pip install --no-deps \
	argbind \
	descript-audio-codec \
	descript-audiotools \
	ffmpy \
	flatten-dict \
	julius \
	markdown2 \
	pyloudnorm \
	pystoi \
	randomname \
	torch-stoi
python -m pip install --no-deps -e "$FISH_DIR"
python -m pip install faster-whisper
python_hf_download "fishaudio/s2-pro" "$MAHTI_MODEL_ROOT/fish-speech/s2-pro"

log "Fish Speech bootstrap completed"