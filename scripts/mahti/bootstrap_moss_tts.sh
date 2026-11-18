#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bootstrap_moss_tts() {
    local env_path="$MAHTI_ENV_ROOT/moss-tts"
    local bootstrap_marker="$env_path/.bootstrap-complete"

    if [[ -d "$env_path" && ! -f "$bootstrap_marker" ]]; then
        rm -rf "$env_path"
    fi
    ensure_python_venv "moss-tts"
    activate_python_venv "moss-tts"

    # transformers 5.0.0 is required for PreTrainedAudioTokenizerBase and Qwen3 internals.
    # torchaudio is used by the processor for audio I/O.
    python -m pip install \
        "transformers==5.0.0" \
        "torchaudio" \
        "soundfile==0.12.1" \
        "tiktoken==0.9.0" \
        "einops==0.8.1" \
        "safetensors==0.5.3" \
        "numpy<3"

    # Pre-fetch the main model weights.
    python_hf_download \
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer" \
        "$MAHTI_MODEL_ROOT/moss-tts/OpenMOSS-Team__MOSS-TTS-Local-Transformer"

    # Pre-fetch the audio codec weights (loaded automatically via trust_remote_code).
    python_hf_download \
        "OpenMOSS-Team/MOSS-Audio-Tokenizer" \
        "$MAHTI_MODEL_ROOT/moss-tts/OpenMOSS-Team__MOSS-Audio-Tokenizer"
}

run_bootstrap_once "moss-tts" bootstrap_moss_tts
activate_python_venv "moss-tts"

log "MOSS-TTS bootstrap completed"
