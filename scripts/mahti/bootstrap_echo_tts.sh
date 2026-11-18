#!/usr/bin/env bash
# Bootstrap Echo-TTS (jordand/echo-tts-base) for Mahti/Puhti.
#
# Model   : jordand/echo-tts-base  (CC-BY-NC-SA-4.0 – non-commercial only)
# GitHub  : https://github.com/jordandare/echo-tts
# Blog    : https://jordandarefsky.com/blog/2025/echo/
#
# Architecture: 2.4B diffusion (DiT) + Fish-Speech S1-DAC autoencoder.
# GPU requirement: CUDA, ≥8 GB VRAM (A100 or V100-32 recommended).
# Output SR: 44 100 Hz.
#
# Note: the Fish-Speech S1-DAC autoencoder weights (loaded by
# load_fish_ae_from_hf() inside inference.py) are downloaded on first
# inference call; they are NOT pre-fetched here because the exact HF
# repo ID is internal to echo-tts/autoencoder.py and may change across
# commits.  Run a short smoke-check job first to populate the cache.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bootstrap_echo_tts() {
    local echo_root="$MAHTI_EXTERNAL_ROOT/echo-tts"
    local env_path="$MAHTI_ENV_ROOT/echo-tts"
    local bootstrap_marker="$env_path/.bootstrap-complete"

    ensure_repo "https://github.com/jordandare/echo-tts.git" "$echo_root"

    if [[ -d "$env_path" && ! -f "$bootstrap_marker" ]]; then
        rm -rf "$env_path"
    fi
    ensure_python_venv "echo-tts"
    activate_python_venv "echo-tts"

    # Requirements from https://github.com/jordandare/echo-tts/blob/main/requirements.txt
    # except torchcodec/gradio. run_echo_tts.py injects a soundfile-backed
    # AudioDecoder shim before importing inference.py, which avoids Puhti's
    # missing libnvrtc.so.13 torchcodec failure.
    # Note: torchcodec is NOT installed – the shim makes it unnecessary.
    python -m pip install \
        "torch>=2.9.1" \
        "torchaudio>=2.9.1" \
        "huggingface-hub" \
        "numpy" \
        "safetensors" \
        "einops" \
        "soundfile"

    # Pre-download main Echo-TTS weights into the model store.
    # inference.py uses hf_hub_download(), which resolves against HF_HOME
    # (set to $MAHTI_CACHE_ROOT/huggingface in common.sh).  snapshot_download
    # (called by python_hf_download) also populates that hub cache, so the
    # runner will work offline on compute nodes after this step.
    python_hf_download \
        "jordand/echo-tts-base" \
        "$MAHTI_MODEL_ROOT/echo-tts/jordand__echo-tts-base"
}

run_bootstrap_once "echo-tts" bootstrap_echo_tts
activate_python_venv "echo-tts"

log "Echo-TTS bootstrap completed"
