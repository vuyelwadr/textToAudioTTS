#!/usr/bin/env bash
# Bootstrap Chatterbox-Turbo (ResembleAI/chatterbox-turbo) for Mahti/Puhti.
#
# Model   : ResembleAI/chatterbox-turbo  (Apache-2.0)
# Package : chatterbox-tts  (pip)
# GitHub  : https://github.com/resemble-ai/chatterbox
#
# Architecture: 350M GPT-2-based token-to-speech model (T3) + distilled
#   S3Gen mel decoder (1-step).  Supports paralinguistic tags ([laugh],
#   [chuckle], etc.) and zero-shot voice cloning via audio_prompt_path.
# GPU requirement: CUDA, ≥4 GB VRAM (A100 / V100-32 recommended).
# Output SR: 24 000 Hz.
#
# HF cache note: python_hf_download calls snapshot_download, which
# populates $HF_HOME/hub/models--ResembleAI--chatterbox-turbo.
# ChatterboxTurboTTS.from_pretrained() resolves that cache offline on
# compute nodes after bootstrap.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

bootstrap_chatterbox_turbo() {
    local env_path="$MAHTI_ENV_ROOT/chatterbox-turbo"
    local bootstrap_marker="$env_path/.bootstrap-complete"
    local python_cmd

    python_cmd="$(resolve_python_command)"

    if [[ -d "$env_path" && ! -f "$bootstrap_marker" ]]; then
        rm -rf "$env_path"
    fi
    ensure_python_venv "chatterbox-turbo"
    activate_python_venv "chatterbox-turbo"

    # chatterbox-tts requires torch==2.6.0 and torchaudio==2.6.0 (Python <3.14).
    # The system pytorch module (load_pytorch_module) does NOT include torchvision,
    # which causes "RuntimeError: operator torchvision::nms does not exist".
    # We must install a complete self-contained PyTorch stack that includes
    # torchvision, matching the CUDA version of the compute nodes (cu121).
    # Install from PyTorch official index to get all three packages together.
    python -m pip install --upgrade pip wheel
    python -m pip install \
        "torch==2.6.0+cu121" \
        "torchvision==0.21.0+cu121" \
        "torchaudio==2.6.0+cu121" \
        --extra-index-url https://download.pytorch.org/whl/cu121

    # Install chatterbox-tts with --no-deps to prevent it from pulling
    # a conflicting torch version. All required non-torch deps are
    # installed explicitly below.
    python -m pip install --no-deps "chatterbox-tts"

    # chatterbox-tts non-torch dependencies (explicit, matching PyPI metadata)
    python -m pip install \
        "numpy>=1.24.0,<2.0.0" \
        "librosa==0.11.0" \
        "s3tokenizer" \
        "transformers==5.2.0" \
        "diffusers==0.29.0" \
        "resemble-perth>=1.0.0" \
        "conformer==0.3.2" \
        "safetensors==0.5.3" \
        "spacy-pkuseg" \
        "pykakasi==2.3.0" \
        "gradio==6.8.0" \
        "pyloudnorm" \
        "omegaconf"

    # Pre-download model weights (T3 backbone + S3Gen decoder safetensors)
    # into $MAHTI_MODEL_ROOT and the HF hub cache simultaneously.
    python_hf_download \
        "ResembleAI/chatterbox-turbo" \
        "$MAHTI_MODEL_ROOT/chatterbox-turbo/ResembleAI__chatterbox-turbo"
}

run_bootstrap_once "chatterbox-turbo" bootstrap_chatterbox_turbo
activate_python_venv "chatterbox-turbo"

log "Chatterbox-Turbo bootstrap completed"
