#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# /projappl/project_2010549 has been observed at 100% *inode* utilization (see
# bootstrap_step_audio_editx.sh, bootstrap_tada.sh), so any new file there --
# venv, flock lock file, or fresh git clone -- fails outright with "Disk quota
# exceeded" regardless of byte size. MAHTI_ENV_ROOT alone is not enough,
# because common.sh derives MAHTI_LOCK_ROOT and MAHTI_EXTERNAL_ROOT from
# MAHTI_PROJECT_ROOT (projappl-backed) independently of MAHTI_ENV_ROOT, and
# run_bootstrap_once/ensure_python_venv both flock a file under
# MAHTI_LOCK_ROOT before touching MAHTI_ENV_ROOT at all. Route the venv, its
# lock directory, and the cloned repo checkout onto scratch instead.
export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
export MAHTI_LOCK_ROOT="$MAHTI_WORK_ROOT/.locks"
mkdir -p "$MAHTI_LOCK_ROOT"

load_pytorch_module 2.9

bootstrap_fish_s2pro() {
    ensure_python_venv "fish-s2pro"
    activate_python_venv "fish-s2pro"

    # Fish Audio S2-Pro ships as a checkpoint on the fishaudio/fish-speech
    # GitHub repo's current inference stack (fish_speech/models/text2semantic
    # and fish_speech/models/dac): there is no standalone pip package that
    # exposes a Python inference class the way qwen-tts/faster-qwen3-tts do,
    # so the repo itself has to be checked out and installed editable. This
    # uses its own scratch-backed checkout under MAHTI_EXTERNAL_ROOT (see
    # quota note above) rather than bootstrap_fish_speech.sh's projappl-backed
    # default, so run_fish_s2pro.py resolves the same MAHTI_EXTERNAL_ROOT at
    # call time -- see the matching override in run_model.slurm's
    # `fish-s2pro` case.
    FISH_DIR="$MAHTI_EXTERNAL_ROOT/fish-speech"
    ensure_repo "https://github.com/fishaudio/fish-speech.git" "$FISH_DIR"

    # Core inference-time dependencies for fish_speech/models/text2semantic/inference.py
    # and fish_speech/models/dac/inference.py. PyTorch/torchaudio themselves come from
    # load_pytorch_module above via --system-site-packages, so they are excluded here.
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
        transformers \
        hydra-core \
        omegaconf \
        pyrootutils \
        resampy \
        "einx[torch]" \
        zstandard \
        pydub \
        modelscope \
        opencc-python-reimplemented \
        silero-vad \
        ormsgpack \
        tiktoken \
        "pydantic==2.9.2" \
        cachetools \
        rich \
        natsort
    # Isolated with --no-deps because descript-audio-codec/descript-audiotools pull in
    # an older protobuf/numpy pin that conflicts with the versions pinned above.
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
    # faster-whisper's onnxruntime dependency drags protobuf up past the <6 pin
    # above (needed for descript-audiotools' generated pb2 code), so re-pin it
    # back down as the last word after faster-whisper installs.
    python -m pip install faster-whisper
    python -m pip install "protobuf>=3.20,<6"

    python_hf_download "fishaudio/s2-pro" "$MAHTI_MODEL_ROOT/fish-speech/s2-pro"
}

run_bootstrap_once "fish-s2pro" bootstrap_fish_s2pro
activate_python_venv "fish-s2pro"

log "Fish Audio S2-Pro bootstrap completed"
