#!/usr/bin/env bash
# Bootstrap script for kugelaudio/kugelaudio-0-open on Puhti/Mahti.
#
# Model: kugelaudio/kugelaudio-0-open  (7B, hybrid AR+Diffusion)
# Repo:  https://github.com/Kugelaudio/kugelaudio-open
# VRAM:  ~19 GB (bfloat16 full-precision) → fits A100-40G on Mahti;
#        tight on V100-16G on Puhti but workable on V100-32G nodes.
#
# Usage (standalone):
#   module load pytorch/2.9
#   bash scripts/mahti/bootstrap_kugelaudio_open.sh
#
# Usage (called from run_kugelaudio_open.py via SLURM):
#   The runner sources this script to get the venv active.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

ENV_NAME="kugelaudio-open"
KUGEL_REPO_URL="https://github.com/Kugelaudio/kugelaudio-open.git"
KUGEL_ROOT="$MAHTI_EXTERNAL_ROOT/kugelaudio-open"
MODEL_HF_ID="kugelaudio/kugelaudio-0-open"
MODEL_LOCAL_DIR="$MAHTI_MODEL_ROOT/kugelaudio-open/kugelaudio__kugelaudio-0-open"

bootstrap_kugelaudio_open() {
    # ── 1. Clone or update the source repo ──────────────────────────────────
    ensure_repo "$KUGEL_REPO_URL" "$KUGEL_ROOT"

    # ── 2. Create / recreate the virtual environment ─────────────────────────
    ensure_python_venv "$ENV_NAME"
    activate_python_venv "$ENV_NAME"

    # ── 3. Install the kugelaudio_open package (no-deps) ─────────────────────
    # We skip gradio (UI-only) to keep the install lean on headless compute nodes.
    python -m pip install --no-deps -e "$KUGEL_ROOT"

    # ── 4. Install required runtime dependencies ──────────────────────────────
    # torch is supplied by the pytorch module; do not reinstall it here.
    python -m pip install \
        "transformers>=4.40.0" \
        "accelerate>=1.0.0" \
        "diffusers>=0.25.0" \
        "numpy>=1.24.0" \
        "scipy>=1.10.0" \
        "librosa>=0.10.0" \
        "soundfile>=0.12.0" \
        "tqdm>=4.65.0" \
        "audioseal>=0.1.0"

    # ── 5. Download model weights from HuggingFace ────────────────────────────
    # Includes model shards, config, voice embeddings (.pt) and voices.json.
    python_hf_download \
        "$MODEL_HF_ID" \
        "$MODEL_LOCAL_DIR"
}

run_bootstrap_once "$ENV_NAME" bootstrap_kugelaudio_open
activate_python_venv "$ENV_NAME"

log "KugelAudio-Open bootstrap completed (env: $ENV_NAME)"
