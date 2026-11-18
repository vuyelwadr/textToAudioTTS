#!/usr/bin/env bash
# Bootstrap TADA (Hume AI's Text-Acoustic Dual-Aligned voice-cloning TTS) for
# Mahti/Puhti. https://github.com/HumeAI/tada
#
# Caveats:
# 1. hume-tada pins torch>=2.7.0,<2.8.0 (see its pyproject.toml), which
#    conflicts with the repo-wide default of `load_pytorch_module 2.9` used
#    by most other bootstrap_*.sh scripts here. Load pytorch/2.7 instead so
#    pip does not try to reinstall torch inside the venv to satisfy that pin.
# 2. TADA-1B/TADA-3B-ML are built on Llama 3.2 1B/3B respectively and ship
#    under the Llama 3.2 Community License. HumeAI's model cards ask users to
#    accept the Llama 3.2 license on Hugging Face (via meta-llama/Llama-3.2-1B
#    or meta-llama/Llama-3.2-3B) before downloading. This relies on the same
#    HUGGINGFACE_API_KEY / HF_TOKEN flow already wired through common.sh for
#    other gated models (e.g. Orpheus).
# 3. Only TADA-1B is prefetched by default here to keep bootstrap fast; pass
#    a different --model-name to run_tada.py (e.g. HumeAI/tada-3b-ml) to
#    exercise the larger checkpoint, which will download on first use.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# /projappl/<project> has been observed at 100% inode utilization on this
# project (confirmed via `df -i /projappl/project_2010549` returning
# 100000/100000 used). Route both the venv and the flock lock directory onto
# scratch instead of the projappl-backed defaults: MAHTI_ENV_ROOT alone is
# not enough, because common.sh derives MAHTI_LOCK_ROOT from
# MAHTI_PROJECT_ROOT (projappl-backed) independently of MAHTI_ENV_ROOT, and
# run_bootstrap_once/ensure_python_venv both flock a file under
# MAHTI_LOCK_ROOT before touching MAHTI_ENV_ROOT at all. Without this second
# override, the very first `exec {fd}> "$MAHTI_LOCK_ROOT/bootstrap-tada.lock"`
# fails with "Disk quota exceeded" before the venv override ever helps.
export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
export MAHTI_LOCK_ROOT="$MAHTI_WORK_ROOT/.locks"
mkdir -p "$MAHTI_LOCK_ROOT"

load_pytorch_module 2.7

bootstrap_tada() {
    ensure_python_venv "tada"
    activate_python_venv "tada"

    python -m pip install hume-tada soundfile
    python_hf_download "HumeAI/tada-codec" "$MAHTI_MODEL_ROOT/tada/HumeAI__tada-codec"
    python_hf_download "HumeAI/tada-1b" "$MAHTI_MODEL_ROOT/tada/HumeAI__tada-1b"
}

run_bootstrap_once "tada" bootstrap_tada
activate_python_venv "tada"

log "TADA bootstrap completed"
