#!/usr/bin/env bash
# Bootstrap script for StepFun's Step-Audio-EditX (3B, Apache-2.0).
#
# Unlike most models wired in under scripts/mahti/, Step-Audio-EditX's own
# inference code (tts.py / model_loader.py) imports vLLM unconditionally to
# run its audio LLM component, and its pyproject.toml pins an exact vLLM dev
# wheel URL via [tool.uv.sources] plus a matching uv.lock. Hand-picking a pip
# subset here would risk an ABI mismatch between torch and that pinned vLLM
# build, so -- matching the upstream README and Dockerfile exactly -- this
# bootstrap uses `uv sync` against the repo's own lockfile instead of
# `ensure_python_venv` + `pip install`. UV_PROJECT_ENVIRONMENT redirects the
# venv uv creates into the shared $MAHTI_ENV_ROOT layout so it still lives
# alongside every other model's venv and `activate_python_venv` keeps working
# for callers.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# /projappl/project_2010549 has been observed at 100% *inode* utilization
# (confirmed via `df -i /projappl/project_2010549`; see also bootstrap_tada.sh)
# -- so any new file there (venv, lock file, or a fresh git clone) currently
# fails outright with "Disk quota exceeded", regardless of byte size. This
# venv is also unusually heavy (uv-managed torch, vLLM, deepspeed, ray, cupy,
# etc. -- hundreds of packages), so route the venv, its flock lock directory,
# and the cloned repo checkout onto scratch instead of the projappl-backed
# defaults: MAHTI_ENV_ROOT alone is not enough, because common.sh derives
# MAHTI_LOCK_ROOT and MAHTI_EXTERNAL_ROOT from MAHTI_PROJECT_ROOT
# (projappl-backed) independently of MAHTI_ENV_ROOT, and
# run_bootstrap_once/ensure_python_venv both flock a file under
# MAHTI_LOCK_ROOT before touching MAHTI_ENV_ROOT at all.
export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
export MAHTI_LOCK_ROOT="$MAHTI_WORK_ROOT/.locks"
mkdir -p "$MAHTI_LOCK_ROOT"

load_pytorch_module 2.9

STEP_AUDIO_EDITX_REPO_URL="https://github.com/stepfun-ai/Step-Audio-EditX.git"

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    local uv_base="$HOME/.local/uv-$(uname -m)"
    if [[ ! -x "$uv_base/bin/uv" ]]; then
        PYTHONUSERBASE="$uv_base" python3 -m pip install --user --quiet uv
    fi
    export PATH="$uv_base/bin:$PATH"
    hash -r
    command -v uv >/dev/null 2>&1 || die "failed to provision uv"
}

bootstrap_step_audio_editx() {
    local repo_dir="$MAHTI_EXTERNAL_ROOT/Step-Audio-EditX"
    local env_path="$MAHTI_ENV_ROOT/step-audio-editx"

    ensure_repo "$STEP_AUDIO_EDITX_REPO_URL" "$repo_dir"
    ensure_uv

    # Let uv fully own this venv (no --system-site-packages): the pinned
    # vLLM wheel and torch build need to match exactly, and uv.lock is the
    # maintainers' own tested pin set (same one baked into their Dockerfile).
    export UV_PROJECT_ENVIRONMENT="$env_path"
    (
        cd "$repo_dir"
        uv sync --python "$(command -v python3)"
    )

    activate_python_venv "step-audio-editx"

    # uv.lock resolves onnxruntime-gpu to 1.27.0, which needs CUDA 13's
    # libcudart.so.13 -- newer than the CUDA 12 runtime the rest of this
    # venv's pip-installed nvidia-*-cu12 packages provide (and newer than
    # what's baked into the pytorch/2.9 module's own container).
    #
    # Install ONLY nvidia-cuda-runtime (cu13's unsuffixed renamed successor
    # to nvidia-cuda-runtime-cu12) here, and nothing else from the cu13
    # family. onnxruntime's CUDAExecutionProvider fails at *runtime* anyway
    # on this cluster (the A100 nodes' driver is too old for CUDA 13) and
    # falls back to CPUExecutionProvider on its own, so cudart is only ever
    # needed to satisfy onnxruntime's *import-time* linkage, not to actually
    # run CUDA ops. Do NOT also install nvidia-cublas/-cudnn-cu13/-curand/
    # -cufft/-cusparse/-cusolver/-nvjitlink/-cuda-nvrtc: those pip packages
    # install into the SAME "nvidia/<component>/lib/" site-packages paths as
    # torch's own nvidia-*-cu12 dependencies (no cu12/cu13 split in the
    # directory name, only in the PyPI package name), so installing them
    # silently overwrites the cu12 libraries torch/CosyVoice actually need,
    # breaking cuDNN-backed ops with CUDNN_STATUS_NOT_INITIALIZED. Confirmed
    # by direct comparison: a venv with only nvidia-cuda-runtime added got a
    # Step-Audio-EditX job all the way to a loaded vLLM engine; adding the
    # rest of the cu13 set broke CosyVoice's first Conv1d call.
    uv pip install --python "$VIRTUAL_ENV/bin/python" nvidia-cuda-runtime
}

run_bootstrap_once "step-audio-editx" bootstrap_step_audio_editx
activate_python_venv "step-audio-editx"

# Deliberately NOT inside the run_bootstrap_once-guarded function above:
# /scratch/project_2010549's Lustre *project* quota is at its 1TiB cap (see
# report), so callers may point MAHTI_MODEL_ROOT at a SLURM job's ephemeral
# local NVMe scratch (--gres=nvme:900, $LOCAL_SCRATCH) instead of the
# default /scratch location to work around that. NVMe is wiped between jobs,
# so the weights must be (re-)fetched on every invocation regardless of
# whether the one-time venv/uv-sync bootstrap already ran. python_hf_download
# resumes/verifies existing files cheaply when MAHTI_MODEL_ROOT is the normal
# persistent /scratch path, so this is a fast no-op there.
python_hf_download \
    "stepfun-ai/Step-Audio-Tokenizer" \
    "$MAHTI_MODEL_ROOT/step-audio-editx/Step-Audio-Tokenizer"
# Full snapshot: this repo also bundles the CosyVoice-300M-25Hz vocoder
# checkpoint under a subdirectory that tts.py expects to find alongside the
# LLM weights.
python_hf_download \
    "stepfun-ai/Step-Audio-EditX" \
    "$MAHTI_MODEL_ROOT/step-audio-editx/Step-Audio-EditX"

log "Step-Audio-EditX bootstrap completed"
