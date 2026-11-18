#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

add_python_libdir_to_ld_library_path() {
    local python_libdir

    python_libdir="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
    if [[ -n "$python_libdir" ]]; then
        export LD_LIBRARY_PATH="$python_libdir:${LD_LIBRARY_PATH:-}"
    fi
}

add_python_libdir_to_ld_library_path

ensure_voxcpm2_venv() {
    local env_name="voxcpm2"
    local env_path="$MAHTI_ENV_ROOT/$env_name"
    local bootstrap_marker="$env_path/.bootstrap-complete"
    local lock_fd=""

    acquire_lock lock_fd "venv-$env_name"

    if [[ -x "$env_path/bin/python" ]] && ! "$env_path/bin/python" -V >/dev/null 2>&1; then
        warn "recreating broken virtual environment $env_path"
        rm -rf "$env_path"
        rm -f "$bootstrap_marker"
    fi

    # CSC's PyTorch Python is a Singularity wrapper. A copied venv embeds the
    # container Python binary and then fails to find libpython on the host.
    # Symlinking keeps the wrapper entrypoint intact.
    if [[ ! -f "$env_path/bin/activate" ]]; then
        rm -f "$bootstrap_marker"
        log "creating virtual environment $env_path"
        python3 -m venv --system-site-packages "$env_path"
    fi

    activate_python_venv "$env_name"
    python -m pip install --upgrade pip setuptools wheel
    release_lock "$lock_fd"
}

bootstrap_voxcpm2() {
    ensure_voxcpm2_venv

    python -m pip install voxcpm soundfile faster-whisper
    python_hf_download "openbmb/VoxCPM2" "$MAHTI_MODEL_ROOT/voxcpm/openbmb__VoxCPM2"
}

run_bootstrap_once "voxcpm2" bootstrap_voxcpm2
activate_python_venv "voxcpm2"

log "VoxCPM2 bootstrap completed"
