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

invalidate_broken_audiobook_video_marker() {
    local env_name="audiobook-video"
    local env_path="$MAHTI_ENV_ROOT/$env_name"
    local marker="$env_path/.bootstrap-complete"

    if [[ -x "$env_path/bin/python" ]] && ! "$env_path/bin/python" -V >/dev/null 2>&1; then
        rm -f "$marker"
        return
    fi
    if [[ -f "$env_path/bin/python3" && ! -L "$env_path/bin/python3" ]]; then
        rm -f "$marker"
    fi
    # A partially-installed venv can still have a working interpreter, so the
    # marker alone is not proof that the audit stack is usable. Re-bootstrap if
    # the whisper/hub imports or the bundled ffmpeg wrapper are missing.
    if [[ -f "$marker" ]]; then
        if ! "$env_path/bin/python" -c 'import numpy, soundfile, faster_whisper, huggingface_hub' >/dev/null 2>&1; then
            rm -f "$marker"
        elif [[ ! -x "$env_path/bin/ffmpeg" ]]; then
            rm -f "$marker"
        fi
    fi
}

ensure_audiobook_video_venv() {
    local env_name="audiobook-video"
    local env_path="$MAHTI_ENV_ROOT/$env_name"
    local bootstrap_marker="$env_path/.bootstrap-complete"
    local lock_fd=""

    acquire_lock lock_fd "venv-$env_name"

    if [[ -x "$env_path/bin/python" ]] && ! "$env_path/bin/python" -V >/dev/null 2>&1; then
        warn "recreating broken virtual environment $env_path"
        rm -rf "$env_path"
        rm -f "$bootstrap_marker"
    fi
    if [[ -f "$env_path/bin/python3" && ! -L "$env_path/bin/python3" ]]; then
        warn "recreating copied virtual environment $env_path to keep the PyTorch wrapper entrypoint"
        rm -rf "$env_path"
        rm -f "$bootstrap_marker"
    fi

    if [[ ! -f "$env_path/bin/activate" ]]; then
        rm -f "$bootstrap_marker"
        log "creating virtual environment $env_path"
        python3 -m venv --system-site-packages "$env_path"
    fi

    activate_python_venv "$env_name"
    python -m pip install --upgrade pip setuptools wheel
    release_lock "$lock_fd"
}

bootstrap_audiobook_video() {
    ensure_audiobook_video_venv

    # Pin huggingface_hub below 1.0: faster-whisper 1.2.x imports the private
    # huggingface_hub._snapshot_download module, which 1.x removed. Without
    # this pin a fresh bootstrap breaks the audit with ModuleNotFoundError.
    python -m pip install \
        "huggingface_hub<1.0" \
        "faster-whisper==1.2.0" \
        pillow numpy soundfile imageio-ffmpeg openai-whisper

    python - <<'PY'
import os
import shlex
from pathlib import Path

import imageio_ffmpeg

venv_bin = Path(os.environ["VIRTUAL_ENV"]) / "bin"
ffmpeg_binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
wrapper_path = venv_bin / "ffmpeg"
wrapper_path.write_text(
    "#!/usr/bin/env bash\n"
    f"exec {shlex.quote(str(ffmpeg_binary))} \"$@\"\n",
    encoding="utf-8",
)
wrapper_path.chmod(0o755)
print(f"[csc] bundled ffmpeg wrapper: {wrapper_path}")
PY
}

invalidate_broken_audiobook_video_marker
run_bootstrap_once "audiobook-video" bootstrap_audiobook_video
activate_python_venv "audiobook-video"

log "Audiobook video bootstrap completed"
