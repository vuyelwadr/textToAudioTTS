#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

load_repo_env() {
    local env_file="$REPO_ROOT/.env"

    if [[ -f "$env_file" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$env_file"
        set +a
    fi

    if [[ -n "${HUGGINGFACE_API_KEY:-}" ]]; then
        export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HUGGINGFACE_API_KEY}"
        export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_API_KEY}"
    elif [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
        export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"
    elif [[ -n "${HF_TOKEN:-}" ]]; then
        export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
    fi
}

load_repo_env

resolve_csc_project_account() {
    local candidate

    for candidate in \
        "${CSC_PROJECT:-}" \
        "${SLURM_JOB_ACCOUNT:-}" \
        "${SLURM_ACCOUNT:-}" \
        "${SBATCH_ACCOUNT:-}"; do
        if [[ -n "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    printf '%s\n' ""
}

export CSC_PROJECT="$(resolve_csc_project_account)"

detect_cluster_name() {
    local raw_name

    raw_name="${CSC_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-$(hostname 2>/dev/null || echo local)}}"
    raw_name="${raw_name,,}"

    if [[ "$raw_name" == *puhti* ]]; then
        printf '%s\n' "puhti"
        return
    fi
    if [[ "$raw_name" == *mahti* ]]; then
        printf '%s\n' "mahti"
        return
    fi
    if [[ "$raw_name" == *roihu* ]]; then
        printf '%s\n' "roihu"
        return
    fi

    printf '%s\n' "local"
}

resolve_project_parent() {
    if [[ -n "${PROJAPPL:-}" && -d "${PROJAPPL:-}" ]]; then
        printf '%s\n' "$PROJAPPL"
        return
    fi
    if [[ -n "${CSC_PROJECT:-}" && -d "/projappl/$CSC_PROJECT" ]]; then
        printf '%s\n' "/projappl/$CSC_PROJECT"
        return
    fi

    printf '%s\n' "$REPO_ROOT/.csc/$(detect_cluster_name)/project"
}

resolve_work_parent() {
    if [[ -n "${SCRATCH:-}" && -d "${SCRATCH:-}" ]]; then
        printf '%s\n' "$SCRATCH"
        return
    fi
    if [[ -n "${CSC_PROJECT:-}" && -d "/scratch/$CSC_PROJECT/$USER" ]]; then
        printf '%s\n' "/scratch/$CSC_PROJECT/$USER"
        return
    fi

    printf '%s\n' "$REPO_ROOT/.csc/$(detect_cluster_name)/work"
}

export CSC_CLUSTER_NAME="${CSC_CLUSTER_NAME:-$(detect_cluster_name)}"
DEFAULT_CLUSTER_ROOT="textToAudioTTS-$CSC_CLUSTER_NAME"
DEFAULT_PROJECT_ROOT="$(resolve_project_parent)/$DEFAULT_CLUSTER_ROOT"
DEFAULT_WORK_ROOT="$(resolve_work_parent)/$DEFAULT_CLUSTER_ROOT"

export MAHTI_REPO_ROOT="${MAHTI_REPO_ROOT:-${CSC_REPO_ROOT:-$REPO_ROOT}}"
export CSC_REPO_ROOT="$MAHTI_REPO_ROOT"
export MAHTI_PROJECT_ROOT="${MAHTI_PROJECT_ROOT:-${CSC_HEAVY_PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}}"
export CSC_HEAVY_PROJECT_ROOT="$MAHTI_PROJECT_ROOT"
export MAHTI_WORK_ROOT="${MAHTI_WORK_ROOT:-${CSC_HEAVY_WORK_ROOT:-$DEFAULT_WORK_ROOT}}"
export CSC_HEAVY_WORK_ROOT="$MAHTI_WORK_ROOT"
export MAHTI_EXTERNAL_ROOT="${MAHTI_EXTERNAL_ROOT:-${CSC_EXTERNAL_ROOT:-$MAHTI_PROJECT_ROOT/external}}"
export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
# Roihu's CPU and GPU node pools are different CPU architectures (x86_64 vs
# aarch64) sharing the same /projappl and /scratch filesystems. A venv with
# compiled extensions (numpy, etc.) built on one pool silently crashes when
# reused on the other, so isolate venvs by architecture on that cluster only
# (Mahti/Puhti are uniformly x86_64 and keep their existing unsuffixed path).
DEFAULT_VENV_ARCH_SUFFIX=""
if [[ "$CSC_CLUSTER_NAME" == "roihu" ]]; then
    DEFAULT_VENV_ARCH_SUFFIX="-$(uname -m)"
fi
export MAHTI_ENV_ROOT="${MAHTI_ENV_ROOT:-${CSC_ENV_ROOT:-$MAHTI_PROJECT_ROOT/venvs$DEFAULT_VENV_ARCH_SUFFIX}}"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
export MAHTI_MODEL_ROOT="${MAHTI_MODEL_ROOT:-${CSC_MODEL_ROOT:-$MAHTI_WORK_ROOT/models}}"
export CSC_MODEL_ROOT="$MAHTI_MODEL_ROOT"
export MAHTI_CACHE_ROOT="${MAHTI_CACHE_ROOT:-${CSC_CACHE_ROOT:-$MAHTI_WORK_ROOT/cache}}"
export CSC_CACHE_ROOT="$MAHTI_CACHE_ROOT"
export MAHTI_OUTPUT_ROOT="${MAHTI_OUTPUT_ROOT:-${CSC_OUTPUT_ROOT:-$MAHTI_REPO_ROOT/experiments/tts_naturalness/generated/all_samples}}"
export CSC_OUTPUT_ROOT="$MAHTI_OUTPUT_ROOT"
export HF_HOME="${HF_HOME:-$MAHTI_CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$MAHTI_CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MAHTI_CACHE_ROOT/xdg}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$MAHTI_CACHE_ROOT/xdg-data}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MAHTI_CACHE_ROOT/pip}"
export NLTK_DATA="${NLTK_DATA:-$MAHTI_CACHE_ROOT/nltk_data}"
export TTS_HOME="${TTS_HOME:-$MAHTI_MODEL_ROOT/tts}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$MAHTI_CACHE_ROOT/matplotlib}"
DEFAULT_TMPDIR="$MAHTI_CACHE_ROOT/tmp"
if [[ "${CSC_PREFER_LOCAL_SCRATCH:-0}" == "1" && -n "${LOCAL_SCRATCH:-}" ]]; then
    DEFAULT_TMPDIR="$LOCAL_SCRATCH"
fi

case "${TMPDIR:-}" in
    ""|/tmp/*|/local_scratch/*|/run/nvme/*)
        # /run/nvme/job_<id> is per-job local NVMe scratch tied to the SLURM job
        # that requested it. submit_audiobook_video.sh's `--export ALL` carries an
        # inherited TMPDIR from the GPU job into the chained CPU video job, where
        # that directory no longer exists/is inaccessible -- fall back instead of
        # trusting it blindly.
        export TMPDIR="$DEFAULT_TMPDIR"
        ;;
    *)
        export TMPDIR="$TMPDIR"
        ;;
esac
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
if [[ -z "${PIP_BUILD_TRACKER:-}" ]]; then
    mkdir -p "$TMPDIR"
    export PIP_BUILD_TRACKER="$(mktemp -d "$TMPDIR/pip-build-tracker-${SLURM_JOB_ID:-$$}-XXXXXX")"
fi
# Lock files are tiny and ephemeral -- put them on MAHTI_WORK_ROOT (scratch)
# rather than MAHTI_PROJECT_ROOT (/projappl), which CSC caps at a 100k-file
# inode quota that heavy venvs/external checkouts already exhaust on their
# own. A lock directory under a full /projappl silently breaks every
# bootstrap script's acquire_lock() with "Disk quota exceeded" before any
# real work starts.
export MAHTI_LOCK_ROOT="${MAHTI_LOCK_ROOT:-$MAHTI_WORK_ROOT/.locks}"

mkdir -p \
    "$MAHTI_PROJECT_ROOT" \
    "$MAHTI_WORK_ROOT" \
    "$MAHTI_EXTERNAL_ROOT" \
    "$MAHTI_ENV_ROOT" \
    "$MAHTI_MODEL_ROOT" \
    "$MAHTI_CACHE_ROOT" \
    "$MAHTI_OUTPUT_ROOT" \
    "$HF_HOME" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$TORCH_HOME" \
    "$XDG_CACHE_HOME" \
    "$XDG_DATA_HOME" \
    "$PIP_CACHE_DIR" \
    "$NLTK_DATA" \
    "$TTS_HOME" \
    "$MPLCONFIGDIR" \
    "$TMPDIR" \
    "$PIP_BUILD_TRACKER" \
    "$MAHTI_LOCK_ROOT"

log() {
    printf '[csc] %s\n' "$*"
}

warn() {
    printf '[csc][warn] %s\n' "$*" >&2
}

die() {
    printf '[csc][error] %s\n' "$*" >&2
    exit 1
}

acquire_lock() {
    local __resultvar="$1"
    local lock_name="$2"
    local lock_file="$MAHTI_LOCK_ROOT/${lock_name}.lock"
    local fd

    exec {fd}> "$lock_file"
    flock "$fd"
    printf -v "$__resultvar" '%s' "$fd"
}

release_lock() {
    local fd="${1:-}"

    if [[ -z "$fd" ]]; then
        return 0
    fi

    flock -u "$fd" || true
    eval "exec ${fd}>&-"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

ensure_module_command() {
    if command -v module >/dev/null 2>&1; then
        return 0
    fi

    if [[ -f /appl/profile/zz-csc-env.sh ]]; then
        local had_errexit=0
        local had_nounset=0

        if [[ $- == *e* ]]; then
            had_errexit=1
        fi
        if [[ $- == *u* ]]; then
            had_nounset=1
        fi

        set +e +u
        # shellcheck disable=SC1091
        source /appl/profile/zz-csc-env.sh >/dev/null 2>&1
        local source_status=$?
        if (( had_nounset )); then
            set -u
        fi
        if (( had_errexit )); then
            set -e
        fi

        if (( source_status == 0 )) && command -v module >/dev/null 2>&1; then
            return 0
        fi
    fi

    local modulecmd_path
    modulecmd_path="$(command -v modulecmd 2>/dev/null || true)"
    if [[ -z "$modulecmd_path" ]]; then
        return 1
    fi

    module() {
        eval "$("$modulecmd_path" bash "$@")"
    }
}

ensure_python310_plus() {
    # Some clusters (e.g. Roihu) ship only Python 3.9 system-wide with no
    # module system to opt into a newer one. Several TTS packages (qwen-tts,
    # faster-qwen3-tts) require >=3.10, so fetch a portable interpreter with
    # uv (no root needed) and put it first on PATH for the rest of this shell.
    local current_version major minor

    current_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 0.0)"
    major="${current_version%%.*}"
    minor="${current_version##*.}"
    if (( major > 3 || (major == 3 && minor >= 10) )); then
        return
    fi

    local arch uv_base uv_bin
    arch="$(uname -m)"
    # $HOME and $MAHTI_CACHE_ROOT are shared filesystems visible from both of
    # Roihu's node pools (x86_64 CPU, aarch64 GPU). A uv binary (or our shim
    # symlinks below) installed while running on one pool is a foreign-arch
    # ELF file on the other ("Exec format error"), so both must be
    # architecture-suffixed rather than living at a single shared path.
    uv_base="$HOME/.local/uv-$arch"
    uv_bin="$uv_base/bin/uv"
    if [[ ! -x "$uv_bin" ]]; then
        PYTHONUSERBASE="$uv_base" python3 -m pip install --user --quiet uv
    fi
    "$uv_bin" python install 3.11 >/dev/null

    local py311_bin shim_dir
    # `uv python find` prefers an active virtualenv's interpreter over its own
    # managed installations. If a venv created through our own shim is already
    # active (e.g. bootstrap_audiobook_video.sh ran earlier in this shell), it
    # would return that venv's bin/python3 -- which itself symlinks back to the
    # shim -- creating a symlink cycle. Unset VIRTUAL_ENV so this always
    # resolves the real standalone interpreter regardless of call order.
    py311_bin="$(env -u VIRTUAL_ENV "$uv_bin" python find 3.11)"
    shim_dir="$MAHTI_CACHE_ROOT/bin-$arch"
    mkdir -p "$shim_dir"
    ln -sf "$py311_bin" "$shim_dir/python3"
    ln -sf "$py311_bin" "$shim_dir/python"
    export PATH="$shim_dir:$PATH"
    hash -r
    log "no Python >=3.10 found; using uv-provisioned $(python3 --version) via $shim_dir"
}

pip_install_pytorch_no_module() {
    # Clusters without an Lmod module system (e.g. Roihu) have no prebuilt
    # PyTorch to inherit via --system-site-packages, so install directly.
    # This always runs against the uv-provisioned interpreter from
    # ensure_python310_plus, which is PEP 668 "externally managed" -- this is
    # a single-user HPC scratch install, not a shared system Python, so
    # override that.
    # Pin 2.9.x: unpinned upgrades landed 2.11 on Roihu and broke
    # qwen3_faster_clone EOS (runaway 327.68s / silent-garbage chunks).
    local want="2.9.1"
    if python -c "import torch; raise SystemExit(0 if torch.__version__.startswith('${want}') else 1)" >/dev/null 2>&1; then
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        # A visible NVIDIA GPU (e.g. Roihu's aarch64 Grace Hopper nodes);
        # PyPI's cu128 index ships CUDA-enabled wheels for aarch64 too.
        python -m pip install --break-system-packages --upgrade \
            "torch==${want}" "torchaudio==${want}" \
            --index-url https://download.pytorch.org/whl/cu128
    else
        # No GPU on this node (e.g. Roihu's CPU-only partitions, which share
        # this same cluster name) -- a plain CPU build is all that's usable.
        python -m pip install --break-system-packages --upgrade \
            "torch==${want}" "torchaudio==${want}"
    fi
}

load_pytorch_module() {
    local version="${1:-}"

    if ! ensure_module_command; then
        ensure_python310_plus
        pip_install_pytorch_no_module
        return
    fi

    module purge >/dev/null 2>&1 || true
    if [[ -n "$version" ]]; then
        module load "pytorch/$version"
    else
        module load pytorch
    fi

    local python_libdir
    python_libdir="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")' 2>/dev/null || true)"
    if [[ -n "$python_libdir" && ":${LD_LIBRARY_PATH:-}:" != *":$python_libdir:"* ]]; then
        export LD_LIBRARY_PATH="$python_libdir:${LD_LIBRARY_PATH:-}"
    fi

    local nvidia_libdirs
    nvidia_libdirs="$(
        python3 - <<'PY' 2>/dev/null
import site
from pathlib import Path

seen = set()
ordered = []
for base in site.getsitepackages():
    root = Path(base) / "nvidia"
    if not root.exists():
        continue
    for libdir in sorted(root.glob("*/lib")):
        if not libdir.is_dir():
            continue
        raw = str(libdir.resolve())
        if raw in seen:
            continue
        seen.add(raw)
        ordered.append(raw)
print(":".join(ordered))
PY
    )"
    if [[ -n "$nvidia_libdirs" ]]; then
        export LD_LIBRARY_PATH="$nvidia_libdirs:${LD_LIBRARY_PATH:-}"
    fi
}

ensure_python_venv() {
    local env_name="$1"
    local env_path="$MAHTI_ENV_ROOT/$env_name"
    local bootstrap_marker="$env_path/.bootstrap-complete"
    local current_python_version
    local existing_python_version
    local existing_python_target=""
    local recreate_env=0
    local lock_fd=""

    acquire_lock lock_fd "venv-$env_name"

    current_python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    if [[ -x "$env_path/bin/python" ]]; then
        existing_python_version="$($env_path/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo invalid)"
        if [[ "$existing_python_version" != "$current_python_version" ]]; then
            warn "recreating virtual environment $env_path (python $existing_python_version -> $current_python_version)"
            recreate_env=1
        fi
    fi

    if [[ -L "$env_path/bin/python3" ]]; then
        existing_python_target="$(readlink -f "$env_path/bin/python3" 2>/dev/null || true)"
        if [[ "$existing_python_target" == /usr/bin/python* ]]; then
            warn "recreating virtual environment $env_path to avoid unstable system python symlink ($existing_python_target)"
            recreate_env=1
        fi
    fi

    if [[ -f "$env_path/pyvenv.cfg" ]] && ! grep -q '^include-system-site-packages = true$' "$env_path/pyvenv.cfg"; then
        warn "recreating virtual environment $env_path to enable system site packages"
        recreate_env=1
    fi

    if (( recreate_env )); then
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

activate_python_venv() {
    local env_name="$1"
    local env_path="$MAHTI_ENV_ROOT/$env_name"

    [[ -f "$env_path/bin/activate" ]] || die "virtual environment does not exist: $env_path"
    # shellcheck disable=SC1090
    source "$env_path/bin/activate"
}

run_bootstrap_once() {
    local env_name="$1"
    local bootstrap_func="$2"
    local marker="$MAHTI_ENV_ROOT/$env_name/.bootstrap-complete"
    local lock_fd=""
    local status=0

    acquire_lock lock_fd "bootstrap-$env_name"
    if [[ -f "$marker" ]]; then
        release_lock "$lock_fd"
        return 0
    fi

    "$bootstrap_func" || status=$?
    if (( status == 0 )); then
        touch "$marker"
    fi
    release_lock "$lock_fd"
    return "$status"
}

download_repo_archive() {
    local repo_url="$1"
    local target_dir="$2"
    local ref="${3:-}"
    local parent_dir repo_name owner_name tmp_dir candidate archive_url archive_root

    if [[ "$repo_url" != https://github.com/* ]]; then
        return 1
    fi

    parent_dir="$(dirname "$target_dir")"
    repo_name="${repo_url##*/}"
    repo_name="${repo_name%.git}"
    owner_name="${repo_url%/*}"
    owner_name="${owner_name##*/}"
    tmp_dir="$parent_dir/.repo-download-${repo_name}-$$"

    rm -rf "$tmp_dir" "$target_dir"
    mkdir -p "$tmp_dir"

    local refs=()
    if [[ -n "$ref" ]]; then
        refs=(
            "https://github.com/$owner_name/$repo_name/archive/$ref.tar.gz::$repo_name-${ref##*/}"
            "https://github.com/$owner_name/$repo_name/archive/refs/tags/$ref.tar.gz::$repo_name-${ref##*/}"
            "https://github.com/$owner_name/$repo_name/archive/refs/heads/$ref.tar.gz::$repo_name-${ref##*/}"
        )
    else
        refs=(
            "https://github.com/$owner_name/$repo_name/archive/refs/heads/main.tar.gz::$repo_name-main"
            "https://github.com/$owner_name/$repo_name/archive/refs/heads/master.tar.gz::$repo_name-master"
        )
    fi

    for candidate in "${refs[@]}"; do
        archive_url="${candidate%%::*}"
        archive_root="${candidate##*::}"

        if curl -LfsS "$archive_url" | tar -xzf - -C "$tmp_dir"; then
            if [[ -d "$tmp_dir/$archive_root" ]]; then
                mv "$tmp_dir/$archive_root" "$target_dir"
                rm -rf "$tmp_dir"
                return 0
            fi
        fi

        rm -rf "$tmp_dir"
        mkdir -p "$tmp_dir"
    done

    rm -rf "$tmp_dir"
    return 1
}

ensure_repo() {
    local repo_url="$1"
    local target_dir="$2"
    local ref="${3:-}"

    if [[ ! -d "$target_dir/.git" ]]; then
        if [[ -d "$target_dir" ]] && find "$target_dir" -mindepth 1 -maxdepth 1 -print -quit >/dev/null 2>&1; then
            warn "keeping existing non-git source tree at $target_dir"
        elif command -v git >/dev/null 2>&1; then
            log "cloning $repo_url into $target_dir"
            git clone "$repo_url" "$target_dir"
        elif download_repo_archive "$repo_url" "$target_dir" "$ref"; then
            log "downloaded archive for $repo_url into $target_dir"
        else
            die "git is unavailable and archive fallback failed for $repo_url"
        fi
    fi

    if [[ ! -d "$target_dir/.git" ]]; then
        if [[ -n "$ref" ]]; then
            warn "archive-based source tree at $target_dir cannot checkout ref $ref; keeping extracted snapshot"
        fi
        return 0
    fi

    if [[ -n "$ref" ]]; then
        git -C "$target_dir" fetch --tags --depth=1 origin "$ref"
        git -C "$target_dir" checkout "$ref"
    else
        git -C "$target_dir" pull --ff-only || warn "unable to fast-forward $target_dir; keeping existing checkout"
    fi
}

resolve_python_command() {
    if command -v python >/dev/null 2>&1; then
        printf '%s\n' python
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        printf '%s\n' python3
        return 0
    fi

    die "python interpreter not found"
}

python_hf_download() {
    local repo_id="$1"
    local local_dir="$2"
    local allow_patterns="${3:-}"
    local python_cmd

    python_cmd="$(resolve_python_command)"

    "$python_cmd" - "$repo_id" "$local_dir" "$allow_patterns" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = Path(sys.argv[2])
raw_patterns = sys.argv[3].strip()
patterns = [part.strip() for part in raw_patterns.split(",") if part.strip()] or None

local_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    allow_patterns=patterns,
)
PY
}

download_zip_to_dir() {
    local url="$1"
    local target_dir="$2"
    local marker_path="$3"
    local python_cmd

    python_cmd="$(resolve_python_command)"

    "$python_cmd" - "$url" "$target_dir" "$marker_path" <<'PY'
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

url = sys.argv[1]
target_dir = Path(sys.argv[2])
marker_path = Path(sys.argv[3])

if marker_path.exists():
    sys.exit(0)

target_dir.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as tmpdir:
    archive_path = Path(tmpdir) / "archive.zip"
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)
PY
}

download_tar_gz_to_dir() {
    local url="$1"
    local target_dir="$2"
    local marker_path="$3"
    local python_cmd

    python_cmd="$(resolve_python_command)"

    "$python_cmd" - "$url" "$target_dir" "$marker_path" <<'PY'
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request

url = sys.argv[1]
target_dir = Path(sys.argv[2])
marker_path = Path(sys.argv[3])

if marker_path.exists():
    sys.exit(0)

target_dir.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as tmpdir:
    archive_path = Path(tmpdir) / "archive.tar.gz"
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(target_dir)
PY
}

write_text_file() {
    local file_path="$1"
    local content="$2"
    mkdir -p "$(dirname -- "$file_path")"
    printf '%s' "$content" > "$file_path"
}
