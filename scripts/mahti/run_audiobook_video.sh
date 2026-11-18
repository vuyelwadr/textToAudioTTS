#!/usr/bin/env bash
#
# Run audiobook video generation inside the dedicated CSC video environment.
#
# Example standalone usage:
#   sbatch --account project_2010549 --partition gpusmall \
#     --gres gpu:a100:1,nvme:900 --cpus-per-task 16 --time 02:00:00 \
#     scripts/mahti/run_audiobook_video.sh \
#       --summary /scratch/project_2010549/$USER/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-full/audiobook_summary.json \
#       --model qwen3_faster_clone \
#       --show-active-line \
#       --active-line-timing asr \
#       --active-line-asr-device cuda
#

set -euo pipefail

resolve_script_dir() {
    local source_dir
    local job_command
    local command_dir

    source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    if [[ -f "$source_dir/common.sh" ]]; then
        printf '%s\n' "$source_dir"
        return 0
    fi

    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
        printf '%s\n' "${SLURM_SUBMIT_DIR}/scripts/mahti"
        return 0
    fi

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        job_command="$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null | sed -n 's/.* Command=\([^ ]*\).*/\1/p')"
        if [[ -n "$job_command" && -f "$job_command" ]]; then
            command_dir=$(cd -- "$(dirname -- "$job_command")" && pwd)
            if [[ -f "$command_dir/common.sh" ]]; then
                printf '%s\n' "$command_dir"
                return 0
            fi
        fi
    fi

    printf '%s\n' "$source_dir"
}

SCRIPT_DIR="$(resolve_script_dir)"
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
load_pytorch_module 2.9

# shellcheck disable=SC1091
source "$SCRIPT_DIR/bootstrap_audiobook_video.sh"

if ensure_module_command; then
    module load ffmpeg >/dev/null 2>&1 || true
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    if ! python -c 'import imageio_ffmpeg' >/dev/null 2>&1; then
        python -m pip install imageio-ffmpeg
    fi

    python - <<'PY'
import os
import shlex
from pathlib import Path

import imageio_ffmpeg

venv_path = os.environ.get("VIRTUAL_ENV")
if not venv_path:
    raise SystemExit("VIRTUAL_ENV is not set; cannot install ffmpeg wrapper")

wrapper_path = Path(venv_path) / "bin" / "ffmpeg"
ffmpeg_binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
wrapper_path.write_text(
    "#!/usr/bin/env bash\n"
    f"exec {shlex.quote(str(ffmpeg_binary))} \"$@\"\n",
    encoding="utf-8",
)
wrapper_path.chmod(0o755)
print(f"[csc] runtime ffmpeg wrapper: {wrapper_path}")
PY
fi

require_command ffmpeg

cd "$REPO_ROOT"
python "$SCRIPT_DIR/run_audiobook_video.py" "$@"
