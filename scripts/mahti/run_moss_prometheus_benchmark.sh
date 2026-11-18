#!/bin/bash
#SBATCH --job-name=moss-prom-bench
#SBATCH --partition=gputest
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1,nvme:900
#SBATCH --mem=24G
#SBATCH --output=%x-%j.out

set -euo pipefail

export CSC_PROJECT="${CSC_PROJECT:-${SLURM_JOB_ACCOUNT:-project_2010549}}"
export MAHTI_PROJECT_ROOT="${MAHTI_PROJECT_ROOT:-/projappl/${CSC_PROJECT}/textToAudioTTS-mahti}"
export CSC_HEAVY_PROJECT_ROOT="$MAHTI_PROJECT_ROOT"

if [[ -n "${LOCAL_SCRATCH:-}" ]]; then
    export CSC_PREFER_LOCAL_SCRATCH=1
    export MAHTI_LOCAL_ROOT="${MAHTI_LOCAL_ROOT:-$LOCAL_SCRATCH/textToAudioTTS-mahti}"
    export MAHTI_ENV_ROOT="${MAHTI_ENV_ROOT:-$MAHTI_LOCAL_ROOT/venvs}"
    export MAHTI_MODEL_ROOT="${MAHTI_MODEL_ROOT:-$MAHTI_LOCAL_ROOT/models}"
    export MAHTI_CACHE_ROOT="${MAHTI_CACHE_ROOT:-$MAHTI_LOCAL_ROOT/cache}"
else
    export MAHTI_ENV_ROOT="${MAHTI_ENV_ROOT:-$MAHTI_PROJECT_ROOT/venvs}"
    export MAHTI_MODEL_ROOT="${MAHTI_MODEL_ROOT:-$MAHTI_PROJECT_ROOT/models}"
    export MAHTI_CACHE_ROOT="${MAHTI_CACHE_ROOT:-$MAHTI_PROJECT_ROOT/cache}"
fi

export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
export CSC_MODEL_ROOT="$MAHTI_MODEL_ROOT"
export CSC_CACHE_ROOT="$MAHTI_CACHE_ROOT"

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mahti"
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bash "$SCRIPT_DIR/bootstrap_moss_tts.sh"
activate_python_venv "moss-tts"

srun python "$SCRIPT_DIR/run_moss_prometheus_benchmark.py" "$@"
