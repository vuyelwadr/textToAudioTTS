#!/bin/bash
#SBATCH --job-name=prom-moss-full
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1,nvme:900
#SBATCH --mem=80G
#SBATCH --output=%x-%A_%a.out

set -euo pipefail

export CSC_PROJECT="${CSC_PROJECT:-${SLURM_JOB_ACCOUNT:-project_2010549}}"
export MAHTI_PROJECT_ROOT="${MAHTI_PROJECT_ROOT:-/projappl/${CSC_PROJECT}/textToAudioTTS-mahti}"
export CSC_HEAVY_PROJECT_ROOT="$MAHTI_PROJECT_ROOT"
export MAHTI_WORK_ROOT="${MAHTI_WORK_ROOT:-/scratch/${CSC_PROJECT}/${USER}/textToAudioTTS-mahti}"
export CSC_HEAVY_WORK_ROOT="$MAHTI_WORK_ROOT"

if [[ -n "${LOCAL_SCRATCH:-}" ]]; then
    export CSC_PREFER_LOCAL_SCRATCH=1
    export MAHTI_LOCAL_ROOT="${MAHTI_LOCAL_ROOT:-$LOCAL_SCRATCH/textToAudioTTS-mahti}"
    export MAHTI_ENV_ROOT="${MAHTI_ENV_ROOT:-$MAHTI_LOCAL_ROOT/venvs}"
    export MAHTI_MODEL_ROOT="${MAHTI_MODEL_ROOT:-$MAHTI_WORK_ROOT/models}"
    export MAHTI_CACHE_ROOT="${MAHTI_CACHE_ROOT:-$MAHTI_LOCAL_ROOT/cache}"
else
    export MAHTI_ENV_ROOT="${MAHTI_ENV_ROOT:-$MAHTI_PROJECT_ROOT/venvs}"
    export MAHTI_MODEL_ROOT="${MAHTI_MODEL_ROOT:-$MAHTI_WORK_ROOT/models}"
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
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

bash "$SCRIPT_DIR/bootstrap_moss_tts.sh"
activate_python_venv "moss-tts"

if [[ "${PREFETCH_ONLY:-0}" == "1" ]]; then
    echo "[moss-full-slurm] prefetch/bootstrap completed"
    exit 0
fi

CHUNK_DIR="${CHUNK_DIR:-$REPO_ROOT/generated/prometheus_benchmark_smoke_20260430/PrometheusAtlas/chunks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/generated/audiobooks-prometheus_atlas-moss_tts_clone_full_20260501/PrometheusAtlas}"
INPUT_TEXT="${INPUT_TEXT:-$REPO_ROOT/library/PrometheusAtlas/PrometheusAtlas.txt}"
REFERENCE_AUDIO="${REFERENCE_AUDIO:-$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.wav}"
CHUNKS_PER_TASK="${CHUNKS_PER_TASK:-20}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_INDEX=$((TASK_ID * CHUNKS_PER_TASK))

extra_args=()
if [[ "${FORCE:-0}" == "1" ]]; then
    extra_args+=(--force)
fi

echo "[moss-full-slurm] array_task=$TASK_ID start_index=$START_INDEX chunks_per_task=$CHUNKS_PER_TASK"
echo "[moss-full-slurm] output_root=$OUTPUT_ROOT"

srun python "$SCRIPT_DIR/run_moss_prometheus_full.py" \
    --chunk-dir "$CHUNK_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --input-text "$INPUT_TEXT" \
    --reference-audio "$REFERENCE_AUDIO" \
    --start-index "$START_INDEX" \
    --max-chunks "$CHUNKS_PER_TASK" \
    --skip-join \
    "${extra_args[@]}"
