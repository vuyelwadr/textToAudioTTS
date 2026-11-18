#!/bin/bash
#SBATCH --job-name=vibebatchopt
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mahti"
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export REPO_ROOT
MAHTI_WORK_ROOT="$REPO_ROOT"
MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
MAHTI_EXTERNAL_ROOT="$REPO_ROOT/external"
export MAHTI_EXTERNAL_ROOT
MAHTI_CACHE_ROOT="$REPO_ROOT/cache"
export REPO_ROOT MAHTI_WORK_ROOT MAHTI_ENV_ROOT MAHTI_EXTERNAL_ROOT MAHTI_CACHE_ROOT

cd "$REPO_ROOT"
source scripts/mahti/bootstrap_vibevoice.sh 2>&1 | tail -3
source scripts/mahti/common.sh
load_pytorch_module 2.9
source "$MAHTI_ENV_ROOT/vibevoice/bin/activate"

AUDIOBOOK_WORK_DIR="$MAHTI_WORK_ROOT/generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun"
INPUT_DIR="$AUDIOBOOK_WORK_DIR/chunks"
OUTPUT_DIR="$AUDIOBOOK_WORK_DIR/vibevoice_realtime/chunks"
LOG_DIR="$AUDIOBOOK_WORK_DIR/vibevoice_realtime/logs"
VIBEVOICE_WORKERS="${VIBEVOICE_WORKERS:-auto}"

python scripts/mahti/run_vibevoice_batch_optimized.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --prefix "chunk_" \
    --speaker "en-Mike_man" \
    --cfg-scale 1.7 \
    --ddpm-steps 8 \
    --workers "$VIBEVOICE_WORKERS" \
    --resume

echo "=== VIBEVOICE BATCH PROCESSING DONE ==="
