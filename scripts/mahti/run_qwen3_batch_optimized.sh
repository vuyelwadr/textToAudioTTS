#!/bin/bash
#SBATCH --job-name=qwen3batchopt
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/local/bin:$PATH"

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
source scripts/mahti/bootstrap_faster_qwen3_tts.sh 2>&1 | tail -3
source scripts/mahti/common.sh
load_pytorch_module 2.9
source "$MAHTI_ENV_ROOT/faster-qwen3-tts/bin/activate"

AUDIOBOOK_WORK_DIR="$MAHTI_WORK_ROOT/generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun"
INPUT_DIR="$AUDIOBOOK_WORK_DIR/chunks"
OUTPUT_DIR="$AUDIOBOOK_WORK_DIR/qwen3_faster_clone/chunks"
LOG_DIR="$AUDIOBOOK_WORK_DIR/qwen3_faster_clone/logs"
QWEN_WORKERS="${QWEN_WORKERS:-1}"
REFERENCE_AUDIO="$MAHTI_WORK_ROOT/outputs/video_downloads/haCxgJSFmFc/haCxgJSFmFc_reference_8s.wav"
REFERENCE_TEXT="$MAHTI_WORK_ROOT/outputs/video_downloads/haCxgJSFmFc/haCxgJSFmFc_reference_8s.transcript.auto.txt"

qwen_args=(
    --input-dir "$INPUT_DIR"
    --output-dir "$OUTPUT_DIR"
    --log-dir "$LOG_DIR"
    --prefix "chunk_"
    --reference-audio "$REFERENCE_AUDIO"
    --generation-mode "clone"
    --no-do-sample
    --non-streaming-mode
    --language "English"
    --max-new-tokens 2048
    --attn-implementation "sdpa"
    --temperature 0.88
    --top-k 55
    --top-p 1.0
    --repetition-penalty 1.05
    --workers "$QWEN_WORKERS"
    --resume
)

if [[ -f "$REFERENCE_TEXT" ]]; then
    qwen_args+=(--reference-text-file "$REFERENCE_TEXT")
fi

python scripts/mahti/run_qwen3_batch_optimized.py "${qwen_args[@]}"

echo "=== QWEN3 BATCH PROCESSING DONE ==="
