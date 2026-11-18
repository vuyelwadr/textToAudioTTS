#!/usr/bin/env bash
#SBATCH --job-name=qwen-ref-sweep
#SBATCH --account=project_2010549
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:4,nvme:900
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/qwen3_half_yellow_sun_clone_refs_%j.log

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mahti"
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
load_pytorch_module 2.9

# shellcheck disable=SC1091
source "$SCRIPT_DIR/bootstrap_faster_qwen3_tts.sh"

MODE="${MODE:-full}"
WORK_BASE="${WORK_BASE:-$MAHTI_WORK_ROOT/generated/audiobooks-half_of_a_yellow_sun-full}"
CHUNK_DIR="${CHUNK_DIR:-$WORK_BASE/half_of_a_yellow_sun/chunks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAHTI_WORK_ROOT/generated/audiobooks-half_of_a_yellow_sun-full-qwen-ref-sweep}"
SHORT_REFERENCE_AUDIO="${SHORT_REFERENCE_AUDIO:-$MAHTI_REPO_ROOT/assets/audio_samples/brothersKarmatzov.clone_ref_15s.wav}"
SHORT_REFERENCE_TEXT_FILE="${SHORT_REFERENCE_TEXT_FILE:-$MAHTI_REPO_ROOT/assets/audio_samples/brothersKarmatzov.clone_ref_15s.transcript.txt}"
LONG_REFERENCE_AUDIO="${LONG_REFERENCE_AUDIO:-$MAHTI_REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.wav}"
LONG_REFERENCE_TEXT_FILE="${LONG_REFERENCE_TEXT_FILE:-$MAHTI_REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.transcript.txt}"
REFERENCES="${REFERENCES:-short long}"
WORKERS="${WORKERS:-auto}"
FORCE="${FORCE:-0}"
MAX_CHUNKS="${MAX_CHUNKS:-}"

args=(
    --chunk-dir "$CHUNK_DIR"
    --output-root "$OUTPUT_ROOT"
    --short-reference-audio "$SHORT_REFERENCE_AUDIO"
    --short-reference-text-file "$SHORT_REFERENCE_TEXT_FILE"
    --long-reference-audio "$LONG_REFERENCE_AUDIO"
    --long-reference-text-file "$LONG_REFERENCE_TEXT_FILE"
    --references $REFERENCES
    --workers "$WORKERS"
)

if [[ "$FORCE" == "1" ]]; then
    args+=(--force)
fi

case "$MODE" in
    smoke)
        args+=(--max-chunks "${MAX_CHUNKS:-2}")
        ;;
    full)
        if [[ -n "$MAX_CHUNKS" ]]; then
            args+=(--max-chunks "$MAX_CHUNKS")
        fi
        ;;
    *)
        die "unknown MODE: $MODE"
        ;;
esac

log "Qwen3 Half of a Yellow Sun clone reference sweep"
log "chunk dir: $CHUNK_DIR"
log "output root: $OUTPUT_ROOT"
log "references: $REFERENCES"
log "workers: $WORKERS"

python -u "$SCRIPT_DIR/run_qwen3_clone_reference_sweep.py" "${args[@]}"
