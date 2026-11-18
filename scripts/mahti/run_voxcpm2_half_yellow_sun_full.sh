#!/usr/bin/env bash
#SBATCH --job-name=vox-halfys-full
#SBATCH --account=project_2010549
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:4,nvme:900
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --output=logs/voxcpm2_half_yellow_sun_full_%j.log

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
source "$SCRIPT_DIR/bootstrap_voxcpm2.sh"

MODE="${MODE:-full}"
VARIANTS="${VARIANTS:-clone_default_clone design_british_baritone clone_brothers_60s}"
WORK_BASE="${WORK_BASE:-$MAHTI_WORK_ROOT/generated/audiobooks-half_of_a_yellow_sun-full}"
CHUNK_DIR="${CHUNK_DIR:-$WORK_BASE/half_of_a_yellow_sun/chunks}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$CHUNK_DIR/half_of_a_yellow_sun_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAHTI_WORK_ROOT/generated/audiobooks-half_of_a_yellow_sun-full-voxcpm2}"
REFERENCE_AUDIO="${REFERENCE_AUDIO:-$MAHTI_REPO_ROOT/assets/audio_samples/brothersKarmatzov.clone_ref_15s.wav}"
REFERENCE_TEXT_FILE="${REFERENCE_TEXT_FILE:-$MAHTI_REPO_ROOT/assets/audio_samples/brothersKarmatzov.clone_ref_15s.transcript.txt}"
LONG_REFERENCE_AUDIO="${LONG_REFERENCE_AUDIO:-$MAHTI_REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.wav}"
LONG_REFERENCE_TEXT_FILE="${LONG_REFERENCE_TEXT_FILE:-$MAHTI_REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.transcript.txt}"
WORKERS="${WORKERS:-auto}"
FORCE="${FORCE:-0}"
MAX_CHUNKS="${MAX_CHUNKS:-}"

args=(
    --chunk-dir "$CHUNK_DIR"
    --output-root "$OUTPUT_ROOT"
    --source-manifest "$SOURCE_MANIFEST"
    --reference-audio "$REFERENCE_AUDIO"
    --reference-text-file "$REFERENCE_TEXT_FILE"
    --long-reference-audio "$LONG_REFERENCE_AUDIO"
    --long-reference-text-file "$LONG_REFERENCE_TEXT_FILE"
    --workers "$WORKERS"
    --variants $VARIANTS
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

log "VoxCPM2 Half of a Yellow Sun run"
log "chunk dir: $CHUNK_DIR"
log "output root: $OUTPUT_ROOT"
log "variants: $VARIANTS"
log "workers: $WORKERS"

python -u "$SCRIPT_DIR/run_voxcpm2_full_audiobook.py" "${args[@]}"
