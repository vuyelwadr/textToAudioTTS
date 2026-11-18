#!/usr/bin/env bash
#
# Tempted by Danger audiobook + CPU video runner.
#
# Usage:
#   INPUT_FILE=$PWD/library/temptedByDanger/temptedByDanger.txt \
#   VIDEO_RENDER_MODE=submit_cpu sbatch --account <project> \
#     --partition gpusmall --gres gpu:a100:2,nvme:900 --cpus-per-task 32 \
#     scripts/mahti/run_tempted_by_danger.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mahti"
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
load_pytorch_module 2.9

BOOK_DIR="$REPO_ROOT/library/temptedByDanger"
INPUT_FILE="${INPUT_FILE:-$BOOK_DIR/temptedByDanger.txt}"
SOURCE_EPUB="${SOURCE_EPUB:-$BOOK_DIR/_OceanofPDF.com_Tempted_by_danger_-_Endiya_Carter.epub}"
OUTPUT_DIR="${OUTPUT_DIR:-$MAHTI_WORK_ROOT/generated/audiobooks-tbd-full}"
WORK_DIR="${WORK_DIR:-$OUTPUT_DIR}"
BOOK_STEM="${BOOK_STEM:-tbd}"
BOOK_SLUG="${BOOK_SLUG:-tbd}"
RUN_LABEL="${RUN_LABEL:-full}"
CHUNK_SIZE="${CHUNK_SIZE:-2200}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-200}"
CHUNK_WORKERS="${CHUNK_WORKERS:-16}"
TARGET_AUDIO_SECONDS="${TARGET_AUDIO_SECONDS:-0}"
FORCE="${FORCE:-0}"
AUDIT_CHUNKS="${AUDIT_CHUNKS:-1}"
AUDIT_ASR_DEVICE="${AUDIT_ASR_DEVICE:-cuda}"
AUDIT_ASR_MODEL="${AUDIT_ASR_MODEL:-small.en}"
AUDIT_WORKERS="${AUDIT_WORKERS:-auto}"
GENERATE_VIDEO="${GENERATE_VIDEO:-1}"
VIDEO_RENDER_MODE="${VIDEO_RENDER_MODE:-submit_cpu}"
VIDEO_ASR_DEVICE="${VIDEO_ASR_DEVICE:-cuda}"
VIDEO_ASR_MODEL="${VIDEO_ASR_MODEL:-small.en}"
VIDEO_ASR_WORKERS="${VIDEO_ASR_WORKERS:-auto}"
VIDEO_ASR_CACHE_DIR="${VIDEO_ASR_CACHE_DIR:-$WORK_DIR/$BOOK_STEM/qwen3_faster_clone/chunk_audit/asr_cache}"
VIDEO_CODEC="${VIDEO_CODEC:-libx264}"
VIDEO_PRESET="${VIDEO_PRESET:-veryfast}"
VIDEO_CRF="${VIDEO_CRF:-18}"
CHAPTER_END_HOLD="${CHAPTER_END_HOLD:-3.0}"
CHAPTER_TRANSITION="${CHAPTER_TRANSITION:-0.6}"
CHAPTER_START_HOLD="${CHAPTER_START_HOLD:-2.0}"
CHAPTER_AUDIO_OUTPUT="${CHAPTER_AUDIO_OUTPUT:-$OUTPUT_DIR/${BOOK_STEM}_qwen3_faster_clone_chapter_gaps.wav}"
VIDEO_SUBMIT_CLUSTER="${VIDEO_SUBMIT_CLUSTER:-mahti}"
VIDEO_SUBMIT_MODE="${VIDEO_SUBMIT_MODE:-full}"
VIDEO_SUBMIT_RESOURCE_MODE="${VIDEO_SUBMIT_RESOURCE_MODE:-cpu}"
VIDEO_SUBMIT_ACCOUNT="${VIDEO_SUBMIT_ACCOUNT:-${CSC_PROJECT:-project_2010549}}"
VIDEO_SUBMIT_PARTITION="${VIDEO_SUBMIT_PARTITION:-}"
VIDEO_SUBMIT_TIME="${VIDEO_SUBMIT_TIME:-}"
VIDEO_SUBMIT_CPUS="${VIDEO_SUBMIT_CPUS:-}"
VIDEO_SUBMIT_GRES="${VIDEO_SUBMIT_GRES:-}"
VIDEO_SUBMIT_MEM="${VIDEO_SUBMIT_MEM:-}"
VIDEO_SUBMIT_ASR_DEVICE="${VIDEO_SUBMIT_ASR_DEVICE:-cpu}"
VIDEO_SUBMIT_JOB_RECORD="${VIDEO_SUBMIT_JOB_RECORD:-$OUTPUT_DIR/${BOOK_STEM}_qwen3_faster_clone_video_job.txt}"

if [[ ! -f "$INPUT_FILE" ]]; then
    if [[ -f "$SOURCE_EPUB" ]]; then
        python "$SCRIPT_DIR/../extract_tempted_by_danger_epub.py" \
            --epub "$SOURCE_EPUB" \
            --output "$INPUT_FILE" \
            --audit-output "$BOOK_DIR/temptedByDanger_text_audit.json"
    else
        echo "Missing input text and source EPUB: $INPUT_FILE / $SOURCE_EPUB" >&2
        exit 2
    fi
fi

args=(
    --input "$INPUT_FILE"
    --models qwen3_faster_clone
    --output-dir "$OUTPUT_DIR"
    --work-dir "$WORK_DIR"
    --chunk-size "$CHUNK_SIZE"
    --min-chunk-size "$MIN_CHUNK_SIZE"
    --chunk-workers "$CHUNK_WORKERS"
    --reference-audio "$REPO_ROOT/assets/audio_samples/brothersKarmatzov.clone_ref_15s.wav"
    --reference-text-file "$REPO_ROOT/assets/audio_samples/brothersKarmatzov.clone_ref_15s.transcript.txt"
)

if [[ "$AUDIT_CHUNKS" == "1" ]]; then
    args+=(
        --audit-chunks
        --audit-asr-model "$AUDIT_ASR_MODEL"
        --audit-asr-device "$AUDIT_ASR_DEVICE"
        --audit-workers "$AUDIT_WORKERS"
    )
else
    args+=(--no-audit-chunks)
fi

if [[ "$FORCE" == "1" ]]; then
    args+=(--force)
fi

if [[ "$TARGET_AUDIO_SECONDS" == "0" ]]; then
    export TARGET_AUDIO_SECONDS=0
else
    export TARGET_AUDIO_SECONDS
fi
export INPUT_FILE BOOK_STEM BOOK_SLUG RUN_LABEL OUTPUT_DIR WORK_DIR GENERATE_VIDEO VIDEO_RENDER_MODE
export VIDEO_ASR_DEVICE VIDEO_ASR_MODEL VIDEO_ASR_WORKERS VIDEO_ASR_CACHE_DIR
export VIDEO_CODEC VIDEO_PRESET VIDEO_CRF
export CHAPTER_END_HOLD CHAPTER_TRANSITION CHAPTER_START_HOLD CHAPTER_AUDIO_OUTPUT
export VIDEO_SUBMIT_CLUSTER VIDEO_SUBMIT_MODE VIDEO_SUBMIT_RESOURCE_MODE VIDEO_SUBMIT_ACCOUNT
export VIDEO_SUBMIT_PARTITION VIDEO_SUBMIT_TIME VIDEO_SUBMIT_CPUS VIDEO_SUBMIT_GRES VIDEO_SUBMIT_MEM
export VIDEO_SUBMIT_ASR_DEVICE VIDEO_SUBMIT_JOB_RECORD

bash "$SCRIPT_DIR/run_qwen_audiobook_video.sh"
