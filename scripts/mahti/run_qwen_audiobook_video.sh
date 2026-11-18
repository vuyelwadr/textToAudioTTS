#!/usr/bin/env bash
#
# Generic Qwen3 audiobook + active-line video runner for Mahti.
#
# Required:
#   INPUT_FILE=/scratch/project_2010549/$USER/textToAudioTTS-mahti/library/Book/Book.txt
#
# Useful controls:
#   TARGET_AUDIO_SECONDS=10800   # Approximate output duration; 0 renders the whole book.
#   MAX_CHUNKS=42                # Overrides TARGET_AUDIO_SECONDS when set.
#   FORCE=1                      # Regenerate existing chunks/audio/video.
#   GENERATE_VIDEO=1             # Render or submit active-line MP4 after audio succeeds.
#   VIDEO_RENDER_MODE=submit_cpu # submit_cpu | inline
#   RUN_LABEL=qwen3_3h           # Output folder suffix.
#
# Example:
#   INPUT_FILE=$PWD/library/PrometheusAtlas/PrometheusAtlas.txt \
#   TARGET_AUDIO_SECONDS=10800 FORCE=1 GENERATE_VIDEO=1 VIDEO_RENDER_MODE=submit_cpu \
#   sbatch --account project_2010549 --partition gpusmall \
#     --gres gpu:a100:1,nvme:900 --cpus-per-task 16 --time 10:00:00 \
#     scripts/mahti/run_qwen_audiobook_video.sh

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

export MAHTI_ENV_ROOT="${MAHTI_ENV_ROOT:-$MAHTI_WORK_ROOT/venvs}"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
export MAHTI_EXTERNAL_ROOT="${MAHTI_EXTERNAL_ROOT:-$MAHTI_WORK_ROOT/external}"
export MAHTI_MODEL_ROOT="${MAHTI_MODEL_ROOT:-$MAHTI_WORK_ROOT/models}"
export MAHTI_CACHE_ROOT="${MAHTI_CACHE_ROOT:-$MAHTI_WORK_ROOT/cache}"

INPUT_FILE="${INPUT_FILE:-}"
if [[ -z "$INPUT_FILE" ]]; then
    echo "INPUT_FILE is required" >&2
    exit 2
fi
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "INPUT_FILE not found: $INPUT_FILE" >&2
    exit 2
fi

MODEL="qwen3_faster_clone"
BOOK_STEM="${BOOK_STEM:-$(basename "${INPUT_FILE%.*}")}"
BOOK_SLUG="${BOOK_SLUG:-$(printf '%s' "$BOOK_STEM" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')}"
RUN_LABEL="${RUN_LABEL:-qwen3_reference_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$MAHTI_WORK_ROOT/generated/audiobooks-${BOOK_SLUG}-${RUN_LABEL}}"
WORK_DIR="${WORK_DIR:-$OUTPUT_DIR}"
CHUNK_SIZE="${CHUNK_SIZE:-2200}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-200}"
CHUNK_WORKERS="${CHUNK_WORKERS:-16}"
TARGET_AUDIO_SECONDS="${TARGET_AUDIO_SECONDS:-10800}"
QWEN_WORDS_PER_SECOND="${QWEN_WORDS_PER_SECOND:-4.62}"
MAX_CHUNKS="${MAX_CHUNKS:-}"
FORCE="${FORCE:-0}"
AUDIT_CHUNKS="${AUDIT_CHUNKS:-1}"
# PyPI's ctranslate2 wheel for aarch64 (Roihu's GH200 nodes) has no CUDA
# build, unlike the x86_64 wheels Mahti/Puhti use. make_autoscroll_video.py's
# _load_whisper_model() detects that specific failure and transparently falls
# back to a PyTorch-native Whisper (still on GPU), so "cuda" stays the
# default here on every cluster.
AUDIT_ASR_DEVICE="${AUDIT_ASR_DEVICE:-cuda}"
AUDIT_ASR_MODEL="${AUDIT_ASR_MODEL:-small.en}"
AUDIT_WORKERS="${AUDIT_WORKERS:-auto}"
GENERATE_VIDEO="${GENERATE_VIDEO:-1}"
VIDEO_RENDER_MODE="${VIDEO_RENDER_MODE:-submit_cpu}"
VIDEO_ASR_DEVICE="${VIDEO_ASR_DEVICE:-cuda}"
VIDEO_ASR_MODEL="${VIDEO_ASR_MODEL:-small.en}"
VIDEO_ASR_WORKERS="${VIDEO_ASR_WORKERS:-auto}"
VIDEO_ASR_CACHE_DIR="${VIDEO_ASR_CACHE_DIR:-$WORK_DIR/$BOOK_STEM/$MODEL/chunk_audit/asr_cache}"
VIDEO_CODEC="${VIDEO_CODEC:-libx264}"
VIDEO_PRESET="${VIDEO_PRESET:-veryfast}"
VIDEO_CRF="${VIDEO_CRF:-18}"
CHAPTER_END_HOLD="${CHAPTER_END_HOLD:-3.0}"
CHAPTER_TRANSITION="${CHAPTER_TRANSITION:-0.6}"
CHAPTER_START_HOLD="${CHAPTER_START_HOLD:-2.0}"
CHAPTER_AUDIO_OUTPUT="${CHAPTER_AUDIO_OUTPUT:-$OUTPUT_DIR/${BOOK_STEM}_${MODEL}_chapter_gaps.wav}"
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
VIDEO_SUBMIT_JOB_RECORD="${VIDEO_SUBMIT_JOB_RECORD:-$OUTPUT_DIR/${BOOK_STEM}_${MODEL}_video_job.txt}"
REFERENCE_AUDIO="${REFERENCE_AUDIO:-$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.wav}"
REFERENCE_TEXT_FILE="${REFERENCE_TEXT_FILE:-$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.transcript.txt}"
QWEN_SAFE_MAX_CHUNK_SIZE="${QWEN_SAFE_MAX_CHUNK_SIZE:-2600}"
ALLOW_LARGE_QWEN_CHUNKS="${ALLOW_LARGE_QWEN_CHUNKS:-0}"

if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]]; then
    echo "CHUNK_SIZE must be a positive integer, got: $CHUNK_SIZE" >&2
    exit 2
fi
if ! [[ "$MIN_CHUNK_SIZE" =~ ^[0-9]+$ ]]; then
    echo "MIN_CHUNK_SIZE must be a positive integer, got: $MIN_CHUNK_SIZE" >&2
    exit 2
fi
if ! [[ "$QWEN_SAFE_MAX_CHUNK_SIZE" =~ ^[0-9]+$ ]]; then
    echo "QWEN_SAFE_MAX_CHUNK_SIZE must be a positive integer, got: $QWEN_SAFE_MAX_CHUNK_SIZE" >&2
    exit 2
fi
if [[ "$ALLOW_LARGE_QWEN_CHUNKS" != "1" && "$CHUNK_SIZE" -gt "$QWEN_SAFE_MAX_CHUNK_SIZE" ]]; then
    echo "Refusing CHUNK_SIZE=$CHUNK_SIZE for Qwen full-book generation." >&2
    echo "Use CHUNK_SIZE<=${QWEN_SAFE_MAX_CHUNK_SIZE}; larger chunks can hit Qwen's fixed audio-token ceiling and produce truncated_current_chunk audit failures." >&2
    echo "Set ALLOW_LARGE_QWEN_CHUNKS=1 only for a measured experiment with chunk audit enabled." >&2
    exit 2
fi

if [[ ! -f "$REFERENCE_AUDIO" ]]; then
    echo "Reference audio not found: $REFERENCE_AUDIO" >&2
    exit 2
fi
if [[ ! -f "$REFERENCE_TEXT_FILE" ]]; then
    echo "Reference transcript not found: $REFERENCE_TEXT_FILE" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$WORK_DIR"

base_args=(
    --input "$INPUT_FILE"
    --models "$MODEL"
    --output-dir "$OUTPUT_DIR"
    --work-dir "$WORK_DIR"
    --reference-audio "$REFERENCE_AUDIO"
    --reference-text-file "$REFERENCE_TEXT_FILE"
    --chunk-size "$CHUNK_SIZE"
    --min-chunk-size "$MIN_CHUNK_SIZE"
    --chunk-workers "$CHUNK_WORKERS"
)

if [[ "$AUDIT_CHUNKS" == "1" ]]; then
    base_args+=(
        --audit-chunks
        --audit-asr-model "$AUDIT_ASR_MODEL"
        --audit-asr-device "$AUDIT_ASR_DEVICE"
        --audit-workers "$AUDIT_WORKERS"
    )
else
    base_args+=(--no-audit-chunks)
fi

if [[ -z "$MAX_CHUNKS" && "$TARGET_AUDIO_SECONDS" != "0" ]]; then
    python "$REPO_ROOT/scripts/build_chunked_audiobook.py" "${base_args[@]}" --dry-run --no-audit-chunks
    manifest_path="$WORK_DIR/$BOOK_STEM/chunks/${BOOK_STEM}_manifest.json"
    MAX_CHUNKS="$(
        python - "$manifest_path" "$TARGET_AUDIO_SECONDS" "$QWEN_WORDS_PER_SECOND" <<'PY'
import json
import math
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target_seconds = float(sys.argv[2])
words_per_second = float(sys.argv[3])
target_words = max(1, int(math.ceil(target_seconds * words_per_second)))
running_words = 0
selected = 0
for item in manifest["chunks"]:
    running_words += int(item["word_count"])
    selected += 1
    if running_words >= target_words:
        break
print(selected)
PY
    )"
    echo "[qwen-audiobook] selected MAX_CHUNKS=$MAX_CHUNKS for target ${TARGET_AUDIO_SECONDS}s"
fi

render_args=("${base_args[@]}")
if [[ "$FORCE" == "1" ]]; then
    render_args+=(--force)
fi
if [[ -n "$MAX_CHUNKS" && "$MAX_CHUNKS" != "all" ]]; then
    render_args+=(--max-chunks "$MAX_CHUNKS")
fi

if [[ "$AUDIT_CHUNKS" == "1" ]]; then
    # Chunk audit uses faster-whisper from the same environment as active-line video.
    # The per-chunk Qwen generation still runs in its own bootstrap inside run_chunks.sh.
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/bootstrap_audiobook_video.sh"
fi

python "$REPO_ROOT/scripts/build_chunked_audiobook.py" "${render_args[@]}"

SUMMARY_PATH="$WORK_DIR/$BOOK_STEM/audiobook_summary.json"
AUDIO_OUTPUT="$OUTPUT_DIR/${BOOK_STEM}_${MODEL}.wav"
VIDEO_OUTPUT="${VIDEO_OUTPUT:-$OUTPUT_DIR/${BOOK_STEM}_${MODEL}_active_line.mp4}"
TIMING_OUTPUT="${TIMING_OUTPUT:-$OUTPUT_DIR/${BOOK_STEM}_${MODEL}_active_line.timing.json}"

if [[ "$GENERATE_VIDEO" == "1" ]]; then
    video_common_args=(
        --summary "$SUMMARY_PATH" \
        --model "$MODEL" \
        --output "$VIDEO_OUTPUT" \
        --timing-output "$TIMING_OUTPUT" \
        --show-active-line \
        --active-line-timing asr \
        --active-line-asr-model "$VIDEO_ASR_MODEL" \
        --active-line-asr-workers "$VIDEO_ASR_WORKERS" \
        --active-line-asr-cache-dir "$VIDEO_ASR_CACHE_DIR" \
        --video-codec "$VIDEO_CODEC" \
        --video-preset "$VIDEO_PRESET" \
        --video-crf "$VIDEO_CRF" \
        --chapter-end-hold "$CHAPTER_END_HOLD" \
        --chapter-transition "$CHAPTER_TRANSITION" \
        --chapter-start-hold "$CHAPTER_START_HOLD" \
        --chapter-audio-output "$CHAPTER_AUDIO_OUTPUT"
    )

    case "$VIDEO_RENDER_MODE" in
        inline)
            bash "$SCRIPT_DIR/run_audiobook_video.sh" \
                "${video_common_args[@]}" \
                --active-line-asr-device "$VIDEO_ASR_DEVICE"
            ;;
        submit_cpu)
            submit_args=(
                --cluster "$VIDEO_SUBMIT_CLUSTER"
                --mode "$VIDEO_SUBMIT_MODE"
                --resource-mode "$VIDEO_SUBMIT_RESOURCE_MODE"
                --account "$VIDEO_SUBMIT_ACCOUNT"
            )
            if [[ -n "$VIDEO_SUBMIT_PARTITION" ]]; then
                submit_args+=(--partition "$VIDEO_SUBMIT_PARTITION")
            fi
            if [[ -n "$VIDEO_SUBMIT_TIME" ]]; then
                submit_args+=(--time "$VIDEO_SUBMIT_TIME")
            fi
            if [[ -n "$VIDEO_SUBMIT_CPUS" ]]; then
                submit_args+=(--cpus "$VIDEO_SUBMIT_CPUS")
            fi
            if [[ -n "$VIDEO_SUBMIT_GRES" ]]; then
                submit_args+=(--gres "$VIDEO_SUBMIT_GRES")
            fi
            if [[ -n "$VIDEO_SUBMIT_MEM" ]]; then
                submit_args+=(--mem "$VIDEO_SUBMIT_MEM")
            fi

            video_submit_output="$(
                bash "$SCRIPT_DIR/submit_audiobook_video.sh" \
                    "${submit_args[@]}" \
                    -- \
                    "${video_common_args[@]}" \
                    --active-line-asr-device "$VIDEO_SUBMIT_ASR_DEVICE"
            )"
            printf '%s\n' "$video_submit_output"
            printf '%s\n' "$video_submit_output" > "$VIDEO_SUBMIT_JOB_RECORD"
            ;;
        *)
            echo "Unsupported VIDEO_RENDER_MODE: $VIDEO_RENDER_MODE" >&2
            exit 2
            ;;
    esac
fi

echo "[qwen-audiobook] summary: $SUMMARY_PATH"
echo "[qwen-audiobook] audio: $AUDIO_OUTPUT"
if [[ "$GENERATE_VIDEO" == "1" ]]; then
    echo "[qwen-audiobook] chapter-gap audio: $CHAPTER_AUDIO_OUTPUT"
    if [[ "$VIDEO_RENDER_MODE" == "inline" ]]; then
        echo "[qwen-audiobook] video: $VIDEO_OUTPUT"
        echo "[qwen-audiobook] video timing: $TIMING_OUTPUT"
    else
        echo "[qwen-audiobook] video submission: $VIDEO_SUBMIT_JOB_RECORD"
        echo "[qwen-audiobook] video target: $VIDEO_OUTPUT"
        echo "[qwen-audiobook] video timing target: $TIMING_OUTPUT"
    fi
fi
