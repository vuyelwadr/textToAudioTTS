#!/bin/bash
#
# Half of a Yellow Sun audiobook runner.
#
# Usage:
#   MODE=preflight MODELS=vibevoice_realtime srun --pty --account <project> \
#     --partition gpu --gres gpu:v100:1 --cpus-per-task 10 --time 00:45:00 \
#     scripts/mahti/run_half_of_a_yellow_sun.sh
#
#   MODE=full PARALLEL_MODELS=1 FORCE=0 sbatch --account <project> \
#     --partition gpusmall --gres gpu:a100:2,nvme:900 --cpus-per-task 32 \
#     scripts/mahti/run_half_of_a_yellow_sun.sh
#
#   MODE=full PARALLEL_MODELS=1 FORCE=0 GENERATE_VIDEO=1 VIDEO_MODEL=qwen3_faster_clone \
#     VIDEO_RENDER_MODE=submit_cpu sbatch --account <project> \
#     --partition gpusmall --gres gpu:a100:2,nvme:900 --cpus-per-task 32 \
#     scripts/mahti/run_half_of_a_yellow_sun.sh
#

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mahti"
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
load_pytorch_module 2.9

export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
export MAHTI_MODEL_ROOT="$MAHTI_WORK_ROOT/models"
export MAHTI_CACHE_ROOT="$MAHTI_WORK_ROOT/cache"

MODE="${MODE:-preflight}"
MODELS=(${MODELS:-qwen3_faster_clone vibevoice_realtime})
CHUNK_LIMIT="${CHUNK_LIMIT:-1}"
PARALLEL_MODELS="${PARALLEL_MODELS:-}"
FORCE="${FORCE:-0}"
GENERATE_VIDEO="${GENERATE_VIDEO:-0}"
VIDEO_MODEL="${VIDEO_MODEL:-qwen3_faster_clone}"
VIDEO_RENDER_MODE="${VIDEO_RENDER_MODE:-submit_cpu}"
VIDEO_ASR_DEVICE="${VIDEO_ASR_DEVICE:-cuda}"
VIDEO_ASR_MODEL="${VIDEO_ASR_MODEL:-small.en}"
VIDEO_ACTIVE_LINE_TIMING="${VIDEO_ACTIVE_LINE_TIMING:-asr}"
VIDEO_OUTPUT="${VIDEO_OUTPUT:-}"
VIDEO_TIMING_OUTPUT="${VIDEO_TIMING_OUTPUT:-}"
VIDEO_ASR_CACHE_DIR="${VIDEO_ASR_CACHE_DIR:-}"
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
VIDEO_SUBMIT_JOB_RECORD="${VIDEO_SUBMIT_JOB_RECORD:-}"
INPUT_FILE="${INPUT_FILE:-$REPO_ROOT/library/halfOfAYellowSun/half_of_a_yellow_sun.txt}"
REFERENCE_AUDIO="${REFERENCE_AUDIO:-$REPO_ROOT/outputs/video_downloads/haCxgJSFmFc/haCxgJSFmFc_reference_8s.wav}"
REFERENCE_TEXT_FILE="${REFERENCE_TEXT_FILE:-$REPO_ROOT/outputs/video_downloads/haCxgJSFmFc/haCxgJSFmFc_reference_8s.transcript.auto.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$MAHTI_WORK_ROOT/generated/audiobooks-half_of_a_yellow_sun-${MODE}}"
WORK_DIR="${WORK_DIR:-$MAHTI_WORK_ROOT/generated/audiobooks-half_of_a_yellow_sun-${MODE}}"
CHUNK_SIZE="${CHUNK_SIZE:-850}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-200}"

args=(
    --input "$INPUT_FILE"
    --models "${MODELS[@]}"
    --output-dir "$OUTPUT_DIR"
    --work-dir "$WORK_DIR"
    --reference-audio "$REFERENCE_AUDIO"
    --chunk-size "$CHUNK_SIZE"
    --min-chunk-size "$MIN_CHUNK_SIZE"
)

if [[ -f "$REFERENCE_TEXT_FILE" ]]; then
    args+=(--reference-text-file "$REFERENCE_TEXT_FILE")
fi

case "$MODE" in
    preflight)
        if [[ -z "$PARALLEL_MODELS" ]]; then
            PARALLEL_MODELS=0
        fi
        args+=(--max-chunks "$CHUNK_LIMIT")
        ;;
    full)
        if [[ -z "$PARALLEL_MODELS" ]]; then
            PARALLEL_MODELS=1
        fi
        ;;
    *)
        die "unknown MODE: $MODE"
        ;;
esac

if [[ "$PARALLEL_MODELS" == "1" ]]; then
    args+=(--parallel-models)
fi

if [[ "$FORCE" == "1" ]]; then
    args+=(--force)
fi

python "$SCRIPT_DIR/../build_chunked_audiobook.py" "${args[@]}"

if [[ "$GENERATE_VIDEO" == "1" ]]; then
    audiobook_stem=$(basename "${INPUT_FILE%.*}")
    summary_path="$WORK_DIR/$audiobook_stem/audiobook_summary.json"
    effective_video_asr_cache_dir="${VIDEO_ASR_CACHE_DIR:-$WORK_DIR/$audiobook_stem/$VIDEO_MODEL/chunk_audit/asr_cache}"
    effective_video_submit_job_record="${VIDEO_SUBMIT_JOB_RECORD:-$OUTPUT_DIR/${VIDEO_MODEL}_video_job.txt}"
    video_common_args=(
        --summary "$summary_path"
        --model "$VIDEO_MODEL"
        --show-active-line
        --active-line-timing "$VIDEO_ACTIVE_LINE_TIMING"
        --active-line-asr-model "$VIDEO_ASR_MODEL"
        --active-line-asr-cache-dir "$effective_video_asr_cache_dir"
    )

    if [[ -n "$VIDEO_OUTPUT" ]]; then
        video_common_args+=(--output "$VIDEO_OUTPUT")
    fi
    if [[ -n "$VIDEO_TIMING_OUTPUT" ]]; then
        video_common_args+=(--timing-output "$VIDEO_TIMING_OUTPUT")
    fi

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
            printf '%s\n' "$video_submit_output" > "$effective_video_submit_job_record"
            ;;
        *)
            die "unsupported VIDEO_RENDER_MODE: $VIDEO_RENDER_MODE"
            ;;
    esac
fi
