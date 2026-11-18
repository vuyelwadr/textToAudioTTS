#!/bin/bash
#SBATCH --job-name=prom-moss-video
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1,nvme:900
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out

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

CHUNK_DIR="${CHUNK_DIR:-$REPO_ROOT/generated/prometheus_benchmark_smoke_20260430/PrometheusAtlas/chunks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/generated/audiobooks-prometheus_atlas-moss_tts_clone_full_20260501/PrometheusAtlas}"
INPUT_TEXT="${INPUT_TEXT:-$REPO_ROOT/library/PrometheusAtlas/PrometheusAtlas.txt}"
REFERENCE_AUDIO="${REFERENCE_AUDIO:-$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.wav}"
AUDIO_OUTPUT="${AUDIO_OUTPUT:-$OUTPUT_ROOT/PrometheusAtlas_moss_tts_local_clone.wav}"
VIDEO_OUTPUT="${VIDEO_OUTPUT:-$OUTPUT_ROOT/PrometheusAtlas_moss_tts_local_clone_active_line.mp4}"
TIMING_OUTPUT="${TIMING_OUTPUT:-$OUTPUT_ROOT/PrometheusAtlas_moss_tts_local_clone_active_line.timing.json}"
VIDEO_ASR_DEVICE="${VIDEO_ASR_DEVICE:-cuda}"
VIDEO_ASR_MODEL="${VIDEO_ASR_MODEL:-small.en}"
VIDEO_ASR_WORKERS="${VIDEO_ASR_WORKERS:-auto}"
VIDEO_CODEC="${VIDEO_CODEC:-libx264}"
VIDEO_PRESET="${VIDEO_PRESET:-veryfast}"
VIDEO_CRF="${VIDEO_CRF:-18}"
ACTIVE_LINE_TIMING="${ACTIVE_LINE_TIMING:-asr}"

# The join step only validates/concatenates WAVs and writes a video-compatible
# summary, so run it in the video environment and avoid re-prefetching MOSS.
bash "$SCRIPT_DIR/bootstrap_audiobook_video.sh"

srun python "$SCRIPT_DIR/run_moss_prometheus_full.py" \
    --chunk-dir "$CHUNK_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --input-text "$INPUT_TEXT" \
    --reference-audio "$REFERENCE_AUDIO" \
    --join-only

bash "$SCRIPT_DIR/run_audiobook_video.sh" \
    --summary "$OUTPUT_ROOT/audiobook_summary.json" \
    --model "moss_tts_local_clone" \
    --output "$VIDEO_OUTPUT" \
    --timing-output "$TIMING_OUTPUT" \
    --show-active-line \
    --active-line-timing "$ACTIVE_LINE_TIMING" \
    --active-line-asr-model "$VIDEO_ASR_MODEL" \
    --active-line-asr-device "$VIDEO_ASR_DEVICE" \
    --active-line-asr-workers "$VIDEO_ASR_WORKERS" \
    --video-codec "$VIDEO_CODEC" \
    --video-preset "$VIDEO_PRESET" \
    --video-crf "$VIDEO_CRF"

python - "$AUDIO_OUTPUT" "$VIDEO_OUTPUT" "$TIMING_OUTPUT" <<'PY'
import json
import sys
import wave
from pathlib import Path

audio = Path(sys.argv[1])
video = Path(sys.argv[2])
timing = Path(sys.argv[3])

with wave.open(str(audio), "rb") as reader:
    duration = reader.getnframes() / reader.getframerate()

payload = {
    "audio": str(audio),
    "audio_exists": audio.exists(),
    "audio_bytes": audio.stat().st_size if audio.exists() else None,
    "audio_duration_seconds": duration,
    "video": str(video),
    "video_exists": video.exists(),
    "video_bytes": video.stat().st_size if video.exists() else None,
    "timing": str(timing),
    "timing_exists": timing.exists(),
}
verify_path = video.with_suffix(".verified.json")
verify_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
