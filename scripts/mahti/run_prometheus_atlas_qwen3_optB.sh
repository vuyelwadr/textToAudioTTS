#!/bin/bash
#SBATCH --job-name=prom-qwen-optB
#SBATCH --account=project_2010549
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2,nvme:900
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out

set -euo pipefail

export CSC_PROJECT="${CSC_PROJECT:-${SLURM_JOB_ACCOUNT:-project_2010549}}"
export MAHTI_WORK_ROOT="${MAHTI_WORK_ROOT:-/scratch/${CSC_PROJECT}/${USER}/textToAudioTTS-mahti}"
export MAHTI_PROJECT_ROOT="${MAHTI_PROJECT_ROOT:-/projappl/${CSC_PROJECT}/textToAudioTTS-mahti}"
export MAHTI_LOCAL_ROOT="${MAHTI_LOCAL_ROOT:-${LOCAL_SCRATCH:-$MAHTI_WORK_ROOT}/textToAudioTTS-mahti}"
export MAHTI_ENV_ROOT="${MAHTI_ENV_ROOT:-$MAHTI_LOCAL_ROOT/venvs}"
export MAHTI_MODEL_ROOT="${MAHTI_MODEL_ROOT:-$MAHTI_WORK_ROOT/models}"
export MAHTI_CACHE_ROOT="${MAHTI_CACHE_ROOT:-$MAHTI_LOCAL_ROOT/cache}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mahti"
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9

# --- CONFIG ---
INPUT_TEXT="$REPO_ROOT/library/PrometheusAtlas/PrometheusAtlas.txt"
REFERENCE_AUDIO="$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.wav"
REFERENCE_TEXT="$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.transcript.txt"
# Use existing chunks from the previous 3h run
EXISTING_WORK_DIR="$REPO_ROOT/generated/audiobooks-prometheus_atlas-qwen3_3h_20260425/.work"
CHUNK_DIR="$EXISTING_WORK_DIR/PrometheusAtlas/chunks"
OUTPUT_ROOT="$REPO_ROOT/generated/audiobooks-prometheus_atlas-qwen3_optB_20260502"
MODEL_NAME="qwen3_faster_clone"
CHUNK_AUDIO_DIR="$OUTPUT_ROOT/PrometheusAtlas/${MODEL_NAME}/chunks"
AUDIO_OUTPUT="$OUTPUT_ROOT/PrometheusAtlas_${MODEL_NAME}_optB.wav"
VIDEO_OUTPUT="$OUTPUT_ROOT/PrometheusAtlas_${MODEL_NAME}_optB_active_line.mp4"
TIMING_OUTPUT="$OUTPUT_ROOT/PrometheusAtlas_${MODEL_NAME}_optB_active_line.timing.json"

# Optional: local Mac destination for auto-copy (set via env or leave empty to skip)
LOCAL_COPY_DEST="${LOCAL_COPY_DEST:-}"

# --- STEP 0: VALIDATE EXISTING CHUNKS ---
echo "[optB] Using existing chunks from $CHUNK_DIR"
CHUNK_COUNT=$(find "$CHUNK_DIR" -name 'chunk_*.txt' | wc -l)
echo "[optB] Existing chunks: $CHUNK_COUNT"
if [[ "$CHUNK_COUNT" -eq 0 ]]; then
    echo "[optB] ERROR: No existing chunks found!"
    exit 1
fi

# --- STEP 1: GENERATE AUDIO with 4096 tokens ---
echo "[optB] Step 1: Generating Qwen3 audio with max_new_tokens=4096 max_seq_len=4096..."
source "$SCRIPT_DIR/bootstrap_faster_qwen3_tts.sh"
activate_python_venv "faster-qwen3-tts"

mkdir -p "$CHUNK_AUDIO_DIR"

python -u "$SCRIPT_DIR/run_qwen3_batch_optimized.py" \
    --input-dir "$CHUNK_DIR" \
    --output-dir "$CHUNK_AUDIO_DIR" \
    --prefix "chunk_" \
    --generation-mode clone \
    --xvec-only \
    --attn-implementation sdpa \
    --max-new-tokens 4096 \
    --max-seq-len 4096 \
    --temperature 0.88 \
    --top-k 55 \
    --workers auto \
    --reference-audio "$REFERENCE_AUDIO" \
    --reference-text-file "$REFERENCE_TEXT" \
    --log-dir "$OUTPUT_ROOT/logs"

# --- STEP 2: JOIN WAVs ---
echo "[optB] Step 2: Joining chunk WAVs..."
python - "$CHUNK_AUDIO_DIR" "$AUDIO_OUTPUT" <<'PY'
import glob, wave, sys, os
from pathlib import Path

chunk_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])

paths = sorted(chunk_dir.glob("*.wav"), key=lambda p: int(p.stem.rsplit("_", 1)[1]) if p.stem.rsplit("_", 1)[1].isdigit() else p.stem)
if not paths:
    print("No chunk WAVs found!")
    sys.exit(1)

params = None
total_frames = 0
tmp_path = output_path.with_suffix(".tmp.wav")
if tmp_path.exists():
    tmp_path.unlink()

with wave.open(str(tmp_path), "wb") as writer:
    for wav_path in paths:
        with wave.open(str(wav_path), "rb") as reader:
            p = reader.getparams()
            if params is None:
                params = p
                writer.setnchannels(p.nchannels)
                writer.setsampwidth(p.sampwidth)
                writer.setframerate(p.framerate)
                writer.setcomptype(p.comptype, p.compname)
            frames = reader.readframes(reader.getnframes())
            writer.writeframes(frames)
            total_frames += reader.getnframes()

tmp_path.replace(output_path)
dur = total_frames / params.framerate
print(f"[join] {len(paths)} chunks -> {dur:.2f}s -> {output_path}")
PY

# --- STEP 3: GENERATE VIDEO ---
echo "[optB] Step 3: Generating active-line video..."
source "$SCRIPT_DIR/bootstrap_audiobook_video.sh"

python "$SCRIPT_DIR/run_audiobook_video.py" \
    --audio "$AUDIO_OUTPUT" \
    --text "$INPUT_TEXT" \
    --chunk-text-dir "$CHUNK_DIR" \
    --chunk-audio-dir "$CHUNK_AUDIO_DIR" \
    --output "$VIDEO_OUTPUT" \
    --timing-output "$TIMING_OUTPUT" \
    --show-active-line \
    --active-line-timing asr \
    --active-line-asr-model small.en \
    --active-line-asr-device cuda \
    --active-line-asr-workers auto \
    --video-codec libx264 \
    --video-preset veryfast \
    --video-crf 18

# --- STEP 4: COMPLETION MARKER + AUTO-COPY ---
echo "[optB] Done!"
echo "  Audio: $AUDIO_OUTPUT"
echo "  Video: $VIDEO_OUTPUT"
echo "  Timing: $TIMING_OUTPUT"

cat > "$OUTPUT_ROOT/COMPLETED.txt" <<EOF
completed_at=$(date -Is)
audio=$AUDIO_OUTPUT
video=$VIDEO_OUTPUT
timing=$TIMING_OUTPUT
EOF

if [[ -n "$LOCAL_COPY_DEST" ]]; then
    echo "[optB] Auto-copying to $LOCAL_COPY_DEST ..."
    rsync -av --progress \
        "$AUDIO_OUTPUT" \
        "$VIDEO_OUTPUT" \
        "$TIMING_OUTPUT" \
        "$OUTPUT_ROOT/COMPLETED.txt" \
        "$LOCAL_COPY_DEST" || echo "[optB] Auto-copy failed (check LOCAL_COPY_DEST and SSH keys)"
fi
