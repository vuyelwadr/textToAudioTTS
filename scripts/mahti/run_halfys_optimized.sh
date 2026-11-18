#!/bin/bash
#
# Optimized Half of a Yellow Sun audiobook runner with parallel chunk processing
#
# Usage:
#   # For preflight (test with 1 chunk)
#   MODE=preflight CHUNK_LIMIT=1 sbatch --partition gpu --gres gpu:v100:1 --cpus-per-task 10 --time 01:00:00 scripts/mahti/run_halfys_optimized.sh
#
#   # For full run with parallel processing
#   BATCH_SIZE=12 sbatch --partition gpu --gres gpu:v100:1 --cpus-per-task 10 --time 08:00:00 scripts/mahti/run_halfys_optimized.sh
#
#   # Resume from existing chunks
#   RESUME=1 sbatch ...scripts/mahti/run_halfys_optimized.sh

#SBATCH --job-name=halfys_optimized
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --mail-user=ruwodda@users.example.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="/scratch/project_2010549/ruwododa/textToAudioTTS-puhti"
export REPO_ROOT
MAHTI_WORK_ROOT="$REPO_ROOT"
export MAHTI_WORK_ROOT
MAHTI_PROJECT_ROOT="$REPO_ROOT"
export MAHTI_PROJECT_ROOT
MAHTI_EXTERNAL_ROOT="$REPO_ROOT/external"
export MAHTI_EXTERNAL_ROOT
MAHTI_MODEL_ROOT="$REPO_ROOT/models"
export MAHTI_MODEL_ROOT
MAHTI_ENV_ROOT="$REPO_ROOT/venvs"
export MAHTI_ENV_ROOT
MAHTI_CACHE_ROOT="$REPO_ROOT/cache"
export MAHTI_CACHE_ROOT

source /scratch/project_2010549/ruwododa/textToAudioTTS-puhti/scripts/mahti/common.sh
load_pytorch_module 2.9

source /scratch/project_2010549/ruwododa/textToAudioTTS-puhti/scripts/mahti/bootstrap_vibevoice.sh

REPO_ROOT="/scratch/project_2010549/ruwododa/textToAudioTTS-puhti"
export REPO_ROOT

MODE="${MODE:-full}"
MODELS=(${MODELS:-vibevoice_realtime})
CHUNK_LIMIT="${CHUNK_LIMIT:-203}"
BATCH_SIZE=1
RESUME="${RESUME:-0}"

INPUT_FILE="/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/library/halfOfAYellowSun/half_of_a_yellow_sun.txt"
REFERENCE_AUDIO="/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/outputs/video_downloads/haCxgJSFmFc/haCxgJSFmFc_reference_8s.wav"
REFERENCE_TEXT="/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/outputs/video_downloads/haCxgJSFmFc/haCxgJSFmFc_reference_8s.transcript.auto.txt"
OUTPUT_DIR="/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/generated/audiobooks-half_of_a_yellow_sun-${MODE}"
WORK_DIR="/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/generated/audiobooks-half_of_a_yellow_sun-${MODE}"
CHUNK_ROOT="$WORK_DIR/chunks"
CHUNK_SIZE="${CHUNK_SIZE:-850}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-200}"

log "=== Half of a Yellow Sun - Optimized Run ==="
log "REPO_ROOT: $REPO_ROOT"
log "SCRIPT_DIR: $SCRIPT_DIR"
log "MODE: $MODE"
log "BATCH_SIZE: $BATCH_SIZE"
log "RESUME: $RESUME"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHUNK_ROOT"

if [[ "$RESUME" != "1" ]]; then
    log "=== Step 1: Create text chunks ==="
    python "$SCRIPT_DIR/../textChunkingAdapter.py" \
        --input "$INPUT_FILE" \
        --output-dir "$CHUNK_ROOT" \
        --max-chunk-size "$CHUNK_SIZE" \
        --min-chunk-size "$MIN_CHUNK_SIZE" \
        --num-workers 10 \
        --prefix "half_of_a_yellow_sun"
    
    log "Chunks created at: $CHUNK_ROOT"
log "Checking manifest at: $CHUNK_ROOT/half_of_a_yellow_sun_manifest.json"
fi

CHUNKS_JSON="$CHUNK_ROOT/half_of_a_yellow_sun_manifest.json"
log "CHUNKS_JSON: $CHUNKS_JSON"
if [[ ! -f "$CHUNKS_JSON" ]]; then
    die "Chunk manifest not found: $CHUNKS_JSON"
fi

TOTAL_CHUNKS=$(python3 -c "import json; print(len(json.load(open('$CHUNKS_JSON'))['chunks']))")
log "Total chunks: $TOTAL_CHUNKS"

if [[ "$CHUNK_LIMIT" != "203" && -n "$CHUNK_LIMIT" ]]; then
    TOTAL_CHUNKS=$CHUNK_LIMIT
fi

log "=== Step 2: Process chunks in parallel batches of $BATCH_SIZE ==="

for model in "${MODELS[@]}"; do
    log "Processing model: $model"
    
    MODEL_OUTPUT_DIR="$WORK_DIR/$model"
    CHUNK_AUDIO_DIR="$MODEL_OUTPUT_DIR/chunks"
    LOGS_DIR="$MODEL_OUTPUT_DIR/logs"
    
    mkdir -p "$CHUNK_AUDIO_DIR"
    mkdir -p "$LOGS_DIR"
    
    case "$model" in
        vibevoice_realtime)
            MODEL_VENV="vibevoice"
            RUNNER_SCRIPT="$SCRIPT_DIR/run_vibevoice.py"
            RUNNER_ARGS=("--mode" "realtime" "--speaker-name" "en-Mike_man" "--cfg-scale" "1.7" "--ddpm-steps" "8")
            ;;
        qwen3_faster_clone)
            MODEL_VENV="faster-qwen3-tts"
            RUNNER_SCRIPT="$SCRIPT_DIR/run_faster_qwen3_tts.py"
            RUNNER_ARGS=(
                "--generation-mode" "clone"
                "--no-do-sample"
                "--non-streaming-mode"
                "--temperature" "0.88"
                "--top-k" "55"
                "--attn-implementation" "sdpa"
                "--reference-audio" "$REFERENCE_AUDIO"
            )
            if [[ -f "$REFERENCE_TEXT" ]]; then
                RUNNER_ARGS+=("--reference-text-file" "$REFERENCE_TEXT")
            fi
            ;;
        *)
            die "Unknown model: $model"
            ;;
    esac
    
    ensure_python_venv "$MODEL_VENV"
    activate_python_venv "$MODEL_VENV"
    
    CHUNK_FILES=("$CHUNK_ROOT"/chunk_*.txt)
    TOTAL_CHUNKS=${#CHUNK_FILES[@]}
    log "Found $TOTAL_CHUNKS txt files in $CHUNK_ROOT"
    
    CHUNK_NUM=0
    BATCH_NUM=0
    DONE_COUNT=0
    
    while [[ $DONE_COUNT -lt $TOTAL_CHUNKS ]]; do
        BATCH_START=$DONE_COUNT
        BATCH_END=$((DONE_COUNT + BATCH_SIZE))
        if [[ $BATCH_END -gt $TOTAL_CHUNKS ]]; then
            BATCH_END=$TOTAL_CHUNKS
        fi
        
        log "=== Processing sequentially (no parallelism) - maximizing GPU ==="
        
        for i in $(seq 0 $((TOTAL_CHUNKS - 1))); do
            CHUNK_FILE="${CHUNK_FILES[$i]}"
            CHUNK_BASE=$(basename "$CHUNK_FILE" .txt)
            CHUNK_OUTPUT="$CHUNK_AUDIO_DIR/${CHUNK_BASE}.wav"
            CHUNK_LOG="$LOGS_DIR/${CHUNK_BASE}.log"
            
            if [[ -s "$CHUNK_OUTPUT" ]]; then
                log "Skip chunk $i: already exists"
                continue
            fi
            
            if [[ ! -f "$CHUNK_FILE" ]]; then
                log "Warning: chunk file not found: $CHUNK_FILE"
                continue
            fi
            
            log "Processing chunk $i/$((TOTAL_CHUNKS-1)): $(basename $CHUNK_FILE)"
            python "$RUNNER_SCRIPT" \
                --input "$CHUNK_FILE" \
                --output "$CHUNK_OUTPUT" \
                "${RUNNER_ARGS[@]}" \
                >"$CHUNK_LOG" 2>&1
            
            if [[ -s "$CHUNK_OUTPUT" ]]; then
                log "Completed chunk $i"
            else
                log "Failed chunk $i"
            fi
        done
        
        DONE_COUNT=$BATCH_END
        BATCH_NUM=$((BATCH_NUM + 1))
        
        COMPLETED_THIS_RUN=0
        for i in $(seq $BATCH_START $((BATCH_END - 1))); do
            CHUNK_FILE="${CHUNK_FILES[$i]}"
            CHUNK_BASE=$(basename "$CHUNK_FILE" .txt)
            CHUNK_OUTPUT="$CHUNK_AUDIO_DIR/${CHUNK_BASE}.wav"
            if [[ -s "$CHUNK_OUTPUT" ]]; then
                COMPLETED_THIS_RUN=$((COMPLETED_THIS_RUN + 1))
            fi
        done
        
        log "Batch $BATCH_NUM completed: $COMPLETED_THIS_RUN/$((BATCH_END - BATCH_START)) chunks"
    done
    
    deactivate
    
    log "=== Step 3: Join all chunks into final audiobook ==="
    
    FINAL_OUTPUT="$OUTPUT_DIR/${model}.wav"
    
    python3 << EOF
import wave
import os
from pathlib import Path

chunk_dir = Path("$CHUNK_AUDIO_DIR")
output_path = Path("$FINAL_OUTPUT")

wav_files = sorted(chunk_dir.glob("half_of_a_yellow_sun_chunk_*.wav"))
if not wav_files:
    raise ValueError("No chunk WAVs found")

params = None
frames = []
for wav_path in wav_files:
    with wave.open(str(wav_path), "rb") as reader:
        if params is None:
            params = reader.getparams()
        frames.append(reader.readframes(reader.getnframes()))

output_path.parent.mkdir(parents=True, exist_ok=True)
with wave.open(str(output_path), "wb") as writer:
    writer.setparams(params)
    writer.writeframes(b"".join(frames))

print(f"Joined {len(wav_files)} chunks to {output_path}")
EOF
    
    log "Final output: $FINAL_OUTPUT"
done

log "=== All done! ==="
log "Output directory: $OUTPUT_DIR"
