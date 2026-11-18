#!/bin/bash
#
# Top-3 audiobook batch generator.
#
# Usage:
#   sbatch --account <project> --partition gpu --gres gpu:v100:1 --mem 24G \
#     scripts/mahti/run_top3_audiobook_samples.sh
#
# Set TOP3_MODEL to one of: qwen3, voxcpm2, vibevoice, all

#SBATCH --job-name=tts_audiobook_top3
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mahti"
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
load_pytorch_module 2.9

INPUT_DIR="$REPO_ROOT/experiments/tts_naturalness/inputs"
OUTPUT_ROOT="$REPO_ROOT/experiments/tts_naturalness/generated/all_samples/top/all"
LOG_ROOT="$REPO_ROOT/experiments/tts_naturalness/generated/runs/top3"
REFERENCE_AUDIO="$REPO_ROOT/assets/audio_samples/brothersKarmatzov.clone_ref_5s.wav"
REFERENCE_TRANSCRIPT="$REPO_ROOT/assets/audio_samples/brothersKarmatzov.clone_ref_5s.transcript.txt"
TOP3_MODEL="${TOP3_MODEL:-all}"

INPUT_FILES=(
    "naturalness_prompt.txt"
    "description_test.txt"
    "dialogue_test.txt"
    "long_audiobook_test.txt"
    "descriptive_test_long.txt"
    "dialogue_test_long.txt"
    "narrative_test_long.txt"
)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

run_python() {
    srun python "$@"
}

run_qwen3() {
    local clone_dir="$OUTPUT_ROOT/qwen3_faster/clone"
    local original_dir="$OUTPUT_ROOT/qwen3_faster/original"

    log "Setting up Faster Qwen3-TTS..."
    ensure_python_venv "faster-qwen3-tts"
    activate_python_venv "faster-qwen3-tts"
    mkdir -p "$clone_dir" "$original_dir"

    for input in "${INPUT_FILES[@]}"; do
        local input_path="$INPUT_DIR/$input"
        local stem="${input%.txt}"
        local original_out="$original_dir/${stem}.wav"
        local clone_out="$clone_dir/${stem}.wav"

        if [[ -s "$original_out" ]]; then
            log "Skipping Faster Qwen3-TTS original (exists): $input"
        elif [[ ! -f "$input_path" ]]; then
            warn "Missing input for Faster Qwen3-TTS original: $input_path"
        else
            log "Generating Faster Qwen3-TTS original: $input"
            if run_python "$SCRIPT_DIR/run_faster_qwen3_tts.py" \
                --input "$input_path" \
                --output "$original_out" \
                --generation-mode custom \
                2>&1 | tee "$LOG_ROOT/qwen3_faster_original_${stem}.log"; then
                :
            else
                warn "Qwen3 original failed for $input"
            fi
        fi

        if [[ -s "$clone_out" ]]; then
            log "Skipping Faster Qwen3-TTS clone (exists): $input"
        elif [[ ! -f "$input_path" ]]; then
            warn "Missing input for Faster Qwen3-TTS clone: $input_path"
        else
            log "Generating Faster Qwen3-TTS clone: $input"
            if run_python "$SCRIPT_DIR/run_faster_qwen3_tts.py" \
                --input "$input_path" \
                --output "$clone_out" \
                --generation-mode clone \
                --xvec-only \
                --reference-audio "$REFERENCE_AUDIO" \
                --reference-text-file "$REFERENCE_TRANSCRIPT" \
                2>&1 | tee "$LOG_ROOT/qwen3_faster_clone_${stem}.log"; then
                :
            else
                warn "Qwen3 clone failed for $input"
            fi
        fi
    done

    deactivate
}

run_voxcpm2() {
    local original_dir="$OUTPUT_ROOT/voxcpm2/original"
    local clone_dir="$OUTPUT_ROOT/voxcpm2/clone"

    log "Setting up VoxCPM2..."
    ensure_python_venv "voxcpm2"
    activate_python_venv "voxcpm2"
    mkdir -p "$original_dir" "$clone_dir"

    for input in "${INPUT_FILES[@]}"; do
        local input_path="$INPUT_DIR/$input"
        local stem="${input%.txt}"
        local original_out="$original_dir/${stem}.wav"
        local clone_out="$clone_dir/${stem}.wav"

        if [[ -s "$original_out" ]]; then
            log "Skipping VoxCPM2 original (exists): $input"
        elif [[ ! -f "$input_path" ]]; then
            warn "Missing input for VoxCPM2 original: $input_path"
        else
            log "Generating VoxCPM2 original: $input"
            if run_python "$SCRIPT_DIR/run_voxcpm2.py" \
                --input "$input_path" \
                --output "$original_out" \
                --generation-mode tts \
                2>&1 | tee "$LOG_ROOT/voxcpm2_original_${stem}.log"; then
                :
            else
                warn "VoxCPM2 original failed for $input"
            fi
        fi

        if [[ -s "$clone_out" ]]; then
            log "Skipping VoxCPM2 clone (exists): $input"
        elif [[ ! -f "$input_path" ]]; then
            warn "Missing input for VoxCPM2 clone: $input_path"
        else
            log "Generating VoxCPM2 clone: $input"
            if run_python "$SCRIPT_DIR/run_voxcpm2.py" \
                --input "$input_path" \
                --output "$clone_out" \
                --generation-mode ultimate \
                --reference-audio "$REFERENCE_AUDIO" \
                --reference-text-file "$REFERENCE_TRANSCRIPT" \
                2>&1 | tee "$LOG_ROOT/voxcpm2_clone_${stem}.log"; then
                :
            else
                warn "VoxCPM2 clone failed for $input"
            fi
        fi
    done

    deactivate
}

run_vibevoice() {
    local plain_dir="$OUTPUT_ROOT/vibevoice/plain_1b"
    local realtime_dir="$OUTPUT_ROOT/vibevoice/realtime"

    log "Setting up VibeVoice..."
    export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
    export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
    ensure_python_venv "vibevoice"
    activate_python_venv "vibevoice"
    mkdir -p "$plain_dir" "$realtime_dir"

    for input in "${INPUT_FILES[@]}"; do
        local input_path="$INPUT_DIR/$input"
        local stem="${input%.txt}"
        local plain_out="$plain_dir/${stem}.wav"
        local realtime_out="$realtime_dir/${stem}.wav"

        if [[ -s "$plain_out" ]]; then
            log "Skipping VibeVoice plain_1b (exists): $input"
        elif [[ ! -f "$input_path" ]]; then
            warn "Missing input for VibeVoice plain_1b: $input_path"
        else
            log "Generating VibeVoice plain_1b: $input"
            if run_python "$SCRIPT_DIR/run_vibevoice.py" \
                --mode plain_1b \
                --input "$input_path" \
                --output "$plain_out" \
                2>&1 | tee "$LOG_ROOT/vibevoice_plain_1b_${stem}.log"; then
                :
            else
                warn "VibeVoice plain_1b failed for $input"
            fi
        fi

        if [[ -s "$realtime_out" ]]; then
            log "Skipping VibeVoice realtime (exists): $input"
        elif [[ ! -f "$input_path" ]]; then
            warn "Missing input for VibeVoice realtime: $input_path"
        else
            log "Generating VibeVoice realtime: $input"
            if run_python "$SCRIPT_DIR/run_vibevoice.py" \
                --mode realtime \
                --input "$input_path" \
                --output "$realtime_out" \
                2>&1 | tee "$LOG_ROOT/vibevoice_realtime_${stem}.log"; then
                :
            else
                warn "VibeVoice realtime failed for $input"
            fi
        fi
    done

    deactivate
}

case "$TOP3_MODEL" in
    all)
        run_qwen3
        run_voxcpm2
        run_vibevoice
        ;;
    qwen3|faster-qwen3|faster_qwen3)
        run_qwen3
        ;;
    voxcpm2)
        run_voxcpm2
        ;;
    vibevoice)
        run_vibevoice
        ;;
    *)
        die "unknown TOP3_MODEL: $TOP3_MODEL"
        ;;
esac

log "=== Top-3 audiobook batch complete ==="
log "Output root: $OUTPUT_ROOT"
