#!/bin/bash
#
# Audiobook Long-Form Sample Generation Script
# Run on CSC Mahti with GPU
#
# Usage:
#   cd scripts/mahti
#   sbatch run_all_audiobook_samples.sh
#

#SBATCH --job-name=tts_audiobook_long
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=ruwodda@users.example.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

INPUT_DIR="$REPO_ROOT/experiments/tts_naturalness/inputs"
OUTPUT_DIR="$REPO_ROOT/experiments/tts_naturalness/generated/all_samples"

INPUT_FILES=(
    "naturalness_prompt.txt"
    "description_test.txt"
    "dialogue_test.txt"
    "long_audiobook_test.txt"
    "descriptive_test_long.txt"
    "dialogue_test_long.txt"
    "narrative_test_long.txt"
)

log "=== Starting audiobook long-form sample generation ==="

# Make sure output directory exists
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# LOCAL MODELS (run directly on login node or with --gres=none)
# =============================================================================

activate_python_venv "tts-experiments"

# Kokoro - best local model
for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="kokoro_$(basename "$input" .txt).wav"
    
    log "Generating Kokoro: $input -> $output_name"
    
    python "$REPO_ROOT/scripts/run_tts_naturalness_experiment.py" \
        --engine kokoro \
        --voice alloy \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        2>&1 | tee "$OUTPUT_DIR/log_kokoro_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# FISH SPEECH S2 PRO (CSC - A100, ~8GB VRAM)
# =============================================================================

log "Setting up Fish Speech..."
load_pytorch_module 2.9
ensure_python_venv "fish-speech"
activate_python_venv "fish-speech"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="fish_s2_pro_$(basename "$input" .txt).wav"
    
    log "Generating Fish Speech S2 Pro: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_fish_speech.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        --half \
        2>&1 | tee "$OUTPUT_DIR/log_fish_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# PARLER TTS LARGE (CSC - A100)
# =============================================================================

log "Setting up Parler TTS Large..."
ensure_python_venv "parler-tts-large"
activate_python_venv "parler-tts-large"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="parler_large_$(basename "$input" .txt).wav"
    
    log "Generating Parler TTS Large: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_parler_tts_large.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        2>&1 | tee "$OUTPUT_DIR/log_parler_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# MOSS-TTS (CSC - V100, 1.7B)
# =============================================================================

log "Setting up MOSS-TTS..."
ensure_python_venv "moss-tts"
activate_python_venv "moss-tts"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="moss_tts_$(basename "$input" .txt).wav"
    
    log "Generating MOSS-TTS: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_moss_tts.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        --mode clone \
        2>&1 | tee "$OUTPUT_DIR/log_moss_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# SPARK-TTS (CSC - V100, 0.5B)
# =============================================================================

log "Setting up Spark-TTS..."
ensure_python_venv "spark-tts"
activate_python_venv "spark-tts"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="spark_tts_$(basename "$input" .txt).wav"
    
    log "Generating Spark-TTS: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_spark_tts.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        --mode clone \
        2>&1 | tee "$OUTPUT_DIR/log_spark_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# VIBEVoice (CSC - A100) - FIXED: use plain_1b mode (no "Speaker one" bug)
# =============================================================================

log "Setting up VibeVoice..."
ensure_python_venv "vibevoice"
activate_python_venv "vibevoice"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="vibevoice_$(basename "$input" .txt).wav"
    
    log "Generating VibeVoice: $input -> $output_name"
    
    # FIX: use --mode plain_1b (NOT multispeaker) to avoid "Speaker one" bug
    python "$SCRIPT_DIR/run_vibevoice.py" \
        --mode plain_1b \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        2>&1 | tee "$OUTPUT_DIR/log_vibevoice_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# F5-TTS (CSC - A100)
# =============================================================================

log "Setting up F5-TTS..."
ensure_python_venv "f5-tts"
activate_python_venv "f5-tts"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="f5_tts_$(basename "$input" .txt).wav"
    
    log "Generating F5-TTS: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_f5_tts.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        2>&1 | tee "$OUTPUT_DIR/log_f5_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# QWEN3-TTS (CSC - A100/V100)
# =============================================================================

log "Setting up Qwen3-TTS..."
ensure_python_venv "qwen3-tts"
activate_python_venv "qwen3-tts"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="qwen3_tts_$(basename "$input" .txt).wav"
    
    log "Generating Qwen3-TTS: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_qwen3_tts.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        --mode custom_voice \
        2>&1 | tee "$OUTPUT_DIR/log_qwen3_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# CSM-1B (CSC - A100, context-aware)
# =============================================================================

log "Setting up CSM-1B..."
ensure_python_venv "csm-1b"
activate_python_venv "csm-1b"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="csm_1b_$(basename "$input" .txt).wav"
    
    log "Generating CSM-1B: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_csm_1b.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        --mode context_clone \
        2>&1 | tee "$OUTPUT_DIR/log_csm_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# DIA 1.6B (CSC - A100, dialogue-optimized)
# =============================================================================

log "Setting up Dia 1.6B..."
ensure_python_venv "dia"
activate_python_venv "dia"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="dia_1_6b_$(basename "$input" .txt).wav"
    
    log "Generating Dia 1.6B: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_dia.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        --mode clone \
        2>&1 | tee "$OUTPUT_DIR/log_dia_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# ECHO-TTS (CSC - V100)
# =============================================================================

log "Setting up Echo-TTS..."
ensure_python_venv "echo-tts"
activate_python_venv "echo-tts"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="echo_tts_$(basename "$input" .txt).wav"
    
    log "Generating Echo-TTS: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_echo_tts.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        --mode clone \
        2>&1 | tee "$OUTPUT_DIR/log_echo_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# OMNIVOICE (CSC - A100)
# =============================================================================

log "Setting up OmniVoice..."
ensure_python_venv "omnivoice"
activate_python_venv "omnivoice"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="omnivoice_$(basename "$input" .txt).wav"
    
    log "Generating OmniVoice: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_omnivoice.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        --mode clone \
        2>&1 | tee "$OUTPUT_DIR/log_omnivoice_$(basename "$input" .txt).log"
done

deactivate

# =============================================================================
# VOX CPM2 (CSC - A100)
# =============================================================================

log "Setting up VoxCPM2..."
ensure_python_venv "voxcpm2"
activate_python_venv "voxcpm2"

for input in "${INPUT_FILES[@]}"; do
    input_path="$INPUT_DIR/$input"
    output_name="voxcpm2_$(basename "$input" .txt).wav"
    
    log "Generating VoxCPM2: $input -> $output_name"
    
    python "$SCRIPT_DIR/run_voxcpm2.py" \
        --input "$input_path" \
        --output "$OUTPUT_DIR/$output_name" \
        --mode ultimate_clone \
        2>&1 | tee "$OUTPUT_DIR/log_voxcpm2_$(basename "$input" .txt).log"
done

deactivate

log "=== All samples generated ==="
log "Output directory: $OUTPUT_DIR"
log "Total files: $(ls "$OUTPUT_DIR"/*.wav "$OUTPUT_DIR"/*.mp3 2>/dev/null | wc -l)"

# List all generated files
echo "=== Generated files ==="
ls -lh "$OUTPUT_DIR"/*.wav "$OUTPUT_DIR"/*.mp3 2>/dev/null | head -50
