#!/bin/bash
#SBATCH --job-name=tts_top3
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# DON'T source common.sh - use this simplified version
# Just activate conda and run

REPO_ROOT="/scratch/project_2006194/textToAudioTTS"
INPUT_DIR="$REPO_ROOT/experiments/tts_naturalness/inputs"
OUTPUT_DIR="$REPO_ROOT/experiments/tts_naturalness/generated/all_samples"

mkdir -p "$OUTPUT_DIR"

echo "=== TOP 3 Audiobook Samples ==="
echo "Output: $OUTPUT_DIR"

# Load conda
source "$MINICONDA3/etc/profile.d/conda.sh"
conda activate base

# ============================================================================
# MODEL 1: Qwen3-TTS (faster) - slot 63
# ============================================================================
echo "[1/3] Running Qwen3-TTS faster..."

source activate qwen3-tts-faster

for f in naturalness_prompt.txt description_test.txt dialogue_test.txt long_audiobook_test.txt descriptive_test_long.txt dialogue_test_long.txt narrative_test_long.txt; do
    echo "  --> qwen3_faster + $f"
    srun python "$REPO_ROOT/scripts/mahti/run_faster_qwen3_tts.py" \
        --input "$INPUT_DIR/$f" \
        --output "$OUTPUT_DIR/qwen3_faster_${f%.txt}.wav" \
        --generation-mode voice_clone
done

conda deactivate

# ============================================================================
# MODEL 2: VoxCPM2 - slot 75  
# ============================================================================
echo "[2/3] Running VoxCPM2..."

source activate voxcpm2

for f in naturalness_prompt.txt description_test.txt dialogue_test.txt long_audiobook_test.txt descriptive_test_long.txt dialogue_test_long.txt narrative_test_long.txt; do
    echo "  --> voxcpm2_ultimate + $f"
    srun python "$REPO_ROOT/scripts/mahti/run_voxcpm2.py" \
        --input "$INPUT_DIR/$f" \
        --output "$OUTPUT_DIR/voxcpm2_ultimate_${f%.txt}.wav" \
        --mode ultimate_clone
done

conda deactivate

# ============================================================================
# MODEL 3: VibeVoice realtime - slot 84
# ============================================================================
echo "[3/3] Running VibeVoice realtime..."

source activate vibevoice

for f in naturalness_prompt.txt description_test.txt dialogue_test.txt long_audiobook_test.txt descriptive_test_long.txt dialogue_test_long.txt narrative_test_long.txt; do
    echo "  --> vibevoice_realtime + $f"
    srun python "$REPO_ROOT/scripts/mahti/run_vibevoice.py" \
        --mode realtime \
        --input "$INPUT_DIR/$f" \
        --output "$OUTPUT_DIR/vibevoice_realtime_${f%.txt}.wav"
done

conda deactivate

echo "=== DONE ==="
ls -lh "$OUTPUT_DIR"/{qwen3_faster,voxcpm2_ultimate,vibevoice_realtime}*.wav 2>/dev/null | head -25
