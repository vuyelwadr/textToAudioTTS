#!/bin/bash
# Run Anna Karenina 5-minute audit pulse with multiple TTS models
# This creates a comparison of different models on the same text

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$REPO_ROOT/generated/anna_karenina_5min_pulse"
MANIFEST="$WORK_DIR/anna_karenina_5min_manifest.json"

# Models to test
MODELS=(
    "qwen3_faster_clone:en-US:female"
    "vibevoice_realtime:en-US:Mike_man"
    "fish_speech_batch:en-US:female"
    "xtts_v2_batch:en-US:female"
    "gpt_sovits_batch:en-US:female"
)

# Reference audio for cloning
REFERENCE_AUDIO="$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.wav"
REFERENCE_TEXT_FILE="$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.transcript.txt"

echo "Running Anna Karenina 5-minute audit pulse with ${#MODELS[@]} models..."
echo "Manifest: $MANIFEST"
echo ""

# Create output directories for each model
for model in "${MODELS[@]}"; do
    IFS=':' read -r MODEL_NAME SPEAKER_LANG SPEAKER_GENDER <<< "$model"
    OUTPUT_DIR="$WORK_DIR/$MODEL_NAME"
    mkdir -p "$OUTPUT_DIR"
    echo "Created $OUTPUT_DIR"
done

# Process each model
for model in "${MODELS[@]}"; do
    IFS=':' read -r MODEL_NAME SPEAKER_LANG SPEAKER_GENDER <<< "$model"
    OUTPUT_DIR="$WORK_DIR/$MODEL_NAME"

    echo ""
    echo "========================================="
    echo "Processing model: $MODEL_NAME"
    echo "========================================="

    # Run the model-specific runner
    case "$MODEL_NAME" in
        qwen3_faster_clone)
            python "$REPO_ROOT/scripts/mahti/run_faster_qwen3_tts.py" \
                --manifest "$MANIFEST" \
                --output-dir "$OUTPUT_DIR" \
                --generation-mode "clone" \
                --reference-audio "$REFERENCE_AUDIO" \
                --reference-text-file "$REFERENCE_TEXT_FILE" \
                --xvec-only \
                --attn-implementation sdpa \
                --temperature 0.88 \
                --top-k 55
            ;;
        vibevoice_realtime)
            python "$REPO_ROOT/scripts/mahti/run_vibevoice.py" \
                --manifest "$MANIFEST" \
                --output-dir "$OUTPUT_DIR" \
                --mode realtime \
                --speaker-name "en-Mike_man" \
                --cfg-scale 1.7 \
                --ddpm-steps 8
            ;;
        fish_speech_batch)
            python "$REPO_ROOT/scripts/mahti/run_fish_speech_batch.py" \
                --manifest "$MANIFEST" \
                --output-dir "$OUTPUT_DIR"
            ;;
        xtts_v2_batch)
            python "$REPO_ROOT/scripts/mahti/run_xtts_v2_batch.py" \
                --manifest "$MANIFEST" \
                --output-dir "$OUTPUT_DIR" \
                --language "en"
            ;;
        gpt_sovits_batch)
            python "$REPO_ROOT/scripts/mahti/run_gpt_sovits_batch.py" \
                --manifest "$MANIFEST" \
                --output-dir "$OUTPUT_DIR" \
                --language "en"
            ;;
        *)
            echo "Unknown model: $MODEL_NAME"
            continue
            ;;
    esac

    echo "Completed $MODEL_NAME"
done

echo ""
echo "========================================="
echo "All models processed!"
echo "========================================="
echo ""
echo "Output directories:"
for model in "${MODELS[@]}"; do
    IFS=':' read -r MODEL_NAME SPEAKER_LANG SPEAKER_GENDER <<< "$model"
    OUTPUT_DIR="$WORK_DIR/$MODEL_NAME"
    echo "  - $MODEL_NAME: $OUTPUT_DIR"
done
echo ""
echo "Next: Run chunk audit on all generated audio files"