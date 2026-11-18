# LONG-FORM AUDIOBOOK TTS RANKING PLAN - EXACT EXECUTION

## INPUT FILES (7 total)
All located in `experiments/tts_naturalness/inputs/`:

1. `naturalness_prompt.txt` - 1-line prompt (~50 words)
2. `description_test.txt` - Short descriptive test (~150 words)
3. `dialogue_test.txt` - Short dialogue test (~120 words)
4. `long_audiobook_test.txt` - Original long test (~500 words)
5. `descriptive_test_long.txt` - NEW long descriptive (~785 words, ~5 min audio)
6. `dialogue_test_long.txt` - NEW long dialogue (~489 words, ~3 min audio)
7. `narrative_test_long.txt` - NEW long narrative (~552 words, ~3.5 min audio)

## YOUR TOP SAMPLES (from the "top" folder)
- `63_faster_qwen3_tts_voice_clone_hacx.wav`
- `75_voxcpm2_ultimate_clone.wav`
- `84_vibevoice_realtime_0_5b.wav` ← HAS "Speaker one" bug (needs fix)

## VIBEVoice BUG FIX
**ALREADY FIXED** in `scripts/mahti/run_vibevoice.py` lines 207-209:

```python
# Line 207-209 - for plain_1b (single speaker) mode:
# DO NOT add Speaker 1: prefix - this is what causes the model to announce "Speaker one"
# For single-speaker narration just pass plain text with no speaker tags at all
content = f"{text}\n"
```

**IMPORTANT**: When running VibeVoice, ALWAYS use `--mode plain_1b` (not `--mode multispeaker`).

---

## HOW TO RUN - EXACT COMMANDS

### Method 1: Single Script (Recommended)
```bash
# SSH to CSC Mahti:
ssh user@mahti.csc.fi

# Navigate to repo:
cd /path/to/textToAudioTTS

# Make sure environments are set up:
cd scripts/mahti

# Option A: Run everything at once (if you have the time allotment)
sbatch run_all_audiobook_samples.sh

# Option B: Run individual models (see below)
```

### Method 2: Individual Models (Run what you need)
The script above handles these models. Here's how to run each manually if needed:

```bash
cd /path/to/textToAudioTTS/scripts/mahti

# Fish Speech S2 Pro:
sbatch run_fish_speech.py --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/fish_s2_pro_narrative.wav --half

# Parler TTS Large:
sbatch run_parler_tts_large.py --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/parler_large_narrative.wav

# MOSS-TTS (voice clone):
sbatch run_moss_tts.py --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/moss_tts_narrative.wav --mode clone

# Spark-TTS (voice clone):
sbatch run_spark_tts.py --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/spark_tts_narrative.wav --mode clone

# VibeVoice (FIXED - use plain_1b mode):
sbatch run_vibevoice.py --mode plain_1b --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/vibevoice_narrative.wav

# F5-TTS:
sbatch run_f5_tts.py --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/f5_tts_narrative.wav

# Qwen3-TTS:
sbatch run_qwen3_tts.py --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/qwen3_tts_narrative.wav --mode custom_voice

# CSM-1B:
sbatch run_csm_1b.py --input ../../experiments/tts_naturalness/inputs/dialogue_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/csm_1b_dialogue.wav --mode context_clone

# Dia 1.6B:
sbatch run_dia.py --input ../../experiments/tts_naturalness/inputs/dialogue_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/dia_dialogue.wav --mode clone

# Echo-TTS:
sbatch run_echo_tts.py --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/echo_tts_narrative.wav --mode clone

# OmniVoice:
sbatch run_omnivoice.py --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/omnivoice_narrative.wav --mode clone

# VoxCPM2:
sbatch run_voxcpm2.py --input ../../experiments/tts_naturalness/inputs/narrative_test_long.txt --output ../../experiments/tts_naturalness/generated/all_samples/voxcpm2_narrative.wav --mode ultimate_clone
```

### Method 3: Local Mac (Kokoro only)
```bash
cd /path/to/textToAudioTTS
source .venv-tts-experiments/bin/activate

python scripts/run_tts_naturalness_experiment.py \
    --engine kokoro \
    --voice alloy \
    --input experiments/tts_naturalness/inputs/narrative_test_long.txt \
    --output experiments/tts_naturalness/generated/all_samples/kokoro_narrative.wav
```

---

## OUTPUT LOCATION
All outputs go to:
```
experiments/tts_naturalness/generated/all_samples/
```

Naming convention: `{model_name}_{passage_name}.wav`

Example:
- `fish_s2_pro_narrative_test_long.wav`
- `vibevoice_dialogue_test_long.wav`
- `kokoro_description_test.txt.wav`

---

## MODELS TO RUN (Priority Order)

| Priority | Model | Command | Est. Time |
|----------|-------|---------|-----------|
| 1 | Fish Speech S2 Pro | `sbatch run_fish_speech.py ...` | 30 min |
| 2 | Parler TTS Large | `sbatch run_parler_tts_large.py ...` | 40 min |
| 3 | MOSS-TTS | `sbatch run_moss_tts.py --mode clone ...` | 35 min |
| 4 | Spark-TTS | `sbatch run_spark_tts.py --mode clone ...` | 20 min |
| 5 | VibeVoice | `sbatch run_vibevoice.py --mode plain_1b ...` | 45 min |
| 6 | F5-TTS | `sbatch run_f5_tts.py ...` | 25 min |
| 7 | Qwen3-TTS | `sbatch run_qwen3_tts.py --mode custom_voice ...` | 20 min |
| 8 | CSM-1B | `sbatch run_csm_1b.py --mode context_clone ...` | 30 min |
| 9 | Dia 1.6B | `sbatch run_dia.py --mode clone ...` | 25 min |
| 10 | Echo-TTS | `sbatch run_echo_tts.py --mode clone ...` | 20 min |
| 11 | OmniVoice | `sbatch run_omnivoice.py --mode clone ...` | 25 min |
| 12 | VoxCPM2 | `sbatch run_voxcpm2.py --mode ultimate_clone ...` | 30 min |
| 13 | Kokoro | Local (see Method 3) | 15 min |

---

## NO EXTRA SPACE NEEDED

The script runs on CSC Mahti GPU nodes - you don't need extra local disk space. 
All output saves directly to the repo, which is synced from the CSC compute nodes.

If you hit quota issues, check with:
```bash
# On Mahti:
csc-workspaces
df -h /scratch /proj /home
```

---

## QUICK TEST (Run one sample first)

To test before running all:
```bash
cd /path/to/textToAudioTTS/scripts/mahti

# Test Fish Speech with shortest input first:
sbatch run_fish_speech.py \
    --input ../../experiments/tts_naturalness/inputs/naturalness_prompt.txt \
    --output ../../experiments/tts_naturalness/generated/all_samples/fish_test.wav \
    --half

# Check status:
sacct -u $USER
```

---

## AFTER GENERATION

1. **Listen and compare** - Use the `audiobook_ranking.md` criteria
2. **Compare against** your Brothers Karmatzov reference sample
3. **Note issues**: chunk boundaries, pacing drift, voice fatigue

Total new samples: ~90 (13 models × 7 input files). Run your top 3-5 models first.
