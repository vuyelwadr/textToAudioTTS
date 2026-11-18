# LONG-FORM AUDIOBOOK TTS RANKING PLAN

**Objective**: Generate and rank TTS samples using genuinely long, meaningful passages to find the best engine for 10+ hour audiobook production.

---

## PROBLEM STATEMENT

1. **Current samples too short**: The existing test passages (300-500 words) cannot reveal issues that only appear in longer audio:
   - Pacing drift over paragraphs
   - Voice fatigue/monotony after 5+ minutes
   - Chunk-boundary artifacts
   - Natural breathing rhythm over long passages

2. **VibeVoice "Speaker one" bug**: The VibeVoice model announces "Speaker one" before audio plays. This is unacceptable for audiobooks. Root cause: the code adds `Speaker 1:` prefix for single-speaker mode.

3. **No definitive ranking**: With 54 samples but short texts, you cannot meaningfully rank for audiobook use. Need 10+ minute passages to properly evaluate.

---

## STEP 1: CREATE LONG TEST PASSAGES (~1500-2500 words each)

Create these new files in `experiments/tts_naturalness/inputs/`:

### Passage A: Literary Prose (Narrative)
**File**: `narrative_test_long.txt` (~2000 words)

Use Chapter 1 of a public domain book. Recommendation: **Proust - Swann's Way** (the opening "Combray" section is excellent for narration testing, available from Project Gutenberg).

```bash
# Download Proust chapter:
curl -s "https://www.gutenberg.org/cache/epub/7178/pg7178.txt" | sed -n '/CHAPTER I/,/CHAPTER II/p' | head -3000 > experiments/tts_naturalness/inputs/narrative_test_long.txt
```

**Why Proust**: Complex sentence structures, varied pacing, introspection, requires natural delivery over 15+ minutes.

### Passage B: Dialogue-Heavy Fiction
**File**: `dialogue_test_long.txt` (~1500 words)

Use a scene with multiple characters. Recommendation: **Dashiell Hammett - The Maltese Falcon** Chapter 1-2 (public domain).

```bash
# Download Hammett:
curl -s "https://www.gutenberg.org/ebooks/34631.txt.utf-8" | sed -n '/Chapter 1/,/Chapter 2/p' | head -2500 > experiments/tts_naturalness/inputs/dialogue_test_long.txt
```

**Why Hammett**: Sharp dialogue, multiple speakers, noir tone - tests speaker differentiation and natural speech patterns.

### Passage C: Descriptive/Atmospheric
**File**: `descriptive_test_long.txt` (~1500 words)

Use vivid descriptive prose. Recommendation: **Sherwood Anderson - Winesburg, Ohio** "The Book of The Grotesque" (public domain).

```bash
# Download Anderson:
curl -s "https://www.gutenberg.org/ebooks/11360.txt.utf-8" | sed -n '/THE BOOK OF THE GROTESQUE/,/SOUNDING IN THE DARKNESS/p' | head -2500 > experiments/tts_naturalness/inputs/descriptive_test_long.txt
```

**Why Anderson**: Rich atmosphere, varied rhythm, tests emotional consistency and descriptive naturalness.

---

## STEP 2: FIX VIBEVoice "Speaker One" BUG

**File to edit**: `scripts/mahti/run_vibevoice.py`

The issue is in function `_write_input_file()` at line 207. Current code:

```python
# Current (WRONG) - adds speaker prefix even for single speaker
content = f"Speaker 1: {text}\n"
```

Fix - for single-speaker narration, pass plain text WITHOUT speaker tags:

```python
# FIX: For plain_1b (single speaker), DO NOT add Speaker 1: prefix
# This prefix causes the model to literally say "Speaker one" in the audio
if mode == "plain_1b":
    content = f"{text}\n"
else:
    content = f"Speaker 1: {text}\n"
```

Or more specifically (matching the existing comment):

```python
# Line 207-209 - the comment is correct, implementation may be wrong
# DO NOT add Speaker 1: prefix - this is what causes the model to announce "Speaker one" out loud
# For single-speaker narration just pass plain text with no speaker tags at all
content = f"{text}\n"
```

**After fix**: Regenerate VibeVoice samples:
```bash
cd scripts/mahti
# Run in vibevoice venv
sbatch run_vibevoice.py --mode plain_1b --input ../../../experiments/tts_naturalness/inputs/narrative_test_long.txt
```

---

## STEP 3: DETERMINE WHICH MODELS TO TEST

### Top Candidates for Audiobook (based on current samples + leaderboard):

**Tier 1 - Must Test** (best quality potential):
1. **Kokoro 82M** - Best local Mac option, already working
2. **Fish Speech S2 Pro** - Top open-weights on Artificial Analysis
3. **Parler TTS Large v1** - Already in repo
4. **MOSS-TTS Local** - 1.7B transformer, voice cloning
5. **Spark-TTS 0.5B** - Good quality, fast
6. **VibeVoice 1.5B** - Large model, but needs fix
7. **Higgs Audio V2** - Best emotion/expressiveness (but needs A100)

**Tier 2 - Worth Testing**:
8. **F5-TTS** - Strong speaker consistency
9. **GPT-SoVITS** - Voice cloning
10. **CSM-1B** - Context-aware (unique)
11. **Dia 1.6B** - Dialogue naturalness

### Recommended Test Matrix:

| Model | Passage A | Passage B | Passage C |
|-------|-----------|-----------|-----------|
| Kokoro 82M | ✓ | ✓ | ✓ |
| Fish S2 Pro | ✓ | ✓ | ✓ |
| Parler Large | ✓ | - | ✓ |
| MOSS-TTS | ✓ | ✓ | - |
| Spark-TTS | ✓ | - | ✓ |
| VibeVoice 1.5B (FIXED) | ✓ | - | ✓ |
| F5-TTS | ✓ | ✓ | - |
| GPT-SoVITS | ✓ | - | - |
| CSM-1B | - | ✓ | - |
| Dia 1.6B | - | ✓ | - |

Minimum: **15 samples** across 3 passages with 8-10 models.

---

## STEP 4: GENERATE LONG SAMPLES

### Local Mac Generation (if model fits in RAM/GPU):

**Kokoro** (best local):
```bash
# Activate venv
source .venv-tts-experiments/bin/activate

# Generate with Kokoro - use long passage
python scripts/run_tts_naturalness_experiment.py \
  --engine kokoro \
  --voice alloy \
  --input experiments/tts_naturalness/inputs/narrative_test_long.txt \
  --output experiments/tts_naturalness/generated/all_samples/kokoro_narrative_long.wav
```

### CSC Mahti Generation (for large models):

Create batch script `generate_audiobook_samples.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=tts_audiobook_long
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Load modules
module load python/3.11
module load cuda/12.4

# Activate environment
source ~/.bashrc
conda activate tts-experiments

# Generate each model/passage combination
# Example for Fish Speech:
python scripts/mahti/run_fish_speech.py \
  --input ../../../experiments/tts_naturalness/inputs/narrative_test_long.txt \
  --output ../../../experiments/tts_naturalness/generated/all_samples/fish_s2_pro_narrative.wav \
  --mode plain

# Repeat for each model in the test matrix
```

### Script for batch generation:

Create `scripts/generate_audiobook_ranking.sh`:

```bash
#!/bin/bash
set -e

INPUT_DIR="experiments/tts_naturalness/inputs"
OUTPUT_DIR="experiments/tts_naturalness/generated/all_samples"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# === NARRATIVE PASSAGE ===
echo "Generating narrative passage samples..."

# Kokoro (local)
source .venv-tts-experiments/bin/activate
python scripts/run_tts_naturalness_experiment.py \
  --engine kokoro --voice alloy \
  --input "$INPUT_DIR/narrative_test_long.txt" \
  --output "$OUTPUT_DIR/kokoro_narrative_long.wav"

# Fish Speech (CSC)
sbatch scripts/mahti/run_fish_speech.sh \
  --input "$INPUT_DIR/narrative_test_long.txt" \
  --output "$OUTPUT_DIR/fish_s2_pro_narrative.wav"

# Parler TTS Large (CSC)
sbatch scripts/mahti/run_parler_tts_large.sh \
  --input "$INPUT_DIR/narrative_test_long.txt" \
  --output "$OUTPUT_DIR/parler_large_narrative.wav"

# MOSS-TTS (CSC)
sbatch scripts/mahti/run_moss_tts.sh \
  --input "$INPUT_DIR/narrative_test_long.txt" \
  --output "$OUTPUT_DIR/moss_tts_narrative.wav"

# Spark-TTS (CSC)
sbatch scripts/mahti/run_spark_tts.sh \
  --input "$INPUT_DIR/narrative_test_long.txt" \
  --output "$OUTPUT_DIR/spark_tts_narrative.wav"

# VibeVoice 1.5B (CSC - after fix)
sbatch scripts/mahti/run_vibevoice.sh \
  --mode plain_1b \
  --input "$INPUT_DIR/narrative_test_long.txt" \
  --output "$OUTPUT_DIR/vibevoice_narrative.wav"

# F5-TTS (CSC)
sbatch scripts/mahti/run_f5_tts.sh \
  --input "$INPUT_DIR/narrative_test_long.txt" \
  --output "$OUTPUT_DIR/f5_tts_narrative.wav"

# === DIALOGUE PASSAGE ===
echo "Generating dialogue passage samples..."

# Kokoro
python scripts/run_tts_naturalness_experiment.py \
  --engine kokoro --voice alloy \
  --input "$INPUT_DIR/dialogue_test_long.txt" \
  --output "$OUTPUT_DIR/kokoro_dialogue_long.wav"

# Fish Speech
sbatch scripts/mahti/run_fish_speech.sh \
  --input "$INPUT_DIR/dialogue_test_long.txt" \
  --output "$OUTPUT_DIR/fish_s2_pro_dialogue.wav"

# CSM-1B (unique context-aware capability)
sbatch scripts/mahti/run_csm_1b.sh \
  --input "$INPUT_DIR/dialogue_test_long.txt" \
  --output "$OUTPUT_DIR/csm_1b_dialogue.wav"

# Dia 1.6B (dialogue-optimized)
sbatch scripts/mahti/run_dia.sh \
  --input "$INPUT_DIR/dialogue_test_long.txt" \
  --output "$OUTPUT_DIR/dia_dialogue.wav"

# === DESCRIPTIVE PASSAGE ===
echo "Generating descriptive passage samples..."

# Kokoro
python scripts/run_tts_naturalness_experiment.py \
  --engine kokoro --voice alloy \
  --input "$INPUT_DIR/descriptive_test_long.txt" \
  --output "$OUTPUT_DIR/kokoro_descriptive_long.wav"

# Fish Speech
sbatch scripts/mahti/run_fish_speech.sh \
  --input "$INPUT_DIR/descriptive_test_long.txt" \
  --output "$OUTPUT_DIR/fish_s2_pro_descriptive.wav"

# Parler Large
sbatch scripts/mahti/run_parler_tts_large.sh \
  --input "$INPUT_DIR/descriptive_test_long.txt" \
  --output "$OUTPUT_DIR/parler_large_descriptive.wav"

# Spark-TTS
sbatch scripts/mahti/run_spark_tts.sh \
  --input "$INPUT_DIR/descriptive_test_long.txt" \
  --output "$OUTPUT_DIR/spark_tts_descriptive.wav"

# VibeVoice (after fix)
sbatch scripts/mahti/run_vibevoice.sh \
  --mode plain_1b \
  --input "$INPUT_DIR/descriptive_test_long.txt" \
  --output "$OUTPUT_DIR/vibevoice_descriptive.wav"

echo "All batch jobs submitted. Check SLURM with: sacct -u $USER"
```

---

## STEP 5: LISTENING PROTOCOL FOR RANKING

### Create ranking spreadsheet: `experiments/tts_naturalness/audiobook_ranking.md`

Structure:

```markdown
# Audiobook TTS Ranking

## Evaluation Criteria (rate 1-5):
- **Naturalness**: Sounds like a human reading, not robotic
- **Pacing**: Appropriate speed, varies with content
- **Clarity**: Every word intelligible
- **Consistency**: Same quality throughout (no drift)
- **Expression**: Appropriate tone for content (not monotone)
- **No Artifacts**: No popping, clicks, weird pauses

## Passages
- A: Narrative (Proust) - ~15 min spoken
- B: Dialogue (Hammett) - ~10 min spoken  
- C: Descriptive (Anderson) - ~10 min spoken

## Results

| Model | Passage | Nat | Pace | Clar | Cons | Expr | Art | Notes |
|-------|---------|-----|------|------|------|------|-----|-------|
| Kokoro | A | | | | | | | |
| Fish S2 Pro | A | | | | | | | |
```

### Listening Workflow:

1. **First pass**: Listen to all samples for ONE passage (e.g., Narrative)
2. **Second pass**: Compare head-to-head between top candidates
3. **Third pass**: Listen to 5+ minute segments to check for fatigue/drift
4. **Final**: Rank top 3, note issues for each

---

## STEP 6: COST/TIME ESTIMATES

### For 15+ samples at ~10-15 minutes each:

| Model | Generation Time | Compute | Est. Cost |
|-------|-----------------|---------|-----------|
| Kokoro | ~20 min | Local | Free |
| Fish S2 Pro | ~30 min | CSC A100 | ~$0.50 |
| Parler Large | ~40 min | CSC A100 | ~$0.70 |
| MOSS-TTS | ~35 min | CSC V100 | ~$0.40 |
| Spark-TTS | ~20 min | CSC V100 | ~$0.25 |
| VibeVoice | ~45 min | CSC A100 | ~$0.80 |
| F5-TTS | ~25 min | CSC A100 | ~$0.45 |
| CSM-1B | ~30 min | CSC A100 | ~$0.50 |
| Dia 1.6B | ~25 min | CSC A100 | ~$0.45 |

**Total estimated CSC cost**: ~$5-8 for the full ranking run.

---

## STEP 7: POST-PROCESSING CHECKLIST

1. **Upsample if needed**: Some models output 16kHz (Spark-TTS needs upsampling)
   ```bash
   # Use LagoSRL or Flow-High for upsampling
   python -m lavasr --input spark_tts_narrative.wav --output spark_tts_narrative_24k.wav
   ```

2. **Normalize volumes**: Ensure consistent loudness across samples
   ```bash
   ffmpeg -i input.wav -filter:a "loudnorm=I=-16:TP=-1.5:LRA=11" output.wav
   ```

3. **Check for artifacts**: Listen for clicks, pops, weird silences

4. **Metadata**: Note which reference voice was used (if any) for cloning models

---

## SUMMARY: EXECUTION ORDER

1. **Create long passages** (Step 1)
2. **Fix VibeVoice bug** (Step 2)
3. **Define test matrix** (Step 3)
4. **Run generation** (Step 4)
5. **Listen and rank** (Step 5)
6. **Post-process** (Step 7)

Total time: ~1-2 days for generation, ~2-3 days for listening evaluation.

---

## REFERENCE: YOUR BROTHERS KARMATZOV SAMPLE

You mentioned `/Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/assets/audio_samples/brothersKarmatzov.mp3` is perfect.

**What makes it good** (analyze it):
- Pacing: Even, not rushed
- Natural pauses: At sentence/paragraph boundaries
- Emotional consistency: Appropriate for content
- No "read-out-loud" feel: Sounds like someone telling you a story
- Clarity: Every word understandable

**Use this as your benchmark**: When ranking, compare each TTS sample against this reference. Ask: "Does this sound 90% as good as the Brothers Karmatzov sample?"

---

## FILES TO CREATE/MODIFY

| File | Action |
|------|--------|
| `experiments/tts_naturalness/inputs/narrative_test_long.txt` | Create - Proust chapter |
| `experiments/tts_naturalness/inputs/dialogue_test_long.txt` | Create - Hammett chapter |
| `experiments/tts_naturalness/inputs/descriptive_test_long.txt` | Create - Anderson chapter |
| `scripts/mahti/run_vibevoice.py` | Fix line 207-209 |
| `scripts/generate_audiobook_ranking.sh` | Create - batch generation script |
| `experiments/tts_naturalness/audiobook_ranking.md` | Create - ranking spreadsheet |
| `experiments/tts_naturalness/generated/all_samples/*_narrative_long.wav` | Generate |
| `experiments/tts_naturalness/generated/all_samples/*_dialogue_long.wav` | Generate |
| `experiments/tts_naturalness/generated/all_samples/*_descriptive_long.wav` | Generate |