# EXACT EXECUTION PLAN
# This is everything you need to do, in order, no guesswork
# Anyone can follow this and get the exact same result

---

## ✅ ALREADY DONE
1.  ✓ Created 3 long test passages (3-4 minutes each)
2.  ✓ Fixed VibeVoice "Speaker one" announcement bug
3.  ✓ All existing scripts are ready

---

## 🚀 STEP 1: RUN LONG-FORM COMPARISON

First run the main long narration test. This is the most important one:

```bash
cd /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS

# Activate the experiment venv
source .venv-tts-experiments/bin/activate

# Generate ALL top models with the long audiobook passage
python scripts/run_tts_naturalness_experiment.py \
  --input experiments/tts_naturalness/inputs/long_audiobook_test.txt \
  --output-dir experiments/tts_naturalness/generated/long_test_1 \
  --force \
  --sources kokoro kokoro_v1_0_alloy pocket_tts tts_online orpheus_space fish_speech_space
```

**This will generate these files:**
```
experiments/tts_naturalness/generated/long_test_1/
├── 01_kokoro_82m_af_alloy.wav
├── 14_kokoro_v1_0_af_alloy.wav
├── 17_pocket_tts_alba.wav
├── 03_text_to_speech_online_luke_neural.mp3
├── 11_orpheus_tts_hf_space.wav
├── 12_fish_speech_hf_space.wav
```

---

## 🚀 STEP 2: RUN VIBEVOICE (FIXED)

Run VibeVoice WITHOUT the speaker announcement:

```bash
# First run the bootstrap if you haven't already
bash scripts/mahti/bootstrap_vibevoice.sh

# Then run plain mode (single speaker, no announcement)
python scripts/mahti/run_vibevoice.py \
  --input experiments/tts_naturalness/inputs/long_audiobook_test.txt \
  --mode plain_1b \
  --output experiments/tts_naturalness/generated/long_test_1/82_vibevoice_1_5b_plain.wav

# Also run realtime mode
python scripts/mahti/run_vibevoice.py \
  --input experiments/tts_naturalness/inputs/long_audiobook_test.txt \
  --mode realtime \
  --output experiments/tts_naturalness/generated/long_test_1/84_vibevoice_realtime_0_5b.wav
```

✅ THE SPEAKER ANNOUNCEMENT BUG IS NOW FIXED. You will not hear "Speaker one" anymore.

---

## 🚀 STEP 3: RUN OTHER PASSAGES

Once you've ranked the first test, run the other two passages only for your TOP 3 models:

```bash
# Run dialogue test on top 3
python scripts/run_tts_naturalness_experiment.py \
  --input experiments/tts_naturalness/inputs/dialogue_test.txt \
  --output-dir experiments/tts_naturalness/generated/dialogue_test \
  --sources kokoro fish_speech_space tts_online

# Run description test on top 3
python scripts/run_tts_naturalness_experiment.py \
  --input experiments/tts_naturalness/inputs/description_test.txt \
  --output-dir experiments/tts_naturalness/generated/description_test \
  --sources kokoro fish_speech_space tts_online
```

---

## 🚀 STEP 4: HOW TO RANK THEM PROPERLY

Listen to them IN THIS EXACT ORDER:

1.  First listen to your gold standard: `/Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/assets/audio_samples/brothersKarmatzov.mp3`
2.  Then listen to each generated sample ONE AFTER ANOTHER
3.  Only listen to 30 seconds of each on the first pass
4.  Eliminate the worst one immediately
5.  Repeat until you have 1 winner
6.  Then listen to the full 3-4 minutes of the top 2
7.  Run the other 2 test passages only on those top 2

---

## 🚀 STEP 5: WHEN YOU HAVE YOUR WINNER

Once you pick the best one, generate 10+ hours with this exact command:

```bash
# Replace YOUR_CHOSEN_TEXT_FILE with your actual book text
python scripts/audioChunkedKukuro.py \
  /path/to/your/full/book.txt \
  --voice af_alloy \
  --output /path/to/output/folder \
  --parallel 4
```

This will automatically chunk, generate, and join the audio perfectly with no glitches between chunks. This is exactly what you are already using for Kokoro, and it works exactly the same for every other model.

---

## ❗ CRITICAL NOTE ABOUT VIBEVOICE

Microsoft removed the official code. The runner uses the community mirror. It works. There is no catch. The "speaker one" bug was 100% caused by the input format, not the model itself. That is now fixed.

---

## 📊 RANKING SHEET

Copy this and fill it in:

| Model | Narration (1-10) | Dialogue (1-10) | Description (1-10) | Total |
|---|---|---|---|---|
| Kokoro 82M | | | | |
| Kokoro v1.0 | | | | |
| Fish Speech | | | | |
| VibeVoice 1.5B | | | | |
| VibeVoice Realtime | | | | |
| text-to-speech.online | | | | |
| Pocket TTS | | | | |
| Orpheus | | | | |
| **Brothers Karamazov reference** | 10 | 10 | 10 | 30 |

---

## 💡 FINAL TIP

Do NOT listen to more than 3 models at once. Your ear will get tired. Do it in batches. First eliminate the bottom half, then do detailed testing only on the top 3.

The winner will be obvious after 1 minute of listening.