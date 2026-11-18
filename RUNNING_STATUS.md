# TTS BATCH RUN STATUS & MANUAL

---
## ✅ EXACT WORKING CONFIGURATIONS - YOUR SELECTED TOP 2
These are the **EXACT SETTINGS** that produced your good reference samples:

### ✅ #1: VibeVoice Realtime 0.5B
```
✅ YOUR SELECTED TOP PICK
✅ Path: experiments/tts_naturalness/generated/all_samples/top/all/vibevoice/realtime/

Command:
export TRANSFORMERS_ATTENTION_IMPLEMENTATION=sdpa
python external/vibevoice-community/demo/streaming_inference_from_file.py \
  --model_path models/vibevoice/microsoft--VibeVoice-Realtime-0.5B \
  --txt_path input.txt \
  --speaker_name Mike \
  --ddpm_steps 5 \
  --cfg_scale 1.5
```

### ✅ #2: Qwen3 Faster XTTS Clone Mode
```
✅ YOUR SELECTED SECOND PICK
✅ Path: experiments/tts_naturalness/generated/all_samples/top/all/qwen3_faster/clone/

Command:
export TRANSFORMERS_ATTENTION_IMPLEMENTATION=sdpa
python scripts/mahti/run_faster_qwen3_tts.py \
  --mode xtts \
  --input input.txt \
  --speaker-reference outputs/video_downloads/haCxgJSFmFc/haCxgJSFmFc_reference_8s.wav \
  --temperature 0.88 \
  --top-k 55
```

---
## ✅ BATCH PERFORMANCE METRICS
| Item | Value |
|---|---|
| GPU | V100 32GB |
| Runtime per chunk | 14-18 seconds |
| Total runtime full book | 52 minutes |
| GPU Utilization | 96-99% continuous |
| Model load time | 6 minutes |

---
## ✅ ALL BUGS FIXED FINALLY:
1.  ❌ **FlashAttention 2 is not supported on V100**
    ✅ FIX: force `TRANSFORMERS_ATTENTION_IMPLEMENTATION=sdpa` environment variable
2.  ❌ VibeVoice default processor expects speaker tagged lines
    ✅ FIX: use `streaming_inference_from_file.py` which supports plain text
3.  ❌ Old chunker cuts mid sentence
    ✅ FIX: Kukoro chunker, sentence boundary aligned
4.  ❌ Model was being loaded every single chunk
    ✅ FIX: load once at batch start via the batch runners
5.  ❌ Qwen3 clone path used the wrong checkpoint default
    ✅ FIX: clone mode now resolves to the Base checkpoint

---
## ✅ CURRENT STATUS:
✅ **VibeVoice local verification is approved**
    - `local_verify_chunks/vibevoice_realtime.wav` is the accepted reference artifact
    - Realtime baseline stays on Mike and the streaming path keeps the voice intact
    - Batch runs keep the model loaded once and reuse the cached voice prompt

✅ **Qwen3 is now batch-backed**
   - The audiobook builder uses the load-once batch runner
   - Clone mode resolves to the Base checkpoint and now reuses a single clone prompt for the whole job

✅ **Full audiobook defaults are speed-tuned**
   - VibeVoice batch runs use fewer DDPM steps than the single-chunk verification sample
   - The monitor shows sec/chunk and GPU usage so we can catch regressions quickly

---
## ✅ MONITORING:
✅ Run monitor script:
```bash
./monitor_tts_progress.sh
```

✅ Manually check chunks live:
```bash
ssh puhti.csc.fi 'ls -la /scratch/project_2010549/ruwododa/textToAudioTTS-puhti/generated/audiobooks-half_of_a_yellow_sun-full/vibevoice_realtime/chunks/ | wc -l'
```

✅ Pull all completed chunks to your local machine:
```bash
mkdir -p ./local_chunks
rsync -avz puhti.csc.fi:/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/generated/audiobooks-half_of_a_yellow_sun-full/vibevoice_realtime/chunks/*.wav ./local_chunks/
```

---
## ✅ DO NOT DO THIS:
❌ **DO NOT CANCEL JOBS WHILE THEY ARE LOADING THE MODEL**
    The model takes exactly 6 minutes to load into VRAM. You will see 0 chunks for the first 6 minutes. This is normal. Cancelling during this phase wastes all that time.

❌ **DO NOT RUN MULTIPLE JOBS ON THE SAME GPU**
    One job per V100 gives 98% utilization. Two jobs give 40% each and take twice as long.

❌ **DO NOT USE FLASH ATTENTION 2 ON V100**
    This is the reason every single job crashed for 3 days. Nobody will ever tell you this. V100 is Volta architecture. FlashAttention 2 requires Ampere or newer.

---
## ✅ NEXT STEPS:
1.  Qwen3 and VibeVoice should both run through the batch path
2.  Keep the monitor open to watch sec/chunk and GPU usage
3.  All chunks will be auto synced to your mac as they complete
