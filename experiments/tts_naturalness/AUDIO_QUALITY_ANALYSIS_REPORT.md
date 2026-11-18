# TTS Audio Quality Analysis Report
## Generated: April 12, 2026
## Location: /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/experiments/tts_naturalness/generated/all_samples/

---

## 🎯 DEFINITIVE CONCLUSION

### **RUBBISH FILES (Pure Garbage - 2 files)**
These files are **objectively unusable** and should be excluded from any evaluation:

| File | Issue | Evidence |
|------|-------|----------|
| **76_csm_1b_context_clone.wav** | 51% SILENCE + EXTREMELY LOW VOLUME | Mean: -33.3dB (max: -12.6dB), 5.1 seconds of pure silence at end (51% of file), Duration: 10s |
| **67_fish_speech_s2_pro_hacx.wav** | EXTREMELY LOW VOLUME | Mean: -31.8dB (max: -11.3dB), Effectively inaudible whisper, Duration: 14.1s |

---

## 📊 FULL ANALYSIS SUMMARY

### **Total Files Analyzed: 50**

| Category | Count | Percentage |
|----------|-------|------------|
| 🔴 **RUBBISH** (unusable) | 2 | 4% |
| 🟠 **POOR** (degraded) | 18 | 36% |
| 🟢 **OK** (good quality) | 30 | 60% |

---

## 🔴 RUBBISH FILES (MUST EXCLUDE)

### 1. `76_csm_1b_context_clone.wav` - CRITICAL FAILURE
```
Duration:        10.00 seconds
Mean Volume:     -33.3 dB (CRITICALLY LOW)
Max Volume:      -12.6 dB
Silence:         5.13 seconds (51.3% of file!)
Longest Silence: 5.10 seconds (from 4.9s to 10s)
File Size:       469KB (suspiciously small)
```
**Verdict:** The file has a **5+ second trailing silence** and extremely low volume throughout. It's 50% empty. The CSM 1B model failed to generate proper audio for this clone attempt.

### 2. `67_fish_speech_s2_pro_hacx.wav` - CRITICAL FAILURE
```
Duration:        14.12 seconds
Mean Volume:     -31.8 dB (CRITICALLY LOW)
Max Volume:      -11.3 dB
Silence:         0%
File Size:       1.2MB
```
**Verdict:** The file plays audio but at **barely audible whisper level** (-31.8dB mean). The Fish Speech S2 Pro model's HACX (clone) output is completely unusable. The audio is there but effectively silent.

---

## 🟠 POOR QUALITY FILES (USABLE BUT DEGRADED)

These files have issues but can still be evaluated:

### Low Volume Files (Mean < -25dB)
| File | Mean Volume | Notes |
|------|-------------|-------|
| 74_voxcpm2_clone.wav | -29.7 dB | Quiet but audible |
| 75_voxcpm2_ultimate_clone.wav | -29.7 dB | Quiet but audible |
| 61_omnivoice_clone_hacx.wav | -29.8 dB | Quiet but audible |
| 65_gpt_sovits_v2_hacx.wav | -29.3 dB | Quiet but audible |
| 62_qwen3_tts_voice_clone_hacx.wav | -29.4 dB | Quiet but audible |
| 63_faster_qwen3_tts_voice_clone_hacx.wav | -28.8 dB | Quiet but audible |
| 30_kyutai_tts_en.wav | -28.1 dB | Quiet but audible |
| 32_csm_1b_plain.wav | -28.1 dB | Quiet but audible |
| 21_orpheus_tts_3b_leah.wav | -27.2 dB | Quiet but usable |
| 64_f5_tts_v1_base_hacx.wav | -27.1 dB | Quiet but usable |
| 80_chatterbox_turbo_plain.wav | -26.6 dB | Slightly quiet |
| 81_chatterbox_turbo_clone.wav | -26.0 dB | Slightly quiet |
| 01_kokoro_82m_af_alloy.wav | -26.9 dB | Slightly quiet |
| 14_kokoro_v1_0_af_alloy.wav | -26.9 dB | Slightly quiet |
| 68_omnivoice_design.wav | -25.8 dB | Marginally low |
| 82_vibevoice_1_5b_plain.wav | -25.7 dB | + ABNORMAL 38s duration |
| 84_vibevoice_realtime_0_5b.wav | -25.5 dB | Slightly quiet |
| 70_qwen3_tts_custom_voice.wav | -25.1 dB | Slightly quiet |

### Pattern Identified: HACX = LOW VOLUME
**All files with "_hacx" suffix have significantly lower volume than their non-hacx counterparts:**

| Model | Normal | HACX Version | Difference |
|-------|--------|--------------|------------|
| OmniVoice Clone | -17.6 dB | -29.8 dB | **-12.2 dB** ❌ |
| GPT-SoVITS v2 | -18.3 dB | -29.3 dB | **-11.0 dB** ❌ |
| Fish Speech S2 Pro | N/A | -31.8 dB | **CRITICALLY LOW** ❌ |

**Conclusion:** The HACX (voice cloning) pipeline has a **systematic volume normalization bug** that reduces output by ~12dB.

### Abnormal Duration Files
| File | Duration | Issue |
|------|----------|-------|
| 82_vibevoice_1_5b_plain.wav | 38.1s | **2.5x longer** than typical (10-15s), 32% silence |

---

## 🟢 GOOD QUALITY FILES (30 files)

These files have proper volume (-15dB to -23dB mean), minimal silence, and normal duration:

**Excellent Examples:**
- `24_xtts_v2.wav` (-15.3 dB, excellent volume)
- `33_dia_1_6b_dialogue.wav` (-16.2 dB)
- `13_parler_tts_hf_space.wav` (-16.7 dB)
- `23_f5_tts_v1_base.wav` (-17.1 dB)
- `27_omnivoice_clone.wav` (-17.6 dB)
- `22_fish_speech_s2_pro.wav` (-18.4 dB)
- `17_pocket_tts_alba.wav` (-19.1 dB)
- `25_gpt_sovits_v2.wav` (-18.3 dB)

---

## 📋 RECOMMENDATIONS

### For Naturalness Evaluation:

1. **EXCLUDE these 2 files from evaluation:**
   - `76_csm_1b_context_clone.wav`
   - `67_fish_speech_s2_pro_hacx.wav`

2. **FLAG these for volume normalization before evaluation:**
   - All `*_hacx.wav` files (10-12dB boost needed)
   - `74_voxcpm2_clone.wav` and `75_voxcpm2_ultimate_clone.wav`

3. **TRIM or EXCLUDE:**
   - `82_vibevoice_1_5b_plain.wav` (trim to first 15 seconds, remove trailing silence)

### Technical Findings:

1. **CSM 1B Context Clone** - Complete failure, generates 50% silence
2. **Fish Speech S2 Pro HACX** - Critical volume bug, inaudible output
3. **HACX Pipeline** - Systematic ~12dB volume reduction across all models
4. **VibeVoice 1.5B** - Excessive duration, possible generation loop

---

## 🔬 VERIFICATION METHOD

Analysis performed using:
- `ffprobe` for format validation (no corruption detected in any file)
- `ffmpeg volumedetect` for precise volume measurement
- `ffmpeg silencedetect` for silence period detection
- Python script for statistical analysis and classification

**Thresholds used:**
- RUBBISH: mean_volume < -30dB OR silence_ratio > 50%
- POOR: mean_volume < -25dB OR silence_ratio > 30%
- WARNING: duration > 20s OR duration < 5s
- OK: All other files

---

*Report generated by automated audio analysis pipeline*
