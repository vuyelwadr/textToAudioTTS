# Anna Karenina Qwen Video Plan

Status: Implemented but not verified

## Requirements

- Use the same successful 1984 Qwen audiobook/video settings.
- Produce an active-line moving-cursor video.
- Queue the Mahti job.
- Keep GPU work focused on text chunking, TTS, and chunk audit.
- Submit final MP4 rendering to a CPU job.

## 1984 settings to preserve

- Model: `qwen3_faster_clone`
- Reference audio: `generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.wav`
- Reference transcript: `generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.transcript.txt`
- Chunk size: `1800`
- Minimum chunk size: `200`
- Chunk workers: `16`
- Audit chunks: enabled
- Audit ASR model/device/workers: `small.en` / `cuda` / `auto`
- Active-line timing: `asr`
- Video codec/preset/CRF: `libx264` / `veryfast` / `18`

## Anna Karenina run

- Input: `library/annaKarenina/AnnaKarenina.txt`
- Output root: `/scratch/project_2010549/$USER/textToAudioTTS-mahti/generated/audiobooks-anna_karenina-qwen3_full_audited_20260525`
- GPU job: `gpusmall`, `gpu:a100:1,nvme:900`, `32` CPUs, `96G`, `1-12:00:00`
- Video job: submitted by wrapper to Mahti CPU `small`, `32` CPUs, `3-00:00:00`
