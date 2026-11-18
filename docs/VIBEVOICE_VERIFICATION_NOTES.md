# VibeVoice verification notes

## Baseline

- Keep the realtime baseline on `Mike`.
- Do not change the voice just to hide the intro artifact.

## Intro artifact

- The unwanted music / SFX appears only at the start of the first realtime chunk.
- The fix is a post-generation trim in `scripts/mahti/run_vibevoice.py`.
- The trim applies only to `chunk_0000_*` inputs unless `--trim-leading-seconds` is set explicitly.
- Upstream realtime inference uses plain text plus a cached `.pt` voice prompt, not `Speaker 1:` tags.
- The upstream repo notes that background music / sound effects can be triggered by prompt text and voice prompts; keep the prompt clean and stay on the streaming path for realtime.
- Upstream streaming calls `VibeVoiceStreamingProcessor.from_pretrained(...)`, `VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(...)`, `process_input_with_cached_prompt(...)`, `model.generate(...)`, and `processor.save_audio(...)`.

## Verified outputs

- `local_verify_chunks/qwen3_faster_clone.wav`
- `local_verify_chunks/vibevoice_realtime.wav`
- Both model chunk dirs contain 2 chunk WAVs.
- The existing `vibevoice_realtime.wav` artifact predates the streaming-path fix and must be regenerated before it is treated as verified.

## Monitoring

- `monitor_tts_progress.sh` now reports active Puhti jobs, chunk speed in sec/chunk, ETA, latest chunk age, and best-effort GPU usage.
- The audiobook builder now uses the load-once batch runners for both Qwen and VibeVoice.

## What made it work

- Use the upstream VibeVoice streaming path: `VibeVoiceStreamingProcessor`, `VibeVoiceStreamingForConditionalGenerationInference`, cached voice prompts, and `processor.save_audio(...)`.
- Keep realtime on the Mike preset and plain text input.
- Keep the model loaded once per batch job instead of launching a fresh process per chunk.
- Use the batch runner directly from `scripts/mahti/` so Puhti resolves the updated code path.

## What did not work

- The old per-chunk subprocess launcher.
- The stale Puhti source tree that still pointed at `scripts/run_vibevoice_batch_optimized.py` instead of `scripts/mahti/run_vibevoice_batch_optimized.py`.
- The speaker-tag workaround for realtime.
