# Verification

## Planned Checks

- Extract TXT from EPUB with only Introduction through Conclusion.
- Write a JSON source-text audit with included sections, counts, hash, and forbidden section heading scan.
- Dry-run chunking locally or on Mahti to confirm the source file is accepted.
- Submit the full Qwen audiobook/video job on Mahti.
- Verify Slurm completion, joined WAV duration, MP4 duration, timing JSON, and chunk audit status.

## Current Status

- `PrometheusAtlas` full MP4 size: `6,432,804,968` bytes (`6.43 GB`, about `5.99 GiB`).
- Clean source TXT created at `library/unmaskingAutism/UnmaskingAutism.txt`.
- Source-text audit created at `library/unmaskingAutism/UnmaskingAutism_text_audit.json`.
- Source-text audit: `85,827` words, `1,063` paragraph blocks, `0` forbidden section heading matches.
- Local dry-run chunking with the old `4500` char chunk size: `118` Qwen chunks, estimated `571.9` minutes.
- Mahti Slurm job `6549062`: failed after producing all `118/118` chunk WAVs because pre-join chunk audit could not import `faster_whisper`.
- Fixed `scripts/mahti/bootstrap_audiobook_video.sh` and `scripts/mahti/run_qwen_audiobook_video.sh` so audit-enabled Qwen runs activate the working `audiobook-video` environment before `build_chunked_audiobook.py`.
- Verified on Mahti: `import faster_whisper` succeeds after the fixed bootstrap.
- Mahti resume Slurm job `6559812`: reused the `118/118` chunk WAVs, but the ASR audit correctly failed the run with `117` blocking truncation issues. Root cause: `4500` char chunks generated audio near the model cap and dropped text.
- Local dry-run chunking with the safer `2400` char chunk size: `225` chunks, `520,239` chunked chars, estimated `572.0` minutes.
- Mahti smoke Slurm job `6560969`: generated `3/3` chunks with `CHUNK_SIZE=2400`; ASR audit passed with `blocking_issue_count = 0`.
- Mahti full Slurm job `6561486`: running with `CHUNK_SIZE=2400`, `AUDIT_CHUNKS=1`, and `GENERATE_VIDEO=1`.
- Live check at `2026-04-27T08:57+03:00`: `155/225` chunk WAVs present, output tree `1.1G`, scratch free about `12G`, latest chunk logs show normal saved WAV durations and no failure signature.
- Mahti full Slurm job `6561486`: generated `225/225` chunks, then ASR audit failed with `7` real blocking issues at chunk indices `53, 78, 91, 93, 102, 132, 203`. These were not bypassed; audit showed truncation/low-match at the `163.04s` Qwen ceiling.
- Local dry-run chunking with `1800` char chunk size: `304` chunks, `520,148` chunked chars, estimated `572.0` minutes.
- Mahti targeted probe Slurm job `6570365`: re-chunked the seven failed segments at `1800` chars into `10` chunks; ASR audit passed with `blocking_issue_count = 0`.
- Removed failed/probe Mahti output trees to recover scratch space before the next full run.
- Mahti full Slurm job `6570794`: submitted with `CHUNK_SIZE=1800`, `AUDIT_CHUNKS=1`, `GENERATE_VIDEO=1`, and output dir `/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-unmasking_autism-qwen3_full_20260427_chunk1800`.
- Scheduler status at `2026-04-27T10:39+03:00`: job `6570794` is `PENDING` for `Priority`; `squeue --start` estimates start at `2026-04-27T15:43:15` on `g5101`; scratch free about `13G`.
