# Verification

## Local Checks

- Session Context MCP and Serena MCP were used before implementation.
- Read `/Users/ruwodda/.codex/instructions.md` and repo `AGENTS.md`.
- `copilot` CLI was checked and is not installed on this machine, so no Copilot subtask was launched.
- Confirmed existing Tolstoy workflow writes:
  - `youtube_captions/*.json3`
  - `*Transcript.json`
  - `*Transcript.txt`
  - `video_work/chunks`
  - `video_work/chunk_audio`
  - `*_neuralsurfer.mp4`

## Playlist Checks

Command:

```bash
uvx yt-dlp --flat-playlist --dump-single-json 'https://www.youtube.com/playlist?list=PLdk-d5njBw7laBIdDCpqxse1ToP9oNis1'
```

Result:

- Playlist title: `Fiction - Russian Literature`
- Entries: 9
- Anna Karenina entries: 4
- Filtered part-order playlist written to `playlist_anna_karenina_4parts.json`.

Detailed `yt-dlp --dump-json` checks found YouTube chapter metadata for all four parts:

- Part 1: 69 chapters
- Part 2: 55 chapters
- Part 3: 65 chapters
- Part 4: 50 chapters

## Code Checks

Commands:

```bash
python3 -m py_compile scripts/process_youtube_hoys_audiobook.py
python3 scripts/process_youtube_hoys_audiobook.py --playlist-json tasks/anna_karenina_youtube_active_line_20260526/playlist_anna_karenina_4parts.json --indices 1-4 --output-root generated/anna_karenina_youtube_active_line_20260526 --use-youtube-chapters --transcript-source youtube-auto --dry-run
```

Result:

- `py_compile` passed.
- Dry run selected exactly 4 items with slugs:
  - `annaKareninaPart1Of4`
  - `annaKareninaPart2Of4`
  - `annaKareninaPart3Of4`
  - `annaKareninaPart4Of4`

## Pending Cluster Verification

- First Slurm array submission `6726509` failed fast for all four tasks.
- Dependent join job `6726513` was cancelled because the array failed.
- Root cause from all four logs: `/projappl/project_2010549/textToAudioTTS-mahti/.locks/audiobook-video-yt-dlp.lock: Disk quota exceeded`.
- Fix: set `MAHTI_LOCK_ROOT="$REPO_ROOT/.locks"` in both Slurm scripts before sourcing Mahti common/bootstrap code.
- Second Slurm array submission `6726518` failed fast for all four tasks.
- Dependent join job `6726522` was cancelled because the array failed.
- Root cause from all four logs: `Disk quota exceeded` while pip tried to install `yt-dlp` into `/projappl/project_2010549/textToAudioTTS-mahti/venvs/audiobook-video/lib/python3.12/site-packages/yt_dlp`.
- Fix: vendor `yt-dlp` under scratch at `$REPO_ROOT/.vendor/yt-dlp` and expose a scratch wrapper at `$REPO_ROOT/.local/bin/yt-dlp`.
- Third Slurm array submission `6726533` failed fast for all four tasks.
- Dependent join job `6726537` was cancelled because the array failed.
- Root cause from all four logs: Python subprocess could not resolve the `yt-dlp` executable even though the scratch wrapper existed after setup.
- Fix: update `scripts/process_youtube_hoys_audiobook.py` to call `yt-dlp` through a helper that falls back to `sys.executable -m yt_dlp` when the executable is not on `PATH`.
- Remote check passed with the Mahti bootstrap environment:

```bash
python -m py_compile scripts/process_youtube_hoys_audiobook.py
python -m yt_dlp --version
```

Result: `yt-dlp 2026.03.17`.

- Need resubmit the array and dependent join job, then verify Slurm state, logs, summaries, and MP4 outputs.

## Active Submission

Submitted after fixes:

- Render array: `6726546`
- Dependent join job: `6726550`

Health check at roughly 3.5 minutes runtime:

- `6726546_1`, `_2`, `_3`, `_4`: `RUNNING`, exit code `0:0`.
- `6726550`: `PENDING (Dependency)`.
- Separate Qwen GPU job `6720307`: still `PENDING (Priority)`.
- Output root size: `6.7G`.
- Logs showed no `Traceback`, `ERROR`, or `RuntimeError` matches for the active submission.
- Metadata, raw audio, caption/transcript/chunk, 16 kHz WAV, and chunk-audio stages were progressing.
- Part 4 had already reached the active-line video render command:

```bash
make_autoscroll_video.py --show-active-line --active-line-timing asr --active-line-asr-model small.en --active-line-asr-device cpu --video-codec libx264 --video-preset veryfast --video-crf 18
```

Status: implemented and running, not complete yet.

## 2026-05-27 13:03 EEST completion check

- Render array `6726546` completed successfully for all four parts with exit `0:0`.
- Dependent join job `6726550` completed successfully with exit `0:0`.
- Output root on Mahti: `/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/anna_karenina_youtube_active_line_20260526`.
- Output root size: `50G`.
- Verified MP4 artifacts:
  - Part 1: `annaKareninaPart1Of4_neuralsurfer.mp4`, `5,420,339,322` bytes.
  - Part 2: `annaKareninaPart2Of4_neuralsurfer.mp4`, `4,627,867,225` bytes.
  - Part 3: `annaKareninaPart3Of4_neuralsurfer.mp4`, `5,309,181,202` bytes.
  - Part 4: `annaKareninaPart4Of4_neuralsurfer.mp4`, `3,312,465,764` bytes.
  - Joined full video: `annaKarenina_youtube_active_line_full.mp4`, `18,673,694,849` bytes.
- Joined full MP4 probe with the audiobook-video `ffmpeg` environment reported `Duration: 41:03:03.00`, video `1920x1080` at `24 fps`, and mono AAC audio at `16000 Hz`.
- Current successful render/join logs showed no `Traceback`, `ERROR`, `RuntimeError`, failed/cancelled state, quota, or no-space matches.

## Local resumable download

- Created local folder: `outputs/video_downloads/YybWtOjzodA_1080p_20260527`.
- Download target: `https://www.youtube.com/watch?v=YybWtOjzodA`.
- Plain `nohup` attempts were killed after the shell exited, so the download was relaunched inside detached `screen` session `ak_ytdlp_YybWtOjzodA`.
- Resumable command uses `yt-dlp` through `uvx` with `--ignore-config`, `--continue`, `--part`, `--retries infinite`, and `--fragment-retries infinite`.
- At 2026-05-27 12:57 EEST, the screen session was detached and active; `YybWtOjzodA.f137.mp4.part` had reached `637M`, about `6.2%` of the `10.05GiB` 1080p video stream.
