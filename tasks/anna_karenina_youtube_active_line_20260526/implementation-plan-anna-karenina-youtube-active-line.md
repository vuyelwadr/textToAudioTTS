# Implementation Plan

## Context

- Existing workflow: `scripts/process_youtube_hoys_audiobook.py`.
- Existing Tolstoy outputs in `library/tolstoy/*` store raw YouTube audio, `youtube_captions/*.json3`, faster-whisper transcript JSON/TXT, chunk text/audio, joined audio, summaries, and rendered `_neuralsurfer.mp4` videos.
- The new playlist contains 9 videos; only entries 6-9 are Anna Karenina, and the playlist order is part 4, part 3, part 2, part 1.
- A filtered playlist was created at `tasks/anna_karenina_youtube_active_line_20260526/playlist_anna_karenina_4parts.json` in part order 1-4.

## Decisions

- Use the existing YouTube audiobook render pipeline rather than creating a parallel renderer.
- Use YouTube-derived transcript text via the new opt-in `--transcript-source youtube-auto` mode for this run.
- Download and retain YouTube caption JSON for traceability.
- Use YouTube metadata chapter markers via the new opt-in `--use-youtube-chapters` mode.
- Run on Mahti CPU as a Slurm array with one task per part. Local disk had too little free space for all intermediates plus joined MP4.
- Leave the separate Anna Karenina Qwen GPU job queued.

## Playlist

Part order:

1. `jbsadC5Qfgs` - `42458s` - Part 1 of 4
2. `Leb4fk_bpYQ` - `36192s` - Part 2 of 4
3. `Jjo3oq52sFk` - `41465s` - Part 3 of 4
4. `TXTIJOH93ww` - `25988s` - Part 4 of 4

Chapter metadata discovered locally with `yt-dlp`:

- Part 1: 69 chapters, `Part 1, Chapter 1` through `Part 2, Chapter 35`
- Part 2: 55 chapters, `Part 3, Chapter 1` through `Part 4, Chapter 23`
- Part 3: 65 chapters, `Part 5, Chapter 1` through `Part 6, Chapter 32`
- Part 4: 50 chapters, `Part 7, Chapter 1` through `Part 8, Chapter 19`

## Implementation Steps

- [x] Recover repo and task memory with Session Context MCP and Serena MCP.
- [x] Inspect existing Tolstoy YouTube transcript/output layout.
- [x] Fetch the corrected playlist metadata with `yt-dlp`.
- [x] Create a filtered four-entry Anna Karenina playlist in part order.
- [x] Patch `scripts/process_youtube_hoys_audiobook.py` to support:
  - unique short Anna part slugs,
  - YouTube metadata chapter extraction,
  - chapter-aware chunk boundaries,
  - YouTube auto-caption transcript source.
- [x] Submit Mahti CPU array render for all four parts.
- [x] Submit dependent join job for a fifth full-length MP4 if all four parts complete and scratch space is sufficient.
- [ ] Verify completed Slurm state, logs, summaries, chapter counts, and produced MP4 artifacts.

## Active Jobs

- Render array: `6726546`, tasks `1-4`, running on Mahti CPU `small`.
- Dependent join: `6726550`, pending on `afterok:6726546`.
- Separate Qwen GPU Anna job: `6720307`, still pending and intentionally left queued.

