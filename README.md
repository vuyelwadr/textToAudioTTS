# textToAudioTTS

Audiobook and TTS pipeline for CSC Puhti and Mahti. This repo is set up to chunk long text, render each chunk with one or more TTS models, join the chunk WAVs into final audiobook files, and keep the result reproducible across clusters.

For live run status, see `RUNNING_STATUS.md`.

## What this repo is for

- Generate and compare audiobook-quality speech from long text.
- Run the heavy model stacks on CSC GPU nodes instead of locally.
- Keep final MP4 rendering on CPU partitions unless a run explicitly needs GPU-backed ASR timing.
- Keep verification outputs separate from full audiobook runs.
- Join chunked WAVs into a single final audio file once a model finishes.

## Main paths

- `library/` - per-book source text + YouTube audiobook outputs, e.g. `library/schopenhauer_parerga/`, `library/tolstoy/`, `library/schopenhauer_kallipolis/`
- `tasks/` - per-batch playlists/metadata (e.g. `tasks/schopenhauer_kallipolis_2026-09-05/`)
- `scripts/build_chunked_audiobook.py` - core chunker, batch runner, and WAV joiner
- `scripts/mahti/` - CSC bootstrap scripts, runners, and cluster-specific helpers
- `monitor_tts_progress.sh` - live Puhti/Mahti progress monitor
- `local_verify_chunks/` - accepted 2-chunk verification outputs
- `outputs/` - locally synced final outputs
- `RUNNING_STATUS.md` - current run notes and operator log

## Fastest way to run the audiobook

Use the Mahti full-run wrapper with two A100s and parallel model rendering:

```bash
PARALLEL_MODELS=1 FORCE=0 sbatch \
  --account project_2010549 \
  --partition gpusmall \
  --gres gpu:a100:2,nvme:900 \
  --cpus-per-task 32 \
  scripts/mahti/run_half_of_a_yellow_sun.sh
```

That path loads each model once, keeps Qwen3 and VibeVoice isolated, and is the preferred route for the full audiobook.

## Quick validation run

For a short smoke test, use one chunk:

```bash
MODE=preflight MODELS=vibevoice_realtime srun --pty \
  --account project_2010549 \
  --partition gpu \
  --gres gpu:v100:1 \
  --cpus-per-task 10 \
  --time 00:45:00 \
  scripts/mahti/run_half_of_a_yellow_sun.sh
```

You can switch `MODELS` to `qwen3_faster_clone` or pass both models if you want a small end-to-end pass.

## Direct model runners

These are the lower-level resume/debug entrypoints:

- `scripts/mahti/run_qwen3_batch_optimized.sh`
- `scripts/mahti/run_vibevoice_batch_optimized.sh`

They expect the audiobook work tree at:

`generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/chunks`

Use them when you need to resume a single model or inspect one runner in isolation.

## Monitoring

```bash
REMOTE_CLUSTERS=puhti,mahti ./monitor_tts_progress.sh
```

The monitor shows:

- active jobs on both clusters
- per-model chunk counts
- average sec/chunk and chunks/hr
- ETA
- GPU util and memory for running jobs

### Speed caveat

The reported speed is a live estimate, not a lab benchmark.

`monitor_tts_progress.sh` now computes speed and ETA from the chunks written by the current job, not from every chunk WAV already present in the run tree. On resumed runs the monitor still shows the total completion as `done / total`, but the throughput and ETA are based only on files whose `mtime` falls within the current job runtime window. The cluster summary uses the same current-job basis.

## What works

- 2-chunk verification outputs for both `qwen3_faster_clone` and `vibevoice_realtime`
- Qwen3 full audiobook generation and local sync
- VibeVoice resume on Mahti A100
- WAV joining through `scripts/build_chunked_audiobook.py`
- GPU usage, ETA, and per-model progress reporting in `monitor_tts_progress.sh`

## What does not work well

- Old hard-coded Puhti paths in ad hoc launchers
- Treating ETA as a precise prediction on resumed jobs
- Relying on the legacy direct runner paths without checking the current work tree

## Current outputs

- Final synced Qwen output: `outputs/audiobooks-half_of_a_yellow_sun-full/qwen3_faster_clone.wav`
- Synced summary: `outputs/audiobooks-half_of_a_yellow_sun-full/audiobook_summary.json`
- Synced Qwen render manifest: `outputs/audiobooks-half_of_a_yellow_sun-full/qwen_render_manifest.json`
- Verification artifacts: `local_verify_chunks/`

## Video generation

The verified local video renderer is `scripts/make_autoscroll_video.py`.

Working verification inputs:

- audio: `local_verify_chunks/qwen3_faster_clone.wav`
- chunk text dir: `local_verify_chunks/chunk_source/`
- matching chunk audio dir: `local_verify_chunks/half_of_a_yellow_sun/qwen3_faster_clone/chunks/`

Verified outputs:

- plain autoscroll: `local_verify_chunks/videos/qwen3_faster_clone_autoscroll.mp4`
- active line: `local_verify_chunks/videos/qwen3_faster_clone_active_line.mp4`

### Plain autoscroll

Use chunk text plus matching chunk audio when you want the scroll timing to follow the real chunk durations:

```bash
python scripts/make_autoscroll_video.py \
  --audio local_verify_chunks/qwen3_faster_clone.wav \
  --chunk-text-dir local_verify_chunks/chunk_source \
  --chunk-audio-dir local_verify_chunks/half_of_a_yellow_sun/qwen3_faster_clone/chunks \
  --output local_verify_chunks/videos/qwen3_faster_clone_autoscroll.mp4
```

### Active line video

Use this when you want the muted red underline that grows left-to-right under only the current line.

For proper sync, do not rely on `--text` alone. Use:

- `--chunk-text-dir` so the renderer knows the exact source text that produced the audio
- `--chunk-audio-dir` so it can match the real chunk boundaries
- `--active-line-timing asr` so the underline is aligned from `faster-whisper` word timestamps instead of coarse weighted line timing
- `--active-line-asr-cache-dir` so reruns reuse the alignment cache

```bash
python scripts/make_autoscroll_video.py \
  --audio local_verify_chunks/qwen3_faster_clone.wav \
  --chunk-text-dir local_verify_chunks/chunk_source \
  --chunk-audio-dir local_verify_chunks/half_of_a_yellow_sun/qwen3_faster_clone/chunks \
  --output local_verify_chunks/videos/qwen3_faster_clone_active_line.mp4 \
  --show-active-line \
  --active-line-timing asr \
  --active-line-asr-cache-dir local_verify_chunks/videos/asr_cache
```

### Notes

- `ffmpeg` must be installed and available on `PATH`.
- The local active-line ASR path uses `faster-whisper`, so `auto` means CUDA when available and CPU otherwise. This path does not use Apple `mps`.
- The active-line render uses `faster-whisper`. An easy way to run it without editing the repo environment is:

```bash
/Users/ruwodda/.local/bin/uv run --python 3.11 \
  --with faster-whisper \
  --with pillow \
  --with numpy \
  --with soundfile \
  python scripts/make_autoscroll_video.py ...
```

- The renderer matches chunk files by chunk id, so the chunk text directory can contain more chunks than the local verification audio directory. Only overlapping chunk ids are used.
- The red underline is only as accurate as the text/audio pair. If the WAV was generated from different text than the chunk text files, the underline will drift.

### Cluster video generation

For future full-book runs on CSC, keep the GPU allocation focused on TTS plus chunk audit and let the wrapper submit the MP4 render to a CPU partition:

```bash
MODE=full PARALLEL_MODELS=1 FORCE=0 GENERATE_VIDEO=1 \
VIDEO_MODEL=qwen3_faster_clone VIDEO_RENDER_MODE=submit_cpu \
sbatch --account project_2010549 \
  --partition gpusmall \
  --gres gpu:a100:2,nvme:900 \
  --cpus-per-task 32 \
  scripts/mahti/run_half_of_a_yellow_sun.sh
```

That path reuses the chunk ASR cache generated during the audio audit, so the CPU video job usually spends its time on layout and `libx264` encoding instead of redoing alignment work.

If you only want the video step for an already-generated audiobook, use:

```bash
CSC_PROJECT=project_2010549 \
bash scripts/mahti/submit_audiobook_video.sh --cluster mahti --mode full --resource-mode cpu -- \
  --summary /scratch/project_2010549/$USER/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-full/audiobook_summary.json \
  --model qwen3_faster_clone \
  --show-active-line \
  --active-line-timing asr \
  --active-line-asr-device cpu \
  --active-line-asr-cache-dir /scratch/project_2010549/$USER/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/qwen3_faster_clone/chunk_audit/asr_cache
```

### Chunking guardrails

Full-book runs now default to two guardrails before the join/video steps:

- `scripts/build_chunked_audiobook.py` strips large numbered note/endnote walls that appear immediately before `CHAPTER` headings, which prevents pathological note-only chunks from being synthesized as audiobook prose.
- `scripts/mahti/run_audiobook_video.py` refuses to render a video when the chunk audit reports blocking chunks, so a broken audio run cannot silently produce a misleading MP4.

The summary JSON records `effective_input` when sanitization changes the source text, and the video runner prefers that cleaned input plus any corrected chunk text directory.

For Qwen full-book runs, keep chunk sizes conservative. The production wrapper default is `CHUNK_SIZE=2200` and refuses values above `2600` unless `ALLOW_LARGE_QWEN_CHUNKS=1` is set for a measured experiment. Larger chunks can hit Qwen's fixed output ceiling and produce WAVs that sound valid but cover only part of the requested chunk. The chunk audit catches this as `truncated_current_chunk`, and any run with that status must be relaunched from a fresh output root with smaller chunks.

## YouTube audiobook flow (human-narration reading videos)

For works that already have good human-narrated readings on YouTube (e.g. Kallipolis Radio), the fastest path to a synced scrolling-text video is to use the narration as the audio and derive the on-screen text from a `faster-whisper` transcript of that audio (so the text always matches the voice). One MP4 is produced per YouTube video, which makes it trivial to deliver one video per essay/chapter.

Pipeline: `scripts/process_youtube_hoys_audiobook.py` (resumable stage by stage):

1. download best audio + metadata (chapters) with `yt-dlp`,
2. convert to 16 kHz mono WAV,
3. transcribe with `faster-whisper` (`small.en` by default) and write a timestamped transcript,
4. split the audio/transcript into paragraph-ish ~60 s chunks,
5. render each item with `scripts/make_autoscroll_video.py` (active red line, ASR word timing).

Usage (per-essay output under `library/<book>/<slug>/..._neuralsurfer.mp4`):

```bash
# 1) snapshot the playlist metadata first
yt-dlp --flat-playlist -J "<playlist-url>" > tasks/<book>_<date>/playlist_raw.json
# 2) build tasks/<book>_<date>/playlist.json (entries: id, title, duration) -
#    keep titles short; the entry slug becomes the output folder name
# 3) render the essays you want; --indices is 1-based playlist order
python scripts/process_youtube_hoys_audiobook.py \
  --playlist-json tasks/schopenhauer_kallipolis_2026-09-05/playlist.json \
  --output-root library/schopenhauer_kallipolis \
  --indices 1-2 --render --use-youtube-chapters --chunk-seconds 60
```

Current Schopenhauer playlist metadata (30 Kallipolis readings, ~19 h): `tasks/schopenhauer_kallipolis_2026-09-05/playlist.json`.

Speed / GPU knobs:

- `--transcript-device cuda --transcript-compute-type int8` (and `--asr-device cuda`) put both whisper passes on a GPU when one exists. Defaults stay CPU.
- Measured, `1920x1080` active-line render is roughly **real-time of the finished video** on a Colab T4 VM (`~28 fps` in, `~1.16x` of 24 fps output, `libx264 veryfast crf26`) and **~3x real time on Apple-silicon CPU** (`~72 fps` in). `faster-whisper small.en int8` runs about **3x real time on Apple-silicon CPU**. So a 1-hour essay costs ~1 hour of render plus ~2 x ~20 min of whisper; splitting long books into per-essay runs is what makes the job tractable and resumable.

### YouTube HTTP 403 on media streams (2026 bot wall) - fix

Since ~mid-2026 YouTube returns `HTTP Error 403: Forbidden` on the actual media stream (metadata still works) from flagged IPs, even with a logged-in browser's cookies. The working fix is a **PO-token provider** plus a recent yt-dlp:

1. `python3 -m pip install -U yt-dlp` (2026.08.x or newer).
2. Install the bgutil Rust provider (single binary, no node/docker): download `bgutil-pot-macos-*`/`-linux-*` from `https://github.com/jim60105/bgutil-ytdlp-pot-provider-rs/releases/latest`, install to `~/bin/bgutil-pot`, and the plugin zip into the yt-dlp plugins dir.
3. Run yt-dlp with:

```bash
yt-dlp --plugin-dirs "$HOME/Library/Application Support/yt-dlp/plugins" \
  --extractor-args "youtubepot-bgutilcli:cli_path=$HOME/bin/bgutil-pot" <url>
```

`scripts/process_youtube_hoys_audiobook.py` reads these via the `YTDLP_EXTRA_ARGS` env var (shlex-split, appended to every yt-dlp call), e.g.:

```bash
export YTDLP_EXTRA_ARGS='--plugin-dirs "$HOME/Library/Application Support/yt-dlp/plugins" --extractor-args youtubepot-bgutilcli:cli_path=$HOME/bin/bgutil-pot'
```

Providing a PO token does not guarantee a bypass; if it still 403s, the provider can also run in HTTP-server mode (`bgutil-pot server`) which the plugin auto-detects on `http://127.0.0.1:4416`.

## Google Colab GPU (fast verification and one-off renders)

The Colab MCP bridge gives a T4 (16 GB) for GPU whisper/TTS work without touching CSC:

- bridge: `~/bin/colab_bridge.py`, started by `start-colab-bridge.sh` (or `-shared`), endpoint `http://127.0.0.1:18023/mcp`
- after the Colab notebook ngrok URL changes, run `./update-colab-url.sh <ngrok-url>` from the repo root and restart the LaunchAgent
- backend health is GET-only: `curl <ngrok-url>/health`; MCP `tools/list` shows `execute_code` + `check_health`
- a minimal python client (JSON-RPC POST to the bridge) lives in `/tmp/sample_build/colab_driver.py` during a session; state (pip installs, HF model cache) persists between `execute_code` calls but resets when the Colab runtime disconnects

Caveats learned the hard way:

- **Colab's datacenter IP is bot-flagged by YouTube** (`Sign in to confirm you're not a bot`). Do YouTube downloads on the home Mac, then upload the audio (or run whisper/render there); Colab is not faster for the render itself, so for a single essay it is usually simpler to run the whole pipeline locally.
- Renders run near real-time of the output everywhere (see measured numbers above), so "fast verification" on Colab means GPU whisper + TTS, not faster MP4 encoding.

## Schopenhauer source hygiene

- The original `library/schopenhauer_parerga/schopenhauer_parerga.txt` was **contaminated**: the last two of its seven volumes were actually *Moonfleet* and *Between Whiles* (113 wrong-book markers; ~43% of the file). Any TTS audiobook/video built from it is unusable past the first five Saunders volumes.
- `scripts/fetch_schopenhauer_anthology.py` downloads the seven Gutenberg ebooks, validates identity + rejects known wrong-book markers, and writes the clean `library/schopenhauer_parerga/schopenhauer_parerga_clean.txt` (raw copies under `gutenberg_raw/`).
- The seven volumes are the T. Bailey Saunders translation set of the essays (Art of Literature, Counsels and Maxims, Art of Controversy + papers, Studies in Pessimism, On Human Nature, The Wisdom of Life, Religion: A Dialogue and Other Essays). Prefer the Kallipolis human readings for per-essay listening videos; the clean text remains the fallback for TTS runs.

## CSC billing notes

These figures are from CSC Docs and should be treated as the source of truth when sizing jobs:

- Mahti GPU billing: each reserved A100 GPU consumes `100 GPU BU/hour`; an A100 slice (`a100_1g.5gb`) consumes `15 GPU BU/hour`; reserved NVMe consumes `0.01 GPU BU/hour` per GiB. Memory is not billed separately on Mahti GPU partitions.
- Puhti GPU billing: `Total GPU BU = (NCores * 1 + MemGiBs * 0.1 + NVMeGiBs * 0.006 + NGPUs * 60) * walltime hours`.
- Billing is based on granted resources and elapsed job time, not achieved utilization. If you reserve 2 GPUs for 1 hour on Mahti, that is `200 GPU BU` plus any NVMe charge even if one GPU is idle.
- If 2 GPUs make the job finish in half the time, the GPU part of the total cost is roughly unchanged. Example on Mahti with the same NVMe request:
  - `1 x A100` for `2h` = `200 GPU BU`
  - `2 x A100` for `1h` = `200 GPU BU`
  - If scaling is worse than linear, the total GPU BU goes up.

Useful partition limits:

- Mahti `gputest`: up to `4` GPUs, `15 minutes`
- Mahti `gpusmall`: up to `2` GPUs, `36 hours`
- Mahti `gpumedium`: up to `24` GPUs, `36 hours`
- Puhti `gputest`: up to `8` GPUs, `15 minutes`
- Puhti `gpu`: up to `80` GPUs, `3 days`

Official docs:

- CSC billing overview: https://docs.csc.fi/accounts/billing/
- Mahti and Puhti BU formulas: https://docs.csc.fi/computing/hpc-billing/
- Partition limits for Puhti and Mahti: https://docs.csc.fi/computing/running/batch-job-partitions/

## More detail

- `scripts/mahti/README.md` - CSC cluster bootstrap and model-specific notes
- `docs/ModelChat.md` - long troubleshooting trail and previous run notes
