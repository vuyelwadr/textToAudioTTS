#!/usr/bin/env python3
"""
Single-video driver for "The Confessions of St. Augustine" [Full Audiobook]
(Ascension Press, YouTube video -mdupzz_41A, 10:12:32 / 36752s).

This is a standalone counterpart to scripts/process_youtube_hoys_audiobook.py,
built for exactly ONE reason: that shared script's load_playlist() has a
hardcoded filter --

    if not title or "confession" in title.lower():
        continue

-- originally added to skip an unrelated Tolstoy "A Confession" playlist
entry. Augustine's title contains "Confessions", so it would be silently
dropped by that filter. Rather than edit the shared filter (other in-progress
work may depend on it), this script imports PlaylistItem and process_item
directly from process_youtube_hoys_audiobook.py and constructs the
PlaylistItem by hand, bypassing load_playlist() entirely. The download,
transcription, paragraphizing, chapter-aligned chunking, and (later)
active-line-ASR rendering all go through the exact same process_item()
pipeline used for the Tolstoy library, so the audiobook video comes out in
the same visual/structural style as library/tolstoy/.

No TTS synthesis happens anywhere in this path: the original narration audio
is preserved and only transcribed (never re-spoken), matching the "HOYS"
(Hear-Original-Your-Self) pattern the shared script already implements.

One deliberate deviation from the shared script: transcribe_audio() there
hardcodes the faster-whisper initial_prompt as "Leo Tolstoy audiobook. ..."
which is simply wrong for this book. Rather than edit that function in the
shared file, this script monkey-patches a corrected copy of it onto the
process_youtube_hoys_audiobook module before calling process_item() -- see
augustine_transcribe_audio() below. Every other line of pipeline logic is
untouched, reused as-is.

A second monkey-patch, augustine_yt_dlp_command(), replaces
hoys.yt_dlp_command(). On the Mahti "small" partition, shutil.which() /
os.access(X_OK) returns False for genuinely executable files on the
/scratch Lustre mount (confirmed: bash's own `which` and direct execution
of the same file both work fine from inside the same job -- only Python's
os.access() check misreports it), so the shared script's PATH-based
yt-dlp discovery never finds a `yt-dlp` wrapper placed on PATH there. Set
the YTDLP_ZIPAPP environment variable to an absolute path to a yt-dlp
zipapp/script and it will be invoked as `sys.executable <path>` (which only
needs the file to be readable, not executable, sidestepping the Lustre
access() issue entirely). Without YTDLP_ZIPAPP set, this falls back to the
shared script's normal PATH/`import yt_dlp` behavior, so this script still
works unmodified on a normal machine (e.g. local runs).

Usage (transcript/download/chunk only, no render -- matches the first pass
of this project):

    python3 scripts/process_augustine_confessions.py \
        --transcript-model small.en \
        --chunk-seconds 600

Add --render once paragraph-correction is done in a later step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import process_youtube_hoys_audiobook as hoys  # noqa: E402

REPO_ROOT = hoys.REPO_ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "library" / "augustine"

VIDEO_ID = "-mdupzz_41A"
VIDEO_TITLE = '"The Confessions of St. Augustine" [Full Audiobook] #audiobook'
VIDEO_DURATION_SECONDS = 36752
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
VIDEO_SLUG = "theConfessionsOfStAugustine"

# Longest of the 279 real YouTube chapter markers on this video is ~471s
# (median ~107s). 600s is comfortably above any single chapter, so with
# --use-youtube-chapters this value only ever acts as a safety fallback --
# in practice every chunk boundary is a real Book/Chapter boundary, not an
# arbitrary time slice.
DEFAULT_CHUNK_SECONDS = 600.0


def augustine_yt_dlp_command() -> list[str]:
    """Copy of hoys.yt_dlp_command() that prefers an explicit zipapp path,
    and (as of 2026-07) also has to route around YouTube's server-side
    anti-bot enforcement for this specific long-form video.

    See the module docstring for why the zipapp override exists: on Mahti's
    small partition, PATH-based discovery of an executable on /scratch fails
    from Python (os.access(X_OK) misreports it) even though the same file is
    findable and runnable from bash. If YTDLP_ZIPAPP is set, invoke it via
    `sys.executable <path>` instead, which only needs read access.

    Two more flags are folded into the returned command prefix (every
    caller in the shared script just does `yt_dlp_command() + [...]`, so
    extra leading flags here apply to every yt-dlp invocation):

    - `--js-runtimes deno:<DENO_BIN>` if DENO_BIN is set: avoids the
      "No supported JavaScript runtime" warning/degraded extraction. This
      alone was NOT enough to fix downloads of this video (still saw a
      503 from the android_vr client's CDN URL even with deno wired in),
      but it's a real improvement worth keeping regardless.
    - `--cookies <path>` if COOKIES_FILE points at an existing file:
      YouTube is currently rejecting anonymous/cookie-less CDN requests
      for this particular ~10h video with 503 (android_vr),
      SABR-experiment URL stripping (android), and explicit 403 with a
      forced format (mweb) -- confirmed reproducible across every client
      and format yt-dlp supports, while a short unrelated video downloads
      fine through the identical toolchain. A logged-in session's cookies
      are the standard yt-dlp workaround for this. No cookies are read or
      shipped anywhere by this script on its own; set COOKIES_FILE to opt
      in explicitly.
    """
    zipapp = os.environ.get("YTDLP_ZIPAPP")
    base = [sys.executable, zipapp] if zipapp else hoys.yt_dlp_command()

    extra: list[str] = []
    deno_bin = os.environ.get("DENO_BIN")
    if deno_bin:
        extra += ["--js-runtimes", f"deno:{deno_bin}"]
    cookies_file = os.environ.get("COOKIES_FILE")
    if cookies_file and Path(cookies_file).exists():
        extra += ["--cookies", cookies_file]

    return base + extra


def augustine_transcribe_audio(
    wav_path: Path,
    transcript_json: Path,
    *,
    title: str,
    model_name: str,
    force: bool,
) -> list["hoys.Segment"]:
    """Copy of hoys.transcribe_audio() with a corrected initial_prompt and
    periodic progress logging.

    The shared script hardcodes 'Leo Tolstoy audiobook. {title}' as the
    faster-whisper initial_prompt, which is wrong for this book and would
    bias decoding toward the wrong author/content.

    The shared script also prints nothing at all while the transcription
    loop runs -- for a ~10 hour audiobook that means hours of silent log
    output with no way to tell "still working" from "stalled". This copy
    adds a heartbeat print every ~30s of wall-clock time so a SLURM log can
    be checked for real progress. Everything else is identical to
    process_youtube_hoys_audiobook.transcribe_audio().
    """
    if transcript_json.exists() and not force:
        payload = json.loads(transcript_json.read_text(encoding="utf-8"))
        return [hoys.Segment(**item) for item in payload["segments"]]

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is required for transcript generation") from exc

    print(f"[transcribe] loading model={model_name} device=cpu compute_type=int8 ...", flush=True)
    load_start = time.monotonic()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print(f"[transcribe] model loaded in {time.monotonic() - load_start:.1f}s; starting {wav_path}", flush=True)

    segments_iter, info = model.transcribe(
        str(wav_path),
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=False,
        initial_prompt=(
            "The Confessions of Saint Augustine, read by Ascension Press. "
            "An autobiographical work by Augustine of Hippo."
        ),
    )

    segments: list[hoys.Segment] = []
    transcribe_start = time.monotonic()
    last_log = transcribe_start
    for segment in segments_iter:
        text = hoys.clean_segment_text(segment.text)
        if not text:
            continue
        segments.append(hoys.Segment(start=float(segment.start), end=float(segment.end), text=text))
        now = time.monotonic()
        if now - last_log >= 30.0:
            elapsed_min = (now - transcribe_start) / 60.0
            pct = (segment.end / VIDEO_DURATION_SECONDS) * 100.0 if VIDEO_DURATION_SECONDS else 0.0
            print(
                f"[transcribe] {len(segments)} segments so far, audio_pos={hoys.timestamp(segment.end)} "
                f"({pct:.1f}% of {hoys.timestamp(VIDEO_DURATION_SECONDS)}), elapsed={elapsed_min:.1f}min",
                flush=True,
            )
            last_log = now

    print(
        f"[transcribe] done: {len(segments)} segments, "
        f"transcription wall time={((time.monotonic() - transcribe_start) / 60.0):.1f}min",
        flush=True,
    )

    if not segments:
        raise RuntimeError(f"Whisper produced no transcript segments for {wav_path}")

    transcript_json.parent.mkdir(parents=True, exist_ok=True)
    transcript_json.write_text(
        json.dumps(
            {
                "audio": str(wav_path),
                "model": model_name,
                "language": getattr(info, "language", "en"),
                "duration": getattr(info, "duration", None),
                "segments": [asdict(segment) for segment in segments],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--transcript-model", default="small.en")
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument(
        "--no-youtube-chapters",
        action="store_true",
        help="Disable chapter-aligned chunking (not recommended -- this video has 279 real chapter markers).",
    )
    parser.add_argument("--render", action="store_true", help="Also render the autoscroll video (2nd ASR pass).")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved PlaylistItem/args and exit.")
    parser.add_argument("--force-metadata", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-audio", action="store_true")
    parser.add_argument("--force-transcript", action="store_true")
    parser.add_argument("--force-chunks", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()

    item = hoys.PlaylistItem(
        index=1,
        video_id=VIDEO_ID,
        title=VIDEO_TITLE,
        duration=VIDEO_DURATION_SECONDS,
        url=VIDEO_URL,
        slug=VIDEO_SLUG,
    )

    args = argparse.Namespace(
        output_root=cli.output_root.resolve(),
        min_free_gb=cli.min_free_gb,
        use_youtube_chapters=not cli.no_youtube_chapters,
        transcript_source="whisper",
        transcript_model=cli.transcript_model,
        chunk_seconds=cli.chunk_seconds,
        render=cli.render,
        force_metadata=cli.force_metadata,
        force_download=cli.force_download,
        force_audio=cli.force_audio,
        force_transcript=cli.force_transcript,
        force_chunks=cli.force_chunks,
        force_render=cli.force_render,
    )

    print(f"=== {item.slug}: {item.title} ===", flush=True)
    print(f"video_id={item.video_id} duration={item.duration}s url={item.url}", flush=True)
    print(f"output_root={args.output_root}", flush=True)
    print(
        f"use_youtube_chapters={args.use_youtube_chapters} chunk_seconds={args.chunk_seconds} "
        f"transcript_source={args.transcript_source} transcript_model={args.transcript_model} render={args.render}",
        flush=True,
    )

    if cli.dry_run:
        print(json.dumps({"item": asdict(item), "args": vars(args) | {"output_root": str(args.output_root)}}, indent=2, default=str))
        return 0

    # Patch in the corrected initial_prompt transcription function (see
    # augustine_transcribe_audio docstring). process_item() looks up
    # `transcribe_audio` as a module-global at call time, so this rebinding
    # takes effect without ever touching the file on disk.
    hoys.transcribe_audio = augustine_transcribe_audio
    hoys.yt_dlp_command = augustine_yt_dlp_command
    if os.environ.get("YTDLP_ZIPAPP"):
        print(f"[yt-dlp] using zipapp override: {os.environ['YTDLP_ZIPAPP']}", flush=True)

    summary = hoys.process_item(args, item)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
