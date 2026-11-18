#!/usr/bin/env python3
"""
Unified CLI entrypoint for the scrolling-video audiobook pipeline.

Accepts a single ``--input-type {youtube,text,audio}`` and ``--input <url-or-path>``
and dispatches to whichever existing pipeline already knows how to handle that
input type:

* ``youtube`` -> ``scripts/process_youtube_hoys_audiobook.py`` internals
  (download, transcribe, chunk, render) via a single-URL ``PlaylistItem``.
* ``text``    -> ``scripts/build_chunked_audiobook.py`` (chunk + TTS) followed
  by ``scripts/mahti/run_audiobook_video.py`` (render), invoked as
  subprocesses -- exactly the sequence ``scripts/mahti/run_qwen_audiobook_video.sh``
  already performs for the currently-running book jobs.
* ``audio``   -> the same transcription/chunking/render building blocks as the
  youtube route, minus the download step, driven directly against a local
  audio file (optionally with ``--manual-chapters`` in place of real YouTube
  chapter markers).

All three routes converge on ``scripts/make_autoscroll_video.py`` for
rendering and on ``scripts/embed_video_chapters.py`` for the optional final
MP4 chapter-bookmark pass.

This file is pure glue: it imports and reuses
``scripts/process_youtube_hoys_audiobook.py`` for the youtube/audio routes and
subprocess-invokes the other pipelines rather than reimplementing chunking,
TTS, or rendering logic. The one deliberate exception is
``_render_with_autoscroll()`` below -- see its docstring for why.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

# Flat sibling import, matching the convention already used inside
# scripts/build_chunked_audiobook.py (``from textChunkingAdapter import ...``).
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import process_youtube_hoys_audiobook as yt_pipeline  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "library" / "_unified"
MAKE_AUTOSCROLL_VIDEO = SCRIPTS_DIR / "make_autoscroll_video.py"
BUILD_CHUNKED_AUDIOBOOK = SCRIPTS_DIR / "build_chunked_audiobook.py"
RUN_AUDIOBOOK_VIDEO = SCRIPTS_DIR / "mahti" / "run_audiobook_video.py"
EMBED_VIDEO_CHAPTERS = SCRIPTS_DIR / "embed_video_chapters.py"
SUBMIT_UNIFIED_PIPELINE = SCRIPTS_DIR / "mahti" / "submit_unified_pipeline.sh"

_CHAPTERS_OUTPUT_SUPPORTED: bool | None = None


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _print_command(command: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)


def _humanize_stem(stem: str) -> str:
    """Turn a filename stem like ``a_talk_with_a_wayfarer`` into a readable
    heading like ``A Talk With A Wayfarer`` for use as the fallback chapter
    heading / ASR bias prompt when no richer title metadata is available."""
    text = re.sub(r"[_\-]+", " ", stem).strip()
    text = re.sub(r"\s+", " ", text)
    return text.title() if text else stem


def _make_autoscroll_supports_chapters_output() -> bool:
    """Whether scripts/make_autoscroll_video.py exposes ``--chapters-output``.

    That flag is added by a sibling task (see the unified pipeline design's
    "MP4 chapter bookmarks" piece). This repo is under active parallel
    development, so the flag may or may not have landed yet; we probe the
    source text (cheap, no import/exec side effects) instead of assuming,
    so this script degrades gracefully -- with a clear warning -- instead of
    crashing with an argparse "unrecognized arguments" error.
    """
    global _CHAPTERS_OUTPUT_SUPPORTED
    if _CHAPTERS_OUTPUT_SUPPORTED is None:
        try:
            source = MAKE_AUTOSCROLL_VIDEO.read_text(encoding="utf-8")
        except OSError:
            source = ""
        _CHAPTERS_OUTPUT_SUPPORTED = "--chapters-output" in source
    return _CHAPTERS_OUTPUT_SUPPORTED


def _build_chunks(
    segments: list[yt_pipeline.Segment],
    *,
    heading: str,
    chunk_seconds: float,
    chapter_cues: list[yt_pipeline.ChapterCue] | None,
    paragraph_strategy: str,
) -> list[tuple[int, float, float, str]]:
    """Wrapper around process_youtube_hoys_audiobook.build_chunks() that only
    forwards ``paragraph_strategy`` when that (sibling-task) kwarg has
    actually landed, so this script keeps working against either version of
    that file instead of hard-failing with a TypeError."""
    kwargs: dict[str, Any] = dict(
        heading=heading, chunk_seconds=chunk_seconds, chapter_cues=chapter_cues
    )
    params = inspect.signature(yt_pipeline.build_chunks).parameters
    if "paragraph_strategy" in params:
        kwargs["paragraph_strategy"] = paragraph_strategy
    elif paragraph_strategy != "naive":
        raise RuntimeError(
            "process_youtube_hoys_audiobook.build_chunks() does not support "
            "paragraph_strategy on this checkout (the LLM-paragraphizer "
            "integration hasn't landed yet). Rerun with "
            "--paragraph-strategy naive, or update process_youtube_hoys_audiobook.py."
        )
    return yt_pipeline.build_chunks(segments, **kwargs)


def _load_manual_chapter_cues(path: Path) -> list[yt_pipeline.ChapterCue]:
    """Load --manual-chapters JSON: [{"start_seconds": float, "title": str}, ...]
    and convert to process_youtube_hoys_audiobook.ChapterCue objects, using
    the same "end = next cue's start, last cue open-ended" convention that
    extract_chapter_cues() uses for real YouTube chapter markers."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"--manual-chapters must contain a JSON array: {path}")

    raw: list[tuple[float, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"--manual-chapters entries must be objects: {entry!r}")
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        start = float(entry["start_seconds"])
        raw.append((max(0.0, start), title))
    raw.sort(key=lambda pair: pair[0])

    cues: list[yt_pipeline.ChapterCue] = []
    for index, (start, title) in enumerate(raw):
        end = raw[index + 1][0] if index + 1 < len(raw) else None
        cues.append(yt_pipeline.ChapterCue(start=start, end=end, title=title))
    return cues


def playlist_item_from_url(url: str, *, slug: str | None = None) -> yt_pipeline.PlaylistItem:
    """Single-URL variant of process_youtube_hoys_audiobook.load_playlist():
    fetches metadata for exactly one YouTube URL via yt-dlp --dump-json and
    builds one PlaylistItem, reusing camel_slug()/short_title() from that
    module so slugs/headings match the batch playlist pipeline exactly."""
    command = yt_pipeline.yt_dlp_command() + ["--skip-download", "--dump-json", url]
    _print_command(command)
    result = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)

    video_id = str(info.get("id") or "").strip()
    if not video_id:
        raise RuntimeError(f"yt-dlp did not return a video id for {url}")
    title = str(info.get("title") or "").strip() or video_id
    resolved_slug = slug or yt_pipeline.camel_slug(title)

    return yt_pipeline.PlaylistItem(
        index=1,
        video_id=video_id,
        title=title,
        duration=int(info.get("duration") or 0),
        url=f"https://www.youtube.com/watch?v={video_id}",
        slug=resolved_slug,
    )


def _render_with_autoscroll(
    *,
    text_path: Path,
    audio_path: Path,
    chunks_dir: Path,
    chunk_audio_dir: Path,
    output_path: Path,
    asr_cache_dir: Path,
    force: bool,
    chapters_output: Path | None,
    args: argparse.Namespace,
) -> Path | None:
    """Render via scripts/make_autoscroll_video.py.

    This mirrors process_youtube_hoys_audiobook.render_video()'s command
    construction (same flags/values for the youtube/audio routes) with two
    additions: explicit --chapter-end-hold/--chapter-transition/
    --chapter-start-hold passthrough (exposed on this unified CLI, whereas
    render_video() always uses make_autoscroll_video.py's own hardcoded
    defaults) and --chapters-output when the caller wants an MP4
    chapter-bookmark JSON out of this render.

    render_video() itself is intentionally NOT called here for the
    chapters-requested case: as of this task's interface contract it has no
    --chapters-output hook, and process_youtube_hoys_audiobook.py is owned by
    a sibling task -- this script may not modify it. Duplicating its short,
    already-stable command list locally (rather than reaching into a file
    outside this task's ownership) is the smallest change that gets real,
    working chapter bookmarks end to end. Returns the chapters JSON path that
    was actually produced (or requested to be produced), or None.
    """
    want_chapters_output = chapters_output is not None
    supports_chapters_output = _make_autoscroll_supports_chapters_output()
    if want_chapters_output and not supports_chapters_output:
        print(
            "[build_scrolling_video] WARNING: scripts/make_autoscroll_video.py "
            "does not support --chapters-output yet; rendering "
            f"{output_path.name} without MP4 chapter bookmarks.",
            file=sys.stderr,
        )

    effective_chapters_output = (
        chapters_output if (want_chapters_output and supports_chapters_output) else None
    )

    if output_path.exists() and not force:
        print(f"Reusing existing render: {output_path}")
        return effective_chapters_output if (
            effective_chapters_output and effective_chapters_output.exists()
        ) else None

    command = [
        sys.executable,
        str(MAKE_AUTOSCROLL_VIDEO),
        "--audio",
        str(audio_path),
        "--text",
        str(text_path),
        "--chunk-text-dir",
        str(chunks_dir),
        "--chunk-audio-dir",
        str(chunk_audio_dir),
        "--output",
        str(output_path),
        "--width",
        "1920",
        "--height",
        "1080",
        "--fps",
        "24",
        "--video-codec",
        "libx264",
        "--video-preset",
        "veryfast",
        "--video-crf",
        "18",
        "--font",
        "auto",
        "--paragraph-size",
        "56",
        "--heading-size",
        "88",
        "--focus-line-ratio",
        "0.62",
        "--chapter-end-hold",
        str(args.chapter_end_hold),
        "--chapter-transition",
        str(args.chapter_transition),
        "--chapter-start-hold",
        str(args.chapter_start_hold),
        "--show-active-line",
        "--active-line-timing",
        "asr",
        "--active-line-asr-model",
        args.transcript_model,
        "--active-line-asr-device",
        "cpu",
        "--active-line-asr-workers",
        "auto",
        "--active-line-asr-cache-dir",
        str(asr_cache_dir),
    ]
    if effective_chapters_output is not None:
        command.extend(["--chapters-output", str(effective_chapters_output)])

    asr_cache_dir.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _print_command(command)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return effective_chapters_output


def embed_chapters_if_requested(
    video_path: Path, chapters_json: Path | None, args: argparse.Namespace
) -> Path:
    """Subprocess call to scripts/embed_video_chapters.py; a clean no-op
    passthrough (returns video_path unchanged) if chapters_json is
    missing/empty, --no-embed-chapters was passed, or embed_video_chapters.py
    hasn't landed on this checkout yet (sibling task)."""
    if not args.embed_chapters:
        return video_path
    if chapters_json is None or not chapters_json.exists():
        return video_path
    if not EMBED_VIDEO_CHAPTERS.exists():
        print(
            f"[build_scrolling_video] WARNING: {EMBED_VIDEO_CHAPTERS} not found; "
            "skipping MP4 chapter-bookmark embed.",
            file=sys.stderr,
        )
        return video_path

    command = [
        sys.executable,
        str(EMBED_VIDEO_CHAPTERS),
        "--video",
        str(video_path),
        "--chapters-json",
        str(chapters_json),
        "--in-place",
    ]
    _print_command(command)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return video_path


# ---------------------------------------------------------------------------
# Route implementations
# ---------------------------------------------------------------------------


def run_youtube_route(args: argparse.Namespace) -> dict[str, Any]:
    item = playlist_item_from_url(args.input, slug=args.slug)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    want_chapters = bool(args.render and args.embed_chapters)

    ns = argparse.Namespace(
        output_root=output_root,
        min_free_gb=args.min_free_gb,
        force_metadata=args.force_metadata,
        force_download=args.force_download,
        force_audio=args.force_audio,
        force_transcript=args.force_transcript,
        force_chunks=args.force_chunks,
        force_render=args.force_render,
        transcript_source=args.transcript_source,
        transcript_model=args.transcript_model,
        chunk_seconds=args.chunk_seconds,
        # Skip process_item()'s own render when we need --chapters-output;
        # we render ourselves once, below, instead of rendering twice.
        render=args.render and not want_chapters,
        use_youtube_chapters=args.use_youtube_chapters,
        paragraph_strategy=args.paragraph_strategy,
    )
    summary = yt_pipeline.process_item(ns, item)

    if want_chapters:
        item_dir = output_root / item.slug
        work_dir = item_dir / "video_work"
        output_path = item_dir / f"{item.slug}_neuralsurfer.mp4"
        chapters_output_target = work_dir / f"{item.slug}_chapters.json"

        used_chapters_output = _render_with_autoscroll(
            text_path=Path(summary["text"]),
            audio_path=Path(summary["joined_audio"]),
            chunks_dir=work_dir / "chunks",
            chunk_audio_dir=work_dir / "chunk_audio",
            output_path=output_path,
            asr_cache_dir=work_dir / "asr_cache",
            force=args.force_render,
            chapters_output=chapters_output_target,
            args=args,
        )
        summary["mp4"] = str(output_path) if output_path.exists() else None
        summary["chapters_json"] = (
            str(used_chapters_output)
            if used_chapters_output and used_chapters_output.exists()
            else None
        )
    else:
        summary["chapters_json"] = None

    if summary.get("mp4"):
        chapters_json_path = (
            Path(summary["chapters_json"]) if summary.get("chapters_json") else None
        )
        final_mp4 = embed_chapters_if_requested(Path(summary["mp4"]), chapters_json_path, args)
        summary["mp4"] = str(final_mp4)

    return summary


def run_audio_route(args: argparse.Namespace) -> dict[str, Any]:
    if args.transcript_source != "whisper":
        raise ValueError(
            "--input-type audio only supports --transcript-source whisper "
            "(there is no YouTube auto-caption source for a local audio file)."
        )

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Audio input not found: {input_path}")

    slug = args.slug or yt_pipeline.camel_slug(input_path.stem)
    heading = _humanize_stem(input_path.stem)

    output_root = args.output_root
    item_dir = output_root / slug
    work_dir = item_dir / "video_work"
    item_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    yt_pipeline.require_free_space(output_root, args.min_free_gb)

    wav_audio = work_dir / f"{slug}_audio_16k.wav"
    yt_pipeline.convert_audio(input_path, wav_audio, args.force_audio)

    transcript_json = item_dir / f"{slug}Transcript.json"
    segments = yt_pipeline.transcribe_audio(
        wav_audio,
        transcript_json,
        title=heading,
        model_name=args.transcript_model,
        force=args.force_transcript,
    )

    transcript_txt = item_dir / f"{slug}Transcript.txt"
    yt_pipeline.write_timestamped_transcript(
        segments,
        transcript_txt,
        yt_pipeline.PlaylistItem(
            index=1, video_id=slug, title=heading, duration=0, url=str(input_path), slug=slug
        ),
    )

    chapter_cues = _load_manual_chapter_cues(args.manual_chapters) if args.manual_chapters else []

    chunks = _build_chunks(
        segments,
        heading=heading,
        chunk_seconds=args.chunk_seconds,
        chapter_cues=chapter_cues,
        paragraph_strategy=args.paragraph_strategy,
    )
    full_text = "\n\n".join(chunk_text for _, _, _, chunk_text in chunks).strip() + "\n"
    text_path = item_dir / f"{slug}.txt"
    text_path.write_text(full_text, encoding="utf-8")

    chunks_dir = work_dir / "chunks"
    chunk_audio_dir = work_dir / "chunk_audio"
    manifest = yt_pipeline.write_chunks_and_audio(
        chunks,
        source_audio=wav_audio,
        chunks_dir=chunks_dir,
        chunk_audio_dir=chunk_audio_dir,
        force=args.force_chunks,
    )
    (work_dir / "chunk_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    joined_audio = work_dir / f"{slug}_chunk_audio_joined.wav"
    yt_pipeline.join_chunk_audio(chunk_audio_dir, joined_audio, args.force_audio)

    output_path = item_dir / f"{slug}_neuralsurfer.mp4"
    chapters_json: str | None = None
    if args.render:
        yt_pipeline.require_free_space(output_root, args.min_free_gb)
        want_chapters = bool(args.embed_chapters)
        chapters_output_target = work_dir / f"{slug}_chapters.json" if want_chapters else None
        used_chapters_output = _render_with_autoscroll(
            text_path=text_path,
            audio_path=joined_audio,
            chunks_dir=chunks_dir,
            chunk_audio_dir=chunk_audio_dir,
            output_path=output_path,
            asr_cache_dir=work_dir / "asr_cache",
            force=args.force_render,
            chapters_output=chapters_output_target,
            args=args,
        )
        chapters_json = (
            str(used_chapters_output)
            if used_chapters_output and used_chapters_output.exists()
            else None
        )

    summary: dict[str, Any] = {
        "slug": slug,
        "item_dir": str(item_dir),
        "input_audio": str(input_path),
        "wav_audio": str(wav_audio),
        "transcript_json": str(transcript_json),
        "transcript_txt": str(transcript_txt),
        "text": str(text_path),
        "chunks": len(chunks),
        "manual_chapters": len(chapter_cues),
        "joined_audio": str(joined_audio),
        "mp4": str(output_path) if output_path.exists() else None,
        "chapters_json": chapters_json,
    }

    if summary["mp4"]:
        chapters_json_path = Path(chapters_json) if chapters_json else None
        final_mp4 = embed_chapters_if_requested(Path(summary["mp4"]), chapters_json_path, args)
        summary["mp4"] = str(final_mp4)

    (item_dir / f"{slug}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_text_route(args: argparse.Namespace) -> dict[str, Any]:
    """Subprocess-invokes scripts/build_chunked_audiobook.py then
    scripts/mahti/run_audiobook_video.py, exactly the sequence
    scripts/mahti/run_qwen_audiobook_video.sh already performs for the
    currently-running book jobs -- reused as a subprocess pipeline here
    instead of a shell script, so all three input types share one Python
    entrypoint. Neither script is modified by this task."""
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Text input not found: {input_path}")

    slug = args.slug or yt_pipeline.camel_slug(input_path.stem)
    output_root = args.output_root
    item_dir = output_root / slug
    work_dir = item_dir / "work"
    item_dir.mkdir(parents=True, exist_ok=True)

    audiobook_stem = input_path.stem
    summary_path = work_dir / audiobook_stem / "audiobook_summary.json"

    build_cmd = [
        sys.executable,
        str(BUILD_CHUNKED_AUDIOBOOK),
        "--input",
        str(input_path),
        "--models",
        args.tts_model,
        "--output-dir",
        str(item_dir),
        "--work-dir",
        str(work_dir),
        "--chunk-size",
        str(args.chunk_size),
        "--min-chunk-size",
        str(args.min_chunk_size),
    ]
    if args.force or args.force_chunks:
        build_cmd.append("--force")
    if args.dry_run:
        build_cmd.append("--dry-run")

    _print_command(build_cmd)
    subprocess.run(build_cmd, cwd=REPO_ROOT, check=True)

    result: dict[str, Any] = {
        "slug": slug,
        "item_dir": str(item_dir),
        "input": str(input_path),
        "audiobook_summary": str(summary_path),
        "mp4": None,
        "chapters_json": None,
    }

    if args.dry_run:
        return result

    if not summary_path.exists():
        raise RuntimeError(f"build_chunked_audiobook.py did not produce a summary at {summary_path}")

    if args.render:
        video_output = item_dir / f"{slug}_{args.tts_model}_active_line.mp4"
        timing_output = item_dir / f"{slug}_{args.tts_model}_active_line.timing.json"
        video_cmd = [
            sys.executable,
            str(RUN_AUDIOBOOK_VIDEO),
            "--summary",
            str(summary_path),
            "--model",
            args.tts_model,
            "--output",
            str(video_output),
            "--timing-output",
            str(timing_output),
            "--show-active-line",
            "--active-line-timing",
            "asr",
            "--active-line-asr-model",
            args.transcript_model,
            "--active-line-asr-device",
            "cpu",
            "--active-line-asr-workers",
            "auto",
            "--video-codec",
            "libx264",
            "--video-preset",
            "veryfast",
            "--video-crf",
            "18",
            "--chapter-end-hold",
            str(args.chapter_end_hold),
            "--chapter-transition",
            str(args.chapter_transition),
            "--chapter-start-hold",
            str(args.chapter_start_hold),
        ]
        _print_command(video_cmd)
        subprocess.run(video_cmd, cwd=REPO_ROOT, check=True)
        result["mp4"] = str(video_output) if video_output.exists() else None

        # No chapters_json hook is available on this route today:
        # scripts/mahti/run_audiobook_video.py does not expose
        # make_autoscroll_video.py's --chapters-output flag (it is used as-is
        # by the currently-running book jobs and is out of scope for this
        # task to change), so embed_chapters_if_requested() below is an
        # intentional, clean no-op for the text route in this version.
        if result["mp4"]:
            final_mp4 = embed_chapters_if_requested(Path(result["mp4"]), None, args)
            result["mp4"] = str(final_mp4)

    (item_dir / f"{slug}_unified_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# Cluster submission
# ---------------------------------------------------------------------------


def _reconstruct_passthrough_args(args: argparse.Namespace) -> list[str]:
    """Rebuild the pipeline-specific CLI flags (excluding submission-only
    flags like --cluster/--cpus/--gpus/--account) so scripts/mahti/
    submit_unified_pipeline.sh can forward an explicit, self-contained
    command line to the runner it launches on the cluster."""
    tokens: list[str] = ["--input-type", args.input_type, "--input", args.input]
    if args.slug:
        tokens += ["--slug", args.slug]
    tokens += ["--output-root", str(args.output_root)]
    tokens += ["--paragraph-strategy", args.paragraph_strategy]
    tokens.append("--use-youtube-chapters" if args.use_youtube_chapters else "--no-use-youtube-chapters")
    if args.manual_chapters:
        tokens += ["--manual-chapters", str(args.manual_chapters)]
    tokens += ["--transcript-source", args.transcript_source]
    tokens += ["--transcript-model", args.transcript_model]
    tokens += ["--tts-model", args.tts_model]
    tokens += ["--chunk-seconds", str(args.chunk_seconds)]
    tokens += ["--chunk-size", str(args.chunk_size)]
    tokens += ["--min-chunk-size", str(args.min_chunk_size)]
    tokens += ["--chapter-end-hold", str(args.chapter_end_hold)]
    tokens += ["--chapter-transition", str(args.chapter_transition)]
    tokens += ["--chapter-start-hold", str(args.chapter_start_hold)]
    tokens.append("--embed-chapters" if args.embed_chapters else "--no-embed-chapters")
    tokens.append("--render" if args.render else "--no-render")
    tokens += ["--min-free-gb", str(args.min_free_gb)]
    if args.force:
        tokens.append("--force")
    if args.force_metadata:
        tokens.append("--force-metadata")
    if args.force_download:
        tokens.append("--force-download")
    if args.force_audio:
        tokens.append("--force-audio")
    if args.force_transcript:
        tokens.append("--force-transcript")
    if args.force_chunks:
        tokens.append("--force-chunks")
    if args.force_render:
        tokens.append("--force-render")
    tokens.append("--local")
    tokens.append("--submit-target-is-cluster-job")
    return tokens


def run_submit(args: argparse.Namespace) -> int:
    submit_cmd = ["bash", str(SUBMIT_UNIFIED_PIPELINE), "--input-type", args.input_type, "--input", args.input]
    if args.cluster != "auto":
        submit_cmd += ["--cluster", args.cluster]
    submit_cmd += ["--resource-mode", args.resource_mode]
    if args.cpus is not None:
        submit_cmd += ["--cpus", str(args.cpus)]
    if args.gpus is not None:
        submit_cmd += ["--gpus", str(args.gpus)]
    if args.time:
        submit_cmd += ["--time", args.time]
    if args.account:
        submit_cmd += ["--account", args.account]
    submit_cmd.append("--")
    submit_cmd += _reconstruct_passthrough_args(args)

    _print_command(submit_cmd)

    if not SUBMIT_UNIFIED_PIPELINE.exists():
        if args.dry_run:
            print(
                f"[build_scrolling_video] NOTE: {SUBMIT_UNIFIED_PIPELINE} does not exist "
                "on this checkout yet; showing the command that would be run once "
                "scripts/mahti/submit_unified_pipeline.sh lands.",
                file=sys.stderr,
            )
            return 0
        raise RuntimeError(
            f"{SUBMIT_UNIFIED_PIPELINE} not found -- the cluster-submission wrapper "
            "(scripts/mahti/submit_unified_pipeline.sh) has not landed on this checkout yet."
        )

    if args.dry_run:
        return 0
    subprocess.run(submit_cmd, cwd=REPO_ROOT, check=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified scrolling-video audiobook pipeline: dispatches a YouTube URL, "
            "a local text file, or a local audio file to the matching existing "
            "pipeline and converges on make_autoscroll_video.py for rendering."
        )
    )
    parser.add_argument("--input-type", choices=("youtube", "text", "audio"), required=True)
    parser.add_argument("--input", required=True, help="YouTube URL, or path to a .txt/audio file")
    parser.add_argument("--slug", default=None, help="Output folder/file stem; derived from title/filename otherwise")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument(
        "--paragraph-strategy",
        choices=("naive", "llm"),
        default="naive",
        help="Within-chapter paragraph split strategy (youtube/audio routes only)",
    )
    parser.add_argument(
        "--use-youtube-chapters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use real YouTube chapter markers as # headings (youtube route only)",
    )
    parser.add_argument(
        "--manual-chapters",
        type=Path,
        default=None,
        help='JSON [{"start_seconds": float, "title": str}, ...] (audio route only)',
    )
    parser.add_argument("--transcript-source", choices=("whisper", "youtube-auto"), default="whisper")
    parser.add_argument("--transcript-model", default="small.en")
    parser.add_argument(
        "--tts-model",
        choices=("qwen3_faster_clone", "vibevoice_realtime"),
        default="qwen3_faster_clone",
        help="Model used by the text route's TTS rendering",
    )

    parser.add_argument("--chunk-seconds", type=float, default=60.0, help="youtube/audio routes")
    parser.add_argument("--chunk-size", type=int, default=850, help="text route passthrough")
    parser.add_argument("--min-chunk-size", type=int, default=200, help="text route passthrough")

    parser.add_argument("--chapter-end-hold", type=float, default=3.0)
    parser.add_argument("--chapter-transition", type=float, default=0.6)
    parser.add_argument("--chapter-start-hold", type=float, default=2.0)

    parser.add_argument("--embed-chapters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--cluster", choices=("auto", "mahti", "puhti", "roihu", "local"), default="auto")
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--resource-mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--time", default=None, help="HH:MM:SS override for cluster submission")
    parser.add_argument("--account", default=None)

    submit_group = parser.add_mutually_exclusive_group()
    submit_group.add_argument("--submit", dest="submit", action="store_true", help="Submit as a cluster job")
    submit_group.add_argument("--local", dest="submit", action="store_false", help="Run locally (default)")
    parser.set_defaults(submit=False)
    parser.add_argument(
        "--submit-target-is-cluster-job",
        action="store_true",
        help=argparse.SUPPRESS,  # informational marker set by submit_unified_pipeline.sh's forwarded call
    )

    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--force", action="store_true", help="Cascades to all --force-* flags below")
    parser.add_argument("--force-metadata", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-audio", action="store_true")
    parser.add_argument("--force-transcript", action="store_true")
    parser.add_argument("--force-chunks", action="store_true")
    parser.add_argument("--force-render", action="store_true")

    args = parser.parse_args(argv)

    if args.force:
        args.force_metadata = True
        args.force_download = True
        args.force_audio = True
        args.force_transcript = True
        args.force_chunks = True
        args.force_render = True

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root = args.output_root.resolve()

    if args.submit:
        return run_submit(args)

    if args.dry_run:
        plan = {
            "input_type": args.input_type,
            "input": args.input,
            "slug": args.slug,
            "output_root": str(args.output_root),
            "paragraph_strategy": args.paragraph_strategy,
            "tts_model": args.tts_model,
            "render": args.render,
            "embed_chapters": args.embed_chapters,
            "submit": args.submit,
        }
        print(json.dumps(plan, indent=2))
        return 0

    if args.input_type == "youtube":
        summary = run_youtube_route(args)
    elif args.input_type == "audio":
        summary = run_audio_route(args)
    elif args.input_type == "text":
        summary = run_text_route(args)
    else:  # pragma: no cover - argparse choices already enforce this
        raise ValueError(f"Unknown --input-type: {args.input_type}")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
