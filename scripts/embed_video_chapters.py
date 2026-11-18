#!/usr/bin/env python3
"""Embed MP4 chapter bookmarks into an already-rendered audiobook video.

Given an existing rendered MP4 and a list of chapter cues (a title plus a
start timestamp, optionally an end timestamp), this module writes an
ffmpeg ``;FFMETADATA1`` sidecar file and remuxes the video (stream copy,
no re-encode) so the chapters show up as real, navigable bookmarks in
players such as VLC and QuickTime.

Callable as a library::

    from scripts.embed_video_chapters import embed_chapters
    embed_chapters(Path("book.mp4"), Path("chapters.json"))

Callable as a standalone CLI for retrofitting an already-rendered video::

    python3 scripts/embed_video_chapters.py \\
        --video book.mp4 --chapters-json chapters.json --output book_chaptered.mp4

``chapters.json`` may be any of:

* the list-of-dicts shape produced by
  ``make_autoscroll_video.build_chapter_marks()``::

      [{"index": 0, "title": "...", "start_seconds": 0.0, "end_seconds": 605.2}, ...]

* a bare list of simpler chapter cues with no ``end_seconds``::

      [{"title": "Chapter One", "start_seconds": 0.0}, ...]

  (``start``/``timestamp``/``start_time`` are also accepted in place of
  ``start_seconds``, and ``end``/``end_time`` in place of ``end_seconds``.)
  Missing end timestamps are inferred from the next chapter's start time,
  or -- for the final chapter -- from the source video's own duration
  (probed via ffprobe only if actually needed).

* a dict wrapping either shape under a ``"chapters"``, ``"chapter_marks"``,
  or ``"chapter_cues"`` key.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def require_binary(name: str) -> str:
    """Resolve `name` (a bare command or a path) to an executable, or raise."""
    binary_path = shutil.which(name)
    if not binary_path:
        raise RuntimeError(f"Required binary '{name}' was not found on PATH")
    return binary_path


@dataclass
class ChapterMark:
    """A single, fully-resolved chapter bookmark (seconds, gap-free)."""

    index: int
    title: str
    start_seconds: float
    end_seconds: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


def probe_duration_seconds(video_path: Path, *, ffprobe_bin: str | None = None) -> float:
    """Return the container duration of `video_path` in seconds via ffprobe."""
    ffprobe = require_binary(ffprobe_bin or "ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    duration = float(payload["format"]["duration"])
    if duration <= 0:
        raise RuntimeError(f"Video duration must be positive, got {duration} for {video_path}")
    return duration


def _coerce_seconds(entry: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = entry.get(key)
        if value is not None:
            return float(value)
    return None


def load_chapter_cues(source: Path | list | dict) -> list[dict]:
    """Load raw chapter-cue dicts from a JSON path, a pre-parsed list, or a
    dict wrapping the list under a "chapters"/"chapter_marks"/"chapter_cues" key.
    """
    if isinstance(source, Path):
        data = json.loads(source.read_text(encoding="utf-8"))
    else:
        data = source

    if isinstance(data, dict):
        for key in ("chapters", "chapter_marks", "chapter_cues"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError(
                "chapters JSON is an object but has no 'chapters' / 'chapter_marks' / "
                "'chapter_cues' key holding the chapter list"
            )

    if not isinstance(data, list):
        raise ValueError("chapters JSON must decode to a list of chapter-cue objects")

    return data


def normalize_chapter_marks(
    raw_cues: list[dict],
    *,
    video_path: Path | None = None,
    ffprobe_bin: str | None = None,
) -> list[ChapterMark]:
    """Turn raw, loosely-shaped chapter-cue dicts into a sorted, gap-free list
    of ChapterMark where start_seconds/end_seconds are always populated.

    Missing end_seconds is inferred from the following chapter's
    start_seconds, or -- for the final chapter -- from the source video's
    own duration (probed lazily via ffprobe, only if actually needed).
    """
    provisional: list[tuple[float, float | None, str]] = []
    for i, entry in enumerate(raw_cues):
        start = _coerce_seconds(entry, ("start_seconds", "start", "timestamp", "start_time"))
        if start is None:
            raise ValueError(f"chapter cue #{i} is missing a start timestamp: {entry!r}")
        end = _coerce_seconds(entry, ("end_seconds", "end", "end_time"))

        title = str(entry.get("title") or entry.get("name") or f"Chapter {i + 1}")
        title = re.sub(r"\s+", " ", title).strip()
        title = title.lstrip("#").strip() or f"Chapter {i + 1}"

        provisional.append((float(start), end, title))

    provisional.sort(key=lambda item: item[0])

    total_duration: float | None = None
    marks: list[ChapterMark] = []
    for i, (start, end, title) in enumerate(provisional):
        if end is None:
            if i + 1 < len(provisional):
                end = provisional[i + 1][0]
            else:
                if total_duration is None and video_path is not None:
                    total_duration = probe_duration_seconds(video_path, ffprobe_bin=ffprobe_bin)
                end = total_duration if total_duration is not None else start + 1.0
        if end <= start:
            end = start + 0.001
        marks.append(ChapterMark(index=i, title=title, start_seconds=start, end_seconds=end))

    return marks


_FFMETADATA_ESCAPE_RE = re.compile(r"([=;#\\\n])")


def _escape_ffmetadata_value(value: str) -> str:
    """Escape the characters ffmpeg's FFMETADATA parser treats as special."""
    return _FFMETADATA_ESCAPE_RE.sub(r"\\\1", value)


def build_ffmetadata(chapter_marks: list[dict] | list[ChapterMark], path: Path) -> None:
    """Write an ffmpeg ``;FFMETADATA1`` chapter file for `chapter_marks` to `path`.

    Each mark may be a ChapterMark or a plain dict with
    ``title``/``start_seconds``/``end_seconds`` keys (the shape produced by
    ``make_autoscroll_video.build_chapter_marks()``). Timestamps use
    TIMEBASE=1/1000 (milliseconds), matching ffmpeg's own chapter convention.
    """
    lines = [";FFMETADATA1"]
    for mark in chapter_marks:
        if isinstance(mark, ChapterMark):
            title = mark.title
            start_seconds = mark.start_seconds
            end_seconds = mark.end_seconds
        else:
            title = str(mark["title"])
            start_seconds = float(mark["start_seconds"])
            end_seconds = float(mark["end_seconds"])

        start_ms = max(0, int(round(start_seconds * 1000)))
        end_ms = int(round(end_seconds * 1000))
        if end_ms <= start_ms:
            end_ms = start_ms + 1

        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={start_ms}")
        lines.append(f"END={end_ms}")
        lines.append(f"title={_escape_ffmetadata_value(title)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def embed_chapters(
    video_path: Path,
    chapters_json: Path,
    *,
    output_path: Path | None = None,
    min_chapters: int = 2,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Embed chapter bookmarks from `chapters_json` into `video_path`.

    Fast remux (stream copy, no re-encode)::

        ffmpeg -y -i <video> -i <ffmetadata> -map_metadata 1 -map 0 -codec copy <output>

    If `output_path` is None (the default), the video is overwritten in
    place -- via a temp file in a separate directory followed by an atomic
    move, so the source is never simultaneously read and written by the
    same ffmpeg invocation.

    If fewer than `min_chapters` chapter marks are found, this is a no-op:
    in true in-place mode (`output_path` is None) the file is left
    completely untouched; if a distinct `output_path` was requested, the
    source video is copied through unchanged. A single-chapter clip gets
    no pointless remux/bookmark.

    Returns the path of the resulting (possibly unchanged) video.
    """
    video_path = Path(video_path)
    chapters_json = Path(chapters_json)
    final_output = Path(output_path) if output_path is not None else video_path

    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not chapters_json.is_file():
        raise FileNotFoundError(f"chapters JSON not found: {chapters_json}")

    raw_cues = load_chapter_cues(chapters_json)
    chapter_marks = normalize_chapter_marks(raw_cues, video_path=video_path)

    if len(chapter_marks) < min_chapters:
        if final_output != video_path:
            final_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, final_output)
        return final_output

    ffmpeg = require_binary(ffmpeg_bin)

    with tempfile.TemporaryDirectory(prefix="embed_chapters_") as tmpdir:
        tmp_dir = Path(tmpdir)
        ffmetadata_path = tmp_dir / "chapters.ffmetadata"
        build_ffmetadata(chapter_marks, ffmetadata_path)

        tmp_output = tmp_dir / f"chaptered_{video_path.name}"
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(ffmetadata_path),
            "-map_metadata",
            "1",
            "-map",
            "0",
            "-codec",
            "copy",
            str(tmp_output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, command, output=result.stdout, stderr=result.stderr
            )

        final_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_output), str(final_output))

    return final_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed real MP4 chapter bookmarks (via ffmpeg FFMETADATA) into an "
            "existing rendered audiobook video, from a JSON list of chapter cues."
        )
    )
    parser.add_argument("--video", required=True, type=Path, help="Path to the existing rendered MP4")
    parser.add_argument(
        "--chapters-json",
        required=True,
        type=Path,
        help="JSON file with a list of chapter cues (title + start_seconds[, end_seconds])",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the chaptered video here instead of overwriting --video",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --video in place (also the default when --output is omitted)",
    )
    parser.add_argument(
        "--min-chapters",
        type=int,
        default=2,
        help="Below this many chapter marks, no-op copy-through instead of remuxing (default: 2)",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg binary to invoke (default: 'ffmpeg' resolved from PATH)",
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.in_place:
        parser.error("--output and --in-place are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result_path = embed_chapters(
            args.video,
            args.chapters_json,
            output_path=args.output,
            min_chapters=args.min_chapters,
            ffmpeg_bin=args.ffmpeg_bin,
        )
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}:\n{exc.stderr}", file=sys.stderr)
        return exc.returncode or 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"embed_video_chapters failed: {exc}", file=sys.stderr)
        return 1

    print(str(result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
