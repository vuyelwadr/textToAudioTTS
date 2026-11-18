#!/usr/bin/env python3
"""
Build local HOYS-style audiobook videos from YouTube audio.

The pipeline is intentionally resumable:
1. download best available audio and automatic caption JSON,
2. convert audio to 16 kHz mono WAV for stable ASR/chunking,
3. transcribe with faster-whisper for cleaner text than YouTube auto captions,
4. split transcript/audio into matched chunks,
5. render with scripts/make_autoscroll_video.py using active-line ASR timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYLIST_JSON = REPO_ROOT / "tasks" / "tolstoy_playlist_2026-05-02" / "playlist.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "library" / "tolstoy"
DEFAULT_INDICES = "3-5,7,9,12-26"


@dataclass(frozen=True)
class PlaylistItem:
    index: int
    video_id: str
    title: str
    duration: int
    url: str
    slug: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ChapterCue:
    start: float
    end: float | None
    title: str


def run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def yt_dlp_command() -> list[str]:
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required on PATH or importable as the yt_dlp module") from exc
    return [sys.executable, "-m", "yt_dlp"]


def yt_dlp_extra_args() -> list[str]:
    """Extra yt-dlp flags from $YTDLP_EXTRA_ARGS (shlex-split), if set.

    Used to inject provider/plugin flags for flagged networks, e.g.
    --plugin-dirs ... and the bgutil PO-token extractor args needed to
    download from YouTube when it returns HTTP 403 on media streams.
    """
    import os
    import shlex

    raw = os.environ.get("YTDLP_EXTRA_ARGS", "").strip()
    return shlex.split(raw) if raw else []


def parse_index_spec(spec: str) -> set[int]:
    selected: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


def short_title(title: str) -> str:
    text = title.strip()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\b([A-Za-z]+)'s\b", r"\1s", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .")

    anna_part = re.search(
        r"^(Anna\s+Karenina)\s*-\s*Complete\s+Audio\s+Book\s+in\s+4\s+Parts\s*-\s*Part\s+(\d+)\s+of\s+4\b",
        text,
        flags=re.IGNORECASE,
    )
    if anna_part:
        return f"{anna_part.group(1)} Part {anna_part.group(2)} of 4"

    if re.match(r"^Leo\s+Tolstoy\.\s+", text, re.IGNORECASE):
        text = re.sub(r"^Leo\s+Tolstoy\.\s+", "", text, flags=re.IGNORECASE)

    split_markers = [
        r"\.?\s+A\s+Story\b",
        r"\.?\s+A\s+Short\s+Story\b",
        r"\.?\s+A\s+Spiritual\s+Short\s+Story\b",
        r"\.?\s+A\s+Spiritual\s+Story\b",
        r"\.?\s+A\s+Dramatic\s+Short\s+Story\b",
        r"\.?\s+An\s+Insightful\s+Essay\b",
        r"\.?\s+A\s+harrowing\s+short\s+story\b",
        r"\.?\s+Excerpts\s+from\b",
        r"\.?\s+Based\s+on\b",
        r",\s+draw\s+from\b",
        r":\s+An\s+Insightful\b",
        r":\s+Leo\s+Tolstoys\b",
    ]
    for marker in split_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)
        if match:
            text = text[: match.start()].strip(" .\"")
            break

    cut_patterns = [
        r"\s+by\s+Leo\s+Tolstoy\b.*$",
        r"\s+By\s+Leo\s+Tolstoy\b.*$",
        r"\s+from\s+Leo\s+Tolstoy\b.*$",
        r"\s+Leo\s+Tolstoy's\b.*$",
        r"\s+Leo\s+Tolstoy\b.*$",
        r"\.\s+A\s+Short\s+Story\b.*$",
        r"\.\s+A\s+Spiritual\s+Story\b.*$",
        r"\.\s+An\s+Insightful\s+Essay\b.*$",
        r"\.\s+A\s+harrowing\s+short\s+story\b.*$",
    ]
    for pattern in cut_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip(" .")

    if ". " in text:
        first, rest = text.split(". ", 1)
        if len(first.split()) <= 8 and re.search(r"\b(short story|essay|classic|spiritual|leo)\b", rest, re.IGNORECASE):
            text = first

    text = text.strip('"').strip(" .")
    return text or title.strip()


def camel_slug(title: str) -> str:
    base = short_title(title)
    base = re.sub(r"\b([A-Za-z]+)'s\b", r"\1s", base)
    words = re.findall(r"[A-Za-z0-9]+", base)
    if not words:
        words = re.findall(r"[A-Za-z0-9]+", title)
    if not words:
        return "youtubeAudiobook"
    first = words[0].lower()
    rest = [word[:1].upper() + word[1:].lower() for word in words[1:]]
    return first + "".join(rest)


def load_playlist(path: Path, index_spec: str) -> list[PlaylistItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    wanted = parse_index_spec(index_spec)
    items: list[PlaylistItem] = []
    for index, entry in enumerate(data.get("entries") or [], start=1):
        if index not in wanted:
            continue
        title = str(entry.get("title") or "").strip()
        if not title or "confession" in title.lower():
            continue
        video_id = str(entry.get("id") or "").strip()
        if not video_id:
            continue
        items.append(
            PlaylistItem(
                index=index,
                video_id=video_id,
                title=title,
                duration=int(entry.get("duration") or 0),
                url=f"https://www.youtube.com/watch?v={video_id}",
                slug=camel_slug(title),
            )
        )
    return items


def write_selection(items: Iterable[PlaylistItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("index\tvideo_id\tduration_seconds\tslug\ttitle\n")
        for item in items:
            handle.write(f"{item.index}\t{item.video_id}\t{item.duration}\t{item.slug}\t{item.title}\n")


def find_downloaded_audio(audio_dir: Path, slug: str) -> Path | None:
    for path in sorted(audio_dir.glob(f"{slug}.*")):
        if path.suffix.lower() in {".json", ".json3", ".part", ".ytdl"}:
            continue
        if path.is_file():
            return path
    return None


def download_assets(item: PlaylistItem, item_dir: Path, force: bool) -> Path:
    raw_audio_dir = item_dir / "raw_audio"
    captions_dir = item_dir / "youtube_captions"
    raw_audio_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for path in raw_audio_dir.glob(f"{item.slug}.*"):
            path.unlink()

    audio_path = find_downloaded_audio(raw_audio_dir, item.slug)
    if audio_path is None:
        run(
            yt_dlp_command()
            + yt_dlp_extra_args()
            + [
                "-f",
                "bestaudio",
                "-o",
                str(raw_audio_dir / f"{item.slug}.%(ext)s"),
                item.url,
            ]
        )
        audio_path = find_downloaded_audio(raw_audio_dir, item.slug)
    if audio_path is None:
        raise RuntimeError(f"yt-dlp did not produce audio for {item.video_id}")

    caption_probe = list(captions_dir.glob(f"{item.video_id}*.json3"))
    if force or not caption_probe:
        try:
            run(
                yt_dlp_command()
                + yt_dlp_extra_args()
                + [
                    "--write-auto-subs",
                    "--write-subs",
                    "--sub-langs",
                    "en.*",
                    "--sub-format",
                    "json3",
                    "--skip-download",
                    "-o",
                    str(captions_dir / "%(id)s.%(ext)s"),
                    item.url,
                ]
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"warning: caption download failed for {item.video_id} "
                f"(rc={exc.returncode}); captions are optional unless "
                f"--transcript-source youtube-auto is used",
                flush=True,
            )
    return audio_path


def fetch_video_info(item: PlaylistItem, item_dir: Path, force: bool) -> dict[str, object]:
    info_path = item_dir / f"{item.slug}_info.json"
    if info_path.exists() and not force:
        return json.loads(info_path.read_text(encoding="utf-8"))

    command = yt_dlp_command() + yt_dlp_extra_args() + ["--skip-download", "--dump-json", item.url]
    print("+ " + " ".join(command) + f" > {info_path}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    info_path.write_text(result.stdout, encoding="utf-8")
    return json.loads(result.stdout)


def normalize_chapter_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip(" .")
    part_chapter = re.search(r"\bpart\s+(\d+)\s*,?\s+chapter\s+(\d+)\b", title, flags=re.IGNORECASE)
    if part_chapter:
        return f"Part {part_chapter.group(1)}, Chapter {part_chapter.group(2)}"
    chapter_only = re.search(r"\bchapter\s+(\d+)\b", title, flags=re.IGNORECASE)
    if chapter_only:
        return f"Chapter {chapter_only.group(1)}"
    return title


def extract_chapter_cues(info: dict[str, object], fallback_duration: int) -> list[ChapterCue]:
    raw_chapters = info.get("chapters")
    if not isinstance(raw_chapters, list):
        return []

    cues: list[ChapterCue] = []
    for raw in raw_chapters:
        if not isinstance(raw, dict):
            continue
        title = normalize_chapter_title(str(raw.get("title") or ""))
        if not title:
            continue
        try:
            start = float(raw.get("start_time") or 0.0)
        except (TypeError, ValueError):
            continue
        end_value = raw.get("end_time")
        try:
            end = float(end_value) if end_value is not None else None
        except (TypeError, ValueError):
            end = None
        cues.append(ChapterCue(start=max(0.0, start), end=end, title=title))

    cues.sort(key=lambda cue: cue.start)
    cleaned: list[ChapterCue] = []
    for index, cue in enumerate(cues):
        next_start = cues[index + 1].start if index + 1 < len(cues) else float(fallback_duration or 0)
        end = cue.end if cue.end is not None else (next_start if next_start > cue.start else None)
        cleaned.append(ChapterCue(start=cue.start, end=end, title=cue.title))
    return cleaned


def convert_audio(raw_audio: Path, wav_path: Path, force: bool) -> None:
    if wav_path.exists() and not force:
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_audio),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ]
    )


def clean_segment_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\[(music|applause|laughter)\]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text


def find_caption_json(captions_dir: Path, video_id: str) -> Path | None:
    preferred = [
        captions_dir / f"{video_id}.en-orig.json3",
        captions_dir / f"{video_id}.en.json3",
    ]
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(captions_dir.glob(f"{video_id}*.json3"))
    return matches[0] if matches else None


def load_youtube_caption_segments(
    captions_dir: Path,
    video_id: str,
    transcript_json: Path,
    *,
    force: bool,
) -> list[Segment]:
    if transcript_json.exists() and not force:
        payload = json.loads(transcript_json.read_text(encoding="utf-8"))
        if payload.get("model") == "youtube-auto-captions":
            return [Segment(**item) for item in payload["segments"]]

    caption_path = find_caption_json(captions_dir, video_id)
    if caption_path is None:
        raise RuntimeError(f"No YouTube caption JSON was found in {captions_dir} for {video_id}")

    payload = json.loads(caption_path.read_text(encoding="utf-8"))
    raw_segments: list[Segment] = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        pieces = [str(seg.get("utf8") or "") for seg in event.get("segs") or [] if isinstance(seg, dict)]
        text = clean_segment_text("".join(pieces).replace("\n", " "))
        if not text:
            continue
        try:
            start = float(event.get("tStartMs") or 0) / 1000.0
            duration = float(event.get("dDurationMs") or 0) / 1000.0
        except (TypeError, ValueError):
            continue
        end = start + duration if duration > 0 else start + 0.25
        raw_segments.append(Segment(start=start, end=end, text=text))

    raw_segments.sort(key=lambda segment: (segment.start, segment.end))
    segments: list[Segment] = []
    for index, segment in enumerate(raw_segments):
        next_start = raw_segments[index + 1].start if index + 1 < len(raw_segments) else None
        end = segment.end
        if next_start is not None and end > next_start:
            end = next_start
        end = max(segment.start + 0.05, end)
        segments.append(Segment(start=segment.start, end=end, text=segment.text))

    if not segments:
        raise RuntimeError(f"No readable caption segments were found in {caption_path}")

    transcript_json.parent.mkdir(parents=True, exist_ok=True)
    transcript_json.write_text(
        json.dumps(
            {
                "caption_json": str(caption_path),
                "model": "youtube-auto-captions",
                "language": "en",
                "segments": [asdict(segment) for segment in segments],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return segments


def transcribe_audio(
    wav_path: Path,
    transcript_json: Path,
    *,
    title: str,
    model_name: str,
    force: bool,
    device: str = "cpu",
    compute_type: str = "int8",
    initial_prompt: str | None = None,
) -> list[Segment]:
    if transcript_json.exists() and not force:
        payload = json.loads(transcript_json.read_text(encoding="utf-8"))
        return [Segment(**item) for item in payload["segments"]]

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is required for transcript generation") from exc

    if initial_prompt is None:
        initial_prompt = f"Audiobook narration. {title}"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(wav_path),
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=False,
        initial_prompt=initial_prompt,
    )

    segments: list[Segment] = []
    for segment in segments_iter:
        text = clean_segment_text(segment.text)
        if not text:
            continue
        segments.append(Segment(start=float(segment.start), end=float(segment.end), text=text))

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


def timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    ms = millis % 1000
    total_seconds = millis // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_timestamped_transcript(segments: list[Segment], path: Path, item: PlaylistItem) -> None:
    lines = [
        f"# YouTube/faster-whisper transcript",
        f"# {item.title}",
        f"# {item.url}",
        "",
    ]
    for segment in segments:
        lines.append(f"{timestamp(segment.start)} {segment.text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def paragraphize(texts: list[str], *, max_words: int = 95) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    current_words = 0

    for raw_text in texts:
        text = clean_segment_text(raw_text)
        if not text:
            continue
        current.append(text)
        current_words += word_count(text)
        if current_words >= max_words and re.search(r"[.!?\"']$", text):
            paragraphs.append(" ".join(current).strip())
            current = []
            current_words = 0

    if current:
        paragraphs.append(" ".join(current).strip())
    return "\n\n".join(paragraphs)


def paragraphize_text(texts: list[str], *, strategy: str = "naive", max_words: int = 95) -> str:
    """Dispatch to the naive paragraphizer (this module's own paragraphize(),
    the default -- zero extra dependencies) or to scripts/llm_paragraphize.py's
    LLM-assisted "llm" strategy.

    llm_paragraphize.py imports ChapterCue/Segment/paragraphize back out of
    *this* module, so it cannot be imported at module scope here without a
    circular import; it is imported lazily, only when strategy != "naive", so
    the default path never depends on it (no `anthropic` package, no API
    credentials) and this module keeps working standalone exactly as before.
    """
    if strategy == "naive":
        return paragraphize(texts, max_words=max_words)
    try:
        from llm_paragraphize import paragraphize_with_strategy
    except ImportError as exc:
        raise RuntimeError(
            f"scripts/llm_paragraphize.py is required for paragraph_strategy={strategy!r} "
            "(the naive strategy has no extra dependency and needs no such import)"
        ) from exc
    return paragraphize_with_strategy(texts, strategy=strategy)


def chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def build_chunks(
    segments: list[Segment],
    *,
    heading: str,
    chunk_seconds: float,
    chapter_cues: list[ChapterCue] | None = None,
    paragraph_strategy: str = "naive",
) -> list[tuple[int, float, float, str]]:
    chunks: list[tuple[int, float, float, str]] = []
    current: list[Segment] = []
    current_start: float | None = None
    pending_headings: list[str] = []
    chapter_index = 0
    chapters = chapter_cues or []

    def flush() -> None:
        nonlocal current, current_start, pending_headings
        if not current:
            return
        start = max(0.0, current_start if current_start is not None else current[0].start)
        end = max(segment.end for segment in current)
        text = paragraphize_text([segment.text for segment in current], strategy=paragraph_strategy)
        blocks = [f"# {title}" for title in pending_headings]
        if not chunks and not chapters:
            blocks.insert(0, f"# {heading}")
        if text:
            blocks.append(text)
        text = "\n\n".join(blocks).strip()
        chunks.append((len(chunks), start, end, text))
        current = []
        current_start = None
        pending_headings = []

    def advance_chapters(segment_start: float) -> None:
        nonlocal chapter_index, current_start
        while chapter_index < len(chapters) and chapters[chapter_index].start <= segment_start + 0.75:
            if current:
                flush()
            cue = chapters[chapter_index]
            pending_headings.append(cue.title)
            if current_start is None:
                current_start = cue.start
            chapter_index += 1

    for segment in segments:
        advance_chapters(segment.start)
        if current_start is None:
            current_start = segment.start
        current.append(segment)
        if segment.end - current_start >= chunk_seconds:
            flush()
    flush()
    return chunks


def write_chunks_and_audio(
    chunks: list[tuple[int, float, float, str]],
    *,
    source_audio: Path,
    chunks_dir: Path,
    chunk_audio_dir: Path,
    force: bool,
) -> list[dict[str, object]]:
    if force and chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    if force and chunk_audio_dir.exists():
        shutil.rmtree(chunk_audio_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_audio_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for index, start, end, text in chunks:
        digest = chunk_hash(text)
        text_path = chunks_dir / f"chunk_{index:04d}_{digest}.txt"
        audio_path = chunk_audio_dir / f"chunk_{index:04d}_{digest}.wav"
        if not text_path.exists() or force:
            text_path.write_text(text.strip() + "\n", encoding="utf-8")
        if not audio_path.exists() or force:
            duration = max(0.1, end - start)
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(source_audio),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(audio_path),
                ]
            )
        manifest.append(
            {
                "chunk_index": index,
                "start": start,
                "end": end,
                "duration_seconds": end - start,
                "text_path": str(text_path),
                "audio_path": str(audio_path),
                "word_count": word_count(text),
            }
        )
    return manifest


def join_chunk_audio(chunk_audio_dir: Path, joined_audio: Path, force: bool) -> None:
    if joined_audio.exists() and not force:
        return
    wavs = sorted(chunk_audio_dir.glob("chunk_*.wav"))
    if not wavs:
        raise RuntimeError(f"No chunk audio found in {chunk_audio_dir}")
    concat_path = joined_audio.with_suffix(".concat.txt")
    concat_path.write_text(
        "".join(f"file '{path.resolve().as_posix()}'\n" for path in wavs),
        encoding="utf-8",
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(joined_audio),
        ]
    )


def render_video(
    *,
    text_path: Path,
    audio_path: Path,
    chunks_dir: Path,
    chunk_audio_dir: Path,
    output_path: Path,
    asr_cache_dir: Path,
    force: bool,
    asr_device: str = "cpu",
    chapters_output: Path | None = None,
) -> None:
    if output_path.exists() and not force:
        return
    command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_autoscroll_video.py"),
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
            "--show-active-line",
            "--active-line-timing",
            "asr",
            "--active-line-asr-model",
            "small.en",
            "--active-line-asr-device",
            asr_device,
            "--active-line-asr-workers",
            "auto",
            "--active-line-asr-cache-dir",
            str(asr_cache_dir),
    ]
    if chapters_output is not None:
        command += ["--chapters-output", str(chapters_output)]
    run(command)


def process_item(args: argparse.Namespace, item: PlaylistItem) -> dict[str, object]:
    require_free_space(args.output_root, args.min_free_gb)
    item_dir = args.output_root / item.slug
    work_dir = item_dir / "video_work"
    item_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = item_dir / f"{item.slug}_metadata.json"
    metadata_path.write_text(json.dumps(asdict(item), indent=2), encoding="utf-8")

    info: dict[str, object] = {}
    chapter_cues: list[ChapterCue] = []
    if args.use_youtube_chapters:
        info = fetch_video_info(item, item_dir, args.force_metadata)
        chapter_cues = extract_chapter_cues(info, item.duration)

    raw_audio = download_assets(item, item_dir, args.force_download)
    wav_audio = work_dir / f"{item.slug}_audio_16k.wav"
    convert_audio(raw_audio, wav_audio, args.force_audio)

    transcript_json = item_dir / f"{item.slug}Transcript.json"
    if args.transcript_source == "youtube-auto":
        segments = load_youtube_caption_segments(
            item_dir / "youtube_captions",
            item.video_id,
            transcript_json,
            force=args.force_transcript,
        )
    else:
        segments = transcribe_audio(
            wav_audio,
            transcript_json,
            title=item.title,
            model_name=args.transcript_model,
            force=args.force_transcript,
            device=args.transcript_device,
            compute_type=args.transcript_compute_type,
            initial_prompt=args.initial_prompt,
        )

    transcript_txt = item_dir / f"{item.slug}Transcript.txt"
    write_timestamped_transcript(segments, transcript_txt, item)

    heading = short_title(item.title)
    chunks = build_chunks(
        segments,
        heading=heading,
        chunk_seconds=args.chunk_seconds,
        chapter_cues=chapter_cues,
        paragraph_strategy=getattr(args, "paragraph_strategy", "naive"),
    )
    full_text = "\n\n".join(chunk_text for _, _, _, chunk_text in chunks).strip() + "\n"
    text_path = item_dir / f"{item.slug}.txt"
    text_path.write_text(full_text, encoding="utf-8")

    chunks_dir = work_dir / "chunks"
    chunk_audio_dir = work_dir / "chunk_audio"
    manifest = write_chunks_and_audio(
        chunks,
        source_audio=wav_audio,
        chunks_dir=chunks_dir,
        chunk_audio_dir=chunk_audio_dir,
        force=args.force_chunks,
    )
    manifest_path = work_dir / "chunk_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    joined_audio = work_dir / f"{item.slug}_chunk_audio_joined.wav"
    join_chunk_audio(chunk_audio_dir, joined_audio, args.force_audio)

    output_path = item_dir / f"{item.slug}_neuralsurfer.mp4"
    chapters_output = item_dir / f"{item.slug}_chapters.json"
    if args.render:
        require_free_space(args.output_root, args.min_free_gb)
        render_video(
            text_path=text_path,
            audio_path=joined_audio,
            chunks_dir=chunks_dir,
            chunk_audio_dir=chunk_audio_dir,
            output_path=output_path,
            asr_cache_dir=work_dir / "asr_cache",
            force=args.force_render,
            asr_device=args.asr_device,
            chapters_output=chapters_output,
        )
        if output_path.exists() and chapters_output.exists():
            from embed_video_chapters import embed_chapters

            embed_chapters(output_path, chapters_output)

    summary = {
        "item": asdict(item),
        "item_dir": str(item_dir),
        "raw_audio": str(raw_audio),
        "wav_audio": str(wav_audio),
        "transcript_json": str(transcript_json),
        "transcript_source": args.transcript_source,
        "transcript_txt": str(transcript_txt),
        "text": str(text_path),
        "chunks": len(chunks),
        "youtube_chapters": len(chapter_cues),
        "joined_audio": str(joined_audio),
        "mp4": str(output_path) if output_path.exists() else None,
    }
    (item_dir / f"{item.slug}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def require_free_space(path: Path, min_free_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB free at {path}; stopping before more downloads/renders. "
            f"Minimum required by --min-free-gb is {min_free_gb:.1f} GiB."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playlist-json", type=Path, default=DEFAULT_PLAYLIST_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default=DEFAULT_INDICES)
    parser.add_argument("--only-slug")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument("--transcript-source", choices=("whisper", "youtube-auto"), default="whisper")
    parser.add_argument("--transcript-model", default="small.en")
    parser.add_argument("--transcript-device", default="cpu", help="faster-whisper device for the full-audio transcript (cpu|cuda)")
    parser.add_argument("--transcript-compute-type", default="int8", help="faster-whisper compute type (int8 for cpu, float16 for cuda)")
    parser.add_argument("--asr-device", default="cpu", help="device for the per-chunk active-line ASR inside make_autoscroll_video.py")
    parser.add_argument("--initial-prompt", default=None, help="optional whisper initial prompt; default 'Audiobook narration. <title>'")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--use-youtube-chapters", action="store_true")
    parser.add_argument(
        "--paragraph-strategy",
        choices=("naive", "llm"),
        default="naive",
        help=(
            "Within-chapter paragraph split strategy: 'naive' (fixed word-count "
            "heuristic, default, no extra dependency) or 'llm' (LLM-assisted "
            "discourse-boundary breaks via scripts/llm_paragraphize.py)"
        ),
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-metadata", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-audio", action="store_true")
    parser.add_argument("--force-transcript", action="store_true")
    parser.add_argument("--force-chunks", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.playlist_json = args.playlist_json.resolve()
    args.output_root = args.output_root.resolve()

    items = load_playlist(args.playlist_json, args.indices)
    if args.only_slug:
        items = [item for item in items if item.slug == args.only_slug]
    if args.max_items is not None:
        items = items[: args.max_items]

    selection_path = args.playlist_json.parent / "selected_items.tsv"
    write_selection(items, selection_path)
    print(f"Selected {len(items)} items. Selection: {selection_path}")
    for item in items:
        print(f"{item.index:03d} {item.slug} {item.duration}s {item.title}")

    if args.dry_run:
        return 0

    summaries = []
    progress_path = args.playlist_json.parent / "batch_progress.json"
    for item in items:
        print(f"\n=== Processing {item.index:03d} {item.slug}: {item.title} ===", flush=True)
        summaries.append(process_item(args, item))
        progress_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    batch_summary = args.playlist_json.parent / "batch_summary.json"
    batch_summary.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote batch summary: {batch_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
