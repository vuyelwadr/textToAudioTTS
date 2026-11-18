#!/usr/bin/env python3
"""Build an autoscrolling reading video from audio and structured text.

This renderer preserves headings and paragraphs, draws the whole document as
one tall page, then scrolls a video viewport over that page for the duration
of the audio.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Block:
    index: int
    kind: str
    text: str
    source_text: str
    word_count: int
    top_y: int = 0
    height: int = 0
    start: float = 0.0
    end: float = 0.0
    caption: str = ""


@dataclass
class PlateSpec:
    path: Path
    caption: str = ""
    after_text_snippet: str = ""


@dataclass
class LineLayout:
    index: int
    block_index: int
    line_index: int
    text: str
    source_start: int = 0
    source_end: int = 0
    top_y: int = 0
    height: int = 0
    width: int = 0
    word_count: int = 0
    start: float = 0.0
    end: float = 0.0


@dataclass
class LayoutBlock:
    block: Block
    font: ImageFont.ImageFont
    spacing: int
    lines: list[LineLayout]


@dataclass
class Chapter:
    index: int
    title: str
    start_offset: int
    end_offset: int
    start_block_index: int
    end_block_index: int


@dataclass
class ChunkTiming:
    chunk_index: int
    start_offset: int
    end_offset: int
    chapter_index: int
    start_time: float
    end_time: float


@dataclass
class ChapterPause:
    chapter_index: int
    previous_chapter_index: int
    gap_start: float
    end_hold_end: float
    transition_end: float
    chapter_audio_start: float


@dataclass
class TimedWord:
    text: str
    normalized: str
    start: float
    end: float


SEGMENT_FILE_PATTERN = re.compile(r"^segment_(\d+)_(\d+)\.wav$")
CHUNK_TEXT_FILE_PATTERN = re.compile(r"^chunk_(\d+)_[0-9a-f]+\.txt$")
CHUNK_AUDIO_FILE_PATTERN = re.compile(r"^chunk_(\d+)_[0-9a-f]+\.wav$")
ALIGNMENT_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z]+(?:['’][0-9A-Za-z]+)*")
ILLUSTRATION_MARKER_PATTERN = re.compile(r"\[Illustration(?::\s*([^\]]*))?\]", re.IGNORECASE | re.DOTALL)
PLATE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
IMAGE_WORD_COUNT = 16
IMAGE_PAUSE_SECONDS = IMAGE_WORD_COUNT / 8.0
_WHISPER_MODEL_CACHE: dict[tuple[str, str], object] = {}
VIDEO_CODEC_CHOICES = ("auto", "libx264", "libx265", "h264_videotoolbox", "hevc_videotoolbox")
VIDEO_PRESET_CHOICES = ("auto", "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow")


def require_binary(name: str) -> str:
    binary_path = shutil.which(name)
    if not binary_path:
        raise RuntimeError(f"Required binary '{name}' was not found on PATH")
    return binary_path


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_chunk_text_entries(chunk_dir: Path) -> list[tuple[int, str]]:
    chunk_paths: list[tuple[int, Path]] = []
    for path in chunk_dir.iterdir():
        if not path.is_file():
            continue
        match = CHUNK_TEXT_FILE_PATTERN.match(path.name)
        if not match:
            continue
        chunk_paths.append((int(match.group(1)), path))

    if not chunk_paths:
        raise RuntimeError(f"No chunk text files were found in {chunk_dir}")

    return [
        (chunk_index, normalize_text(path.read_text(encoding="utf-8")))
        for chunk_index, path in sorted(chunk_paths, key=lambda item: item[0])
    ]


def load_chunk_texts_directory(chunk_dir: Path) -> list[str]:
    return [chunk_text for _, chunk_text in load_chunk_text_entries(chunk_dir)]


def load_chunk_text_directory(chunk_dir: Path) -> str:
    return "".join(load_chunk_texts_directory(chunk_dir))


def source_text_for_chunk_subset(reference_text: str, chunk_texts: list[str]) -> str:
    normalized_reference = normalize_text(reference_text)
    if not chunk_texts:
        return normalized_reference

    try:
        spans = locate_chunk_spans(normalized_reference, chunk_texts)
    except RuntimeError:
        return "\n\n".join(chunk_texts)
    start = spans[0][0]
    end = spans[-1][1]
    return normalized_reference[start:end].strip()


def load_chunk_audio_duration_entries(chunk_dir: Path) -> list[tuple[int, float]]:
    chunk_paths: list[tuple[int, Path]] = []
    for path in chunk_dir.iterdir():
        if not path.is_file():
            continue
        match = CHUNK_AUDIO_FILE_PATTERN.match(path.name)
        if not match:
            continue
        chunk_paths.append((int(match.group(1)), path))

    if not chunk_paths:
        raise RuntimeError(f"No chunk audio files were found in {chunk_dir}")

    return [(chunk_index, probe_audio_duration(path)) for chunk_index, path in sorted(chunk_paths, key=lambda item: item[0])]


def load_chunk_duration_manifest_entries(manifest_path: Path) -> list[tuple[int, float]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Chunk duration manifest must be a non-empty list: {manifest_path}")

    entries: list[tuple[int, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError(f"Chunk duration manifest entry must be an object: {manifest_path}")
        try:
            chunk_index = int(item["chunk_index"])
            duration = float(item["duration_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Chunk duration manifest entries must contain numeric chunk_index and duration_seconds: {manifest_path}"
            ) from exc
        entries.append((chunk_index, duration))

    if not entries:
        raise RuntimeError(f"No chunk duration entries were found in {manifest_path}")
    return sorted(entries, key=lambda item: item[0])


def load_chunk_paragraph_flags(manifest_path: Path) -> dict[int, bool]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") if isinstance(payload, dict) else None
    if not isinstance(chunks, list):
        raise RuntimeError(f"Chunk manifest is missing a chunks list: {manifest_path}")

    flags: dict[int, bool] = {}
    for item in chunks:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("chunk_id", item.get("chunk_index"))
        if raw_id is None:
            continue
        flags[int(raw_id)] = bool(item.get("ends_at_paragraph", False))
    return flags


def load_chunk_audio_path_entries(chunk_dir: Path) -> list[tuple[int, Path]]:
    chunk_paths: list[tuple[int, Path]] = []
    for path in chunk_dir.iterdir():
        if not path.is_file():
            continue
        match = CHUNK_AUDIO_FILE_PATTERN.match(path.name)
        if not match:
            continue
        chunk_paths.append((int(match.group(1)), path))

    if not chunk_paths:
        raise RuntimeError(f"No chunk audio files were found in {chunk_dir}")

    return sorted(chunk_paths, key=lambda item: item[0])


def load_chunk_audio_durations(chunk_dir: Path) -> list[float]:
    return [duration for _, duration in load_chunk_audio_duration_entries(chunk_dir)]


def match_chunk_subset(
    chunk_text_entries: list[tuple[int, str]],
    chunk_audio_entries: list[tuple[int, float]],
) -> tuple[list[int], list[str], list[float]]:
    audio_by_chunk_index = dict(chunk_audio_entries)
    missing_text_ids = [chunk_index for chunk_index, _ in chunk_audio_entries if chunk_index not in dict(chunk_text_entries)]
    if missing_text_ids:
        missing_text_label = ", ".join(str(chunk_index) for chunk_index in missing_text_ids[:10])
        raise RuntimeError(f"Missing chunk text files for audio chunk ids: {missing_text_label}")

    matched: list[tuple[int, str, float]] = []
    for chunk_index, chunk_text in chunk_text_entries:
        duration = audio_by_chunk_index.get(chunk_index)
        if duration is None:
            continue
        matched.append((chunk_index, chunk_text, duration))

    if not matched:
        raise RuntimeError("No overlapping chunk ids were found between --chunk-text-dir and --chunk-audio-dir")

    ordered_chunk_indices = [chunk_index for chunk_index, _, _ in matched]
    chunk_texts = [chunk_text for _, chunk_text, _ in matched]
    chunk_durations = [duration for _, _, duration in matched]
    return ordered_chunk_indices, chunk_texts, chunk_durations


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    raw_blocks = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]

    normalized_blocks: list[str] = []
    for raw_block in raw_blocks:
        normalized_block = re.sub(r"\s*\n\s*", " ", raw_block)
        normalized_block = re.sub(r"[ \t]+", " ", normalized_block)
        normalized_blocks.append(normalized_block.strip())

    return "\n\n".join(normalized_blocks)


def normalize_alignment_token(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token.replace("’", "'").replace("`", "'"))
    stripped = "".join(character for character in normalized if not unicodedata.combining(character))
    stripped = re.sub(r"[^0-9A-Za-z']+", "", stripped).strip("'").lower()
    return stripped


def tokenize_alignment_text(text: str) -> list[str]:
    tokens = [normalize_alignment_token(match.group(0)) for match in ALIGNMENT_TOKEN_PATTERN.finditer(text)]
    return [token for token in tokens if token]


def build_line_token_ranges(line_layouts: list[LineLayout]) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    ranges: list[tuple[int, int]] = []
    for line in line_layouts:
        start_index = len(tokens)
        tokens.extend(tokenize_alignment_text(line.text))
        ranges.append((start_index, len(tokens)))
    return tokens, ranges


def resolve_visible_cuda_devices() -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices and visible_devices not in {"-1", "none", "None"}:
        return [token.strip() for token in visible_devices.split(",") if token.strip()]

    try:
        import torch
    except Exception:
        return []

    if not torch.cuda.is_available():
        return []
    return [str(index) for index in range(torch.cuda.device_count())]


def resolve_asr_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if resolve_visible_cuda_devices():
        return "cuda"
    return "cpu"


def resolve_asr_worker_count(requested: str | None, device: str, task_count: int) -> int:
    if task_count <= 1:
        return 1

    raw_value = (requested or "auto").strip().lower()
    if raw_value != "auto":
        return max(1, min(int(raw_value), task_count))

    if device == "cuda":
        return min(max(1, len(resolve_visible_cuda_devices())), task_count)

    cpu_count = os.cpu_count() or 1
    return min(task_count, max(1, min(cpu_count - 1, 8)))


def build_asr_cache_path(audio_path: Path, model_name: str, cache_dir: Path | None) -> Path:
    safe_model = re.sub(r"[^0-9A-Za-z._-]+", "_", model_name)
    cache_name = f"{audio_path.stem}.active_line_asr.{safe_model}.json"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / cache_name
    return audio_path.with_name(cache_name)


@dataclass
class _OpenAIWhisperWord:
    word: str
    start: float | None
    end: float | None


class _OpenAIWhisperSegment:
    def __init__(self, raw_segment: dict) -> None:
        self.words = [
            _OpenAIWhisperWord(word=w.get("word", ""), start=w.get("start"), end=w.get("end"))
            for w in raw_segment.get("words", []) or []
        ]


class _OpenAIWhisperAdapter:
    """Stands in for faster-whisper's WhisperModel when ctranslate2 has no
    CUDA build for this platform (e.g. Roihu's aarch64 GH200 nodes, where
    PyPI's ctranslate2 wheel is CPU-only). Uses the plain PyTorch-based
    openai-whisper package instead, which shares PyTorch's own CUDA support."""

    def __init__(self, model_name: str, device: str) -> None:
        import whisper

        self._model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio_path: str, *, beam_size: int = 5, **_ignored_kwargs):
        result = self._model.transcribe(
            audio_path,
            word_timestamps=True,
            condition_on_previous_text=False,
            beam_size=beam_size,
        )
        segments = [_OpenAIWhisperSegment(segment) for segment in result.get("segments", [])]
        return segments, None


def _load_whisper_model(model_name: str, device: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is required for audio-aligned active-line timing") from exc

    compute_type = "float16" if device == "cuda" else "int8"
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except ValueError as exc:
        if device != "cuda" or "CUDA support" not in str(exc):
            raise
        return _OpenAIWhisperAdapter(model_name, device=device)


def transcribe_audio_words(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    cache_dir: Path | None = None,
) -> list[TimedWord]:
    cache_path = build_asr_cache_path(audio_path, model_name, cache_dir)
    audio_stat = audio_path.stat()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("audio_size") == audio_stat.st_size
            and cached.get("audio_mtime_ns") == audio_stat.st_mtime_ns
            and cached.get("model_name") == model_name
        ):
            words = cached.get("words", [])
            if words:
                return [TimedWord(**item) for item in words]

    cache_key = (model_name, device)
    model = _WHISPER_MODEL_CACHE.get(cache_key)
    if model is None:
        model = _load_whisper_model(model_name, device)
        _WHISPER_MODEL_CACHE[cache_key] = model
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=True,
    )

    timed_words: list[TimedWord] = []
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            if word.start is None or word.end is None:
                continue
            normalized = normalize_alignment_token(word.word)
            if not normalized:
                continue
            timed_words.append(
                TimedWord(
                    text=word.word.strip(),
                    normalized=normalized,
                    start=float(word.start),
                    end=float(word.end),
                )
            )

    if not timed_words:
        raise RuntimeError(f"no timed words were produced for {audio_path}")

    cache_payload = {
        "audio_size": audio_stat.st_size,
        "audio_mtime_ns": audio_stat.st_mtime_ns,
        "model_name": model_name,
        "words": [asdict(word) for word in timed_words],
    }
    cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    return timed_words


def fill_missing_word_timings(
    word_starts: list[float | None],
    word_ends: list[float | None],
    expected_tokens: list[str],
    duration: float,
) -> tuple[list[float], list[float]]:
    known_indices = [index for index, value in enumerate(word_starts) if value is not None and word_ends[index] is not None]
    if not known_indices:
        raise RuntimeError("no aligned words were available to estimate line timing")

    resolved_starts = list(word_starts)
    resolved_ends = list(word_ends)

    def distribute(indexes: list[int], start_time: float, end_time: float) -> None:
        if not indexes:
            return
        if end_time <= start_time:
            average_duration = max(0.06, duration / max(1, len(expected_tokens)))
            end_time = start_time + average_duration * len(indexes)
        weights = [max(1, len(expected_tokens[index])) for index in indexes]
        total_weight = sum(weights)
        cursor = start_time
        span = max(0.01 * len(indexes), end_time - start_time)
        for offset, index in enumerate(indexes):
            if offset == len(indexes) - 1:
                resolved_starts[index] = cursor
                resolved_ends[index] = end_time
                continue
            slice_duration = span * (weights[offset] / total_weight) if total_weight else span / len(indexes)
            resolved_starts[index] = cursor
            resolved_ends[index] = min(end_time, cursor + slice_duration)
            cursor = resolved_ends[index]

    first_known = known_indices[0]
    distribute(list(range(0, first_known)), 0.0, float(resolved_starts[first_known]))

    for previous_index, next_index in zip(known_indices, known_indices[1:]):
        gap_indexes = list(range(previous_index + 1, next_index))
        distribute(gap_indexes, float(resolved_ends[previous_index]), float(resolved_starts[next_index]))

    last_known = known_indices[-1]
    distribute(list(range(last_known + 1, len(expected_tokens))), float(resolved_ends[last_known]), duration)

    return [float(value or 0.0) for value in resolved_starts], [float(value or 0.0) for value in resolved_ends]


def align_expected_tokens_to_timed_words(
    expected_tokens: list[str],
    timed_words: list[TimedWord],
    duration: float,
) -> tuple[list[float], list[float], float]:
    actual_tokens = [word.normalized for word in timed_words]
    matcher = SequenceMatcher(a=expected_tokens, b=actual_tokens, autojunk=False)
    word_starts: list[float | None] = [None] * len(expected_tokens)
    word_ends: list[float | None] = [None] * len(expected_tokens)
    match_count = 0

    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            expected_index = block.a + offset
            actual_index = block.b + offset
            word_starts[expected_index] = timed_words[actual_index].start
            word_ends[expected_index] = timed_words[actual_index].end
            match_count += 1

    match_ratio = match_count / max(1, len(expected_tokens))
    starts, ends = fill_missing_word_timings(word_starts, word_ends, expected_tokens, duration)
    return starts, ends, match_ratio


def assign_line_timing_from_audio_asr(
    line_layouts: list[LineLayout],
    audio_path: Path,
    *,
    duration: float,
    model_name: str,
    device: str,
    cache_dir: Path | None = None,
    time_offset: float = 0.0,
) -> tuple[list[LineLayout], float]:
    if not line_layouts:
        return [], 0.0

    expected_tokens, line_ranges = build_line_token_ranges(line_layouts)
    if not expected_tokens:
        return line_layouts, 0.0

    timed_words = transcribe_audio_words(audio_path, model_name=model_name, device=device, cache_dir=cache_dir)
    word_starts, word_ends, match_ratio = align_expected_tokens_to_timed_words(expected_tokens, timed_words, duration)
    if match_ratio < 0.35:
        raise RuntimeError(f"ASR alignment only matched {match_ratio:.1%} of expected words")

    resolved_lines: list[LineLayout] = []
    previous_end = time_offset
    for line, (start_index, end_index) in zip(line_layouts, line_ranges):
        if end_index <= start_index:
            line.start = previous_end
            line.end = previous_end
        else:
            line_start = time_offset + word_starts[start_index]
            line_end = time_offset + word_ends[end_index - 1]
            line.start = max(previous_end, line_start)
            line.end = max(line.start + 0.01, line_end)
            previous_end = line.end
        resolved_lines.append(line)

    if resolved_lines:
        resolved_lines[-1].end = max(resolved_lines[-1].end, time_offset + duration)

    return resolved_lines, match_ratio


def assign_line_timing_within_span(
    line_layouts: list[LineLayout],
    *,
    start_time: float,
    end_time: float,
) -> list[LineLayout]:
    if not line_layouts:
        return []

    cursor = start_time
    total_weight = sum(max(1, line.word_count) for line in line_layouts)
    remaining_weight = total_weight
    resolved_lines: list[LineLayout] = []

    for index, line in enumerate(line_layouts):
        weight = max(1, line.word_count)
        if index == len(line_layouts) - 1:
            line.start = cursor
            line.end = max(line.start + 0.01, end_time)
        else:
            remaining_span = max(0.01, end_time - cursor)
            slice_duration = remaining_span * (weight / max(1, remaining_weight))
            line.start = cursor
            line.end = max(line.start + 0.01, min(end_time, cursor + slice_duration))
            cursor = line.end
            remaining_weight -= weight
        resolved_lines.append(line)

    return resolved_lines


def _align_chunk_asr_shard(
    chunk_tasks: list[dict[str, object]],
    *,
    model_name: str,
    device: str,
    cache_dir: str | None,
    cuda_visible_device: str | None,
) -> list[dict[str, object]]:
    if device == "cuda" and cuda_visible_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
    elif device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    global _WHISPER_MODEL_CACHE
    _WHISPER_MODEL_CACHE = {}

    shard_results: list[dict[str, object]] = []
    cache_dir_path = Path(cache_dir) if cache_dir else None

    for task in chunk_tasks:
        chunk_lines = [LineLayout(**line_layout) for line_layout in task["chunk_lines"]]
        chunk_duration = float(task["chunk_duration"])
        time_offset = float(task["time_offset"])
        try:
            aligned_lines, match_ratio = assign_line_timing_from_audio_asr(
                chunk_lines,
                Path(task["chunk_audio_path"]),
                duration=chunk_duration,
                model_name=model_name,
                device=device,
                cache_dir=cache_dir_path,
                time_offset=time_offset,
            )
        except RuntimeError as exc:
            print(
                f"warning: chunk {int(task['chunk_index'])} fell back to chunk timing: {exc}",
                file=sys.stderr,
                flush=True,
            )
            aligned_lines = assign_line_timing_within_span(
                chunk_lines,
                start_time=time_offset,
                end_time=time_offset + chunk_duration,
            )
            match_ratio = 0.0
        shard_results.append(
            {
                "chunk_index": int(task["chunk_index"]),
                "expected_word_total": int(task["expected_word_total"]),
                "match_ratio": float(match_ratio),
                "aligned_lines": [asdict(line) for line in aligned_lines],
            }
        )

    return shard_results


def is_heading_block(text: str) -> bool:
    """Return whether a block should be *styled* as a heading.

    Styling is intentionally broader than chapter detection: short all-caps
    source headings may still get the heading font, but they must not create
    timeline chapter pauses.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if re.match(r"^(chapter|part|section)\b", stripped, re.IGNORECASE) and len(stripped.split()) <= 10:
        return True
    if re.search(r"\bWAR IS PEACE\b|\bFREEDOM IS SLAVERY\b|\bIGNORANCE IS STRENGTH\b", stripped):
        return False
    if re.match(r"^(DOWN WITH BIG BROTHER|BIG BROTHER IS WATCHING YOU)\b", stripped, re.IGNORECASE):
        return False
    words = stripped.split()
    return len(words) <= 10 and stripped.upper() == stripped and any(char.isalpha() for char in stripped)


def is_chapter_heading_block(text: str) -> bool:
    """Return whether a block is a real chapter boundary.

    Do NOT treat arbitrary short ALL-CAPS blocks as chapters. That old
    heuristic turned title-page fragments, roman numerals and essay headings
    into hundreds of fake chapters and injected seconds of silence at each one.
    Markdown headings are explicit structure and are therefore authoritative;
    plain-text PART/CHAPTER headings remain supported for books such as 1984.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if re.match(r"^#{1,6}\s+\S", stripped):
        return True
    return bool(re.match(r"^(?:part|chapter)\b", stripped, re.IGNORECASE) and len(stripped.split()) <= 14)


def parse_blocks(text: str) -> list[Block]:
    normalized = normalize_text(text)
    raw_blocks = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    blocks: list[Block] = []
    for index, raw_block in enumerate(raw_blocks):
        block_text = raw_block.replace("#", "").strip()
        kind = "heading" if is_heading_block(raw_block) else "paragraph"
        blocks.append(
            Block(
                index=index,
                kind=kind,
                text=block_text,
                source_text=raw_block,
                word_count=max(1, len(re.findall(r"\S+", block_text))),
            )
        )
    if not blocks:
        raise RuntimeError("No readable blocks were produced from the input text")
    return blocks


def natural_sort_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def is_logo_name(name: str) -> bool:
    return "logo" in Path(name).stem.lower()


def strip_illustration_markers(text: str) -> str:
    stripped = ILLUSTRATION_MARKER_PATTERN.sub("", text)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_illustration_markers(markers_text: str) -> list[dict[str, str]]:
    markers: list[dict[str, str]] = []
    for match in ILLUSTRATION_MARKER_PATTERN.finditer(markers_text):
        caption = collapse_whitespace(match.group(1) or "")
        before = markers_text[: match.start()]
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", before) if part.strip()]
        while paragraphs:
            candidate = ILLUSTRATION_MARKER_PATTERN.sub("", paragraphs[-1]).strip()
            if candidate:
                paragraphs[-1] = candidate
                break
            paragraphs.pop()
        snippet = collapse_whitespace(paragraphs[-1]) if paragraphs else ""
        if len(snippet) > 220:
            snippet = snippet[-220:]
        markers.append({"caption": caption, "after_text_snippet": snippet})
    return markers


def _normalize_fig_token(token: str) -> str:
    token = token.lower().replace(" ", "")
    token = token.replace("–", "-").replace("—", "-")
    return token


def extract_fig_keys(text: str) -> set[str]:
    keys: set[str] = set()
    if not text:
        return keys
    lowered = text.lower()
    if "frontispiece" in lowered or "colorchart" in lowered or "colour chart" in lowered or "meaning of the colour" in lowered:
        keys.add("frontispiece")
    for match in re.finditer(
        r"figs?\.?\s*([0-9]+[a-z]?(?:\s*[-–—]\s*[0-9]+[a-z]?)?)",
        text,
        flags=re.IGNORECASE,
    ):
        keys.add(_normalize_fig_token(match.group(1)))
    for match in re.finditer(r"figs?([0-9]+[a-z]?(?:[-–—][0-9]+[a-z]?)?)", text, flags=re.IGNORECASE):
        keys.add(_normalize_fig_token(match.group(1)))
    for match in re.finditer(r"plate\s*\.?\s*([a-z0-9]+)", text, flags=re.IGNORECASE):
        keys.add(f"plate:{match.group(1).lower()}")
    stem_match = re.fullmatch(r"fig([a-z])", Path(text).stem.lower()) if "/" not in text and " " not in text else None
    if stem_match:
        keys.add(f"plate:{stem_match.group(1)}")
    return keys


def expand_fig_keys(keys: set[str]) -> set[str]:
    expanded = set(keys)
    for key in list(keys):
        if key.startswith("plate:") or key == "frontispiece":
            continue
        range_match = re.fullmatch(r"([0-9]+)[a-z]?-([0-9]+)[a-z]?", key)
        if not range_match:
            continue
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if end < start or end - start > 20:
            continue
        for value in range(start, end + 1):
            expanded.add(str(value))
    return expanded


def list_plate_images_from_dir(plates_dir: Path) -> list[Path]:
    plates: list[Path] = []
    for path in plates_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in PLATE_IMAGE_EXTENSIONS:
            continue
        if is_logo_name(path.name):
            continue
        plates.append(path)
    return sorted(plates, key=lambda path: natural_sort_key(path.name))


def list_plate_images_from_html(book_dir: Path, plates_dir: Path) -> list[Path] | None:
    for candidate_name in ("source.htm", "source.html"):
        source_path = book_dir / candidate_name
        if source_path.exists():
            break
    else:
        return None

    html = source_path.read_text(encoding="utf-8", errors="ignore")
    srcs = re.findall(r"""<img[^>]+src=["']([^"']+)["']""", html, flags=re.IGNORECASE)
    if not srcs:
        return None

    by_lower_name = {path.name.lower(): path for path in list_plate_images_from_dir(plates_dir)}
    # Include logo-skipped files only for lookup of non-logo HTML refs; rebuild full name map.
    for path in plates_dir.iterdir():
        if path.is_file() and path.suffix.lower() in PLATE_IMAGE_EXTENSIONS:
            by_lower_name.setdefault(path.name.lower(), path)

    ordered: list[Path] = []
    seen: set[Path] = set()
    for src in srcs:
        name = Path(src).name
        if is_logo_name(name):
            continue
        path = by_lower_name.get(name.lower())
        if path is None or path in seen:
            continue
        ordered.append(path)
        seen.add(path)
    return ordered or None


def resolve_plate_images(plates_dir: Path, book_dir: Path | None = None) -> list[Path]:
    search_dir = book_dir or plates_dir.parent
    html_plates = list_plate_images_from_html(search_dir, plates_dir)
    if html_plates:
        return html_plates
    return list_plate_images_from_dir(plates_dir)


def resolve_markers_path(text_path: Path | None) -> Path | None:
    if text_path is None:
        return None
    sibling = text_path.with_name(f"{text_path.stem}_with_illustration_markers.txt")
    if sibling.exists():
        return sibling
    return None


def match_plates_to_markers(plates: list[Path], markers: list[dict[str, str]]) -> list[PlateSpec]:
    usable_markers = [
        marker
        for marker in markers
        if not is_logo_name(marker.get("caption", "")) and "publisher logo" not in marker.get("caption", "").lower()
    ]
    used_marker_indexes: set[int] = set()
    matched: list[PlateSpec] = []

    for plate in plates:
        plate_keys = expand_fig_keys(extract_fig_keys(plate.stem) | extract_fig_keys(plate.name))
        matched_index: int | None = None
        for index, marker in enumerate(usable_markers):
            if index in used_marker_indexes:
                continue
            marker_keys = expand_fig_keys(extract_fig_keys(marker.get("caption", "")))
            if plate_keys and marker_keys and plate_keys & marker_keys:
                matched_index = index
                break
        if matched_index is None:
            for index, marker in enumerate(usable_markers):
                if index not in used_marker_indexes:
                    matched_index = index
                    break
        if matched_index is None:
            matched.append(PlateSpec(path=plate, caption="", after_text_snippet=""))
            continue

        marker = usable_markers[matched_index]
        used_marker_indexes.add(matched_index)
        if plate_keys:
            expanded_plate_keys = expand_fig_keys(plate_keys)
            for index, other in enumerate(usable_markers):
                if index in used_marker_indexes:
                    continue
                other_keys = expand_fig_keys(extract_fig_keys(other.get("caption", "")))
                if other_keys and other_keys <= expanded_plate_keys:
                    used_marker_indexes.add(index)
        matched.append(
            PlateSpec(
                path=plate,
                caption=marker.get("caption", ""),
                after_text_snippet=marker.get("after_text_snippet", ""),
            )
        )
    return matched


def locate_snippet_offset(clean_text: str, snippet: str) -> int:
    normalized = normalize_text(clean_text)
    needle = collapse_whitespace(snippet)
    if not needle:
        return 0

    direct = normalized.find(needle)
    if direct >= 0:
        return direct + len(needle)

    for length in (180, 140, 100, 70, 40):
        if len(needle) < length:
            continue
        piece = needle[-length:]
        found = normalized.find(piece)
        if found >= 0:
            return found + len(piece)

    best_ratio = 0.0
    best_end = 0
    window = min(len(needle), 160)
    probe = needle[-window:]
    step = max(1, window // 4)
    for start in range(0, max(1, len(normalized) - len(probe) + 1), step):
        candidate = normalized[start : start + len(probe)]
        ratio = SequenceMatcher(None, probe, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_end = start + len(candidate)
    if best_ratio >= 0.72:
        return best_end
    return 0


def build_plates_manifest(
    plates_dir: Path,
    *,
    markers_text: str | None = None,
    markers_path: Path | None = None,
    book_dir: Path | None = None,
) -> dict[str, object]:
    if markers_text is None and markers_path is not None:
        markers_text = markers_path.read_text(encoding="utf-8")
    if markers_text is None:
        raise RuntimeError("markers_text or markers_path is required to build a plates manifest")

    resolved_book_dir = book_dir or plates_dir.parent
    plates = resolve_plate_images(plates_dir, resolved_book_dir)
    markers = parse_illustration_markers(markers_text)
    specs = match_plates_to_markers(plates, markers)
    return {
        "plates": [
            {
                "path": str(spec.path.resolve()),
                "caption": spec.caption,
                "after_text_snippet": spec.after_text_snippet,
            }
            for spec in specs
        ]
    }


def load_plates_manifest(manifest_path: Path) -> list[PlateSpec]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_plates = payload.get("plates") if isinstance(payload, dict) else None
    if not isinstance(raw_plates, list) or not raw_plates:
        raise RuntimeError(f"Plates manifest must contain a non-empty plates list: {manifest_path}")

    specs: list[PlateSpec] = []
    for item in raw_plates:
        if not isinstance(item, dict) or "path" not in item:
            raise RuntimeError(f"Each plates manifest entry must include path: {manifest_path}")
        path = Path(str(item["path"])).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        if not path.exists():
            raise RuntimeError(f"Plate image not found: {path}")
        specs.append(
            PlateSpec(
                path=path,
                caption=str(item.get("caption") or ""),
                after_text_snippet=str(item.get("after_text_snippet") or ""),
            )
        )
    return specs


def resolve_plates_for_render(
    *,
    plates_dir: Path | None,
    plates_manifest: Path | None,
    text_path: Path | None,
    source_text: str,
) -> tuple[list[PlateSpec], str]:
    """Return plate specs and speech-safe source text.

    Speech timing always uses clean text without ``[Illustration...]`` markers.
    """
    if plates_dir is None and plates_manifest is None:
        return [], source_text

    speech_text = source_text
    markers_text: str | None = None

    if ILLUSTRATION_MARKER_PATTERN.search(source_text):
        markers_text = source_text
        speech_text = normalize_text(strip_illustration_markers(source_text))

    markers_path = resolve_markers_path(text_path)
    if markers_text is None and markers_path is not None:
        markers_text = markers_path.read_text(encoding="utf-8")

    if plates_manifest is not None:
        specs = load_plates_manifest(plates_manifest)
        return specs, speech_text

    assert plates_dir is not None
    if not plates_dir.exists():
        raise RuntimeError(f"Plates directory not found: {plates_dir}")
    if markers_text is None:
        raise RuntimeError(
            "Illustrated mode needs --plates-manifest, illustration markers in --text, "
            "or a sibling *_with_illustration_markers.txt next to --text"
        )

    manifest = build_plates_manifest(
        plates_dir,
        markers_text=markers_text,
        book_dir=plates_dir.parent if text_path is None else text_path.parent,
    )
    specs = [
        PlateSpec(
            path=Path(str(item["path"])),
            caption=str(item.get("caption") or ""),
            after_text_snippet=str(item.get("after_text_snippet") or ""),
        )
        for item in manifest["plates"]  # type: ignore[index]
    ]
    return specs, speech_text


def inject_image_blocks(blocks: list[Block], plates: list[PlateSpec], clean_text: str) -> list[Block]:
    if not plates:
        return blocks

    spans = build_block_spans(blocks)
    insertions: dict[int, list[PlateSpec]] = {}
    for plate in plates:
        if not plate.path.exists():
            raise RuntimeError(f"Plate image not found: {plate.path}")
        offset = locate_snippet_offset(clean_text, plate.after_text_snippet)
        host_index = blocks[0].index if blocks else 0
        for start, end, block in spans:
            if start <= offset <= end:
                host_index = block.index
                break
            if offset >= end:
                host_index = block.index
        insertions.setdefault(host_index, []).append(plate)

    merged: list[Block] = []
    for block in blocks:
        merged.append(block)
        for plate in insertions.get(block.index, []):
            merged.append(
                Block(
                    index=-1,
                    kind="image",
                    text=str(plate.path),
                    source_text="",
                    word_count=IMAGE_WORD_COUNT,
                    caption=plate.caption,
                )
            )

    for index, block in enumerate(merged):
        block.index = index
    return merged


def probe_audio_duration(audio_path: Path) -> float:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        command = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
    else:
        info = sf.info(str(audio_path))
        duration = float(info.duration)
    if duration <= 0:
        raise RuntimeError(f"Audio duration must be positive, got {duration}")
    return duration


def load_kokoro_chunk_texts(source_text: str) -> list[str]:
    try:
        from audioChunkedKukuro import LANG_CODE, improved_chunk_text
    except Exception as exc:
        raise RuntimeError(f"Could not import Kokoro chunker: {exc}") from exc

    return [chunk_text for chunk_text, _ in improved_chunk_text(source_text, LANG_CODE)]


def load_kokoro_segment_groups(segment_dir: Path) -> dict[int, list[Path]]:
    grouped_segments: dict[int, list[tuple[int, Path]]] = {}
    for path in segment_dir.iterdir():
        if not path.is_file():
            continue
        match = SEGMENT_FILE_PATTERN.match(path.name)
        if not match:
            continue
        chunk_index = int(match.group(1))
        part_index = int(match.group(2))
        grouped_segments.setdefault(chunk_index, []).append((part_index, path))

    if not grouped_segments:
        raise RuntimeError(f"No Kokoro segment files were found in {segment_dir}")

    ordered_groups: dict[int, list[Path]] = {}
    for chunk_index in sorted(grouped_segments):
        ordered_groups[chunk_index] = [path for _, path in sorted(grouped_segments[chunk_index], key=lambda item: item[0])]

    return ordered_groups


def probe_kokoro_chunk_durations(segment_groups: dict[int, list[Path]]) -> list[float]:
    durations: list[float] = []
    for chunk_index in sorted(segment_groups):
        total_duration = 0.0
        for path in segment_groups[chunk_index]:
            total_duration += probe_audio_duration(path)
        durations.append(total_duration)

    return durations


def build_block_spans(blocks: list[Block]) -> list[tuple[int, int, Block]]:
    spans: list[tuple[int, int, Block]] = []
    cursor = 0
    for block in blocks:
        if block.kind == "image":
            # Zero-width: keep speech character offsets aligned with clean TTS text.
            spans.append((cursor, cursor, block))
            continue
        start = cursor
        end = start + len(block.source_text)
        spans.append((start, end, block))
        cursor = end + 2
    return spans


def build_chapters(block_spans: list[tuple[int, int, Block]]) -> list[Chapter]:
    chapters: list[Chapter] = []
    current_chapter: Chapter | None = None

    for start_offset, end_offset, block in block_spans:
        if block.kind == "image":
            if current_chapter is not None:
                current_chapter.end_block_index = block.index
            continue
        chapter_boundary = is_chapter_heading_block(block.source_text)
        if chapter_boundary or current_chapter is None:
            if current_chapter is not None:
                chapters.append(current_chapter)

            title = block.text if chapter_boundary else f"Section {len(chapters) + 1}"
            current_chapter = Chapter(
                index=len(chapters),
                title=title,
                start_offset=start_offset,
                end_offset=end_offset,
                start_block_index=block.index,
                end_block_index=block.index,
            )

        current_chapter.end_offset = end_offset
        current_chapter.end_block_index = block.index

    if current_chapter is not None:
        chapters.append(current_chapter)

    return chapters


def find_chapter_index(chapters: list[Chapter], offset: int) -> int:
    if not chapters:
        return 0

    for chapter in chapters:
        if chapter.start_offset <= offset <= chapter.end_offset:
            return chapter.index

    if offset < chapters[0].start_offset:
        return chapters[0].index

    return chapters[-1].index


def locate_chunk_spans(normalized_text: str, chunk_texts: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    search_start = 0
    for chunk_text in chunk_texts:
        chunk_start = normalized_text.find(chunk_text, search_start)
        if chunk_start < 0:
            raise RuntimeError("Could not map a Kokoro chunk back to the source document")
        chunk_end = chunk_start + len(chunk_text)
        spans.append((chunk_start, chunk_end))
        search_start = chunk_end
    return spans


def build_chunk_timeline(
    chunk_indices: list[int],
    chunk_spans: list[tuple[int, int]],
    chunk_durations: list[float],
    chapters: list[Chapter],
    chapter_end_hold: float,
    chapter_transition: float,
    chapter_start_hold: float,
    *,
    paragraph_pause_after: dict[int, bool] | None = None,
    paragraph_pause_seconds: float = 0.0,
) -> tuple[list[ChunkTiming], list[ChapterPause], float]:
    if not (len(chunk_indices) == len(chunk_spans) == len(chunk_durations)):
        raise RuntimeError("Chunk indices, spans, and durations must have matching lengths")

    chunk_timings: list[ChunkTiming] = []
    chapter_pauses: list[ChapterPause] = []
    cursor = 0.0
    previous_chapter_index: int | None = None

    for chunk_index, (start_offset, end_offset), duration in zip(chunk_indices, chunk_spans, chunk_durations):
        chapter_index = find_chapter_index(chapters, int((start_offset + end_offset) / 2))

        if previous_chapter_index is not None and chapter_index != previous_chapter_index:
            gap_start = cursor
            end_hold_end = gap_start + chapter_end_hold
            transition_end = end_hold_end + chapter_transition
            chapter_audio_start = transition_end + chapter_start_hold
            chapter_pauses.append(
                ChapterPause(
                    chapter_index=chapter_index,
                    previous_chapter_index=previous_chapter_index,
                    gap_start=gap_start,
                    end_hold_end=end_hold_end,
                    transition_end=transition_end,
                    chapter_audio_start=chapter_audio_start,
                )
            )
            cursor = chapter_audio_start

        chunk_start_time = cursor
        chunk_end_time = chunk_start_time + duration
        chunk_timings.append(
            ChunkTiming(
                chunk_index=chunk_index,
                start_offset=start_offset,
                end_offset=end_offset,
                chapter_index=chapter_index,
                start_time=chunk_start_time,
                end_time=chunk_end_time,
            )
        )
        cursor = chunk_end_time
        if (
            paragraph_pause_after
            and paragraph_pause_after.get(chunk_index, False)
            and chunk_index != chunk_indices[-1]
        ):
            cursor += max(0.0, paragraph_pause_seconds)
        previous_chapter_index = chapter_index

    return chunk_timings, chapter_pauses, cursor


def build_chapter_chunk_boundaries(chunk_timings: list[ChunkTiming]) -> tuple[dict[int, int], dict[int, int]]:
    first_chunks: dict[int, int] = {}
    last_chunks: dict[int, int] = {}

    for chunk_timing in chunk_timings:
        first_chunks.setdefault(chunk_timing.chapter_index, chunk_timing.chunk_index)
        last_chunks[chunk_timing.chapter_index] = chunk_timing.chunk_index

    return first_chunks, last_chunks


def build_chapter_marks(
    chapters: list[Chapter],
    *,
    chunk_timings: list[ChunkTiming] | None = None,
    blocks: list[Block] | None = None,
    total_duration: float | None = None,
) -> list[dict]:
    """Build real start/end-timestamped chapter marks from the Chapter objects
    already computed during rendering, for ``--chapters-output``.

    Prefers exact per-chunk timing (``chunk_timings``, populated whenever
    --chunk-audio-dir / --chunk-duration-manifest / --kokoro-segment-dir drives
    an exact_timing_payload render): those ``start_time``/``end_time`` values
    are already absolute seconds into the final rendered audio (chapter-pause
    gaps included -- see build_chunk_timeline()). Falls back to the coarser
    per-block ``.start``/``.end`` timing assigned by assign_block_timing()
    when only duration-weighted timing was available (plain --audio/--text
    mode with no chunk timing source).

    Returns a flat list of {"index", "title", "start_seconds", "end_seconds"}
    dicts -- the shape scripts/embed_video_chapters.py's load_chapter_cues()/
    normalize_chapter_marks() already know how to consume directly.
    """
    marks: list[dict] = []

    if chunk_timings:
        bounds: dict[int, list[float]] = {}
        for timing in chunk_timings:
            bound = bounds.setdefault(timing.chapter_index, [timing.start_time, timing.end_time])
            bound[0] = min(bound[0], timing.start_time)
            bound[1] = max(bound[1], timing.end_time)
        for chapter in chapters:
            bound = bounds.get(chapter.index)
            if bound is None:
                continue
            marks.append(
                {
                    "index": chapter.index,
                    "title": chapter.title,
                    "start_seconds": bound[0],
                    "end_seconds": bound[1],
                }
            )
    elif blocks:
        for chapter in chapters:
            start_block = blocks[chapter.start_block_index]
            end_block = blocks[chapter.end_block_index]
            marks.append(
                {
                    "index": chapter.index,
                    "title": chapter.title,
                    "start_seconds": start_block.start,
                    "end_seconds": end_block.end,
                }
            )

    if marks and total_duration is not None:
        marks[-1]["end_seconds"] = max(marks[-1]["end_seconds"], total_duration)

    return marks


def resolve_scroll_target(
    offset: int,
    block_spans: list[tuple[int, int, Block]],
    viewport_height: int,
    max_scroll: float,
    focus_line_ratio: float,
) -> float:
    if not block_spans:
        return 0.0

    for start, end, block in block_spans:
        if start <= offset <= end:
            if end <= start:
                ratio = 0.0
            else:
                ratio = (offset - start) / float(end - start)
            raw_target = float(block.top_y + int(block.height * ratio) - int(viewport_height * focus_line_ratio))
            return min(max(raw_target, 0.0), max_scroll)

    if offset < block_spans[0][0]:
        return 0.0
    return max_scroll


def resolve_chapter_start_target(
    chapter: Chapter,
    blocks: list[Block],
    viewport_height: int,
    max_scroll: float,
    chapter_top_margin_ratio: float,
) -> float:
    start_block = blocks[chapter.start_block_index]
    raw_target = float(start_block.top_y - int(viewport_height * chapter_top_margin_ratio))
    return min(max(raw_target, 0.0), max_scroll)


def resolve_chapter_end_target(chapter: Chapter, blocks: list[Block], viewport_height: int, max_scroll: float) -> float:
    end_block = blocks[chapter.end_block_index]
    raw_target = float(end_block.top_y + end_block.height - viewport_height)
    return min(max(raw_target, 0.0), max_scroll)


def collapse_anchors(anchors: list[tuple[float, float]]) -> list[tuple[float, float]]:
    collapsed: list[tuple[float, float]] = []
    for time_value, y_value in anchors:
        if collapsed and abs(collapsed[-1][0] - time_value) < 0.01:
            collapsed[-1] = (time_value, y_value)
        elif collapsed and abs(collapsed[-1][1] - y_value) < 0.5 and abs(collapsed[-1][0] - time_value) < 0.15:
            collapsed[-1] = (time_value, y_value)
        else:
            collapsed.append((time_value, y_value))
    return collapsed


def build_exact_scroll_anchors(
    blocks: list[Block],
    chunk_spans: list[tuple[int, int]],
    chunk_timings: list[ChunkTiming],
    chapters: list[Chapter],
    chapter_pauses: list[ChapterPause],
    document_height: int,
    viewport_height: int,
    focus_line_ratio: float,
    chapter_top_margin_ratio: float,
) -> list[tuple[float, float]]:
    block_spans = build_block_spans(blocks)
    max_scroll = max(0.0, float(document_height - viewport_height))
    first_chunks, last_chunks = build_chapter_chunk_boundaries(chunk_timings)
    pause_by_chapter = {pause.chapter_index: pause for pause in chapter_pauses}

    anchors: list[tuple[float, float]] = [(0.0, 0.0)]
    for chunk_timing, chunk_span in zip(chunk_timings, chunk_spans):
        chapter = chapters[chunk_timing.chapter_index]
        chapter_start_target = resolve_chapter_start_target(
            chapter,
            blocks,
            viewport_height,
            max_scroll,
            chapter_top_margin_ratio,
        )
        chapter_end_target = resolve_chapter_end_target(chapter, blocks, viewport_height, max_scroll)

        chapter_pause = pause_by_chapter.get(chunk_timing.chapter_index)
        if chapter_pause is not None and first_chunks.get(chunk_timing.chapter_index) == chunk_timing.chunk_index:
            previous_chapter = chapters[chapter_pause.previous_chapter_index]
            previous_end_target = resolve_chapter_end_target(previous_chapter, blocks, viewport_height, max_scroll)
            anchors.extend(
                [
                    (chapter_pause.gap_start, previous_end_target),
                    (chapter_pause.end_hold_end, previous_end_target),
                    (chapter_pause.transition_end, chapter_start_target),
                    (chapter_pause.chapter_audio_start, chapter_start_target),
                ]
            )

        start_target = resolve_scroll_target(
            chunk_span[0],
            block_spans,
            viewport_height,
            max_scroll,
            focus_line_ratio,
        )
        end_target = resolve_scroll_target(
            chunk_span[1],
            block_spans,
            viewport_height,
            max_scroll,
            focus_line_ratio,
        )

        if first_chunks.get(chunk_timing.chapter_index) == chunk_timing.chunk_index:
            start_target = chapter_start_target
            end_target = max(end_target, chapter_start_target)

        if last_chunks.get(chunk_timing.chapter_index) == chunk_timing.chunk_index:
            end_target = chapter_end_target

        anchors.append((chunk_timing.start_time, start_target))
        anchors.append((chunk_timing.end_time, end_target))

    if chunk_timings:
        last_chapter = chapters[chunk_timings[-1].chapter_index]
        anchors.append((chunk_timings[-1].end_time, resolve_chapter_end_target(last_chapter, blocks, viewport_height, max_scroll)))

    return collapse_anchors(anchors)


def make_silence(duration: float, sample_rate: int, template_audio: np.ndarray) -> np.ndarray:
    sample_count = max(1, int(round(duration * sample_rate)))
    if template_audio.ndim == 1:
        return np.zeros(sample_count, dtype=template_audio.dtype)
    return np.zeros((sample_count, template_audio.shape[1]), dtype=template_audio.dtype)


def build_chapter_pause_audio(
    segment_groups: dict[int, list[Path]],
    chunk_timings: list[ChunkTiming],
    chapter_pauses: list[ChapterPause],
    output_path: Path,
) -> Path:
    """Rebuild audio so it exactly follows the computed chunk timeline.

    The timeline can contain both chapter holds and paragraph pauses. The old
    implementation inserted only chapter holds, silently deleting paragraph
    pacing whenever a chapter-paused video audio track was reconstructed.
    """
    del chapter_pauses  # pauses are already encoded in chunk_timings.start_time
    audio_segments: list[np.ndarray] = []
    sample_rate: int | None = None
    template_audio: np.ndarray | None = None
    assembled_seconds = 0.0

    for chunk_timing in chunk_timings:
        chunk_paths = segment_groups.get(chunk_timing.chunk_index)
        if not chunk_paths:
            raise RuntimeError(f"Missing audio segments for chunk {chunk_timing.chunk_index}")

        # Read the first chunk before inserting any gap so we know its format.
        loaded_segments: list[np.ndarray] = []
        for segment_path in chunk_paths:
            audio_data, segment_sample_rate = sf.read(segment_path)
            if sample_rate is None:
                sample_rate = segment_sample_rate
            elif segment_sample_rate != sample_rate:
                raise RuntimeError(
                    f"Segment {segment_path} uses sample rate {segment_sample_rate}, expected {sample_rate}"
                )
            template_audio = audio_data
            loaded_segments.append(audio_data)

        assert sample_rate is not None and template_audio is not None
        gap_duration = chunk_timing.start_time - assembled_seconds
        if gap_duration > 0.0005:
            audio_segments.append(make_silence(gap_duration, sample_rate, template_audio))
            assembled_seconds += gap_duration
        elif gap_duration < -0.02:
            raise RuntimeError(
                f"Chunk timeline overlaps assembled audio by {-gap_duration:.3f}s at chunk {chunk_timing.chunk_index}"
            )

        for audio_data in loaded_segments:
            audio_segments.append(audio_data)
            assembled_seconds += len(audio_data) / sample_rate

    if not audio_segments or sample_rate is None:
        raise RuntimeError("No audio segments were available to assemble a timeline-aligned track")

    combined_audio = np.concatenate(audio_segments, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, combined_audio, sample_rate)
    return output_path


def assign_block_timing(blocks: list[Block], total_duration: float) -> None:
    weights = [
        max(IMAGE_WORD_COUNT, block.word_count)
        if block.kind == "image"
        else max(6 if block.kind == "heading" else 20, block.word_count)
        for block in blocks
    ]
    total_weight = sum(weights)
    min_duration = min(2.2, total_duration / max(1, len(blocks)))
    base_total = min_duration * len(blocks)
    extra_pool = max(0.0, total_duration - base_total)

    cursor = 0.0
    for block, weight in zip(blocks, weights):
        extra = extra_pool * (weight / total_weight) if total_weight else 0.0
        duration = min_duration + extra
        block.start = cursor
        block.end = min(total_duration, cursor + duration)
        cursor = block.end

    if blocks:
        blocks[-1].end = total_duration


def assign_line_timing(layout_blocks: list[LayoutBlock]) -> list[LineLayout]:
    line_layouts: list[LineLayout] = []

    for layout_block in layout_blocks:
        block = layout_block.block
        if not layout_block.lines:
            continue

        weights = [max(1, line.word_count) for line in layout_block.lines]
        total_weight = sum(weights)
        block_duration = max(0.01, block.end - block.start)
        cursor = block.start

        for line_index, (line, weight) in enumerate(zip(layout_block.lines, weights)):
            if line_index == len(layout_block.lines) - 1:
                line.start = cursor
                line.end = block.end
            else:
                duration = block_duration * (weight / total_weight) if total_weight else block_duration / len(layout_block.lines)
                line.start = cursor
                line.end = min(block.end, cursor + duration)
                cursor = line.end

            line_layouts.append(line)

    return line_layouts


def assign_line_source_spans(
    line_layouts: list[LineLayout],
    block_spans: list[tuple[int, int, Block]],
) -> list[LineLayout]:
    if not line_layouts:
        return []

    block_span_map = {block.index: (start_offset, end_offset, block) for start_offset, end_offset, block in block_spans}
    lines_by_block: dict[int, list[LineLayout]] = {}
    for line in line_layouts:
        lines_by_block.setdefault(line.block_index, []).append(line)

    for block_index, block_lines in lines_by_block.items():
        span_info = block_span_map.get(block_index)
        if span_info is None:
            raise RuntimeError(f"Missing source span for block {block_index}")
        block_start_offset, _, block = span_info
        ordered_lines = sorted(block_lines, key=lambda item: item.line_index)
        searchable_chars: list[str] = []
        searchable_offsets: list[int] = []
        previous_was_space = True

        for local_offset, char in enumerate(block.source_text):
            if char == "#":
                continue
            if char.isspace():
                if previous_was_space:
                    continue
                searchable_chars.append(" ")
                searchable_offsets.append(local_offset)
                previous_was_space = True
                continue
            searchable_chars.append(char)
            searchable_offsets.append(local_offset)
            previous_was_space = False

        while searchable_chars and searchable_chars[0] == " ":
            searchable_chars.pop(0)
            searchable_offsets.pop(0)
        while searchable_chars and searchable_chars[-1] == " ":
            searchable_chars.pop()
            searchable_offsets.pop()

        if block.kind == "image" or not searchable_chars:
            for line in ordered_lines:
                line.source_start = block_start_offset
                line.source_end = block_start_offset
            continue

        searchable_text = "".join(searchable_chars)
        if not searchable_text:
            for line in ordered_lines:
                line.source_start = block_start_offset
                line.source_end = block_start_offset
            continue

        search_start = 0
        local_line_starts: list[int] = []
        local_line_ends: list[int] = []

        for line in ordered_lines:
            local_start = searchable_text.find(line.text, search_start)
            if local_start < 0:
                local_start = searchable_text.find(line.text)
            if local_start < 0:
                raise RuntimeError(f"Could not map rendered line back to source text: {line.text!r}")
            local_line_starts.append(local_start)
            local_line_ends.append(local_start + len(line.text))
            search_start = local_start + len(line.text)

        for line_index, line in enumerate(ordered_lines):
            next_local_start = (
                searchable_offsets[local_line_starts[line_index + 1]]
                if line_index + 1 < len(local_line_starts)
                else searchable_offsets[-1] + 1
            )
            source_start = searchable_offsets[local_line_starts[line_index]]
            source_end = searchable_offsets[local_line_ends[line_index] - 1] + 1
            line.source_start = block_start_offset + source_start
            line.source_end = block_start_offset + max(source_end, next_local_start)

    return sorted(line_layouts, key=lambda item: (item.source_start, item.line_index, item.index))


def expand_image_line_timing(line_layouts: list[LineLayout], blocks: list[Block]) -> list[LineLayout]:
    """Give illustration lines a brief pause so scroll lingers on the plate."""
    if not line_layouts:
        return line_layouts

    block_by_index = {block.index: block for block in blocks}
    pause = max(0.75, float(IMAGE_PAUSE_SECONDS))

    for index, line in enumerate(line_layouts):
        block = block_by_index.get(line.block_index)
        if block is None or block.kind != "image":
            continue

        previous_line = next(
            (
                candidate
                for candidate in reversed(line_layouts[:index])
                if block_by_index.get(candidate.block_index) is not None
                and block_by_index[candidate.block_index].kind != "image"
            ),
            None,
        )
        next_line = next(
            (
                candidate
                for candidate in line_layouts[index + 1 :]
                if block_by_index.get(candidate.block_index) is not None
                and block_by_index[candidate.block_index].kind != "image"
            ),
            None,
        )
        start = previous_line.end if previous_line is not None else line.start
        if next_line is not None and next_line.start > start:
            end = min(next_line.start, start + pause)
        else:
            end = start + pause
        line.start = start
        line.end = max(start + 0.01, end)
        block.start = line.start
        block.end = line.end

    return line_layouts


def map_source_offset_to_time(
    offset: int,
    chunk_spans: list[tuple[int, int]],
    chunk_timings: list[ChunkTiming],
    *,
    bias: str,
) -> float:
    if not chunk_spans or not chunk_timings:
        return 0.0

    if bias not in {"start", "end"}:
        raise ValueError(f"Unsupported offset bias: {bias}")

    for chunk_index, ((start_offset, end_offset), chunk_timing) in enumerate(zip(chunk_spans, chunk_timings)):
        span_length = max(1, end_offset - start_offset)

        if bias == "start":
            if offset < start_offset:
                return chunk_timing.start_time
            if start_offset <= offset < end_offset or (chunk_index == len(chunk_spans) - 1 and offset == end_offset):
                ratio = max(0.0, min(1.0, (offset - start_offset) / span_length))
                return chunk_timing.start_time + ((chunk_timing.end_time - chunk_timing.start_time) * ratio)
            continue

        if offset <= start_offset:
            if chunk_index == 0:
                return chunk_timing.start_time
            return chunk_timings[chunk_index - 1].end_time
        if offset <= end_offset:
            ratio = max(0.0, min(1.0, (offset - start_offset) / span_length))
            return chunk_timing.start_time + ((chunk_timing.end_time - chunk_timing.start_time) * ratio)

    return chunk_timings[-1].end_time


def assign_line_timing_from_chunks(
    line_layouts: list[LineLayout],
    block_spans: list[tuple[int, int, Block]],
    chunk_spans: list[tuple[int, int]],
    chunk_timings: list[ChunkTiming],
) -> list[LineLayout]:
    if not line_layouts:
        return []

    if len(chunk_spans) != len(chunk_timings):
        raise RuntimeError("Chunk spans and chunk timings must have matching lengths")

    ordered_lines = assign_line_source_spans(line_layouts, block_spans)
    for line in ordered_lines:
        line.start = map_source_offset_to_time(line.source_start, chunk_spans, chunk_timings, bias="start")
        line.end = map_source_offset_to_time(line.source_end, chunk_spans, chunk_timings, bias="end")
        if line.end < line.start:
            line.end = line.start

    return ordered_lines


def assign_line_timing_from_chunk_asr(
    line_layouts: list[LineLayout],
    block_spans: list[tuple[int, int, Block]],
    chunk_spans: list[tuple[int, int]],
    chunk_timings: list[ChunkTiming],
    chunk_audio_paths: dict[int, Path],
    *,
    model_name: str,
    device: str,
    cache_dir: Path | None = None,
    workers: str | None = "auto",
) -> tuple[list[LineLayout], float]:
    if not line_layouts:
        return [], 0.0

    ordered_lines = assign_line_source_spans(line_layouts, block_spans)
    lines_by_chunk: dict[int, list[LineLayout]] = {chunk_timing.chunk_index: [] for chunk_timing in chunk_timings}
    chunk_end_offsets = [end_offset for _, end_offset in chunk_spans]

    for line in ordered_lines:
        chunk_index = bisect_right(chunk_end_offsets, line.source_start)
        if chunk_index >= len(chunk_timings):
            chunk_index = len(chunk_timings) - 1
        lines_by_chunk[chunk_timings[chunk_index].chunk_index].append(line)

    resolved_lines: list[LineLayout] = []
    weighted_match_total = 0.0
    expected_word_total = 0
    chunk_tasks: list[dict[str, object]] = []

    for chunk_timing in chunk_timings:
        chunk_lines = lines_by_chunk.get(chunk_timing.chunk_index, [])
        if not chunk_lines:
            continue
        chunk_audio_path = chunk_audio_paths.get(chunk_timing.chunk_index)
        if chunk_audio_path is None:
            raise RuntimeError(f"Missing audio file for chunk {chunk_timing.chunk_index}")
        chunk_duration = max(0.01, chunk_timing.end_time - chunk_timing.start_time)
        chunk_tasks.append(
            {
                "chunk_index": chunk_timing.chunk_index,
                "chunk_audio_path": str(chunk_audio_path),
                "chunk_duration": chunk_duration,
                "time_offset": chunk_timing.start_time,
                "expected_word_total": sum(max(1, line.word_count) for line in chunk_lines),
                "chunk_lines": [asdict(line) for line in chunk_lines],
            }
        )

    cuda_devices = resolve_visible_cuda_devices() if device == "cuda" else []
    worker_count = resolve_asr_worker_count(workers, device, len(chunk_tasks))

    if worker_count > 1:
        shard_count = worker_count
        shards = [[] for _ in range(shard_count)]
        for task_index, chunk_task in enumerate(chunk_tasks):
            shards[task_index % shard_count].append(chunk_task)
        shard_devices = cuda_devices[:shard_count] if device == "cuda" else [None] * shard_count

        with ProcessPoolExecutor(
            max_workers=shard_count,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(
                    _align_chunk_asr_shard,
                    shard,
                    model_name=model_name,
                    device=device,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    cuda_visible_device=shard_devices[index],
                )
                for index, shard in enumerate(shards)
                if shard
            ]
            shard_results: list[dict[str, object]] = []
            for future in as_completed(futures):
                shard_results.extend(future.result())
    else:
        shard_results = _align_chunk_asr_shard(
            chunk_tasks,
            model_name=model_name,
            device=device,
            cache_dir=str(cache_dir) if cache_dir else None,
            cuda_visible_device=cuda_devices[0] if cuda_devices else None,
        )

    for result in sorted(shard_results, key=lambda item: int(item["chunk_index"])):
        chunk_word_total = int(result["expected_word_total"])
        weighted_match_total += float(result["match_ratio"]) * chunk_word_total
        expected_word_total += chunk_word_total
        resolved_lines.extend(LineLayout(**line_layout) for line_layout in result["aligned_lines"])

    resolved_lines.sort(key=lambda item: (item.start, item.index))
    overall_match_ratio = weighted_match_total / max(1, expected_word_total)
    return resolved_lines, overall_match_ratio


def interpolate_anchors(anchors: list[tuple[float, float]], time_value: float) -> float:
    if not anchors:
        return 0.0

    if time_value <= anchors[0][0]:
        return anchors[0][1]

    for index in range(len(anchors) - 1):
        start_time, start_y = anchors[index]
        end_time, end_y = anchors[index + 1]
        if time_value <= end_time:
            if end_time <= start_time:
                return end_y
            ratio = max(0.0, min(1.0, (time_value - start_time) / (end_time - start_time)))
            return start_y + ((end_y - start_y) * ratio)

    return anchors[-1][1]


def clamp_color(value: int) -> int:
    return max(0, min(255, value))


def build_paper_background_image(width: int, height: int, seed: int) -> Image.Image:
    base_rgb = np.array([216, 171, 118], dtype=np.int16)
    randomizer = np.random.default_rng(seed)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    x_positions = np.arange(width, dtype=np.float32)
    column_wave = np.rint(2.0 * np.sin(x_positions / 220.0)).astype(np.int16)[None, :]
    denominator = max(1, height - 1)
    row_chunk_size = 1024

    for row_start in range(0, height, row_chunk_size):
        row_end = min(height, row_start + row_chunk_size)
        y_positions = np.arange(row_start, row_end, dtype=np.float32)[:, None]
        vertical_ratio = y_positions / float(denominator)
        gradient = np.rint(10.0 * (0.5 - vertical_ratio)).astype(np.int16)
        band = np.rint(3.0 * np.sin(y_positions / 180.0)).astype(np.int16)
        noise = randomizer.integers(-1, 2, size=(row_end - row_start, width), dtype=np.int16)
        shade = gradient + band + column_wave + noise
        rgb[row_start:row_end] = np.clip(base_rgb + shade[..., None], 0, 255).astype(np.uint8)

    return Image.fromarray(rgb, mode="RGB").convert("RGBA")


DEFAULT_FONT_PATH_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
    Path("/Library/Fonts/Georgia.ttf"),
    Path.home() / "Library" / "Fonts" / "Georgia.ttf",
    Path("/usr/share/fonts/truetype/msttcorefonts/Georgia.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/georgia.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSerif.ttf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf"),
    Path("/usr/share/fonts/urw-base35/P052-Roman.otf"),
    Path("/usr/share/fonts/urw-base35/C059-Roman.otf"),
)

DEFAULT_FONT_NAME_CANDIDATES = (
    "Georgia",
    "Times New Roman",
    "DejaVu Serif",
    "Nimbus Roman",
    "NimbusRoman-Regular.otf",
    "DejaVuSerif.ttf",
)


def _matplotlib_dejavu_serif_path() -> Path | None:
    # matplotlib bundles DejaVuSerif.ttf on every platform it supports, so it
    # works as a last-resort font source on minimal systems (e.g. Roihu) with
    # no system font packages installed at all.
    try:
        import matplotlib
    except ImportError:
        return None
    candidate = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf"
    return candidate if candidate.exists() else None


def choose_default_font_argument() -> str:
    for candidate in DEFAULT_FONT_PATH_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    matplotlib_font = _matplotlib_dejavu_serif_path()
    if matplotlib_font is not None:
        return str(matplotlib_font)

    for candidate_name in DEFAULT_FONT_NAME_CANDIDATES:
        try:
            ImageFont.truetype(candidate_name, 72)
            return candidate_name
        except OSError:
            continue

    tried = ", ".join([str(path) for path in DEFAULT_FONT_PATH_CANDIDATES] + list(DEFAULT_FONT_NAME_CANDIDATES))
    raise RuntimeError(f"Unable to find a default scalable serif font. Tried: {tried}")


def resolve_font(font_name: str, font_size: int) -> ImageFont.ImageFont:
    requested_font_name = font_name.strip()
    allow_default_family_fallback = requested_font_name.lower() in {"", "auto", "default"}
    if allow_default_family_fallback:
        font_name = choose_default_font_argument()
    else:
        font_name = requested_font_name
    font_path = Path(font_name)
    candidates: list[Path] = []
    attempted: list[str] = []
    if font_path.exists():
        candidates.append(font_path)
    else:
        font_dirs = [
            Path("/System/Library/Fonts/Supplemental"),
            Path("/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ]
        for font_dir in font_dirs:
            for extension in (".ttf", ".otf", ".ttc"):
                candidates.append(font_dir / f"{font_name}{extension}")

    for candidate in candidates:
        attempted.append(str(candidate))
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), font_size)
            except OSError:
                continue

    fallback_names = (font_name,) if not allow_default_family_fallback else (font_name, *DEFAULT_FONT_NAME_CANDIDATES)
    for candidate_name in fallback_names:
        attempted.append(candidate_name)
        try:
            return ImageFont.truetype(candidate_name, font_size)
        except OSError:
            continue

    tried = ", ".join(dict.fromkeys(attempted))
    raise RuntimeError(
        f"Unable to resolve a scalable font for {font_name!r} at size {font_size}. "
        f"Tried: {tried or font_name}"
    )


def wrap_text(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str, max_width: int) -> str:
    words = text.split()
    if not words:
        return text

    lines: list[str] = []
    current_words: list[str] = []
    if hasattr(font, "getlength"):
        measure = font.getlength
    else:
        measure = lambda value: draw.textlength(value, font=font)
    space_width = float(measure(" "))
    current_width = 0.0

    for word in words:
        word_width = float(measure(word))
        candidate_width = word_width if not current_words else current_width + space_width + word_width
        if candidate_width <= max_width or not current_words:
            current_words.append(word)
            current_width = candidate_width
        else:
            lines.append(" ".join(current_words))
            current_words = [word]
            current_width = word_width

    if current_words:
        lines.append(" ".join(current_words))

    return "\n".join(lines)


def resolve_horizontal_margin(width: int) -> int:
    return max(52, int(round(width * 0.065)))


def resolve_top_margin(viewport_height: int, paragraph_size: int) -> int:
    return max(int(round(viewport_height * 0.08)), int(round(paragraph_size * 1.8)))


def measure_text_width(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str) -> int:
    if hasattr(font, "getlength"):
        return int(round(float(font.getlength(text))))
    return int(round(float(draw.textlength(text, font=font))))


def measure_line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    ascent, descent = font.getmetrics()
    estimated_height = ascent + descent
    if estimated_height > 0:
        return estimated_height
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return max(1, bbox[3] - bbox[1])


def validate_font_metrics(font_name: str, label: str, requested_size: int, line_height: int) -> None:
    minimum_expected_height = max(12, int(round(requested_size * 0.72)))
    if line_height < minimum_expected_height:
        raise RuntimeError(
            f"Resolved {label} font metrics are too small for {font_name!r}: "
            f"requested {requested_size}px, measured line height {line_height}px. "
            "Refusing to render because this would produce unreadably tiny text."
        )


def layout_document(
    source_text: str,
    blocks: list[Block],
    width: int,
    viewport_height: int,
    font_name: str,
    paragraph_size: int,
    heading_size: int,
) -> tuple[int, list[LayoutBlock], list[LineLayout]]:
    dummy_image = Image.new("RGBA", (width, max(viewport_height, 2000)), (255, 255, 255, 0))
    draw = ImageDraw.Draw(dummy_image)
    heading_font = resolve_font(font_name, heading_size)
    paragraph_font = resolve_font(font_name, paragraph_size)

    layouts: list[LayoutBlock] = []
    line_layouts: list[LineLayout] = []
    horizontal_margin = resolve_horizontal_margin(width)
    text_width = max(240, width - (horizontal_margin * 2))
    cursor_y = resolve_top_margin(viewport_height, paragraph_size)
    line_index = 0
    heading_line_height = measure_line_height(draw, heading_font)
    paragraph_line_height = measure_line_height(draw, paragraph_font)
    validate_font_metrics(font_name, "heading", heading_size, heading_line_height)
    validate_font_metrics(font_name, "paragraph", paragraph_size, paragraph_line_height)

    caption_font = resolve_font(font_name, max(28, int(round(paragraph_size * 0.62))))
    caption_line_height = measure_line_height(draw, caption_font)
    max_image_height = max(160, int(round(viewport_height * 0.85)))

    for block in blocks:
        if block.kind == "image":
            cursor_y += max(28, int(round(paragraph_size * 0.55)))
            display_width, display_height = measure_plate_display_size(
                Path(block.text),
                content_width=text_width,
                max_height=max_image_height,
            )
            caption_lines: list[str] = []
            caption_height = 0
            if block.caption.strip():
                wrapped_caption = wrap_text(draw, caption_font, block.caption.strip(), text_width)
                caption_lines = wrapped_caption.splitlines() if wrapped_caption else [block.caption.strip()]
                caption_height = len(caption_lines) * caption_line_height + max(8, int(round(paragraph_size * 0.12)))

            block_lines = [
                LineLayout(
                    index=line_index,
                    block_index=block.index,
                    line_index=0,
                    text="",
                    top_y=cursor_y,
                    height=display_height + caption_height,
                    width=display_width,
                    word_count=max(IMAGE_WORD_COUNT, block.word_count),
                )
            ]
            line_layouts.append(block_lines[0])
            line_index += 1
            block.top_y = cursor_y
            block.height = display_height + caption_height
            layouts.append(LayoutBlock(block=block, font=caption_font, spacing=0, lines=block_lines))
            cursor_y = cursor_y + block.height + max(40, int(round(paragraph_size * 0.9)))
            continue

        if block.kind == "heading":
            cursor_y += max(24, int(round(heading_size * 0.45)))
            font = heading_font
            spacing = max(16, int(round(heading_size * 0.28)))
            line_height = heading_line_height
        else:
            font = paragraph_font
            spacing = max(12, int(round(paragraph_size * 0.22)))
            line_height = paragraph_line_height

        wrapped = wrap_text(draw, font, block.text, text_width)
        wrapped_lines = wrapped.splitlines() if wrapped else [block.text]
        block_lines = []
        line_cursor_y = cursor_y

        for wrapped_line_index, line_text in enumerate(wrapped_lines):
            line_width = measure_text_width(draw, font, line_text)

            line_layout = LineLayout(
                index=line_index,
                block_index=block.index,
                line_index=wrapped_line_index,
                text=line_text,
                top_y=line_cursor_y,
                height=line_height,
                width=line_width,
                word_count=max(1, len(re.findall(r"\S+", line_text))),
            )
            block_lines.append(line_layout)
            line_layouts.append(line_layout)
            line_index += 1

            line_cursor_y += line_height
            if wrapped_line_index < len(wrapped_lines) - 1:
                line_cursor_y += spacing

        block.top_y = cursor_y
        block.height = max(0, line_cursor_y - cursor_y)

        layouts.append(LayoutBlock(block=block, font=font, spacing=spacing, lines=block_lines))

        cursor_y = line_cursor_y + (
            max(56, int(round(heading_size * 0.95))) if block.kind == "heading" else max(34, int(round(paragraph_size * 0.82)))
        )

    total_height = max(viewport_height + 1, cursor_y + max(120, int(round(viewport_height * 0.14))))
    return total_height, layouts, line_layouts


def measure_plate_display_size(image_path: Path, *, content_width: int, max_height: int) -> tuple[int, int]:
    with Image.open(image_path) as plate:
        width, height = plate.size
    if width <= 0 or height <= 0:
        return max(1, content_width), max(1, min(max_height, content_width))
    scale = min(content_width / float(width), max_height / float(height), 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def build_document_image(layout_blocks: list[LayoutBlock], width: int, total_height: int, seed: int) -> Image.Image:
    image = build_paper_background_image(width, total_height, seed)
    draw = ImageDraw.Draw(image)
    left_margin = resolve_horizontal_margin(width)
    content_width = max(240, width - (left_margin * 2))

    for item in layout_blocks:
        block = item.block
        font = item.font

        if block.kind == "image":
            plate_path = Path(block.text)
            with Image.open(plate_path) as plate:
                original_width, original_height = plate.size
                display_width = max(1, item.lines[0].width if item.lines else content_width)
                scale = display_width / float(max(1, original_width))
                display_height = max(1, int(round(original_height * scale)))
                plate = plate.convert("RGBA").resize((display_width, display_height), Image.Resampling.LANCZOS)
                paste_x = left_margin + max(0, (content_width - display_width) // 2)
                image.alpha_composite(plate, (paste_x, block.top_y))
            if block.caption.strip():
                wrapped_caption = wrap_text(draw, font, block.caption.strip(), content_width)
                caption_lines = wrapped_caption.splitlines() if wrapped_caption else [block.caption.strip()]
                caption_y = block.top_y + display_height + 8
                caption_fill = (70, 48, 30, 255)
                for caption_line in caption_lines:
                    caption_width = measure_text_width(draw, font, caption_line)
                    caption_x = left_margin + max(0, (content_width - caption_width) // 2)
                    draw.text((caption_x, caption_y), caption_line, font=font, fill=caption_fill)
                    caption_y += measure_line_height(draw, font)
            continue

        if block.kind == "heading":
            fill = (59, 39, 24, 255)
        else:
            fill = (43, 31, 22, 255)

        for line in item.lines:
            draw.text((left_margin, line.top_y), line.text, font=font, fill=fill)

    return image


def render_document_image(path: Path, layout_blocks: list[LayoutBlock], width: int, total_height: int, seed: int) -> int:
    image = build_document_image(layout_blocks, width, total_height, seed)
    image.convert("RGB").save(path, format="PNG")
    image.close()
    return total_height


def build_scroll_anchors(
    blocks: list[Block],
    document_height: int,
    viewport_height: int,
    duration: float,
    focus_line_ratio: float,
) -> list[tuple[float, float]]:
    max_scroll = max(0.0, float(document_height - viewport_height))
    anchors: list[tuple[float, float]] = [(0.0, 0.0)]
    for block in blocks:
        target = min(max(float(block.top_y - int(viewport_height * focus_line_ratio)), 0.0), max_scroll)
        anchors.append((block.start, target))
    anchors.append((duration, max_scroll))

    collapsed: list[tuple[float, float]] = []
    for time_value, y_value in anchors:
        if collapsed and abs(collapsed[-1][0] - time_value) < 0.01:
            collapsed[-1] = (time_value, y_value)
        else:
            collapsed.append((time_value, y_value))
    return collapsed


def build_line_scroll_anchors(
    line_layouts: list[LineLayout], viewport_height: int, focus_line_ratio: float, document_height: int
) -> list[tuple[float, float]]:
    max_scroll = max(0.0, float(document_height - viewport_height))
    anchors: list[tuple[float, float]] = [(0.0, 0.0)]

    for line in line_layouts:
        target = min(max(float(line.top_y - int(viewport_height * focus_line_ratio)), 0.0), max_scroll)
        anchors.append((line.start, target))
        anchors.append((line.end, target))

    anchors.append((line_layouts[-1].end if line_layouts else 0.0, max_scroll))

    collapsed: list[tuple[float, float]] = []
    for time_value, y_value in anchors:
        if collapsed and abs(collapsed[-1][0] - time_value) < 0.01:
            collapsed[-1] = (time_value, y_value)
        elif collapsed and abs(collapsed[-1][1] - y_value) < 0.5 and abs(collapsed[-1][0] - time_value) < 0.15:
            collapsed[-1] = (time_value, y_value)
        else:
            collapsed.append((time_value, y_value))
    return collapsed


def build_scroll_expression(anchors: list[tuple[float, float]], max_scroll: float) -> str:
    expression = f"{anchors[-1][1]:.2f}"
    for index in range(len(anchors) - 2, -1, -1):
        start_t, start_y = anchors[index]
        end_t, end_y = anchors[index + 1]
        delta_t = max(0.01, end_t - start_t)
        segment_expr = (
            f"({start_y:.2f}+({end_y - start_y:.2f})*max(min((t-{start_t:.2f})/{delta_t:.2f},1),0))"
        )
        expression = f"if(lt(t,{end_t:.2f}),{segment_expr},{expression})"
    return f"min(max({expression},0),{max_scroll:.2f})"


def resolve_video_codec(ffmpeg_path: str, requested_codec: str) -> str:
    if requested_codec != "auto":
        return requested_codec
    return "libx264"


def compute_hardware_bitrate(width: int, height: int, fps: int, *, hevc: bool) -> tuple[str, str, str]:
    reference_pixels = 1920 * 1080 * 24
    scale = (width * height * max(1, fps)) / float(reference_pixels)
    base_bitrate = 8_000_000 if hevc else 12_000_000
    bitrate = max(4_000_000, int(round(base_bitrate * max(scale, 0.35))))
    maxrate = int(round(bitrate * 1.35))
    bufsize = bitrate * 2
    return str(bitrate), str(maxrate), str(bufsize)


def resolve_video_preset(requested_preset: str, resolved_codec: str) -> str:
    if requested_preset != "auto":
        return requested_preset
    if resolved_codec in {"libx264", "libx265"} and sys.platform == "darwin":
        return "veryfast"
    return "medium"


def build_video_codec_args(
    ffmpeg_path: str,
    video_codec: str,
    video_crf: int,
    width: int,
    height: int,
    fps: int,
    video_preset: str,
) -> list[str]:
    resolved_codec = resolve_video_codec(ffmpeg_path, video_codec)
    resolved_preset = resolve_video_preset(video_preset, resolved_codec)
    if resolved_codec == "hevc_videotoolbox":
        bitrate, maxrate, bufsize = compute_hardware_bitrate(width, height, fps, hevc=True)
        return [
            "-c:v",
            "hevc_videotoolbox",
            "-tag:v",
            "hvc1",
            "-allow_sw",
            "0",
            "-profile:v",
            "main",
            "-prio_speed",
            "1",
            "-b:v",
            bitrate,
            "-maxrate",
            maxrate,
            "-bufsize",
            bufsize,
            "-pix_fmt",
            "yuv420p",
        ]
    if resolved_codec == "h264_videotoolbox":
        bitrate, maxrate, bufsize = compute_hardware_bitrate(width, height, fps, hevc=False)
        return [
            "-c:v",
            "h264_videotoolbox",
            "-allow_sw",
            "0",
            "-profile:v",
            "high",
            "-prio_speed",
            "1",
            "-b:v",
            bitrate,
            "-maxrate",
            maxrate,
            "-bufsize",
            bufsize,
            "-pix_fmt",
            "yuv420p",
        ]
    if resolved_codec == "libx265":
        return [
            "-c:v",
            "libx265",
            "-tag:v",
            "hvc1",
            "-preset",
            resolved_preset,
            "-crf",
            str(video_crf),
            "-pix_fmt",
            "yuv420p",
        ]

    return [
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-preset",
        resolved_preset,
        "-crf",
        str(video_crf),
        "-pix_fmt",
        "yuv420p",
    ]


def run_ffmpeg_with_filter_script(command_prefix: list[str], filter_expression: str, command_suffix: list[str]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".ffscript", encoding="utf-8", delete=False) as handle:
        handle.write(filter_expression)
        handle.write("\n")
        filter_script_path = Path(handle.name)
    try:
        subprocess.run(
            [*command_prefix, "-filter_script:v", str(filter_script_path), *command_suffix],
            check=True,
        )
    finally:
        filter_script_path.unlink(missing_ok=True)


def is_streaming_output(output_path: Path) -> bool:
    return str(output_path) == "-"


def build_output_target_args(output_path: Path) -> list[str]:
    if is_streaming_output(output_path):
        return [
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ]
    return ["-movflags", "+faststart", str(output_path)]


def render_video(
    document_path: Path,
    audio_path: Path,
    output_path: Path,
    width: int,
    height: int,
    duration: float,
    fps: int,
    scroll_expression: str,
    video_crf: int,
    video_codec: str,
    video_preset: str,
) -> None:
    ffmpeg = require_binary("ffmpeg")
    filter_expression = f"crop={width}:{height}:0:'{scroll_expression}',fps={fps},format=yuv420p"
    command = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(document_path),
        "-i",
        str(audio_path),
    ]
    suffix = [
        *build_video_codec_args(ffmpeg, video_codec, video_crf, width, height, fps, video_preset),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration:.3f}",
        *build_output_target_args(output_path),
    ]
    run_ffmpeg_with_filter_script(command, filter_expression, suffix)


def render_video_with_active_line(
    document_image: Image.Image,
    audio_path: Path,
    output_path: Path,
    width: int,
    height: int,
    duration: float,
    fps: int,
    anchors: list[tuple[float, float]],
    line_layouts: list[LineLayout],
    video_crf: int,
    video_codec: str,
    video_preset: str,
) -> None:
    ffmpeg = require_binary("ffmpeg")
    document_array = np.asarray(document_image, dtype=np.uint8)
    line_starts = [line.start for line in line_layouts]
    line_color = (176, 68, 58, 150)
    line_rgb = np.array(line_color[:3], dtype=np.uint16).reshape(1, 1, 3)
    line_alpha = int(line_color[3])
    inverse_alpha = 255 - line_alpha
    underline_thickness = max(4, int(round(height * 0.008)))
    left_margin = resolve_horizontal_margin(width)
    max_scroll = max(0, document_image.height - height)
    frame_count = max(1, int(math.floor(duration * fps)) + 1)
    process: subprocess.Popen[bytes] | None = None

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(audio_path),
        *build_video_codec_args(ffmpeg, video_codec, video_crf, width, height, fps, video_preset),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration:.3f}",
        *build_output_target_args(output_path),
    ]

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None

        for frame_index in range(frame_count):
            time_value = min(duration, frame_index / float(fps))
            scroll_y = int(round(interpolate_anchors(anchors, time_value)))
            scroll_y = max(0, min(max_scroll, scroll_y))

            frame = document_array[scroll_y : scroll_y + height, :, :].copy()

            if line_layouts:
                line_index = bisect_right(line_starts, time_value) - 1
                if 0 <= line_index < len(line_layouts):
                    active_line = line_layouts[line_index]
                    if active_line.start <= time_value < active_line.end or (
                        frame_index == frame_count - 1 and math.isclose(time_value, active_line.end, abs_tol=1e-6)
                    ):
                        line_duration = max(0.01, active_line.end - active_line.start)
                        progress = max(0.0, min(1.0, (time_value - active_line.start) / line_duration))

                        underline_width = max(2, int(round(active_line.width * progress)))
                        underline_y = int(round(active_line.top_y - scroll_y + active_line.height + 8))
                        if -underline_thickness < underline_y < height:
                            x0 = max(0, left_margin)
                            x1 = min(width, left_margin + underline_width)
                            y0 = max(0, underline_y)
                            y1 = min(height, underline_y + underline_thickness)
                            if x0 < x1 and y0 < y1:
                                region = frame[y0:y1, x0:x1, :3].astype(np.uint16)
                                blended = ((region * inverse_alpha) + (line_rgb * line_alpha)) // 255
                                frame[y0:y1, x0:x1, :3] = blended.astype(np.uint8)
                                frame[y0:y1, x0:x1, 3] = 255

            assert process.stdin is not None
            process.stdin.write(memoryview(frame))

        if process.stdin is not None:
            process.stdin.close()
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
    finally:
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an autoscrolling reading video from audio and text")
    parser.add_argument("--audio", required=True, type=Path, help="Rendered audio file")
    parser.add_argument("--text", type=Path, help="Source text file")
    parser.add_argument(
        "--chunk-text-dir",
        type=Path,
        help="Directory containing chunk_*.txt files to concatenate into the source text",
    )
    parser.add_argument(
        "--chunk-audio-dir",
        type=Path,
        help="Directory containing chunk_*.wav files that match --chunk-text-dir and provide exact chunk timing",
    )
    parser.add_argument(
        "--chunk-duration-manifest",
        type=Path,
        help="JSON manifest with exact chunk_index/duration_seconds pairs for --chunk-text-dir",
    )
    parser.add_argument(
        "--chunk-manifest",
        type=Path,
        help="Chunker manifest containing ends_at_paragraph metadata for exact pause timing",
    )
    parser.add_argument(
        "--paragraph-pause-seconds",
        type=float,
        default=0.0,
        help="Silence inserted after chunk boundaries marked ends_at_paragraph in --chunk-manifest",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path")
    parser.add_argument("--width", type=int, default=1920, help="Viewport width")
    parser.add_argument("--height", type=int, default=1080, help="Viewport height")
    parser.add_argument("--fps", type=int, default=24, help="Output frames per second")
    parser.add_argument("--video-crf", type=int, default=18, help="libx264 CRF value; higher is smaller and softer")
    parser.add_argument("--video-codec", choices=VIDEO_CODEC_CHOICES, default="auto", help="Video codec")
    parser.add_argument("--video-preset", choices=VIDEO_PRESET_CHOICES, default="auto", help="Software encoder preset")
    parser.add_argument(
        "--font",
        default="auto",
        help="Document font family or path. 'auto' selects a verified scalable serif font.",
    )
    parser.add_argument("--paragraph-size", type=int, default=56, help="Paragraph font size")
    parser.add_argument("--heading-size", type=int, default=88, help="Heading font size")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for paper texture")
    parser.add_argument(
        "--kokoro-segment-dir",
        type=Path,
        help="Directory containing Kokoro segment_<chunk>_<part>.wav files to drive scroll timing from real chunk durations",
    )
    parser.add_argument(
        "--focus-line-ratio",
        type=float,
        default=0.62,
        help="Vertical placement of the active reading line as a fraction of the viewport height",
    )
    parser.add_argument(
        "--chapter-top-margin-ratio",
        type=float,
        default=0.16,
        help="Top margin used when settling onto a fresh chapter heading",
    )
    parser.add_argument(
        "--chapter-end-hold",
        type=float,
        default=3.0,
        help="Seconds to hold on the finished chapter page before transitioning to the next chapter",
    )
    parser.add_argument(
        "--chapter-transition",
        type=float,
        default=0.6,
        help="Seconds used for the visual move from the old chapter page to the next chapter heading",
    )
    parser.add_argument(
        "--chapter-start-hold",
        type=float,
        default=2.0,
        help="Seconds to hold on the new chapter heading before audio resumes",
    )
    parser.add_argument(
        "--chapter-audio-output",
        type=Path,
        help="Optional path for the silence-padded chapter audio assembled from existing Kokoro segments",
    )
    parser.add_argument(
        "--show-active-line",
        action="store_true",
        help="Draw a muted red progress underline beneath the current reading line",
    )
    parser.add_argument(
        "--active-line-timing",
        choices=("auto", "chunk", "asr"),
        default="auto",
        help="How to time the active-line underline: chunk mapping, ASR alignment, or auto fallback",
    )
    parser.add_argument(
        "--active-line-asr-model",
        default="small.en",
        help="faster-whisper model used when active-line timing runs in ASR mode",
    )
    parser.add_argument(
        "--active-line-asr-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for faster-whisper active-line alignment",
    )
    parser.add_argument(
        "--active-line-asr-cache-dir",
        type=Path,
        help="Optional directory for cached faster-whisper word timings",
    )
    parser.add_argument(
        "--active-line-asr-workers",
        default="auto",
        help="Worker count for chunked ASR active-line timing, or 'auto'",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate assets")
    parser.add_argument(
        "--chapters-output",
        type=Path,
        default=None,
        help=(
            "Optional path to write the real, timestamped chapter bookmarks "
            "(index/title/start_seconds/end_seconds) computed during this render, "
            "as a JSON list -- consumable by scripts/embed_video_chapters.py. "
            "When omitted, no chapters file is written and rendering is unaffected."
        ),
    )
    parser.add_argument(
        "--plates-dir",
        type=Path,
        help=(
            "Optional directory of book illustration images. When set, embeds plates "
            "into the scrolling document at marker positions (illustrated video style)."
        ),
    )
    parser.add_argument(
        "--plates-manifest",
        type=Path,
        help=(
            "Optional JSON manifest {plates:[{path, caption, after_text_snippet}]}. "
            "When omitted with --plates-dir, auto-built from markers + plates + source.htm."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.audio.exists():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 1
    if args.text is None and args.chunk_text_dir is None:
        print("Either --text or --chunk-text-dir must be provided", file=sys.stderr)
        return 1
    if args.text is not None and not args.text.exists():
        print(f"Text file not found: {args.text}", file=sys.stderr)
        return 1
    if args.chunk_text_dir is not None and not args.chunk_text_dir.exists():
        print(f"Chunk text directory not found: {args.chunk_text_dir}", file=sys.stderr)
        return 1
    if args.chunk_audio_dir is not None and not args.chunk_audio_dir.exists():
        print(f"Chunk audio directory not found: {args.chunk_audio_dir}", file=sys.stderr)
        return 1
    if args.chunk_duration_manifest is not None and not args.chunk_duration_manifest.exists():
        print(f"Chunk duration manifest not found: {args.chunk_duration_manifest}", file=sys.stderr)
        return 1
    if args.chunk_manifest is not None and not args.chunk_manifest.exists():
        print(f"Chunk manifest not found: {args.chunk_manifest}", file=sys.stderr)
        return 1
    if args.chunk_audio_dir is not None and args.chunk_duration_manifest is not None:
        print("Use either --chunk-audio-dir or --chunk-duration-manifest, not both", file=sys.stderr)
        return 1
    if args.kokoro_segment_dir and not args.kokoro_segment_dir.exists():
        print(f"Kokoro segment directory not found: {args.kokoro_segment_dir}", file=sys.stderr)
        return 1
    if args.plates_dir is not None and not args.plates_dir.exists():
        print(f"Plates directory not found: {args.plates_dir}", file=sys.stderr)
        return 1
    if args.plates_manifest is not None and not args.plates_manifest.exists():
        print(f"Plates manifest not found: {args.plates_manifest}", file=sys.stderr)
        return 1
    plates_enabled = args.plates_dir is not None or args.plates_manifest is not None

    chunk_text_entries = load_chunk_text_entries(args.chunk_text_dir) if args.chunk_text_dir is not None else None
    if args.chunk_audio_dir is not None:
        chunk_audio_entries = load_chunk_audio_duration_entries(args.chunk_audio_dir)
    elif args.chunk_duration_manifest is not None:
        chunk_audio_entries = load_chunk_duration_manifest_entries(args.chunk_duration_manifest)
    else:
        chunk_audio_entries = None
    chunk_audio_path_entries = load_chunk_audio_path_entries(args.chunk_audio_dir) if args.chunk_audio_dir is not None else None
    paragraph_pause_after = load_chunk_paragraph_flags(args.chunk_manifest) if args.chunk_manifest is not None else {}
    ordered_chunk_indices: list[int] | None = None
    chunk_texts: list[str] | None = None
    chunk_durations: list[float] | None = None

    if chunk_text_entries is not None and chunk_audio_entries is not None:
        ordered_chunk_indices, chunk_texts, chunk_durations = match_chunk_subset(chunk_text_entries, chunk_audio_entries)
        if args.text is not None:
            source_text = source_text_for_chunk_subset(read_text_file(args.text), chunk_texts)
        else:
            source_text = "\n\n".join(chunk_texts)
    elif chunk_text_entries is not None:
        ordered_chunk_indices = [chunk_index for chunk_index, _ in chunk_text_entries]
        chunk_texts = [chunk_text for _, chunk_text in chunk_text_entries]
        if args.text is not None:
            source_text = source_text_for_chunk_subset(read_text_file(args.text), chunk_texts)
        else:
            source_text = "\n\n".join(chunk_texts)
    else:
        assert args.text is not None
        source_text = read_text_file(args.text)

    try:
        plate_specs: list[PlateSpec] = []
        if plates_enabled:
            plate_specs, source_text = resolve_plates_for_render(
                plates_dir=args.plates_dir,
                plates_manifest=args.plates_manifest,
                text_path=args.text,
                source_text=source_text,
            )
            print(f"Illustrated mode: embedding {len(plate_specs)} plate(s)", file=sys.stderr)

        blocks = parse_blocks(source_text)
        if plate_specs:
            blocks = inject_image_blocks(blocks, plate_specs, source_text)
        block_spans = build_block_spans(blocks)
        chapters = build_chapters(block_spans)
        chunk_timings: list[ChunkTiming] | None = None
        chapter_pauses: list[ChapterPause] | None = None
        exact_timing_payload = None
        render_anchors: list[tuple[float, float]] | None = None
        active_line_timing_used = "disabled"
        active_line_match_ratio: float | None = None

        if args.kokoro_segment_dir:
            segment_groups = load_kokoro_segment_groups(args.kokoro_segment_dir)
            chunk_texts = load_kokoro_chunk_texts(source_text)
            ordered_chunk_indices = sorted(segment_groups)
            if len(chunk_texts) != len(ordered_chunk_indices):
                raise RuntimeError(
                    f"Kokoro chunk count ({len(chunk_texts)}) does not match segment group count ({len(ordered_chunk_indices)})"
                )
            chunk_durations = probe_kokoro_chunk_durations(segment_groups)
            block_spans = build_block_spans(blocks)
            chapters = build_chapters(block_spans)
            chunk_spans = locate_chunk_spans(normalize_text(source_text), chunk_texts)
            chunk_timings, chapter_pauses, duration = build_chunk_timeline(
                ordered_chunk_indices,
                chunk_spans,
                chunk_durations,
                chapters,
                args.chapter_end_hold,
                args.chapter_transition,
                args.chapter_start_hold,
                paragraph_pause_after=paragraph_pause_after,
                paragraph_pause_seconds=args.paragraph_pause_seconds,
            )
            exact_timing_payload = {
                "segment_groups": segment_groups,
                "chunk_spans": chunk_spans,
                "chunk_timings": chunk_timings,
                "chapters": chapters,
                "chapter_pauses": chapter_pauses,
            }
        elif args.chunk_audio_dir is not None or args.chunk_duration_manifest is not None:
            if chunk_texts is None:
                raise RuntimeError("--chunk-duration-manifest requires --chunk-text-dir")
            if chunk_durations is None or ordered_chunk_indices is None:
                raise RuntimeError("Matched chunk timings were not prepared for exact chunk timing input")
            if len(chunk_texts) != len(chunk_durations):
                raise RuntimeError(
                    f"Chunk text count ({len(chunk_texts)}) does not match chunk audio count ({len(chunk_durations)})"
                )
            chunk_spans = locate_chunk_spans(normalize_text(source_text), chunk_texts)
            chapter_end_hold = args.chapter_end_hold if args.chunk_audio_dir is not None else 0.0
            chapter_transition = args.chapter_transition if args.chunk_audio_dir is not None else 0.0
            chapter_start_hold = args.chapter_start_hold if args.chunk_audio_dir is not None else 0.0
            chunk_timings, chapter_pauses, duration = build_chunk_timeline(
                ordered_chunk_indices,
                chunk_spans,
                chunk_durations,
                chapters,
                chapter_end_hold,
                chapter_transition,
                chapter_start_hold,
                paragraph_pause_after=paragraph_pause_after,
                paragraph_pause_seconds=args.paragraph_pause_seconds,
            )
            segment_groups: dict[int, list[Path]] = {}
            if chunk_audio_path_entries is not None:
                chunk_audio_path_map = dict(chunk_audio_path_entries)
                segment_groups = {
                    chunk_index: [chunk_audio_path_map[chunk_index]]
                    for chunk_index in ordered_chunk_indices
                    if chunk_index in chunk_audio_path_map
                }
            exact_timing_payload = {
                "segment_groups": segment_groups,
                "chunk_spans": chunk_spans,
                "chunk_timings": chunk_timings,
                "chapters": chapters,
                "chapter_pauses": chapter_pauses,
            }
        else:
            duration = probe_audio_duration(args.audio)
            assign_block_timing(blocks, duration)

        document_height, layout_blocks, line_layouts = layout_document(
            source_text,
            blocks,
            width=args.width,
            viewport_height=args.height,
            font_name=args.font,
            paragraph_size=args.paragraph_size,
            heading_size=args.heading_size,
        )
        chunk_spans_for_lines = exact_timing_payload["chunk_spans"] if exact_timing_payload else None
        if args.show_active_line:
            active_line_timing_used = args.active_line_timing
            active_line_warning: str | None = None

            if args.active_line_timing in {"auto", "asr"} and chunk_timings is not None and chunk_spans_for_lines is not None:
                if ordered_chunk_indices is None or chunk_audio_path_entries is None:
                    raise RuntimeError("Chunk ASR timing requires matched chunk indices and chunk audio paths")
                chunk_audio_path_map = dict(chunk_audio_path_entries)
                matched_chunk_audio_paths = {
                    chunk_index: chunk_audio_path_map[chunk_index]
                    for chunk_index in ordered_chunk_indices
                    if chunk_index in chunk_audio_path_map
                }
                try:
                    asr_device = resolve_asr_device(args.active_line_asr_device)
                    line_layouts, active_line_match_ratio = assign_line_timing_from_chunk_asr(
                        line_layouts,
                        block_spans,
                        chunk_spans_for_lines,
                        chunk_timings,
                        matched_chunk_audio_paths,
                        model_name=args.active_line_asr_model,
                        device=asr_device,
                        cache_dir=args.active_line_asr_cache_dir,
                        workers=args.active_line_asr_workers,
                    )
                    active_line_timing_used = f"asr:{args.active_line_asr_model}"
                except Exception as exc:
                    if args.active_line_timing == "asr":
                        raise RuntimeError(f"ASR active-line timing failed: {exc}") from exc
                    active_line_warning = f"ASR active-line timing unavailable; falling back to chunk timing ({exc})"
                    line_layouts = assign_line_timing_from_chunks(line_layouts, block_spans, chunk_spans_for_lines, chunk_timings)
                    active_line_timing_used = "chunk"
                render_anchors = build_line_scroll_anchors(line_layouts, args.height, args.focus_line_ratio, document_height)
            elif args.active_line_timing == "asr":
                asr_device = resolve_asr_device(args.active_line_asr_device)
                line_layouts, active_line_match_ratio = assign_line_timing_from_audio_asr(
                    line_layouts,
                    args.audio,
                    duration=duration,
                    model_name=args.active_line_asr_model,
                    device=asr_device,
                    cache_dir=args.active_line_asr_cache_dir,
                )
                active_line_timing_used = f"asr:{args.active_line_asr_model}"
                render_anchors = build_line_scroll_anchors(line_layouts, args.height, args.focus_line_ratio, document_height)
            elif chunk_timings is not None and chunk_spans_for_lines is not None:
                line_layouts = assign_line_timing_from_chunks(line_layouts, block_spans, chunk_spans_for_lines, chunk_timings)
                active_line_timing_used = "chunk"
                render_anchors = build_line_scroll_anchors(line_layouts, args.height, args.focus_line_ratio, document_height)
            else:
                line_layouts = assign_line_timing(layout_blocks)
                active_line_timing_used = "weighted_blocks"
                render_anchors = build_line_scroll_anchors(line_layouts, args.height, args.focus_line_ratio, document_height)

            if plate_specs:
                line_layouts = expand_image_line_timing(line_layouts, blocks)
                render_anchors = build_line_scroll_anchors(line_layouts, args.height, args.focus_line_ratio, document_height)

            if active_line_warning:
                print(active_line_warning, file=sys.stderr)
    except Exception as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_temp:
        temp_dir_path = args.output.parent / f"{args.output.stem}_autoscroll_assets"
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        cleanup_needed = False
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="autoscroll_video_", dir=str(args.output.parent))
        temp_dir_path = Path(temp_dir.name)
        cleanup_needed = True

    document_path = temp_dir_path / "document.png"
    metadata_path = temp_dir_path / "scroll_map.json"
    render_audio_path = args.audio
    document_image: Image.Image | None = None

    try:
        if exact_timing_payload and exact_timing_payload["chapter_pauses"]:
            render_audio_path = args.chapter_audio_output or (temp_dir_path / "chapter_pause_audio.wav")
            build_chapter_pause_audio(
                exact_timing_payload["segment_groups"],
                exact_timing_payload["chunk_timings"],
                exact_timing_payload["chapter_pauses"],
                render_audio_path,
            )
            duration = probe_audio_duration(render_audio_path)

        if args.show_active_line:
            document_image = build_document_image(
                layout_blocks=layout_blocks,
                width=args.width,
                total_height=document_height,
                seed=args.seed,
            )
            document_image.convert("RGB").save(document_path, format="PNG")
        else:
            render_document_image(
                path=document_path,
                layout_blocks=layout_blocks,
                width=args.width,
                total_height=document_height,
                seed=args.seed,
            )
        if args.show_active_line and render_anchors is not None:
            anchors = render_anchors
        elif exact_timing_payload:
            anchors = build_exact_scroll_anchors(
                blocks,
                chunk_spans=exact_timing_payload["chunk_spans"],
                chunk_timings=exact_timing_payload["chunk_timings"],
                chapters=exact_timing_payload["chapters"],
                chapter_pauses=exact_timing_payload["chapter_pauses"],
                document_height=document_height,
                viewport_height=args.height,
                focus_line_ratio=args.focus_line_ratio,
                chapter_top_margin_ratio=args.chapter_top_margin_ratio,
            )
        else:
            anchors = build_scroll_anchors(
                blocks,
                document_height=document_height,
                viewport_height=args.height,
                duration=duration,
                focus_line_ratio=args.focus_line_ratio,
            )
        metadata_path.write_text(
            json.dumps(
                {
                    "duration": duration,
                    "document_height": document_height,
                    "audio_path": str(render_audio_path),
                    "source_mode": "chunk_text_dir" if args.chunk_text_dir else "text_file",
                    "chunk_audio_dir": str(args.chunk_audio_dir) if args.chunk_audio_dir else None,
                    "selected_chunk_indices": ordered_chunk_indices,
                    "show_active_line": args.show_active_line,
                    "active_line_timing": active_line_timing_used if args.show_active_line else "disabled",
                    "active_line_asr_match_ratio": active_line_match_ratio,
                    "timing_mode": (
                        "kokoro_segments_chapter_paused"
                        if exact_timing_payload and exact_timing_payload["chapter_pauses"]
                        else "chunk_audio_exact"
                        if args.chunk_audio_dir and exact_timing_payload
                        else "kokoro_segments"
                        if exact_timing_payload
                        else "weighted_blocks"
                    ),
                    "focus_line_ratio": args.focus_line_ratio,
                    "chapter_top_margin_ratio": args.chapter_top_margin_ratio,
                    "chapter_end_hold": args.chapter_end_hold,
                    "chapter_transition": args.chapter_transition,
                    "chapter_start_hold": args.chapter_start_hold,
                    "anchors": anchors,
                    "blocks": [asdict(block) for block in blocks],
                    "chapters": [asdict(chapter) for chapter in exact_timing_payload["chapters"]] if exact_timing_payload else [],
                    "chapter_pauses": [asdict(chapter_pause) for chapter_pause in exact_timing_payload["chapter_pauses"]] if exact_timing_payload else [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if args.show_active_line:
            render_video_with_active_line(
                document_image if document_image is not None else Image.open(document_path).convert("RGBA"),
                render_audio_path,
                args.output,
                args.width,
                args.height,
                duration,
                args.fps,
                anchors,
                line_layouts,
                args.video_crf,
                args.video_codec,
                args.video_preset,
            )
        else:
            scroll_expression = build_scroll_expression(anchors, max_scroll=max(0.0, float(document_height - args.height)))
            render_video(
                document_path,
                render_audio_path,
                args.output,
                args.width,
                args.height,
                duration,
                args.fps,
                scroll_expression,
                args.video_crf,
                args.video_codec,
                args.video_preset,
            )
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"Video build failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if document_image is not None:
            document_image.close()
        if cleanup_needed:
            temp_dir.cleanup()

    if args.chapters_output is not None:
        chapter_marks = build_chapter_marks(
            chapters,
            chunk_timings=chunk_timings,
            blocks=blocks,
            total_duration=duration,
        )
        args.chapters_output.parent.mkdir(parents=True, exist_ok=True)
        args.chapters_output.write_text(json.dumps(chapter_marks, indent=2), encoding="utf-8")

    print(f"Created autoscroll video: {args.output}", file=sys.stderr if is_streaming_output(args.output) else sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
