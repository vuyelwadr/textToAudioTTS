#!/usr/bin/env python3
"""
LLM-assisted chapter+paragraph structuring for ASR-sourced transcripts.

process_youtube_hoys_audiobook.paragraphize() breaks raw ASR segments into
paragraphs with a fixed 95-word heuristic -- it has no notion of discourse
structure (topic/speaker/temporal shifts), so long, single-register passages
get cut at an arbitrary word count instead of a real paragraph boundary. This
module is the opt-in, higher-quality replacement: it batches ASR segments
into review-sized windows, asks Claude to insert paragraph breaks only at
genuine discourse boundaries, and verifies -- via a word-multiset diff -- that
nothing was silently dropped, invented, or rewritten before the result ever
reaches TTS/rendering.

Output convention (unchanged from process_youtube_hoys_audiobook.build_chunks()
and consumed as-is by make_autoscroll_video.is_heading_block()/build_chapters()):
    - one "# {chapter title}" heading line per chapter
    - paragraphs separated by exactly one blank line

Two entry points:
    llm_paragraphize(segments, ...)              -- the LLM-assisted transform,
                                                      operating on Segment objects
                                                      (so it can align chapter_cues
                                                      by timestamp).
    paragraphize_with_strategy(texts, ...)        -- the single dispatcher seam:
                                                      strategy="naive" is a byte-
                                                      identical passthrough to
                                                      process_youtube_hoys_audiobook
                                                      .paragraphize(); strategy="llm"
                                                      calls llm_paragraphize() above.

Called by: process_youtube_hoys_audiobook.build_chunks() (via
paragraphize_with_strategy(), once that file threads its own
--paragraph-strategy flag through) and, for manual review/testing, directly
via this file's CLI:

    python3 scripts/llm_paragraphize.py \
        --transcript-json library/tolstoy/someBook/someBookTranscript.json \
        --heading "Some Book" \
        --chapters-json manual_chapters.json \
        --out /tmp/someBook_llm_paragraphized.txt

`--chapters-json` (optional) is a JSON list of {"start_seconds": float,
"title": str} objects -- the same shape the unified pipeline's
--manual-chapters flag is designed to accept.

Requires the `anthropic` package and API credentials (ANTHROPIC_API_KEY, or an
`ant auth login` profile) only when strategy="llm" is actually invoked; the
"naive" path and every pure helper in this file have no such dependency.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# process_youtube_hoys_audiobook.py lives alongside this file in scripts/.
# When this module is imported by a caller that was itself launched as
# `python scripts/<something>.py`, scripts/ is already on sys.path (Python
# puts the entry-point script's own directory at sys.path[0]) and this is a
# no-op. Bootstrap it explicitly so this file is also independently
# importable/runnable regardless of caller cwd.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from process_youtube_hoys_audiobook import (  # noqa: E402
    ChapterCue,
    Segment,
    paragraphize as _naive_paragraphize,
)

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_WINDOW_WORDS = 1200
DEFAULT_INTEGRITY_TOLERANCE = 0.01  # 1% word-multiset divergence budget

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

_SYSTEM_PROMPT = """You are a careful text-structuring assistant. You are given one \
window of a raw ASR (automatic speech recognition) transcript excerpt from an \
audiobook-style narration and must restructure it into paragraphs for on-screen \
scrolling-text video rendering.

Your ONLY job is to insert paragraph breaks at genuine discourse boundaries -- a \
change of topic, a change of speaker, or a clear temporal/scene shift -- and to \
apply light whitespace/punctuation normalization (collapse repeated spaces, fix \
obviously mangled sentence-ending punctuation left over from ASR). Paragraph \
length should follow the discourse structure, not a fixed word count: a short \
paragraph is fine if the topic genuinely changes quickly, and a long paragraph is \
fine if the passage is a single sustained thought.

You must NOT:
- summarize, paraphrase, condense, or rewrite any wording
- delete, add, invent, or reorder any words
- change word choice, tense, number, or spelling (beyond trivial whitespace/\
punctuation touch-ups)
- add chapter headings, titles, bullet points, section markers, or commentary of \
any kind -- the caller adds headings separately

Return paragraphs separated by exactly one blank line between them. Respond with \
ONLY the restructured transcript text -- no preamble, no explanation, no markdown \
code fences, and no quotation marks wrapping the output."""

_USER_PROMPT_TEMPLATE = """This is window {window_index} of {window_count} for the \
chapter "{chapter_title}". Restructure it into paragraphs per your instructions. \
Preserve every word exactly; only insert paragraph breaks and normalize stray \
whitespace/punctuation.

<transcript_excerpt>
{window_text}
</transcript_excerpt>"""


@dataclass(frozen=True)
class ParagraphizeIntegrityError(Exception):
    """Raised when an LLM-paragraphized window's word-multiset diverges from
    the input window's word-multiset beyond `tolerance` -- i.e. the model
    silently dropped, invented, or substantially rewrote words. This is the
    hard guard that keeps llm_paragraphize() lossless: catching this means the
    LLM output must be discarded (retried, or the caller should fall back to
    strategy="naive"), never passed through to TTS/rendering.
    """

    message: str
    expected_word_count: int
    actual_word_count: int
    tolerance: float
    window_index: int | None = None

    def __post_init__(self) -> None:
        # BaseException.args is a C-level slot, not a regular attribute, so
        # this does not trip the frozen dataclass's __setattr__ guard.
        Exception.__init__(self, self.message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def _normalize_words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text)]


def _check_integrity(
    input_text: str,
    output_text: str,
    *,
    tolerance: float,
    window_index: int | None = None,
) -> None:
    input_words = _normalize_words(input_text)
    output_words = _normalize_words(output_text)
    input_counts = Counter(input_words)
    output_counts = Counter(output_words)
    diff = sum((input_counts - output_counts).values()) + sum(
        (output_counts - input_counts).values()
    )
    total = max(1, len(input_words))
    divergence_ratio = diff / total
    if divergence_ratio > tolerance:
        location = f" in window {window_index}" if window_index is not None else ""
        raise ParagraphizeIntegrityError(
            message=(
                f"LLM paragraphize output diverges from its input word-multiset "
                f"by {divergence_ratio:.2%} (tolerance {tolerance:.2%}){location}"
            ),
            expected_word_count=len(input_words),
            actual_word_count=len(output_words),
            tolerance=tolerance,
            window_index=window_index,
        )


def _group_segments_by_chapter(
    segments: list[Segment],
    chapter_cues: list[ChapterCue] | None,
    heading: str,
) -> list[tuple[str, list[Segment]]]:
    """Groups segments into (chapter_title, segments) buckets.

    Mirrors process_youtube_hoys_audiobook.build_chunks()'s advance_chapters()
    boundary rule: a cue fires once a segment's start reaches cue.start (with
    a 0.75s tolerance for ASR/caption jitter, matching build_chunks()
    exactly). When chapter_cues is empty or None, the entire segment list
    becomes a single bucket titled `heading` -- matching build_chunks()'s own
    single-heading-for-the-whole-document fallback when there are no chapter
    cues at all.
    """
    if not chapter_cues:
        return [(heading, list(segments))]

    buckets: list[tuple[str, list[Segment]]] = []
    current_title: str | None = None
    current_segments: list[Segment] = []
    cues = sorted(chapter_cues, key=lambda cue: cue.start)
    cue_index = 0

    def flush() -> None:
        nonlocal current_segments
        if current_segments and current_title is not None:
            buckets.append((current_title, current_segments))
        current_segments = []

    for segment in segments:
        while cue_index < len(cues) and cues[cue_index].start <= segment.start + 0.75:
            flush()
            current_title = cues[cue_index].title
            cue_index += 1
        if current_title is None:
            # No chapter cue has fired yet -- fall back to the caller-supplied
            # heading for this leading span (mirrors build_chunks()'s
            # single-heading fallback for content before the first cue).
            current_title = heading
        current_segments.append(segment)
    flush()
    return buckets


def _split_into_windows(
    segments: list[Segment], window_words: int
) -> list[list[Segment]]:
    """Batches a chapter's segments into review-sized windows so a single LLM
    call never has to hold an entire multi-hour chapter in context. Windows
    close on a sentence-ending boundary once the running word count reaches
    `window_words`, mirroring the closing condition
    process_youtube_hoys_audiobook.paragraphize() already uses for its own
    95-word paragraph flush -- this keeps each window's tail a complete
    sentence, which materially improves the LLM's paragraph-break judgment at
    the window edge.
    """
    windows: list[list[Segment]] = []
    current: list[Segment] = []
    current_words = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        current.append(segment)
        current_words += len(text.split())
        if current_words >= window_words and re.search(r"[.!?\"']$", text):
            windows.append(current)
            current = []
            current_words = 0
    if current:
        windows.append(current)
    return windows


def _call_llm(
    client: Any,
    *,
    model: str,
    chapter_title: str,
    window_index: int,
    window_count: int,
    window_text: str,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": _USER_PROMPT_TEMPLATE.format(
                    window_index=window_index + 1,
                    window_count=window_count,
                    chapter_title=chapter_title,
                    window_text=window_text,
                ),
            }
        ],
    )
    if getattr(response, "stop_reason", None) == "refusal":
        raise RuntimeError(
            f"LLM declined to paragraphize window {window_index + 1}/{window_count} "
            f"of chapter {chapter_title!r} (stop_reason=refusal)"
        )
    parts = [
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    ]
    text = "".join(parts).strip()
    # Defensive cleanup in case the model wraps the output despite instructions.
    text = re.sub(r"^```(?:text)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def llm_paragraphize(
    segments: list[Segment],
    *,
    chapter_cues: list[ChapterCue] | None = None,
    heading: str,
    window_words: int = DEFAULT_WINDOW_WORDS,
    model: str = DEFAULT_MODEL,
    integrity_tolerance: float = DEFAULT_INTEGRITY_TOLERANCE,
) -> str:
    """Returns chapter+paragraph-structured text using the SAME textual convention
    consumed downstream: one '# {chapter title}' heading line per chapter
    (matching process_youtube_hoys_audiobook.build_chunks()'s existing
    heading-block format, which make_autoscroll_video.is_heading_block()/
    build_chapters() already parse), and paragraphs separated by a blank line,
    split on genuine discourse boundaries (topic/speaker/temporal shift)
    rather than a fixed word count. Must be lossless over input words
    (verified via ParagraphizeIntegrityError) -- may only insert paragraph/
    heading boundaries and light whitespace/punctuation normalization, never
    summarize or rewrite content.

    Raises:
        RuntimeError: the `anthropic` package/credentials are unavailable, or
            the model declined a window (stop_reason == "refusal").
        ParagraphizeIntegrityError: a window's LLM output diverges from that
            window's input word-multiset beyond `integrity_tolerance`.
    """
    clean_segments = [
        segment for segment in segments if segment.text and segment.text.strip()
    ]
    if not clean_segments:
        return ""

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The 'anthropic' package is required for the LLM paragraphize "
            "strategy (pip install anthropic). Use "
            "paragraphize_with_strategy(..., strategy='naive') instead if the "
            "LLM-assisted path is unavailable in this environment."
        ) from exc

    client = anthropic.Anthropic()

    chapters = _group_segments_by_chapter(clean_segments, chapter_cues, heading)

    document_blocks: list[str] = []
    for chapter_title, chapter_segments in chapters:
        if not chapter_segments:
            continue
        windows = _split_into_windows(chapter_segments, window_words)
        window_paragraphs: list[str] = []
        for window_index, window_segments in enumerate(windows):
            if not window_segments:
                continue
            window_text = " ".join(
                segment.text.strip() for segment in window_segments
            )
            output_text = _call_llm(
                client,
                model=model,
                chapter_title=chapter_title,
                window_index=window_index,
                window_count=len(windows),
                window_text=window_text,
            )
            _check_integrity(
                window_text,
                output_text,
                tolerance=integrity_tolerance,
                window_index=window_index,
            )
            if output_text:
                window_paragraphs.append(output_text)
        chapter_text = "\n\n".join(window_paragraphs)
        document_blocks.append(f"# {chapter_title}")
        if chapter_text:
            document_blocks.append(chapter_text)

    return "\n\n".join(document_blocks).strip()


def _segments_from_texts(texts: list[str]) -> list[Segment]:
    """Synthesizes Segment objects from plain strings for the "llm" strategy
    path of paragraphize_with_strategy(), which -- like
    process_youtube_hoys_audiobook.paragraphize() itself -- only receives
    flat text, not timestamps. The synthetic start/end values preserve
    ordering only; real chapter-cue timestamp alignment requires calling
    llm_paragraphize() directly with real Segment objects (see module
    docstring).
    """
    segments: list[Segment] = []
    for index, text in enumerate(texts):
        if not text or not text.strip():
            continue
        segments.append(Segment(start=float(index), end=float(index + 1), text=text))
    return segments


def paragraphize_with_strategy(
    texts: list[str],
    *,
    strategy: str = "naive",
    chapter_cues: list[ChapterCue] | None = None,
    heading: str = "",
    max_words: int = 95,
) -> str:
    """Dispatcher: strategy="naive" calls
    process_youtube_hoys_audiobook.paragraphize() unchanged (byte-identical
    output to today); strategy="llm" calls llm_paragraphize() above. This is
    the single seam process_youtube_hoys_audiobook.build_chunks() threads
    through, so its own default behavior is provably unchanged when the new
    flag isn't passed.
    """
    if strategy == "naive":
        return _naive_paragraphize(texts, max_words=max_words)
    if strategy == "llm":
        segments = _segments_from_texts(texts)
        return llm_paragraphize(segments, chapter_cues=chapter_cues, heading=heading)
    raise ValueError(
        f"Unknown paragraph strategy {strategy!r}; expected 'naive' or 'llm'"
    )


# --------------------------------------------------------------------------
# CLI -- manual review/testing, independent of any other script in the
# unified pipeline (see design doc Task B: "independently buildable and
# independently callable").
# --------------------------------------------------------------------------


def _load_segments(path: Path) -> list[Segment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Segment(**item) for item in payload["segments"]]


def _load_chapter_cues(path: Path | None) -> list[ChapterCue] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    cues: list[ChapterCue] = []
    for index, entry in enumerate(raw):
        start = float(entry["start_seconds"])
        title = str(entry["title"])
        next_start = float(raw[index + 1]["start_seconds"]) if index + 1 < len(raw) else None
        cues.append(ChapterCue(start=start, end=next_start, title=title))
    return cues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcript-json",
        type=Path,
        required=True,
        help="Path to a *Transcript.json produced by "
        "process_youtube_hoys_audiobook.py (whisper or youtube-auto format)",
    )
    parser.add_argument("--heading", required=True)
    parser.add_argument(
        "--chapters-json",
        type=Path,
        default=None,
        help='Optional JSON [{"start_seconds": float, "title": str}, ...]',
    )
    parser.add_argument("--window-words", type=int, default=DEFAULT_WINDOW_WORDS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--integrity-tolerance", type=float, default=DEFAULT_INTEGRITY_TOLERANCE
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write result here instead of stdout"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    segments = _load_segments(args.transcript_json)
    chapter_cues = _load_chapter_cues(args.chapters_json)
    text = llm_paragraphize(
        segments,
        chapter_cues=chapter_cues,
        heading=args.heading,
        window_words=args.window_words,
        model=args.model,
        integrity_tolerance=args.integrity_tolerance,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text.strip() + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
