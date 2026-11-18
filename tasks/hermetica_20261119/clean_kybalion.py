#!/usr/bin/env python3
"""Clean the YOGeBooks Kybalion text layer into a TTS-ready manuscript."""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

RAW = Path("/tmp/kyb/kybalion_raw.txt")
OUT = Path("/tmp/kyb/kybalion_clean.txt")

PAGE_NUMBER_RE = re.compile(r"^\s*(?:[ivxlcdm]+|\d{1,3})\s*$", re.IGNORECASE)
RUNNING_HEADERS = {
    "the kybalion",
    "introduction",
    "the hermetic philosophy",
    "the seven hermetic principles",
    "mental transmutation",
    "the all",
    "the mental universe",
    "the divine paradox",
    "“the all” in all",
    "the planes of correspondence",
    "vibration",
    "polarity",
    "rhythm",
    "causation",
    "gender",
    "mental gender",
    "hermetic axioms",
    "bibliography",
}
CHAPTER_RE = re.compile(r"^Chapter\s+([IVXL]+)\.\s*$")
TERMINAL = tuple(".!?\"”’")


def is_header(line: str) -> bool:
    return line.strip().lower().rstrip(".") in RUNNING_HEADERS


def join_lines(lines: list[str]) -> str:
    """Join OCR/text-layer lines, repairing hyphenation and drop caps."""
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        line = line.replace("\u00ad", "").replace("\u2010", "-")
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        if len(prev) == 1 and prev.isalpha() and line[:1].islower():
            out[-1] = prev + line
        elif prev.endswith("-") and len(prev) > 1:
            out[-1] = prev[:-1] + line
        else:
            out[-1] = prev + " " + line
    return " ".join(out)


def paragraphs_from_lines(lines: list[str]) -> list[str]:
    """Split body lines into paragraphs using short-line + punctuation cues."""
    cleaned = [line.rstrip() for line in lines if line.strip()]
    if not cleaned:
        return []
    lengths = [len(line.strip()) for line in cleaned if len(line.strip()) > 0]
    median = statistics.median(lengths) if lengths else 0
    paragraphs: list[list[str]] = [[]]
    for line in cleaned:
        stripped = line.strip()
        paragraphs[-1].append(stripped)
        short = median and len(stripped) < median * 0.72
        if short and stripped.endswith(TERMINAL):
            paragraphs.append([])
    result = []
    for block in paragraphs:
        text = join_lines(block)
        if text:
            result.append(re.sub(r"\s+", " ", text).strip())
    return result


def main() -> int:
    raw = RAW.read_text(encoding="utf-8-sig")
    pages = raw.split("\f")
    body: list[str] = []
    started = False
    chapter_lines: list[str] = []

    for page in pages:
        lines = page.splitlines()
        # Drop leading/trailing blanks and the page number / running header.
        lines = [line for line in lines if not PAGE_NUMBER_RE.match(line)]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue
        first = lines[0].strip()
        if first.lower() == "introduction.":
            started = True
            body.append("INTRODUCTION")
            lines = lines[1:]
        elif CHAPTER_RE.match(first):
            started = True
            chapter_lines = [first]
            lines = lines[1:]
        elif not started:
            continue

        if chapter_lines:
            while lines and not lines[0].strip():
                lines.pop(0)
            title = lines[0].strip() if lines else ""
            body.append(" ".join(chapter_lines + ([title] if title else [])))
            chapter_lines = []
            lines = lines[1:]

        # Remove a running header at the top of a continuation page.
        if lines and is_header(lines[0]):
            lines = lines[1:]
        # Remove a trailing running header / page furniture.
        while lines and is_header(lines[-1]):
            lines.pop()

        for paragraph in paragraphs_from_lines(lines):
            if paragraph.lower() in {"finis.", "finis"}:
                body.append("FINIS")
                break
            body.append(paragraph)
        if body and body[-1] == "FINIS":
            break

    merged: list[str] = []
    for paragraph in body:
        is_heading = bool(
            re.match(r"^(?:Chapter\s+[IVXL]+\.|INTRODUCTION$|FINIS$)", paragraph)
        )
        if (
            merged
            and not is_heading
            and not re.match(r"^(?:Chapter\s+[IVXL]+\.|INTRODUCTION$|FINIS$)", merged[-1])
            and not re.search(r"[.!?\"”’]$", merged[-1])
        ):
            merged[-1] = merged[-1].rstrip() + " " + paragraph
        else:
            merged.append(paragraph)
    body = merged

    text = "\n\n".join(body)
    # The text layer uses non-breaking hyphens inside compounds; normalise.
    text = text.replace("\u2010", "-").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if " Finis." in text:
        text = text[: text.rfind(" Finis.")].rstrip() + "\n\nFINIS\n"
    text = text.replace("Yon simply", "You simply")
    OUT.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(text.split())} words, {len(text)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
