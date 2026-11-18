#!/usr/bin/env python3
"""Download and validate the seven Schopenhauer Gutenberg volumes used by the audiobook.

This exists because the previous local anthology was silently contaminated with
Moonfleet and Between Whiles.  The script validates every downloaded ebook
before it can enter the combined source file.
"""
from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Volume:
    ebook_id: int
    heading: str
    required_phrases: tuple[str, ...]


VOLUMES = (
    Volume(10714, "On Authorship and Style / The Art of Literature", ("arthur schopenhauer", "art of literature")),
    Volume(10715, "Counsels and Maxims", ("arthur schopenhauer", "counsels and maxims")),
    Volume(10731, "The Art of Controversy", ("arthur schopenhauer", "art of controversy")),
    Volume(10732, "Studies in Pessimism", ("arthur schopenhauer", "studies in pessimism")),
    Volume(10739, "On Human Nature", ("arthur schopenhauer", "human nature")),
    Volume(10741, "The Wisdom of Life", ("arthur schopenhauer", "wisdom of life")),
    Volume(10833, "Religion: A Dialogue and Other Essays", ("arthur schopenhauer", "religion")),
)

BAD_MARKERS = (
    "moonfleet",
    "j. meade falkner",
    "between whiles",
    "helen jackson",
)


def candidate_urls(ebook_id: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-8.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
    ]


def download_text(volume: Volume) -> tuple[str, str]:
    errors: list[str] = []
    request_headers = {"User-Agent": "schopenhauer-audiobook-source-validator/1.0"}
    for url in candidate_urls(volume.ebook_id):
        try:
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read()
            text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
            return text, url
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download Gutenberg ebook " + str(volume.ebook_id) + "\n" + "\n".join(errors))


def strip_gutenberg_wrapper(text: str) -> str:
    # Keep only the book payload between Gutenberg's explicit START/END markers.
    start = re.search(r"^\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*{3}\s*$", text, re.I | re.M)
    end = re.search(r"^\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*{3}\s*$", text, re.I | re.M)
    if not start or not end or end.start() <= start.end():
        raise RuntimeError("Gutenberg START/END markers were not found in the expected order")
    body = text[start.end() : end.start()].strip()
    # Normalize pathological vertical whitespace while preserving paragraphs.
    body = re.sub(r"\n[ \t]+\n", "\n\n", body)
    body = re.sub(r"\n{4,}", "\n\n\n", body)
    return body


def _normalized_search(text: str) -> str:
    # Collapse all whitespace runs to single spaces so that title-page and
    # typesetting artifacts (e.g. "ARTHUR  SCHOPENHAUER") cannot defeat a
    # phrase match that a human reader would accept.
    return re.sub(r"\s+", " ", text.lower())


def validate_volume(volume: Volume, body: str) -> None:
    lowered = body.lower()
    searchable = _normalized_search(body)
    missing = [phrase for phrase in volume.required_phrases if phrase not in searchable]
    if missing:
        raise RuntimeError(f"ebook {volume.ebook_id} failed identity validation; missing {missing}")
    bad = [marker for marker in BAD_MARKERS if marker in searchable]
    if bad:
        raise RuntimeError(f"ebook {volume.ebook_id} contains known wrong-book markers: {bad}")
    if len(body) < 10_000:
        raise RuntimeError(f"ebook {volume.ebook_id} is implausibly short ({len(body)} chars)")


def validate_combined(text: str) -> None:
    lowered = _normalized_search(text)
    bad = [marker for marker in BAD_MARKERS if marker in lowered]
    if bad:
        raise RuntimeError(f"combined anthology contains known contamination markers: {bad}")
    if "project gutenberg ebook moonfleet" in lowered or "project gutenberg ebook between whiles" in lowered:
        raise RuntimeError("combined anthology contains a wrong Project Gutenberg end marker")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("library/schopenhauer_parerga/schopenhauer_parerga_clean.txt"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("library/schopenhauer_parerga/gutenberg_raw"))
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    for volume in VOLUMES:
        raw_path = args.cache_dir / f"pg{volume.ebook_id}.txt"
        if raw_path.exists():
            raw = raw_path.read_text(encoding="utf-8")
            source = str(raw_path)
        else:
            raw, source = download_text(volume)
            raw_path.write_text(raw, encoding="utf-8")
        body = strip_gutenberg_wrapper(raw)
        validate_volume(volume, body)
        print(f"ok: {volume.ebook_id} {volume.heading} <- {source}")
        sections.append(f"## {volume.heading}\n\n{body}")

    combined = "\n\n\n".join(sections).strip() + "\n"
    validate_combined(combined)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(combined, encoding="utf-8")
    print(f"wrote {args.output} ({len(combined):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
