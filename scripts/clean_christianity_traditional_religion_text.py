#!/usr/bin/env python3
"""Extract audiobook-ready text from the Christianity/Traditional Religion PDF."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = (
    REPO_ROOT
    / "library"
    / "christianityAndTraditioanReligion"
    / "christianityAndTraditioanReligion.pdf"
)
DEFAULT_OUTPUT = DEFAULT_PDF.with_suffix(".txt")

CHAPTER_TITLES = (
    "Preface",
    "Ndebeleland: society and religion before 1859",
    "The Ndebele and the first missionaries, 1859-68",
    "Christianity and western Zimbabwe under Lobengula, 1870-89",
    "Religious institutions and the fall of the Ndebele kingdom, 1888-97",
    "Missions and the traditional societies, 1897-1923",
    "Christianity and education in western Zimbabwe before 1923",
    "Conclusion",
)

TITLE_ALIASES = {
    "preface": "Preface",
    "ndebeleland: society and religion before 1859": "Ndebeleland: society and religion before 1859",
    "the ndebele and the first missionaries, 1859-68": "The Ndebele and the first missionaries, 1859-68",
    "the ndebele and the first missionaries, 1859—68": "The Ndebele and the first missionaries, 1859-68",
    "christianity and western zimbabwe under lobengula, 1870-89": (
        "Christianity and western Zimbabwe under Lobengula, 1870-89"
    ),
    "religious institutions and the fall of the ndebele kingdom, 1888-97": (
        "Religious institutions and the fall of the Ndebele kingdom, 1888-97"
    ),
    "missions and the traditional societies, 1897-1923": "Missions and the traditional societies, 1897-1923",
    "christianity and education in western zimbabwe before 1923": (
        "Christianity and education in western Zimbabwe before 1923"
    ),
    "conclusion": "Conclusion",
}

STOP_SECTION_PATTERN = re.compile(r"^(Note on archival sources|Bibliography|Index)\b", re.IGNORECASE)
RUNNING_HEADER_PATTERNS = (
    re.compile(r"^(?:[ivxlcdm]+|\d+)\s+Christianity and Traditional Religion", re.IGNORECASE),
    re.compile(r"^Christianity and Traditional Religion.*(?:[ivxlcdm]+|\d+)$", re.IGNORECASE),
)


def run_pdftotext(pdf_path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        raise RuntimeError("pdftotext is required; install poppler-utils/poppler")

    with tempfile.TemporaryDirectory(prefix="christianity_pdf_text_") as tmp_dir:
        raw_path = Path(tmp_dir) / "raw.txt"
        subprocess.run([pdftotext, str(pdf_path), str(raw_path)], check=True)
        return raw_path.read_text(encoding="utf-8", errors="replace")


def clean_text_fragment(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\ufb01": "fi",
        "\ufb02": "fl",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalized_heading_key(lines: list[str]) -> str:
    text = " ".join(clean_text_fragment(line) for line in lines)
    text = re.sub(r"^\d+\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def match_chapter_heading(lines: list[str]) -> tuple[str, int] | None:
    significant = [clean_text_fragment(line) for line in lines if clean_text_fragment(line)]
    significant = [line for line in significant if line not in {".", ",", "°", "oh"}]
    candidates = significant[:5]

    for start in range(min(2, len(candidates))):
        for end in range(start + 1, min(len(candidates), start + 4) + 1):
            key = normalized_heading_key(candidates[start:end])
            title = TITLE_ALIASES.get(key)
            if title:
                consumed = 0
                seen = 0
                for original in lines:
                    consumed += 1
                    if clean_text_fragment(original):
                        if clean_text_fragment(original) not in {".", ",", "°", "oh"}:
                            seen += 1
                        if seen >= end:
                            break
                return title, consumed
    return None


def is_running_header(line: str) -> bool:
    stripped = clean_text_fragment(line)
    if not stripped:
        return False
    if re.fullmatch(r"(?:[ivxlcdm]+|\d+)", stripped, flags=re.IGNORECASE):
        return True
    if stripped == "Christianity and Traditional Religion in Western Zimbabwe":
        return True
    if any(pattern.search(stripped) for pattern in RUNNING_HEADER_PATTERNS):
        return True
    for title in CHAPTER_TITLES:
        prefix = title[: min(len(title), 42)]
        if stripped.lower().startswith(prefix.lower()) and re.search(r"\s+\d+$", stripped):
            return True
    return False


def is_noise_line(line: str) -> bool:
    stripped = clean_text_fragment(line)
    if not stripped:
        return False
    alpha_count = sum(character.isalpha() for character in stripped)
    return len(stripped) > 8 and alpha_count <= 1


def is_map_page(lines: list[str]) -> bool:
    page_text = " ".join(clean_text_fragment(line) for line in lines)
    markers = (
        "Distribution of Mission Stations",
        "Distribution of peoples",
        "Area surveyed for white",
        "Government industrial school",
    )
    return any(marker in page_text for marker in markers)


def reflow_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        paragraph = " ".join(current)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
        current.clear()

    for raw_line in lines:
        line = clean_text_fragment(raw_line)
        if not line:
            flush()
            continue
        if line.startswith("# "):
            flush()
            paragraphs.append(line)
            continue
        if current and current[-1].endswith("-") and line[:1].islower():
            current[-1] = current[-1][:-1] + line
            continue
        current.append(line)

    flush()
    return "\n\n".join(paragraphs).strip() + "\n"


def extract_clean_text(raw_text: str) -> str:
    pages = raw_text.split("\f")
    collected: list[str] = []
    started = False
    in_notes = False
    seen_titles: set[str] = set()

    for page in pages:
        raw_lines = page.splitlines()
        lines = [line.rstrip() for line in raw_lines]
        if not any(line.strip() for line in lines):
            continue
        if is_map_page(lines):
            continue

        heading = match_chapter_heading(lines)
        if not started:
            if not heading or heading[0] != "Preface":
                continue
            started = True

        if any(STOP_SECTION_PATTERN.match(clean_text_fragment(line)) for line in lines if clean_text_fragment(line)):
            break

        if in_notes:
            if heading and heading[0] not in seen_titles:
                in_notes = False
            else:
                continue

        page_lines = list(lines)
        heading = match_chapter_heading(page_lines)
        if heading:
            title, consumed = heading
            if title not in seen_titles:
                collected.append(f"# {title}")
                collected.append("")
                seen_titles.add(title)
            page_lines = page_lines[consumed:]

        for line in page_lines:
            stripped = clean_text_fragment(line)
            if not stripped:
                collected.append("")
                continue
            if stripped in {".", ",", "°", "oh"}:
                continue
            if stripped == "Notes":
                in_notes = True
                break
            if stripped in seen_titles:
                continue
            if any(
                title in seen_titles
                and len(stripped) <= len(title)
                and (title.startswith(stripped) or stripped.startswith(title[:40]))
                for title in CHAPTER_TITLES
            ):
                continue
            if is_running_header(line) or is_noise_line(line):
                continue
            collected.append(line)

        collected.append("")

    return reflow_lines(collected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_text = run_pdftotext(args.pdf)
    clean_text = extract_clean_text(raw_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(clean_text, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Characters: {len(clean_text)}")
    print(f"Words: {len(clean_text.split())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
