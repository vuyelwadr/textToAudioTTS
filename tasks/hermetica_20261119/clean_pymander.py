#!/usr/bin/env python3
"""Clean the Bodleian OCR of The Divine Pymander into a TTS manuscript."""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

RAW = Path("/tmp/pymander_ocr/pymander_raw.txt")
OUT = Path("/tmp/pymander_ocr/divinePymander_clean.txt")

PAGE_MARKER_RE = re.compile(r"^===== PAGE (\d+) =====$", re.M)
ORDINALS = [
    "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH",
    "EIGHTH", "NINTH", "TENTH", "ELEVENTH", "TWELFTH", "THIRTEENTH",
    "FOURTEENTH", "FIFTEENTH", "SIXTEENTH", "SEVENTEENTH",
]
HEADING_RE = re.compile(
    r"^(?:THE\s+)?(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|"
    r"TENTH|ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH|FIFTEENTH|SIXTEENTH|SEVENTEENTH|"
    r"FPOURTEENTA|TITHE|LAE\s+TWELP\s+TH)\b[^\n]{0,70}\b(?:BOOK|BOUK|BOCK)\b[^\n]{0,40}$",
    re.IGNORECASE,
)
VERSE_RE = re.compile(r"^\s*(\d{1,3})\s*[.)]\s*(.*)$")
TERMINAL = tuple(".!?\"”’")
ROMAN_OR_NUM_RE = re.compile(r"^[\s\(\)\[\]\.\,\:\;\-\u2013\u2014\dIVXLCDMivxlcdm%°oO]*$")

BOOK_TITLES = {
    "FIRST": "THE FIRST BOOK OF HERMES TRISMEGISTUS.",
    "SECOND": "THE SECOND BOOK, CALLED PIMANDER.",
    "THIRD": "THE THIRD BOOK, CALLED THE HOLY SERMON.",
    "FOURTH": "THE FOURTH BOOK, CALLED THE KEY.",
    "FIFTH": "THE FIFTH BOOK: THAT GOD IS NOT MANIFEST, AND YET MOST MANIFEST.",
    "SIXTH": "THE SIXTH BOOK: THAT IN GOD ALONE IS GOOD.",
    "SEVENTH": "THE SEVENTH BOOK: HIS SECRET SERMON IN THE MOUNT OF REGENERATION, AND THE PROFESSION OF SILENCE.",
    "EIGHTH": "THE EIGHTH BOOK: THAT THE GREATEST EVIL IN MAN IS THE NOT KNOWING GOD.",
    "NINTH": "THE NINTH BOOK: A UNIVERSAL SERMON TO ASCLEPIUS.",
    "TENTH": "THE TENTH BOOK: THE MIND TO HERMES.",
    "ELEVENTH": "THE ELEVENTH BOOK: OF THE COMMON MIND, TO TAT.",
    "TWELFTH": "THE TWELFTH BOOK: HIS CRATER, OR MONAS.",
    "THIRTEENTH": "THE THIRTEENTH BOOK: OF SENSE AND UNDERSTANDING.",
    "FOURTEENTH": "THE FOURTEENTH BOOK: OF OPERATION AND SENSE.",
    "FIFTEENTH": "THE FIFTEENTH BOOK: OF TRUTH, TO HIS SON TAT.",
    "SIXTEENTH": "THE SIXTEENTH BOOK: THAT NONE OF THE THINGS THAT ARE CAN PERISH.",
    "SEVENTEENTH": "THE SEVENTEENTH BOOK: TO ASCLEPIUS, TO BE TRULY WISE.",
}
HEADING_ORDINAL_FIX = {
    "FPOURTEENTA": "FOURTEENTH",
    "TITHE": "FOURTH",
    "LAE TWELP TH": "TWELFTH",
}
DROP_CAP_FIXES = {
    "HE glory of all things": "THE glory of all things",
    "N the general speeches": "IN the general speeches",
    "HITHER are you carried": "WHITHER are you carried",
    "LL that is moved": "ALL that is moved",
    "ORBEAR thy Speech": "FORBEAR thy Speech",
    "HE Mind, O Tat": "THE Mind, O Tat",
    "HE Workman made": "THE Workman made",
    "ESTERDAY, Asclepius": "YESTERDAY, Asclepius",
    "ESTERDAY’S Speech": "YESTERDAY’S Speech",
    "HOU hast well explained": "THOU hast well explained",
    "F TRUTH, O Tat": "OF TRUTH, O Tat",
    "E must now speak": "WE must now speak",
    "ECAUSE, my Son": "BECAUSE, my Son",
    "OD, O Asclepius": "GOD, O Asclepius",
    "Y THOUGHTS being": "MY THOUGHTS being",
    "HIS Discourse I will": "THIS Discourse I will",
    "HIS «SECRET SERMON": "HIS SECRET SERMON",
    "Tat. HOU hast": "Tat. THOU hast",
}
TOKEN_FIXES = {
    r"\bYat\.": "Tat.",
    r"\bTut\.": "Tat.",
    r"\bZrism\.": "Trismegistus.",
    r"\bTrism\.": "Trismegistus.",
    r"\bPim\.": "Pimander.",
    r"\bPmand\.": "Pimander.",
    r"\bPoemander\b": "Pimander",
    r"\bHerm\.": "Hermes.",
    r"\bAsclep\.": "Asclepius.",
    r"\bAZind\b": "Mind",
    r"\bAZzxd\b": "Mind",
    r"\bZrruth\b": "Truth",
    r"\bZruth\b": "Truth",
    r"\bTrnth\b": "Truth",
    r"\bJndia\b": "India",
    r"\bCefius\b": "Cælius",
    r"\bTothe\b": "To the",
    r"\bisa\b": "is a",
    r"\bitis\b": "it is",
    r"\bthethings\b": "the things",
    r"\bofthe\b": "of the",
    r"\binthe\b": "in the",
    r"\bMax\b": "Man",
    r"\bSévaroc\b": "thanatos",
    r"\bNotall\b": "Not all",
    r"\bTirice\b": "Thrice",
    r"\bTurice\b": "Thrice",
    r"\bTrismecistus\b": "Trismegistus",
    r"\bA‘sculapius\b": "Asclepius",
    r"\bAfsculapius\b": "Asclepius",
    r"\binuence\b": "influence",
    r"\bHWestory\b": "History",
    r"\basa\b": "as a",
    r"\btn this world\b": "in this world",
    r"\bEgy ptian\b": "Egyptian",
    r"\bwhere where\b": "where",
    r"\bJat\b": "Tat",
    r"\bMusitians\b": "Musicians",
    r"\bfourfooted\b": "four-footed",
    r"\bthy son's\b": "thy soul",
    r"\battain-to\b": "attain to",
    r"\bknow both \{where\b": "know both where",
    r"§ ": "",
    r"\bPismures\b": "Pismires",
    r"\[and and\b": "and",
}


def fix_tokens(text: str) -> str:
    for pattern, good in TOKEN_FIXES.items():
        text = re.sub(pattern, good, text)
    text = re.sub(r"\b1\b(?=\s+[a-z])", "I", text)
    text = re.sub(
        r"\b1\s+(am|will|have|know|say|saw|thought|pray|give|shall|would|could|must)\b",
        r"I \1",
        text,
    )
    text = re.sub(
        r"\|\s+(am|will|have|know|say|saw|thought|pray|give|shall|would|could|must)\b",
        r"I \1",
        text,
    )
    text = re.sub(r"\|\s+(?=[a-z])", "I ", text)
    text = text.replace("  ", " ")
    return text


def fix_drop_cap(line: str) -> str:
    for bad, good in DROP_CAP_FIXES.items():
        if line.startswith(bad):
            return good + line[len(bad):]
    return line


def is_page_number(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 14:
        return False
    if ROMAN_OR_NUM_RE.match(stripped):
        return True
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    digits = re.sub(r"[^0-9]", "", stripped)
    return bool(digits) and len(letters) <= 3 and len(stripped) <= 12


def is_decorative(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    alpha = sum(ch.isalpha() for ch in stripped)
    if alpha <= 1 or alpha / len(stripped) < 0.40:
        return True
    tokens = stripped.split()
    if tokens and sum(len(token) for token in tokens) / len(tokens) < 2.6:
        return True
    return False


def clean_page_lines(lines: list[str]) -> list[str]:
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if is_page_number(stripped) or is_decorative(stripped):
            continue
        if re.match(r"(?i)^the end of the .{0,50}book[.,]?$", stripped):
            continue
        kept.append(stripped)
    while kept and not kept[0]:
        kept.pop(0)
    while kept and not kept[-1]:
        kept.pop()
    return kept


def join_lines(lines: list[str]) -> str:
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
        if prev.endswith("-") and len(prev) > 1:
            out[-1] = prev[:-1] + line
        else:
            out[-1] = prev + " " + line
    return " ".join(out)


def flow_paragraphs(lines: list[str]) -> list[str]:
    lengths = [len(line.strip()) for line in lines if line.strip()]
    median = statistics.median(lengths) if lengths else 0
    paragraphs: list[list[str]] = [[]]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if paragraphs[-1]:
                paragraphs.append([])
            continue
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


def verse_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[list[str]] = [[]]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = VERSE_RE.match(stripped)
        if match:
            rest = match.group(2).strip()
            if rest:
                paragraphs.append([rest])
            continue
        paragraphs[-1].append(stripped)
    result = []
    for block in paragraphs:
        text = join_lines(block)
        if text:
            text = fix_drop_cap(re.sub(r"\s+", " ", text).strip())
            result.append(fix_tokens(text))
    return result


def is_title_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.upper() == stripped and len(stripped) < 100:
        return True
    if stripped.upper().startswith("OF "):
        return True
    return False


def ordinal_from_heading(line: str) -> str | None:
    upper = re.sub(r"[^A-Z ]", " ", line.upper())
    upper = re.sub(r"\s+", " ", upper).strip()
    if upper.startswith("THE "):
        upper = upper[4:]
    for raw, fixed in HEADING_ORDINAL_FIX.items():
        if upper.startswith(raw):
            return fixed
    for ordinal in ORDINALS:
        if upper.startswith(ordinal):
            return ordinal
    return None


def main() -> int:
    raw = RAW.read_text(encoding="utf-8")
    pages: dict[int, list[str]] = {}
    matches = list(PAGE_MARKER_RE.finditer(raw))
    for i, match in enumerate(matches):
        page = int(match.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        pages[page] = raw[match.end():end].splitlines()

    out: list[str] = ["THE DIVINE PYMANDER", "Hermes Mercurius Trismegistus", ""]

    # Introduction (pages 18-31).
    intro_lines: list[str] = []
    for page in range(18, 32):
        intro_lines.extend(clean_page_lines(pages.get(page, [])))
    out.append("INTRODUCTION.")
    for paragraph in flow_paragraphs(intro_lines):
        if paragraph.upper().startswith("INTRODUCTION"):
            paragraph = paragraph[len("INTRODUCTION"):].lstrip(". ")
        paragraph = paragraph.replace("SE os Greeks applied", "The Greeks applied")
        if paragraph.startswith("os Greeks applied"):
            paragraph = "The " + paragraph
        if paragraph.startswith("*"):
            continue
        out.append(fix_tokens(paragraph))
        out.append("")

    # Body (pages 34-145).
    body_lines: list[str] = []
    for page in range(34, 146):
        body_lines.extend(clean_page_lines(pages.get(page, [])))
    joined_for_cut = "\n".join(body_lines)
    cut = joined_for_cut.lower().find("the end of the seventeenth book")
    if cut != -1:
        body_lines = joined_for_cut[:cut].splitlines()

    current_book: str | None = None
    block: list[str] = []
    first_body_line_pending = False

    def flush_block() -> None:
        nonlocal block, first_body_line_pending
        if current_book is None:
            block = []
            return
        if block:
            if first_body_line_pending:
                block[0] = fix_drop_cap(block[0])
                first_body_line_pending = False
            for paragraph in verse_paragraphs(block):
                out.append(paragraph)
                out.append("")
        block = []

    for line in body_lines:
        ordinal = ordinal_from_heading(line) if HEADING_RE.match(line) else None
        if ordinal is not None:
            flush_block()
            current_book = ordinal
            out.append("")
            out.append(BOOK_TITLES[ordinal])
            out.append("")
            first_body_line_pending = True
            continue
        if current_book is None:
            continue
        if not block and is_title_line(line):
            continue
        if not block:
            tokens = line.split()
            if tokens and sum(len(token) for token in tokens) / len(tokens) < 3.5:
                continue
        block.append(line)
    flush_block()

    text = "\n".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text.split())} words, {len(text)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
