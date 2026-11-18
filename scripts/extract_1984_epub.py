#!/usr/bin/env python3
"""Extract audiobook-ready text from the local 1984 EPUB."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from bs4 import BeautifulSoup

try:
    import wordninja
except ImportError:  # pragma: no cover - Mahti/local bootstrap normally installs this.
    wordninja = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPUB = REPO_ROOT / "library" / "1984" / "1984_George Orwell_liber3.epub"
DEFAULT_OUTPUT = REPO_ROOT / "library" / "1984" / "1984.txt"
DEFAULT_AUDIT = REPO_ROOT / "library" / "1984" / "1984_text_audit.json"

TITLE_BLOCKS = ("Nineteen Eighty-Four", "George Orwell")
BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote")
ROMAN_RE = re.compile(r"(?=[IVXLCDM]+\b)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
PART_RE = re.compile(r"^PART\s+([IVXLCDM]+)$", re.IGNORECASE)
FOOTNOTE_RE = re.compile(r"\[[0-9]+\]")
BACKMATTER_RE = re.compile(
    r"^(appendix|the principles of newspeak|notes?|copyright|contents|about the author|"
    r"orwell\.ru|first published|source:)",
    re.IGNORECASE,
)
SLOGAN = "WAR IS PEACEFREEDOM IS SLAVERYIGNORANCE IS STRENGTH"
SLOGAN_REPAIRED = "WAR IS PEACE. FREEDOM IS SLAVERY. IGNORANCE IS STRENGTH."
PHRASE_REPLACEMENTS = (
    ("ChapterIIgnorance is Strength", "Chapter I: Ignorance is Strength"),
    ("WAR IS PEACEFREEDOM IS SLAVERYIGNORANCE IS STRENGTH", SLOGAN_REPAIRED),
    ("acrorss", "across"),
    ("anyactual", "any actual"),
    ("be get children", "beget children"),
    ("broughtdoublethink", "brought doublethink"),
    ("conferno", "confer no"),
    ("Crimestopmeans", "Crimestop means"),
    ("calleddoublethink", "called doublethink"),
    ("comuni cations", "communications"),
    ("dontcare", "dont care"),
    ("doublethinkare", "doublethink are"),
    ("Doublethinklies", "Doublethink lies"),
    ("Doublethinkmeans", "Doublethink means"),
    ("doublethinkhe", "doublethink he"),
    ("double think", "doublethink"),
    ("exercisedoublethink", "exercise doublethink"),
    ("facesuggested", "face suggested"),
    ("fora place", "for a place"),
    ("I\u2019Il", "I\u2019ll"),
    ("inventeddoublethinkand", "invented doublethink and"),
    ("jewesssitting", "Jewess sitting"),
    ("keyword here is black white", "keyword here is blackwhite"),
    ("knowingat", "knowing at"),
    ("dustswirled", "dust swirled"),
    ("ofdoublethink", "of doublethink"),
    ("endvictory", "end -- victory"),
    ("longhoped-for", "long-hoped-for"),
    ("pol its", "polits"),
    ("positionhe", "position he"),
    ("Soand-so", "so-and-so"),
    ("Soandso", "so-and-so"),
    ("tenminutes", "ten minutes"),
    ("they ll", "they'll"),
    ("thoughdoublethinkcomprises", "though doublethink comprises"),
    ("Timesdid", "Times did"),
    ("upondoublethink", "upon doublethink"),
    ("vapourizedwas", "vapourized was"),
    ("whathe", "what he"),
    ("whitecounterpaned", "white-counterpaned"),
    ("worddoublethink", "word doublethink"),
    ("wordscrimestop", "words crimestop"),
    ("were shape him", "we reshape him"),
    ("a like", "alike"),
    ("in form", "inform"),
)

PREFIX_PARTS = {
    "a",
    "after",
    "against",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "before",
    "between",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "down",
    "even",
    "ever",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "his",
    "if",
    "in",
    "inside",
    "into",
    "is",
    "its",
    "my",
    "never",
    "no",
    "not",
    "of",
    "off",
    "on",
    "once",
    "one",
    "only",
    "or",
    "our",
    "out",
    "outside",
    "over",
    "she",
    "so",
    "that",
    "the",
    "their",
    "they",
    "there",
    "these",
    "this",
    "those",
    "through",
    "to",
    "under",
    "up",
    "very",
    "was",
    "we",
    "were",
    "why",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "whose",
    "with",
    "within",
    "without",
    "would",
    "it",
    "you",
    "your",
}
SUFFIX_PARTS = {
    "about",
    "after",
    "again",
    "against",
    "all",
    "along",
    "and",
    "away",
    "back",
    "before",
    "behind",
    "between",
    "but",
    "by",
    "could",
    "down",
    "even",
    "ever",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "him",
    "his",
    "if",
    "in",
    "inside",
    "into",
    "is",
    "it",
    "its",
    "more",
    "never",
    "not",
    "of",
    "off",
    "on",
    "one",
    "only",
    "or",
    "out",
    "outside",
    "over",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "through",
    "to",
    "too",
    "under",
    "up",
    "was",
    "were",
    "when",
    "where",
    "what",
    "which",
    "while",
    "who",
    "whom",
    "whose",
    "with",
    "within",
    "without",
    "would",
    "you",
    "us",
}
NEWSPEAK_WORDS = {
    "antefiling",
    "bellyfeel",
    "blackwhite",
    "crimestop",
    "doubleplusgood",
    "doublethink",
    "duckspeak",
    "facecrime",
    "goodthinkful",
    "ingsoc",
    "miniluv",
    "minipax",
    "miniplenty",
    "minitrue",
    "newspeak",
    "oldspeak",
    "ownlife",
    "prolefeed",
    "rectify",
    "speakwrite",
    "telescreen",
    "thoughtcrime",
    "thoughtcriminal",
    "unperson",
}
DO_NOT_SPLIT = NEWSPEAK_WORDS | {
    "aaronson",
    "alike",
    "blackberry",
    "beget",
    "comrade",
    "demeanour",
    "everlastingly",
    "hitherto",
    "inform",
    "incautiously",
    "neighbourhood",
    "nowadays",
    "overfulfilled",
    "overfulfilment",
    "pannikin",
    "polits",
    "revolutionary",
    "somehow",
    "therefore",
    "understanding",
    "upsub",
    "whereas",
}
MORPHEME_PARTS = {"ed", "er", "ers", "est", "ible", "ing", "ion", "ity", "ive", "ly", "ment", "ness", "ous", "s", "tion"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epub", type=Path, default=DEFAULT_EPUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def clean_inline_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = FOOTNOTE_RE.sub("", text)
    for old, new in PHRASE_REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])I\b", r"\1 I", text)
    for old, new in PHRASE_REPLACEMENTS:
        text = text.replace(old, new)
    return text.strip()


def split_camel_token(token: str) -> list[str] | None:
    if not re.search(r"[a-z][A-Z]", token):
        return None
    parts = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", token).split()
    return parts if len(parts) > 1 else None


def restore_case(token: str, parts: list[str]) -> str:
    output: list[str] = []
    cursor = 0
    for part in parts:
        width = len(part)
        raw = token[cursor : cursor + width]
        output.append(raw if raw else part)
        cursor += width
    return " ".join(output)


def should_accept_wordninja_split(token: str, parts: list[str]) -> bool:
    lower = token.lower()
    if lower in DO_NOT_SPLIT:
        return False
    if len(parts) < 2 or len(parts) > 3:
        return False
    if "".join(parts) != lower:
        return False
    if any(part in MORPHEME_PARTS for part in parts):
        return False
    if min(len(part) for part in parts) == 1 and not any(part in {"a", "i"} for part in parts):
        return False
    if parts[0] in PREFIX_PARTS or parts[-1] in SUFFIX_PARTS:
        return True
    if parts[0].endswith("ly") and len(parts[-1]) >= 4:
        return True
    if len(parts) == 2 and min(len(part) for part in parts) >= 5:
        return True
    return False


def repair_glued_words(text: str, repairs: Counter[str]) -> str:
    if wordninja is None:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) < 5 or token.isupper():
            return token

        camel_parts = split_camel_token(token)
        if camel_parts is not None:
            replacement = " ".join(camel_parts)
            repairs[f"{token} -> {replacement}"] += 1
            return replacement

        lower = token.lower()
        if lower in DO_NOT_SPLIT:
            return token

        parts = wordninja.split(lower)
        if not should_accept_wordninja_split(token, parts):
            return token

        replacement = restore_case(token, parts)
        repairs[f"{token} -> {replacement}"] += 1
        return replacement

    repaired = re.sub(r"\b[A-Za-z][A-Za-z]{4,}\b", replace, text)
    for old, new in PHRASE_REPLACEMENTS:
        repaired = repaired.replace(old, new)
    return repaired


def clean_soup(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["script", "style", "img", "svg", "nav"]):
        tag.decompose()
    for tag in soup.find_all(True):
        epub_type = " ".join(str(tag.get(attr, "")) for attr in ("epub:type", "type")).lower()
        role = str(tag.get("role", "")).lower()
        if "pagebreak" in epub_type or "pagebreak" in role:
            tag.decompose()


def is_nested_block(tag) -> bool:
    parent = tag.parent
    while parent is not None and getattr(parent, "name", None) != "body":
        if getattr(parent, "name", None) in BLOCK_TAGS:
            return True
        parent = parent.parent
    return False


def classify_block(text: str) -> tuple[str, str]:
    part_match = PART_RE.fullmatch(text)
    if part_match:
        return "heading", f"Part {part_match.group(1).upper()}"
    if ROMAN_RE.fullmatch(text) and len(text) <= 8:
        return "heading", f"Chapter {text.upper()}"
    if text == "THE END":
        return "heading", text
    return "paragraph", text


def ends_with_terminal(text: str) -> bool:
    return bool(re.search(r"[.!?;:][\"')\]]*$", text.rstrip()))


def starts_with_lowercase(text: str) -> bool:
    stripped = text.lstrip("\"'([{")
    return bool(stripped) and stripped[0].islower()


def append_block(
    blocks: list[dict[str, str]],
    block: dict[str, str],
    merged_fragments: list[dict[str, str]],
) -> None:
    if (
        block["kind"] == "paragraph"
        and blocks
        and blocks[-1]["kind"] == "paragraph"
        and not ends_with_terminal(blocks[-1]["text"])
        and starts_with_lowercase(block["text"])
    ):
        merged_fragments.append(
            {
                "previous_tail": blocks[-1]["text"][-120:],
                "current_head": block["text"][:120],
                "href": block["href"],
            }
        )
        blocks[-1]["text"] = f"{blocks[-1]['text']} {block['text']}"
        return
    blocks.append(block)


def html_files(epub: ZipFile) -> list[str]:
    names = [name for name in epub.namelist() if name.startswith("content/") and name.endswith(".html")]
    return [name for name in names if not name.endswith("split_0_split_000.html")]


def extract_text(epub_path: Path) -> tuple[str, dict]:
    repairs: Counter[str] = Counter()
    merged_fragments: list[dict[str, str]] = []
    blocks: list[dict[str, str]] = [
        {"kind": "heading", "text": TITLE_BLOCKS[0], "href": "title"},
        {"kind": "heading", "text": TITLE_BLOCKS[1], "href": "title"},
    ]
    sections: list[dict[str, object]] = []
    stopped_at: dict[str, object] | None = None

    with ZipFile(epub_path) as epub:
        for href in html_files(epub):
            soup = BeautifulSoup(epub.read(href), "xml")
            clean_soup(soup)
            body = soup.find("body")
            section_count = 0
            if body is None:
                continue
            for tag in body.find_all(BLOCK_TAGS):
                if is_nested_block(tag):
                    continue
                raw = tag.get_text("", strip=True)
                text = repair_glued_words(clean_inline_text(raw), repairs)
                text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    continue
                if BACKMATTER_RE.match(text) and text != "THE END":
                    stopped_at = {"href": href, "text": text, "reason": "backmatter_heading"}
                    break
                kind, text = classify_block(text)
                append_block(
                    blocks,
                    {"kind": kind, "text": text, "href": href},
                    merged_fragments,
                )
                section_count += 1
                if text == "THE END":
                    stopped_at = {"href": href, "text": text, "reason": "first_the_end"}
                    break
            sections.append({"href": href, "block_count": section_count})
            if stopped_at is not None:
                break

    output = "\n\n".join(block["text"] for block in blocks if block["text"]).strip() + "\n"
    forbidden_matches = [
        line.strip()
        for line in output.splitlines()
        if BACKMATTER_RE.match(line.strip()) and line.strip() != "THE END"
    ]
    audit = {
        "source_epub": str(epub_path),
        "output": str(DEFAULT_OUTPUT),
        "characters": len(output),
        "words": len(output.split()),
        "blocks": len(blocks),
        "headings": [block["text"] for block in blocks if block["kind"] == "heading"],
        "sections": sections,
        "stopped_at": stopped_at,
        "merged_fragment_count": len(merged_fragments),
        "merged_fragment_samples": merged_fragments[:30],
        "word_repair_count": sum(repairs.values()),
        "word_repair_samples": [
            {"repair": repair, "count": count}
            for repair, count in repairs.most_common(120)
        ],
        "forbidden_backmatter_matches": forbidden_matches,
        "sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }
    return output, audit


def main() -> int:
    args = parse_args()
    text, audit = extract_text(args.epub)
    audit["output"] = str(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.audit_output}")
    print(f"Characters: {audit['characters']}")
    print(f"Words: {audit['words']}")
    print(f"Blocks: {audit['blocks']}")
    print(f"Word repairs: {audit['word_repair_count']}")
    print(f"Merged fragments: {audit['merged_fragment_count']}")
    if audit["forbidden_backmatter_matches"]:
        print("Forbidden backmatter matches remain:")
        for match in audit["forbidden_backmatter_matches"]:
            print(match)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
