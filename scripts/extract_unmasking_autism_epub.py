#!/usr/bin/env python3
"""Extract audiobook-ready text from the local Unmasking Autism EPUB."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPUB = (
    REPO_ROOT
    / "library"
    / "unmaskingAutism"
    / "Unmasking Autism _ Discovering the New Faces of Neurodiversity_Devon Price_liber3.epub"
)
DEFAULT_OUTPUT = REPO_ROOT / "library" / "unmaskingAutism" / "UnmaskingAutism.txt"
DEFAULT_AUDIT = REPO_ROOT / "library" / "unmaskingAutism" / "UnmaskingAutism_text_audit.json"

INCLUDED_SPINE_IDS = (
    "itr",
    "c001",
    "c002",
    "c003",
    "c004",
    "c005",
    "c006",
    "c007",
    "c008",
    "c009",
)

BOOK_TITLE_LINES = (
    "Unmasking Autism",
    "Discovering the New Faces of Neurodiversity",
    "Devon Price",
)

BAD_SECTION_RE = re.compile(
    r"^(contents|copyright|dedication|acknowledgments|notes|index|glossary|appendix|what's next)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epub", type=Path, default=DEFAULT_EPUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def opf_path_for_epub(epub: ZipFile) -> str:
    container = BeautifulSoup(epub.read("META-INF/container.xml"), "xml")
    rootfile = container.find("rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise RuntimeError("Could not find EPUB OPF rootfile")
    return rootfile["full-path"]


def load_spine(epub: ZipFile, opf_path: str) -> list[tuple[str, str]]:
    opf = BeautifulSoup(epub.read(opf_path), "xml")
    manifest = {
        item["id"]: item["href"]
        for item in opf.find_all("item")
        if item.get("id") and item.get("href")
    }
    base = str(Path(opf_path).parent)
    spine: list[tuple[str, str]] = []
    for itemref in opf.find_all("itemref"):
        idref = itemref.get("idref")
        href = manifest.get(idref or "")
        if idref and href:
            spine.append((idref, str(Path(base) / href)))
    return spine


def clean_soup(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["script", "style", "img", "svg", "nav"]):
        tag.decompose()

    for tag in soup.find_all(True):
        epub_type = " ".join(
            str(tag.get(attr, ""))
            for attr in ("epub:type", "type")
        ).lower()
        role = str(tag.get("role", "")).lower()
        href = str(tag.get("href", "")).lower()
        text = normalize_text(tag.get_text(" ", strip=True))

        if "pagebreak" in epub_type or "pagebreak" in role:
            tag.decompose()
            continue
        if "noteref" in epub_type or "doc-noteref" in role:
            tag.decompose()
            continue
        if tag.name == "a" and (href.startswith("#") or "notes" in href) and re.fullmatch(r"\[?\d+\]?", text):
            tag.decompose()
            continue

    for tag in soup.find_all("a"):
        tag.unwrap()


def is_nested_block(tag) -> bool:
    block_names = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "table"}
    parent = tag.parent
    while parent is not None and getattr(parent, "name", None) != "body":
        if getattr(parent, "name", None) in block_names:
            return True
        parent = parent.parent
    return False


def table_rows(table) -> list[str]:
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(". ".join(cells))
    return rows


def extract_blocks(soup: BeautifulSoup) -> list[str]:
    body = soup.find("body")
    if body is None:
        return []

    blocks: list[str] = []
    for tag in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "table"]):
        if is_nested_block(tag):
            continue
        if tag.name == "table":
            blocks.extend(table_rows(tag))
            continue
        text = normalize_text(tag.get_text(" ", strip=True))
        if not text:
            continue
        if BAD_SECTION_RE.match(text):
            continue
        blocks.append(text)
    return blocks


def collapse_heading_pairs(blocks: list[str]) -> list[str]:
    collapsed: list[str] = []
    index = 0
    while index < len(blocks):
        current = blocks[index]
        next_block = blocks[index + 1] if index + 1 < len(blocks) else ""
        if current.upper() in {"INTRODUCTION", "CONCLUSION"} and next_block:
            collapsed.append(f"{current.title()}: {next_block}")
            index += 2
            continue
        if re.fullmatch(r"CHAPTER\s+\d+", current, flags=re.IGNORECASE) and next_block:
            chapter_number = re.search(r"\d+", current).group(0)
            collapsed.append(f"Chapter {chapter_number}: {next_block}")
            index += 2
            continue
        collapsed.append(current)
        index += 1
    return collapsed


def extract_text(epub_path: Path) -> tuple[str, dict]:
    with ZipFile(epub_path) as epub:
        opf_path = opf_path_for_epub(epub)
        spine = load_spine(epub, opf_path)
        selected = [(idref, href) for idref, href in spine if idref in INCLUDED_SPINE_IDS]

        sections: list[dict] = []
        output_blocks = list(BOOK_TITLE_LINES)
        for idref, href in selected:
            soup = BeautifulSoup(epub.read(href), "xml")
            clean_soup(soup)
            blocks = collapse_heading_pairs(extract_blocks(soup))
            if blocks:
                output_blocks.append("")
                output_blocks.extend(blocks)
            sections.append(
                {
                    "spine_id": idref,
                    "href": href,
                    "paragraph_count": len(blocks),
                    "first_block": blocks[0] if blocks else "",
                }
            )

        clean_text = "\n\n".join(block for block in output_blocks if block != "").strip() + "\n"
        forbidden_matches = [
            line
            for line in clean_text.splitlines()
            if BAD_SECTION_RE.match(line.strip())
        ]
        audit = {
            "source_epub": str(epub_path),
            "included_spine_ids": list(INCLUDED_SPINE_IDS),
            "included_sections": sections,
            "excluded_back_matter": ["dedication", "acknowledgments", "notes", "index", "next-reads"],
            "characters": len(clean_text),
            "words": len(clean_text.split()),
            "paragraphs": len([block for block in clean_text.split("\n\n") if block.strip()]),
            "sha256": hashlib.sha256(clean_text.encode("utf-8")).hexdigest(),
            "forbidden_section_heading_matches": forbidden_matches,
        }
        return clean_text, audit


def main() -> int:
    args = parse_args()
    clean_text, audit = extract_text(args.epub)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(clean_text, encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.audit_output}")
    print(f"Characters: {audit['characters']}")
    print(f"Words: {audit['words']}")
    print(f"Paragraphs: {audit['paragraphs']}")
    if audit["forbidden_section_heading_matches"]:
        print("Forbidden heading matches remain:")
        for match in audit["forbidden_section_heading_matches"]:
            print(match)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
