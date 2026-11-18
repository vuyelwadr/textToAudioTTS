#!/usr/bin/env python3
"""Extract audiobook-ready text from the local *Tempted by Danger* EPUB."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import posixpath
import re
from html.parser import HTMLParser
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPUB = REPO_ROOT / "library" / "temptedByDanger" / "_OceanofPDF.com_Tempted_by_danger_-_Endiya_Carter.epub"
DEFAULT_OUTPUT = REPO_ROOT / "library" / "temptedByDanger" / "temptedByDanger.txt"
DEFAULT_AUDIT = REPO_ROOT / "library" / "temptedByDanger" / "temptedByDanger_text_audit.json"

TITLE = "Tempted by Danger"
AUTHOR = "Endiya Carter"
TITLE_LINES = (TITLE, AUTHOR)

START_RE = re.compile(r"^(prologue|chapter\s+1|chapter\s+i)\b", re.IGNORECASE)
STOP_RE = re.compile(r"^(message|connect with me|about the author|also by)\b", re.IGNORECASE)
SKIP_RE = re.compile(r"^(copyright|contents|synopsis|dedication|acknowledg(?:e)?ments?)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epub", type=Path, default=DEFAULT_EPUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = html.unescape(text.replace("\xa0", " "))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return text.strip()


def opf_path(epub: ZipFile) -> str:
    container = ET.fromstring(epub.read("META-INF/container.xml"))
    rootfile = container.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
    if rootfile is None or not rootfile.attrib.get("full-path"):
        raise RuntimeError("Could not find EPUB OPF rootfile")
    return rootfile.attrib["full-path"]


def spine_paths(epub: ZipFile) -> list[str]:
    opf = ET.fromstring(epub.read(opf_path(epub)))
    ns = {"opf": "http://www.idpf.org/2007/opf"}
    manifest = {
        item.attrib["id"]: item.attrib["href"]
        for item in opf.findall(".//opf:item", ns)
        if item.attrib.get("id") and item.attrib.get("href")
    }
    opf_dir = posixpath.dirname(opf_path(epub))
    paths: list[str] = []
    for itemref in opf.findall(".//opf:itemref", ns):
        idref = itemref.attrib.get("idref")
        href = manifest.get(idref or "")
        if idref and href:
            paths.append(posixpath.normpath(posixpath.join(opf_dir, href)))
    return paths


class BlockTextParser(HTMLParser):
    block_tags = {"p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "br"}
    ignored_tags = {"script", "style", "svg", "nav", "img"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.current: list[str] = []
        self.ignore_depth = 0

    def flush(self) -> None:
        if not self.current:
            return
        block = normalize_text(" ".join(self.current))
        self.current = []
        if block:
            self.blocks.append(block)

    def handle_starttag(self, tag: str, attrs):
        if tag in self.ignored_tags:
            self.ignore_depth += 1
            return
        if self.ignore_depth:
            return
        if tag == "br":
            self.flush()

    def handle_endtag(self, tag: str):
        if tag in self.ignored_tags:
            self.ignore_depth = max(0, self.ignore_depth - 1)
            return
        if self.ignore_depth:
            return
        if tag in self.block_tags:
            self.flush()

    def handle_data(self, data: str):
        if self.ignore_depth:
            return
        if data.strip():
            self.current.append(data)


def extract_blocks(epub: ZipFile, href: str) -> list[str]:
    parser = BlockTextParser()
    parser.feed(epub.read(href).decode("utf-8", "ignore"))
    parser.flush()
    return parser.blocks


def extract_text(epub_path: Path) -> tuple[str, dict]:
    story_started = False
    stop_marker = None
    included_blocks: list[str] = list(TITLE_LINES) + [""]
    skipped_front_matter = 0
    story_block_count = 0

    with ZipFile(epub_path) as epub:
        for href in spine_paths(epub):
            if not href.lower().endswith((".html", ".xhtml", ".htm")):
                continue
            for raw_block in extract_blocks(epub, href):
                block = normalize_text(raw_block.replace("OceanofPDF.com", ""))
                if not block:
                    continue

                lower = block.lower()
                if not story_started:
                    if lower in {TITLE.lower(), AUTHOR.lower(), "dymon danger"}:
                        skipped_front_matter += 1
                        continue
                    if SKIP_RE.match(block):
                        skipped_front_matter += 1
                        continue
                    if START_RE.match(block):
                        story_started = True
                        included_blocks.append(block)
                        story_block_count += 1
                        continue
                    skipped_front_matter += 1
                    continue

                if STOP_RE.match(block):
                    stop_marker = block
                    break

                if lower in {TITLE.lower(), AUTHOR.lower(), "oceanofpdf.com"}:
                    continue

                included_blocks.append(block)
                story_block_count += 1

            if stop_marker is not None:
                break

    text = "\n\n".join(block for block in included_blocks if block is not None).strip() + "\n"
    audit = {
        "source_epub": str(epub_path),
        "output": str(DEFAULT_OUTPUT),
        "title": TITLE,
        "author": AUTHOR,
        "characters": len(text),
        "words": len(text.split()),
        "included_blocks": len([block for block in included_blocks if block]),
        "story_block_count": story_block_count,
        "skipped_front_matter_blocks": skipped_front_matter,
        "start_marker_found": story_started,
        "stop_marker": stop_marker,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return text, audit


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
    print(f"Included blocks: {audit['included_blocks']}")
    print(f"Story blocks: {audit['story_block_count']}")
    if audit["stop_marker"]:
        print(f"Stop marker: {audit['stop_marker']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
