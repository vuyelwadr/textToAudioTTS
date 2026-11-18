#!/usr/bin/env python3
"""OCR the Bodleian Pymander scan page by page with tesseract."""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PDF = Path("/tmp/bodleian_590481166.pdf")
OUT = Path("/tmp/pymander_ocr")
PAGES = 153
DPI = 300


def ocr_page(page: int) -> tuple[int, str]:
    png = OUT / "png" / f"page_{page:03d}.png"
    txt = OUT / "txt" / f"page_{page:03d}.txt"
    if txt.exists() and txt.stat().st_size > 0:
        return page, "cached"
    png.parent.mkdir(parents=True, exist_ok=True)
    txt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(DPI), "-gray", "-png",
         "-singlefile", str(PDF), str(png.with_suffix(""))],
        check=True,
    )
    proc = subprocess.run(
        ["tesseract", str(png.resolve()), "stdout", "--psm", "3"],
        capture_output=True,
    )
    text = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        txt.write_text("", encoding="utf-8")
        png.unlink(missing_ok=True)
        return page, f"FAILED rc={proc.returncode}: {err[-1] if err else 'no stderr'}"
    txt.write_text(text, encoding="utf-8")
    png.unlink(missing_ok=True)
    return page, f"{len(text)} chars"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for page, status in pool.map(ocr_page, range(1, PAGES + 1)):
            if page % 10 == 0:
                print(f"page {page}: {status}", flush=True)
    combined = OUT / "pymander_raw.txt"
    parts = []
    for page in range(1, PAGES + 1):
        txt = OUT / "txt" / f"page_{page:03d}.txt"
        parts.append(f"\n\n===== PAGE {page} =====\n\n" + txt.read_text(encoding="utf-8"))
    combined.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {combined} ({combined.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
