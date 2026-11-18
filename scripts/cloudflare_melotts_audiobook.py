#!/usr/bin/env python3
"""Cheap/free-tier audiobook generation with Cloudflare Workers AI MeloTTS.

Environment:
  CLOUDFLARE_ACCOUNT_ID=...
  CLOUDFLARE_API_TOKEN=...
Requires ffmpeg only for the final MP3 concatenation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "@cf/myshell-ai/melotts"


def chunk_text(text: str, max_chars: int) -> list[str]:
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidates = [paragraph]
        if len(paragraph) > max_chars:
            candidates = re.split(r"(?<=[.!?])\s+", paragraph)
        for piece in candidates:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) > max_chars:
                for i in range(0, len(piece), max_chars):
                    sub = piece[i : i + max_chars]
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(sub)
                continue
            proposed = f"{current} {piece}".strip()
            if current and len(proposed) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = proposed
    if current:
        chunks.append(current)
    return chunks


def synthesize(account_id: str, token: str, text: str, lang: str, retries: int = 6) -> bytes:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL}"
    payload = json.dumps({"prompt": text, "lang": lang}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                content_type = response.headers.get("Content-Type", "")
            if "audio" not in content_type.lower() and data[:1] in {b"{", b"["}:
                raise RuntimeError(f"Cloudflare returned JSON instead of MP3: {data[:1000]!r}")
            if len(data) < 500:
                raise RuntimeError(f"Cloudflare returned an implausibly small audio response ({len(data)} bytes)")
            return data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise RuntimeError(f"Cloudflare HTTP {exc.code}: {body[:2000]}") from exc
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Final MP3")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--work-dir", type=Path, default=Path("generated/cloudflare_melotts_chunks"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        raise SystemExit("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")

    chunks = chunk_text(args.input.read_text(encoding="utf-8"), args.max_chars)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    mp3_paths: list[Path] = []
    for index, chunk in enumerate(chunks):
        path = args.work_dir / f"chunk_{index:05d}.mp3"
        mp3_paths.append(path)
        if path.exists() and path.stat().st_size > 500 and not args.force:
            print(f"[skip] {index+1}/{len(chunks)} {path.name}")
            continue
        print(f"[run]  {index+1}/{len(chunks)} {len(chunk)} chars")
        path.write_bytes(synthesize(account_id, token, chunk, args.lang))

    concat = args.work_dir / "concat.txt"
    concat.write_text("".join(f"file '{path.resolve()}'\n" for path in mp3_paths), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-c", "copy", str(args.output)
    ], check=True)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
