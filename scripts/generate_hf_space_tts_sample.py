#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from gradio_client import Client


DESCRIPTION = (
    "A clear, natural audiobook narrator with steady pacing, warm tone, and minimal background noise."
)
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def build_fish_args(text: str) -> list[object]:
    return [text, False, None, "", 0, 200, 0.7, 1.2, 0.7, "0", 1]


def build_orpheus_args(text: str) -> list[object]:
    return [text, "Kara", 0.6, 0.95, 1.1, 1200]


def build_parler_args(text: str) -> list[object]:
    return [text, DESCRIPTION, False]


SOURCE_BUILDERS: dict[str, dict[str, str | Callable[[str], list[object]]]] = {
    "orpheus_space": {
        "space": "Karayakar/Orpheus-TTS-Turkish",
        "api_name": "/generate_speech",
        "builder": build_orpheus_args,
    },
    "fish_speech_space": {
        "space": "pineconeT94/Fish-Speech",
        "api_name": "/inference_wrapper",
        "builder": build_fish_args,
    },
    "parler_tts_space": {
        "space": "parler-tts/parler_tts",
        "api_name": "/gen_tts",
        "builder": build_parler_args,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one short TTS sample from a public Hugging Face Space")
    parser.add_argument("input", type=Path, help="Input text file")
    parser.add_argument("--source-id", choices=sorted(SOURCE_BUILDERS), required=True, help="Configured Space source")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory that will receive the generated audio")
    return parser.parse_args()


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input text file is empty: {path}")
    return text


def iter_audio_candidates(value: Any):
    if value is None:
        return
    if isinstance(value, Path):
        yield value
        return
    if isinstance(value, str):
        lowered = value.lower()
        if lowered.endswith(AUDIO_SUFFIXES):
            yield Path(value)
        return
    if isinstance(value, dict):
        for key in ("value", "path", "url"):
            if key in value:
                yield from iter_audio_candidates(value[key])
        for nested in value.values():
            yield from iter_audio_candidates(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from iter_audio_candidates(nested)


def resolve_downloaded_audio(result: Any) -> Path:
    for candidate in iter_audio_candidates(result):
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No downloaded audio file found in Space result: {result!r}")


def copy_audio(source_id: str, downloaded_audio: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{source_id}{downloaded_audio.suffix.lower() or '.wav'}"
    shutil.copy2(downloaded_audio, final_path)
    return final_path


def main() -> int:
    args = parse_args()
    text = read_text(args.input)
    source = SOURCE_BUILDERS[args.source_id]
    builder = source["builder"]
    if not callable(builder):
        raise TypeError(f"Builder is not callable for source {args.source_id}")

    client = Client(str(source["space"]), verbose=False)
    result = client.predict(*builder(text), api_name=str(source["api_name"]))
    downloaded_audio = resolve_downloaded_audio(result)
    final_path = copy_audio(args.source_id, downloaded_audio, args.output_dir)
    print(final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())