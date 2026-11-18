from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import torch

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_external_root,
    resolve_model_root,
    resolve_output_dir,
    resolve_torch_dtype,
)

ORPHEUS_PACKAGE_DIR = resolve_external_root() / "Orpheus-TTS" / "orpheus_tts_pypi"
if str(ORPHEUS_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(ORPHEUS_PACKAGE_DIR))

from orpheus_tts import OrpheusModel  # noqa: E402


DEFAULT_ORPHEUS_MODEL = "canopylabs/orpheus-3b-0.1-ft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Orpheus TTS for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--voice", default="leah")
    parser.add_argument("--model-name", default=DEFAULT_ORPHEUS_MODEL)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / "21_orpheus_tts_3b_leah.wav")
    ensure_parent(output_path)

    local_model_dir = resolve_model_root() / args.model_name.rsplit("/", 1)[-1]
    model_source = str(local_model_dir if local_model_dir.exists() else args.model_name)
    dtype = resolve_torch_dtype(args.dtype)
    prompt = f"{args.voice}: {text}" if args.voice else text

    model = OrpheusModel(
        model_name=model_source,
        tokenizer=model_source,
        dtype=dtype,
        max_model_len=args.max_model_len,
    )

    start = time.monotonic()
    chunks = model.generate_speech(
        prompt=prompt,
        voice=None,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )

    total_frames = 0
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

        for chunk in chunks:
            frame_count = len(chunk) // (wav_file.getsampwidth() * wav_file.getnchannels())
            total_frames += frame_count
            wav_file.writeframes(chunk)

    record_run(
        model_id="orpheus_tts_3b",
        output_path=output_path,
        input_path=args.input,
        reference_audio=None,
        manifest_path=args.manifest,
        extra={
            "voice": args.voice,
            "duration_seconds": round(total_frames / 24000.0, 2),
            "generation_seconds": round(time.monotonic() - start, 2),
        },
    )


if __name__ == "__main__":
    main()