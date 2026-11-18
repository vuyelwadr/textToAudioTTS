from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_external_root,
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
)

DEFAULT_OUTPUT = "90_breeze_tts2_voice_direction.wav"
DEFAULT_INSTRUCTION = "Warm, steady literary audiobook narration; natural phrasing, restrained emotion, clear diction, and unhurried pacing."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Breeze TTS 2 voice cloning/direction through the official infer.py CLI.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-all", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    ref_text = resolve_reference_text(args.reference_audio, args.reference_text, args.reference_text_file)
    output_dir = resolve_output_dir(args.output_dir)
    output = (args.output or output_dir / DEFAULT_OUTPUT).resolve()
    ensure_parent(output)

    repo = resolve_external_root() / "breeze-tts"
    infer = repo / "infer.py"
    model = resolve_model_root() / "breeze-tts2" / "BreezeBlue__Breeze-TTS-2"
    if not infer.exists():
        raise FileNotFoundError(f"Breeze inference repo not found: {infer}")
    if not model.exists():
        raise FileNotFoundError(f"Breeze model checkpoint not found: {model}")

    command = [
        sys.executable,
        str(infer),
        str(model),
        "--ref-audio",
        str(args.reference_audio.resolve()),
        "--ref-text",
        ref_text,
        "--text",
        text,
        "--output",
        str(output),
        "--seed",
        str(args.seed),
    ]
    if args.instruction:
        command += ["--instruction", args.instruction, "--cfg-scale", str(args.cfg_scale)]
    if args.fast_all:
        command.append("--fast-all")

    started = time.perf_counter()
    subprocess.run(command, cwd=repo, check=True)
    elapsed = time.perf_counter() - started
    record_run(
        model_id="breeze_tts2_voice_direction",
        output_path=output,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={"instruction": args.instruction, "cfg_scale": args.cfg_scale, "seed": args.seed, "fast_all": args.fast_all, "elapsed_seconds": elapsed},
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
