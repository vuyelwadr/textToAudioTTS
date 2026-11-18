from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
    resolve_device,
    resolve_external_root,
    resolve_output_dir,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kyutai TTS for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--voice-repo", default=None)
    parser.add_argument("--voice", default="expresso/ex03-ex01_happy_001_channel1_334s.wav")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / "30_kyutai_tts_en.wav")
    ensure_parent(output_path)

    kyutai_root = resolve_external_root() / "delayed-streams-modeling"
    script_path = kyutai_root / "scripts" / "tts_pytorch.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Kyutai script not found: {script_path}")

    device = resolve_device(args.device)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    with tempfile.TemporaryDirectory(prefix="kyutai_tts_") as tmpdir:
        input_file = Path(tmpdir) / "prompt.txt"
        write_text(input_file, text + "\n")

        command = [
            sys.executable,
            str(script_path),
            str(input_file),
            str(output_path),
            "--device",
            device,
        ]
        if args.hf_repo:
            command.extend(["--hf-repo", args.hf_repo])
        if args.voice_repo:
            command.extend(["--voice-repo", args.voice_repo])
        if args.voice:
            command.extend(["--voice", args.voice])

        subprocess.run(command, cwd=str(kyutai_root), env=env, check=True)

    record_run(
        model_id="kyutai_tts",
        output_path=output_path,
        input_path=args.input,
        reference_audio=None,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "hf_repo": args.hf_repo or "default",
            "voice": args.voice,
            "voice_repo": args.voice_repo or "default",
        },
    )


if __name__ == "__main__":
    main()