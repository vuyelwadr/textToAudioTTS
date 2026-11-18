from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_device,
    resolve_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XTTS v2 voice cloning for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / "24_xtts_v2.wav")
    ensure_parent(output_path)

    device = resolve_device(args.device)
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    if hasattr(tts, "to"):
        tts = tts.to(device)

    tts.tts_to_file(
        text=text,
        speaker_wav=str(args.reference_audio),
        language=args.language,
        file_path=str(output_path),
    )

    record_run(
        model_id="xtts_v2",
        output_path=output_path,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={"language": args.language, "device": device},
    )


if __name__ == "__main__":
    main()
