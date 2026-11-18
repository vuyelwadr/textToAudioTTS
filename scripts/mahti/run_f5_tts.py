from __future__ import annotations

import argparse
import os
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from f5_tts.api import F5TTS

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    discover_first,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
)


def install_soundfile_torchaudio_load() -> None:
    def _soundfile_load(path: str | os.PathLike[str], *args, **kwargs):
        data, sample_rate = sf.read(Path(path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(data.T.copy())
        return waveform, sample_rate

    torchaudio.load = _soundfile_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F5-TTS for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--ckpt-file", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--remove-silence", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_ckpt_file(explicit: Path | None) -> str:
    if explicit is not None:
        return str(explicit)

    model_root = resolve_model_root() / "F5-TTS"
    discovered = discover_first(
        model_root,
        "F5TTS_v1_Base/model_1250000.safetensors",
        "F5TTS_Base/model_1200000.safetensors",
        "**/model_1250000.safetensors",
        "**/model_1200000.safetensors",
    )
    return str(discovered) if discovered else ""


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / "23_f5_tts_v1_base.wav")
    ensure_parent(output_path)

    if args.reference_text is not None:
        reference_text = args.reference_text.strip()
    else:
        reference_text = resolve_reference_text(args.reference_audio, args.reference_text, args.reference_text_file)

    install_soundfile_torchaudio_load()

    tts = F5TTS(
        model=args.model,
        ckpt_file=resolve_ckpt_file(args.ckpt_file),
        device=args.device,
        hf_cache_dir=os.environ.get("HF_HOME"),
    )
    tts.infer(
        ref_file=str(args.reference_audio),
        ref_text=reference_text,
        gen_text=text,
        nfe_step=args.nfe_step,
        speed=args.speed,
        remove_silence=args.remove_silence,
        file_wave=str(output_path),
    )

    record_run(
        model_id="f5_tts_v1_base",
        output_path=output_path,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={
            "speed": args.speed,
            "nfe_step": args.nfe_step,
            "reference_text_mode": "auto_transcribed" if args.reference_text is None and args.reference_text_file is None else "manual",
        },
    )


if __name__ == "__main__":
    main()