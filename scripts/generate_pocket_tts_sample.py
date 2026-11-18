#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pocket_tts import TTSModel
from scipy.io import wavfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Pocket TTS sample for the naturalness comparison")
    parser.add_argument("input", type=Path, help="Input text file")
    parser.add_argument("--voice", default="alba", help="Built-in Pocket TTS voice name or local/HF prompt path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory that will receive the rendered wav")
    parser.add_argument("--output-name", default="pocket_tts.wav", help="Output wav file name")
    return parser.parse_args()


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input text file is empty: {path}")
    return text


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.dtype.kind != "f":
        return audio

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1.0 + 1e-6:
        return np.clip(audio, -1.0, 1.0) * 32767.0
    return audio


def main() -> int:
    args = parse_args()
    text = read_text(args.input)

    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt(args.voice)
    audio = model.generate_audio(voice_state, text)
    audio_np = normalize_audio(audio.detach().cpu().numpy()).astype(np.int16, copy=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    wavfile.write(output_path, model.sample_rate, audio_np)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())