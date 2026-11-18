from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    discover_first,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_external_root,
    resolve_output_dir,
    resolve_reference_text,
    write_text,
)

LANGUAGE_MAP = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "mix_zh_en": "中英混合",
    "mix_ja_en": "日英混合",
    "multi": "多语种混合",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-SoVITS for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--ref-language", default="en", choices=sorted(LANGUAGE_MAP))
    parser.add_argument("--target-language", default="en", choices=sorted(LANGUAGE_MAP))
    parser.add_argument("--gpt-model", type=Path, default=None)
    parser.add_argument("--sovits-model", type=Path, default=None)
    parser.add_argument("--version", default="v2")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_paths(gpt_root: Path, gpt_model: Path | None, sovits_model: Path | None) -> tuple[Path, Path]:
    if gpt_model is None:
        gpt_model = discover_first(
            gpt_root,
            "gsv-v2final-pretrained/*.ckpt",
            "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "**/s1*.ckpt",
        )
    if sovits_model is None:
        sovits_model = discover_first(
            gpt_root,
            "gsv-v2final-pretrained/*s2G*.pth",
            "s2G488k.pth",
            "**/s2G*.pth",
        )
    if gpt_model is None or sovits_model is None:
        raise FileNotFoundError("Could not discover GPT-SoVITS pretrained checkpoints. Run bootstrap_gpt_sovits.sh first.")
    return Path(gpt_model), Path(sovits_model)


def prepare_reference_audio(reference_audio: Path, temp_root: Path, minimum_seconds: float = 3.1) -> Path:
    samples, sample_rate = sf.read(reference_audio, always_2d=True, dtype="float32")

    if sample_rate <= 0:
        return reference_audio

    duration_seconds = samples.shape[0] / float(sample_rate)
    normalized_path = temp_root / f"normalized_{reference_audio.stem}.wav"

    if duration_seconds < minimum_seconds:
        target_frames = int(sample_rate * minimum_seconds)
        missing_frames = max(target_frames - samples.shape[0], 0)
        if missing_frames:
            silence = np.zeros((missing_frames, samples.shape[1]), dtype=samples.dtype)
            samples = np.vstack([samples, silence])

    sf.write(normalized_path, samples, sample_rate, subtype="PCM_16")
    return normalized_path


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    reference_text = resolve_reference_text(args.reference_audio, args.reference_text, args.reference_text_file)
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / "25_gpt_sovits_v2.wav")
    ensure_parent(output_path)

    gpt_root = resolve_external_root() / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"
    gpt_model_path, sovits_model_path = resolve_model_paths(gpt_root, args.gpt_model, args.sovits_model)

    external_root = resolve_external_root() / "GPT-SoVITS"
    env = os.environ.copy()
    env["version"] = args.version

    with tempfile.TemporaryDirectory(prefix="gpt_sovits_") as tmpdir:
        tmp_root = Path(tmpdir)
        ref_text_file = tmp_root / "reference.txt"
        target_text_file = tmp_root / "target.txt"
        temp_output_dir = tmp_root / "out"
        prepared_reference_audio = prepare_reference_audio(args.reference_audio, tmp_root)
        write_text(ref_text_file, reference_text + "\n")
        write_text(target_text_file, text + "\n")
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "GPT_SoVITS/inference_cli.py",
            "--gpt_model",
            str(gpt_model_path),
            "--sovits_model",
            str(sovits_model_path),
            "--ref_audio",
            str(prepared_reference_audio),
            "--ref_text",
            str(ref_text_file),
            "--ref_language",
            LANGUAGE_MAP[args.ref_language],
            "--target_text",
            str(target_text_file),
            "--target_language",
            LANGUAGE_MAP[args.target_language],
            "--output_path",
            str(temp_output_dir),
        ]
        subprocess.run(command, cwd=str(external_root), env=env, check=True)

        generated = temp_output_dir / "output.wav"
        if not generated.exists():
            raise FileNotFoundError(f"GPT-SoVITS did not create expected output: {generated}")
        shutil.move(str(generated), str(output_path))

    record_run(
        model_id="gpt_sovits_v2",
        output_path=output_path,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={
            "version": args.version,
            "ref_language": args.ref_language,
            "target_language": args.target_language,
        },
    )


if __name__ == "__main__":
    main()