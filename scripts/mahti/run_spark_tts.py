from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
    resolve_device,
    resolve_external_root,
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
    safe_model_dirname,
)

DEFAULT_MODEL_NAME = "SparkAudio/Spark-TTS-0.5B"
DEFAULT_OUTPUT = "40_spark_tts_0_5b_clone.wav"

SPARK_ROOT = resolve_external_root() / "Spark-TTS"
if str(SPARK_ROOT) not in sys.path:
    sys.path.insert(0, str(SPARK_ROOT))

from cli.SparkTTS import SparkTTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spark-TTS for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_name(model_name: str) -> Path | str:
    local_dir = resolve_model_root() / "spark-tts" / safe_model_dirname(model_name)
    return local_dir if local_dir.exists() else model_name


def resolve_torch_device(device: str | None) -> torch.device:
    resolved = resolve_device(device)
    if resolved == "cuda":
        return torch.device("cuda:0")
    return torch.device(resolved)


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    reference_text = resolve_reference_text(
        args.reference_audio,
        args.reference_text,
        args.reference_text_file,
    )
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / DEFAULT_OUTPUT)
    ensure_parent(output_path)

    model_name = resolve_model_name(args.model_name)
    device = resolve_torch_device(args.device)
    spark = SparkTTS(Path(model_name), device)
    wav = spark.inference(
        text=text,
        prompt_speech_path=str(args.reference_audio),
        prompt_text=reference_text,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    sf.write(str(output_path), wav, samplerate=16000)

    record_run(
        model_id="spark_tts_0_5b_clone",
        output_path=output_path,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": str(device),
            "model_name": str(model_name),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
    )


if __name__ == "__main__":
    main()
