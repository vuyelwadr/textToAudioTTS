from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

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
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
    resolve_torch_dtype,
    safe_model_dirname,
)

from transformers import AutoProcessor, CsmForConditionalGeneration

MODE_OUTPUTS = {
    "clone": "76_csm_1b_context_clone.wav",
    "plain": "32_csm_1b_plain.wav",
}
TARGET_SAMPLE_RATE = 24_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sesame CSM-1B for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--generation-mode", choices=sorted(MODE_OUTPUTS), default="plain")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--speaker-id", default="0")
    parser.add_argument("--model-name", default="sesame/csm-1b")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--depth-decoder-temperature", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--max-length", type=int, default=250)
    parser.add_argument("--static-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_name(model_name: str) -> str:
    local_dir = resolve_model_root() / "csm" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def load_reference_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = torch.as_tensor(audio, dtype=torch.float32)
    if audio.ndim > 1:
        audio = audio.mean(dim=1)
    audio = audio.unsqueeze(0)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, TARGET_SAMPLE_RATE)
    return audio.squeeze(0).cpu().numpy()


def build_conversation(args: argparse.Namespace, text: str) -> tuple[list[dict[str, object]], Path | None]:
    if args.generation_mode == "plain":
        return (
            [
                {
                    "role": args.speaker_id,
                    "content": [{"type": "text", "text": text}],
                }
            ],
            None,
        )

    reference_text = resolve_reference_text(
        args.reference_audio,
        args.reference_text,
        args.reference_text_file,
    )
    reference_audio = load_reference_audio(args.reference_audio)
    return (
        [
            {
                "role": args.speaker_id,
                "content": [
                    {"type": "text", "text": reference_text},
                    {"type": "audio", "path": reference_audio},
                ],
            },
            {
                "role": args.speaker_id,
                "content": [{"type": "text", "text": text}],
            },
        ],
        args.reference_audio,
    )


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)
    model_name = resolve_model_name(args.model_name)

    processor = AutoProcessor.from_pretrained(model_name)
    model = CsmForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
    ).to(device)
    if args.max_length is not None:
        model.generation_config.max_length = args.max_length
    if args.static_cache:
        model.generation_config.cache_implementation = "static"
        if hasattr(model, "depth_decoder") and hasattr(model.depth_decoder, "generation_config"):
            model.depth_decoder.generation_config.cache_implementation = "static"

    conversation, reference_audio = build_conversation(args, text)
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
    ).to(model.device)
    if "input_values" in inputs and dtype is not None:
        inputs["input_values"] = inputs["input_values"].to(dtype=dtype)

    audio = model.generate(
        **inputs,
        output_audio=True,
        do_sample=args.do_sample,
        depth_decoder_do_sample=args.do_sample,
        temperature=args.temperature,
        depth_decoder_temperature=args.depth_decoder_temperature,
    )
    processor.save_audio(audio, str(output_path))

    record_run(
        model_id=f"csm_1b_{args.generation_mode}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=reference_audio,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "dtype": str(dtype),
            "mode": args.generation_mode,
            "model_name": model_name,
            "speaker_id": args.speaker_id,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "depth_decoder_temperature": args.depth_decoder_temperature,
        },
    )


if __name__ == "__main__":
    main()
