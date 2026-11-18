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
from transformers import AutoProcessor, DiaForConditionalGeneration

MODE_OUTPUTS = {
    "clone": "77_dia_1_6b_clone.wav",
    "dialogue": "33_dia_1_6b_dialogue.wav",
}
TARGET_SAMPLE_RATE = 44_100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dia 1.6B for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--generation-mode", choices=sorted(MODE_OUTPUTS), default="dialogue")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--model-name", default="nari-labs/Dia-1.6B-0626")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=45)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_name(model_name: str) -> str:
    local_dir = resolve_model_root() / "dia" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def ensure_dialogue_tags(text: str) -> str:
    stripped = " ".join(text.split())
    if "[S1]" in stripped or "[S2]" in stripped:
        return stripped
    return f"[S1] {stripped}"


def load_prompt_audio(path: Path) -> list[float]:
    audio, sample_rate = sf.read(path, always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = torch.as_tensor(audio, dtype=torch.float32)
    if audio.ndim > 1:
        audio = audio.mean(dim=1)
    audio = audio.unsqueeze(0)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, TARGET_SAMPLE_RATE)
    return audio.squeeze(0).to(dtype=torch.float32).cpu().numpy()


def build_text(args: argparse.Namespace, text: str) -> tuple[str, Path | None]:
    target_text = ensure_dialogue_tags(text)
    if args.generation_mode != "clone":
        return target_text, None

    reference_text = ensure_dialogue_tags(
        resolve_reference_text(
            args.reference_audio,
            args.reference_text,
            args.reference_text_file,
        )
    )
    combined = f"{reference_text} {target_text}"
    return combined, args.reference_audio


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)
    model_name = resolve_model_name(args.model_name)
    prompt_text, reference_audio = build_text(args, text)

    processor = AutoProcessor.from_pretrained(model_name)
    model = DiaForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
    ).to(device)

    processor_kwargs = {
        "text": [prompt_text],
        "padding": True,
        "return_tensors": "pt",
    }
    prompt_len = None
    if args.generation_mode == "clone":
        processor_kwargs["audio"] = load_prompt_audio(args.reference_audio)

    inputs = processor(**processor_kwargs).to(device)
    if args.generation_mode == "clone":
        prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    if prompt_len is None:
        decoded = processor.batch_decode(outputs)
    else:
        decoded = processor.batch_decode(outputs, audio_prompt_len=prompt_len)
    processor.save_audio(decoded, str(output_path))

    record_run(
        model_id=f"dia_1_6b_{args.generation_mode}",
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
            "guidance_scale": args.guidance_scale,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
    )


if __name__ == "__main__":
    main()
