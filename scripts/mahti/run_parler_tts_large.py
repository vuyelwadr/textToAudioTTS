from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_attn_implementation,
    resolve_cluster_name,
    resolve_device,
    resolve_model_root,
    resolve_output_dir,
    resolve_torch_dtype,
    safe_model_dirname,
)

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

DEFAULT_MODEL_NAME = "parler-tts/parler-tts-large-v1"
DEFAULT_OUTPUT = "54_parler_tts_large_v1_plain.wav"
DEFAULT_SPEAKER = "Jon"
DEFAULT_DESCRIPTION_TEMPLATE = (
    "{speaker}'s voice is warm, steady, and close-mic with restrained expressiveness, "
    "natural long-form audiobook pacing, and very clear audio."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Parler-TTS Large v1 for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--description", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_name(model_name: str) -> str:
    local_dir = resolve_model_root() / "parler-tts" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def resolve_description(args: argparse.Namespace) -> str:
    if args.description:
        return args.description.strip()
    return DEFAULT_DESCRIPTION_TEMPLATE.format(speaker=args.speaker)


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / DEFAULT_OUTPUT)
    ensure_parent(output_path)

    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)
    model_name = resolve_model_name(args.model_name)
    description = resolve_description(args)

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation or "eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    description_inputs = tokenizer(description, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(text, return_tensors="pt").to(device)
    generation = model.generate(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
    )
    audio = generation.detach().cpu().float().numpy().squeeze()
    sf.write(str(output_path), audio, model.config.sampling_rate)

    record_run(
        model_id="parler_tts_large_v1",
        output_path=output_path,
        input_path=args.input,
        reference_audio=None,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "dtype": str(dtype),
            "description": description,
            "speaker": args.speaker,
            "model_name": model_name,
        },
    )


if __name__ == "__main__":
    main()
