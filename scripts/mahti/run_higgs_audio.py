from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_attn_implementation,
    resolve_cluster_name,
    resolve_device,
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
    resolve_torch_dtype,
    safe_model_dirname,
)

DEFAULT_MODEL_NAME = "bosonai/higgs-audio-v2-generation-3B-base"
SYSTEM_PROMPT = "Generate audio following instruction."
SCENE_DESCRIPTION = "Audio is recorded from a quiet room."

MODE_OUTPUTS = {
    "plain": "36_higgs_audio_v2_3b_plain.wav",
    "clone": "37_higgs_audio_v2_3b_clone.wav",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Higgs Audio v2 for the shared naturalness prompt."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--generation-mode", choices=sorted(MODE_OUTPUTS), default="plain"
    )
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_name(model_name: str) -> str:
    local_dir = resolve_model_root() / "higgs-audio" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def build_conversation(
    args: argparse.Namespace,
    text: str,
) -> tuple[list[dict], Path | None]:
    conversation: list[dict] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "scene",
            "content": [{"type": "text", "text": SCENE_DESCRIPTION}],
        },
    ]

    if args.generation_mode == "plain":
        conversation.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        )
        return conversation, None

    reference_text = resolve_reference_text(
        args.reference_audio,
        args.reference_text,
        args.reference_text_file,
    )
    conversation.extend(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": reference_text}],
            },
            {
                "role": "assistant",
                "content": [{"type": "audio", "url": str(args.reference_audio.resolve())}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        ]
    )
    return conversation, args.reference_audio


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)
    attn_implementation = resolve_attn_implementation()
    model_name = resolve_model_name(args.model_name)

    processor = AutoProcessor.from_pretrained(model_name)
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()

    conversation, reference_audio = build_conversation(args, text)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        sampling_rate=24_000,
        return_tensors="pt",
    ).to(device)

    generate_kwargs: dict[str, object] = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    decoded = processor.batch_decode(outputs.cpu())
    processor.save_audio(decoded, str(output_path))

    record_run(
        model_id=f"higgs_audio_v2_3b_{args.generation_mode}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=reference_audio,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "dtype": str(dtype),
            "attn_implementation": attn_implementation,
            "model_name": args.model_name,
            "mode": args.generation_mode,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    )


if __name__ == "__main__":
    main()
