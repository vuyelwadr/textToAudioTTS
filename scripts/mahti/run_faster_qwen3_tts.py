from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import soundfile as sf

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

from faster_qwen3_tts import FasterQwen3TTS

DEFAULT_MODELS = {
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
DEFAULT_OUTPUTS = {
    "clone": "29_faster_qwen3_tts_voice_clone.wav",
    "custom": "29_faster_qwen3_tts_custom_voice.wav",
    "design": "29_faster_qwen3_tts_voice_design.wav",
}
DEFAULT_INSTRUCT = "Warm, steady audiobook narrator with clear diction and restrained expressiveness."
DEFAULT_SPEAKER_CANDIDATES = ("aiden", "Ryan", "Aiden", "Dylan", "Serena")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Faster Qwen3-TTS for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--generation-mode", choices=sorted(DEFAULT_MODELS), default="custom")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic sampling during generation",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_name(mode: str, explicit: str | None) -> str:
    model_name = explicit or DEFAULT_MODELS[mode]
    local_dir = resolve_model_root() / "qwen3-tts" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def list_supported_speakers(model: FasterQwen3TTS) -> list[str]:
    getter = getattr(model.model, "get_supported_speakers", None)
    if callable(getter):
        speakers = getter()
        if speakers:
            return list(speakers)
    return []


def resolve_speaker(speakers: Iterable[str], requested: str | None) -> str:
    speaker_list = list(speakers)
    if not speaker_list:
        raise RuntimeError("Faster Qwen3-TTS custom voice model did not report any speakers")

    if requested:
        requested_lower = requested.lower()
        for speaker in speaker_list:
            if speaker.lower() == requested_lower:
                return speaker
        available = ", ".join(speaker_list)
        raise KeyError(f"speaker {requested!r} not found; available speakers: {available}")

    for candidate in DEFAULT_SPEAKER_CANDIDATES:
        candidate_lower = candidate.lower()
        for speaker in speaker_list:
            if speaker.lower() == candidate_lower:
                return speaker

    return speaker_list[0]


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / DEFAULT_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    device = resolve_device(args.device)
    if device != "cuda":
        raise RuntimeError("Faster Qwen3-TTS requires a CUDA device")

    model_name = resolve_model_name(args.generation_mode, args.model_name)
    dtype = resolve_torch_dtype(args.dtype)
    model = FasterQwen3TTS.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        attn_implementation=resolve_attn_implementation(args.attn_implementation),
        max_seq_len=args.max_seq_len,
    )

    generate_kwargs = {
        "language": args.language,
        "non_streaming_mode": args.non_streaming_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
    }

    reference_audio: Path | None = None
    if args.generation_mode == "design":
        wavs, sample_rate = model.generate_voice_design(
            text=text,
            instruct=args.instruct or DEFAULT_INSTRUCT,
            **generate_kwargs,
        )
        model_id = "faster_qwen3_tts_voice_design"
        extra = {"instruct": args.instruct or DEFAULT_INSTRUCT}
    elif args.generation_mode == "custom":
        speaker = resolve_speaker(list_supported_speakers(model), args.speaker)
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            instruct=args.instruct or None,
            **generate_kwargs,
        )
        model_id = "faster_qwen3_tts_custom_voice"
        extra = {"speaker": speaker, "instruct": args.instruct or None}
    else:
        reference_audio = args.reference_audio
        reference_text = resolve_reference_text(
            args.reference_audio,
            args.reference_text,
            args.reference_text_file,
        )
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            ref_audio=str(args.reference_audio),
            ref_text=reference_text,
            xvec_only=args.xvec_only,
            instruct=args.instruct or None,
            **generate_kwargs,
        )
        model_id = "faster_qwen3_tts_voice_clone"
        extra = {
            "reference_text_mode": "x_vector_only" if args.xvec_only else "icl",
            "instruct": args.instruct or None,
        }

    sf.write(str(output_path), wavs[0], sample_rate)

    extra.update(
        {
            "cluster": resolve_cluster_name(),
            "device": device,
            "dtype": str(dtype),
            "language": args.language,
            "mode": args.generation_mode,
            "model_name": model_name,
        }
    )
    record_run(
        model_id=model_id,
        output_path=output_path,
        input_path=args.input,
        reference_audio=reference_audio,
        manifest_path=args.manifest,
        extra=extra,
    )


if __name__ == "__main__":
    main()
