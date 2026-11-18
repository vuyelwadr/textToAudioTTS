from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
    safe_model_dirname,
)

from voxcpm import VoxCPM

MODE_OUTPUTS = {
    "tts": "72_voxcpm2_tts.wav",
    "design": "73_voxcpm2_voice_design.wav",
    "clone": "74_voxcpm2_clone.wav",
    "ultimate": "75_voxcpm2_ultimate_clone.wav",
}

DEFAULT_DESIGN_PROMPT = "Warm, steady audiobook narrator with clear diction and restrained expressiveness."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VoxCPM2 for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--generation-mode", choices=sorted(MODE_OUTPUTS), default="tts")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--model-name", default="openbmb/VoxCPM2")
    parser.add_argument("--device", default="auto", help="Device for VoxCPM2: auto, cpu, mps, cuda, cuda:0")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--voice-description", default=DEFAULT_DESIGN_PROMPT)
    parser.add_argument("--load-denoiser", action="store_true")
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_name(model_name: str) -> str:
    local_dir = resolve_model_root() / "voxcpm" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def prepend_design_prompt(text: str, voice_description: str) -> str:
    description = voice_description.strip()
    if not description:
        raise ValueError("voice description is empty")
    return f"({description}){text}"


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    model_name = resolve_model_name(args.model_name)
    use_local_files = Path(model_name).exists()
    model = VoxCPM.from_pretrained(
        model_name,
        optimize=args.optimize,
        load_denoiser=args.load_denoiser,
        local_files_only=use_local_files,
    )

    generate_kwargs: dict[str, object] = {
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
    }

    reference_audio: Path | None = None
    if args.generation_mode == "design":
        rendered_text = prepend_design_prompt(text, args.voice_description)
        model_id = "voxcpm2_voice_design"
        extra = {"voice_description": args.voice_description}
    else:
        rendered_text = text
        model_id = "voxcpm2_tts"
        extra = {}

    if args.generation_mode == "clone":
        reference_audio = args.reference_audio
        generate_kwargs["reference_wav_path"] = str(args.reference_audio)
        model_id = "voxcpm2_clone"
    elif args.generation_mode == "ultimate":
        reference_audio = args.reference_audio
        reference_text = resolve_reference_text(
            args.reference_audio,
            args.reference_text,
            args.reference_text_file,
        )
        generate_kwargs["reference_wav_path"] = str(args.reference_audio)
        generate_kwargs["prompt_wav_path"] = str(args.reference_audio)
        generate_kwargs["prompt_text"] = reference_text
        model_id = "voxcpm2_ultimate_clone"
        extra["reference_text_mode"] = (
            "manual" if args.reference_text is not None or args.reference_text_file is not None else "auto_transcribed"
        )

    waveform = model.generate(
        text=rendered_text,
        **generate_kwargs,
    )
    sf.write(str(output_path), waveform, int(model.tts_model.sample_rate))

    extra.update(
        {
            "cluster": resolve_cluster_name(),
            "device": args.device,
            "cfg_value": args.cfg_value,
            "inference_timesteps": args.inference_timesteps,
            "load_denoiser": args.load_denoiser,
            "model_name": args.model_name,
            "mode": args.generation_mode,
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
