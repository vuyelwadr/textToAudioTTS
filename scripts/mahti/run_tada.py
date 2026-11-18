from __future__ import annotations

import argparse
from pathlib import Path

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

# TADA's decoder always emits 24 kHz audio regardless of the reference clip's
# sample rate (confirmed in HumeAI/tada's inference.ipynb).
OUTPUT_SAMPLE_RATE = 24000
DEFAULT_ENCODER_NAME = "HumeAI/tada-codec"
DEFAULT_MODEL_NAME = "HumeAI/tada-1b"
DEFAULT_OUTPUT_NAME = "tada_voice_clone.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TADA zero-shot voice cloning for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME)
    parser.add_argument(
        "--language",
        default=None,
        help="Non-English aligner language code for the encoder, e.g. de/es/fr/ja/pl/ar/ch/it/pt",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--num-transition-steps", type=int, default=5)
    parser.add_argument("--num-extra-steps", type=int, default=0)
    parser.add_argument("--num-flow-matching-steps", type=int, default=None)
    parser.add_argument("--acoustic-cfg-scale", type=float, default=None)
    parser.add_argument("--duration-cfg-scale", type=float, default=None)
    parser.add_argument("--speed-up-factor", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_pretrained_name(repo_id: str) -> str:
    local_dir = resolve_model_root() / "tada" / safe_model_dirname(repo_id)
    return str(local_dir if local_dir.exists() else repo_id)


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / DEFAULT_OUTPUT_NAME)
    ensure_parent(output_path)

    device = resolve_device(args.device)
    if device != "cuda":
        raise RuntimeError("TADA smoke testing here requires a CUDA device")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Imported lazily so --help works even before the venv has hume-tada installed.
    from tada.modules.encoder import Encoder
    from tada.modules.tada import InferenceOptions, TadaForCausalLM

    encoder_name = resolve_pretrained_name(args.encoder_name)
    model_name = resolve_pretrained_name(args.model_name)
    dtype = resolve_torch_dtype(args.dtype)

    encoder_kwargs = {}
    if args.language:
        encoder_kwargs["language"] = args.language
    encoder = Encoder.from_pretrained(encoder_name, **encoder_kwargs).to(device)
    model = TadaForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    reference_text = resolve_reference_text(
        args.reference_audio,
        args.reference_text,
        args.reference_text_file,
    )

    audio, sample_rate = torchaudio.load(str(args.reference_audio))
    audio = audio.mean(dim=0, keepdim=True)  # mono
    audio = audio / audio.abs().max()  # normalize
    audio = audio.to(device)

    prompt = encoder(audio, text=[reference_text], sample_rate=sample_rate)

    inference_option_kwargs = {}
    if args.num_flow_matching_steps is not None:
        inference_option_kwargs["num_flow_matching_steps"] = args.num_flow_matching_steps
    if args.acoustic_cfg_scale is not None:
        inference_option_kwargs["acoustic_cfg_scale"] = args.acoustic_cfg_scale
    if args.duration_cfg_scale is not None:
        inference_option_kwargs["duration_cfg_scale"] = args.duration_cfg_scale
    if args.speed_up_factor is not None:
        inference_option_kwargs["speed_up_factor"] = args.speed_up_factor

    generate_kwargs = {
        "num_transition_steps": args.num_transition_steps,
        "num_extra_steps": args.num_extra_steps,
    }
    if inference_option_kwargs:
        generate_kwargs["inference_options"] = InferenceOptions(**inference_option_kwargs)

    output = model.generate(prompt=prompt, text=text, **generate_kwargs)

    waveform = output.audio[0].detach().to("cpu").float().numpy().squeeze()
    sf.write(str(output_path), waveform, OUTPUT_SAMPLE_RATE)

    record_run(
        model_id="tada_voice_clone",
        output_path=output_path,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "dtype": str(dtype),
            "model_name": model_name,
            "encoder_name": encoder_name,
            "language": args.language,
            "num_transition_steps": args.num_transition_steps,
            "num_extra_steps": args.num_extra_steps,
        },
    )


if __name__ == "__main__":
    main()
