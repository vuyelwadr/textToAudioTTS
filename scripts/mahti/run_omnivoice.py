from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment

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

from omnivoice import OmniVoice

MODE_OUTPUTS = {
    "auto": "27_omnivoice_auto.wav",
    "clone": "27_omnivoice_clone.wav",
    "design": "27_omnivoice_design.wav",
}
DEFAULT_DESIGN_INSTRUCT = "female, young adult, moderate pitch"


def load_reference_audio(reference_audio: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    try:
        waveform_np, prompt_sampling_rate = sf.read(str(reference_audio), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform_np.T.copy())
    except (RuntimeError, OSError):
        segment = AudioSegment.from_file(reference_audio)
        audio_data = np.array(segment.get_array_of_samples()).astype(np.float32) / 32768.0
        if segment.channels == 1:
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_data.reshape(-1, segment.channels).T)
        prompt_sampling_rate = segment.frame_rate

    if prompt_sampling_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, prompt_sampling_rate, sample_rate)
        prompt_sampling_rate = sample_rate

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, int(prompt_sampling_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmniVoice for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--generation-mode", choices=sorted(MODE_OUTPUTS), default="clone")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--model-name", default="k2-fsa/OmniVoice")
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_path(model_name: str) -> str:
    local_dir = resolve_model_root() / "omnivoice" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)
    model = OmniVoice.from_pretrained(
        resolve_model_path(args.model_name),
        device_map=device,
        dtype=dtype,
        attn_implementation=resolve_attn_implementation(args.attn_implementation),
    )

    generate_kwargs: dict[str, object] = {
        "text": text,
        "language": args.language,
        "num_step": args.num_step,
        "guidance_scale": args.guidance_scale,
        "speed": args.speed,
    }
    if args.duration is not None:
        generate_kwargs["duration"] = args.duration

    reference_audio: Path | None = None
    if args.generation_mode == "clone":
        reference_audio = args.reference_audio
        generate_kwargs["ref_audio"] = load_reference_audio(args.reference_audio, int(model.sampling_rate))
        generate_kwargs["ref_text"] = resolve_reference_text(
            args.reference_audio,
            args.reference_text,
            args.reference_text_file,
        )
    elif args.generation_mode == "design":
        generate_kwargs["instruct"] = args.instruct or DEFAULT_DESIGN_INSTRUCT

    generated = model.generate(**generate_kwargs)
    waveform = generated[0].detach().cpu()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    sample_rate = int(getattr(model, "sampling_rate", 24000))
    audio_array = waveform.squeeze(0).numpy() if waveform.size(0) == 1 else waveform.transpose(0, 1).numpy()
    sf.write(str(output_path), audio_array, sample_rate)

    record_run(
        model_id=f"omnivoice_{args.generation_mode}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=reference_audio,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "dtype": str(dtype),
            "language": args.language,
            "mode": args.generation_mode,
            "model_name": args.model_name,
        },
    )


if __name__ == "__main__":
    main()
