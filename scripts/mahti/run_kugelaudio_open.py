"""Run kugelaudio/kugelaudio-0-open for the shared naturalness prompt.

Model:  kugelaudio/kugelaudio-0-open  (7B, hybrid AR+Diffusion)
Repo:   https://github.com/Kugelaudio/kugelaudio-open
Paper:  –  (trained on ~200k hours YODAS2 for 24 European languages)
VRAM:   ~19 GB bfloat16; use --no-strip-encoders only if VRAM > 25 GB.

Inference path
--------------
1. Load KugelAudioForConditionalGenerationInference.from_pretrained()
2. Call model.model.strip_encoders() to free encoder VRAM after loading
3. Load KugelAudioProcessor.from_pretrained()
4. processor(text=..., voice=...) → input dict
5. model.generate(inputs, cfg_scale=...) → ModelOutput with .speech_outputs
6. processor.save_audio(speech_outputs, path)

Usage examples
--------------
python scripts/mahti/run_kugelaudio_open.py
python scripts/mahti/run_kugelaudio_open.py --voice warm --cfg-scale 4.0
python scripts/mahti/run_kugelaudio_open.py --output-dir /scratch/project/tts/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
    resolve_model_root,
    resolve_output_dir,
    safe_model_dirname,
)

DEFAULT_MODEL_NAME = "kugelaudio/kugelaudio-0-open"
DEFAULT_VOICE = "default"
DEFAULT_CFG_SCALE = 3.0
DEFAULT_OUTPUT = "56_kugelaudio_0_open_default.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KugelAudio-0-Open for the shared naturalness prompt."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help='Pre-encoded voice name (e.g. "default", "warm", "clear"). '
        "Run --list-voices to see all available voices.",
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="Print available voices and exit."
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=DEFAULT_CFG_SCALE,
        help="Classifier-free guidance scale (default: 3.0).",
    )
    parser.add_argument(
        "--no-strip-encoders",
        action="store_true",
        help="Keep encoder weights loaded (uses ~6 GB extra VRAM; not normally needed).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device override (auto-detected if omitted).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_path(model_name: str) -> str:
    """Return local path when model weights have been pre-downloaded, else HF ID."""
    local_dir = resolve_model_root() / "kugelaudio-open" / safe_model_dirname(model_name)
    return str(local_dir) if local_dir.exists() else model_name


def resolve_torch_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_inputs(
    processor,
    text: str,
    voice: str | None,
):
    if voice is None:
        return processor(text=text, return_tensors="pt"), None

    try:
        return processor(text=text, voice=voice, return_tensors="pt"), voice
    except ValueError as exc:
        if "Could not find voice file" not in str(exc):
            raise
        print(
            f"[warn] preset voice '{voice}' is unavailable for this model snapshot; "
            "falling back to unconditioned generation",
            file=sys.stderr,
        )
        return processor(text=text, return_tensors="pt"), None


def move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


def main() -> None:
    args = parse_args()

    from kugelaudio_open import (  # noqa: PLC0415 – import after bootstrap activates venv
        KugelAudioForConditionalGenerationInference,
        KugelAudioProcessor,
    )

    model_path = resolve_model_path(args.model_name)
    device = resolve_torch_device(args.device)

    # ── Load processor first (lightweight) ───────────────────────────────────
    processor = KugelAudioProcessor.from_pretrained(model_path)

    if args.list_voices:
        print("Available voices:", processor.get_available_voices())
        return

    # ── Load model ────────────────────────────────────────────────────────────
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Strip encoder weights to recover ~6 GB VRAM; they are not used for inference.
    if not args.no_strip_encoders:
        model.model.strip_encoders()

    # ── Prepare text input ────────────────────────────────────────────────────
    text = normalize_target_text(load_text(args.input))

    output_file_stem = f"56_kugelaudio_0_open_{args.voice}"
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / f"{output_file_stem}.wav")
    ensure_parent(output_path)

    # ── Build model inputs ────────────────────────────────────────────────────
    inputs, resolved_voice = build_inputs(processor, text, args.voice)
    inputs = move_to_device(dict(inputs), device)

    # ── Generate ──────────────────────────────────────────────────────────────
    generate_kwargs = {
        "text_ids": inputs["text_ids"],
        "speech_input_mask": inputs.get("speech_input_mask"),
        "cfg_scale": args.cfg_scale,
    }
    if inputs.get("voice_cache") is not None:
        generate_kwargs["voice_cache"] = inputs["voice_cache"]
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    # ── Save audio (processor handles WAV encoding) ───────────────────────────
    processor.save_audio(outputs.speech_outputs, str(output_path))

    record_run(
        model_id="kugelaudio_0_open",
        output_path=output_path,
        input_path=args.input,
        reference_audio=None,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": str(device),
            "model_name": args.model_name,
            "voice": resolved_voice,
            "requested_voice": args.voice,
            "cfg_scale": args.cfg_scale,
            "strip_encoders": not args.no_strip_encoders,
        },
    )


if __name__ == "__main__":
    main()
