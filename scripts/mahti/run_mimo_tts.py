from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    safe_model_dirname,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MIMO_REPO_ROOT = resolve_external_root() / "mimo-audio"
if str(MIMO_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMO_REPO_ROOT))

from src.mimo_audio.mimo_audio import MimoAudio  # noqa: E402

MODEL_ID = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
TOKENIZER_ID = "XiaomiMiMo/MiMo-Audio-Tokenizer"

MODE_OUTPUTS = {
    "clone": "56_mimo_audio_7b_instruct_clone.wav",
    "plain": "56_mimo_audio_7b_instruct_plain.wav",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MiMo-Audio-7B-Instruct for plain or prompt-speech-clone TTS."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--generation-mode", choices=sorted(MODE_OUTPUTS), default="clone")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--model-name", default=MODEL_ID)
    parser.add_argument("--tokenizer-name", default=TOKENIZER_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--instruct", default=None)
    return parser.parse_args()


def _resolve_hf_path(model_id: str) -> str:
    local = resolve_model_root() / "mimo-audio" / safe_model_dirname(model_id)
    return str(local) if local.exists() else model_id


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    device = resolve_device(args.device)
    model_path = _resolve_hf_path(args.model_name)
    tokenizer_path = _resolve_hf_path(args.tokenizer_name)

    model = MimoAudio(model_path, tokenizer_path, device=device)

    prompt_speech = None
    reference_audio_recorded: Path | None = None
    if args.generation_mode == "clone":
        if not args.reference_audio.exists():
            raise FileNotFoundError(f"reference audio not found: {args.reference_audio}")
        prompt_speech = str(args.reference_audio)
        reference_audio_recorded = args.reference_audio

    model.tts_sft(
        text,
        str(output_path),
        instruct=args.instruct,
        prompt_speech=prompt_speech,
    )

    record_run(
        model_id=f"mimo_audio_7b_instruct_{args.generation_mode}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=reference_audio_recorded,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "mode": args.generation_mode,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "instruct": args.instruct,
            "sample_rate": 24000,
        },
    )


if __name__ == "__main__":
    main()
