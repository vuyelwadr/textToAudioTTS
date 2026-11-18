"""Run Chatterbox-Turbo for the shared naturalness prompt.

Model   : ResembleAI/chatterbox-turbo  (Apache-2.0)
Package : chatterbox-tts  (pip)
GitHub  : https://github.com/resemble-ai/chatterbox

Architecture overview
---------------------
Chatterbox-Turbo is a 350M-parameter English TTS model consisting of:
  • T3  – GPT-2-based text-to-speech-token backbone
  • S3Gen – distilled 1-step mel spectrogram decoder (from 10 steps → 1)
It supports paralinguistic expression tags in text (e.g. [laugh], [chuckle])
and zero-shot voice cloning via a short reference audio clip (~10 s).

Inference path
--------------
1.  bootstrap_chatterbox_turbo.sh installs chatterbox-tts and pre-downloads
    ResembleAI/chatterbox-turbo into the HF hub cache at
    $MAHTI_CACHE_ROOT/huggingface/hub.
2.  ChatterboxTurboTTS.from_pretrained(device="cuda") resolves the weights
    from that cache (offline-safe on compute nodes).
3.  model.generate(text [, audio_prompt_path=...]) returns a (1, N) CPU tensor.
4.  Output is saved with torchaudio at 24 000 Hz (model.sr).

Modes
-----
  plain  (80_chatterbox_turbo_plain.wav)  – no reference audio
  clone  (81_chatterbox_turbo_clone.wav)  – zero-shot voice clone

Recommended reference audio source (local, for pre-production testing):
  /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/outputs/
      video_downloads/haCxgJSFmFc/haCxgJSFmFc_reference_15s.wav

GPU: requires CUDA with ≥4 GB VRAM (A100 achieves RTF < 0.01 for 30 s output).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
    resolve_output_dir,
)

DEFAULT_MODEL_REPO = "ResembleAI/chatterbox-turbo"
DEFAULT_OUTPUT_PLAIN = "80_chatterbox_turbo_plain.wav"
DEFAULT_OUTPUT_CLONE = "81_chatterbox_turbo_clone.wav"
USER_REFERENCE_AUDIO = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "video_downloads"
    / "haCxgJSFmFc"
    / "haCxgJSFmFc_reference_8s.wav"
)
DEFAULT_CHATTERBOX_REFERENCE_AUDIO = (
    USER_REFERENCE_AUDIO if USER_REFERENCE_AUDIO.exists() else DEFAULT_REFERENCE_AUDIO
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chatterbox-Turbo for the shared naturalness prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--mode",
        choices=["plain", "clone"],
        default="plain",
        help=(
            "plain: generate without a speaker reference; "
            "clone: zero-shot voice clone from --reference-audio."
        ),
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=DEFAULT_CHATTERBOX_REFERENCE_AUDIO,
        help="Speaker reference WAV (used in clone mode only; ~10 s recommended).",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help=(
            "HuggingFace repo ID or local path passed to "
            "ChatterboxTurboTTS.from_pretrained().  "
            "Bootstrap pre-caches ResembleAI/chatterbox-turbo."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    text = normalize_target_text(load_text(args.input))

    output_dir = resolve_output_dir(args.output_dir)
    default_fname = (
        DEFAULT_OUTPUT_CLONE if args.mode == "clone" else DEFAULT_OUTPUT_PLAIN
    )
    output_path = args.output or (output_dir / default_fname)
    ensure_parent(output_path)

    # from_pretrained resolves weights from the HF hub cache populated by
    # bootstrap_chatterbox_turbo.sh; no network access required on compute nodes.
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    ref_path: Path | None = None
    if args.mode == "clone":
        ref_path = args.reference_audio
        wav = model.generate(text, audio_prompt_path=str(ref_path))
    else:
        wav = model.generate(text)

    # model.sr == 24000 (Hz); use soundfile to avoid torchaudio's torchcodec path.
    sf.write(str(output_path), wav.detach().cpu().squeeze(0).numpy(), model.sr)

    record_run(
        model_id=f"chatterbox_turbo_{args.mode}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=ref_path,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "model_repo": args.model_repo,
            "mode": args.mode,
            "sample_rate": model.sr,
        },
    )


if __name__ == "__main__":
    main()
