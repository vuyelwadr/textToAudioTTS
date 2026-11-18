from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_device,
    resolve_external_root,
    resolve_output_dir,
)

EXTERNAL_ROOT = resolve_external_root()
MELO_ROOT = EXTERNAL_ROOT / "MeloTTS"
OPENVOICE_ROOT = EXTERNAL_ROOT / "OpenVoice"

for repo_root in (MELO_ROOT, OPENVOICE_ROOT):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from melo.api import TTS as MeloTTS  # noqa: E402
from openvoice import se_extractor  # noqa: E402
from openvoice.api import ToneColorConverter  # noqa: E402


def prepare_reference_audio(reference_audio: Path, temp_root: Path, minimum_seconds: float = 6.0) -> Path:
    samples, sample_rate = sf.read(reference_audio, always_2d=True, dtype="float32")
    if sample_rate <= 0:
        return reference_audio

    duration_seconds = samples.shape[0] / float(sample_rate)
    if duration_seconds >= minimum_seconds:
        return reference_audio

    repeat_count = max(2, int(np.ceil(minimum_seconds / duration_seconds)))
    repeated = np.vstack([samples] * repeat_count)
    prepared_path = temp_root / f"prepared_{reference_audio.stem}.wav"
    sf.write(prepared_path, repeated, sample_rate, subtype="PCM_16")
    return prepared_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVoice V2 with MeloTTS base speakers.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--language", default="EN")
    parser.add_argument("--source-speaker", default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--watermark-message", default="@MyShell")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_source_speaker(model: MeloTTS, requested: str | None) -> tuple[str, int]:
    speaker_ids = model.hps.data.spk2id
    if requested is None:
        speaker_keys = sorted(str(key) for key in speaker_ids.keys())
        if not speaker_keys:
            raise KeyError("no speakers found in MeloTTS speaker map")
        speaker_key = speaker_keys[0]
        return speaker_key, speaker_ids[speaker_key]

    for speaker_key, speaker_id in speaker_ids.items():
        normalized = speaker_key.lower().replace("_", "-")
        if requested.lower() in {speaker_key.lower(), normalized}:
            return speaker_key, speaker_id
    raise KeyError(f"speaker {requested!r} not found; available speakers: {', '.join(sorted(speaker_ids))}")


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / "26_openvoice_v2_en.wav")
    ensure_parent(output_path)

    device = resolve_device(args.device)
    checkpoint_root = args.checkpoint_root or (OPENVOICE_ROOT / "checkpoints_v2")
    tone_color_converter = ToneColorConverter(str(checkpoint_root / "converter" / "config.json"), device=device)
    tone_color_converter.load_ckpt(str(checkpoint_root / "converter" / "checkpoint.pth"))

    base_tts = MeloTTS(language=args.language, device=device)
    speaker_key, speaker_id = resolve_source_speaker(base_tts, args.source_speaker)
    embedding_name = speaker_key.lower().replace("_", "-")
    source_embedding = checkpoint_root / "base_speakers" / "ses" / f"{embedding_name}.pth"
    source_se = torch.load(source_embedding, map_location=device).to(device)

    with tempfile.TemporaryDirectory(prefix="openvoice_v2_") as tmpdir:
        prepared_reference_audio = prepare_reference_audio(args.reference_audio, Path(tmpdir))
        target_se, _ = se_extractor.get_se(
            str(prepared_reference_audio),
            tone_color_converter,
            target_dir=str(output_path.parent / "_openvoice_processed"),
            vad=not args.no_vad,
        )
        source_audio = Path(tmpdir) / "melo_source.wav"
        base_tts.tts_to_file(text, speaker_id, str(source_audio), speed=args.speed)
        tone_color_converter.convert(
            audio_src_path=str(source_audio),
            src_se=source_se,
            tgt_se=target_se,
            output_path=str(output_path),
            message=args.watermark_message,
        )

    record_run(
        model_id="openvoice_v2",
        output_path=output_path,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={
            "language": args.language,
            "speaker": speaker_key,
            "speed": args.speed,
        },
    )


if __name__ == "__main__":
    main()