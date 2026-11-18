from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio

# As of torchaudio 2.9 (the module-provided build this repo pins via
# `load_pytorch_module 2.9`), torchaudio.load() unconditionally delegates to
# TorchCodec's AudioDecoder -- the `backend=` argument is accepted only for
# backwards compatibility and is silently ignored. TorchCodec's default PyPI
# wheel links against CUDA 13 (libnvrtc.so.13), which this cluster's
# module-provided PyTorch build (CUDA 12.9) doesn't ship, so importing it
# fails at first use -- the same CUDA-version-mismatch class of problem
# already documented for onnxruntime-gpu in bootstrap_cosyvoice3.sh.
# cosyvoice/utils/file_utils.py's load_wav() (the only torchaudio.load()
# call site on the zero-shot inference path) only needs a plain WAV decode,
# so replace torchaudio.load with a `soundfile`-backed implementation here
# (soundfile has no CUDA/FFmpeg dependency) instead of pulling in TorchCodec.
def _load_wav_via_soundfile(
    uri,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: str | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
):
    import soundfile as sf

    data, sample_rate = sf.read(str(uri), dtype="float32", always_2d=True)
    if num_frames is not None and num_frames != -1:
        data = data[frame_offset : frame_offset + num_frames]
    elif frame_offset:
        data = data[frame_offset:]
    tensor = torch.from_numpy(data)
    if channels_first:
        tensor = tensor.transpose(0, 1).contiguous()
    return tensor, sample_rate


# torchaudio.save() has the same TorchCodec-only redirection in 2.9 (routing
# through TorchCodec's AudioEncoder), hitting the identical CUDA-13/libnvrtc
# failure -- this script's own final `torchaudio.save(...)` call is the only
# save() call site on the zero-shot inference path, so replace it the same
# way with a plain `soundfile`-backed writer.
def _save_wav_via_soundfile(
    uri,
    src,
    sample_rate: int,
    channels_first: bool = True,
    format: str | None = None,
    encoding: str | None = None,
    bits_per_sample: int | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
    compression=None,
):
    import soundfile as sf

    data = src.detach().cpu().numpy()
    if channels_first:
        data = data.T  # [channels, samples] -> [samples, channels]
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    sf.write(str(uri), data, sample_rate)


torchaudio.load = _load_wav_via_soundfile
torchaudio.save = _save_wav_via_soundfile

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
    resolve_reference_text,
    safe_model_dirname,
)

DEFAULT_MODEL_NAME = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_OUTPUT = "43_cosyvoice3_zero_shot_clone.wav"
# CosyVoice3's zero-shot prompt_text expects a leading system-style prompt
# terminated by <|endofprompt|> before the actual reference transcript --
# see cosyvoice3_example() in the upstream repo's example.py.
DEFAULT_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"

COSYVOICE_ROOT = resolve_external_root() / "CosyVoice"
MATCHA_ROOT = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"
for _extra_path in (MATCHA_ROOT, COSYVOICE_ROOT):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CosyVoice3 zero-shot voice cloning for the shared naturalness prompt.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--text-frontend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable wetext-based text normalization for the target text",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_dir(model_name: str) -> str:
    local_dir = resolve_model_root() / "cosyvoice3" / safe_model_dirname(model_name)
    return str(local_dir) if local_dir.exists() else model_name


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    reference_text = resolve_reference_text(
        args.reference_audio,
        args.reference_text,
        args.reference_text_file,
    )
    prompt_text = f"{args.prompt_prefix}{reference_text}"

    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / DEFAULT_OUTPUT)
    ensure_parent(output_path)

    device = resolve_device(args.device)
    if device != "cuda":
        raise RuntimeError("CosyVoice3 smoke test expects a CUDA device")

    model_dir = resolve_model_dir(args.model_name)
    cosyvoice = AutoModel(model_dir=model_dir, fp16=args.fp16)

    start = time.perf_counter()
    chunks = []
    for _, result in enumerate(
        cosyvoice.inference_zero_shot(
            text,
            prompt_text,
            str(args.reference_audio),
            stream=False,
            speed=args.speed,
            text_frontend=args.text_frontend,
        )
    ):
        chunks.append(result["tts_speech"].cpu())
    elapsed = time.perf_counter() - start

    if not chunks:
        raise RuntimeError("CosyVoice3 produced no audio for the given input")
    wav = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]

    # CosyVoice yields torch tensors shaped [channels, samples] (matching
    # torchaudio.save's expected layout directly), unlike some other models
    # in this repo that return numpy arrays consumed via soundfile.write.
    torchaudio.save(str(output_path), wav, cosyvoice.sample_rate)

    audio_seconds = wav.shape[-1] / cosyvoice.sample_rate
    rtf = elapsed / audio_seconds if audio_seconds else float("nan")

    record_run(
        model_id="cosyvoice3_zero_shot_clone",
        output_path=output_path,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "model_name": model_dir,
            "speed": args.speed,
            "fp16": args.fp16,
            "text_frontend": args.text_frontend,
            "generation_seconds": elapsed,
            "audio_seconds": audio_seconds,
            "rtf": rtf,
        },
    )
    print(f"wrote {output_path} ({audio_seconds:.2f}s audio in {elapsed:.2f}s, rtf={rtf:.3f})")


if __name__ == "__main__":
    main()
