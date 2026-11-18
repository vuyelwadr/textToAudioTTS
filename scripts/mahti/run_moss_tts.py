from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _patch_torchaudio() -> None:
    """Monkey-patch torchaudio.load and torchaudio.save when torchcodec is absent.

    Newer torchaudio (≥2.6) defaults to torchcodec for both loading and saving,
    which is not installed on Puhti's PyTorch module.  Fall back to soundfile.
    """
    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401 – just probe
        return  # torchcodec is available; no patch needed
    except Exception:
        pass

    import torchaudio as _ta

    def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True, **_kw):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T)  # (channels, time)
        if frame_offset:
            tensor = tensor[:, frame_offset:]
        if num_frames > 0:
            tensor = tensor[:, :num_frames]
        if not normalize:
            # soundfile always normalises floats; scale back to int range if
            # the caller requests raw samples (rarely needed here).
            tensor = tensor * 32768.0
        return tensor, sr

    def _sf_save(uri, src, sample_rate, channels_first=True, **_kw):
        data = src.detach().cpu().numpy()
        if channels_first:  # (C, T) -> (T, C)
            data = data.T
        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]  # squeeze mono channel
        sf.write(str(uri), data, samplerate=sample_rate)

    _ta.load = _sf_load
    _ta.save = _sf_save


_patch_torchaudio()

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
    resolve_torch_dtype,
    safe_model_dirname,
)

# cuDNN SDPA has known issues with some models; disable it and rely on math/efficient.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
CODEC_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

MODE_OUTPUTS = {
    "clone": "55_moss_tts_local_transformer_clone.wav",
    "plain": "55_moss_tts_local_transformer_plain.wav",
}

# Recommended hyperparameters from the model card for MossTTSLocal-1.7B.
DEFAULT_AUDIO_TEMPERATURE = 1.0
DEFAULT_AUDIO_TOP_P = 0.95
DEFAULT_AUDIO_TOP_K = 50
DEFAULT_AUDIO_REPETITION_PENALTY = 1.1
# 12.5 tokens/s × 30 s ≈ 375 tokens for a ~30 s utterance; 4096 is the max.
DEFAULT_MAX_NEW_TOKENS = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOSS-TTS-Local-Transformer for the shared naturalness prompt."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--generation-mode", choices=sorted(MODE_OUTPUTS), default="clone")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--model-name", default=MODEL_ID)
    parser.add_argument("--codec-name", default=CODEC_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--audio-temperature", type=float, default=DEFAULT_AUDIO_TEMPERATURE)
    parser.add_argument("--audio-top-p", type=float, default=DEFAULT_AUDIO_TOP_P)
    parser.add_argument("--audio-top-k", type=int, default=DEFAULT_AUDIO_TOP_K)
    parser.add_argument("--audio-repetition-penalty", type=float, default=DEFAULT_AUDIO_REPETITION_PENALTY)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _resolve_hf_path(model_id: str, subdir: str) -> str:
    """Return local cache path if pre-downloaded, otherwise fall back to HF hub ID."""
    local = resolve_model_root() / "moss-tts" / safe_model_dirname(model_id)
    return str(local) if local.exists() else model_id


def _resolve_attn_implementation(device: str, dtype: torch.dtype) -> str:
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)

    model_path = _resolve_hf_path(args.model_name, "MOSS-TTS-Local-Transformer")
    codec_path = _resolve_hf_path(args.codec_name, "MOSS-Audio-Tokenizer")
    attn_impl = _resolve_attn_implementation(device, dtype)

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        codec_path=codec_path,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    if args.generation_mode == "clone":
        reference_path = args.reference_audio
        conversations = [
            [processor.build_user_message(text=text, reference=[str(reference_path)])]
        ]
        reference_audio_recorded: Path | None = reference_path
    else:
        conversations = [[processor.build_user_message(text=text)]]
        reference_audio_recorded = None

    batch = processor(conversations, mode="generation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.audio_temperature,
            top_p=args.audio_top_p,
            top_k=args.audio_top_k,
            repetition_penalty=args.audio_repetition_penalty,
        )

    messages = processor.decode(outputs)
    audio = messages[0].audio_codes_list[0]
    sample_rate = processor.model_config.sampling_rate
    # Bypass torchaudio.save — newer torchaudio (≥2.6) dispatches through
    # torchcodec internally even after monkey-patching the module attribute.
    _wav = audio.cpu().float().numpy()
    if _wav.ndim == 2:  # (channels, time) — take first channel if mono-ish
        _wav = _wav.T  # (time, channels) for soundfile
    sf.write(str(output_path), _wav, samplerate=sample_rate)

    record_run(
        model_id=f"moss_tts_local_transformer_{args.generation_mode}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=reference_audio_recorded,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "dtype": str(dtype),
            "mode": args.generation_mode,
            "model_name": args.model_name,
            "attn_implementation": attn_impl,
            "audio_temperature": args.audio_temperature,
            "audio_top_p": args.audio_top_p,
            "audio_top_k": args.audio_top_k,
            "audio_repetition_penalty": args.audio_repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "sample_rate": sample_rate,
        },
    )


if __name__ == "__main__":
    main()
