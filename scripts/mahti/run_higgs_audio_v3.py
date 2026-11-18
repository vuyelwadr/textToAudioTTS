from __future__ import annotations

import argparse
import time
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
    resolve_cluster_name,
    resolve_device,
    resolve_output_dir,
    resolve_reference_text,
)

# Unlike Higgs Audio v2 (`HiggsAudioV2ForConditionalGeneration`, native in
# transformers), the v3 weights repo ships config + safetensors only -- no
# modeling code, and transformers has no "higgs_multimodal_qwen3" class to
# load it with. Boson's model card documents SGLang-Omni (Docker) or
# vLLM-Omni as the only supported serving stacks. vLLM-Omni is pip
# installable and ships an *offline* inference path (vllm_omni.Omni +
# HiggsAudioV3TokenizerAdapter), which is what this script drives directly --
# no HTTP server needed for a single clip. See:
# https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/text_to_speech/higgs_audio_v3/end2end.py
DEFAULT_MODEL_NAME = "bosonai/higgs-audio-v3-tts-4b"
SAMPLE_RATE = 24_000

MODE_OUTPUTS = {
    "plain": "85_higgs_audio_v3_4b_plain.wav",
    "clone": "86_higgs_audio_v3_4b_clone.wav",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Higgs Audio v3 (via vLLM-Omni offline inference) for the shared naturalness prompt."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--generation-mode", choices=sorted(MODE_OUTPUTS), default="plain"
    )
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        default=None,
        help="Informational only -- vLLM-Omni manages device placement "
        "internally. Used to fail fast if no CUDA device is visible.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _extract_pcm(multimodal_output: dict) -> torch.Tensor:
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    if audio is None:
        raise ValueError(f"no audio key in multimodal_output: {list(multimodal_output.keys())}")
    if isinstance(audio, list):
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not valid:
            raise ValueError("audio list is empty")
        return torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]
    return torch.as_tensor(audio).float().cpu().reshape(-1)


def _pcm_to_int16(pcm: torch.Tensor) -> np.ndarray:
    arr = pcm.numpy()
    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    else:
        arr = arr.astype(np.int16)
    return arr


def build_prompt(
    args: argparse.Namespace,
    text: str,
    adapter,
) -> tuple[dict, Path | None, str | None]:
    if args.generation_mode == "plain":
        prompt_ids = adapter.build_prompt(text)
        return {"prompt_token_ids": prompt_ids}, None, None

    from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
        apply_delay_pattern,
        encode_reference_audio,
    )

    reference_text = resolve_reference_text(
        args.reference_audio,
        args.reference_text,
        args.reference_text_file,
    )

    ref_wav, ref_sr = sf.read(str(args.reference_audio), always_2d=False)
    if ref_wav.ndim == 2:
        ref_wav = ref_wav.mean(axis=1)
    ref_codes_raw = encode_reference_audio(ref_wav, int(ref_sr))
    ref_codes_delayed = apply_delay_pattern(ref_codes_raw)

    prompt_ids = adapter.build_prompt(
        text,
        num_ref_tokens=int(ref_codes_delayed.shape[0]),
        reference_text=reference_text,
    )
    prompt = {
        "prompt_token_ids": prompt_ids,
        "additional_information": {
            "audio_input_ids": ref_codes_delayed.to(torch.long),
            "audio_input_ids_mask": torch.ones(ref_codes_delayed.shape[0], dtype=torch.bool),
        },
    }
    return prompt, args.reference_audio, reference_text


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    device = resolve_device(args.device)
    if device != "cuda":
        raise RuntimeError(
            f"Higgs Audio v3 via vLLM-Omni requires a CUDA device; resolved device={device!r}"
        )

    # Imported lazily so --help works even before the venv has vllm-omni
    # installed, and so import errors surface with the model name in context.
    from transformers import AutoTokenizer
    from vllm_omni import Omni
    from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
        HiggsAudioV3TokenizerAdapter,
    )

    t_load_start = time.perf_counter()
    engine = Omni(model=args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    adapter = HiggsAudioV3TokenizerAdapter(tokenizer)
    load_elapsed = time.perf_counter() - t_load_start

    prompt, reference_audio, reference_text = build_prompt(args, text, adapter)

    t_gen_start = time.perf_counter()
    outputs = engine.generate([prompt])
    gen_elapsed = time.perf_counter() - t_gen_start

    multimodal_output = outputs[0].outputs[0].multimodal_output
    pcm = _extract_pcm(multimodal_output)
    sf.write(str(output_path), _pcm_to_int16(pcm), SAMPLE_RATE, format="WAV", subtype="PCM_16")

    audio_duration = pcm.numel() / SAMPLE_RATE
    rtf = gen_elapsed / audio_duration if audio_duration > 0 else float("inf")

    record_run(
        model_id=f"higgs_audio_v3_4b_{args.generation_mode}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=reference_audio,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "model_name": args.model_name,
            "mode": args.generation_mode,
            "max_new_tokens": args.max_new_tokens,
            "reference_text": reference_text,
            "engine_load_seconds": round(load_elapsed, 3),
            "generation_seconds": round(gen_elapsed, 3),
            "audio_duration_seconds": round(audio_duration, 3),
            "rtf": round(rtf, 4),
            "engine": "vllm-omni",
        },
    )
    print(
        f"wrote {output_path} | audio={audio_duration:.2f}s "
        f"gen={gen_elapsed:.2f}s load={load_elapsed:.2f}s rtf={rtf:.3f}"
    )


if __name__ == "__main__":
    main()
