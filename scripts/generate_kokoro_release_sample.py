#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download


SAMPLE_RATE = 24000


@dataclass(frozen=True)
class ReleaseSpec:
    release_id: str
    label: str
    loader_kind: str
    model_repo: str
    config_filename: str
    model_filename: str
    voice_repo: str
    voice_filename_template: str


RELEASES: dict[str, ReleaseSpec] = {
    "v1_0": ReleaseSpec(
        release_id="v1_0",
        label="Kokoro v1.0",
        loader_kind="pipeline",
        model_repo="hexgrad/Kokoro-82M",
        config_filename="config.json",
        model_filename="kokoro-v1_0.pth",
        voice_repo="hexgrad/Kokoro-82M",
        voice_filename_template="voices/{voice}.pt",
    ),
    "v0_19": ReleaseSpec(
        release_id="v0_19",
        label="Kokoro v0.19",
        loader_kind="legacy_code",
        model_repo="hexgrad/kLegacy",
        config_filename="v0.19/config.json",
        model_filename="v0.19/kokoro-v0_19.pth",
        voice_repo="hexgrad/kLegacy",
        voice_filename_template="v0.19/voices/{voice}.pt",
    ),
    "v0_19_fp16": ReleaseSpec(
        release_id="v0_19_fp16",
        label="Kokoro v0.19 fp16",
        loader_kind="legacy_code",
        model_repo="hexgrad/kLegacy",
        config_filename="v0.19/config.json",
        model_filename="v0.19/fp16/kokoro-v0_19-half.pth",
        voice_repo="hexgrad/kLegacy",
        voice_filename_template="v0.19/voices/{voice}.pt",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a short sample from a specific Kokoro model release")
    parser.add_argument("input", type=Path, help="Input text file")
    parser.add_argument("--release-id", choices=sorted(RELEASES), required=True, help="Kokoro release to compare")
    parser.add_argument("--voice", default="am_adam", help="Shared voice to use across Kokoro releases")
    parser.add_argument("--voice-repo", default=None, help="Optional Hugging Face repo to source the voice pack from")
    parser.add_argument("--lang-code", default="a", help="KPipeline language code")
    parser.add_argument("--speed", type=float, default=None, help="Speech speed; defaults to the repo Kokoro baseline heuristic")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--output-name", default=None, help="Output wav filename")
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_assets(spec: ReleaseSpec, voice: str, voice_repo_override: str | None = None) -> tuple[Path, Path, Path]:
    config_path = Path(hf_hub_download(spec.model_repo, spec.config_filename))
    model_path = Path(hf_hub_download(spec.model_repo, spec.model_filename))
    voice_repo = voice_repo_override or spec.voice_repo
    voice_filename = spec.voice_filename_template.format(voice=voice)
    if voice_repo_override and voice_repo_override != spec.voice_repo and voice_filename.startswith("v0.19/"):
        voice_filename = f"voices/{voice}.pt"
    voice_path = Path(hf_hub_download(voice_repo, voice_filename))
    return config_path, model_path, voice_path


def download_legacy_runtime(spec: ReleaseSpec) -> dict[str, Path]:
    legacy_files = {
        "kokoro": "v0.19/kokoro.py",
        "models": "v0.19/models.py",
        "istftnet": "v0.19/istftnet.py",
        "plbert": "v0.19/plbert.py",
    }
    return {
        name: Path(hf_hub_download(spec.model_repo, filename))
        for name, filename in legacy_files.items()
    }


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prepare_audio(audio: np.ndarray) -> np.ndarray:
    prepared = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(prepared))) if len(prepared) else 0.0
    if peak > 0.999:
        prepared = prepared * (0.999 / peak)
    return prepared


def dynamic_speed(text: str) -> float:
    if '?' in text:
        return 0.95
    if '!' in text:
        return 1.05
    if re.search(r'".*"', text) or re.search(r'".*"', text):
        return 1.02
    if len(text.split()) > 20 or text.count(',') > 2:
        return 0.97
    return 1.0


def run_pipeline_release(spec: ReleaseSpec, text: str, voice_path: Path, config_path: Path, model_path: Path, device: torch.device, speed: float) -> np.ndarray:
    from kokoro import KPipeline
    from kokoro.model import KModel

    model = KModel(config=str(config_path), model=str(model_path)).to(device).eval()
    pipeline = KPipeline(lang_code="a", repo_id=spec.voice_repo, model=model, device=device.type)

    audio_chunks: list[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=str(voice_path), speed=speed, split_pattern=r"\n+"):
        if isinstance(audio, torch.Tensor):
            audio_chunks.append(prepare_audio(audio.detach().cpu().numpy()))
        else:
            audio_chunks.append(prepare_audio(np.asarray(audio)))

    if not audio_chunks:
        raise RuntimeError(f"{spec.label} did not produce any audio chunks")
    return np.concatenate(audio_chunks)


def run_legacy_release(spec: ReleaseSpec, text: str, voice_path: Path, model_path: Path, device: torch.device, speed: float) -> np.ndarray:
    runtime_files = download_legacy_runtime(spec)
    runtime_root = runtime_files["kokoro"].parent
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))

    models_mod = load_module(f"legacy_models_{spec.release_id}", runtime_files["models"])
    kokoro_mod = load_module(f"legacy_kokoro_{spec.release_id}", runtime_files["kokoro"])

    device_candidates = [device.type]
    if device.type == "mps":
        device_candidates.append("cpu")

    last_error: Exception | None = None
    for device_name in device_candidates:
        try:
            model = models_mod.build_model(str(model_path), device_name)
            voicepack = torch.load(str(voice_path), weights_only=True).to(device_name)
            generated = kokoro_mod.generate_full(model, text, voicepack, lang=voice_path.stem[0], speed=speed)
            if generated is None:
                raise RuntimeError(f"{spec.label} returned no audio")
            audio, _ = generated
            return prepare_audio(np.asarray(audio))
        except Exception as error:
            last_error = error
            print(f"Legacy Kokoro attempt on {device_name} failed: {error}")

    raise RuntimeError(f"Legacy Kokoro generation failed for {spec.release_id}: {last_error}")


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    spec = RELEASES[args.release_id]
    device = resolve_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text = args.input.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {args.input}")
    resolved_speed = args.speed if args.speed is not None else dynamic_speed(text)

    print(f"Using device: {device.type}")
    print(f"Preparing {spec.label} with voice {args.voice}")
    print(f"Using speed: {resolved_speed}")
    config_path, model_path, voice_path = download_assets(spec, args.voice, args.voice_repo)
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Voice: {voice_path}")

    if spec.loader_kind == "pipeline":
        combined_audio = run_pipeline_release(spec, text, voice_path, config_path, model_path, device, resolved_speed)
    elif spec.loader_kind == "legacy_code":
        combined_audio = run_legacy_release(spec, text, voice_path, model_path, device, resolved_speed)
    else:
        raise ValueError(f"Unsupported loader kind: {spec.loader_kind}")

    output_name = args.output_name or f"{args.release_id}_{args.voice}.wav"
    output_path = args.output_dir / output_name
    sf.write(output_path, combined_audio, SAMPLE_RATE, subtype="PCM_24")
    print(f"Saved audio to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())