from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
    safe_model_dirname,
)

from run_voxcpm2 import prepend_design_prompt
from voxcpm import VoxCPM

try:
    import torch
except ImportError:  # pragma: no cover - Mahti runner always has torch.
    torch = None
else:
    torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class VoiceReference:
    key: str
    audio_path: Path
    text: str | None
    text_file: Path | None


@dataclass(frozen=True)
class Variant:
    key: str
    mode: str
    cfg_value: float
    inference_timesteps: int
    voice_description: str | None = None
    reference_key: str | None = None


VOICE_DESIGNS: tuple[tuple[str, str], ...] = (
    (
        "warm_audiobook",
        "Warm, steady audiobook narrator with clear diction, intimate pacing, and restrained emotion.",
    ),
    (
        "deep_thriller",
        "Deep gravelly male thriller narrator with controlled suspense, close microphone tone, and crisp consonants.",
    ),
    (
        "british_baritone",
        "Mature British baritone literary narrator, elegant phrasing, dry warmth, and studio audiobook polish.",
    ),
    (
        "elder_literary",
        "Older literary male narrator with reflective cadence, soft authority, and careful sentence endings.",
    ),
    (
        "neutral_documentary",
        "Neutral documentary narrator with clean articulation, low drama, even pacing, and broadcast clarity.",
    ),
    (
        "clear_alto",
        "Clear alto female audiobook narrator, calm intelligence, warm tone, and precise pronunciation.",
    ),
    (
        "mezzo_dramatic",
        "Expressive mezzo-soprano female narrator with controlled drama, gentle warmth, and polished studio sound.",
    ),
    (
        "young_precise",
        "Younger precise narrator with bright articulation, natural pacing, and minimal theatricality.",
    ),
    (
        "intimate_storyteller",
        "Close intimate storyteller voice, soft dynamics, natural pauses, and a confidential audiobook feel.",
    ),
    (
        "classic_radio",
        "Classic radio narrator with rounded vowels, authoritative projection, and smooth vintage cadence.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-voice VoxCPM2 audiobook variant sweep.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-name", default="openbmb/VoxCPM2")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--rich-reference-audio", type=Path, default=None)
    parser.add_argument("--rich-reference-text", default=None)
    parser.add_argument("--rich-reference-text-file", type=Path, default=None)
    parser.add_argument("--cfg-values", nargs="+", type=float, default=[1.4, 2.0, 2.8])
    parser.add_argument("--plain-timesteps", nargs="+", type=int, default=[10, 16])
    parser.add_argument("--design-cfg-value", type=float, default=2.0)
    parser.add_argument("--design-timesteps", type=int, default=10)
    parser.add_argument("--clone-cfg-value", type=float, default=2.0)
    parser.add_argument("--clone-timesteps", type=int, default=10)
    parser.add_argument("--load-denoiser", action="store_true")
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-variants", type=int, default=None)
    return parser.parse_args()


def resolve_model_name(model_name: str) -> str:
    local_dir = resolve_model_root() / "voxcpm" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def slug_number(value: float | int) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def build_references(args: argparse.Namespace) -> dict[str, VoiceReference]:
    references = {
        "default_clone": VoiceReference(
            key="default_clone",
            audio_path=args.reference_audio,
            text=args.reference_text,
            text_file=args.reference_text_file,
        )
    }
    if args.rich_reference_audio:
        references["rich_miller"] = VoiceReference(
            key="rich_miller",
            audio_path=args.rich_reference_audio,
            text=args.rich_reference_text,
            text_file=args.rich_reference_text_file,
        )
    return references


def build_variants(args: argparse.Namespace, references: dict[str, VoiceReference]) -> list[Variant]:
    variants: list[Variant] = []

    for cfg_value in args.cfg_values:
        for timesteps in args.plain_timesteps:
            variants.append(
                Variant(
                    key=f"plain_cfg{slug_number(cfg_value)}_steps{timesteps}",
                    mode="tts",
                    cfg_value=cfg_value,
                    inference_timesteps=timesteps,
                )
            )

    for design_key, voice_description in VOICE_DESIGNS:
        variants.append(
            Variant(
                key=f"design_{design_key}",
                mode="design",
                cfg_value=args.design_cfg_value,
                inference_timesteps=args.design_timesteps,
                voice_description=voice_description,
            )
        )

    for reference_key in references:
        variants.append(
            Variant(
                key=f"clone_{reference_key}",
                mode="clone",
                cfg_value=args.clone_cfg_value,
                inference_timesteps=args.clone_timesteps,
                reference_key=reference_key,
            )
        )
        variants.append(
            Variant(
                key=f"ultimate_{reference_key}",
                mode="ultimate",
                cfg_value=args.clone_cfg_value,
                inference_timesteps=args.clone_timesteps,
                reference_key=reference_key,
            )
        )

    if args.start_index:
        if args.start_index < 0:
            raise ValueError("--start-index must be non-negative")
        variants = variants[args.start_index :]
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    return variants


def resolve_reference_texts(references: dict[str, VoiceReference]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, reference in references.items():
        resolved[key] = resolve_reference_text(
            reference.audio_path,
            reference.text,
            reference.text_file,
        )
    return resolved


def generate_variant(
    *,
    model: Any,
    text: str,
    variant: Variant,
    references: dict[str, VoiceReference],
    reference_texts: dict[str, str],
    output_dir: Path,
    input_path: Path,
    manifest_path: Path,
    model_name: str,
    load_denoiser: bool,
) -> dict[str, Any]:
    rendered_text = text
    reference_audio: Path | None = None
    generate_kwargs: dict[str, object] = {
        "cfg_value": variant.cfg_value,
        "inference_timesteps": variant.inference_timesteps,
    }

    extra: dict[str, Any] = {
        "cluster": resolve_cluster_name(),
        "cfg_value": variant.cfg_value,
        "inference_timesteps": variant.inference_timesteps,
        "load_denoiser": load_denoiser,
        "model_name": model_name,
        "mode": variant.mode,
        "sweep_variant": variant.key,
    }

    if variant.mode == "design":
        if not variant.voice_description:
            raise ValueError(f"{variant.key} missing voice description")
        rendered_text = prepend_design_prompt(text, variant.voice_description)
        model_id = "voxcpm2_voice_design_sweep"
        extra["voice_description"] = variant.voice_description
    elif variant.mode in {"clone", "ultimate"}:
        if not variant.reference_key:
            raise ValueError(f"{variant.key} missing reference key")
        reference = references[variant.reference_key]
        reference_audio = reference.audio_path
        generate_kwargs["reference_wav_path"] = str(reference.audio_path)
        extra["reference_key"] = variant.reference_key
        if variant.mode == "ultimate":
            generate_kwargs["prompt_wav_path"] = str(reference.audio_path)
            generate_kwargs["prompt_text"] = reference_texts[variant.reference_key]
            model_id = "voxcpm2_ultimate_clone_sweep"
        else:
            model_id = "voxcpm2_clone_sweep"
    elif variant.mode == "tts":
        model_id = "voxcpm2_tts_sweep"
    else:
        raise ValueError(f"unsupported mode: {variant.mode}")

    output_path = output_dir / f"{variant.key}.wav"
    ensure_parent(output_path)

    started = time.monotonic()
    waveform = model.generate(text=rendered_text, **generate_kwargs)
    elapsed_seconds = time.monotonic() - started
    sample_rate = int(model.tts_model.sample_rate)
    sf.write(str(output_path), waveform, sample_rate)

    duration_seconds = len(waveform) / sample_rate if sample_rate else 0.0
    extra.update(
        {
            "duration_seconds": duration_seconds,
            "elapsed_seconds": elapsed_seconds,
            "sample_rate": sample_rate,
        }
    )
    record_run(
        model_id=model_id,
        output_path=output_path,
        input_path=input_path,
        reference_audio=reference_audio,
        manifest_path=manifest_path,
        extra=extra,
    )

    return {
        "variant": variant.key,
        "mode": variant.mode,
        "output_path": str(output_path),
        "duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "cfg_value": variant.cfg_value,
        "inference_timesteps": variant.inference_timesteps,
        "voice_description": variant.voice_description,
        "reference_key": variant.reference_key,
        "sample_rate": sample_rate,
    }


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest
    ensure_parent(manifest_path)

    references = build_references(args)
    variants = build_variants(args, references)
    reference_texts = resolve_reference_texts(references)

    model_name = resolve_model_name(args.model_name)
    use_local_files = Path(model_name).exists()
    model = VoxCPM.from_pretrained(
        model_name,
        optimize=args.optimize,
        load_denoiser=args.load_denoiser,
        local_files_only=use_local_files,
    )

    manifest: dict[str, Any] = {
        "input": str(args.input),
        "model_name": args.model_name,
        "resolved_model_name": model_name,
        "load_denoiser": args.load_denoiser,
        "variant_count": len(variants),
        "references": {
            key: {
                "audio_path": str(reference.audio_path),
                "has_manual_text": bool(reference.text or reference.text_file),
                "resolved_text_words": len(reference_texts[key].split()),
            }
            for key, reference in references.items()
        },
        "results": [],
    }

    for index, variant in enumerate(variants, start=1):
        print(f"[voxcpm2-sweep] {index}/{len(variants)} {variant.key}", flush=True)
        try:
            result = generate_variant(
                model=model,
                text=text,
                variant=variant,
                references=references,
                reference_texts=reference_texts,
                output_dir=output_dir,
                input_path=args.input,
                manifest_path=manifest_path,
                model_name=args.model_name,
                load_denoiser=args.load_denoiser,
            )
        except Exception as exc:
            result = {
                "variant": variant.key,
                "mode": variant.mode,
                "cfg_value": variant.cfg_value,
                "inference_timesteps": variant.inference_timesteps,
                "voice_description": variant.voice_description,
                "reference_key": variant.reference_key,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"[voxcpm2-sweep][error] {variant.key}: {exc!r}", flush=True)
        manifest["results"].append(result)
        (output_dir / "variant_sweep_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
