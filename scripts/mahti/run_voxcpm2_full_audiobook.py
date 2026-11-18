#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

from common import (
    normalize_target_text,
    resolve_model_root,
    resolve_visible_cuda_devices,
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNK_DIR = (
    Path(os.environ.get("MAHTI_WORK_ROOT", REPO_ROOT))
    / "generated"
    / "audiobooks-half_of_a_yellow_sun-full"
    / "half_of_a_yellow_sun"
    / "chunks"
)
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("MAHTI_WORK_ROOT", REPO_ROOT))
    / "generated"
    / "audiobooks-half_of_a_yellow_sun-full-voxcpm2"
)
DEFAULT_REFERENCE_AUDIO = REPO_ROOT / "assets" / "audio_samples" / "brothersKarmatzov.clone_ref_15s.wav"
DEFAULT_REFERENCE_TEXT_FILE = DEFAULT_REFERENCE_AUDIO.with_suffix(".transcript.txt")
DEFAULT_BOOK_STEM = "Half_Of_A_Yellow_Sun"


@dataclass(frozen=True)
class Variant:
    key: str
    output_name: str
    mode: str
    cfg_value: float
    inference_timesteps: int
    voice_description: str | None = None
    reference_key: str | None = None


VARIANTS: dict[str, Variant] = {
    "clone_default_clone": Variant(
        key="clone_default_clone",
        output_name="voxcpm2_clone_default_clone",
        mode="clone",
        cfg_value=2.0,
        inference_timesteps=10,
        reference_key="short",
    ),
    "clone_brothers_60s": Variant(
        key="clone_brothers_60s",
        output_name="voxcpm2_clone_brothers_60s",
        mode="clone",
        cfg_value=2.0,
        inference_timesteps=10,
        reference_key="long",
    ),
    "design_british_baritone": Variant(
        key="design_british_baritone",
        output_name="voxcpm2_design_british_baritone",
        mode="design",
        cfg_value=2.0,
        inference_timesteps=10,
        voice_description=(
            "Mature British baritone literary narrator, elegant phrasing, dry warmth, "
            "and studio audiobook polish."
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full Half of a Yellow Sun VoxCPM2 audiobook variants from existing chunk text."
    )
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--book-stem", default=DEFAULT_BOOK_STEM)
    parser.add_argument("--model-name", default="openbmb/VoxCPM2")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text-file", type=Path, default=DEFAULT_REFERENCE_TEXT_FILE)
    parser.add_argument("--long-reference-audio", type=Path, default=None)
    parser.add_argument("--long-reference-text-file", type=Path, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=["clone_default_clone", "design_british_baritone"],
    )
    parser.add_argument("--workers", default="auto", help="GPU worker count, integer or auto")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate existing chunk WAVs")
    parser.add_argument("--load-denoiser", action="store_true")
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-manifest", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=-1, help=argparse.SUPPRESS)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_model_name(model_name: str) -> str:
    local_dir = resolve_model_root() / "voxcpm" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def discover_chunks(chunk_dir: Path, max_chunks: int | None) -> list[Path]:
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"chunk dir not found: {chunk_dir}")
    chunks = sorted(chunk_dir.glob("chunk_*.txt"))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.txt files found in {chunk_dir}")
    if max_chunks is not None:
        if max_chunks < 1:
            raise ValueError("--max-chunks must be at least 1")
        chunks = chunks[:max_chunks]
    return chunks


def load_worker_chunks(worker_manifest: Path) -> list[Path]:
    raw_paths = json.loads(worker_manifest.read_text(encoding="utf-8"))
    return [Path(raw_path) for raw_path in raw_paths]


def variant_dir(output_root: Path, variant: Variant) -> Path:
    return output_root / "half_of_a_yellow_sun" / variant.output_name


def variant_chunk_path(output_root: Path, variant: Variant, chunk_path: Path) -> Path:
    return variant_dir(output_root, variant) / "chunks" / f"{chunk_path.stem}.wav"


def final_output_path(output_root: Path, book_stem: str, variant: Variant) -> Path:
    return output_root / f"{book_stem}_{variant.output_name}.wav"


def is_valid_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getnframes() > 0 and reader.getframerate() > 0
    except wave.Error:
        return False


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def partition_chunks(chunks: list[Path], worker_count: int) -> list[list[Path]]:
    buckets: list[list[Path]] = [[] for _ in range(worker_count)]
    weights = [0] * worker_count
    for chunk in sorted(chunks, key=lambda path: (path.stat().st_size, path.name), reverse=True):
        bucket_index = min(range(worker_count), key=lambda index: weights[index])
        buckets[bucket_index].append(chunk)
        weights[bucket_index] += chunk.stat().st_size
    return [sorted(bucket) for bucket in buckets if bucket]


def resolve_worker_count(requested: str, chunks: list[Path]) -> tuple[int, list[str]]:
    visible_devices = resolve_visible_cuda_devices()
    if requested != "auto":
        worker_count = max(1, min(int(requested), len(chunks)))
        return worker_count, visible_devices[:worker_count]
    if visible_devices:
        worker_count = min(len(visible_devices), len(chunks))
        return worker_count, visible_devices[:worker_count]
    return 1, []


def variant_generate_kwargs(
    variant: Variant, references: dict[str, Path]
) -> tuple[str, dict[str, object], Path | None]:
    kwargs: dict[str, object] = {
        "cfg_value": variant.cfg_value,
        "inference_timesteps": variant.inference_timesteps,
    }
    reference_path: Path | None = None
    if variant.mode == "clone":
        reference_key = variant.reference_key or "short"
        if reference_key not in references:
            raise ValueError(f"{variant.key} requested missing reference key: {reference_key}")
        reference_path = references[reference_key]
        kwargs["reference_wav_path"] = str(reference_path)
    return variant.mode, kwargs, reference_path


def generate_one(
    *,
    model: Any,
    text_path: Path,
    output_path: Path,
    variant: Variant,
    references: dict[str, Path],
    force: bool,
) -> dict[str, Any]:
    if force and output_path.exists():
        output_path.unlink()
    if not force and is_valid_wav(output_path):
        return {
            "chunk": text_path.name,
            "variant": variant.key,
            "output_path": str(output_path),
            "status": "skipped",
            "duration_seconds": round(wav_duration(output_path), 4),
        }

    text = normalize_target_text(text_path.read_text(encoding="utf-8"))
    rendered_text = text
    if variant.mode == "design":
        if not variant.voice_description:
            raise ValueError(f"{variant.key} missing voice description")
        rendered_text = prepend_design_prompt(text, variant.voice_description)

    _, generate_kwargs, reference_path = variant_generate_kwargs(variant, references)
    if reference_path is not None and not reference_path.exists():
        raise FileNotFoundError(f"reference audio not found: {reference_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    waveform = model.generate(text=rendered_text, **generate_kwargs)
    elapsed = time.monotonic() - started
    sample_rate = int(model.tts_model.sample_rate)
    sf.write(str(output_path), waveform, sample_rate)
    if not is_valid_wav(output_path):
        raise RuntimeError(f"invalid or empty output wav: {output_path}")

    return {
        "chunk": text_path.name,
        "variant": variant.key,
        "output_path": str(output_path),
        "status": "generated",
        "duration_seconds": round(len(waveform) / sample_rate, 4),
        "elapsed_seconds": round(elapsed, 4),
        "sample_rate": sample_rate,
        "bytes": output_path.stat().st_size,
    }


def run_worker(args: argparse.Namespace) -> int:
    assert args.worker_manifest is not None
    chunks = load_worker_chunks(args.worker_manifest)
    variants = [VARIANTS[key] for key in args.variants]
    references = {"short": args.reference_audio}
    if args.long_reference_audio is not None:
        references["long"] = args.long_reference_audio
    model_name = resolve_model_name(args.model_name)
    use_local_files = Path(model_name).exists()

    model = VoxCPM.from_pretrained(
        model_name,
        optimize=args.optimize,
        load_denoiser=args.load_denoiser,
        local_files_only=use_local_files,
    )

    worker_log_dir = args.output_root / "half_of_a_yellow_sun" / "logs"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    worker_summary_path = worker_log_dir / f"worker_{args.worker_index:02d}_summary.json"
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for chunk_index, chunk_path in enumerate(chunks, start=1):
        for variant in variants:
            output_path = variant_chunk_path(args.output_root, variant, chunk_path)
            label = f"[worker {args.worker_index}] {chunk_index}/{len(chunks)} {variant.key} {chunk_path.name}"
            print(label, flush=True)
            try:
                result = generate_one(
                    model=model,
                    text_path=chunk_path,
                    output_path=output_path,
                    variant=variant,
                    references=references,
                    force=args.force,
                )
                print(
                    f"[worker {args.worker_index}]   {result['status']} "
                    f"{Path(result['output_path']).name} duration={result.get('duration_seconds')}s",
                    flush=True,
                )
                results.append(result)
            except Exception as exc:
                failure = {
                    "chunk": chunk_path.name,
                    "variant": variant.key,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"[worker {args.worker_index}][error] {failure}", flush=True)
                failures.append(failure)

            worker_summary_path.write_text(
                json.dumps(
                    {
                        "worker_index": args.worker_index,
                        "updated_at": utc_now(),
                        "chunks": [str(path) for path in chunks],
                        "results": results,
                        "failures": failures,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    return 1 if failures else 0


def join_wavs(chunk_wavs: Iterable[Path], output_path: Path) -> dict[str, Any]:
    paths = list(chunk_wavs)
    if not paths:
        raise ValueError("no chunk wavs to join")

    params = None
    signature = None
    total_frames = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        for wav_path in paths:
            if not is_valid_wav(wav_path):
                raise RuntimeError(f"missing or invalid chunk wav: {wav_path}")
            with wave.open(str(wav_path), "rb") as reader:
                current_params = reader.getparams()
                current_signature = (
                    current_params.nchannels,
                    current_params.sampwidth,
                    current_params.framerate,
                    current_params.comptype,
                    current_params.compname,
                )
                if params is None:
                    params = current_params
                    signature = current_signature
                    writer.setnchannels(params.nchannels)
                    writer.setsampwidth(params.sampwidth)
                    writer.setframerate(params.framerate)
                    writer.setcomptype(params.comptype, params.compname)
                elif current_signature != signature:
                    raise RuntimeError(
                        f"WAV format mismatch in {wav_path}: expected {signature}, got {current_signature}"
                    )
                frames = reader.readframes(reader.getnframes())
                total_frames += reader.getnframes()
                writer.writeframes(frames)

    if params is None:
        raise RuntimeError("could not read wav params")
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "duration_seconds": round(total_frames / params.framerate, 4),
        "sample_rate": params.framerate,
        "channels": params.nchannels,
        "sample_width": params.sampwidth,
    }


def spawn_workers(args: argparse.Namespace, chunks: list[Path]) -> list[dict[str, Any]]:
    worker_count, visible_devices = resolve_worker_count(args.workers, chunks)
    partitions = partition_chunks(chunks, worker_count)
    logs_dir = args.output_root / "half_of_a_yellow_sun" / "logs"
    manifests_dir = logs_dir / "worker_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[voxcpm2-full] launching {len(partitions)} workers "
        f"for {len(chunks)} chunks; visible_devices={visible_devices or ['cpu/default']}",
        flush=True,
    )

    worker_processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    script = Path(__file__).resolve()
    for index, worker_chunks in enumerate(partitions):
        manifest_path = manifests_dir / f"worker_{index:02d}.json"
        manifest_path.write_text(
            json.dumps([str(path) for path in worker_chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        worker_log_path = logs_dir / f"worker_{index:02d}.log"
        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--chunk-dir",
            str(args.chunk_dir),
            "--output-root",
            str(args.output_root),
            "--book-stem",
            args.book_stem,
            "--model-name",
            args.model_name,
            "--reference-audio",
            str(args.reference_audio),
            "--reference-text-file",
            str(args.reference_text_file),
            "--variants",
            *args.variants,
            "--workers",
            "1",
            "--worker-manifest",
            str(manifest_path),
            "--worker-index",
            str(index),
        ]
        if args.source_manifest:
            cmd.extend(["--source-manifest", str(args.source_manifest)])
        if args.long_reference_audio is not None:
            cmd.extend(["--long-reference-audio", str(args.long_reference_audio)])
        if args.long_reference_text_file is not None:
            cmd.extend(["--long-reference-text-file", str(args.long_reference_text_file)])
        if args.load_denoiser:
            cmd.append("--load-denoiser")
        cmd.append("--optimize" if args.optimize else "--no-optimize")
        if args.force:
            cmd.append("--force")

        worker_env = os.environ.copy()
        if visible_devices:
            worker_env["CUDA_VISIBLE_DEVICES"] = visible_devices[index]
        worker_env["OMP_NUM_THREADS"] = "1"
        worker_env["MKL_NUM_THREADS"] = "1"
        worker_env["PYTHONUNBUFFERED"] = "1"
        log_handle = worker_log_path.open("w", encoding="utf-8")
        print(f"[voxcpm2-full] worker {index}: {len(worker_chunks)} chunks -> {worker_log_path}", flush=True)
        worker_processes.append((index, subprocess.Popen(cmd, env=worker_env, stdout=log_handle, stderr=subprocess.STDOUT), log_handle))

    failures: list[dict[str, Any]] = []
    for index, process, log_handle in worker_processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append({"worker": index, "return_code": return_code})

    if failures:
        raise RuntimeError(f"worker failures: {failures}")

    summaries: list[dict[str, Any]] = []
    for index in range(len(partitions)):
        summary_path = logs_dir / f"worker_{index:02d}_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"missing worker summary: {summary_path}")
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return summaries


def verify_and_join(args: argparse.Namespace, chunks: list[Path]) -> dict[str, Any]:
    final_outputs: dict[str, Any] = {}
    for variant_key in args.variants:
        variant = VARIANTS[variant_key]
        chunk_wavs = [variant_chunk_path(args.output_root, variant, chunk) for chunk in chunks]
        missing = [str(path) for path in chunk_wavs if not is_valid_wav(path)]
        if missing:
            raise RuntimeError(f"{variant.key} missing/invalid chunk wavs: {missing[:10]}")
        final_outputs[variant.key] = join_wavs(
            chunk_wavs,
            final_output_path(args.output_root, args.book_stem, variant),
        )
    return final_outputs


def main() -> int:
    args = parse_args()
    if args.worker_manifest is not None:
        return run_worker(args)

    chunks = discover_chunks(args.chunk_dir, args.max_chunks)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.source_manifest is None:
        inferred_manifest = args.chunk_dir / "half_of_a_yellow_sun_manifest.json"
        if inferred_manifest.exists():
            args.source_manifest = inferred_manifest

    manifest_path = args.output_root / "half_of_a_yellow_sun" / "voxcpm2_full_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selected_variants = [VARIANTS[key] for key in args.variants]
    references = {
        "short": {
            "audio_path": str(args.reference_audio),
            "text_file": str(args.reference_text_file),
        }
    }
    if args.long_reference_audio is not None:
        references["long"] = {
            "audio_path": str(args.long_reference_audio),
            "text_file": str(args.long_reference_text_file) if args.long_reference_text_file else None,
        }
    preflight = {
        "status": "dry_run" if args.dry_run else "running",
        "created_at": utc_now(),
        "chunk_dir": str(args.chunk_dir),
        "source_manifest": str(args.source_manifest) if args.source_manifest else None,
        "chunk_count": len(chunks),
        "output_root": str(args.output_root),
        "model_name": args.model_name,
        "resolved_model_name": resolve_model_name(args.model_name),
        "reference_audio": str(args.reference_audio),
        "reference_text_file": str(args.reference_text_file),
        "references": references,
        "book_stem": args.book_stem,
        "workers": args.workers,
        "force": args.force,
        "load_denoiser": args.load_denoiser,
        "optimize": args.optimize,
        "variants": [variant.__dict__ for variant in selected_variants],
    }
    manifest_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0

    started = time.monotonic()
    worker_summaries = spawn_workers(args, chunks)
    final_outputs = verify_and_join(args, chunks)
    manifest = dict(preflight)
    manifest.update(
        {
            "status": "ok",
            "completed_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "worker_summaries": worker_summaries,
            "final_outputs": final_outputs,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
