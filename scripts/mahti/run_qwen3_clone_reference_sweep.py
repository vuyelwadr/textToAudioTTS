#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


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
    / "audiobooks-half_of_a_yellow_sun-full-qwen-ref-sweep"
)
DEFAULT_SHORT_REFERENCE_AUDIO = REPO_ROOT / "assets" / "audio_samples" / "brothersKarmatzov.clone_ref_15s.wav"
DEFAULT_SHORT_REFERENCE_TEXT_FILE = DEFAULT_SHORT_REFERENCE_AUDIO.with_suffix(".transcript.txt")
DEFAULT_LONG_REFERENCE_AUDIO = (
    REPO_ROOT
    / "generated"
    / "clone_references_2026-04-25"
    / "brothersKarmatzov.clone_ref_60s.wav"
)
DEFAULT_LONG_REFERENCE_TEXT_FILE = DEFAULT_LONG_REFERENCE_AUDIO.with_suffix(".transcript.txt")
DEFAULT_BOOK_STEM = "Half_Of_A_Yellow_Sun"


@dataclass(frozen=True)
class ReferenceVariant:
    key: str
    output_name: str
    audio_path: Path
    text_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Faster Qwen3-TTS full-book clone outputs with short and long references."
    )
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--book-stem", default=DEFAULT_BOOK_STEM)
    parser.add_argument("--short-reference-audio", type=Path, default=DEFAULT_SHORT_REFERENCE_AUDIO)
    parser.add_argument("--short-reference-text-file", type=Path, default=DEFAULT_SHORT_REFERENCE_TEXT_FILE)
    parser.add_argument("--long-reference-audio", type=Path, default=DEFAULT_LONG_REFERENCE_AUDIO)
    parser.add_argument("--long-reference-text-file", type=Path, default=DEFAULT_LONG_REFERENCE_TEXT_FILE)
    parser.add_argument("--references", nargs="+", choices=("short", "long"), default=["short", "long"])
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_valid_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getnframes() > 0 and reader.getframerate() > 0
    except wave.Error:
        return False


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


def join_wavs(chunk_wavs: Iterable[Path], output_path: Path) -> dict[str, object]:
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
                writer.writeframes(reader.readframes(reader.getnframes()))
                total_frames += reader.getnframes()

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


def build_reference_variants(args: argparse.Namespace) -> dict[str, ReferenceVariant]:
    return {
        "short": ReferenceVariant(
            key="short",
            output_name="qwen3_faster_clone_brothers_15s",
            audio_path=args.short_reference_audio,
            text_file=args.short_reference_text_file,
        ),
        "long": ReferenceVariant(
            key="long",
            output_name="qwen3_faster_clone_brothers_60s",
            audio_path=args.long_reference_audio,
            text_file=args.long_reference_text_file,
        ),
    }


def run_reference(
    *,
    args: argparse.Namespace,
    ref: ReferenceVariant,
    chunks: list[Path],
) -> dict[str, object]:
    if not ref.audio_path.exists():
        raise FileNotFoundError(f"reference audio not found: {ref.audio_path}")
    if not ref.text_file.exists():
        raise FileNotFoundError(f"reference transcript not found: {ref.text_file}")

    variant_root = args.output_root / "half_of_a_yellow_sun" / ref.output_name
    chunk_output_dir = variant_root / "chunks"
    logs_dir = variant_root / "logs"
    model_run_log = logs_dir / "model_run.log"
    final_output = args.output_root / f"{args.book_stem}_{ref.output_name}.wav"
    logs_dir.mkdir(parents=True, exist_ok=True)
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        shutil.rmtree(chunk_output_dir, ignore_errors=True)
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        final_output.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "mahti" / "run_qwen3_batch_optimized.py"),
        "--input-dir",
        str(args.chunk_dir),
        "--output-dir",
        str(chunk_output_dir),
        "--prefix",
        "chunk_",
        "--generation-mode",
        "clone",
        "--xvec-only",
        "--attn-implementation",
        "sdpa",
        "--temperature",
        "0.88",
        "--top-k",
        "55",
        "--workers",
        args.workers,
        "--log-dir",
        str(logs_dir),
        "--reference-audio",
        str(ref.audio_path),
        "--reference-text-file",
        str(ref.text_file),
    ]

    worker_manifest = None
    if args.max_chunks is not None:
        worker_manifest = logs_dir / "selected_chunks.json"
        worker_manifest.write_text(
            json.dumps([str(path) for path in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        cmd.extend(["--worker-manifest", str(worker_manifest)])

    started = time.monotonic()
    if args.dry_run:
        return {
            "reference": ref.key,
            "status": "dry_run",
            "command": cmd,
            "chunk_count": len(chunks),
            "final_output": str(final_output),
        }

    with model_run_log.open("w", encoding="utf-8") as log_handle:
        subprocess.run(cmd, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=True)

    chunk_wavs = [chunk_output_dir / f"{chunk.stem}.wav" for chunk in chunks]
    missing = [str(path) for path in chunk_wavs if not is_valid_wav(path)]
    if missing:
        raise RuntimeError(f"{ref.key} missing/invalid chunk wavs: {missing[:10]}")

    final = join_wavs(chunk_wavs, final_output)
    manifest = {
        "reference": ref.key,
        "status": "ok",
        "chunk_count": len(chunks),
        "chunk_dir": str(args.chunk_dir),
        "chunk_output_dir": str(chunk_output_dir),
        "reference_audio": str(ref.audio_path),
        "reference_text_file": str(ref.text_file),
        "model_run_log": str(model_run_log),
        "runner_script": "scripts/mahti/run_qwen3_batch_optimized.py",
        "runner_args": cmd[cmd.index("--generation-mode") :],
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "final_output": final,
    }
    (variant_root / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    chunks = discover_chunks(args.chunk_dir, args.max_chunks)
    references = build_reference_variants(args)
    selected_refs = [references[key] for key in args.references]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "half_of_a_yellow_sun" / "qwen3_reference_sweep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "status": "running",
        "created_at": utc_now(),
        "chunk_dir": str(args.chunk_dir),
        "chunk_count": len(chunks),
        "output_root": str(args.output_root),
        "book_stem": args.book_stem,
        "workers": args.workers,
        "force": args.force,
        "references": [
            {
                "key": ref.key,
                "output_name": ref.output_name,
                "audio_path": str(ref.audio_path),
                "text_file": str(ref.text_file),
            }
            for ref in selected_refs
        ],
        "results": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    results = []
    try:
        for ref in selected_refs:
            print(f"[qwen-ref-sweep] running {ref.key}: {ref.audio_path}", flush=True)
            result = run_reference(args=args, ref=ref, chunks=chunks)
            results.append(result)
            manifest["results"] = results
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        manifest["results"] = results
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise

    manifest["status"] = "ok" if not args.dry_run else "dry_run"
    manifest["completed_at"] = utc_now()
    manifest["results"] = results
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
