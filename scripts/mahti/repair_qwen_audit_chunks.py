#!/usr/bin/env python3
"""Prepare/finalize split repairs for Qwen audiobook chunks that fail audit.

Some citation-heavy audiobook chunks repeatedly hit the decoder ceiling when
generated as one long prompt. This helper keeps the public chunk manifest
stable: it splits only the audit-blocking text files into smaller temporary
part files, lets the existing Qwen batch runner synthesize those parts, then
concatenates the part WAVs back into the original chunk WAV paths.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "finalize"], required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--asr-cache-dir", type=Path, required=True)
    parser.add_argument("--max-part-chars", type=int, default=520)
    parser.add_argument("--min-part-chars", type=int, default=120)
    parser.add_argument("--silence-ms", type=int, default=120)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


def split_oversized_unit(unit: str, max_chars: int) -> list[str]:
    words = unit.split()
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        next_length = length + len(word) + (1 if current else 0)
        if current and next_length > max_chars:
            parts.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length = next_length
    if current:
        parts.append(" ".join(current))
    return parts


def split_text(text: str, *, max_chars: int, min_chars: int) -> list[str]:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return [normalized]

    coarse_units = [unit.strip() for unit in re.split(r"\n{2,}", normalized) if unit.strip()]
    units: list[str] = []
    for unit in coarse_units:
        if len(unit) <= max_chars:
            units.append(unit)
            continue
        sentence_units = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?;:])\s+", unit)
            if sentence.strip()
        ]
        for sentence in sentence_units:
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                units.extend(split_oversized_unit(sentence, max_chars))

    parts: list[str] = []
    current = ""
    for unit in units:
        separator = "\n\n" if "\n" in current else " "
        candidate = (current + separator + unit).strip() if current else unit
        if current and len(candidate) > max_chars and len(current) >= min_chars:
            parts.append(current.strip())
            current = unit
        else:
            current = candidate
    if current.strip():
        parts.append(current.strip())

    return parts


def cache_path_for_audio(asr_cache_dir: Path, audio_path: Path) -> Path:
    return asr_cache_dir / f"{audio_path.stem}.active_line_asr.small.en.json"


def prepare(args: argparse.Namespace) -> int:
    summary = read_json(args.audit_summary)
    blocking_indices = list(summary.get("blocking_chunk_indices") or [])
    chunks = list(summary.get("chunks") or [])
    if not blocking_indices:
        print("blocking_count=0")
        return 0

    split_text_dir = args.repair_dir / "split_text"
    split_audio_dir = args.repair_dir / "split_audio"
    split_text_dir.mkdir(parents=True, exist_ok=True)
    split_audio_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    worker_paths: list[str] = []
    for index in blocking_indices:
        item = chunks[index]
        text_path = Path(item["text_path"])
        audio_path = Path(item["audio_path"])
        audio_cache_path = cache_path_for_audio(args.asr_cache_dir, audio_path)
        if audio_path.exists():
            audio_path.unlink()
        if audio_cache_path.exists():
            audio_cache_path.unlink()

        text = text_path.read_text(encoding="utf-8")
        parts = split_text(
            text,
            max_chars=args.max_part_chars,
            min_chars=args.min_part_chars,
        )
        part_text_paths: list[str] = []
        part_audio_paths: list[str] = []
        for part_index, part_text in enumerate(parts):
            part_stem = f"{text_path.stem}__part{part_index:03d}"
            part_text_path = split_text_dir / f"{part_stem}.txt"
            part_audio_path = split_audio_dir / f"{part_stem}.wav"
            part_text_path.write_text(part_text + "\n", encoding="utf-8")
            if part_audio_path.exists():
                part_audio_path.unlink()
            part_text_paths.append(str(part_text_path))
            part_audio_paths.append(str(part_audio_path))
            worker_paths.append(str(part_text_path))

        manifest.append(
            {
                "chunk_index": index,
                "text_path": str(text_path),
                "audio_path": str(audio_path),
                "part_text_paths": part_text_paths,
                "part_audio_paths": part_audio_paths,
            }
        )

    write_json(args.repair_dir / "split_repair_manifest.json", manifest)
    write_json(args.repair_dir / "split_worker_manifest.json", worker_paths)
    print(f"blocking_count={len(blocking_indices)}")
    print(f"part_count={len(worker_paths)}")
    print(f"worker_manifest={args.repair_dir / 'split_worker_manifest.json'}")
    print(f"part_input_dir={split_text_dir}")
    print(f"part_output_dir={split_audio_dir}")
    return 0


def finalize(args: argparse.Namespace) -> int:
    import numpy as np
    import soundfile as sf

    manifest_path = args.repair_dir / "split_repair_manifest.json"
    manifest = read_json(manifest_path)
    silence_seconds = max(0.0, args.silence_ms / 1000.0)

    repaired = 0
    for item in manifest:
        audio_path = Path(item["audio_path"])
        arrays = []
        sample_rate = None
        channel_shape: tuple[int, ...] | None = None
        part_audio_paths = [Path(raw) for raw in item["part_audio_paths"]]
        for part_audio_path in part_audio_paths:
            if not part_audio_path.exists() or part_audio_path.stat().st_size <= 44:
                raise FileNotFoundError(f"Missing split repair audio: {part_audio_path}")
            data, part_sample_rate = sf.read(part_audio_path, dtype="float32", always_2d=False)
            if sample_rate is None:
                sample_rate = part_sample_rate
                channel_shape = data.shape[1:] if data.ndim > 1 else ()
            elif part_sample_rate != sample_rate:
                raise ValueError(
                    f"Sample-rate mismatch for {part_audio_path}: {part_sample_rate} != {sample_rate}"
                )
            if (data.shape[1:] if data.ndim > 1 else ()) != channel_shape:
                raise ValueError(f"Channel mismatch for {part_audio_path}")
            arrays.append(data)
            if silence_seconds:
                silence_frames = int(round(sample_rate * silence_seconds))
                if data.ndim == 1:
                    arrays.append(np.zeros((silence_frames,), dtype="float32"))
                else:
                    arrays.append(np.zeros((silence_frames, *data.shape[1:]), dtype="float32"))

        if arrays and silence_seconds:
            arrays = arrays[:-1]
        if not arrays or sample_rate is None:
            raise RuntimeError(f"No part audio for {audio_path}")

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        joined = np.concatenate(arrays, axis=0)
        sf.write(audio_path, joined, sample_rate, subtype="PCM_16")
        cache_path = cache_path_for_audio(args.asr_cache_dir, audio_path)
        if cache_path.exists():
            cache_path.unlink()
        repaired += 1
        duration = len(joined) / sample_rate
        print(f"repaired {audio_path.name}: parts={len(part_audio_paths)} duration={duration:.2f}s")

    print(f"repaired_count={repaired}")
    return 0


def main() -> int:
    args = parse_args()
    args.repair_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        return prepare(args)
    return finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
