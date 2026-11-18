from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNK_DIR = (
    REPO_ROOT / "generated" / "prometheus_benchmark_smoke_20260430" / "PrometheusAtlas" / "chunks"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "generated" / "prometheus_benchmark_candidates_20260501" / "mimo_audio_multichunk"
)
DEFAULT_REFERENCE_AUDIO = (
    REPO_ROOT / "generated" / "clone_references_2026-04-25" / "brothersKarmatzov.clone_ref_60s.wav"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-chunk Prometheus benchmark samples with MiMo-Audio."
    )
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--max-chunks", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["plain", "clone"])
    return parser.parse_args()


def discover_chunks(chunk_dir: Path, max_chunks: int) -> list[Path]:
    if max_chunks < 1:
        raise ValueError("--max-chunks must be at least 1")
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"chunk dir not found: {chunk_dir}")
    chunks = sorted(chunk_dir.glob("chunk_*.txt"))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.txt files found in {chunk_dir}")
    return chunks[:max_chunks]


def is_valid_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb"):
            return True
    except wave.Error:
        return False


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


def run_mode(
    *,
    mode: str,
    chunks: list[Path],
    output_root: Path,
    reference_audio: Path,
) -> dict[str, object]:
    mode_root = output_root / mode
    chunk_output_dir = mode_root / "chunks"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chunks[0].parent / "PrometheusAtlas_manifest.json"
    run_script = REPO_ROOT / "scripts" / "mahti" / "run_mimo_tts.py"
    chunk_outputs: list[Path] = []

    for chunk_path in chunks:
        output_path = chunk_output_dir / f"{chunk_path.stem}_{mode}.wav"
        cmd = [
            sys.executable,
            str(run_script),
            "--input",
            str(chunk_path),
            "--generation-mode",
            mode,
            "--reference-audio",
            str(reference_audio),
            "--output",
            str(output_path),
        ]
        if manifest_path.exists():
            cmd.extend(["--manifest", str(manifest_path)])
        subprocess.run(cmd, check=True)
        chunk_outputs.append(output_path)

    final_output = output_root / f"PrometheusAtlas_mimo_audio_{mode}.wav"
    joined = join_wavs(chunk_outputs, final_output)
    return {
        "mode": mode,
        "chunk_count": len(chunk_outputs),
        "chunks": [str(path) for path in chunks],
        "chunk_outputs": [str(path) for path in chunk_outputs],
        "final_output": joined,
    }


def main() -> None:
    args = parse_args()
    chunks = discover_chunks(args.chunk_dir, args.max_chunks)
    if "clone" in args.modes and not args.reference_audio.exists():
        raise FileNotFoundError(f"reference audio not found: {args.reference_audio}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = {
        "chunk_dir": str(args.chunk_dir),
        "reference_audio": str(args.reference_audio),
        "max_chunks": args.max_chunks,
        "modes": [],
    }

    for mode in args.modes:
        if mode not in {"plain", "clone"}:
            raise ValueError(f"unsupported mode: {mode}")
        results["modes"].append(
            run_mode(
                mode=mode,
                chunks=chunks,
                output_root=args.output_root,
                reference_audio=args.reference_audio,
            )
        )

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
