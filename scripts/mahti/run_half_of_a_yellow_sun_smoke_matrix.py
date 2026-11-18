#!/usr/bin/env python3
"""Run the Half of a Yellow Sun two-chunk smoke matrix for Qwen or VibeVoice."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_MANIFEST = (
    REPO_ROOT / "generated" / "audiobooks-half_of_a_yellow_sun-smoke-matrix" / "variant_manifest.json"
)
DEFAULT_SOURCE_DIR = (
    REPO_ROOT
    / "generated"
    / "audiobooks-half_of_a_yellow_sun-smoke-matrix"
    / "source_two_chunks_narrative"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "vibevoice"), required=True)
    parser.add_argument("--matrix-manifest", type=Path, default=DEFAULT_MATRIX_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument(
        "--variant-ids",
        nargs="*",
        default=None,
        help="Optional list of variant ids to run. Defaults to the full matrix.",
    )
    parser.add_argument(
        "--summary-name",
        default="matrix_run_summary.json",
        help="Summary filename written under the output root.",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def run_command(command: list[str], log_path: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, text=True)


def load_matrix_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expand_vibevoice_variants(manifest: dict) -> list[dict]:
    if "vibevoice_speakers" not in manifest or "vibevoice_confs" not in manifest:
        return manifest["vibevoice_variants"]

    variants: list[dict] = []
    for speaker in manifest["vibevoice_speakers"]:
        speaker_id = speaker.lower().replace("-", "_")
        for config in manifest["vibevoice_confs"]:
            variant_id = f"vibe_{speaker_id}_{config['id']}"
            variants.append(
                {
                    "id": variant_id,
                    "description": f"{speaker} / {config['description']}",
                    "args": ["--speaker", speaker, *config["args"]],
                }
            )
    return variants


def qwen_variants(manifest: dict) -> list[dict]:
    return manifest["qwen_variants"]


def vibevoice_variants(manifest: dict) -> list[dict]:
    return expand_vibevoice_variants(manifest)


def select_variants(variants: list[dict], selected_ids: list[str] | None) -> list[dict]:
    if not selected_ids:
        return variants

    selected = [variant for variant in variants if variant["id"] in set(selected_ids)]
    missing = sorted(set(selected_ids) - {variant["id"] for variant in selected})
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Unknown variant ids: {missing_text}")
    return selected


def parse_variant_args(args: list[str]) -> dict[str, str | bool]:
    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument in variant args: {token}")
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            parsed[token] = args[index + 1]
            index += 2
        else:
            parsed[token] = True
            index += 1
    return parsed


def run_qwen(
    *,
    python_exe: str,
    source_dir: Path,
    output_root: Path,
    manifest: dict,
    reference_audio: Path | None,
    reference_text_file: Path | None,
    selected_ids: list[str] | None,
) -> list[dict]:
    runner = REPO_ROOT / "scripts" / "mahti" / "run_qwen3_batch_optimized.py"
    results: list[dict] = []

    for variant in select_variants(qwen_variants(manifest), selected_ids):
        variant_root = output_root / variant["id"]
        chunks_dir = variant_root / "chunks"
        logs_dir = variant_root / "logs"
        variant_root.mkdir(parents=True, exist_ok=True)
        command = [
            python_exe,
            str(runner),
            "--input-dir",
            str(source_dir),
            "--output-dir",
            str(chunks_dir),
            "--prefix",
            "chunk_",
            "--log-dir",
            str(logs_dir),
        ]
        if reference_audio is not None:
            command.extend(["--reference-audio", str(reference_audio)])
        if reference_text_file is not None and not any(arg == "--xvec-only" for arg in variant["args"]):
            command.extend(["--reference-text-file", str(reference_text_file)])
        command.extend(variant["args"])
        run_command(command)
        results.append(
            {
                "id": variant["id"],
                "description": variant["description"],
                "output_dir": str(variant_root),
                "status": "completed",
            }
        )

    return results


def run_vibevoice(
    *,
    python_exe: str,
    source_dir: Path,
    output_root: Path,
    manifest: dict,
    selected_ids: list[str] | None,
) -> list[dict]:
    runner = REPO_ROOT / "scripts" / "mahti" / "run_vibevoice.py"
    results: list[dict] = []
    chunk_paths = sorted(source_dir.glob("chunk_*.txt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.txt files found in {source_dir}")

    for variant in select_variants(vibevoice_variants(manifest), selected_ids):
        variant_root = output_root / variant["id"]
        chunks_dir = variant_root / "chunks"
        logs_dir = variant_root / "logs"
        variant_root.mkdir(parents=True, exist_ok=True)
        variant_args = parse_variant_args(variant["args"])
        speaker_name = variant_args.get("--speaker")
        cfg_scale = variant_args.get("--cfg-scale")
        ddpm_steps = variant_args.get("--ddpm-steps")

        if not isinstance(speaker_name, str) or not isinstance(cfg_scale, str) or not isinstance(ddpm_steps, str):
            raise ValueError(f"Variant {variant['id']} is missing speaker/cfg/ddpm arguments")

        for chunk_path in chunk_paths:
            output_path = chunks_dir / f"{chunk_path.stem}.wav"
            log_path = logs_dir / f"{chunk_path.stem}.log"
            command = [
                python_exe,
                str(runner),
                "--mode",
                "realtime",
                "--input",
                str(chunk_path),
                "--output",
                str(output_path),
                "--speaker-name",
                speaker_name,
                "--cfg-scale",
                cfg_scale,
                "--ddpm-steps",
                ddpm_steps,
            ]
            run_command(command, log_path=log_path)
        results.append(
            {
                "id": variant["id"],
                "description": variant["description"],
                "output_dir": str(variant_root),
                "status": "completed",
            }
        )

    return results


def main() -> int:
    args = parse_args()
    manifest = load_matrix_manifest(args.matrix_manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.model == "qwen":
        results = run_qwen(
            python_exe=args.python,
            source_dir=args.source_dir,
            output_root=args.output_root,
            manifest=manifest,
            reference_audio=args.reference_audio,
            reference_text_file=args.reference_text_file,
            selected_ids=args.variant_ids,
        )
    else:
        results = run_vibevoice(
            python_exe=args.python,
            source_dir=args.source_dir,
            output_root=args.output_root,
            manifest=manifest,
            selected_ids=args.variant_ids,
        )

    summary_path = args.output_root / args.summary_name
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
