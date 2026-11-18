#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()

TARGETS = {
    "venv": REPO_ROOT / ".venv-tts-experiments",
    "experiment-runs": REPO_ROOT / "experiments" / "tts_naturalness" / "generated" / "runs",
    "chatterbox-hf": HOME / ".cache" / "huggingface" / "hub" / "models--ResembleAI--chatterbox",
    "kokoro-hf": HOME / ".cache" / "huggingface" / "hub" / "models--hexgrad--Kokoro-82M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean up local disk usage from the TTS experiment workflow")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGETS),
        default=[],
        help="Named targets to remove",
    )
    parser.add_argument("--execute", action="store_true", help="Actually delete files instead of showing a dry run")
    return parser.parse_args()


def describe_path(path: Path) -> str:
    if not path.exists():
        return f"missing: {path}"
    if path.is_file():
        size = path.stat().st_size
        return f"file: {path} ({size:,} bytes)"
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return f"dir: {path} ({total:,} bytes)"


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    args = parse_args()
    if not args.targets:
        print("Available cleanup targets:")
        for name, path in TARGETS.items():
            print(f"- {name}: {describe_path(path)}")
        print("\nAdd --targets ... to select paths, and --execute to actually delete them.")
        return 0

    print("Selected cleanup targets:")
    for name in args.targets:
        print(f"- {name}: {describe_path(TARGETS[name])}")

    if not args.execute:
        print("\nDry run only. Re-run with --execute to delete these targets.")
        return 0

    for name in args.targets:
        path = TARGETS[name]
        remove_path(path)
        print(f"removed: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())