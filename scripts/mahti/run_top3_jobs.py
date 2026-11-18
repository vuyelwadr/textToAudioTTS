#!/usr/bin/env python3
"""
Audiobook TOP 3 Sample Generator
Run via: sbatch run_top3_jobs.sh

This Python script handles the TOP 3 models directly without shell script deps.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/scratch/project_2006194/textToAudioTTS")
INPUT_DIR = REPO_ROOT / "experiments/tts_naturalness/inputs"
OUTPUT_DIR = REPO_ROOT / "experiments/tts_naturalness/generated/all_samples"

INPUT_FILES = [
    "naturalness_prompt.txt",
    "description_test.txt",
    "dialogue_test.txt",
    "long_audiobook_test.txt",
    "descriptive_test_long.txt",
    "dialogue_test_long.txt",
    "narrative_test_long.txt",
]

MODELS = [
    {
        "name": "qwen3_faster",
        "script": "run_faster_qwen3_tts.py",
        "args": ["--generation-mode", "voice_clone"],
    },
    {
        "name": "voxcpm2_ultimate",
        "script": "run_voxcpm2.py",
        "args": ["--mode", "ultimate_clone"],
    },
    {
        "name": "vibevoice_realtime",
        "script": "run_vibevoice.py",
        "args": ["--mode", "realtime"],
    },
]


def run_job(
    model_script: str, input_file: str, output_name: str, extra_args: list
) -> str:
    """Submit a single SLURM job."""
    input_path = INPUT_DIR / input_file
    output_path = OUTPUT_DIR / output_name

    cmd = [
        "sbatch",
        "-J",
        f"{model_script.replace('.py', '')}_{input_file.replace('.txt', '')}",
        "--time=02:00:00",
        "--gres=gpu:1",
        model_script,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ] + extra_args

    print(f"Submitting: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=Path(__file__).parent, capture_output=True, text=True
    )
    return result.stdout


def main():
    print(f"=== Generating TOP 3 models with {len(INPUT_FILES)} inputs ===")
    print(f"Repo: {REPO_ROOT}")
    print(f"Inputs: {INPUT_FILES}")
    print()

    for model in MODELS:
        script = model["script"]
        model_name = model["name"]
        extra_args = model["args"]

        for input_file in INPUT_FILES:
            output_name = f"{model_name}_{Path(input_file).stem}.wav"

            print(f"--> {model_name} + {input_file}")
            run_job(script, input_file, output_name, extra_args)
            print(f"    submitted: {output_name}")

    print(f"\n=== All jobs submitted ===")
    print("Check with: sacct -u $USER")


if __name__ == "__main__":
    main()
