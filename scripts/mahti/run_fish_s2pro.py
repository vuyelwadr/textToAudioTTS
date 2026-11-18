from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_device,
    resolve_external_root,
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
)

DEFAULT_OUTPUT_NAME = "fish_s2pro_voice_clone.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fish Audio S2-Pro zero-shot voice cloning against a short reference clip."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-length", type=int, default=300)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    reference_text = resolve_reference_text(args.reference_audio, args.reference_text, args.reference_text_file)
    device = resolve_device(args.device)
    output_dir = resolve_output_dir(args.output_dir)
    output_path = (args.output or (output_dir / DEFAULT_OUTPUT_NAME)).resolve()
    ensure_parent(output_path)
    reference_audio = args.reference_audio.resolve()

    # S2-Pro is distributed as a checkpoint for fishaudio/fish-speech's current
    # text2semantic + dac inference stack rather than a standalone pip package,
    # so this shells into the upstream CLI script exactly like the rest of the
    # heavy CSC stacks in this folder shell into their own upstream entrypoints.
    fish_root = resolve_external_root() / "fish-speech"
    checkpoint_path = (args.checkpoint_path or (resolve_model_root() / "fish-speech" / "s2-pro")).resolve()
    scratch_output_dir = (output_path.parent / "_fish_s2pro_intermediates").resolve()
    scratch_output_dir.mkdir(parents=True, exist_ok=True)

    inference_script = fish_root / "fish_speech" / "models" / "text2semantic" / "inference.py"

    # The upstream script is launched with cwd=fish_root below, so every path
    # argument must be absolute -- relative paths (as typically passed on the
    # command line, relative to the caller's cwd) would otherwise resolve
    # against fish_root instead and fail with "Path ... does not exist".
    runner_args = [
        str(inference_script),
        "--text",
        text,
        "--prompt-text",
        reference_text,
        "--prompt-audio",
        str(reference_audio),
        "--output",
        str(output_path),
        "--output-dir",
        str(scratch_output_dir),
        "--checkpoint-path",
        str(checkpoint_path),
        "--device",
        device,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--chunk-length",
        str(args.chunk_length),
        "--compile" if args.compile else "--no-compile",
    ]
    if args.half:
        runner_args.append("--half")

    # The pytorch module's torchaudio build does not reliably ship a working
    # audio backend for plain wav decoding on CSC nodes, so swap torchaudio.load
    # for a soundfile-backed shim before the upstream script's own import runs.
    with tempfile.NamedTemporaryFile("w", suffix="_fish_s2pro_wrapper.py", delete=False) as wrapper_file:
        wrapper_file.write(
            "import runpy\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "import soundfile as sf\n"
            "import torch\n"
            "import torchaudio\n"
            "\n"
            "def _soundfile_load(path, *args, **kwargs):\n"
            "    data, sample_rate = sf.read(Path(path), always_2d=True, dtype='float32')\n"
            "    waveform = torch.from_numpy(data.T.copy())\n"
            "    return waveform, sample_rate\n"
            "\n"
            "torchaudio.load = _soundfile_load\n"
            f"sys.argv = {runner_args!r}\n"
            f"runpy.run_path({str(inference_script)!r}, run_name='__main__')\n"
        )
        wrapper_path = Path(wrapper_file.name)

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        subprocess.run([sys.executable, str(wrapper_path)], cwd=str(fish_root), env=env, check=True)
    finally:
        wrapper_path.unlink(missing_ok=True)

    record_run(
        model_id="fish_s2pro_voice_clone",
        output_path=output_path,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={
            "device": device,
            "seed": args.seed,
            "chunk_length": args.chunk_length,
            "checkpoint_path": str(checkpoint_path),
        },
    )


if __name__ == "__main__":
    main()
