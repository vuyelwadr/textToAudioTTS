"""Run Step-Audio-EditX for the shared naturalness prompt.

Model family : stepfun-ai/Step-Audio-EditX (3B, Apache-2.0)
Code source  : https://github.com/stepfun-ai/Step-Audio-EditX
Weights      : HuggingFace - stepfun-ai/Step-Audio-EditX, stepfun-ai/Step-Audio-Tokenizer

Step-Audio-EditX's own inference code (tts.py / model_loader.py) imports vLLM
unconditionally and is only exposed as the standalone `tts_infer.py` CLI
script shipped in the cloned repo -- there is no pip-installed package to
import directly. Rather than re-implementing StepAudioTokenizer/StepAudioTTS
wiring here (and risking drift from upstream's own loading logic), this
runner delegates to that CLI via subprocess, the same approach already used
for VibeVoice in run_vibevoice.py.

Notable features (see bootstrap_step_audio_editx.sh for install notes):
  - Zero-shot cloning from ~3-30s of reference audio (--edit-type clone).
  - Iterative emotion/style/paralinguistic editing on top of a first clone
    pass (--edit-type emotion|style|paralinguistic|speed|denoise|vad).
  - Paralinguistic tags (insert directly into --target-text for clone, or
    into the edited text for --edit-type paralinguistic), e.g.:
        [breath] [sigh] [laugh] [chuckle] [inhale] [exhale] [cough]
        [uhm] [clears throat] [snort] [giggle] [Confirmation-en]
        [Surprise-oh] [Surprise-ah] [Surprise-wa] [Surprise-yo]
        [Dissatisfaction-hnn] [Question-ei|ah|en|yi|oh]
    These are plausible candidates for natural paragraph-boundary pacing
    (e.g. a [breath] tag at a paragraph break) but have only been smoke
    tested for the plain "clone" path so far -- verify paralinguistic-tag
    placement quality separately before wiring into the chunking pipeline.

Output slot: 30_step_audio_editx_voice_clone.wav
"""

from __future__ import annotations

import argparse
import os
import site
import subprocess
import sys
import time
from pathlib import Path

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
    resolve_device,
    resolve_external_root,
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
)

STEP_AUDIO_EDITX_DIR = resolve_external_root() / "Step-Audio-EditX"
_MODEL_ROOT = resolve_model_root() / "step-audio-editx"
DEFAULT_MODEL_PATH = _MODEL_ROOT / "Step-Audio-EditX"
DEFAULT_TOKENIZER_PATH = _MODEL_ROOT / "Step-Audio-Tokenizer"

EDIT_TYPES = ("clone", "emotion", "style", "vad", "denoise", "paralinguistic", "speed")
DEFAULT_OUTPUT_NAME = "30_step_audio_editx_voice_clone.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step-Audio-EditX for zero-shot voice cloning / iterative audio editing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument(
        "--edit-type",
        choices=EDIT_TYPES,
        default="clone",
        help="clone: zero-shot TTS from --input text. Others edit --reference-audio in place.",
    )
    parser.add_argument(
        "--edit-info",
        default="",
        help="Sub-type for emotion/style/speed edits, e.g. happy, whisper, 'more faster'.",
    )
    parser.add_argument(
        "--target-text",
        default=None,
        help="Override text for clone/paralinguistic edits. Defaults to --input's contents.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--repo-dir", type=Path, default=STEP_AUDIO_EDITX_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--quantization", choices=("awq", "gptq", "fp8"), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _apptainer_env_with_cuda13_libs() -> dict[str, str]:
    """Patch in the nvidia-cuda-runtime/-cublas/-cudnn-cu13/etc. lib dirs
    bootstrap_step_audio_editx.sh installs for onnxruntime-gpu==1.27.0
    (pinned by upstream's uv.lock), which needs CUDA 13's libcudart.so.13.

    This venv's base interpreter is Mahti's Apptainer-wrapped pytorch/2.9
    module python3 (every venv python3 symlinks back to it). That wrapper
    script (/appl/soft/ai/wrap/pytorch-2.9/bin/python3) has two branches:
    outside any container it runs `singularity exec ... python3`, where
    Apptainer only honors APPTAINERENV_-prefixed vars and drops a plain
    LD_LIBRARY_PATH; but once already inside the container (as this process
    already is, having been launched the same way) it just does a bare
    `exec .../_bin/python3 "$@"` with no more Apptainer involved -- at that
    point a plain LD_LIBRARY_PATH passed via subprocess env *does* take
    effect, and APPTAINERENV_LD_LIBRARY_PATH alone is silently ignored
    (nothing left to interpret that prefix). Set both so this helper works
    whether or not the caller already went through the container boundary.
    """
    lib_dirs: list[str] = []
    seen: set[str] = set()
    for base in site.getsitepackages():
        nvidia_root = Path(base) / "nvidia"
        if not nvidia_root.exists():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                resolved = str(lib_dir.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    lib_dirs.append(resolved)

    env = dict(os.environ)
    if not lib_dirs:
        return env

    # By the time this process is running, the container has already
    # applied its own default LD_LIBRARY_PATH into os.environ -- prepend
    # our extra lib dirs rather than dropping that default entirely.
    existing = env.get("LD_LIBRARY_PATH", "")
    combined = ":".join(lib_dirs + ([existing] if existing else []))
    env["LD_LIBRARY_PATH"] = combined
    env["APPTAINERENV_LD_LIBRARY_PATH"] = combined
    return env


def _resolve_produced_wav(scratch_dir: Path, edit_type: str, edit_info: str) -> Path:
    """Mirror tts_infer.py's own output-naming convention (infer())."""
    if edit_type == "clone":
        candidate = scratch_dir / "test_clone.wav"
    elif edit_info:
        candidate = scratch_dir / f"test_{edit_type}_{edit_info}.wav"
    else:
        candidate = scratch_dir / f"test_{edit_type}.wav"

    if candidate.exists():
        return candidate

    matches = sorted(scratch_dir.glob("test_*.wav"))
    if not matches:
        raise FileNotFoundError(f"Step-Audio-EditX did not produce an output wav under {scratch_dir}")
    return matches[0]


def main() -> None:
    args = parse_args()

    # tts_infer.py is invoked with cwd=repo_dir (the external Step-Audio-EditX
    # checkout), not the caller's own cwd -- any relative path the caller
    # passed (e.g. --reference-audio generated/foo.wav from a SLURM script
    # running with --chdir on the main repo) would otherwise resolve against
    # the wrong directory and fail with "No such file or directory".
    args.reference_audio = args.reference_audio.resolve()
    if args.reference_text_file is not None:
        args.reference_text_file = args.reference_text_file.resolve()
    args.input = args.input.resolve()
    args.model_path = args.model_path.resolve()
    args.tokenizer_path = args.tokenizer_path.resolve()
    args.repo_dir = args.repo_dir.resolve()

    device = resolve_device(args.device)
    if device != "cuda":
        raise RuntimeError("Step-Audio-EditX requires a CUDA device")

    text = normalize_target_text(load_text(args.input))
    target_text = args.target_text if args.target_text is not None else text

    reference_text = resolve_reference_text(
        args.reference_audio, args.reference_text, args.reference_text_file
    )

    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / DEFAULT_OUTPUT_NAME)
    ensure_parent(output_path)

    repo_dir = args.repo_dir
    infer_script = repo_dir / "tts_infer.py"
    if not infer_script.exists():
        raise FileNotFoundError(
            f"tts_infer.py not found under {repo_dir}; run bootstrap_step_audio_editx.sh first"
        )

    scratch_output_dir = output_path.parent / f".step_audio_editx_scratch_{output_path.stem}"
    scratch_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(infer_script),
        "--model-path", str(args.model_path),
        "--model-source", "local",
        "--tokenizer-path", str(args.tokenizer_path),
        "--prompt-text", reference_text,
        "--prompt-audio", str(args.reference_audio),
        "--edit-type", args.edit_type,
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--output-dir", str(scratch_output_dir),
    ]
    if args.edit_type in {"clone", "paralinguistic"}:
        cmd += ["--generated-text", target_text]
    if args.edit_info:
        cmd += ["--edit-info", args.edit_info]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.enforce_eager:
        cmd.append("--enforce-eager")

    print("Running:", " ".join(cmd))
    start = time.monotonic()
    subprocess.run(cmd, cwd=str(repo_dir), check=True, env=_apptainer_env_with_cuda13_libs())
    elapsed = time.monotonic() - start

    produced_path = _resolve_produced_wav(scratch_output_dir, args.edit_type, args.edit_info)
    produced_path.replace(output_path)

    audio_info = sf.info(str(output_path))
    audio_duration = audio_info.duration
    rtf = (elapsed / audio_duration) if audio_duration else None

    print(
        f"[Saved] {output_path} "
        f"({audio_duration:.2f}s audio, {elapsed:.1f}s wall"
        + (f", RTF={rtf:.2f}" if rtf else "")
        + ")"
    )

    record_run(
        model_id=f"step_audio_editx_{args.edit_type}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=args.reference_audio,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "device": device,
            "dtype": args.dtype,
            "edit_type": args.edit_type,
            "edit_info": args.edit_info or None,
            "model_path": str(args.model_path),
            "generation_seconds": round(elapsed, 2),
            "audio_seconds": round(audio_duration, 2) if audio_duration else None,
            "rtf": round(rtf, 3) if rtf else None,
        },
    )


if __name__ == "__main__":
    main()
