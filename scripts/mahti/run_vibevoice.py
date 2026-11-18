"""Run VibeVoice for the shared naturalness prompt.

Model family  : microsoft/VibeVoice-1.5B  (multi-speaker diffusion TTS)
                microsoft/VibeVoice-Realtime-0.5B  (streaming diffusion TTS)
Code source   : vibevoice-community/VibeVoice  (community mirror, MIT)
                https://github.com/vibevoice-community/VibeVoice
Weights       : HuggingFace – microsoft/VibeVoice-{1.5B,Realtime-0.5B}

⚠ CAVEATS
----------
1.  Code availability: Microsoft deleted their official GitHub inference repo
    in September 2025.  Weights remain on HuggingFace.  This runner delegates
    to the community mirror installed by bootstrap_vibevoice.sh.
    Fallback archive: https://github.com/shijincai/VibeVoice

2.  Transformers pin: VibeVoice requires transformers==4.51.3 exactly.
    This conflicts with Higgs Audio (>=4.45.1,<4.47.0); both need isolated venvs.
    Always run this script inside the 'vibevoice' venv activated by
    bootstrap_vibevoice.sh.

3.  Weights license: microsoft/VibeVoice-{1.5B,Realtime-0.5B} model cards
    say "research purpose use only".  The community code is MIT.
    Do not use in commercial or production contexts.

4.  Voice cloning is NOT supported on Realtime-0.5B; it uses preset .pt files
    shipped with the repo (demo/voices/streaming_model/).

Inference path
--------------
Inference is delegated via subprocess to the CLI scripts shipped with the
community repo rather than importing VibeVoice internals directly.  This avoids
having to replicate the non-trivial VibeVoiceForConditionalGenerationInference
loading API and keeps the runner decoupled from internal class layout changes.

  demo/inference_from_file.py            – 1.5B model  (modes: plain_1b, multispeaker)
  demo/streaming_inference_from_file.py  – Realtime-0.5B (mode: realtime)

Input format:
  1.5B modes use speaker-tagged lines for the community demo.
  Realtime uses plain text plus a cached voice prompt (.pt).

Output slots
------------
  plain_1b      → 82_vibevoice_1_5b_plain.wav      (1 speaker, --disable_prefill)
  multispeaker  → 83_vibevoice_1_5b_dialogue.wav   (2 speakers, built-in presets)
  realtime      → 84_vibevoice_realtime_0_5b.wav   (Realtime-0.5B, Mike preset)

Voice presets (Realtime-0.5B): Carter, Davis, Emma, Frank, Grace, Mike, Samuel.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave
from pathlib import Path

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
    resolve_external_root,
    resolve_model_root,
    resolve_output_dir,
    safe_model_dirname,
)

VIBEVOICE_DIR = resolve_external_root() / "vibevoice-community"
_MODEL_ROOT = resolve_model_root() / "vibevoice"

DEFAULT_MODEL_1_5B = "microsoft/VibeVoice-1.5B"
DEFAULT_MODEL_REALTIME = "microsoft/VibeVoice-Realtime-0.5B"

MODE_OUTPUTS: dict[str, str] = {
    "plain_1b": "82_vibevoice_1_5b_plain.wav",
    "multispeaker": "83_vibevoice_1_5b_dialogue.wav",
    "realtime": "84_vibevoice_realtime_0_5b.wav",
}

# Built-in voice preset names used by 1.5B (demo/voices/model_1_5b/).
_SPEAKER_A = "Alice"
_SPEAKER_B = "Frank"
# Preset for Realtime-0.5B (demo/voices/streaming_model/{name}.pt).
# Keep the selected audiobook baseline voice unless the caller overrides it.
_REALTIME_SPEAKER = "en-Mike_man"
_REALTIME_LEADIN_TRIM_SECONDS = 12.5


def _resolve_model_path(repo_id: str) -> str:
    """Return the local pre-downloaded model path, or the HF repo ID as fallback."""
    local = _MODEL_ROOT / safe_model_dirname(repo_id)
    return str(local) if local.exists() else repo_id


def _resolve_attn_implementation() -> str:
    try:
        import torch
    except ImportError:
        return "sdpa"

    if not torch.cuda.is_available():
        return "sdpa"

    major, _minor = torch.cuda.get_device_capability()
    return "flash_attention_2" if major >= 8 else "sdpa"


def _resolve_realtime_trim_seconds(source_path: Path, override: float | None) -> float:
    if override is not None:
        return max(0.0, override)

    if source_path.name.startswith("chunk_0000_") or source_path.stem == "chunk_0000":
        return _REALTIME_LEADIN_TRIM_SECONDS

    return 0.0


def _trim_wav_prefix(path: Path, trim_seconds: float) -> None:
    if trim_seconds <= 0:
        return

    with wave.open(str(path), "rb") as reader:
        params = reader.getparams()
        trim_frames = int(round(trim_seconds * params.framerate))
        if trim_frames <= 0:
            return
        if trim_frames >= params.nframes:
            print(
                f"Warning: trim request ({trim_seconds:.2f}s) exceeds audio duration; skipping trim"
            )
            return
        raw_audio = reader.readframes(params.nframes)

    bytes_per_frame = params.nchannels * params.sampwidth
    trimmed_audio = raw_audio[trim_frames * bytes_per_frame :]

    tmp_path = path.with_name(f"{path.stem}.trimmed{path.suffix}")
    with wave.open(str(tmp_path), "wb") as writer:
        writer.setnchannels(params.nchannels)
        writer.setsampwidth(params.sampwidth)
        writer.setframerate(params.framerate)
        writer.setcomptype(params.comptype, params.compname)
        writer.writeframes(trimmed_audio)

    tmp_path.replace(path)
    print(f"Trimmed {trim_seconds:.2f}s from start of {path.name}")


def _run_demo_script(
    script_path: Path, argv: list[str], attn_implementation: str
) -> None:
    patch_code = f"""
import runpy
import sys
from transformers import modeling_utils as _mu

_orig = _mu.PreTrainedModel.from_pretrained.__func__

def _patched(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    attn = kwargs.get("attn_implementation")
    if attn is None or attn == "flash_attention_2":
        kwargs["attn_implementation"] = {attn_implementation!r}
    return _orig(cls, pretrained_model_name_or_path, *model_args, **kwargs)

_mu.PreTrainedModel.from_pretrained = classmethod(_patched)
sys.argv = {([str(script_path)] + argv)!r}
runpy.run_path({str(script_path)!r}, run_name="__main__")
"""
    subprocess.run(
        [sys.executable, "-c", patch_code], cwd=str(VIBEVOICE_DIR), check=True
    )


def _sanitize_artifact_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VibeVoice (1.5B and Realtime-0.5B) for the shared naturalness prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_OUTPUTS),
        default="plain_1b",
        help=(
            "plain_1b: single-speaker narration (slot 82, --disable_prefill); "
            "multispeaker: two-speaker dialogue (slot 83, built-in presets); "
            "realtime: Realtime-0.5B model with Mike preset (slot 84)."
        ),
    )
    parser.add_argument(
        "--model-1-5b",
        default=DEFAULT_MODEL_1_5B,
        help="HF repo ID or local path for the 1.5B model.",
    )
    parser.add_argument(
        "--model-realtime",
        default=DEFAULT_MODEL_REALTIME,
        help="HF repo ID or local path for the Realtime-0.5B model.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help=(
            "Diffusion steps for the 1.5B model.  Defaults to the model's built-in "
            "value (~50).  Use 6 for ~2× speedup with a small quality drop."
        ),
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.7,
        help="CFG guidance scale for Realtime-0.5B.",
    )
    parser.add_argument(
        "--ddpm-steps",
        type=int,
        default=8,
        help="DDPM inference steps for Realtime-0.5B.",
    )
    parser.add_argument(
        "--speaker-name",
        default=_REALTIME_SPEAKER,
        help="Realtime preset to use from demo/voices/streaming_model/.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--trim-leading-seconds",
        type=float,
        default=None,
        help=(
            "Trim this many seconds from the start of realtime outputs after generation. "
            "If omitted, chunk_0000_* inputs get the baseline lead-in trim automatically."
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _write_input_file(text: str, mode: str, scratch_dir: Path, artifact_stem: str) -> Path:
    """Write the prompt in VibeVoice's speaker-tagged line format.

    For multispeaker mode the text is split roughly in half at a sentence
    boundary so both voices produce coherent segments. Realtime expects plain
    text and a cached prompt, so we do not add a speaker tag there.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    txt_path = scratch_dir / f"{artifact_stem}.txt"

    if mode == "multispeaker":
        sentences = [s.strip() for s in text.replace(".", ".|").split("|") if s.strip()]
        mid = max(1, len(sentences) // 2)
        part_a = " ".join(sentences[:mid]).strip()
        part_b = (
            " ".join(sentences[mid:]).strip() or part_a
        )  # guard for very short text
        content = f"Speaker 1: {part_a}\nSpeaker 2: {part_b}\n"
    elif mode == "realtime":
        content = f"{text}\n"
    else:
        # The parser still expects a speaker tag in single-speaker mode.
        content = f"Speaker 1: {text}\n"

    txt_path.write_text(content, encoding="utf-8")
    return txt_path


def _collect_output(
    scratch_out: Path,
    txt_path: Path,
    output_path: Path,
    *,
    suffix: str = "_generated.wav",
) -> None:
    """Move the inferred output filename to our canonical output path."""
    inferred = scratch_out / f"{txt_path.stem}{suffix}"
    if not inferred.exists():
        raise FileNotFoundError(
            f"Expected output not found at {inferred}. "
            "Check that the VibeVoice demo script completed without errors."
        )
    inferred.rename(output_path)


def _run_1_5b(
    args: argparse.Namespace,
    txt_path: Path,
    output_path: Path,
    scratch_out: Path,
    attn_implementation: str,
) -> None:
    """Call demo/inference_from_file.py as a subprocess."""
    model_path = _resolve_model_path(args.model_1_5b)

    argv = [
        "--model_path",
        model_path,
        "--txt_path",
        str(txt_path),
        "--output_dir",
        str(scratch_out),
    ]

    if args.mode == "plain_1b":
        # --disable_prefill skips voice conditioning (no reference audio required).
        argv += ["--speaker_names", _SPEAKER_A, "--disable_prefill"]
    else:
        argv += ["--speaker_names", _SPEAKER_A, _SPEAKER_B]

    if args.num_steps is not None:
        argv += ["--num_steps", str(args.num_steps)]

    _run_demo_script(
        VIBEVOICE_DIR / "demo" / "inference_from_file.py",
        argv,
        attn_implementation,
    )
    _collect_output(scratch_out, txt_path, output_path, suffix="_generated.wav")


def _run_realtime(
    args: argparse.Namespace,
    txt_path: Path,
    output_path: Path,
    scratch_out: Path,
    attn_implementation: str,
) -> None:
    """Call demo/streaming_inference_from_file.py as a subprocess."""
    model_path = _resolve_model_path(args.model_realtime)

    argv = [
        "--model_path",
        model_path,
        "--txt_path",
        str(txt_path),
        "--speaker_name",
        args.speaker_name,
        "--cfg_scale",
        str(args.cfg_scale),
        "--ddpm_steps",
        str(args.ddpm_steps),
        "--output_dir",
        str(scratch_out),
    ]

    _run_demo_script(
        VIBEVOICE_DIR / "demo" / "streaming_inference_from_file.py",
        argv,
        attn_implementation,
    )
    _collect_output(
        scratch_out,
        txt_path,
        output_path,
        suffix="_streaming_generated.wav",
    )
    trim_seconds = _resolve_realtime_trim_seconds(args.input, args.trim_leading_seconds)
    _trim_wav_prefix(output_path, trim_seconds)


def main() -> None:
    args = parse_args()

    text = normalize_target_text(load_text(args.input))

    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.mode])
    ensure_parent(output_path)

    # Scratch space for intermediate text inputs and raw demo outputs.
    cache_root = Path(os.environ.get("MAHTI_CACHE_ROOT", str(VIBEVOICE_DIR / ".cache")))
    scratch_dir = cache_root / "vibevoice" / "inputs"
    scratch_out = cache_root / "vibevoice" / "outputs"
    scratch_out.mkdir(parents=True, exist_ok=True)

    artifact_stem = _sanitize_artifact_component(
        "__".join(
            [
                "vibevoice",
                args.mode,
                args.speaker_name,
                args.input.stem,
                output_path.parent.parent.name if output_path.parent.name == "chunks" else output_path.stem,
                str(os.getpid()),
            ]
        )
    )
    txt_path = _write_input_file(text, args.mode, scratch_dir, artifact_stem)
    attn_implementation = _resolve_attn_implementation()

    if args.mode == "realtime":
        _run_realtime(args, txt_path, output_path, scratch_out, attn_implementation)
        model_id_key = "vibevoice_realtime_0_5b"
        model_used = args.model_realtime
    else:
        _run_1_5b(args, txt_path, output_path, scratch_out, attn_implementation)
        model_id_key = f"vibevoice_1_5b_{args.mode}"
        model_used = args.model_1_5b

    record_run(
        model_id=model_id_key,
        output_path=output_path,
        input_path=args.input,
        reference_audio=None,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "model": model_used,
            "mode": args.mode,
            "attn_implementation": attn_implementation,
            "num_steps": args.num_steps,
            "cfg_scale": args.cfg_scale if args.mode == "realtime" else None,
            "ddpm_steps": args.ddpm_steps if args.mode == "realtime" else None,
        },
    )


if __name__ == "__main__":
    main()
