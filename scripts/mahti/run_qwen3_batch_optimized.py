#!/usr/bin/env python3
"""
Optimized Qwen3-TTS batch processor.

The Faster Qwen wrapper still exposes single-text generation calls, so higher
throughput comes from sharding chunks across a small number of worker processes
that each keep one CUDA-graphed model instance resident on the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import soundfile as sf
import torch

from common import (
    resolve_attn_implementation,
    resolve_device,
    resolve_model_root,
    resolve_reference_text,
    resolve_visible_cuda_devices,
    safe_model_dirname,
)

os.environ["PATH"] = "/usr/bin:/bin:/usr/local/bin:" + os.environ.get("PATH", "")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

FASTER_QWEN3_TTS_DIR = (
    Path(os.environ.get("MAHTI_EXTERNAL_ROOT", "")) / "faster-qwen3-tts"
)
sys.path.insert(0, str(FASTER_QWEN3_TTS_DIR))

try:
    from faster_qwen3_tts import FasterQwen3TTS
except ImportError as e:
    print(f"Failed to import FasterQwen3TTS: {e}")
    print(f"FASTER_QWEN3_TTS_DIR: {FASTER_QWEN3_TTS_DIR}")
    sys.exit(1)


DEFAULT_MODELS = {
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
MIN_VALID_WAV_BYTES = 1000


def resolve_model_name(mode: str, explicit: str | None) -> str:
    model_name = explicit or DEFAULT_MODELS[mode]
    local_dir = resolve_model_root() / "qwen3-tts" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def resolve_default_reference_audio() -> Path:
    work_root = Path(os.environ.get("MAHTI_WORK_ROOT", Path(__file__).resolve().parents[2]))
    return work_root / "outputs" / "video_downloads" / "haCxgJSFmFc" / "haCxgJSFmFc_reference_8s.wav"


def configure_cuda_runtime(device: str) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        return

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def is_valid_output(path: Path) -> bool:
    return path.exists() and path.stat().st_size > MIN_VALID_WAV_BYTES


def discover_txt_files(
    input_dir: Path, prefix: str, worker_manifest: Path | None
) -> list[Path]:
    if worker_manifest is None:
        return sorted(input_dir.glob(f"{prefix}*.txt"))

    payload = json.loads(worker_manifest.read_text(encoding="utf-8"))
    return [Path(raw_path) for raw_path in payload]


def partition_txt_files(txt_files: list[Path], workers: int) -> list[list[Path]]:
    buckets: list[list[Path]] = [[] for _ in range(workers)]
    weights = [0] * workers

    for txt_file in sorted(
        txt_files,
        key=lambda path: (path.stat().st_size, path.name),
        reverse=True,
    ):
        bucket_index = min(range(workers), key=lambda idx: weights[idx])
        buckets[bucket_index].append(txt_file)
        weights[bucket_index] += txt_file.stat().st_size

    return [bucket for bucket in buckets if bucket]


def resolve_worker_count(requested: str, pending_count: int, device: str) -> int:
    if pending_count <= 1:
        return pending_count

    if requested != "auto":
        return max(1, min(int(requested), pending_count))

    if device != "cuda" or not torch.cuda.is_available():
        return 1

    total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_memory_gib >= 35:
        return min(2, pending_count)
    return 1


def spawn_multi_gpu_workers(
    *,
    args: argparse.Namespace,
    model_path: str,
    resolved_attn: str,
    txt_files: list[Path],
    pending_files: list[Path],
    output_dir: Path,
) -> int:
    visible_devices = resolve_visible_cuda_devices()
    gpu_partitions = partition_txt_files(
        pending_files, min(len(visible_devices), len(pending_files))
    )
    log_dir = Path(args.log_dir) if args.log_dir else output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = log_dir / ".gpu_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    assigned_devices = visible_devices[: len(gpu_partitions)]
    print(
        f"Launching {len(gpu_partitions)} Faster Qwen GPU shards across visible devices: "
        f"{', '.join(assigned_devices)}"
    )

    processes: list[tuple[str, subprocess.Popen[bytes]]] = []
    script_path = Path(__file__).resolve()
    for gpu_index, (device_id, gpu_files) in enumerate(
        zip(assigned_devices, gpu_partitions)
    ):
        manifest_path = manifest_dir / f"gpu_{gpu_index:02d}.json"
        manifest_path.write_text(
            json.dumps([str(path) for path in gpu_files], indent=2),
            encoding="utf-8",
        )
        gpu_log_dir = log_dir / f"gpu_{gpu_index:02d}"
        gpu_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"  gpu {device_id}: {len(gpu_files)} pending chunks")

        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--model",
            model_path,
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir),
            "--prefix",
            args.prefix,
            "--generation-mode",
            args.generation_mode,
            "--language",
            args.language,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-seq-len",
            str(args.max_seq_len),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--attn-implementation",
            resolved_attn,
            "--dtype",
            args.dtype,
            "--workers",
            args.workers,
            "--worker-manifest",
            str(manifest_path),
            "--worker-index",
            str(gpu_index),
            "--log-dir",
            str(gpu_log_dir),
        ]
        if args.instruct is not None:
            cmd.extend(["--instruct", args.instruct])
        if args.speaker is not None:
            cmd.extend(["--speaker", args.speaker])
        if args.reference_audio is not None:
            cmd.extend(["--reference-audio", str(args.reference_audio)])
        if args.reference_text is not None:
            cmd.extend(["--reference-text", args.reference_text])
        if args.reference_text_file is not None:
            cmd.extend(["--reference-text-file", str(args.reference_text_file)])
        if args.xvec_only:
            cmd.append("--xvec-only")
        cmd.append("--do-sample" if args.do_sample else "--no-do-sample")
        if args.non_streaming_mode:
            cmd.append("--non-streaming-mode")
        if args.resume:
            cmd.append("--resume")

        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = device_id
        worker_env["OMP_NUM_THREADS"] = "1"
        worker_env["MKL_NUM_THREADS"] = "1"
        worker_env["PYTHONUNBUFFERED"] = "1"
        processes.append((device_id, subprocess.Popen(cmd, env=worker_env)))

    failures = 0
    for device_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures += 1
            print(f"gpu {device_id} shard failed with exit code {return_code}")

    completed_count = sum(
        1 for txt_file in txt_files if is_valid_output(output_dir / f"{txt_file.stem}.wav")
    )
    if failures:
        return -failures
    return completed_count


def spawn_workers(
    *,
    args: argparse.Namespace,
    model_path: str,
    resolved_attn: str,
    txt_files: list[Path],
    output_dir: Path,
) -> int:
    pending_files = [
        txt_file for txt_file in txt_files if not is_valid_output(output_dir / f"{txt_file.stem}.wav")
    ]
    if not pending_files:
        print("All chunk outputs already exist; nothing to do.")
        return len(txt_files)

    visible_devices = resolve_visible_cuda_devices()
    if (
        args.worker_manifest is None
        and args.workers != "1"
        and len(visible_devices) > 1
        and len(pending_files) > 1
    ):
        return spawn_multi_gpu_workers(
            args=args,
            model_path=model_path,
            resolved_attn=resolved_attn,
            txt_files=txt_files,
            pending_files=pending_files,
            output_dir=output_dir,
        )

    worker_count = resolve_worker_count(args.workers, len(pending_files), "cuda")
    if worker_count <= 1:
        summary = process_chunk_batch(
            model_path=model_path,
            txt_files=txt_files,
            output_dir=output_dir,
            generation_mode=args.generation_mode,
            language=args.language,
            instruct=args.instruct,
            speaker=args.speaker,
            use_xvec=args.xvec_only,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            non_streaming_mode=args.non_streaming_mode,
            reference_audio=args.reference_audio,
            reference_text=args.reference_text,
            reference_text_file=args.reference_text_file,
            attn_implementation=resolved_attn,
            dtype=args.dtype,
            worker_index=args.worker_index,
        )
        if summary["failed"]:
            return -summary["failed"]
        return summary["completed"]

    log_dir = Path(args.log_dir) if args.log_dir else output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = log_dir / ".worker_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    partitions = partition_txt_files(pending_files, worker_count)
    print(
        f"Launching {len(partitions)} Faster Qwen worker processes on one GPU "
        f"for {len(pending_files)} pending chunks"
    )

    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    script_path = Path(__file__).resolve()
    for worker_index, worker_files in enumerate(partitions):
        manifest_path = manifest_dir / f"worker_{worker_index:02d}.json"
        manifest_path.write_text(
            json.dumps([str(path) for path in worker_files], indent=2),
            encoding="utf-8",
        )
        worker_log = log_dir / f"worker_{worker_index:02d}.log"
        print(
            f"  worker {worker_index}: {len(worker_files)} chunks -> {worker_log}"
        )

        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--model",
            model_path,
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir),
            "--prefix",
            args.prefix,
            "--generation-mode",
            args.generation_mode,
            "--language",
            args.language,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-seq-len",
            str(args.max_seq_len),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--attn-implementation",
            resolved_attn,
            "--dtype",
            args.dtype,
            "--workers",
            "1",
            "--worker-manifest",
            str(manifest_path),
            "--worker-index",
            str(worker_index),
            "--log-dir",
            str(log_dir),
        ]
        if args.instruct is not None:
            cmd.extend(["--instruct", args.instruct])
        if args.speaker is not None:
            cmd.extend(["--speaker", args.speaker])
        if args.reference_audio is not None:
            cmd.extend(["--reference-audio", str(args.reference_audio)])
        if args.reference_text is not None:
            cmd.extend(["--reference-text", args.reference_text])
        if args.reference_text_file is not None:
            cmd.extend(["--reference-text-file", str(args.reference_text_file)])
        if args.xvec_only:
            cmd.append("--xvec-only")
        cmd.append("--do-sample" if args.do_sample else "--no-do-sample")
        if args.non_streaming_mode:
            cmd.append("--non-streaming-mode")
        if args.resume:
            cmd.append("--resume")

        worker_env = os.environ.copy()
        worker_env["OMP_NUM_THREADS"] = "1"
        worker_env["MKL_NUM_THREADS"] = "1"
        worker_env["PYTHONUNBUFFERED"] = "1"
        log_handle = worker_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=worker_env,
        )
        processes.append((worker_index, process, log_handle))

    failures = 0
    for worker_index, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures += 1
            print(f"worker {worker_index} failed with exit code {return_code}")

    completed_count = sum(
        1 for txt_file in txt_files if is_valid_output(output_dir / f"{txt_file.stem}.wav")
    )
    if failures:
        return -failures
    return completed_count


def process_chunk_batch(
    *,
    model_path: str,
    txt_files: list[Path],
    output_dir: Path,
    generation_mode: str,
    language: str,
    instruct: str | None,
    speaker: str | None,
    use_xvec: bool,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
    non_streaming_mode: bool,
    reference_audio: Path | None,
    reference_text: str | None,
    reference_text_file: Path | None,
    attn_implementation: str,
    dtype: str,
    worker_index: int = -1,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(txt_files)
    label = f"worker {worker_index}" if worker_index >= 0 else "main"

    print(f"[{label}] Found {total} text files to process")
    if total == 0:
        return {"completed": 0, "failed": 0}

    configure_cuda_runtime("cuda")
    if resolve_device("cuda") != "cuda":
        raise RuntimeError("Faster Qwen3-TTS requires CUDA")

    resolved_dtype = getattr(torch, dtype)
    print(f"[{label}] Loading Qwen3-TTS model from: {model_path}")
    print(f"[{label}] Generation mode: {generation_mode}")
    print(f"[{label}] X-Vector only: {use_xvec}")
    print(f"[{label}] Sampling enabled: {do_sample}")
    print(f"[{label}] dtype: {resolved_dtype}")
    print(f"[{label}] attn_implementation: {attn_implementation}")
    model = FasterQwen3TTS.from_pretrained(
        model_path,
        device="cuda",
        dtype=resolved_dtype,
        attn_implementation=attn_implementation,
        max_seq_len=max_seq_len,
    )

    reference_audio_path = str(reference_audio or resolve_default_reference_audio())
    if generation_mode == "clone" and not os.path.exists(reference_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

    resolved_reference_text = None
    if generation_mode == "clone" and not use_xvec:
        resolved_reference_text = resolve_reference_text(
            Path(reference_audio_path),
            reference_text,
            reference_text_file,
        )

    voice_clone_prompt = None
    if generation_mode == "clone":
        print(f"[{label}] Creating reusable voice clone prompt...")
        voice_clone_prompt = model.model.create_voice_clone_prompt(
            ref_audio=reference_audio_path,
            ref_text=resolved_reference_text or "",
            x_vector_only_mode=use_xvec,
        )

    completed = 0
    failed = 0
    for index, txt_file in enumerate(txt_files, start=1):
        wav_file = output_dir / f"{txt_file.stem}.wav"
        if is_valid_output(wav_file):
            print(f"[{label}] [{index}/{total}] Skip {txt_file.name}: already exists")
            completed += 1
            continue

        try:
            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                print(f"[{label}] [{index}/{total}] Warning: Empty text in {txt_file.name}")
                failed += 1
                continue

            print(f"[{label}] [{index}/{total}] Processing {txt_file.name}...")
            start_time = time.time()
            generate_kwargs = {
                "language": language,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
                "non_streaming_mode": non_streaming_mode,
            }

            with torch.inference_mode():
                if generation_mode == "design":
                    wavs, sample_rate = model.generate_voice_design(
                        text=text,
                        instruct=instruct
                        or "Warm, steady audiobook narrator with clear diction and restrained expressiveness.",
                        **generate_kwargs,
                    )
                elif generation_mode == "custom":
                    active_speaker = speaker
                    if not active_speaker:
                        speakers = list(model.model.get_supported_speakers())
                        active_speaker = speakers[0] if speakers else "unknown"
                    wavs, sample_rate = model.generate_custom_voice(
                        text=text,
                        speaker=active_speaker,
                        instruct=instruct,
                        **generate_kwargs,
                    )
                else:
                    wavs, sample_rate = model.generate_voice_clone(
                        text=text,
                        voice_clone_prompt=voice_clone_prompt,
                        instruct=instruct or None,
                        **generate_kwargs,
                    )
            torch.cuda.synchronize()
            generation_time = time.time() - start_time

            audio = wavs[0]
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            sf.write(str(wav_file), audio, sample_rate)

            if is_valid_output(wav_file):
                audio_duration = len(audio) / sample_rate if sample_rate else 0.0
                rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")
                print(f"[{label}]   ✓ Saved {wav_file.name} ({wav_file.stat().st_size} bytes)")
                print(
                    f"[{label}]     generation: {generation_time:.2f}s | "
                    f"audio: {audio_duration:.2f}s | RTF: {rtf:.2f}x"
                )
                completed += 1
            else:
                raise RuntimeError(f"Failed to save valid audio for {txt_file.name}")
        except Exception as exc:
            failed += 1
            print(f"[{label}]   ✗ Error processing {txt_file.name}: {exc}")
            traceback.print_exc()

    print(f"[{label}] Batch complete: completed={completed}, failed={failed}")
    return {"completed": completed, "failed": failed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized Qwen3-TTS batch processing")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to Qwen3-TTS model",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=None,
        help="Reference audio used for clone mode",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing chunk_*.txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output wav files",
    )
    parser.add_argument(
        "--prefix", type=str, default="chunk_", help="Prefix for chunk files"
    )
    parser.add_argument(
        "--generation-mode",
        type=str,
        default="clone",
        choices=["clone", "custom", "design"],
        help="Generation mode",
    )
    parser.add_argument(
        "--language", type=str, default="English", help="Language for generation"
    )
    parser.add_argument(
        "--instruct", type=str, default=None, help="Instruction for voice design mode"
    )
    parser.add_argument(
        "--speaker", type=str, default=None, help="Speaker for custom voice mode"
    )
    parser.add_argument(
        "--reference-text",
        type=str,
        default=None,
        help="Reference transcript used for clone mode",
    )
    parser.add_argument(
        "--reference-text-file",
        type=Path,
        default=None,
        help="Reference transcript file used for clone mode",
    )
    parser.add_argument(
        "--xvec-only",
        action="store_true",
        help="Use x-vector only for cloning (vs ICL)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Static cache sequence budget for the Faster Qwen talker graph",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.88, help="Sampling temperature"
    )
    parser.add_argument("--top-k", type=int, default=55, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic sampling during generation",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.05, help="Repetition penalty"
    )
    parser.add_argument(
        "--non-streaming-mode", action="store_true", help="Use non-streaming mode"
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Attention implementation override",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Torch dtype for model loading",
    )
    parser.add_argument(
        "--workers",
        type=str,
        default="auto",
        help="Concurrent worker count per GPU, or 'auto'",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-worker logs",
    )
    parser.add_argument(
        "--worker-manifest",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume", action="store_true", help="Retained for CLI compatibility"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_name(args.generation_mode, args.model)
    txt_files = discover_txt_files(args.input_dir, args.prefix, args.worker_manifest)
    if not txt_files:
        print("No files found!")
        return 1

    resolved_attn = resolve_attn_implementation(args.attn_implementation)
    completed = spawn_workers(
        args=args,
        model_path=model_path,
        resolved_attn=resolved_attn,
        txt_files=txt_files,
        output_dir=args.output_dir,
    )
    if completed < 0:
        return 1

    missing = [
        txt_file
        for txt_file in txt_files
        if not is_valid_output(args.output_dir / f"{txt_file.stem}.wav")
    ]
    if missing:
        print(f"Missing {len(missing)} chunk outputs after batch run")
        for sample in missing[:5]:
            print(f"  - {sample.name}")
        return 1

    print(f"\nSuccessfully processed {completed} chunks!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
