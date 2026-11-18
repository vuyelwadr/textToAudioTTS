#!/usr/bin/env python3
"""
Optimized VibeVoice batch processor for the realtime 0.5B model.

This runner keeps the approved streaming path and Mike preset from the verified
sample output, restores the first-chunk lead-in trim used by the single-file
runner, and can launch multiple deterministic workers on the same A100 when the
installed runtime still only supports batch size 1.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path

import torch

from common import (
    resolve_attn_implementation,
    resolve_device,
    resolve_model_root,
    resolve_torch_dtype,
    resolve_visible_cuda_devices,
    safe_model_dirname,
)

os.environ["PATH"] = "/usr/bin:/bin:/usr/local/bin:" + os.environ.get("PATH", "")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

VIBEVOICE_DIR = Path(os.environ.get("MAHTI_EXTERNAL_ROOT", "")) / "vibevoice-community"
sys.path.insert(0, str(VIBEVOICE_DIR))
sys.path.insert(0, str(VIBEVOICE_DIR / "demo"))

try:
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )
except ImportError as e:
    print(f"Failed to import VibeVoice streaming modules: {e}")
    print(f"VIBEVOICE_DIR: {VIBEVOICE_DIR}")
    sys.exit(1)


DEFAULT_MODEL = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_SPEAKER = "Mike"
DEFAULT_LEADIN_TRIM_SECONDS = 12.5


class VoiceMapper:
    """Map speaker names to streaming voice prompt files."""

    def __init__(self) -> None:
        self.setup_voice_presets()

        aliases: dict[str, str] = {}
        for name, path in self.voice_presets.items():
            alias = name
            if "_" in alias:
                alias = alias.split("_")[0]
            if "-" in alias:
                alias = alias.split("-")[-1]
            aliases[alias] = path
        self.voice_presets.update(aliases)

    def setup_voice_presets(self) -> None:
        voices_dir = VIBEVOICE_DIR / "demo" / "voices" / "streaming_model"

        if not voices_dir.exists():
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        self.voice_presets = {
            path.stem: str(path) for path in sorted(voices_dir.glob("*.pt"))
        }
        self.available_voices = {
            name: path for name, path in self.voice_presets.items() if Path(path).exists()
        }

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path

        if self.voice_presets:
            default_voice = list(self.voice_presets.values())[0]
            print(
                f"Warning: No voice preset found for '{speaker_name}', using default voice"
            )
            return default_voice

        raise ValueError(
            "No voice presets available. Please check the voices/streaming_model directory."
        )


def resolve_model_name(explicit: str | None) -> str:
    model_name = explicit or DEFAULT_MODEL
    local_dir = resolve_model_root() / "vibevoice" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def _resolve_realtime_trim_seconds(
    source_path: Path, override: float | None = None
) -> float:
    if override is not None:
        return max(0.0, override)

    if source_path.name.startswith("chunk_0000_") or source_path.stem == "chunk_0000":
        return DEFAULT_LEADIN_TRIM_SECONDS

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
                f"Warning: trim request ({trim_seconds:.2f}s) exceeds audio duration; "
                f"skipping trim for {path.name}"
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


def _collect_work_items(
    input_dir_path: Path,
    output_dir_path: Path,
    prefix: str,
    worker_manifest: Path | None = None,
) -> tuple[list[Path], list[tuple[Path, Path]], int]:
    if worker_manifest is None:
        txt_files = sorted(input_dir_path.glob(f"{prefix}*.txt"))
    else:
        payload = json.loads(worker_manifest.read_text(encoding="utf-8"))
        txt_files = [Path(raw_path) for raw_path in payload]
    pending_items: list[tuple[Path, Path]] = []
    existing_count = 0

    for txt_file in txt_files:
        wav_file = output_dir_path / f"{txt_file.stem}.wav"
        if wav_file.exists() and wav_file.stat().st_size > 1000:
            existing_count += 1
        else:
            pending_items.append((txt_file, wav_file))

    return txt_files, pending_items, existing_count


def _resolve_worker_count(requested: int | None, resolved_device: str) -> int:
    raw_value = requested
    if raw_value is None:
        env_value = os.environ.get("VIBEVOICE_WORKERS", "").strip()
        raw_value = env_value or None

    if raw_value is not None and str(raw_value).lower() != "auto":
        return max(1, int(raw_value))

    if resolved_device != "cuda":
        return 1

    try:
        total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return 1

    if total_memory_gib >= 70:
        return 4
    if total_memory_gib >= 38:
        return 3 if major >= 8 else 2
    if total_memory_gib >= 28:
        return 2
    return 1


def _resolve_runtime_dtype(dtype: str | None, resolved_device: str):
    if dtype is not None:
        return resolve_torch_dtype(dtype)

    if resolved_device != "cuda" or not torch.cuda.is_available():
        return resolve_torch_dtype(dtype)

    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return resolve_torch_dtype(dtype)

    # V100 + SDPA has been more stable with fp32, while A100 can use bf16 safely.
    if major < 8:
        return torch.float32
    return resolve_torch_dtype(dtype)


def _shard_work_items(
    pending_items: list[tuple[Path, Path]], worker_count: int
) -> list[list[tuple[Path, Path]]]:
    shards: list[list[tuple[Path, Path]]] = [[] for _ in range(worker_count)]
    shard_loads = [0 for _ in range(worker_count)]

    for item in sorted(
        pending_items, key=lambda pair: pair[0].stat().st_size, reverse=True
    ):
        shard_index = min(range(worker_count), key=lambda idx: shard_loads[idx])
        shards[shard_index].append(item)
        shard_loads[shard_index] += item[0].stat().st_size

    return [shard for shard in shards if shard]


def _spawn_multi_gpu_runners(
    *,
    model_path: str,
    input_dir_path: Path,
    output_dir_path: Path,
    prefix: str,
    speaker_name: str,
    cfg_scale: float,
    ddpm_steps: int,
    dtype: str | None,
    attn_implementation: str | None,
    worker_count: str | None,
    trim_leading_seconds: float | None,
    log_dir: str | None,
    txt_files: list[Path],
    pending_items: list[tuple[Path, Path]],
) -> tuple[int, int]:
    visible_devices = resolve_visible_cuda_devices()
    gpu_shards = _shard_work_items(
        pending_items, min(len(visible_devices), len(pending_items))
    )
    manifest_root = Path(log_dir) if log_dir else output_dir_path.parent / "logs"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = manifest_root / ".gpu_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    processes: list[tuple[str, subprocess.Popen[bytes]]] = []

    assigned_devices = visible_devices[: len(gpu_shards)]
    print(
        "Launching "
        f"{len(gpu_shards)} VibeVoice GPU shards across visible devices: "
        f"{', '.join(assigned_devices)}"
    )

    for gpu_index, (device_id, shard) in enumerate(zip(assigned_devices, gpu_shards)):
        manifest_path = manifest_dir / f"gpu_{gpu_index:02d}.json"
        manifest_path.write_text(
            json.dumps([str(txt_file) for txt_file, _ in shard], indent=2),
            encoding="utf-8",
        )
        print(f"  gpu {device_id}: {len(shard)} pending chunks")
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--model",
            model_path,
            "--input-dir",
            str(input_dir_path),
            "--output-dir",
            str(output_dir_path),
            "--prefix",
            prefix,
            "--speaker",
            speaker_name,
            "--cfg-scale",
            str(cfg_scale),
            "--ddpm-steps",
            str(ddpm_steps),
            "--device",
            "cuda",
            "--workers",
            worker_count or "auto",
            "--worker-manifest",
            str(manifest_path),
            "--worker-index",
            str(gpu_index),
        ]
        if dtype is not None:
            cmd.extend(["--dtype", dtype])
        if attn_implementation is not None:
            cmd.extend(["--attn-implementation", attn_implementation])
        if log_dir is not None:
            cmd.extend(["--log-dir", log_dir])
        if trim_leading_seconds is not None:
            cmd.extend(["--trim-leading-seconds", str(trim_leading_seconds)])

        shard_env = os.environ.copy()
        shard_env["CUDA_VISIBLE_DEVICES"] = device_id
        shard_env["OMP_NUM_THREADS"] = "1"
        shard_env["MKL_NUM_THREADS"] = "1"
        shard_env["PYTHONUNBUFFERED"] = "1"
        processes.append((device_id, subprocess.Popen(cmd, env=shard_env)))

    failures = 0
    for device_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures += 1
            print(f"gpu {device_id} shard failed with exit code {return_code}")

    completed = sum(
        1
        for txt_file in txt_files
        if (output_dir_path / f"{txt_file.stem}.wav").exists()
        and (output_dir_path / f"{txt_file.stem}.wav").stat().st_size > 1000
    )
    failed = len(txt_files) - completed
    if failures and failed == 0:
        failed = failures
    return completed, failed


def _load_runtime(
    model_path: str,
    speaker_name: str,
    ddpm_steps: int,
    resolved_device: str,
    resolved_dtype,
    resolved_attn: str,
):
    if resolved_device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"Loading VibeVoice streaming model from: {model_path}")
    print(f"Device: {resolved_device}")
    print(f"torch_dtype: {resolved_dtype}")
    print(f"attn_implementation: {resolved_attn}")
    print("Loading processor...")
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
    print("Loading model...")

    if resolved_device == "cuda":
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=resolved_dtype,
            device_map="cuda",
            attn_implementation=resolved_attn,
        )
    elif resolved_device == "mps":
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=resolved_dtype,
            device_map=None,
            attn_implementation=resolved_attn,
        )
        model.to("mps")
    else:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=resolved_dtype,
            device_map="cpu",
            attn_implementation=resolved_attn,
        )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=ddpm_steps)
    print("Model loaded successfully")

    voice_mapper = VoiceMapper()
    voice_sample = voice_mapper.get_voice_path(speaker_name)
    print(f"Loading voice preset: {voice_sample}")
    target_device = resolved_device if resolved_device != "cpu" else "cpu"
    all_prefilled_outputs = torch.load(
        voice_sample, map_location=target_device, weights_only=False
    )
    all_prefilled_outputs = _cast_prefilled_outputs(
        all_prefilled_outputs,
        device=target_device,
        dtype=resolved_dtype,
    )

    return processor, model, all_prefilled_outputs, target_device


def _cast_prefilled_outputs(payload, *, device: str, dtype):
    if torch.is_tensor(payload):
        tensor = payload.to(device)
        if dtype is not None and torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=dtype)
        return tensor
    if isinstance(payload, dict):
        if type(payload) is not dict:
            cloned = copy.copy(payload)
            for key, value in payload.items():
                cloned[key] = _cast_prefilled_outputs(
                    value,
                    device=device,
                    dtype=dtype,
                )
            return cloned
        return {
            key: _cast_prefilled_outputs(value, device=device, dtype=dtype)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _cast_prefilled_outputs(item, device=device, dtype=dtype)
            for item in payload
        ]
    if isinstance(payload, tuple):
        return tuple(
            _cast_prefilled_outputs(item, device=device, dtype=dtype)
            for item in payload
        )
    return payload


def _render_chunk(
    txt_file: Path,
    wav_file: Path,
    *,
    processor,
    model,
    all_prefilled_outputs,
    target_device: str,
    resolved_dtype,
    cfg_scale: float,
    trim_leading_seconds: float | None,
    log_prefix: str,
) -> tuple[bool, float]:
    text = txt_file.read_text(encoding="utf-8").strip()
    if not text:
        print(f"{log_prefix}Warning: empty text in {txt_file.name}")
        return False, 0.0

    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            tensor = value.to(target_device)
            if resolved_dtype is not None and torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=resolved_dtype)
            inputs[key] = tensor

    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
        )
    generation_time = time.time() - start_time

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        raise RuntimeError(f"No audio generated for {txt_file.name}")

    audio = outputs.speech_outputs[0]
    if hasattr(audio, "shape") and len(audio.shape) > 0:
        audio_duration = float(audio.shape[-1]) / 24000.0
    else:
        audio_duration = 0.0
    rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

    processor.save_audio(audio, output_path=str(wav_file))
    trim_seconds = _resolve_realtime_trim_seconds(txt_file, trim_leading_seconds)
    if trim_seconds > 0:
        _trim_wav_prefix(wav_file, trim_seconds)

    if wav_file.exists() and wav_file.stat().st_size > 1000:
        print(f"{log_prefix}✓ Saved {wav_file.name} ({wav_file.stat().st_size} bytes)")
        print(
            f"{log_prefix}  generation: {generation_time:.2f}s | "
            f"audio: {audio_duration:.2f}s | RTF: {rtf:.2f}x"
        )
        return True, generation_time

    print(f"{log_prefix}✗ Failed to save valid audio for {txt_file.name}")
    return False, generation_time


def _worker_main(
    worker_id: int,
    shard: list[tuple[Path, Path]],
    *,
    model_path: str,
    speaker_name: str,
    cfg_scale: float,
    ddpm_steps: int,
    resolved_device: str,
    resolved_dtype,
    resolved_attn: str,
    trim_leading_seconds: float | None,
    result_queue,
) -> None:
    completed = 0
    failed = 0
    log_prefix = f"[worker {worker_id}] "

    try:
        processor, model, all_prefilled_outputs, target_device = _load_runtime(
            model_path=model_path,
            speaker_name=speaker_name,
            ddpm_steps=ddpm_steps,
            resolved_device=resolved_device,
            resolved_dtype=resolved_dtype,
            resolved_attn=resolved_attn,
        )

        for index, (txt_file, wav_file) in enumerate(shard, start=1):
            try:
                print(
                    f"{log_prefix}[{index}/{len(shard)}] Processing {txt_file.name}..."
                )
                ok, _generation_time = _render_chunk(
                    txt_file,
                    wav_file,
                    processor=processor,
                    model=model,
                    all_prefilled_outputs=all_prefilled_outputs,
                    target_device=target_device,
                    resolved_dtype=resolved_dtype,
                    cfg_scale=cfg_scale,
                    trim_leading_seconds=trim_leading_seconds,
                    log_prefix=log_prefix,
                )
                if ok:
                    completed += 1
                else:
                    failed += 1
            except Exception as exc:
                print(f"{log_prefix}✗ Error processing {txt_file.name}: {exc}")
                traceback.print_exc()
                failed += 1

        result_queue.put(
            {"worker_id": worker_id, "completed": completed, "failed": failed}
        )
    except Exception as exc:
        print(f"{log_prefix}Worker failed before rendering: {exc}")
        traceback.print_exc()
        result_queue.put(
            {"worker_id": worker_id, "completed": completed, "failed": len(shard)}
        )
        raise


def process_chunk_batch(
    model_path: str,
    input_dir: str,
    output_dir: str,
    prefix: str = "chunk_",
    speaker_name: str = DEFAULT_SPEAKER,
    cfg_scale: float = 1.5,
    ddpm_steps: int = 5,
    device: str | None = None,
    dtype: str | None = None,
    attn_implementation: str | None = None,
    worker_count: str | None = None,
    trim_leading_seconds: float | None = None,
    log_dir: str | None = None,
    worker_manifest: str | None = None,
) -> tuple[int, int]:
    """Process all text chunks while preserving the approved realtime settings."""

    worker_manifest_path = Path(worker_manifest) if worker_manifest is not None else None
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    txt_files, pending_items, existing_count = _collect_work_items(
        input_dir_path, output_dir_path, prefix, worker_manifest_path
    )
    total = len(txt_files)
    print(f"Found {total} text files to process")

    if total == 0:
        print("No files found!")
        return 0, 0

    print(f"Existing chunk outputs: {existing_count}")
    print(f"Pending chunk outputs: {len(pending_items)}")
    if not pending_items:
        print("All chunk outputs already exist; skipping model load.")
        return existing_count, 0

    resolved_device = resolve_device(device)
    visible_devices = resolve_visible_cuda_devices()
    if (
        worker_manifest_path is None
        and resolved_device == "cuda"
        and worker_count != "1"
        and len(visible_devices) > 1
        and len(pending_items) > 1
    ):
        completed, failed = _spawn_multi_gpu_runners(
            model_path=model_path,
            input_dir_path=input_dir_path,
            output_dir_path=output_dir_path,
            prefix=prefix,
            speaker_name=speaker_name,
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
            dtype=dtype,
            attn_implementation=attn_implementation,
            worker_count=worker_count,
            trim_leading_seconds=trim_leading_seconds,
            log_dir=log_dir,
            txt_files=txt_files,
            pending_items=pending_items,
        )
        print("\nBatch processing complete!")
        print(f"  ✓ Completed: {completed}")
        print(f"  ✗ Failed: {failed}")
        print(f"  📁 Output directory: {output_dir_path}")
        return completed, failed

    resolved_dtype = _resolve_runtime_dtype(dtype, resolved_device)
    resolved_attn = resolve_attn_implementation(attn_implementation)
    selected_worker_count = _resolve_worker_count(worker_count, resolved_device)
    print(f"Selected worker count: {selected_worker_count}")

    if (
        resolved_device == "cuda"
        and selected_worker_count > 1
        and len(pending_items) > 1
    ):
        shards = _shard_work_items(pending_items, selected_worker_count)
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []

        print(f"Launching {len(shards)} VibeVoice workers on the same visible GPU")
        for worker_id, shard in enumerate(shards, start=1):
            process = ctx.Process(
                target=_worker_main,
                kwargs={
                    "worker_id": worker_id,
                    "shard": shard,
                    "model_path": model_path,
                    "speaker_name": speaker_name,
                    "cfg_scale": cfg_scale,
                    "ddpm_steps": ddpm_steps,
                    "resolved_device": resolved_device,
                    "resolved_dtype": resolved_dtype,
                    "resolved_attn": resolved_attn,
                    "trim_leading_seconds": trim_leading_seconds,
                    "result_queue": result_queue,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        worker_results = []
        while not result_queue.empty():
            worker_results.append(result_queue.get())

        if len(worker_results) != len(processes):
            raise RuntimeError(
                "One or more VibeVoice worker processes exited without reporting results"
            )
        for process in processes:
            if process.exitcode not in (0, None):
                raise RuntimeError(
                    f"VibeVoice worker exited with status {process.exitcode}"
                )

        completed = existing_count + sum(item["completed"] for item in worker_results)
        failed = sum(item["failed"] for item in worker_results)
    else:
        result_queue = mp.get_context("spawn").Queue()
        _worker_main(
            1,
            pending_items,
            model_path=model_path,
            speaker_name=speaker_name,
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
            resolved_device=resolved_device,
            resolved_dtype=resolved_dtype,
            resolved_attn=resolved_attn,
            trim_leading_seconds=trim_leading_seconds,
            result_queue=result_queue,
        )
        result = result_queue.get()
        completed = existing_count + result["completed"]
        failed = result["failed"]

    print("\nBatch processing complete!")
    print(f"  ✓ Completed: {completed}")
    print(f"  ✗ Failed: {failed}")
    print(f"  📁 Output directory: {output_dir_path}")

    return completed, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimized VibeVoice realtime batch processing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the VibeVoice streaming model",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing chunk_*.txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output wav files",
    )
    parser.add_argument("--prefix", type=str, default="chunk_", help="Prefix for chunk files")
    parser.add_argument(
        "--speaker", type=str, default=DEFAULT_SPEAKER, help="Voice preset speaker name"
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=1.5, help="CFG guidance scale"
    )
    parser.add_argument(
        "--ddpm-steps", type=int, default=5, help="DDPM inference steps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference: cuda | mps | cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch dtype override: fp16 | bf16 | fp32",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Attention implementation override",
    )
    parser.add_argument(
        "--workers",
        "--worker-count",
        dest="worker_count",
        type=str,
        default=None,
        help="Concurrent VibeVoice workers per GPU, or 'auto'",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-manifest",
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
        "--trim-leading-seconds",
        type=float,
        default=None,
        help=(
            "Trim this many seconds from the start of every generated chunk. "
            "If omitted, chunk_0000_* gets the approved lead-in trim automatically."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed chunks",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = resolve_model_name(args.model)

    completed, failed = process_chunk_batch(
        model_path=model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        speaker_name=args.speaker,
        cfg_scale=args.cfg_scale,
        ddpm_steps=args.ddpm_steps,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        worker_count=args.worker_count,
        trim_leading_seconds=args.trim_leading_seconds,
        log_dir=args.log_dir,
        worker_manifest=args.worker_manifest,
    )

    if failed == 0 and completed > 0:
        print(f"\nSuccessfully processed {completed} chunks!")
        return 0

    print(f"\nChunk processing finished with failures. Completed: {completed}, failed: {failed}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
