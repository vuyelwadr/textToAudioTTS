from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    ensure_parent,
    load_text,
    normalize_target_text,
    resolve_cluster_name,
    resolve_device,
    resolve_model_root,
    resolve_torch_dtype,
    safe_model_dirname,
)


DEFAULT_CHUNK_DIR = (
    REPO_ROOT / "generated" / "prometheus_benchmark_smoke_20260430" / "PrometheusAtlas" / "chunks"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "generated" / "audiobooks-prometheus_atlas-moss_tts_clone_full_20260501" / "PrometheusAtlas"
)
DEFAULT_REFERENCE_AUDIO = (
    REPO_ROOT / "generated" / "clone_references_2026-04-25" / "brothersKarmatzov.clone_ref_60s.wav"
)
DEFAULT_INPUT_TEXT = REPO_ROOT / "library" / "PrometheusAtlas" / "PrometheusAtlas.txt"
MODEL_NAME = "moss_tts_local_clone"
FINAL_WAV_NAME = "PrometheusAtlas_moss_tts_local_clone.wav"
MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
CODEC_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
DEFAULT_AUDIO_TEMPERATURE = 1.0
DEFAULT_AUDIO_TOP_P = 0.95
DEFAULT_AUDIO_TOP_K = 50
DEFAULT_AUDIO_REPETITION_PENALTY = 1.1
DEFAULT_MAX_NEW_TOKENS = 4096


def _resolve_hf_path(model_id: str) -> str:
    local = resolve_model_root() / "moss-tts" / safe_model_dirname(model_id)
    return str(local) if local.exists() else model_id


def _resolve_attn_implementation(device: str, dtype: torch.dtype) -> str:
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


def _patch_torchaudio() -> None:
    try:
        import torchaudio as _ta
    except Exception as exc:
        raise RuntimeError("torchaudio is required for MOSS generation") from exc

    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401
        return
    except Exception:
        pass

    def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True, **_kw):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T)
        if frame_offset:
            tensor = tensor[:, frame_offset:]
        if num_frames > 0:
            tensor = tensor[:, :num_frames]
        if not normalize:
            tensor = tensor * 32768.0
        return tensor, sr

    def _sf_save(uri, src, sample_rate, channels_first=True, **_kw):
        data = src.detach().cpu().numpy()
        if channels_first:
            data = data.T
        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]
        sf.write(str(uri), data, samplerate=sample_rate)

    _ta.load = _sf_load
    _ta.save = _sf_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the full Prometheus Atlas audiobook with MOSS-TTS clone mode."
    )
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-text", type=Path, default=DEFAULT_INPUT_TEXT)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--model-name", default=MODEL_ID)
    parser.add_argument("--codec-name", default=CODEC_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--audio-temperature", type=float, default=DEFAULT_AUDIO_TEMPERATURE)
    parser.add_argument("--audio-top-p", type=float, default=DEFAULT_AUDIO_TOP_P)
    parser.add_argument("--audio-top-k", type=int, default=DEFAULT_AUDIO_TOP_K)
    parser.add_argument("--audio-repetition-penalty", type=float, default=DEFAULT_AUDIO_REPETITION_PENALTY)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Number of chunks to process from --start-index; 0 means through the end.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate valid existing chunk WAVs.")
    parser.add_argument("--join-only", action="store_true", help="Only validate/join existing chunks and write summary.")
    parser.add_argument("--skip-join", action="store_true", help="Do not join the final WAV after generation.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_chunks(chunk_dir: Path, start_index: int = 0, max_chunks: int = 0) -> list[Path]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if max_chunks < 0:
        raise ValueError("--max-chunks must be >= 0")
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"chunk dir not found: {chunk_dir}")

    chunks = sorted(chunk_dir.glob("chunk_*.txt"))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.txt files found in {chunk_dir}")

    selected = chunks[start_index:]
    if max_chunks:
        selected = selected[:max_chunks]
    return selected


def is_valid_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb"):
            return True
    except wave.Error:
        return False


def probe_wav(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as reader:
        params = reader.getparams()
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "duration_seconds": round(reader.getnframes() / params.framerate, 4),
            "sample_rate": params.framerate,
            "channels": params.nchannels,
            "sample_width": params.sampwidth,
        }


def join_wavs(chunk_wavs: Iterable[Path], output_path: Path) -> dict[str, object]:
    paths = list(chunk_wavs)
    if not paths:
        raise ValueError("no chunk wavs to join")

    params = None
    signature = None
    total_frames = 0
    tmp_path = output_path.with_suffix(".tmp.wav")
    ensure_parent(output_path)
    if tmp_path.exists():
        tmp_path.unlink()

    with wave.open(str(tmp_path), "wb") as writer:
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
                frames = reader.getnframes()
                writer.writeframes(reader.readframes(frames))
                total_frames += frames

    tmp_path.replace(output_path)
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


def output_path_for_chunk(chunk_path: Path, chunk_output_dir: Path) -> Path:
    return chunk_output_dir / f"{chunk_path.stem}.wav"


def build_summary(
    *,
    args: argparse.Namespace,
    chunks: list[Path],
    chunk_outputs: list[Path],
    final_output: dict[str, object] | None,
    elapsed_seconds: float,
    generated_count: int,
    skipped_count: int,
) -> dict[str, object]:
    model_entry = {
        "model": MODEL_NAME,
        "final_output": str(args.output_root / FINAL_WAV_NAME),
        "chunk_outputs": [str(path) for path in chunk_outputs],
        "blocking_chunk_indices": [],
        "generated_chunk_count": generated_count,
        "skipped_existing_chunk_count": skipped_count,
        "chunk_count": len(chunk_outputs),
        "final_output_probe": final_output,
        "elapsed_seconds": elapsed_seconds,
        "cluster": resolve_cluster_name(),
        "device": resolve_device(args.device),
        "dtype": str(resolve_torch_dtype(args.dtype)),
        "mode": "clone",
        "model_name": args.model_name,
        "codec_name": args.codec_name,
        "reference_audio": str(args.reference_audio),
        "audio_temperature": args.audio_temperature,
        "audio_top_p": args.audio_top_p,
        "audio_top_k": args.audio_top_k,
        "audio_repetition_penalty": args.audio_repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }
    return {
        "created_at": utc_now(),
        "input": str(args.input_text) if args.input_text.exists() else None,
        "chunk_manifest": str(args.chunk_dir / "PrometheusAtlas_manifest.json"),
        "chunk_dir": str(args.chunk_dir),
        "output_root": str(args.output_root),
        "chunks": [str(path) for path in chunks],
        "models": [model_entry],
    }


def write_summary(output_root: Path, payload: dict[str, object], *, range_only: bool = False) -> Path:
    if range_only:
        start_index = int(payload.get("range_start_index", 0))
        max_chunks = payload.get("range_max_chunks", "all")
        summary_path = output_root / "range_summaries" / f"summary_{start_index:04d}_{max_chunks}.json"
    else:
        summary_path = output_root / "audiobook_summary.json"
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return summary_path


def write_progress(output_root: Path, payload: dict[str, object], *, start_index: int) -> None:
    progress_path = output_root / "range_summaries" / f"progress_{start_index:04d}.json"
    ensure_parent(progress_path)
    progress_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_model(args: argparse.Namespace) -> tuple[object, object, str, torch.dtype]:
    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)
    _patch_torchaudio()
    model_path = _resolve_hf_path(args.model_name)
    codec_path = _resolve_hf_path(args.codec_name)
    attn_impl = _resolve_attn_implementation(device, dtype)

    from transformers import AutoModel, AutoProcessor

    print(
        f"[moss-full] loading model={args.model_name} codec={args.codec_name} "
        f"device={device} dtype={dtype} attn={attn_impl}",
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        codec_path=codec_path,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return processor, model, device, dtype


def generate_chunk(
    *,
    processor: object,
    model: object,
    device: str,
    args: argparse.Namespace,
    chunk_path: Path,
    output_path: Path,
) -> dict[str, object]:
    text = normalize_target_text(load_text(chunk_path))
    conversations = [[processor.build_user_message(text=text, reference=[str(args.reference_audio)])]]

    started = time.perf_counter()
    batch = processor(conversations, mode="generation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.audio_temperature,
            top_p=args.audio_top_p,
            top_k=args.audio_top_k,
            repetition_penalty=args.audio_repetition_penalty,
        )

    messages = processor.decode(outputs)
    audio = messages[0].audio_codes_list[0]
    sample_rate = processor.model_config.sampling_rate
    wav = audio.cpu().float().numpy()
    if wav.ndim == 2:
        wav = wav.T

    tmp_path = output_path.with_suffix(".tmp.wav")
    ensure_parent(output_path)
    if tmp_path.exists():
        tmp_path.unlink()
    sf.write(str(tmp_path), wav, samplerate=sample_rate)
    if not is_valid_wav(tmp_path):
        raise RuntimeError(f"generated invalid temporary WAV: {tmp_path}")
    tmp_path.replace(output_path)

    elapsed = time.perf_counter() - started
    probe = probe_wav(output_path)
    probe["elapsed_seconds"] = round(elapsed, 4)

    del batch, input_ids, attention_mask, outputs, messages, audio
    if device == "cuda":
        torch.cuda.empty_cache()
    return probe


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    start_time = time.perf_counter()

    if not args.reference_audio.exists():
        raise FileNotFoundError(f"reference audio not found: {args.reference_audio}")

    selected_chunks = discover_chunks(args.chunk_dir, args.start_index, args.max_chunks)
    all_chunks = discover_chunks(args.chunk_dir, 0, 0)
    chunk_output_dir = args.output_root / MODEL_NAME / "chunks"
    args.output_root.mkdir(parents=True, exist_ok=True)
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    selected_outputs = [output_path_for_chunk(chunk_path, chunk_output_dir) for chunk_path in selected_chunks]
    all_outputs = [output_path_for_chunk(chunk_path, chunk_output_dir) for chunk_path in all_chunks]

    generated_count = 0
    skipped_count = 0
    chunk_records: list[dict[str, object]] = []

    print(
        f"[moss-full] selected {len(selected_chunks)} chunk(s), start_index={args.start_index}, "
        f"max_chunks={args.max_chunks or 'all'}, output_root={args.output_root}",
        flush=True,
    )

    if not args.join_only:
        processor, model, device, _dtype = load_model(args)
        for offset, (chunk_path, output_path) in enumerate(zip(selected_chunks, selected_outputs), start=1):
            chunk_index = args.start_index + offset - 1
            if is_valid_wav(output_path) and not args.force:
                skipped_count += 1
                record = probe_wav(output_path)
                record.update({"chunk_index": chunk_index, "chunk": str(chunk_path), "status": "skipped_existing"})
                print(f"[moss-full] skip valid chunk {chunk_index:04d}: {output_path}", flush=True)
            else:
                print(f"[moss-full] generate chunk {chunk_index:04d}/{len(all_chunks) - 1:04d}: {chunk_path.name}", flush=True)
                record = generate_chunk(
                    processor=processor,
                    model=model,
                    device=device,
                    args=args,
                    chunk_path=chunk_path,
                    output_path=output_path,
                )
                generated_count += 1
                record.update({"chunk_index": chunk_index, "chunk": str(chunk_path), "status": "generated"})
                print(
                    f"[moss-full] wrote chunk {chunk_index:04d}: "
                    f"{record['duration_seconds']}s in {record['elapsed_seconds']}s",
                    flush=True,
                )

            chunk_records.append(record)
            write_progress(
                args.output_root,
                {
                    "started_at": started_at,
                    "updated_at": utc_now(),
                    "start_index": args.start_index,
                    "max_chunks": args.max_chunks,
                    "selected_chunk_count": len(selected_chunks),
                    "generated_chunk_count": generated_count,
                    "skipped_existing_chunk_count": skipped_count,
                    "last_record": record,
                    "records": chunk_records,
                },
                start_index=args.start_index,
            )

    final_probe: dict[str, object] | None = None
    if not args.skip_join:
        missing_outputs = [path for path in all_outputs if not is_valid_wav(path)]
        if missing_outputs:
            if args.join_only:
                missing_label = "\n".join(str(path) for path in missing_outputs[:20])
                raise RuntimeError(
                    f"cannot join; {len(missing_outputs)} chunk WAV(s) are missing or invalid:\n{missing_label}"
                )
            print(f"[moss-full] not joining yet; {len(missing_outputs)} chunk WAV(s) are still missing", flush=True)
        else:
            final_probe = join_wavs(all_outputs, args.output_root / FINAL_WAV_NAME)
            print(
                f"[moss-full] joined final WAV: {final_probe['path']} "
                f"duration={final_probe['duration_seconds']}s bytes={final_probe['bytes']}",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time
    summary = build_summary(
        args=args,
        chunks=all_chunks,
        chunk_outputs=all_outputs,
        final_output=final_probe,
        elapsed_seconds=elapsed,
        generated_count=generated_count,
        skipped_count=skipped_count,
    )
    summary["range_start_index"] = args.start_index
    summary["range_max_chunks"] = args.max_chunks or "all"
    summary_path = write_summary(args.output_root, summary, range_only=args.skip_join and not args.join_only)
    print(f"[moss-full] summary: {summary_path}", flush=True)
    print(f"[moss-full] elapsed_seconds={elapsed:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
