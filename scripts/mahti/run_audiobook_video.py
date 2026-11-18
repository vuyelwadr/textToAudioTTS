from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mahti.common import ensure_parent
from scripts.make_autoscroll_video import probe_audio_duration

VIDEO_SCRIPT = REPO_ROOT / "scripts" / "make_autoscroll_video.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run audiobook video generation and record wall-clock timing",
    )
    parser.add_argument("--summary", type=Path, help="Path to audiobook_summary.json")
    parser.add_argument("--model", default="qwen3_faster_clone", help="Model id inside audiobook_summary.json")
    parser.add_argument("--audio", type=Path, help="Final joined WAV to render")
    parser.add_argument("--text", type=Path, help="Full source text file")
    parser.add_argument("--chunk-text-dir", type=Path, help="Directory of chunk_*.txt files")
    parser.add_argument("--chunk-audio-dir", type=Path, help="Directory of chunk_*.wav files")
    parser.add_argument("--output", type=Path, help="Output MP4 path")
    parser.add_argument("--timing-output", type=Path, help="Where to write timing JSON")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--video-codec", default="auto")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", default="auto")
    parser.add_argument("--font", default="auto")
    parser.add_argument("--paragraph-size", type=int, default=56)
    parser.add_argument("--heading-size", type=int, default=88)
    parser.add_argument("--focus-line-ratio", type=float, default=0.62)
    parser.add_argument("--chapter-end-hold", type=float, default=3.0)
    parser.add_argument("--chapter-transition", type=float, default=0.6)
    parser.add_argument("--chapter-start-hold", type=float, default=2.0)
    parser.add_argument("--chapter-audio-output", type=Path)
    parser.add_argument("--chunk-manifest", type=Path, help="Chunk manifest carrying paragraph-boundary metadata")
    parser.add_argument("--paragraph-pause-seconds", type=float, default=None)
    parser.add_argument("--show-active-line", action="store_true", help="Render the red progress line")
    parser.add_argument(
        "--active-line-timing",
        choices=("auto", "chunk", "asr"),
        default="asr",
        help="Timing mode passed to make_autoscroll_video.py when active line is enabled",
    )
    parser.add_argument("--active-line-asr-model", default="small.en")
    parser.add_argument(
        "--active-line-asr-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--active-line-asr-cache-dir", type=Path)
    parser.add_argument("--active-line-asr-workers", default="auto")
    parser.add_argument(
        "--plates-dir",
        type=Path,
        help="Optional book illustration directory (illustrated scrolling-text video)",
    )
    parser.add_argument(
        "--plates-manifest",
        type=Path,
        help="Optional plates manifest JSON for illustrated scrolling-text video",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from_summary(summary_path: Path, model_name: str) -> tuple[Path | None, Path | None, Path | None, Path, Path | None, float]:
    summary = load_summary(summary_path)
    model_entry = next((item for item in summary.get("models", []) if item.get("model") == model_name), None)
    if model_entry is None:
        available = ", ".join(item.get("model", "<unknown>") for item in summary.get("models", []))
        raise RuntimeError(f"Model {model_name!r} not found in {summary_path}. Available: {available}")

    blocking_chunk_indices = model_entry.get("blocking_chunk_indices") or []
    if blocking_chunk_indices:
        raise RuntimeError(
            f"Refusing to render video for {model_name}: chunk audit reported blocking chunks "
            f"{', '.join(str(index) for index in blocking_chunk_indices[:20])}"
        )

    chunk_manifest = summary.get("chunk_manifest")
    chunk_text_dir = Path(chunk_manifest).parent if chunk_manifest else None
    corrected_chunk_text_dir = model_entry.get("corrected_chunk_text_dir")
    if corrected_chunk_text_dir:
        corrected_path = Path(corrected_chunk_text_dir)
        if corrected_path.exists():
            chunk_text_dir = corrected_path

    chunk_outputs = [Path(item) for item in model_entry.get("chunk_outputs", [])]
    chunk_audio_dir = chunk_outputs[0].parent if chunk_outputs else None

    effective_input = summary.get("effective_input") or summary.get("input")
    text_path = Path(effective_input) if effective_input else None
    audio_path = Path(model_entry["final_output"])
    manifest_path = Path(chunk_manifest) if chunk_manifest else None
    paragraph_pause_seconds = float(summary.get("paragraph_pause_seconds", 0.0))
    return text_path, chunk_text_dir, chunk_audio_dir, audio_path, manifest_path, paragraph_pause_seconds


def choose_output_path(audio_path: Path, explicit_output: Path | None, show_active_line: bool) -> Path:
    if explicit_output is not None:
        return explicit_output
    suffix = "_active_line" if show_active_line else "_autoscroll"
    return audio_path.with_name(f"{audio_path.stem}{suffix}.mp4")


def choose_timing_path(output_path: Path, explicit_timing_path: Path | None) -> Path:
    if explicit_timing_path is not None:
        return explicit_timing_path
    return output_path.with_suffix(".timing.json")


def build_command(
    *,
    audio_path: Path,
    text_path: Path | None,
    chunk_text_dir: Path | None,
    chunk_audio_dir: Path | None,
    chunk_manifest: Path | None,
    paragraph_pause_seconds: float,
    output_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(VIDEO_SCRIPT),
        "--audio",
        str(audio_path),
        "--output",
        str(output_path),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--video-codec",
        args.video_codec,
        "--video-crf",
        str(args.video_crf),
        "--video-preset",
        args.video_preset,
        "--font",
        args.font,
        "--paragraph-size",
        str(args.paragraph_size),
        "--heading-size",
        str(args.heading_size),
        "--focus-line-ratio",
        str(args.focus_line_ratio),
        "--chapter-end-hold",
        str(args.chapter_end_hold),
        "--chapter-transition",
        str(args.chapter_transition),
        "--chapter-start-hold",
        str(args.chapter_start_hold),
    ]

    if text_path is not None and text_path.exists():
        command.extend(["--text", str(text_path)])

    if chunk_text_dir is not None and chunk_text_dir.exists():
        command.extend(["--chunk-text-dir", str(chunk_text_dir)])
    elif text_path is None or not text_path.exists():
        raise RuntimeError("No usable text source was found for video generation")

    if chunk_audio_dir is not None and chunk_audio_dir.exists():
        command.extend(["--chunk-audio-dir", str(chunk_audio_dir)])
    if chunk_manifest is not None and chunk_manifest.exists():
        command.extend(["--chunk-manifest", str(chunk_manifest)])
        command.extend(["--paragraph-pause-seconds", str(paragraph_pause_seconds)])
    if args.chapter_audio_output is not None:
        command.extend(["--chapter-audio-output", str(args.chapter_audio_output)])

    if args.show_active_line:
        command.append("--show-active-line")
        command.extend(["--active-line-timing", args.active_line_timing])
        command.extend(["--active-line-asr-model", args.active_line_asr_model])
        command.extend(["--active-line-asr-device", args.active_line_asr_device])
        command.extend(["--active-line-asr-workers", args.active_line_asr_workers])
        if args.active_line_asr_cache_dir is not None:
            command.extend(["--active-line-asr-cache-dir", str(args.active_line_asr_cache_dir)])

    if args.plates_dir is not None:
        command.extend(["--plates-dir", str(args.plates_dir)])
    if args.plates_manifest is not None:
        command.extend(["--plates-manifest", str(args.plates_manifest)])

    return command


def main() -> int:
    args = parse_args()

    text_path = args.text
    chunk_text_dir = args.chunk_text_dir
    chunk_audio_dir = args.chunk_audio_dir
    chunk_manifest = args.chunk_manifest
    paragraph_pause_seconds = float(args.paragraph_pause_seconds or 0.0)
    audio_path = args.audio

    if args.summary is not None:
        (
            summary_text_path,
            summary_chunk_text_dir,
            summary_chunk_audio_dir,
            summary_audio_path,
            summary_chunk_manifest,
            summary_paragraph_pause_seconds,
        ) = resolve_from_summary(args.summary, args.model)
        text_path = text_path or summary_text_path
        chunk_text_dir = chunk_text_dir or summary_chunk_text_dir
        chunk_audio_dir = chunk_audio_dir or summary_chunk_audio_dir
        chunk_manifest = chunk_manifest or summary_chunk_manifest
        if args.paragraph_pause_seconds is None:
            paragraph_pause_seconds = summary_paragraph_pause_seconds
        audio_path = audio_path or summary_audio_path

    if audio_path is None:
        raise RuntimeError("You must provide either --audio or --summary")
    if not audio_path.exists():
        raise RuntimeError(f"Audio file not found: {audio_path}")

    output_path = choose_output_path(audio_path, args.output, args.show_active_line)
    ensure_parent(output_path)

    if args.show_active_line and args.active_line_asr_cache_dir is None:
        args.active_line_asr_cache_dir = output_path.parent / f"{output_path.stem}_asr_cache"

    timing_path = choose_timing_path(output_path, args.timing_output)
    ensure_parent(timing_path)

    command = build_command(
        audio_path=audio_path,
        text_path=text_path,
        chunk_text_dir=chunk_text_dir,
        chunk_audio_dir=chunk_audio_dir,
        chunk_manifest=chunk_manifest,
        paragraph_pause_seconds=paragraph_pause_seconds,
        output_path=output_path,
        args=args,
    )

    audio_duration = probe_audio_duration(audio_path)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    elapsed = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc)

    timing_payload = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed,
        "audio_duration_seconds": audio_duration,
        "speed_x_realtime": audio_duration / elapsed if elapsed else None,
        "summary": str(args.summary) if args.summary else None,
        "model": args.model,
        "audio": str(audio_path),
        "text": str(text_path) if text_path else None,
        "chunk_text_dir": str(chunk_text_dir) if chunk_text_dir else None,
        "chunk_audio_dir": str(chunk_audio_dir) if chunk_audio_dir else None,
        "chunk_manifest": str(chunk_manifest) if chunk_manifest else None,
        "paragraph_pause_seconds": paragraph_pause_seconds,
        "output": str(output_path),
        "video_codec": args.video_codec,
        "video_crf": args.video_crf,
        "video_preset": args.video_preset,
        "show_active_line": args.show_active_line,
        "active_line_timing": args.active_line_timing if args.show_active_line else None,
        "active_line_asr_model": args.active_line_asr_model if args.show_active_line else None,
        "active_line_asr_device": args.active_line_asr_device if args.show_active_line else None,
        "active_line_asr_workers": args.active_line_asr_workers if args.show_active_line else None,
        "active_line_asr_cache_dir": str(args.active_line_asr_cache_dir) if args.active_line_asr_cache_dir else None,
        "chapter_end_hold": args.chapter_end_hold,
        "chapter_transition": args.chapter_transition,
        "chapter_start_hold": args.chapter_start_hold,
        "chapter_audio_output": str(args.chapter_audio_output) if args.chapter_audio_output else None,
        "command": command,
    }
    timing_path.write_text(json.dumps(timing_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Created video: {output_path}")
    print(f"Timing JSON: {timing_path}")
    if timing_payload["speed_x_realtime"] is not None:
        print(f"Speed: {timing_payload['speed_x_realtime']:.2f}x realtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
