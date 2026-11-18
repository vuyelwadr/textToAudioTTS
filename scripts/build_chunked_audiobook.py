#!/usr/bin/env python3
"""Build one or more long-form audiobooks from a txt file by chunking and joining audio."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from chunk_alignment_audit import audit_chunk_alignment
from textChunkingAdapter import WebTTSChunker

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "tts_naturalness" / "generated" / "audiobooks"
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_DIR / ".work"
USER_REFERENCE_AUDIO = (
    REPO_ROOT / "outputs" / "video_downloads" / "haCxgJSFmFc" / "haCxgJSFmFc_reference_8s.wav"
)
USER_REFERENCE_TEXT_FILE = USER_REFERENCE_AUDIO.with_suffix(".transcript.auto.txt")
DEFAULT_REFERENCE_AUDIO = (
    USER_REFERENCE_AUDIO
    if USER_REFERENCE_AUDIO.exists()
    else REPO_ROOT / "assets" / "audio_samples" / "brothersKarmatzov.clone_ref_5s.wav"
)
DEFAULT_REFERENCE_TEXT_FILE: Path | None = (
    USER_REFERENCE_TEXT_FILE if USER_REFERENCE_TEXT_FILE.exists() else None
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    venv_name: str
    bootstrap_script: Path
    runner_script: Path
    runner_args: tuple[str, ...]
    batch_runner_script: Path | None = None
    batch_runner_args: tuple[str, ...] = ()


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3_faster_clone": ModelSpec(
        name="qwen3_faster_clone",
        label="Faster Qwen3-TTS clone",
        venv_name="faster-qwen3-tts",
        bootstrap_script=REPO_ROOT / "scripts" / "mahti" / "bootstrap_faster_qwen3_tts.sh",
        runner_script=REPO_ROOT / "scripts" / "mahti" / "run_faster_qwen3_tts.py",
        runner_args=(
            "--generation-mode",
            "clone",
            "--attn-implementation",
            "sdpa",
            "--temperature",
            "0.88",
            "--top-k",
            "55",
        ),
        batch_runner_script=REPO_ROOT / "scripts" / "mahti" / "run_qwen3_batch_optimized.py",
        batch_runner_args=(
            "--generation-mode",
            "clone",
            "--attn-implementation",
            "sdpa",
            "--temperature",
            "0.88",
            "--top-k",
            "55",
            "--workers",
            "1",
        ),
    ),
    "qwen3_faster_clone_xvec": ModelSpec(
        name="qwen3_faster_clone_xvec",
        label="Faster Qwen3-TTS clone (legacy x-vector-only baseline)",
        venv_name="faster-qwen3-tts",
        bootstrap_script=REPO_ROOT / "scripts" / "mahti" / "bootstrap_faster_qwen3_tts.sh",
        runner_script=REPO_ROOT / "scripts" / "mahti" / "run_faster_qwen3_tts.py",
        runner_args=(
            "--generation-mode",
            "clone",
            "--xvec-only",
            "--attn-implementation",
            "sdpa",
            "--temperature",
            "0.88",
            "--top-k",
            "55",
        ),
        batch_runner_script=REPO_ROOT / "scripts" / "mahti" / "run_qwen3_batch_optimized.py",
        batch_runner_args=(
            "--generation-mode",
            "clone",
            "--xvec-only",
            "--attn-implementation",
            "sdpa",
            "--temperature",
            "0.88",
            "--top-k",
            "55",
            "--workers",
            "1",
        ),
    ),
    "vibevoice_realtime": ModelSpec(
        name="vibevoice_realtime",
        label="VibeVoice realtime",
        venv_name="vibevoice",
        bootstrap_script=REPO_ROOT / "scripts" / "mahti" / "bootstrap_vibevoice.sh",
        runner_script=REPO_ROOT / "scripts" / "mahti" / "run_vibevoice.py",
        runner_args=(
            "--mode",
            "realtime",
            "--speaker-name",
            "en-Mike_man",
            "--cfg-scale",
            "1.7",
            "--ddpm-steps",
            "8",
        ),
        batch_runner_script=REPO_ROOT / "scripts" / "mahti" / "run_vibevoice_batch_optimized.py",
        batch_runner_args=(
            "--speaker",
            "en-Mike_man",
            "--cfg-scale",
            "1.7",
            "--ddpm-steps",
            "8",
            "--workers",
            "auto",
        ),
    ),
}
MIN_VALID_WAV_BYTES = 1000
CHAPTER_HEADING_PATTERN = re.compile(r"^\s*CHAPTER\s+[IVXLCDM]+\s*$", re.IGNORECASE)
PRECHAPTER_NOTE_LINE_PATTERN = re.compile(r"^\s*\d{3,4}\s+")
DEFAULT_PRECHAPTER_NOTE_LOOKBACK = 260
DEFAULT_PRECHAPTER_NOTE_MIN_LINES = 100
DEFAULT_PRECHAPTER_NOTE_MIN_RATIO = 0.55
DEFAULT_MAX_HEAL_ATTEMPTS = 2
# Matches scripts/chunk_alignment_audit.py's _chunk_index_from_path filename
# convention (chunk_<index>_<hash>.<ext>) so blocking_chunk_indices reported by
# the audit can be mapped back to this module's own chunk_files/chunk_outputs
# lists without importing a private helper from that module.
CHUNK_INDEX_PATTERN = re.compile(r"^chunk_(\d+)_[0-9a-f]+$")


def chunk_index_from_path(path: Path) -> int:
    match = CHUNK_INDEX_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"Could not parse chunk index from {path}")
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk a txt book and generate full audiobook WAVs for the selected models."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source txt file to render")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=["qwen3_faster_clone", "vibevoice_realtime"],
        help="Audiobook models to render",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for final audiobook WAVs",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Directory for chunk text, chunk audio, and logs",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=850,
        help="Maximum characters per text chunk",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=200,
        help="Minimum characters per chunk",
    )
    parser.add_argument(
        "--chunk-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker count recorded in the chunk manifest",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit rendering to the first N chunks for smoke tests or interactive preflight",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=DEFAULT_REFERENCE_AUDIO,
        help="Reference audio used for Faster Qwen3 clone mode",
    )
    parser.add_argument(
        "--reference-text-file",
        type=Path,
        default=DEFAULT_REFERENCE_TEXT_FILE,
        help="Reference transcript used for Faster Qwen3 clone mode",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate chunk audio and overwrite existing final outputs",
    )
    parser.add_argument(
        "--cleanup-work",
        action="store_true",
        help="Delete the per-model work directories after a successful run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk the text and print commands without invoking TTS",
    )
    parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="Render multiple models concurrently, one CUDA device per model",
    )
    parser.add_argument(
        "--audit-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Audit chunk audio/text alignment before joining the final audiobook WAV",
    )
    parser.add_argument(
        "--audit-asr-model",
        default="small.en",
        help="faster-whisper model used for chunk alignment audit",
    )
    parser.add_argument(
        "--audit-asr-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for chunk alignment audit",
    )
    parser.add_argument(
        "--audit-neighbor-window",
        type=int,
        default=1,
        help="Neighbor chunk window used during chunk alignment audit",
    )
    parser.add_argument(
        "--audit-cache-dir",
        type=Path,
        default=None,
        help="Optional shared ASR cache directory for chunk alignment audit",
    )
    parser.add_argument(
        "--audit-workers",
        default="auto",
        help="Parallel worker count for chunk alignment audit, or 'auto'",
    )
    parser.add_argument(
        "--heal-blocking-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the chunk alignment audit reports blocking chunks, archive the "
            "failed chunk artifacts and resynthesize only those chunk indices "
            "(retrying up to --max-heal-attempts times) instead of immediately "
            "failing the whole render. Pass --no-heal-blocking-chunks to restore "
            "the old fail-fast-on-first-blocking-chunk behavior."
        ),
    )
    parser.add_argument(
        "--max-heal-attempts",
        type=int,
        default=DEFAULT_MAX_HEAL_ATTEMPTS,
        help=(
            "Maximum number of resynthesis retries per still-blocking chunk when "
            "--heal-blocking-chunks is enabled (default 2, i.e. up to 3 total "
            "tries including the original synthesis)."
        ),
    )
    parser.add_argument(
        "--sanitize-prechapter-note-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strip large citation/endnote blocks that appear immediately before chapter headings "
            "before chunking the audiobook input."
        ),
    )
    parser.add_argument(
        "--sanitize-prechapter-note-lookback",
        type=int,
        default=DEFAULT_PRECHAPTER_NOTE_LOOKBACK,
        help="Maximum number of lines to inspect before each chapter heading when stripping note blocks.",
    )
    parser.add_argument(
        "--sanitize-prechapter-note-min-lines",
        type=int,
        default=DEFAULT_PRECHAPTER_NOTE_MIN_LINES,
        help="Minimum count of numbered note lines required before a pre-chapter note block is stripped.",
    )
    parser.add_argument(
        "--sanitize-prechapter-note-min-ratio",
        type=float,
        default=DEFAULT_PRECHAPTER_NOTE_MIN_RATIO,
        help="Minimum numbered-note-line ratio required before a pre-chapter block is stripped.",
    )
    return parser.parse_args()


def ensure_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_audit_dependencies_available() -> None:
    try:
        import faster_whisper  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "faster-whisper is required when --audit-chunks is enabled. "
            "On Mahti, source scripts/mahti/bootstrap_audiobook_video.sh before "
            "running scripts/build_chunked_audiobook.py, or rerun with "
            "--no-audit-chunks only for an unaudited recovery."
        ) from exc


def compute_source_text_sha256(input_path: Path) -> str:
    return hashlib.sha256(input_path.read_bytes()).hexdigest()


def sanitize_prechapter_note_blocks(
    text: str,
    *,
    max_lookback: int,
    min_note_lines: int,
    min_note_ratio: float,
) -> tuple[str, list[dict[str, object]]]:
    lines = text.splitlines()
    if not lines:
        return text, []

    keep_line = [True] * len(lines)
    removed_blocks: list[dict[str, object]] = []
    chapter_indices = [index for index, line in enumerate(lines) if CHAPTER_HEADING_PATTERN.match(line)]
    previous_chapter_index = -1

    for chapter_index in chapter_indices:
        search_start = max(previous_chapter_index + 1, chapter_index - max_lookback)
        candidate_start: int | None = None
        candidate_nonblank_count = 0
        candidate_note_line_count = 0

        for start_index in range(search_start, chapter_index):
            if not PRECHAPTER_NOTE_LINE_PATTERN.match(lines[start_index]):
                continue
            segment = lines[start_index:chapter_index]
            nonblank_lines = [line for line in segment if line.strip()]
            if not nonblank_lines:
                continue
            note_line_count = sum(
                1 for line in nonblank_lines if PRECHAPTER_NOTE_LINE_PATTERN.match(line)
            )
            if note_line_count < min_note_lines:
                continue
            note_line_ratio = note_line_count / len(nonblank_lines)
            if note_line_ratio < min_note_ratio:
                continue
            candidate_start = start_index
            candidate_nonblank_count = len(nonblank_lines)
            candidate_note_line_count = note_line_count
            break

        if candidate_start is not None:
            for line_index in range(candidate_start, chapter_index):
                keep_line[line_index] = False
            removed_blocks.append(
                {
                    "start_line": candidate_start + 1,
                    "end_line": chapter_index,
                    "removed_line_count": chapter_index - candidate_start,
                    "chapter_line": chapter_index + 1,
                    "chapter_heading": lines[chapter_index].strip(),
                    "note_line_count": candidate_note_line_count,
                    "nonblank_line_count": candidate_nonblank_count,
                    "note_line_ratio": round(candidate_note_line_count / candidate_nonblank_count, 4),
                    "excerpt": lines[candidate_start].strip()[:160],
                }
            )

        previous_chapter_index = chapter_index

    if not removed_blocks:
        return text, []

    sanitized_lines: list[str] = []
    for line, keep in zip(lines, keep_line):
        if not keep:
            continue
        if CHAPTER_HEADING_PATTERN.match(line) and sanitized_lines and sanitized_lines[-1].strip():
            sanitized_lines.append("")
        sanitized_lines.append(line)

    sanitized_text = "\n".join(sanitized_lines)
    if text.endswith("\n"):
        sanitized_text += "\n"
    return sanitized_text, removed_blocks


def prepare_input_text(
    input_path: Path,
    work_dir: Path,
    *,
    sanitize_prechapter_note_blocks_enabled: bool,
    sanitize_prechapter_note_lookback: int,
    sanitize_prechapter_note_min_lines: int,
    sanitize_prechapter_note_min_ratio: float,
) -> tuple[Path, dict[str, object]]:
    if not sanitize_prechapter_note_blocks_enabled:
        return input_path, {"enabled": False, "changed": False, "removed_blocks": []}

    original_text = input_path.read_text(encoding="utf-8")
    sanitized_text, removed_blocks = sanitize_prechapter_note_blocks(
        original_text,
        max_lookback=sanitize_prechapter_note_lookback,
        min_note_lines=sanitize_prechapter_note_min_lines,
        min_note_ratio=sanitize_prechapter_note_min_ratio,
    )
    if not removed_blocks:
        return input_path, {"enabled": True, "changed": False, "removed_blocks": []}

    prepared_dir = work_dir / "_prepared_input"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / input_path.name
    prepared_path.write_text(sanitized_text, encoding="utf-8")
    return prepared_path, {
        "enabled": True,
        "changed": True,
        "prepared_input": str(prepared_path),
        "removed_blocks": removed_blocks,
    }


def audiobook_output_path(output_dir: Path, audiobook_stem: str, model_name: str) -> Path:
    return output_dir / f"{audiobook_stem}_{model_name}.wav"


def legacy_audiobook_output_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / f"{model_name}.wav"


def chunk_input_text(
    input_path: Path,
    chunk_root: Path,
    *,
    chunk_size: int,
    min_chunk_size: int,
    chunk_workers: int,
) -> tuple[Path, Path, list[Path], dict]:
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunker = WebTTSChunker(
        max_chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        output_dir=str(chunk_root),
    )
    manifest_path_str, plan_path_str = chunker.process_file(
        input_file=str(input_path),
        prefix=input_path.stem,
        num_workers=chunk_workers,
    )

    manifest_path = Path(manifest_path_str)
    plan_path = Path(plan_path_str)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = sorted(manifest["chunks"], key=lambda chunk: int(chunk["chunk_id"]))
    chunk_files = [chunk_root / chunk["filename"] for chunk in chunks]

    for chunk_file in chunk_files:
        ensure_path(chunk_file, "Chunk text file")

    return manifest_path, plan_path, chunk_files, manifest


def load_chunk_manifest(
    input_path: Path,
    chunk_root: Path,
    *,
    chunk_size: int,
    min_chunk_size: int,
) -> tuple[Path, Path, list[Path], dict]:
    manifest_path = chunk_root / f"{input_path.stem}_manifest.json"
    plan_path = chunk_root / "processing_plan.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Chunk manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    source_text_sha256 = compute_source_text_sha256(input_path)
    if metadata.get("max_chunk_size") != chunk_size:
        raise FileNotFoundError(
            f"Chunk manifest max_chunk_size={metadata.get('max_chunk_size')} does not match requested {chunk_size}"
        )
    if metadata.get("min_chunk_size") != min_chunk_size:
        raise FileNotFoundError(
            f"Chunk manifest min_chunk_size={metadata.get('min_chunk_size')} does not match requested {min_chunk_size}"
        )
    if metadata.get("source_file") != input_path.name:
        raise FileNotFoundError(
            f"Chunk manifest source_file={metadata.get('source_file')} does not match requested {input_path.name}"
        )
    if metadata.get("source_text_sha256") != source_text_sha256:
        raise FileNotFoundError(
            "Chunk manifest source_text_sha256 does not match the requested input file"
        )
    chunks = sorted(manifest["chunks"], key=lambda chunk: int(chunk["chunk_id"]))
    chunk_files = [chunk_root / chunk["filename"] for chunk in chunks]

    for chunk_file in chunk_files:
        ensure_path(chunk_file, "Chunk text file")

    return manifest_path, plan_path, chunk_files, manifest


def build_runner_args(
    model_name: str,
    *,
    reference_audio: Path,
    reference_text_file: Path | None,
) -> list[str]:
    spec = MODEL_SPECS[model_name]
    runner_args = list(spec.runner_args)
    if model_name in {"qwen3_faster_clone", "qwen3_faster_clone_xvec"}:
        apply_qwen_env_overrides(runner_args, batch=False)
        runner_args.extend(["--reference-audio", str(reference_audio)])
        if reference_text_file is not None:
            runner_args.extend(["--reference-text-file", str(reference_text_file)])
    return runner_args


def set_arg_value(args: list[str], flag: str, value: str | None) -> None:
    if value is None or value == "":
        return
    if flag in args:
        index = args.index(flag)
        if index + 1 >= len(args):
            raise ValueError(f"Runner flag {flag} is missing a value")
        args[index + 1] = value
    else:
        args.extend([flag, value])


def apply_qwen_env_overrides(runner_args: list[str], *, batch: bool) -> None:
    set_arg_value(runner_args, "--temperature", os.environ.get("QWEN_TEMPERATURE"))
    set_arg_value(runner_args, "--top-k", os.environ.get("QWEN_TOP_K"))
    set_arg_value(runner_args, "--top-p", os.environ.get("QWEN_TOP_P"))
    set_arg_value(runner_args, "--max-new-tokens", os.environ.get("QWEN_MAX_NEW_TOKENS"))
    set_arg_value(runner_args, "--max-seq-len", os.environ.get("QWEN_MAX_SEQ_LEN"))
    if batch:
        set_arg_value(runner_args, "--workers", os.environ.get("QWEN_BATCH_WORKERS"))


def build_batch_runner_args(
    model_name: str,
    *,
    reference_audio: Path,
    reference_text_file: Path | None,
) -> list[str]:
    spec = MODEL_SPECS[model_name]
    if spec.batch_runner_script is None:
        raise ValueError(f"{model_name} does not define a batch runner")

    runner_args = list(spec.batch_runner_args)
    if model_name in {"qwen3_faster_clone", "qwen3_faster_clone_xvec"}:
        apply_qwen_env_overrides(runner_args, batch=True)
        runner_args.extend(["--reference-audio", str(reference_audio)])
        if reference_text_file is not None:
            runner_args.extend(["--reference-text-file", str(reference_text_file)])
    return runner_args


def resolve_cuda_visible_devices() -> list[str]:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw_value and raw_value.lower() != "all":
        return [
            device
            for device in (part.strip() for part in raw_value.split(","))
            if device and device.lower() not in {"none"}
        ]

    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    return [str(index) for index in range(torch.cuda.device_count())]


def is_valid_wav(path: Path) -> bool:
    return path.exists() and path.stat().st_size > MIN_VALID_WAV_BYTES


def audit_summary_has_truncated_chunks(
    audit_summary: dict,
    *,
    min_expected_ratio: float = 0.80,
    min_actual_ratio_for_truncation: float = 0.85,
    decoder_ceiling_seconds: float = 158.0,
    min_expected_ratio_hard: float = 0.65,
) -> bool:
    for chunk in audit_summary.get("chunks", []):
        if chunk.get("blocking_reasons"):
            return True
        try:
            duration = float(chunk.get("duration_seconds", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        if decoder_ceiling_seconds > 0 and duration >= decoder_ceiling_seconds:
            return True
        score = chunk.get("current_score") or chunk.get("best_candidate") or {}
        try:
            expected_ratio = float(score.get("expected_ratio", 1.0))
            actual_ratio = float(score.get("actual_ratio", 0.0))
        except (TypeError, ValueError):
            continue
        if min_expected_ratio_hard > 0 and expected_ratio < min_expected_ratio_hard:
            return True
        if (
            expected_ratio < min_expected_ratio
            and actual_ratio >= min_actual_ratio_for_truncation
        ):
            return True
    return False


def load_existing_render_result(
    model_name: str,
    chunk_files: list[Path],
    *,
    audiobook_stem: str,
    model_root: Path,
    output_dir: Path,
    require_audit: bool,
) -> dict | None:
    model_dir = model_root / model_name
    manifest_path = model_dir / "render_manifest.json"
    final_output = audiobook_output_path(output_dir, audiobook_stem, model_name)
    legacy_output = legacy_audiobook_output_path(output_dir, model_name)

    existing_output: Path | None = None
    if is_valid_wav(final_output):
        existing_output = final_output
    elif is_valid_wav(legacy_output):
        legacy_output.replace(final_output)
        existing_output = final_output

    if not manifest_path.exists() or existing_output is None:
        return None

    try:
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if result.get("status") != "ok":
        return None
    if int(result.get("chunk_count", -1)) != len(chunk_files):
        return None

    chunk_outputs = [Path(raw_path) for raw_path in result.get("chunk_outputs", [])]
    if len(chunk_outputs) != len(chunk_files):
        return None
    if not all(is_valid_wav(path) for path in chunk_outputs):
        return None
    if require_audit:
        audit_summary_path = result.get("audit_summary_path")
        if not audit_summary_path:
            return None
        audit_path = Path(audit_summary_path)
        if not audit_path.exists():
            return None
        try:
            audit_summary = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if audit_summary.get("blocking_issue_count", 0):
            return None
        if audit_summary_has_truncated_chunks(audit_summary):
            return None

    reused_result = dict(result)
    reused_result["final_output"] = str(existing_output)
    reused_result["chunk_count"] = len(chunk_files)
    if result.get("final_output") != reused_result["final_output"]:
        result["final_output"] = reused_result["final_output"]
        manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return reused_result


def write_model_script(
    model_name: str,
    chunk_files: list[Path],
    *,
    model_root: Path,
    chunk_root: Path,
    reference_audio: Path,
    reference_text_file: Path | None,
    force: bool,
    restrict_to_chunk_subset: bool,
    script_filename: str = "run_chunks.sh",
    worker_manifest_filename: str = "batch_worker_manifest.json",
) -> tuple[Path, list[Path]]:
    spec = MODEL_SPECS[model_name]
    model_dir = model_root / model_name
    chunk_audio_dir = model_dir / "chunks"
    logs_dir = model_dir / "logs"
    script_path = model_dir / script_filename
    batch_mode = spec.batch_runner_script is not None
    runner_script = spec.batch_runner_script if batch_mode else spec.runner_script
    worker_manifest_path = logs_dir / worker_manifest_filename

    chunk_outputs = [chunk_audio_dir / f"{chunk_text.stem}.wav" for chunk_text in chunk_files]
    runner_args = (
        build_batch_runner_args(
            model_name,
            reference_audio=reference_audio,
            reference_text_file=reference_text_file,
        )
        if batch_mode
        else build_runner_args(
            model_name,
            reference_audio=reference_audio,
            reference_text_file=reference_text_file,
        )
    )

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        'export PATH="/usr/bin:/bin:/usr/local/bin:$PATH"',
    ]
    lines.extend(
        [
            "source scripts/mahti/common.sh",
            'export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"',
            'export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"',
            'export MAHTI_MODEL_ROOT="$MAHTI_WORK_ROOT/models"',
            'export MAHTI_CACHE_ROOT="$MAHTI_WORK_ROOT/cache"',
        ]
    )
    lines.extend(
        [
            f"source {shlex.quote(str(spec.bootstrap_script.relative_to(REPO_ROOT)))}",
            f"runner_script={shlex.quote(str(runner_script.relative_to(REPO_ROOT)))}",
            "runner_args=(",
        ]
    )
    lines.extend(f"  {shlex.quote(arg)}" for arg in runner_args)
    lines.extend(
        [
            ")",
            f"force_flag={1 if force else 0}",
            f"logs_dir={shlex.quote(str(logs_dir))}",
            f"model_run_log={shlex.quote(str(logs_dir / 'model_run.log'))}",
            "chunk_texts=(",
        ]
    )
    lines.extend(f"  {shlex.quote(str(path))}" for path in chunk_files)
    lines.extend(
        [
            ")",
            "chunk_outputs=(",
        ]
    )
    lines.extend(f"  {shlex.quote(str(path))}" for path in chunk_outputs)
    if batch_mode:
        lines.extend(
            [
                ")",
                f"input_dir={shlex.quote(str(chunk_root))}",
                f"output_dir={shlex.quote(str(chunk_audio_dir))}",
                'if [[ "$force_flag" -eq 1 ]]; then',
                '  rm -f "${chunk_outputs[@]}"',
                "fi",
                'echo "[run] batch: $(basename "$runner_script")"',
            'for chunk_output in "${chunk_outputs[@]}"; do',
                '  if [[ ! -s "$chunk_output" ]]; then',
                '    echo "[error] missing chunk output: $chunk_output"',
                "    exit 1",
                "  fi",
                "done",
            ]
        )
        if restrict_to_chunk_subset:
            worker_manifest_path.write_text(
                json.dumps([str(path) for path in chunk_files], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            manifest_line = f"worker_manifest={shlex.quote(str(worker_manifest_path))}"
            run_line = (
                'python -u "$runner_script" --input-dir "$input_dir" --output-dir "$output_dir" '
                '--prefix "chunk_" --log-dir "$logs_dir" --worker-manifest "$worker_manifest" '
                '"${runner_args[@]}" >"$model_run_log" 2>&1'
            )
            lines.insert(lines.index('if [[ "$force_flag" -eq 1 ]]; then'), manifest_line)
            lines.insert(lines.index('for chunk_output in "${chunk_outputs[@]}"; do'), run_line)
        else:
            run_line = (
                'python -u "$runner_script" --input-dir "$input_dir" --output-dir "$output_dir" '
                '--prefix "chunk_" --log-dir "$logs_dir" "${runner_args[@]}" >"$model_run_log" 2>&1'
            )
            lines.insert(lines.index('for chunk_output in "${chunk_outputs[@]}"; do'), run_line)
    else:
        lines.extend(
            [
                ")",
                "chunk_logs=(",
            ]
        )
        lines.extend(f"  {shlex.quote(str(path))}" for path in [logs_dir / f"{chunk_text.stem}.log" for chunk_text in chunk_files])
        lines.extend(
            [
                ")",
                'for i in "${!chunk_texts[@]}"; do',
                '  chunk_text="${chunk_texts[$i]}"',
                '  chunk_output="${chunk_outputs[$i]}"',
                '  chunk_log="${chunk_logs[$i]}"',
                '  if [[ "$force_flag" -eq 0 && -s "$chunk_output" ]]; then',
                '    echo "[skip] chunk ${i}: $(basename "$chunk_output")"',
                "    continue",
                "  fi",
                '  echo "[run] chunk ${i}: $(basename "$chunk_text")"',
                '  python -u "$runner_script" --input "$chunk_text" --output "$chunk_output" "${runner_args[@]}" >"$chunk_log" 2>&1',
                "done",
            ]
        )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return script_path, chunk_outputs


PARAGRAPH_PAUSE_SECONDS = 0.6


def join_wavs(
    chunk_wavs: Iterable[Path],
    output_path: Path,
    *,
    pause_after: list[bool] | None = None,
    pause_seconds: float = PARAGRAPH_PAUSE_SECONDS,
) -> Path:
    """Concatenate chunk wavs into one file.

    pause_after[i] (if provided) inserts pause_seconds of silence between
    chunk i and chunk i+1 -- used to mark real paragraph boundaries so
    narration pacing isn't uniformly back-to-back regardless of structure.
    """
    chunk_list = list(chunk_wavs)
    if not chunk_list:
        raise ValueError("No chunk wavs were provided for joining")

    params = None
    signature = None
    frames: list[bytes] = []
    for index, wav_path in enumerate(chunk_list):
        ensure_path(wav_path, "Chunk audio")
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
            elif current_signature != signature:
                raise ValueError(
                    "WAV format mismatch in "
                    f"{wav_path}: expected {signature}, got {current_signature}"
                )
            frames.append(reader.readframes(reader.getnframes()))

        if (
            pause_after is not None
            and index < len(pause_after)
            and pause_after[index]
            and index < len(chunk_list) - 1
        ):
            silence_frame_count = int(round(current_params.framerate * pause_seconds))
            silence_bytes = silence_frame_count * current_params.nchannels * current_params.sampwidth
            frames.append(b"\x00" * silence_bytes)

    if params is None:
        raise ValueError("Could not read WAV parameters from chunk audio")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(params.nchannels)
        writer.setsampwidth(params.sampwidth)
        writer.setframerate(params.framerate)
        writer.setcomptype(params.comptype, params.compname)
        writer.writeframes(b"".join(frames))
    return output_path


def archive_failed_chunk_artifacts(
    *,
    blocking_indices: list[int],
    chunk_index_to_files: dict[int, tuple[Path, Path]],
    audit_summary: dict,
    corrected_chunk_text_dir: Path,
    archive_root: Path,
    attempt_number: int,
) -> Path:
    """Copy (never move) each blocking chunk's audio WAV, source text, corrected
    text, and per-chunk audit diagnostic into chunk_audit/original_failed_chunks/
    attempt_<N>/ before any resynthesis attempt overwrites that chunk index.

    This runs before the resynthesis subprocess for the attempt, so the archive
    is written even if that resynthesis attempt (or a later heal attempt) fails --
    the evidence of what went wrong is never destroyed.
    """
    attempt_dir = archive_root / f"attempt_{attempt_number}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    chunk_result_by_index = {
        int(chunk["chunk_index"]): chunk for chunk in audit_summary.get("chunks", [])
    }

    for chunk_index in blocking_indices:
        text_path, audio_path = chunk_index_to_files[chunk_index]

        if audio_path.exists():
            shutil.copy2(audio_path, attempt_dir / audio_path.name)
        if text_path.exists():
            shutil.copy2(text_path, attempt_dir / text_path.name)

        corrected_path = corrected_chunk_text_dir / text_path.name
        if corrected_path.exists():
            shutil.copy2(corrected_path, attempt_dir / f"{text_path.stem}.corrected.txt")

        chunk_result = chunk_result_by_index.get(chunk_index)
        if chunk_result is not None:
            diagnostic_path = attempt_dir / f"{text_path.stem}.audit_result.json"
            diagnostic_path.write_text(
                json.dumps(chunk_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    return attempt_dir


def heal_blocking_chunk_indices(
    *,
    model_name: str,
    audit_summary: dict,
    chunk_files: list[Path],
    expected_outputs: list[Path],
    model_root: Path,
    reference_audio: Path,
    reference_text_file: Path | None,
    audit_root: Path,
    corrected_chunk_text_dir: Path,
    audit_asr_model: str,
    audit_asr_device: str,
    audit_neighbor_window: int,
    audit_cache_dir: Path | None,
    audit_workers: int,
    max_heal_attempts: int,
    cuda_visible_device: str | None,
) -> dict:
    """Resynthesize only the chunk indices the alignment audit flagged as
    blocking, splicing the healed WAV(s) back in at the same chunk_audio_dir
    paths the rest of the (unchanged) chunks already occupy, then re-run the
    full-set alignment audit to confirm they now pass. Retries up to
    max_heal_attempts times per still-blocking chunk.

    Returns the final passing audit_summary (with a "heal_attempts_used" key
    added) on success. Raises RuntimeError naming the chunk indices that could
    not be healed and where the archived original failed-attempt artifacts live
    if attempts are exhausted while chunks are still blocking.
    """
    spec = MODEL_SPECS[model_name]
    model_dir = model_root / model_name
    # render_model() already creates these before it ever reaches the audit
    # step, but this function must also be safe to call on its own (as the
    # tests do), so don't rely on that.
    (model_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (model_dir / "logs").mkdir(parents=True, exist_ok=True)
    archive_root = audit_root / "original_failed_chunks"
    effective_cache_dir = (
        audit_cache_dir / model_name if audit_cache_dir is not None else audit_root / "asr_cache"
    )

    chunk_index_to_files: dict[int, tuple[Path, Path]] = {
        chunk_index_from_path(text_path): (text_path, audio_path)
        for text_path, audio_path in zip(chunk_files, expected_outputs)
    }

    current_audit_summary = audit_summary
    for attempt_number in range(1, max_heal_attempts + 1):
        blocking_indices = list(current_audit_summary["blocking_chunk_indices"])

        archive_dir = archive_failed_chunk_artifacts(
            blocking_indices=blocking_indices,
            chunk_index_to_files=chunk_index_to_files,
            audit_summary=current_audit_summary,
            corrected_chunk_text_dir=corrected_chunk_text_dir,
            archive_root=archive_root,
            attempt_number=attempt_number,
        )
        print(
            f"[heal] {spec.label}: attempt {attempt_number}/{max_heal_attempts} "
            f"resynthesizing {len(blocking_indices)} blocking chunk(s) "
            f"({', '.join(str(index) for index in blocking_indices[:20])}); "
            f"original artifacts archived to {archive_dir}"
        )

        heal_chunk_files = [chunk_index_to_files[index][0] for index in blocking_indices]
        heal_script_path, heal_chunk_outputs = write_model_script(
            model_name,
            heal_chunk_files,
            model_root=model_root,
            chunk_root=heal_chunk_files[0].parent,
            reference_audio=reference_audio,
            reference_text_file=reference_text_file,
            force=True,
            restrict_to_chunk_subset=True,
            script_filename=f"run_chunks_heal_attempt_{attempt_number}.sh",
            worker_manifest_filename=f"batch_worker_manifest_heal_attempt_{attempt_number}.json",
        )

        run_env = os.environ.copy()
        if cuda_visible_device is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        heal_log_path = model_dir / "logs" / f"heal_attempt_{attempt_number}.log"
        with heal_log_path.open("w", encoding="utf-8") as log_handle:
            subprocess.run(
                ["bash", str(heal_script_path)],
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=run_env,
                check=True,
            )
        for heal_output in heal_chunk_outputs:
            ensure_path(heal_output, "Healed chunk audio output")

        current_audit_summary = audit_chunk_alignment(
            chunk_files,
            expected_outputs,
            output_root=audit_root,
            transcription_model=audit_asr_model,
            device=audit_asr_device,
            cache_dir=effective_cache_dir,
            neighbor_window=audit_neighbor_window,
            workers=audit_workers,
        )

        if current_audit_summary["blocking_issue_count"] == 0:
            print(
                f"[heal] {spec.label}: attempt {attempt_number}/{max_heal_attempts} "
                "succeeded -- all previously blocking chunk(s) now pass the alignment audit"
            )
            healed_summary = dict(current_audit_summary)
            healed_summary["heal_attempts_used"] = attempt_number
            return healed_summary

    still_blocking = list(current_audit_summary["blocking_chunk_indices"])
    raise RuntimeError(
        f"{spec.label} chunk audit still failed for chunk indices "
        f"{', '.join(str(index) for index in still_blocking[:20])} "
        f"after {max_heal_attempts} heal attempt(s) (including the original synthesis, "
        f"{max_heal_attempts + 1} total tries per chunk). Original failed-attempt "
        f"artifacts for inspection are archived under {archive_root}"
    )


def render_model(
    model_name: str,
    chunk_files: list[Path],
    *,
    audiobook_stem: str,
    model_root: Path,
    output_dir: Path,
    reference_audio: Path,
    reference_text_file: Path | None,
    force: bool,
    dry_run: bool,
    restrict_batch_to_manifest_subset: bool,
    audit_chunks: bool,
    audit_asr_model: str,
    audit_asr_device: str,
    audit_neighbor_window: int,
    audit_cache_dir: Path | None,
    audit_workers: int,
    heal_blocking_chunks: bool = True,
    max_heal_attempts: int = DEFAULT_MAX_HEAL_ATTEMPTS,
    cuda_visible_device: str | None = None,
    chunk_ends_at_paragraph: list[bool] | None = None,
) -> dict:
    spec = MODEL_SPECS[model_name]
    model_dir = model_root / model_name
    chunk_audio_dir = model_dir / "chunks"
    logs_dir = model_dir / "logs"
    final_output = audiobook_output_path(output_dir, audiobook_stem, model_name)
    legacy_output = legacy_audiobook_output_path(output_dir, model_name)
    manifest_path = model_dir / "render_manifest.json"
    model_run_log = logs_dir / "model_run.log"

    chunk_audio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if force and final_output.exists():
        final_output.unlink()
    if force and legacy_output.exists():
        legacy_output.unlink()
    if force and manifest_path.exists():
        manifest_path.unlink()

    script_path, chunk_outputs = write_model_script(
        model_name,
        chunk_files,
        model_root=model_root,
        chunk_root=chunk_files[0].parent,
        reference_audio=reference_audio,
        reference_text_file=reference_text_file,
        force=force,
        restrict_to_chunk_subset=restrict_batch_to_manifest_subset,
    )

    if dry_run:
        print(f"[dry-run] {script_path}")
        expected_outputs = chunk_outputs
    else:
        print(f"[run] {spec.label}: {script_path.name}")
        run_env = os.environ.copy()
        if cuda_visible_device is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        with model_run_log.open("w", encoding="utf-8") as log_handle:
            subprocess.run(
                ["bash", str(script_path)],
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=run_env,
                check=True,
            )
        expected_outputs = chunk_outputs
        for chunk_output in expected_outputs:
            ensure_path(chunk_output, "Chunk audio output")

    result = {
        "model": model_name,
        "label": spec.label,
        "chunk_count": len(chunk_files),
        "chunk_outputs": [str(path) for path in expected_outputs],
        "final_output": str(final_output),
        "script_path": str(script_path),
        "model_run_log": str(model_run_log),
        "runner_script": str(
            (spec.batch_runner_script if spec.batch_runner_script is not None else spec.runner_script).relative_to(REPO_ROOT)
        ),
        "runner_args": (
            build_batch_runner_args(
                model_name,
                reference_audio=reference_audio,
                reference_text_file=reference_text_file,
            )
            if spec.batch_runner_script is not None
            else build_runner_args(
                model_name,
                reference_audio=reference_audio,
                reference_text_file=reference_text_file,
            )
        ),
        "status": "dry_run" if dry_run else "ok",
    }

    if dry_run:
        manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    if len(expected_outputs) != len(chunk_files):
        missing = len(chunk_files) - len(expected_outputs)
        raise RuntimeError(f"{spec.label} did not produce all chunk wavs ({missing} missing)")

    audit_summary_path = model_dir / "chunk_audit" / "audit_summary.json"
    corrected_chunk_text_dir = model_dir / "chunk_audit" / "corrected_chunk_text"
    if audit_chunks:
        audit_root = model_dir / "chunk_audit"
        effective_cache_dir = audit_cache_dir / model_name if audit_cache_dir is not None else audit_root / "asr_cache"
        audit_summary = audit_chunk_alignment(
            chunk_files,
            expected_outputs,
            output_root=audit_root,
            transcription_model=audit_asr_model,
            device=audit_asr_device,
            cache_dir=effective_cache_dir,
            neighbor_window=audit_neighbor_window,
            workers=audit_workers,
        )
        if audit_summary["blocking_issue_count"] > 0:
            if heal_blocking_chunks:
                print(
                    f"[audit] {spec.label}: {audit_summary['blocking_issue_count']} blocking "
                    "chunk(s) detected on first pass "
                    f"({', '.join(str(index) for index in audit_summary['blocking_chunk_indices'][:20])}); "
                    "attempting chunk-level self-heal instead of failing the whole render"
                )
                try:
                    audit_summary = heal_blocking_chunk_indices(
                        model_name=model_name,
                        audit_summary=audit_summary,
                        chunk_files=chunk_files,
                        expected_outputs=expected_outputs,
                        model_root=model_root,
                        reference_audio=reference_audio,
                        reference_text_file=reference_text_file,
                        audit_root=audit_root,
                        corrected_chunk_text_dir=corrected_chunk_text_dir,
                        audit_asr_model=audit_asr_model,
                        audit_asr_device=audit_asr_device,
                        audit_neighbor_window=audit_neighbor_window,
                        audit_cache_dir=audit_cache_dir,
                        audit_workers=audit_workers,
                        max_heal_attempts=max_heal_attempts,
                        cuda_visible_device=cuda_visible_device,
                    )
                except Exception:
                    result["status"] = "audit_failed"
                    result["audit_summary_path"] = str(audit_summary_path)
                    result["corrected_chunk_text_dir"] = str(corrected_chunk_text_dir)
                    manifest_path.write_text(
                        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    raise
            else:
                result["status"] = "audit_failed"
                result["audit_summary_path"] = str(audit_summary_path)
                result["corrected_chunk_text_dir"] = str(corrected_chunk_text_dir)
                result["blocking_chunk_indices"] = audit_summary["blocking_chunk_indices"]
                result["trimmed_chunk_indices"] = audit_summary["trimmed_chunk_indices"]
                manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                raise RuntimeError(
                    f"{spec.label} chunk audit failed for chunk indices: "
                    f"{', '.join(str(index) for index in audit_summary['blocking_chunk_indices'][:20])}"
                )

        result["audit_summary_path"] = str(audit_summary_path)
        result["corrected_chunk_text_dir"] = str(corrected_chunk_text_dir)
        result["blocking_chunk_indices"] = audit_summary["blocking_chunk_indices"]
        result["trimmed_chunk_indices"] = audit_summary["trimmed_chunk_indices"]
        if audit_summary.get("heal_attempts_used"):
            result["heal_attempts_used"] = audit_summary["heal_attempts_used"]

    join_wavs(expected_outputs, final_output, pause_after=chunk_ends_at_paragraph)
    if legacy_output.exists():
        legacy_output.unlink()
    result["status"] = "ok"
    result["final_output"] = str(final_output)
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def cleanup_work_dir(work_dir: Path) -> None:
    if work_dir.exists():
        for child in sorted(work_dir.glob("*")):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def main() -> int:
    args = parse_args()
    ensure_path(args.input, "Input txt")
    if args.audit_chunks and not args.dry_run:
        ensure_audit_dependencies_available()

    audiobook_stem = args.input.stem
    output_dir = args.output_dir
    work_dir = args.work_dir / audiobook_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    effective_input_path, input_sanitization = prepare_input_text(
        args.input,
        work_dir,
        sanitize_prechapter_note_blocks_enabled=args.sanitize_prechapter_note_blocks,
        sanitize_prechapter_note_lookback=args.sanitize_prechapter_note_lookback,
        sanitize_prechapter_note_min_lines=args.sanitize_prechapter_note_min_lines,
        sanitize_prechapter_note_min_ratio=args.sanitize_prechapter_note_min_ratio,
    )
    if input_sanitization.get("changed"):
        removed_blocks = input_sanitization["removed_blocks"]
        print(
            "Prepared sanitized input: "
            f"{effective_input_path} "
            f"({len(removed_blocks)} pre-chapter note block(s) removed)"
        )

    chunk_root = work_dir / "chunks"
    if args.force and chunk_root.exists():
        shutil.rmtree(chunk_root)

    if not args.force:
        try:
            manifest_path, plan_path, chunk_files, manifest = load_chunk_manifest(
                effective_input_path,
                chunk_root,
                chunk_size=args.chunk_size,
                min_chunk_size=args.min_chunk_size,
            )
            print(f"Reusing existing chunk manifest: {manifest_path}")
        except FileNotFoundError as exc:
            if chunk_root.exists():
                print(
                    "Existing chunk manifest cannot be reused; "
                    f"removing stale chunk tree before rebuilding: {chunk_root} ({exc})"
                )
                shutil.rmtree(chunk_root)
            manifest_path, plan_path, chunk_files, manifest = chunk_input_text(
                effective_input_path,
                chunk_root,
                chunk_size=args.chunk_size,
                min_chunk_size=args.min_chunk_size,
                chunk_workers=args.chunk_workers,
            )
    else:
        manifest_path, plan_path, chunk_files, manifest = chunk_input_text(
            effective_input_path,
            chunk_root,
            chunk_size=args.chunk_size,
            min_chunk_size=args.min_chunk_size,
            chunk_workers=args.chunk_workers,
        )
    total_chunk_count = len(chunk_files)

    if args.max_chunks is not None:
        if args.max_chunks < 1:
            raise ValueError("--max-chunks must be at least 1")
        chunk_files = chunk_files[: args.max_chunks]

    if args.max_heal_attempts < 0:
        raise ValueError("--max-heal-attempts must be at least 0")

    # Which chunk boundaries land on a real paragraph break (set by the
    # paragraph-aware chunker) -- used to insert a pause between that
    # chunk's audio and the next one, so narration pacing reflects the
    # book's actual paragraph structure instead of running on uniformly.
    chunk_ends_at_paragraph = [
        bool(chunk.get("ends_at_paragraph", False)) for chunk in manifest["chunks"]
    ][: len(chunk_files)]

    print(f"Chunk manifest: {manifest_path}")
    print(f"Chunk plan: {plan_path}")
    print(f"Chunk count: {total_chunk_count}")
    if args.max_chunks is not None:
        print(f"Chunk render limit: {args.max_chunks}")
        print(f"Chunk render count: {len(chunk_files)}")
    print(f"Chunk output dir: {chunk_root}")

    run_summaries_by_model: dict[str, dict] = {}
    models_to_render: list[str] = []
    if not args.force and not args.dry_run:
        for model_name in args.models:
            existing_result = load_existing_render_result(
                model_name,
                chunk_files,
                audiobook_stem=audiobook_stem,
                model_root=work_dir,
                output_dir=output_dir,
                require_audit=args.audit_chunks,
            )
            if existing_result is not None:
                print(
                    f"Reusing existing render result for {model_name}: "
                    f"{existing_result['final_output']}"
                )
                run_summaries_by_model[model_name] = existing_result
            else:
                model_dir = work_dir / model_name
                if model_dir.exists():
                    print(
                        f"Existing render state for {model_name} is not reusable; "
                        f"PRESERVING {model_dir} (skip cleanup; batch runner skips existing wavs)"
                    )
                    # cleanup_work_dir(model_dir)  # disabled 2026-11-14 resume
                models_to_render.append(model_name)
    else:
        models_to_render = list(args.models)

    if args.parallel_models and len(models_to_render) > 1:
        cuda_devices = resolve_cuda_visible_devices()
        if len(cuda_devices) < len(models_to_render):
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            raise RuntimeError(
                "--parallel-models requires one visible CUDA device per model; "
                f"found {len(cuda_devices)} from CUDA_VISIBLE_DEVICES={visible!r}"
            )

        if args.force:
            for model_name in models_to_render:
                model_dir = work_dir / model_name
                if model_dir.exists():
                    cleanup_work_dir(model_dir)

        results_by_model: dict[str, dict] = {}
        errors: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(max_workers=len(args.models)) as executor:
            futures = {
                executor.submit(
                    render_model,
                    model_name,
                    chunk_files,
                    audiobook_stem=audiobook_stem,
                    model_root=work_dir,
                    output_dir=output_dir,
                    reference_audio=args.reference_audio,
                    reference_text_file=args.reference_text_file,
                    force=args.force,
                    dry_run=args.dry_run,
                    restrict_batch_to_manifest_subset=args.max_chunks is not None,
                    audit_chunks=args.audit_chunks,
                    audit_asr_model=args.audit_asr_model,
                    audit_asr_device=args.audit_asr_device,
                    audit_neighbor_window=args.audit_neighbor_window,
                    audit_cache_dir=args.audit_cache_dir,
                    audit_workers=args.audit_workers,
                    heal_blocking_chunks=args.heal_blocking_chunks,
                    max_heal_attempts=args.max_heal_attempts,
                    cuda_visible_device=cuda_devices[index],
                    chunk_ends_at_paragraph=chunk_ends_at_paragraph,
                ): model_name
                for index, model_name in enumerate(models_to_render)
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    results_by_model[model_name] = future.result()
                except Exception as exc:
                    errors.append((model_name, exc))

        if errors:
            details = "; ".join(f"{model}: {exc}" for model, exc in errors)
            raise RuntimeError(f"Parallel model rendering failed: {details}")

        run_summaries_by_model.update(results_by_model)
    else:
        for model_name in models_to_render:
            if args.force:
                model_dir = work_dir / model_name
                if model_dir.exists():
                    cleanup_work_dir(model_dir)
            run_summaries_by_model[model_name] = render_model(
                model_name,
                chunk_files,
                audiobook_stem=audiobook_stem,
                model_root=work_dir,
                output_dir=output_dir,
                reference_audio=args.reference_audio,
                reference_text_file=args.reference_text_file,
                force=args.force,
                dry_run=args.dry_run,
                restrict_batch_to_manifest_subset=args.max_chunks is not None,
                audit_chunks=args.audit_chunks,
                audit_asr_model=args.audit_asr_model,
                audit_asr_device=args.audit_asr_device,
                audit_neighbor_window=args.audit_neighbor_window,
                audit_cache_dir=args.audit_cache_dir,
                audit_workers=args.audit_workers,
                heal_blocking_chunks=args.heal_blocking_chunks,
                max_heal_attempts=args.max_heal_attempts,
                chunk_ends_at_paragraph=chunk_ends_at_paragraph,
            )

    run_summaries = [run_summaries_by_model[model_name] for model_name in args.models]

    summary = {
        "input": str(args.input),
        "effective_input": str(effective_input_path),
        "input_sanitization": input_sanitization,
        "chunk_manifest": str(manifest_path),
        "chunk_plan": str(plan_path),
        "chunk_count": total_chunk_count,
        "render_chunk_count": len(chunk_files),
        "max_chunks": args.max_chunks,
        "chunk_metadata": manifest.get("metadata", {}),
        "paragraph_pause_seconds": PARAGRAPH_PAUSE_SECONDS,
        "models": run_summaries,
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
    }
    (work_dir / "audiobook_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.cleanup_work and not args.dry_run:
        cleanup_work_dir(work_dir)

    print("Audiobook build complete")
    for item in run_summaries:
        print(f"  {item['model']}: {item['final_output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
