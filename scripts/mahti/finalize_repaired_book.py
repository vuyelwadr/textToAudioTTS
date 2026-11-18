#!/usr/bin/env python3
"""Targeted re-audit + finalize for a book whose only remaining problem is a
handful of already-repaired chunks.

Two build_chunked_audiobook.py behaviors combine to make every audit-repair
round redo the entire book instead of just the fixed chunks:

1. repair_qwen_audit_chunks.py --mode finalize rewrites a repaired chunk's
   WAV but never touches chunk_audit/audit_summary.json, so its
   blocking_issue_count/blocking_chunk_indices stay stale (still showing the
   pre-repair failures).
2. render_model() writes render_manifest.json with status="audit_failed"
   the moment any chunk audit fails, and raises before ever reaching
   join_wavs() -- so final_output (the joined book WAV) never gets created
   for a book that has ever had a bad audit round. On the next full run,
   load_existing_render_result() sees status != "ok" (or, once that's
   patched, sees no final_output on disk) and refuses to reuse anything --
   build_chunked_audiobook.py then wipes model_dir (chunks/, chunk_audit/,
   the ASR cache, render_manifest.json) and regenerates all chunk audio via
   fresh TTS sampling, which re-rolls the model's random repetition-loop
   failure mode across every chunk, including ones that were already fine.

This script re-audits only the chunks that were just repaired (plus a
neighbor window, needed as alignment context by audit_chunk_alignment),
merges those results into the existing audit_summary.json, and -- if that
leaves zero blocking chunks -- performs the one remaining step render_model()
never got to run: joins the existing per-chunk WAVs (290-plus already-good,
handful freshly repaired) into final_output via the same join_wavs() used by
the normal pipeline, and rewrites render_manifest.json to status="ok" with a
cleared blocking list. The next full pipeline run then sees a valid,
already-complete render and reuses it instead of rebuilding from scratch.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_chunked_audiobook import is_valid_wav, join_wavs  # noqa: E402
from chunk_alignment_audit import audit_chunk_alignment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="e.g. .../theGayScience/qwen3_faster_clone (contains render_manifest.json, chunk_audit/, chunks/)",
    )
    parser.add_argument("--repaired-indices", required=True, help="comma-separated chunk indices")
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--transcription-model", default="small.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", default="auto")
    return parser.parse_args()


def find_chunk_path(directory: Path, index: int, suffix: str) -> Path | None:
    matches = list(directory.glob(f"chunk_{index:04d}_*{suffix}"))
    return matches[0] if matches else None


def load_chunk_ends_at_paragraph(work_dir: Path, chunk_output_count: int) -> list[bool]:
    candidates = list((work_dir / "chunks").glob("*_manifest.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one chunk manifest under {work_dir / 'chunks'}, found {candidates}")
    chunk_manifest = json.loads(candidates[0].read_text(encoding="utf-8"))
    return [bool(chunk.get("ends_at_paragraph", False)) for chunk in chunk_manifest["chunks"]][:chunk_output_count]


def main() -> int:
    args = parse_args()
    repaired_indices = sorted(int(part) for part in args.repaired_indices.split(",") if part.strip())
    if not repaired_indices:
        raise ValueError("--repaired-indices produced no indices")

    model_dir = args.model_dir
    work_dir = model_dir.parent
    chunk_text_dir = work_dir / "chunks"
    chunk_audio_dir = model_dir / "chunks"
    audit_dir = model_dir / "chunk_audit"
    audit_summary_path = audit_dir / "audit_summary.json"
    real_corrected_text_dir = audit_dir / "corrected_chunk_text"
    real_asr_cache_dir = audit_dir / "asr_cache"
    render_manifest_path = model_dir / "render_manifest.json"

    audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    existing_chunks = {int(item["chunk_index"]): item for item in audit_summary["chunks"]}

    audit_pool = sorted(
        {
            idx
            for repaired in repaired_indices
            for idx in range(repaired - args.neighbor_window, repaired + args.neighbor_window + 1)
        }
    )
    text_paths = []
    audio_paths = []
    for idx in audit_pool:
        text_path = find_chunk_path(chunk_text_dir, idx, ".txt")
        audio_path = find_chunk_path(chunk_audio_dir, idx, ".wav")
        if text_path is None or audio_path is None:
            continue
        text_paths.append(text_path)
        audio_paths.append(audio_path)

    real_corrected_text_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=audit_dir, prefix="targeted_patch_tmp_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        mini_summary = audit_chunk_alignment(
            text_paths,
            audio_paths,
            output_root=tmp_root,
            transcription_model=args.transcription_model,
            device=args.device,
            cache_dir=real_asr_cache_dir,
            neighbor_window=args.neighbor_window,
            workers=args.workers,
        )
        mini_by_index = {int(item["chunk_index"]): item for item in mini_summary["chunks"]}

        still_blocking = []
        for idx in repaired_indices:
            fresh = mini_by_index.get(idx)
            if fresh is None:
                raise RuntimeError(f"Targeted audit produced no result for repaired chunk {idx}")
            tmp_corrected_path = Path(fresh["corrected_text_path"])
            real_corrected_path = real_corrected_text_dir / tmp_corrected_path.name
            real_corrected_path.write_text(tmp_corrected_path.read_text(encoding="utf-8"), encoding="utf-8")
            fresh["corrected_text_path"] = str(real_corrected_path)
            existing_chunks[idx] = fresh
            if fresh.get("blocking_reasons"):
                still_blocking.append(idx)

    merged_chunks = [existing_chunks[idx] for idx in sorted(existing_chunks)]
    blocking_chunk_indices = [int(item["chunk_index"]) for item in merged_chunks if item.get("blocking_reasons")]
    trimmed_chunk_indices = [int(item["chunk_index"]) for item in merged_chunks if item.get("status") == "trimmed_start"]

    audit_summary["chunks"] = merged_chunks
    audit_summary["blocking_chunk_indices"] = blocking_chunk_indices
    audit_summary["blocking_issue_count"] = len(blocking_chunk_indices)
    audit_summary["trimmed_chunk_indices"] = trimmed_chunk_indices
    audit_summary["trimmed_chunk_count"] = len(trimmed_chunk_indices)
    audit_summary_path.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"targeted audit checked chunks: {repaired_indices}")
    print(f"still blocking after repair: {still_blocking}")
    print(f"merged blocking_issue_count: {audit_summary['blocking_issue_count']}")

    if still_blocking:
        print("Not finalizing: some repaired chunks are still blocking.")
        return 2

    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    chunk_outputs = [Path(path) for path in render_manifest["chunk_outputs"]]
    invalid = [str(path) for path in chunk_outputs if not is_valid_wav(path)]
    if invalid:
        raise RuntimeError(f"Refusing to join: {len(invalid)} chunk outputs are missing/invalid: {invalid[:10]}")

    final_output = Path(render_manifest["final_output"])
    if final_output.exists():
        print(f"final_output already exists, leaving as-is: {final_output}")
    else:
        chunk_ends_at_paragraph = load_chunk_ends_at_paragraph(work_dir, len(chunk_outputs))
        join_wavs(chunk_outputs, final_output, pause_after=chunk_ends_at_paragraph)
        print(f"joined {len(chunk_outputs)} chunks -> {final_output}")

    render_manifest["status"] = "ok"
    render_manifest["blocking_chunk_indices"] = []
    render_manifest["trimmed_chunk_indices"] = trimmed_chunk_indices
    render_manifest_path.write_text(json.dumps(render_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"render_manifest.json patched to status=ok: {render_manifest_path}")
    print("SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
