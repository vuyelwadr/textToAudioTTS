#!/usr/bin/env python3
"""Audit audiobook chunk text/audio alignment before join or video generation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
import os
import re
from pathlib import Path
from typing import Iterable

from make_autoscroll_video import (
    ALIGNMENT_TOKEN_PATTERN,
    load_chunk_audio_path_entries,
    load_chunk_text_entries,
    normalize_alignment_token,
    probe_audio_duration,
    resolve_asr_device,
    transcribe_audio_words,
)


@dataclass(frozen=True)
class TokenSpan:
    normalized: str
    start: int
    end: int


@dataclass(frozen=True)
class AlignmentScore:
    label: str
    matched_tokens: int
    expected_token_count: int
    actual_token_count: int
    expected_ratio: float
    actual_ratio: float
    score: float
    first_expected_index: int | None
    first_actual_index: int | None
    last_expected_index: int | None
    last_actual_index: int | None


def tokenize_text_with_spans(text: str) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for match in ALIGNMENT_TOKEN_PATTERN.finditer(text):
        normalized = normalize_alignment_token(match.group(0))
        if not normalized:
            continue
        spans.append(TokenSpan(normalized=normalized, start=match.start(), end=match.end()))
    return spans


def score_alignment(expected_tokens: list[str], actual_tokens: list[str], *, label: str) -> AlignmentScore:
    matcher = SequenceMatcher(a=expected_tokens, b=actual_tokens, autojunk=False)
    matched_tokens = 0
    first_expected_index: int | None = None
    first_actual_index: int | None = None
    last_expected_index: int | None = None
    last_actual_index: int | None = None

    for block in matcher.get_matching_blocks():
        if block.size <= 0:
            continue
        matched_tokens += block.size
        if first_expected_index is None:
            first_expected_index = block.a
            first_actual_index = block.b
        last_expected_index = block.a + block.size - 1
        last_actual_index = block.b + block.size - 1

    expected_ratio = matched_tokens / max(1, len(expected_tokens))
    actual_ratio = matched_tokens / max(1, len(actual_tokens))
    if expected_ratio <= 0.0 or actual_ratio <= 0.0:
        score = 0.0
    else:
        score = 2.0 * expected_ratio * actual_ratio / (expected_ratio + actual_ratio)

    return AlignmentScore(
        label=label,
        matched_tokens=matched_tokens,
        expected_token_count=len(expected_tokens),
        actual_token_count=len(actual_tokens),
        expected_ratio=expected_ratio,
        actual_ratio=actual_ratio,
        score=score,
        first_expected_index=first_expected_index,
        first_actual_index=first_actual_index,
        last_expected_index=last_expected_index,
        last_actual_index=last_actual_index,
    )


def safe_excerpt(text: str, *, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def write_corrected_text(
    output_path: Path,
    original_text: str,
    *,
    trim_start_char: int | None,
) -> str:
    corrected_text = original_text
    if trim_start_char is not None and trim_start_char > 0:
        corrected_text = corrected_text[trim_start_char:].lstrip()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(corrected_text, encoding="utf-8")
    return corrected_text


def _chunk_index_from_path(path: Path) -> int:
    match = re.match(r"^chunk_(\d+)_[0-9a-f]+$", path.stem)
    if not match:
        raise ValueError(f"Could not parse chunk index from {path}")
    return int(match.group(1))


def _audit_chunk_task(
    task: dict[str, object],
    *,
    transcription_model: str,
    device: str,
    cache_dir: str,
    previous_tail_tokens: int,
    trim_min_chars: int,
    trim_min_tokens: int,
    candidate_margin: float,
    replay_margin: float,
    min_current_score: float,
    min_expected_ratio: float,
    min_actual_ratio_for_truncation: float,
    decoder_ceiling_seconds: float,
    min_expected_ratio_hard: float,
) -> dict[str, object]:
    if device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    chunk_index = int(task["chunk_index"])
    audio_path = Path(task["audio_path"])
    text_path = Path(task["text_path"])
    corrected_path = Path(task["corrected_text_path"])
    current_text = str(task["current_text"])
    current_tokens = list(task["current_tokens"])
    current_token_spans = [tuple(item) for item in task["current_token_spans"]]
    candidate_token_map: dict[str, list[str]] = {
        label: list(tokens)
        for label, tokens in dict(task["candidate_token_map"]).items()
    }
    previous_tokens = list(task.get("previous_tokens") or [])

    timed_words = transcribe_audio_words(
        audio_path,
        model_name=transcription_model,
        device=device,
        cache_dir=Path(cache_dir),
    )
    transcript_tokens = [word.normalized for word in timed_words]
    transcript_text = " ".join(word.text for word in timed_words)
    duration = probe_audio_duration(audio_path)

    current_score = score_alignment(current_tokens, transcript_tokens, label="current")
    candidate_scores: dict[str, AlignmentScore] = {"current": current_score}
    for candidate_label, candidate_tokens in candidate_token_map.items():
        if candidate_label == "current":
            continue
        candidate_scores[candidate_label] = score_alignment(candidate_tokens, transcript_tokens, label=candidate_label)

    prefix_replay = None
    previous_tail_count = 0
    if previous_tokens:
        previous_tail_tokens_list = previous_tokens[-previous_tail_tokens:]
        previous_tail_count = len(previous_tail_tokens_list)
        combined_tokens = previous_tail_tokens_list + current_tokens
        combined_score = score_alignment(
            combined_tokens,
            transcript_tokens,
            label=f"previous_tail_{int(task['previous_chunk_index'])}_plus_current",
        )
        candidate_scores[combined_score.label] = combined_score
        if (
            previous_tail_count > 0
            and combined_score.score >= current_score.score + replay_margin
            and combined_score.first_expected_index is not None
            and combined_score.first_expected_index < previous_tail_count
        ):
            prefix_replay = {
                "previous_chunk_index": int(task["previous_chunk_index"]),
                "previous_tail_token_count": previous_tail_count,
                "first_expected_index": combined_score.first_expected_index,
                "score_delta": combined_score.score - current_score.score,
            }

    best_candidate = max(candidate_scores.values(), key=lambda item: item.score)
    better_neighbor_index = None
    if best_candidate.label.startswith("neighbor_") and best_candidate.score >= current_score.score + candidate_margin:
        better_neighbor_index = int(best_candidate.label.split("_", 1)[1])

    trim_start_char = None
    trim_start_token = None
    if current_score.first_expected_index is not None and current_score.first_expected_index < len(current_token_spans):
        trim_start_token = current_score.first_expected_index
        trim_start_char = int(current_token_spans[current_score.first_expected_index][0])

    blocking_reasons: list[str] = []
    status = "ok"
    if better_neighbor_index is not None:
        status = "wrong_chunk_match"
        blocking_reasons.append(f"audio matched chunk {better_neighbor_index} better than current chunk {chunk_index}")
    elif prefix_replay is not None:
        status = "replays_previous_chunk"
        blocking_reasons.append(f"audio begins with previous chunk {prefix_replay['previous_chunk_index']} tail")
    elif decoder_ceiling_seconds > 0 and duration >= decoder_ceiling_seconds:
        # Qwen3-TTS stops at its fixed 2048 audio-token ceiling (163.84 s at
        # 12.5 Hz). A chunk at/above that duration almost always means the
        # decoder ran away or never emitted EOS, so the audio is unusable even
        # when the partial transcript happens to align.
        status = "decoder_ceiling"
        blocking_reasons.append(
            f"audio duration {duration:.2f}s reached the decoder ceiling "
            f"(>= {decoder_ceiling_seconds:.2f}s); generation likely ran away or was truncated"
        )
    elif current_score.score < min_current_score:
        status = "low_match"
        blocking_reasons.append(f"current chunk score {current_score.score:.3f} below threshold {min_current_score:.3f}")
    elif min_expected_ratio_hard > 0 and current_score.expected_ratio < min_expected_ratio_hard:
        # A runaway / heavily damaged take can have a low transcript
        # self-match ratio too, which makes the truncation branch below miss
        # it. Coverage this low means large parts of the chunk text were never
        # spoken, so regenerate regardless of the transcript self-match.
        status = "low_coverage"
        blocking_reasons.append(
            "audio covered too little of the expected chunk text "
            f"({current_score.expected_ratio:.3f} < {min_expected_ratio_hard:.3f})"
        )
    elif (
        current_score.expected_ratio < min_expected_ratio
        and current_score.actual_ratio >= min_actual_ratio_for_truncation
    ):
        status = "truncated_current_chunk"
        blocking_reasons.append(
            "audio covered too little of the expected chunk text "
            f"({current_score.expected_ratio:.3f} < {min_expected_ratio:.3f}) "
            "while still matching its own transcript strongly "
            f"({current_score.actual_ratio:.3f} >= {min_actual_ratio_for_truncation:.3f})"
        )
    elif (
        trim_start_char is not None
        and trim_start_char >= trim_min_chars
        and trim_start_token is not None
        and trim_start_token >= trim_min_tokens
    ):
        status = "trimmed_start"

    corrected_text = write_corrected_text(
        corrected_path,
        current_text,
        trim_start_char=trim_start_char if status == "trimmed_start" else None,
    )

    return {
        "chunk_index": chunk_index,
        "audio_path": str(audio_path),
        "text_path": str(text_path),
        "corrected_text_path": str(corrected_path),
        "duration_seconds": duration,
        "status": status,
        "blocking_reasons": blocking_reasons,
        "best_candidate": asdict(best_candidate),
        "current_score": asdict(current_score),
        "candidate_scores": {label: asdict(score) for label, score in candidate_scores.items()},
        "prefix_replay": prefix_replay,
        "trim_start_char": trim_start_char if status == "trimmed_start" else None,
        "trim_start_token": trim_start_token if status == "trimmed_start" else None,
        "transcript_excerpt": safe_excerpt(transcript_text),
        "corrected_text_excerpt": safe_excerpt(corrected_text),
    }


def audit_chunk_alignment(
    chunk_text_paths: Iterable[Path],
    chunk_audio_paths: Iterable[Path],
    *,
    output_root: Path,
    transcription_model: str = "small.en",
    device: str = "auto",
    cache_dir: Path | None = None,
    neighbor_window: int = 1,
    previous_tail_tokens: int = 160,
    trim_min_chars: int = 80,
    trim_min_tokens: int = 6,
    candidate_margin: float = 0.08,
    replay_margin: float = 0.05,
    min_current_score: float = 0.70,
    min_expected_ratio: float = 0.80,
    min_actual_ratio_for_truncation: float = 0.85,
    decoder_ceiling_seconds: float = 158.0,
    min_expected_ratio_hard: float = 0.65,
    progress_every: int = 10,
    workers: str | int = "auto",
) -> dict:
    text_paths = sorted((Path(path) for path in chunk_text_paths), key=_chunk_index_from_path)
    audio_paths = sorted((Path(path) for path in chunk_audio_paths), key=_chunk_index_from_path)
    if not text_paths:
        raise RuntimeError("No chunk text paths were provided for audit")
    if not audio_paths:
        raise RuntimeError("No chunk audio paths were provided for audit")

    text_by_index = {_chunk_index_from_path(path): path for path in text_paths}
    audio_by_index = {_chunk_index_from_path(path): path for path in audio_paths}
    shared_indices = sorted(index for index in text_by_index if index in audio_by_index)
    if not shared_indices:
        raise RuntimeError("No overlapping chunk indices were found for audit")

    resolved_device = resolve_asr_device(device)
    report_dir = output_root
    corrected_text_dir = report_dir / "corrected_chunk_text"
    report_dir.mkdir(parents=True, exist_ok=True)
    corrected_text_dir.mkdir(parents=True, exist_ok=True)
    effective_cache_dir = cache_dir if cache_dir is not None else report_dir / "asr_cache"
    effective_cache_dir.mkdir(parents=True, exist_ok=True)

    text_entries: dict[int, dict[str, object]] = {}
    for chunk_index in shared_indices:
        text = text_by_index[chunk_index].read_text(encoding="utf-8")
        token_spans = tokenize_text_with_spans(text)
        text_entries[chunk_index] = {
            "path": text_by_index[chunk_index],
            "text": text,
            "token_spans": token_spans,
            "tokens": [item.normalized for item in token_spans],
        }

    results: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    for chunk_index in shared_indices:
        text_entry = text_entries[chunk_index]
        candidate_token_map = {"current": list(text_entry["tokens"])}
        for neighbor_offset in range(-neighbor_window, neighbor_window + 1):
            if neighbor_offset == 0:
                continue
            candidate_index = chunk_index + neighbor_offset
            candidate_entry = text_entries.get(candidate_index)
            if candidate_entry is None:
                continue
            candidate_token_map[f"neighbor_{candidate_index}"] = list(candidate_entry["tokens"])

        previous_entry = text_entries.get(chunk_index - 1)
        tasks.append(
            {
                "chunk_index": chunk_index,
                "audio_path": str(audio_by_index[chunk_index]),
                "text_path": str(text_by_index[chunk_index]),
                "corrected_text_path": str(corrected_text_dir / text_by_index[chunk_index].name),
                "current_text": str(text_entry["text"]),
                "current_tokens": list(text_entry["tokens"]),
                "current_token_spans": [(item.start, item.end) for item in text_entry["token_spans"]],
                "candidate_token_map": candidate_token_map,
                "previous_chunk_index": chunk_index - 1,
                "previous_tokens": list(previous_entry["tokens"]) if previous_entry is not None else [],
            }
        )

    if isinstance(workers, str):
        raw_workers = workers.strip().lower()
    else:
        raw_workers = str(workers)

    if raw_workers == "auto":
        if resolved_device == "cpu":
            cpu_count = os.cpu_count() or 1
            worker_count = min(len(tasks), max(1, min(cpu_count - 1, 8)))
        else:
            worker_count = 1
    else:
        worker_count = max(1, min(int(raw_workers), len(tasks)))
    if worker_count == 1:
        for position, task in enumerate(tasks, start=1):
            if progress_every > 0 and (position == 1 or position % progress_every == 0 or position == len(tasks)):
                print(f"[audit] chunk {position}/{len(tasks)} -> {task['chunk_index']}", flush=True)
            results.append(
                _audit_chunk_task(
                    task,
                    transcription_model=transcription_model,
                    device=resolved_device,
                    cache_dir=str(effective_cache_dir),
                    previous_tail_tokens=previous_tail_tokens,
                    trim_min_chars=trim_min_chars,
                    trim_min_tokens=trim_min_tokens,
                    candidate_margin=candidate_margin,
                    replay_margin=replay_margin,
                    min_current_score=min_current_score,
                    min_expected_ratio=min_expected_ratio,
                    min_actual_ratio_for_truncation=min_actual_ratio_for_truncation,
                    decoder_ceiling_seconds=decoder_ceiling_seconds,
                    min_expected_ratio_hard=min_expected_ratio_hard,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _audit_chunk_task,
                    task,
                    transcription_model=transcription_model,
                    device=resolved_device,
                    cache_dir=str(effective_cache_dir),
                    previous_tail_tokens=previous_tail_tokens,
                    trim_min_chars=trim_min_chars,
                    trim_min_tokens=trim_min_tokens,
                    candidate_margin=candidate_margin,
                    replay_margin=replay_margin,
                    min_current_score=min_current_score,
                    min_expected_ratio=min_expected_ratio,
                    min_actual_ratio_for_truncation=min_actual_ratio_for_truncation,
                    decoder_ceiling_seconds=decoder_ceiling_seconds,
                    min_expected_ratio_hard=min_expected_ratio_hard,
                )
                for task in tasks
            ]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if progress_every > 0 and (completed == 1 or completed % progress_every == 0 or completed == len(futures)):
                    print(f"[audit] completed {completed}/{len(futures)} chunks", flush=True)
                results.append(future.result())

    results.sort(key=lambda item: int(item["chunk_index"]))
    blocking_chunk_indices = [int(item["chunk_index"]) for item in results if item["blocking_reasons"]]
    trimmed_chunk_indices = [int(item["chunk_index"]) for item in results if item["status"] == "trimmed_start"]

    summary = {
        "chunk_count": len(results),
        "blocking_issue_count": len(blocking_chunk_indices),
        "blocking_chunk_indices": blocking_chunk_indices,
        "trimmed_chunk_count": len(trimmed_chunk_indices),
        "trimmed_chunk_indices": trimmed_chunk_indices,
        "transcription_model": transcription_model,
        "device": resolved_device,
        "corrected_text_dir": str(corrected_text_dir),
        "thresholds": {
            "min_current_score": min_current_score,
            "min_expected_ratio": min_expected_ratio,
            "min_actual_ratio_for_truncation": min_actual_ratio_for_truncation,
            "decoder_ceiling_seconds": decoder_ceiling_seconds,
            "min_expected_ratio_hard": min_expected_ratio_hard,
        },
        "chunks": results,
    }
    summary_path = report_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit audiobook chunk alignment against generated chunk audio.")
    parser.add_argument("--chunk-text-dir", type=Path, required=True, help="Directory containing chunk_XXXX text files")
    parser.add_argument("--chunk-audio-dir", type=Path, required=True, help="Directory containing chunk_XXXX wav files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for audit report and corrected text")
    parser.add_argument("--asr-model", default="small.en", help="faster-whisper model name")
    parser.add_argument("--asr-device", default="auto", choices=("auto", "cpu", "cuda"), help="ASR device")
    parser.add_argument("--asr-cache-dir", type=Path, default=None, help="Optional ASR cache directory")
    parser.add_argument("--neighbor-window", type=int, default=1, help="Neighbor chunk window to compare against")
    parser.add_argument("--previous-tail-tokens", type=int, default=160, help="How many tokens from the previous chunk tail to test for replay")
    parser.add_argument("--trim-min-chars", type=int, default=80, help="Minimum leading character skip before emitting trimmed video text")
    parser.add_argument("--trim-min-tokens", type=int, default=6, help="Minimum leading token skip before emitting trimmed video text")
    parser.add_argument("--min-current-score", type=float, default=0.70, help="Minimum current-chunk F1 score before the chunk is marked blocking")
    parser.add_argument(
        "--min-expected-ratio",
        type=float,
        default=0.80,
        help="Minimum fraction of expected chunk tokens that must appear when the transcript otherwise matches strongly",
    )
    parser.add_argument(
        "--min-actual-ratio-for-truncation",
        type=float,
        default=0.85,
        help="Minimum transcript self-match ratio before a low expected-ratio chunk is treated as truncation",
    )
    parser.add_argument(
        "--decoder-ceiling-seconds",
        type=float,
        default=158.0,
        help=(
            "Mark a chunk blocking when its audio reaches this duration. Qwen3-TTS "
            "has a fixed 2048-token / 163.84 s cap; near-cap audio means the take "
            "ran away. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--min-expected-ratio-hard",
        type=float,
        default=0.65,
        help=(
            "Mark a chunk blocking when this fraction of its expected words is "
            "missing, regardless of transcript self-match. Set 0 to disable."
        ),
    )
    parser.add_argument("--workers", default="auto", help="Parallel worker count for chunk audit, or 'auto'")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text_paths = sorted(args.chunk_text_dir.glob("chunk_*.txt"), key=_chunk_index_from_path)
    audio_paths = [path for _, path in load_chunk_audio_path_entries(args.chunk_audio_dir)]

    summary = audit_chunk_alignment(
        text_paths,
        audio_paths,
        output_root=args.output_dir,
        transcription_model=args.asr_model,
        device=args.asr_device,
        cache_dir=args.asr_cache_dir,
        neighbor_window=args.neighbor_window,
        previous_tail_tokens=args.previous_tail_tokens,
        trim_min_chars=args.trim_min_chars,
        trim_min_tokens=args.trim_min_tokens,
        min_current_score=args.min_current_score,
        min_expected_ratio=args.min_expected_ratio,
        min_actual_ratio_for_truncation=args.min_actual_ratio_for_truncation,
        decoder_ceiling_seconds=args.decoder_ceiling_seconds,
        min_expected_ratio_hard=args.min_expected_ratio_hard,
        workers=args.workers,
    )
    print(json.dumps({
        "chunk_count": summary["chunk_count"],
        "blocking_issue_count": summary["blocking_issue_count"],
        "blocking_chunk_indices": summary["blocking_chunk_indices"][:20],
        "trimmed_chunk_count": summary["trimmed_chunk_count"],
        "trimmed_chunk_indices": summary["trimmed_chunk_indices"][:20],
        "corrected_text_dir": summary["corrected_text_dir"],
    }, indent=2))
    return 0 if summary["blocking_issue_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
