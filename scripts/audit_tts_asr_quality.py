#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import soundfile as sf
from faster_whisper import WhisperModel


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe TTS WAVs and score them against a reference text."
    )
    parser.add_argument("--reference-text", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--max-window-hypothesis-words",
        type=int,
        default=120,
        help="Use full-reference WER instead of expensive sliding windows above this hypothesis length.",
    )
    parser.add_argument("audio", nargs="+", type=Path)
    return parser.parse_args()


def normalize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (token_a != token_b),
                )
            )
        previous = current
    return previous[-1]


def best_window_wer(reference: list[str], hypothesis: list[str]) -> dict[str, object]:
    if not hypothesis:
        return {
            "scoring_mode": "best_window",
            "wer": 1.0,
            "start_word": 0,
            "end_word": 0,
            "matched_reference": "",
        }

    if not reference:
        return {
            "scoring_mode": "best_window",
            "wer": 1.0,
            "start_word": 0,
            "end_word": 0,
            "matched_reference": "",
        }

    min_window = max(1, int(len(hypothesis) * 0.75))
    max_window = min(len(reference), max(min_window, int(len(hypothesis) * 1.25) + 3))
    best: tuple[float, int, int] | None = None

    for window_size in range(min_window, max_window + 1):
        for start in range(0, len(reference) - window_size + 1):
            end = start + window_size
            distance = edit_distance(reference[start:end], hypothesis)
            wer = distance / max(window_size, 1)
            if best is None or wer < best[0]:
                best = (wer, start, end)

    assert best is not None
    wer, start, end = best
    return {
        "scoring_mode": "best_window",
        "wer": round(wer, 4),
        "start_word": start,
        "end_word": end,
        "matched_reference": " ".join(reference[start:end]),
    }


def full_reference_wer(reference: list[str], hypothesis: list[str]) -> dict[str, object]:
    distance = edit_distance(reference, hypothesis)
    excerpt_words = reference[: min(len(reference), 120)]
    return {
        "scoring_mode": "full_reference",
        "wer": round(distance / max(len(reference), 1), 4),
        "start_word": 0,
        "end_word": len(reference),
        "matched_reference": " ".join(excerpt_words),
    }


def prefix_wer(reference: list[str], hypothesis: list[str]) -> float:
    if not hypothesis:
        return 1.0
    prefix = reference[: len(hypothesis)]
    return round(edit_distance(prefix, hypothesis) / max(len(hypothesis), 1), 4)


def transcribe_audio(
    model: WhisperModel, audio_path: Path, beam_size: int
) -> tuple[str, list[dict[str, object]]]:
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=beam_size,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segment_rows: list[dict[str, object]] = []
    texts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        texts.append(text)
        segment_rows.append(
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": text,
            }
        )
    return " ".join(texts).strip(), segment_rows


def iter_audio_paths(paths: Iterable[Path]) -> list[Path]:
    audio_paths: list[Path] = []
    for path in paths:
        if path.is_dir():
            audio_paths.extend(sorted(path.glob("*.wav")))
        else:
            audio_paths.append(path)
    return audio_paths


def write_payload(
    *,
    output_json: Path,
    reference_text: Path,
    model: str,
    device: str,
    compute_type: str,
    results: list[dict[str, object]],
) -> None:
    payload = {
        "reference_text": str(reference_text),
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "results": results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    reference_text = args.reference_text.read_text(encoding="utf-8")
    reference_words = normalize_words(reference_text)
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    audio_paths = iter_audio_paths(args.audio)
    results: list[dict[str, object]] = []
    for index, audio_path in enumerate(audio_paths, start=1):
        print(
            f"[asr-audit] {index}/{len(audio_paths)} transcribing {audio_path}",
            file=sys.stderr,
            flush=True,
        )
        info = sf.info(str(audio_path))
        transcript, segments = transcribe_audio(model, audio_path, args.beam_size)
        hypothesis_words = normalize_words(transcript)
        if len(hypothesis_words) > args.max_window_hypothesis_words:
            window = full_reference_wer(reference_words, hypothesis_words)
        else:
            window = best_window_wer(reference_words, hypothesis_words)
        results.append(
            {
                "audio_path": str(audio_path),
                "duration_sec": round(float(info.duration), 4),
                "sample_rate": int(info.samplerate),
                "bytes": audio_path.stat().st_size,
                "transcript": transcript,
                "hypothesis_words": len(hypothesis_words),
                "reference_words": len(reference_words),
                "coverage_ratio": round(
                    len(hypothesis_words) / max(len(reference_words), 1), 4
                ),
                "prefix_wer": prefix_wer(reference_words, hypothesis_words),
                "best_window": window,
                "segments": segments,
            }
        )
        write_payload(
            output_json=args.output_json,
            reference_text=args.reference_text,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            results=results,
        )

    payload = json.loads(args.output_json.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
