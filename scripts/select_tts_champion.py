#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best TTS output from an ASR quality audit JSON."
    )
    parser.add_argument("--asr-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-duration-sec", type=float, default=15.0)
    parser.add_argument("--min-coverage", type=float, default=0.9)
    parser.add_argument("--max-wer", type=float, default=0.05)
    parser.add_argument("--max-prefix-wer", type=float, default=0.08)
    return parser.parse_args()


def fail_reasons(result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    wer = float(result.get("best_window", {}).get("wer", 1.0))
    prefix_wer = float(result.get("prefix_wer", 1.0))
    coverage = float(result.get("coverage_ratio", 0.0))
    duration = float(result.get("duration_sec", 0.0))

    if duration < args.min_duration_sec:
        reasons.append(f"duration {duration:.2f}s < {args.min_duration_sec:.2f}s")
    if coverage < args.min_coverage:
        reasons.append(f"coverage {coverage:.4f} < {args.min_coverage:.4f}")
    if wer > args.max_wer:
        reasons.append(f"WER {wer:.4f} > {args.max_wer:.4f}")
    if prefix_wer > args.max_prefix_wer:
        reasons.append(f"prefix WER {prefix_wer:.4f} > {args.max_prefix_wer:.4f}")
    return reasons


def score_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(result.get("best_window", {}).get("wer", 1.0)),
        float(result.get("prefix_wer", 1.0)),
        -float(result.get("coverage_ratio", 0.0)),
        -float(result.get("duration_sec", 0.0)),
    )


def compact_result(result: dict[str, Any], *, passed: bool, reasons: list[str]) -> dict[str, Any]:
    return {
        "audio_path": result.get("audio_path"),
        "passed": passed,
        "fail_reasons": reasons,
        "duration_sec": result.get("duration_sec"),
        "coverage_ratio": result.get("coverage_ratio"),
        "prefix_wer": result.get("prefix_wer"),
        "wer": result.get("best_window", {}).get("wer"),
        "hypothesis_words": result.get("hypothesis_words"),
        "reference_words": result.get("reference_words"),
        "transcript_preview": str(result.get("transcript", ""))[:240],
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.asr_json.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    evaluated: list[dict[str, Any]] = []
    passing: list[dict[str, Any]] = []

    for result in results:
        reasons = fail_reasons(result, args)
        passed = not reasons
        compact = compact_result(result, passed=passed, reasons=reasons)
        evaluated.append(compact)
        if passed:
            passing.append(result)

    selected = min(passing, key=score_key) if passing else None
    output = {
        "asr_json": str(args.asr_json),
        "thresholds": {
            "min_duration_sec": args.min_duration_sec,
            "min_coverage": args.min_coverage,
            "max_wer": args.max_wer,
            "max_prefix_wer": args.max_prefix_wer,
        },
        "selected": compact_result(selected, passed=True, reasons=[]) if selected else None,
        "evaluated": sorted(evaluated, key=lambda row: (not row["passed"], row["wer"] or 1.0)),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    if selected:
        print(f"PASS {selected['audio_path']}")
    else:
        print("FAIL no candidate met the quality gate")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
