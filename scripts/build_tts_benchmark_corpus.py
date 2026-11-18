#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "tts_naturalness"
DEFAULT_INPUTS_DIR = EXPERIMENT_ROOT / "inputs"
DEFAULT_REFERENCE_AUDIO = REPO_ROOT / "assets" / "audio_samples" / "brothersKarmatzov.mp3"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUTS_DIR / "passage_bank"
DEFAULT_TARGET_WORDS = 900
DEFAULT_MIN_WORDS = 800
DEFAULT_MAX_WORDS = 1100
DEFAULT_PARAGRAPH_WORDS = 180
DEFAULT_MAX_GENERATED_PASSAGES = 3

sys.path.insert(0, str(REPO_ROOT / "scripts" / "mahti"))
from common import auto_transcribe_reference, load_text  # noqa: E402


INPUT_METADATA: dict[str, tuple[str, str]] = {
    "naturalness_prompt.txt": ("smoke", "Short smoke check."),
    "long_audiobook_test.txt": ("long_narration", "Original long narration baseline."),
    "dialogue_test.txt": ("dialogue_stress", "Short dialogue stress test."),
    "description_test.txt": ("description_stress", "Short description stress test."),
    "descriptive_test_long.txt": ("long_description", "Long descriptive stress test."),
    "dialogue_test_long.txt": ("long_dialogue", "Long dialogue stress test."),
    "narrative_test_long.txt": ("long_narrative", "Long narrative stress test."),
}


@dataclass(frozen=True)
class PassageRecord:
    passage_id: str
    order: int
    source_type: str
    source_path: Path
    role: str
    notes: str
    generated: bool
    word_count: int
    paragraph_count: int
    target_word_count: int | None = None

    def to_manifest_entry(self) -> dict[str, object]:
        entry: dict[str, object] = {
            "passage_id": self.passage_id,
            "order": self.order,
            "source_type": self.source_type,
            "source_path": relpath(self.source_path),
            "role": self.role,
            "notes": self.notes,
            "generated": self.generated,
            "word_count": self.word_count,
            "paragraph_count": self.paragraph_count,
        }
        if self.target_word_count is not None:
            entry["target_word_count"] = self.target_word_count
        return entry


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<![.!?])\n", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) > 1:
        return sentences

    if count_words(text) <= DEFAULT_PARAGRAPH_WORDS:
        return [text.strip()]

    words = text.split()
    chunk_size = max(40, DEFAULT_PARAGRAPH_WORDS // 2)
    return [" ".join(words[index : index + chunk_size]).strip() for index in range(0, len(words), chunk_size)]


def split_into_paragraphs(text: str, target_words: int) -> list[str]:
    normalized = normalize_text(text)
    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    result: list[str] = []
    for paragraph in paragraphs:
        sentence_buffer: list[str] = []
        sentence_words = 0
        for sentence in split_sentences(paragraph):
            sentence_word_count = count_words(sentence)
            if sentence_word_count > target_words:
                if sentence_buffer:
                    result.append(" ".join(sentence_buffer).strip())
                    sentence_buffer = []
                    sentence_words = 0

                words = sentence.split()
                for index in range(0, len(words), target_words):
                    chunk = " ".join(words[index : index + target_words]).strip()
                    if chunk:
                        result.append(chunk)
                continue

            if sentence_buffer and sentence_words + sentence_word_count > target_words and sentence_words >= max(40, target_words // 2):
                result.append(" ".join(sentence_buffer).strip())
                sentence_buffer = [sentence]
                sentence_words = sentence_word_count
                continue

            sentence_buffer.append(sentence)
            sentence_words += sentence_word_count

        if sentence_buffer:
            result.append(" ".join(sentence_buffer).strip())

    return [paragraph for paragraph in result if paragraph.strip()]


def pack_paragraphs_into_passages(
    paragraphs: list[str],
    min_words: int,
    target_words: int,
    max_words: int,
    max_passages: int,
) -> list[str]:
    passages: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_words = count_words(paragraph)

        if current and current_words + paragraph_words > max_words and current_words >= min_words:
            passages.append("\n\n".join(current).strip())
            current = []
            current_words = 0

        current.append(paragraph)
        current_words += paragraph_words

        if current_words >= target_words and current_words >= min_words:
            passages.append("\n\n".join(current).strip())
            current = []
            current_words = 0

    if current:
        if passages and current_words < min_words:
            passages[-1] = f"{passages[-1]}\n\n" + "\n\n".join(current).strip()
        else:
            passages.append("\n\n".join(current).strip())

    passages = [passage for passage in passages if passage.strip()]
    if len(passages) > max_passages:
        merged: list[str] = []
        group_size = math.ceil(len(passages) / max_passages)
        for index in range(0, len(passages), group_size):
            merged.append("\n\n".join(passages[index : index + group_size]).strip())
        passages = merged[:max_passages]

    return passages


def count_paragraphs(text: str) -> int:
    normalized = normalize_text(text)
    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
    return max(1, len(paragraphs))


def load_existing_input_records(inputs_dir: Path, start_order: int = 1) -> tuple[list[PassageRecord], int]:
    files = sorted(
        (path for path in inputs_dir.glob("*.txt") if path.is_file()),
        key=lambda path: (preferred_input_order(path.name), path.name),
    )

    records: list[PassageRecord] = []
    order = start_order
    for path in files:
        text = normalize_text(load_text(path))
        role, notes = INPUT_METADATA.get(path.name, ("benchmark", "Existing benchmark input."))
        records.append(
            PassageRecord(
                passage_id=path.stem,
                order=order,
                source_type="existing_input",
                source_path=path,
                role=role,
                notes=notes,
                generated=False,
                word_count=count_words(text),
                paragraph_count=count_paragraphs(text),
            )
        )
        order += 1

    return records, order


def preferred_input_order(filename: str) -> int:
    order = [
        "naturalness_prompt.txt",
        "long_audiobook_test.txt",
        "dialogue_test.txt",
        "description_test.txt",
        "descriptive_test_long.txt",
        "dialogue_test_long.txt",
        "narrative_test_long.txt",
    ]
    try:
        return order.index(filename)
    except ValueError:
        return len(order)


def build_generated_records(
    transcript_text: str,
    output_dir: Path,
    start_order: int,
    target_words: int,
    min_words: int,
    max_words: int,
    paragraph_words: int,
    max_passages: int,
    transcript_source: Path,
) -> list[PassageRecord]:
    paragraphs = split_into_paragraphs(transcript_text, target_words=paragraph_words)
    passages = pack_paragraphs_into_passages(
        paragraphs,
        min_words=min_words,
        target_words=target_words,
        max_words=max_words,
        max_passages=max_passages,
    )

    if not passages:
        raise RuntimeError("Transcript did not produce any benchmark passages")

    generated_records: list[PassageRecord] = []
    for index, passage_text in enumerate(passages, start=1):
        passage_id = f"bk_{index:02d}"
        passage_path = output_dir / f"{passage_id}.txt"
        passage_path.write_text(passage_text.strip() + "\n", encoding="utf-8")
        generated_records.append(
            PassageRecord(
                passage_id=passage_id,
                order=start_order + index - 1,
                source_type="transcript_passage",
                source_path=passage_path,
                role="brothers_karamazov",
                notes=f"Transcript-backed passage from {relpath(transcript_source)}",
                generated=True,
                word_count=count_words(passage_text),
                paragraph_count=count_paragraphs(passage_text),
                target_word_count=target_words,
            )
        )

    return generated_records


def build_manifest(
    *,
    reference_audio: Path,
    transcript_path: Path,
    inputs_dir: Path,
    output_dir: Path,
    existing_records: list[PassageRecord],
    generated_records: list[PassageRecord],
    target_words: int,
    min_words: int,
    max_words: int,
    paragraph_words: int,
) -> dict[str, object]:
    entries = [record.to_manifest_entry() for record in (*existing_records, *generated_records)]
    return {
        "created_at": _utc_now(),
        "reference_audio": relpath(reference_audio),
        "reference_transcript": relpath(transcript_path),
        "inputs_dir": relpath(inputs_dir),
        "output_dir": relpath(output_dir),
        "target_words": target_words,
        "min_words": min_words,
        "max_words": max_words,
        "paragraph_words": paragraph_words,
        "existing_input_count": len(existing_records),
        "generated_passage_count": len(generated_records),
        "entry_count": len(entries),
        "entries": entries,
        "existing_inputs": [record.to_manifest_entry() for record in existing_records],
        "generated_passages": [record.to_manifest_entry() for record in generated_records],
    }


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the long-form TTS benchmark corpus.")
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--transcript-path", type=Path, default=None)
    parser.add_argument("--target-words", type=int, default=DEFAULT_TARGET_WORDS)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--paragraph-words", type=int, default=DEFAULT_PARAGRAPH_WORDS)
    parser.add_argument("--max-generated-passages", type=int, default=DEFAULT_MAX_GENERATED_PASSAGES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.inputs_dir.exists():
        raise FileNotFoundError(f"inputs directory not found: {args.inputs_dir}")
    if not args.reference_audio.exists():
        raise FileNotFoundError(f"reference audio not found: {args.reference_audio}")

    transcript_path = args.transcript_path or args.reference_audio.with_suffix(".transcript.auto.txt")
    if transcript_path.exists():
        transcript_text = load_text(transcript_path)
    else:
        transcript_text = auto_transcribe_reference(args.reference_audio, transcript_path)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing_records, next_order = load_existing_input_records(args.inputs_dir)
    generated_records = build_generated_records(
        transcript_text=transcript_text,
        output_dir=args.output_dir,
        start_order=next_order,
        target_words=args.target_words,
        min_words=args.min_words,
        max_words=args.max_words,
        paragraph_words=args.paragraph_words,
        max_passages=args.max_generated_passages,
        transcript_source=args.reference_audio,
    )

    manifest = build_manifest(
        reference_audio=args.reference_audio,
        transcript_path=transcript_path,
        inputs_dir=args.inputs_dir,
        output_dir=args.output_dir,
        existing_records=existing_records,
        generated_records=generated_records,
        target_words=args.target_words,
        min_words=args.min_words,
        max_words=args.max_words,
        paragraph_words=args.paragraph_words,
    )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Built TTS benchmark corpus")
    print(f"  inputs: {len(existing_records)}")
    print(f"  generated passages: {len(generated_records)}")
    print(f"  total entries: {manifest['entry_count']}")
    print(f"  manifest: {relpath(manifest_path)}")
    for record in generated_records:
        print(f"  - {record.passage_id}: {record.word_count} words")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
